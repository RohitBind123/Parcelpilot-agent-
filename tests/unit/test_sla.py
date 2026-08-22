"""First-response target status.

Acceptance is GS-011 through GS-016. Two properties carry the file.

`measurable` is always False. There is no `first_response_at` column, so the
honest claim is elapsed-versus-target, and a system that reports a breach it
cannot measure will eventually be asked to prove one.

AS_OF is a Sunday. A 24x7 target runs immediately; a business-hours target does
not start until Monday 09:00. Two tickets raised ninety minutes apart on the
same Sunday can therefore be one past its target and the other not yet started,
and that is not an edge case in this dataset - it is most of it.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.errors import WrongEvidence
from src.domain.calculators.sla import sla_first_response_status
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import SeverityVerdict, deterministic_severity

NORTHSTAR_SLA = "northstar_logistics_enterprise_agreement::§1"
LUMENWORKS_SLA = "lumenworks_service_agreement::§1"
V3_TARGETS = "support_policy_v3_current::§3"
V2 = "support_policy_v2_deprecated::§-"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


def inferred(severity: str, confidence: float = 1.0) -> SeverityVerdict:
    """A model-graded verdict. The default is what the real classifier returns
    for the tickets in the pack it grades stably (see the M4 calibration in
    `severity.py`); the low-confidence cases below pass an explicit value."""
    return SeverityVerdict(
        severity=severity,
        confidence=confidence,
        basis_clause="support_policy_v3_current::§2",
        basis="inferred",
    )


@pytest.fixture(scope="module")
def db_path():
    return get_settings().db_path


@pytest.fixture
def chain(db_path):
    stores: dict[str, tuple] = {}

    def factory(persona_id: str, ticket_id: str, account_id: str):
        principal = persona(persona_id)
        if persona_id not in stores:
            ctx = open_evidence_store(run_id="run_1", principal=principal)
            stores[persona_id] = (ctx, ctx.__enter__())
        store = stores[persona_id][1]

        with open_repository(principal, db_path) as repo:
            ticket = repo.get_ticket(ticket_id)
            account = repo.get_account(account_id)
        tsnap = store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())
        asnap = store.mint(EvidenceKind.ACCOUNT_SNAPSHOT, account.to_payload())

        with PolicyResolver.open(db_path) as resolver:
            resolution = resolver.resolve("first_response_target", principal, account_id=account_id)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[tsnap]
        )
        return store, tsnap, asnap, handle, ticket

    yield factory
    for ctx, _ in stores.values():
        ctx.__exit__(None, None, None)


def run(chain, persona_id, ticket_id, account_id, severity=None, surface="ops"):
    store, tsnap, asnap, resolution, ticket = chain(persona_id, ticket_id, account_id)
    verdict = severity or deterministic_severity(ticket.subject, ticket.description or "")
    return sla_first_response_status(
        store,
        snapshot_id=tsnap,
        account_snapshot_id=asnap,
        resolution_id=resolution,
        severity=verdict,
        surface=surface,
    )


class TestTwentyFourSevenTargetsIgnoreTheWeekend:
    def test_northstar_p1_is_fifteen_minutes_from_creation(self, chain):
        # GS-011. Created Sunday 10:30, due 10:45 the same Sunday.
        outcome = run(chain, "maya_agent", "TKT-501", "ACCT-001")
        assert outcome.severity == "P1"
        assert outcome.severity_inferred is False  # matched by guard, not sampled
        assert outcome.target == "15 minutes, 24x7"
        assert outcome.clock_type == "24x7"
        assert outcome.due_at == "2026-08-16T10:45:00+05:30"
        assert outcome.elapsed_minutes == 30
        assert outcome.past_target_by_minutes == 15
        assert outcome.target_clause == NORTHSTAR_SLA
        assert outcome.overridden_clauses == (V3_TARGETS,)

    def test_a_security_ticket_uses_the_default_enterprise_grid(self, chain):
        # GS-015. Axis is Enterprise with no agreement: 30 minutes, 24x7.
        outcome = run(chain, "rohit_agent", "TKT-505", "ACCT-004")
        assert outcome.severity == "P1"
        assert outcome.target == "30 minutes, 24x7"
        assert outcome.due_at == "2026-08-16T09:00:00+05:30"
        assert outcome.elapsed_minutes == 150
        assert outcome.target_clause == V3_TARGETS
        assert outcome.overridden_clauses == ()
        assert outcome.escalate is True


class TestBusinessHoursTargetsWaitForMonday:
    def test_a_growth_p2_starts_monday_morning(self, chain):
        # GS-012. Created Sunday 09:45; four business hours lands Monday 13:00,
        # not Sunday 13:45. The naive answer reports a breach that has not
        # happened.
        outcome = run(chain, "maya_agent", "TKT-502", "ACCT-002", severity=inferred("P2"))
        assert outcome.target == "4 business hours"
        assert outcome.clock_type == "business_hours"
        assert outcome.clock_starts_at == "2026-08-17T09:00:00+05:30"
        assert outcome.due_at == "2026-08-17T13:00:00+05:30"
        assert outcome.past_target_by_minutes < 0
        assert outcome.target_clause == LUMENWORKS_SLA

    def test_a_standard_p3_is_two_business_days(self, chain):
        # GS-013.
        outcome = run(chain, "rohit_agent", "TKT-503", "ACCT-003", severity=inferred("P3"))
        assert outcome.target == "2 business days"
        assert outcome.due_at == "2026-08-19T09:00:00+05:30"
        assert outcome.target_clause == V3_TARGETS

    def test_an_agreement_p3_is_eight_business_hours(self, chain):
        # GS-014.
        outcome = run(chain, "maya_agent", "TKT-504", "ACCT-001", severity=inferred("P3"))
        assert outcome.target == "8 business hours"
        assert outcome.due_at == "2026-08-17T17:00:00+05:30"
        assert outcome.target_clause == NORTHSTAR_SLA

    def test_the_weekend_start_is_surfaced_as_a_warning(self, chain):
        outcome = run(chain, "maya_agent", "TKT-502", "ACCT-002", severity=inferred("P2"))
        assert any("business hours" in w for w in outcome.warnings)


class TestMeasurability:
    @pytest.mark.parametrize(
        ("ticket_id", "account_id", "severity"),
        [("TKT-501", "ACCT-001", None), ("TKT-502", "ACCT-002", inferred("P2"))],
    )
    def test_no_outcome_ever_claims_a_measured_breach(self, chain, ticket_id, account_id, severity):
        outcome = run(chain, "maya_agent", ticket_id, account_id, severity=severity)
        assert outcome.measurable is False
        assert "first replied" in outcome.measurability_note

    def test_past_target_is_stated_as_elapsed_not_as_a_breach(self, chain):
        outcome = run(chain, "maya_agent", "TKT-501", "ACCT-001")
        assert outcome.past_target is True
        assert outcome.measurable is False


class TestLowConfidenceSeverity:
    def test_the_customer_surface_declines_to_quote_a_target(self, chain):
        # D25. Quoting a target from a severity we do not trust is a promise
        # ParcelPilot may not keep.
        outcome = run(
            chain,
            "maya_agent",
            "TKT-502",
            "ACCT-002",
            severity=inferred("P3", confidence=0.4),
            surface="customer",
        )
        assert outcome.severity is None
        assert outcome.target is None
        assert outcome.due_at is None
        assert outcome.escalate is True

    def test_ops_triage_rounds_up_instead(self, chain):
        # An over-prioritised ticket costs an analyst two minutes; a missed P1
        # costs an outage.
        outcome = run(
            chain,
            "maya_agent",
            "TKT-502",
            "ACCT-002",
            severity=inferred("P3", confidence=0.4),
            surface="ops",
        )
        assert outcome.severity == "P2"
        assert outcome.severity_inferred is True
        assert outcome.target is not None
        assert any("triaged up" in w for w in outcome.warnings)

    def test_a_deterministic_verdict_is_never_marked_inferred(self, chain):
        outcome = run(chain, "rohit_agent", "TKT-505", "ACCT-004")
        assert outcome.severity_inferred is False
        assert outcome.severity_confidence == 1.0


class TestTierDiscipline:
    @pytest.mark.parametrize(
        ("ticket_id", "account_id"),
        [("TKT-501", "ACCT-001"), ("TKT-503", "ACCT-003"), ("TKT-505", "ACCT-004")],
    )
    def test_the_deprecated_grid_is_never_the_target_clause(self, chain, ticket_id, account_id):
        outcome = run(chain, "rohit_agent", ticket_id, account_id, severity=inferred("P1"))
        assert outcome.target_clause != V2
        assert V2 not in outcome.overridden_clauses


class TestEvidenceDiscipline:
    def test_mismatched_ticket_and_account_snapshots_are_refused(self, chain):
        store, tsnap, _, resolution, _t = chain("maya_agent", "TKT-501", "ACCT-001")
        _, _, foreign_account, _, _ = chain("maya_agent", "TKT-502", "ACCT-002")
        with pytest.raises(WrongEvidence, match="account"):
            sla_first_response_status(
                store,
                snapshot_id=tsnap,
                account_snapshot_id=foreign_account,
                resolution_id=resolution,
                severity=inferred("P1"),
            )

    def test_an_order_snapshot_cannot_stand_in_for_a_ticket(self, chain):
        from src.domain.evidence import EvidenceKindError

        store, _, asnap, resolution, _t = chain("maya_agent", "TKT-501", "ACCT-001")
        order_like = store.mint(EvidenceKind.ORDER_SNAPSHOT, {"order_id": "ORD-1001"})
        with pytest.raises(EvidenceKindError):
            sla_first_response_status(
                store,
                snapshot_id=order_like,
                account_snapshot_id=asnap,
                resolution_id=resolution,
                severity=inferred("P1"),
            )

    def test_the_calc_handle_records_all_three_inputs(self, chain):
        store, tsnap, asnap, resolution, _t = chain("maya_agent", "TKT-501", "ACCT-001")
        outcome = sla_first_response_status(
            store,
            snapshot_id=tsnap,
            account_snapshot_id=asnap,
            resolution_id=resolution,
            severity=inferred("P1"),
        )
        assert store.provenance(outcome.calc_id) == (
            tsnap.evidence_id,
            asnap.evidence_id,
            resolution.evidence_id,
        )


class TestDeterministicGuard:
    def test_the_two_named_triggers_match(self):
        assert (
            deterministic_severity(
                "All shipment creation is failing",
                "Every user gets HTTP 500 when creating any shipment.",
            ).severity
            == "P1"
        )
        assert (
            deterministic_severity(
                "Possible API key exposure",
                "An employee posted a screenshot containing a production API key publicly.",
            ).severity
            == "P1"
        )

    def test_a_partial_failure_does_not_trip_the_outage_guard(self):
        # TKT-502's shape: bulk upload fails, one-by-one still works. A looser
        # pattern would page someone for a degraded feature.
        assert (
            deterministic_severity(
                "Bulk upload fails for 4,200-row CSV",
                "The CSV reaches roughly 70% and fails. Creating shipments one-by-one still works.",
            )
            is None
        )

    def test_an_ordinary_question_does_not_trip_any_guard(self):
        assert (
            deterministic_severity(
                "How do we change the billing contact?",
                "Customer wants to replace the billing-contact email on their account.",
            )
            is None
        )

    def test_the_guard_returns_none_rather_than_guessing_low(self):
        # "No guard fired" and "the guard says P3" are different statements,
        # and only inference may make the second.
        assert deterministic_severity("Minor typo on invoice", "Cosmetic.") is None
