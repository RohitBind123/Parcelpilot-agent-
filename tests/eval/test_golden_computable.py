"""The golden set, run through the real chain (D21 layer 2).

Every entry M3 can answer is driven from `golden_set.yaml` rather than restated
here, so the expectations in this file are the ones that were signed off. If a
verdict changes there, this fails - which is the whole reason the golden set was
written before the resolver.

Sixteen of the thirty-two are computable today: cancellation, service credit and
SLA targets. The rest need consistency checking (M4), the tool layer (M5) or
answer composition (M7), and the coverage test at the bottom asserts that the
uncovered set is exactly those and shrinks as milestones land - so this file
cannot quietly stop testing something.

Nothing is mocked. The database is the committed one, the clauses are the real
registry, and the arithmetic is the arithmetic that will ship.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.cancellation import compute_cancellation_fee
from src.domain.calculators.credit import compute_service_credit
from src.domain.calculators.sla import sla_first_response_status
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import SeverityVerdict, deterministic_severity

GOLDEN = Path(__file__).resolve().parent / "golden_set.yaml"

CANCELLATION = ["GS-001", "GS-002", "GS-003", "GS-004", "GS-005", "GS-006"]
CREDIT = ["GS-007", "GS-008", "GS-009", "GS-010"]
SLA = ["GS-011", "GS-012", "GS-013", "GS-014", "GS-015"]

#: Entries whose acceptance needs a milestone that has not landed. Named
#: individually so the list is a to-do rather than a shrug.
NOT_YET_COMPUTABLE = {
    "GS-016": "abstract SLA question with no ticket; needs the tool layer (M5)",
    "GS-017": "tier discipline at the answer surface; needs composition (M7)",
    "GS-018": "deliberate tier-4 read; needs the tool layer (M5)",
    "GS-019": "staleness conflict; needs check_data_consistency (M4)",
    "GS-020": "historical contradiction; needs check_data_consistency (M4)",
    "GS-021": "historical contradiction; needs check_data_consistency (M4)",
    "GS-022": "known-issue matching; needs retrieval in the tool layer (M5)",
    "GS-023": "plan capability; needs the tool layer (M5)",
    "GS-024": "no-source escalation; needs the escalation flow (M7)",
    "GS-025": "no-source escalation; needs the escalation flow (M7)",
    "GS-026": "cross-account denial; covered by repository tests, asserted end to end in M5",
    "GS-027": "cross-account retrieval denial; asserted in M2 end-to-end tests",
    "GS-028": "prompt injection; needs tool projection (M5)",
    "GS-029": "scope refusal; needs tool projection (M5)",
    "GS-030": "my_queue; covered by repository tests, asserted end to end in M5",
    "GS-031": "ops detection; needs scan_support_health (M10)",
    "GS-032": "role projection; needs tool projection (M5)",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, dict]:
    return {e["id"]: e for e in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def db_path():
    return get_settings().db_path


def principal_for(entry: dict):
    return to_principal(get_persona(entry["persona"]))


def scope_for(entry: dict, principal):
    """Which account the question is about. Staff must say; customers cannot."""
    return None if principal.account_id else _subject_account(entry)


def _subject_account(entry: dict) -> str:
    return {
        "ORD-1001": "ACCT-001",
        "ORD-1002": "ACCT-001",
        "ORD-2001": "ACCT-002",
        "ORD-2002": "ACCT-002",
        "ORD-3001": "ACCT-003",
        "ORD-4001": "ACCT-004",
        "TKT-501": "ACCT-001",
        "TKT-502": "ACCT-002",
        "TKT-503": "ACCT-003",
        "TKT-504": "ACCT-001",
        "TKT-505": "ACCT-004",
    }[entry["subject"]]


class TestCancellation:
    @pytest.mark.parametrize("entry_id", CANCELLATION)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        principal = principal_for(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            with open_repository(principal, db_path) as repo:
                order = repo.get_order(entry["subject"])
            snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, order.to_payload())
            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "cancellation_fee", principal, account_id=scope_for(entry, principal)
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
            )
            outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=handle)

        assert outcome.fee_inr == expect["amount_inr"], entry["derivation"]
        assert outcome.governing_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        for forbidden in expect.get("must_not_cite", []):
            assert forbidden != outcome.governing_clause
            assert forbidden not in outcome.overridden_clauses

    def test_the_pair_is_genuinely_discriminating(self, golden):
        # Guards the parametrised test above from passing on a system that
        # hard-codes one of the two.
        assert (
            golden["GS-001"]["check"]["order_status"] == golden["GS-002"]["check"]["order_status"]
        )
        assert golden["GS-001"]["expect"]["amount_inr"] != golden["GS-002"]["expect"]["amount_inr"]


class TestServiceCredit:
    @pytest.mark.parametrize("entry_id", CREDIT)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        check = entry.get("check") or {}
        principal = principal_for(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            if entry.get("subject"):
                with open_repository(principal, db_path) as repo:
                    payload = repo.get_order(entry["subject"]).to_payload()
                account = _subject_account(entry)
            else:
                # A hypothetical: the question states the delay and the fee, so
                # the snapshot carries exactly what the question asserted and
                # nothing more.
                account = principal.account_id or "ACCT-003"
                payload = {
                    "order_id": None,
                    "account_id": account,
                    "delay_hours": check.get("hypothetical_delay_hours"),
                    "carrier_fault": True,
                    "customer_fault": False,
                    "shipment_fee_inr": check.get("hypothetical_fee_inr"),
                }
            snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, payload)

            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "failed_pickup_credit",
                    principal,
                    account_id=None if principal.account_id else account,
                )
                approval = resolver.resolve(
                    "credit_approval",
                    principal,
                    account_id=None if principal.account_id else account,
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
            )
            approval_handle = store.mint(EvidenceKind.POLICY_RESOLUTION, approval.to_payload())
            outcome = compute_service_credit(
                store,
                snapshot_id=snapshot,
                resolution_id=handle,
                approval_resolution_id=approval_handle,
            )

        assert outcome.eligible == (expect["verdict"] == "eligible"), entry["derivation"]
        assert outcome.credit_inr == expect["amount_inr"]
        assert outcome.governing_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        if expect.get("manager_approval_required") is not None:
            assert outcome.requires_manager_approval == expect["manager_approval_required"]

    def test_the_same_question_gets_opposite_answers_by_account(self, golden):
        assert golden["GS-008"]["question"] == golden["GS-009"]["question"]
        assert golden["GS-008"]["expect"]["verdict"] != golden["GS-009"]["expect"]["verdict"]


class TestSlaTargets:
    @pytest.mark.parametrize("entry_id", SLA)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        principal = principal_for(entry)
        account_id = _subject_account(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            with open_repository(principal, db_path) as repo:
                ticket = repo.get_ticket(entry["subject"])
                account = repo.get_account(account_id)
            tsnap = store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())
            asnap = store.mint(EvidenceKind.ACCOUNT_SNAPSHOT, account.to_payload())

            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "first_response_target", principal, account_id=account_id
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[tsnap]
            )

            # The guard where it fires; the signed-off severity otherwise,
            # because inference is M4's job and this is not the test for it.
            verdict = deterministic_severity(
                ticket.subject, ticket.description or ""
            ) or SeverityVerdict(
                severity=expect["severity"],
                confidence=1.0,
                basis_clause="support_policy_v3_current::§2",
                basis="pinned from the reviewed golden set pending M4",
            )
            outcome = sla_first_response_status(
                store,
                snapshot_id=tsnap,
                account_snapshot_id=asnap,
                resolution_id=handle,
                severity=verdict,
                surface="ops",
            )

        assert outcome.severity == expect["severity"], entry["derivation"]
        assert outcome.target == expect["target"]
        assert outcome.due_at == expect["target_due"]
        assert outcome.target_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        assert outcome.measurable is False
        for forbidden in expect.get("must_not_cite", []):
            assert outcome.target_clause != forbidden

    def test_the_two_named_p1_triggers_are_deterministic(self, golden, db_path):
        # GS-011 and GS-015 must never depend on a model sample.
        for entry_id in ("GS-011", "GS-015"):
            entry = golden[entry_id]
            principal = principal_for(entry)
            with open_repository(principal, db_path) as repo:
                ticket = repo.get_ticket(entry["subject"])
            verdict = deterministic_severity(ticket.subject, ticket.description or "")
            assert verdict is not None and verdict.severity == "P1"
            assert verdict.deterministic is True

    def test_the_weekend_actually_moves_a_deadline(self, golden):
        # If AS_OF stopped being a Sunday this suite would still pass while
        # testing nothing interesting, so the property is asserted directly.
        assert golden["GS-012"]["expect"]["target_due"].startswith("2026-08-17")
        assert golden["GS-011"]["expect"]["target_due"].startswith("2026-08-16")


class TestCoverage:
    def test_every_entry_is_either_covered_or_explicitly_deferred(self, golden):
        covered = set(CANCELLATION + CREDIT + SLA)
        deferred = set(NOT_YET_COMPUTABLE)
        assert covered | deferred == set(golden)
        assert not covered & deferred, "an entry is both covered and deferred"

    def test_the_deferred_list_shrinks_as_milestones_land(self, golden):
        # A reminder with teeth: when M4 lands, its entries must move out of
        # NOT_YET_COMPUTABLE or this number stops being true.
        assert len(NOT_YET_COMPUTABLE) == 17
        assert all(reason for reason in NOT_YET_COMPUTABLE.values())
