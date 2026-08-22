"""`check_data_consistency` (D19).

SOP v4 section 3 says, verbatim: "When data conflicts, identify the conflict and
request verification before a state-changing action." This module is that
sentence in code, which is the reason it exists as a tool rather than as a
warning field on a calculator - a warning is advice, and a blocking conflict has
to be able to stop `prepare_action` from minting a token.

Four situations, and the pack contains three of them for real:

  ORD-1001  the order says BOOKED, an open ticket says the driver has been, and
            a current known issue says the webhook that would prove it is late
  TKT-450   a closed ticket records a fee that this customer's agreement waives
  TKT-451   a closed ticket records a plan limit that is really a bug threshold

The fourth, two clauses of the same tier disagreeing, does not occur in the
corpus and is covered by a synthetic fixture.

What the tests below are mostly guarding is restraint. A detector that fires on
ORD-2001 as well - same status, same carrier, no ticket saying otherwise - is
worse than no detector, because every answer then arrives hedged and the hedge
stops meaning anything.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.consistency import (
    ConflictClass,
    ConflictSeverity,
    ConsistencyChecker,
    ConsistencyReport,
)
from src.domain.evidence import (
    EvidenceKind,
    EvidenceKindError,
    EvidenceScopeError,
    open_evidence_store,
)
from src.domain.resolver import PolicyResolver


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture
def chain():
    """One store, one repository, one resolver, one checker - as a run has.

    Cached per persona rather than rebuilt per call. A fresh store each time
    would put every handle in a different database, and the cross-principal
    refusals below would pass for the wrong reason.
    """
    opened: dict[str, tuple] = {}
    # One evidence database for every persona, which is what a deployment has.
    # Separate ones would make the cross-principal refusal below pass because
    # the row was missing rather than because the scope check fired.
    shared = sqlite3.connect(":memory:")

    def build(persona_id: str):
        if persona_id not in opened:
            principal = persona(persona_id)
            ctx = open_evidence_store(run_id="run_a", principal=principal, connection=shared)
            store = ctx.__enter__()
            repo = open_repository(principal, get_settings().db_path)
            checker = ConsistencyChecker(
                store=store, repository=repo, resolver=PolicyResolver(repo.connection)
            )
            opened[persona_id] = (ctx, store, repo, checker)
        _, store, repo, checker = opened[persona_id]
        return store, repo, checker

    yield build
    for ctx, _, repo, _ in opened.values():
        repo.close()
        ctx.__exit__(None, None, None)
    shared.close()


def order_report(chain, persona_id, order_id, **kwargs) -> ConsistencyReport:
    store, repo, checker = chain(persona_id)
    snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, repo.get_order(order_id).to_payload())
    return checker.check(snapshot_id=snapshot, **kwargs)


def ticket_report(chain, persona_id, ticket_id, **kwargs) -> ConsistencyReport:
    store, repo, checker = chain(persona_id)
    snapshot = store.mint(EvidenceKind.TICKET_SNAPSHOT, repo.get_ticket(ticket_id).to_payload())
    return checker.check(snapshot_id=snapshot, **kwargs)


def classes(report: ConsistencyReport) -> set[str]:
    return {c.conflict_class for c in report.conflicts}


def only(report: ConsistencyReport, conflict_class: ConflictClass):
    matching = [c for c in report.conflicts if c.conflict_class == conflict_class]
    assert len(matching) == 1, f"expected exactly one {conflict_class}, got {len(matching)}"
    return matching[0]


class TestStaleStatus:
    """ORD-1001. Three sources, and no two of them agree."""

    def test_the_conflict_is_detected(self, chain):
        report = order_report(chain, "northstar_customer", "ORD-1001")
        assert ConflictClass.STALE_STATUS in classes(report)

    def test_it_blocks_rather_than_advises(self, chain):
        # If BOOKED is stale the shipment is already collected, cancellation is
        # not permitted, and return-to-origin applies instead. Acting on the
        # stored status would be acting on a fact nobody has checked.
        assert (
            only(
                order_report(chain, "northstar_customer", "ORD-1001"), ConflictClass.STALE_STATUS
            ).severity
            == ConflictSeverity.BLOCKING
        )

    def test_all_three_sources_are_named(self, chain):
        conflict = only(
            order_report(chain, "northstar_customer", "ORD-1001"), ConflictClass.STALE_STATUS
        )
        assert set(conflict.sources) == {
            "ORD-1001",
            "TKT-504",
            "product_operations_guide_and_known_issues::KI-211",
        }

    def test_the_ticket_link_is_reported_as_an_inference(self, chain):
        # Assumption A3: tickets carry no order_id. The link is that ORD-1001 is
        # Northstar's only SwiftShip order, which is strong and is still a
        # guess. Stating it as fact is how a caveat becomes a claim.
        conflict = only(
            order_report(chain, "northstar_customer", "ORD-1001"), ConflictClass.STALE_STATUS
        )
        assert conflict.inference_note is not None
        assert "only" in conflict.inference_note.lower()
        assert conflict.confidence < 1.0

    def test_the_known_issue_instruction_is_carried_verbatim(self, chain):
        # KI-211 does not merely describe the lag, it says what to do about it.
        # The instruction is the answer, so it must survive to the fact block
        # rather than being paraphrased by the model.
        conflict = only(
            order_report(chain, "northstar_customer", "ORD-1001"), ConflictClass.STALE_STATUS
        )
        assert "verify the carrier status" in conflict.instruction.lower()

    def test_the_report_does_not_pick_a_side(self, chain):
        conflict = only(
            order_report(chain, "northstar_customer", "ORD-1001"), ConflictClass.STALE_STATUS
        )
        assert "may" in conflict.detail.lower() or "cannot be confirmed" in conflict.detail.lower()


class TestStaleStatusRestraint:
    def test_a_booked_order_with_no_corroborating_ticket_is_not_stale(self, chain):
        # ORD-2001 is the control: same status, same carrier, same missing
        # pickup timestamp, and nobody has said a driver came.
        report = order_report(chain, "lumenworks_customer", "ORD-2001")
        assert ConflictClass.STALE_STATUS not in classes(report)

    def test_a_delivered_order_is_not_stale(self, chain):
        assert ConflictClass.STALE_STATUS not in classes(
            order_report(chain, "axis_customer", "ORD-4001")
        )

    def test_an_order_whose_pickup_is_recorded_is_not_stale(self, chain):
        # ORD-1002 has pickup_actual_at set, so there is nothing for a late
        # webhook to explain.
        assert ConflictClass.STALE_STATUS not in classes(
            order_report(chain, "northstar_customer", "ORD-1002")
        )

    def test_a_resolved_known_issue_cannot_corroborate(self, chain, monkeypatch):
        # KI-176 is resolved and says in terms not to use it for new incidents.
        # Detection must read the issue's status, not merely its existence.
        from src.domain import consistency

        monkeypatch.setattr(consistency, "_ACTIVE_ISSUE_STATUSES", frozenset())
        assert ConflictClass.STALE_STATUS not in classes(
            order_report(chain, "northstar_customer", "ORD-1001")
        )


class TestHistoricalContradiction:
    def test_tkt_450_contradicts_the_northstar_agreement(self, chain):
        report = ticket_report(chain, "maya_agent", "TKT-450")
        conflict = only(report, ConflictClass.HISTORICAL_CONTRADICTION)
        assert "northstar_logistics_enterprise_agreement::§2" in conflict.sources
        assert "TKT-450" in conflict.sources

    def test_the_recorded_answer_and_the_current_rule_are_both_stated(self, chain):
        conflict = only(
            ticket_report(chain, "maya_agent", "TKT-450"), ConflictClass.HISTORICAL_CONTRADICTION
        )
        assert conflict.claimed_value == 250
        assert conflict.current_value == 0

    def test_it_advises_rather_than_blocks(self, chain):
        # Nothing about the current data is in doubt. A past answer was wrong,
        # which the answer must say and no action needs to wait for.
        assert (
            only(
                ticket_report(chain, "maya_agent", "TKT-450"),
                ConflictClass.HISTORICAL_CONTRADICTION,
            ).severity
            == ConflictSeverity.ADVISORY
        )

    def test_the_tier_five_text_is_never_the_basis(self, chain):
        # Policy v3 §1: historical tickets are context only. The closed ticket
        # is the thing being corrected, never the authority for the correction.
        conflict = only(
            ticket_report(chain, "maya_agent", "TKT-450"), ConflictClass.HISTORICAL_CONTRADICTION
        )
        assert conflict.basis_clause == "northstar_logistics_enterprise_agreement::§2"

    def test_tkt_451_contradicts_the_product_guide(self, chain):
        conflict = only(
            ticket_report(chain, "rohit_agent", "TKT-451"), ConflictClass.HISTORICAL_CONTRADICTION
        )
        assert conflict.claimed_value == 3000
        assert conflict.current_value == 5000
        assert conflict.basis_clause == "product_operations_guide_and_known_issues::§1"

    def test_tkt_451_also_surfaces_the_known_issue_that_explains_the_number(self, chain):
        # The subtlest trap in the pack. 3,000 is real - it is the threshold
        # above which uploads intermittently fail - and it is not the plan
        # limit. Correcting the number without KI-208 tells the customer their
        # 3,500-row upload should work, which it will not.
        conflict = only(
            ticket_report(chain, "rohit_agent", "TKT-451"), ConflictClass.HISTORICAL_CONTRADICTION
        )
        assert "product_operations_guide_and_known_issues::KI-208" in conflict.sources
        assert "workaround" in (conflict.instruction or "").lower()

    def test_the_contradiction_is_relative_to_the_account(self, chain):
        # The same recorded sentence is wrong for Northstar and right for an
        # account with no agreement, because the SOP does charge INR 250.
        # A checker comparing against "the policy" rather than "this account's
        # governing clause" gets this backwards.
        store, repo, checker = chain("rohit_agent")
        payload = repo.get_ticket("TKT-450").to_payload() | {"account_id": "ACCT-003"}
        snapshot = store.mint(EvidenceKind.TICKET_SNAPSHOT, payload)
        report = checker.check(snapshot_id=snapshot)
        assert ConflictClass.HISTORICAL_CONTRADICTION not in classes(report)


class TestHistoricalRestraint:
    @pytest.mark.parametrize("ticket_id", ["TKT-501", "TKT-502", "TKT-503", "TKT-504", "TKT-505"])
    def test_an_open_ticket_has_no_recorded_resolution_to_contradict(self, chain, ticket_id):
        assert ConflictClass.HISTORICAL_CONTRADICTION not in classes(
            ticket_report(chain, "rohit_agent", ticket_id)
        )

    def test_a_recorded_resolution_that_agrees_is_not_a_contradiction(self, chain):
        store, repo, checker = chain("rohit_agent")
        payload = repo.get_ticket("TKT-450").to_payload() | {
            "account_id": "ACCT-003",
            "historical_resolution": "Agent told customer a INR 250 cancellation fee applied.",
        }
        snapshot = store.mint(EvidenceKind.TICKET_SNAPSHOT, payload)
        assert ConflictClass.HISTORICAL_CONTRADICTION not in classes(
            checker.check(snapshot_id=snapshot)
        )

    def test_prose_stating_no_rule_produces_no_contradiction(self, chain):
        store, repo, checker = chain("rohit_agent")
        payload = repo.get_ticket("TKT-450").to_payload() | {
            "historical_resolution": "Agent apologised and closed the ticket.",
        }
        snapshot = store.mint(EvidenceKind.TICKET_SNAPSHOT, payload)
        assert checker.check(snapshot_id=snapshot).conflicts == ()


class TestTopicChecks:
    """`missing_source` and `unresolved_same_tier` are properties of a question,
    so they fire only when the caller names the topic under discussion."""

    def test_a_topic_with_no_citable_clause_is_missing_source(self, chain):
        # TKT-503 asks how to change a billing contact. Only Northstar's
        # agreement says anything about account contacts, and Beacon is not
        # Northstar.
        report = ticket_report(chain, "rohit_agent", "TKT-503", topics=("account_contact",))
        conflict = only(report, ConflictClass.MISSING_SOURCE)
        assert conflict.severity == ConflictSeverity.BLOCKING
        assert "account_contact" in conflict.detail

    def test_a_topic_that_does_resolve_is_not_missing(self, chain):
        report = ticket_report(chain, "maya_agent", "TKT-450", topics=("cancellation_fee",))
        assert ConflictClass.MISSING_SOURCE not in classes(report)

    def test_no_topics_named_means_no_topic_checks(self, chain):
        assert ConflictClass.MISSING_SOURCE not in classes(
            ticket_report(chain, "rohit_agent", "TKT-503")
        )

    def test_an_unresolved_same_tier_conflict_blocks(self, tmp_path: Path):
        db = _conflicted_db(tmp_path)
        principal = persona("northstar_customer")
        with open_evidence_store(run_id="run_a", principal=principal) as store:
            repo = open_repository(principal, db)
            checker = ConsistencyChecker(
                store=store, repository=repo, resolver=PolicyResolver(repo.connection)
            )
            snapshot = store.mint(
                EvidenceKind.ORDER_SNAPSHOT, {"order_id": "ORD-9001", "account_id": "ACCT-001"}
            )
            report = checker.check(snapshot_id=snapshot, topics=("cancellation_fee",))
            conflict = only(report, ConflictClass.UNRESOLVED_SAME_TIER)
            assert conflict.severity == ConflictSeverity.BLOCKING
            repo.close()


class TestTheReportIsEvidence:
    def test_it_mints_as_a_consistency_report(self, chain):
        store, repo, checker = chain("northstar_customer")
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, repo.get_order("ORD-1001").to_payload())
        report = checker.check(snapshot_id=snapshot)
        assert store.kind_of(report.report_id) is EvidenceKind.CONSISTENCY_REPORT

    def test_the_snapshot_is_in_its_provenance(self, chain):
        store, repo, checker = chain("northstar_customer")
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, repo.get_order("ORD-1001").to_payload())
        report = checker.check(snapshot_id=snapshot)
        assert store.provenance(report.report_id) == (snapshot.evidence_id,)

    def test_the_payload_round_trips(self, chain):
        store, repo, checker = chain("northstar_customer")
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, repo.get_order("ORD-1001").to_payload())
        report = checker.check(snapshot_id=snapshot)
        payload = store.read(report.report_id, expect=EvidenceKind.CONSISTENCY_REPORT)
        assert payload["blocking"] is True
        assert payload["conflicts"][0]["conflict_class"] == ConflictClass.STALE_STATUS

    def test_a_snapshot_from_another_principal_is_refused(self, chain):
        store, repo, _ = chain("northstar_customer")
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, repo.get_order("ORD-1001").to_payload())
        _, other_repo, other_checker = chain("lumenworks_customer")
        assert other_repo is not repo
        with pytest.raises(EvidenceScopeError):
            other_checker.check(snapshot_id=snapshot)

    def test_a_resolution_handle_is_not_a_snapshot(self, chain):
        store, _, checker = chain("northstar_customer")
        wrong = store.mint(EvidenceKind.POLICY_RESOLUTION, {"topic": "cancellation_fee"})
        with pytest.raises(EvidenceKindError):
            checker.check(snapshot_id=wrong)


class TestReportShape:
    def test_a_clean_order_reports_no_conflicts_rather_than_failing(self, chain):
        report = order_report(chain, "axis_customer", "ORD-4001")
        assert report.conflicts == ()
        assert report.blocking is False

    def test_the_checks_that_ran_are_listed(self, chain):
        # A report saying "no conflicts" is only meaningful alongside what was
        # looked for. Silence from a check that never ran reads identically.
        report = order_report(chain, "axis_customer", "ORD-4001")
        assert ConflictClass.STALE_STATUS in report.checked

    def test_blocking_is_true_only_when_a_blocking_conflict_stands(self, chain):
        assert order_report(chain, "northstar_customer", "ORD-1001").blocking is True
        assert ticket_report(chain, "maya_agent", "TKT-450").blocking is False


def _conflicted_db(tmp_path: Path) -> Path:
    """Two Tier 1 clauses on one topic, disagreeing. Synthetic, and labelled so."""
    db = tmp_path / "conflict.db"
    conn = sqlite3.connect(db)
    conn.executescript(Path("src/datastore/schema.sql").read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO accounts (account_id, account_name, plan, status, premium_support) "
        "VALUES (?,?,?,?,?)",
        ("ACCT-001", "Northstar Logistics", "Enterprise", "active", 1),
    )
    for ref, params in (
        ("§A", '{"fee_inr": 0, "overrides": true}'),
        ("§B", '{"fee_inr": 99, "overrides": true}'),
    ):
        clause_id = f"synthetic_agreement::{ref}"
        conn.execute(
            "INSERT INTO clauses (clause_id, doc_id, doc_title, clause_ref, title, tier, "
            "account_id, status, params, text) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                clause_id,
                "synthetic_agreement",
                "Synthetic Agreement",
                ref,
                "Cancellation",
                1,
                "ACCT-001",
                "CURRENT",
                params,
                "synthetic",
            ),
        )
        conn.execute(
            "INSERT INTO clause_topics (clause_id, topic) VALUES (?, ?)",
            (clause_id, "cancellation_fee"),
        )
    conn.commit()
    conn.close()
    return db
