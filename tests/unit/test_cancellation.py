"""The cancellation fee calculator.

Acceptance criteria are golden-set entries GS-001 through GS-006. The
discriminating pair lives here: ORD-1001 and ORD-2001 are the same shape and
must come out INR 0 and INR 250, purely because of the agreement.

Two properties beyond arithmetic get attention. The calculator must refuse a
bare order id, because that refusal is what forces the model through the
resolver (D13a). And a shipment that cannot be cancelled must report `fee_inr:
None`, never 0 - "cancelled free of charge" and "not cancellable" are opposite
answers and zero renders as the wrong one.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.cancellation import (
    CancellationOutcome,
    compute_cancellation_fee,
)
from src.domain.calculators.errors import NoBasis, WrongEvidence
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver

NORTHSTAR_CANCEL = "northstar_logistics_enterprise_agreement::§2"
LUMENWORKS_CANCEL = "lumenworks_service_agreement::§2"
SOP_CANCEL = "cancellation_and_service_credit_sop_v4::§1"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture(scope="module")
def db_path():
    return get_settings().db_path


@pytest.fixture
def chain(db_path):
    """Mints the handles a calculator needs, the way the tool layer will.

    One store per test, shared across every chain built in it, so a handle
    minted by one call is visible to another - which is also what makes the
    cross-chain refusal tests meaningful rather than accidental.
    """
    stores: dict[str, tuple] = {}

    def factory(persona_id: str, order_id: str, topic: str = "cancellation_fee"):
        principal = persona(persona_id)
        if persona_id not in stores:
            ctx = open_evidence_store(run_id="run_1", principal=principal)
            stores[persona_id] = (ctx, ctx.__enter__())
        store = stores[persona_id][1]

        with open_repository(principal, db_path) as repo:
            order = repo.get_order(order_id)
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, order.to_payload())

        with PolicyResolver.open(db_path) as resolver:
            resolution = resolver.resolve(topic, principal)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
        )
        return store, snapshot, handle

    yield factory
    for ctx, _ in stores.values():
        ctx.__exit__(None, None, None)


class TestTheDiscriminatingPair:
    def test_northstar_pays_nothing_because_the_agreement_waives_it(self, chain):
        store, snapshot, resolution = chain("northstar_customer", "ORD-1001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)

        assert outcome.cancellable is True
        assert outcome.fee_inr == 0
        assert outcome.minutes_since_booking == 120
        assert outcome.governing_clause == NORTHSTAR_CANCEL
        assert outcome.overridden_clauses == (SOP_CANCEL,)
        assert outcome.is_override is True

    def test_lumenworks_pays_the_standard_fee_despite_holding_an_agreement(self, chain):
        store, snapshot, resolution = chain("lumenworks_customer", "ORD-2001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)

        assert outcome.fee_inr == 250
        assert outcome.minutes_since_booking == 75
        assert outcome.governing_clause == SOP_CANCEL
        assert outcome.overridden_clauses == ()
        assert outcome.is_override is False
        # The agreement was consulted and declined. Dropping it would make this
        # indistinguishable from an account with no contract at all.
        assert outcome.deferred_clauses == (LUMENWORKS_CANCEL,)

    def test_the_pair_differs_only_by_the_agreement(self, chain):
        store_a, snap_a, res_a = chain("northstar_customer", "ORD-1001")
        store_b, snap_b, res_b = chain("lumenworks_customer", "ORD-2001")
        first = compute_cancellation_fee(store_a, snapshot_id=snap_a, resolution_id=res_a)
        second = compute_cancellation_fee(store_b, snapshot_id=snap_b, resolution_id=res_b)

        assert first.order_status == second.order_status == "BOOKED"
        assert first.minutes_since_booking > 30 and second.minutes_since_booking > 30
        assert (first.fee_inr, second.fee_inr) == (0, 250)


class TestTheWindow:
    def test_inside_the_free_window_there_is_no_fee_and_no_override(self, chain):
        store, snapshot, resolution = chain("beacon_customer", "ORD-3001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)

        assert outcome.fee_inr == 0
        assert outcome.minutes_since_booking == 15
        assert outcome.within_free_window is True
        assert outcome.governing_clause == SOP_CANCEL
        # Same number as Northstar, entirely different reason. An answer that
        # cannot tell "inside the window" from "waived by contract" has not
        # answered either question.
        assert outcome.is_override is False
        assert outcome.fee_basis == "inside_free_window"

    def test_the_waiver_basis_is_named_distinctly(self, chain):
        store, snapshot, resolution = chain("northstar_customer", "ORD-1001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        assert outcome.within_free_window is False
        assert outcome.fee_basis == "agreement_waiver"

    def test_the_window_comes_from_params_not_from_a_constant(self, chain):
        store, snapshot, resolution = chain("beacon_customer", "ORD-3001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        assert outcome.free_window_minutes == 30
        assert outcome.window_source == SOP_CANCEL


class TestStatusRules:
    def test_a_picked_up_shipment_cannot_be_cancelled(self, chain):
        store, snapshot, resolution = chain("northstar_customer", "ORD-1002")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)

        assert outcome.cancellable is False
        assert outcome.next_action == "return_to_origin"
        # Not zero. There is no fee because no cancellation can happen, and
        # rendering that as INR 0 tells the customer it succeeded free.
        assert outcome.fee_inr is None

    def test_a_delivered_shipment_cannot_be_cancelled(self, chain):
        store, snapshot, resolution = chain("axis_customer", "ORD-4001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)

        assert outcome.cancellable is False
        assert outcome.fee_inr is None
        assert outcome.next_action is None

    def test_the_agreement_waiver_does_not_rescue_a_picked_up_shipment(self, chain):
        # Northstar's waiver has applies_to_status [BOOKED]. A calculator that
        # applies the waiver first and checks status second returns INR 0 for a
        # shipment already in transit.
        store, snapshot, resolution = chain("northstar_customer", "ORD-1002")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        assert outcome.fee_inr is None
        assert outcome.cancellable is False

    def test_status_rules_are_read_from_the_registry(self, chain):
        store, snapshot, resolution = chain("axis_customer", "ORD-4001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        # Sourced from a clause even though the governing clause for the fee is
        # a different one, so the citation for "cannot cancel" is real.
        assert outcome.status_rule_source == SOP_CANCEL


class TestEvidenceDiscipline:
    def test_a_bare_order_id_is_refused(self, chain):
        store, _, resolution = chain("northstar_customer", "ORD-1001")
        # The refusal that forces the chain. Without it the model can compute a
        # fee having never consulted precedence.
        with pytest.raises(Exception) as refused:
            compute_cancellation_fee(store, snapshot_id="ORD-1001", resolution_id=resolution)
        assert "ORD-1001" in str(refused.value)

    def test_a_resolution_handle_cannot_stand_in_for_a_snapshot(self, chain):
        store, _, resolution = chain("northstar_customer", "ORD-1001")
        with pytest.raises(Exception):
            compute_cancellation_fee(store, snapshot_id=resolution, resolution_id=resolution)

    def test_a_resolution_for_the_wrong_topic_is_refused(self, chain):
        store, snapshot, _ = chain("northstar_customer", "ORD-1001")
        _, _, credit = chain("northstar_customer", "ORD-1001", topic="failed_pickup_credit")
        # Both handles are valid and both belong to this run. Only the topic is
        # wrong, and nothing but an explicit check would catch it.
        with pytest.raises(WrongEvidence, match="cancellation_fee"):
            compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=credit)

    def test_a_resolution_with_no_governing_clause_is_refused(self, chain):
        store, snapshot, _ = chain("northstar_customer", "ORD-1001")
        # Constructed rather than resolved: every real account resolves
        # cancellation_fee to at least the SOP, so the empty case cannot be
        # produced from the pack. The guard still has to exist - a topic with
        # no authority is exactly the GS-024 shape, and a calculator that
        # returns a default there invents policy.
        empty = store.mint(
            EvidenceKind.POLICY_RESOLUTION,
            {
                "topic": "cancellation_fee",
                "account_id": "ACCT-001",
                "governing": None,
                "overridden": [],
                "deferred": [],
                "supporting": [],
                "excluded": [],
                "unresolved_conflict": None,
            },
        )
        with pytest.raises(NoBasis, match="nothing governs"):
            compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=empty)

    def test_an_unresolved_conflict_is_refused_rather_than_decided(self, chain):
        store, snapshot, _ = chain("northstar_customer", "ORD-1001")
        conflicted = store.mint(
            EvidenceKind.POLICY_RESOLUTION,
            {
                "topic": "cancellation_fee",
                "account_id": "ACCT-001",
                "governing": None,
                "overridden": [],
                "deferred": [],
                "supporting": [],
                "excluded": [],
                "unresolved_conflict": {
                    "tier": 1,
                    "clauses": [
                        {"clause_id": "a::§1", "citation": "A §1", "tier": 1, "params": {}},
                        {"clause_id": "b::§1", "citation": "B §1", "tier": 1, "params": {}},
                    ],
                    "differing_params": ["fee_inr"],
                },
            },
        )
        with pytest.raises(NoBasis, match="conflict"):
            compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=conflicted)

    def test_the_snapshot_and_the_resolution_must_agree_on_the_account(self, chain, db_path):
        store, snapshot, _ = chain("northstar_customer", "ORD-1001")
        principal = persona("northstar_customer")
        with PolicyResolver.open(db_path) as resolver:
            foreign = resolver.resolve("cancellation_fee", persona("lumenworks_customer"))
        handle = store.mint(EvidenceKind.POLICY_RESOLUTION, foreign.to_payload())
        # Both handles are this run's and this Principal's; they simply describe
        # two different accounts. Computing across them would apply one
        # customer's contract to another's shipment.
        with pytest.raises(WrongEvidence, match="account"):
            compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=handle)


class TestOutcomeShape:
    def test_the_outcome_mints_a_calc_handle(
        self,
        chain,
    ):
        store, snapshot, resolution = chain("northstar_customer", "ORD-1001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        assert outcome.calc_id
        assert store.provenance(outcome.calc_id) == (
            snapshot.evidence_id,
            resolution.evidence_id,
        )

    def test_the_outcome_serialises_for_a_fact_block(self, chain):
        store, snapshot, resolution = chain("northstar_customer", "ORD-1001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        payload = outcome.to_payload()
        import json

        assert json.loads(json.dumps(payload)) == payload
        assert payload["fee_inr"] == 0

    def test_every_returned_number_carries_the_clause_it_came_from(self, chain):
        store, snapshot, resolution = chain("lumenworks_customer", "ORD-2001")
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=resolution)
        assert isinstance(outcome, CancellationOutcome)
        # The invariant that keeps citation and computation from drifting: if a
        # number is present, so is its source.
        assert outcome.fee_source is not None
        assert outcome.window_source is not None
