"""The failed-pickup service credit calculator.

Acceptance is GS-007 through GS-010. The interesting property is that an
agreement can replace the *threshold* as well as the amount, which means an
override can make a customer worse off: at three hours LumenWorks is ineligible
where a no-agreement account would be paid. A system that surfaces overrides
only when they are favourable is not reporting precedence.

The amount is also the place where "lower of" has to be read as a cap rather
than a choice, and where a credit with no shipment to attach to must come back
as None rather than a plausible number.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.credit import ServiceCreditOutcome, compute_service_credit
from src.domain.calculators.errors import NoBasis, WrongEvidence
from src.domain.evidence import EvidenceKind, EvidenceNotFound, open_evidence_store
from src.domain.resolver import PolicyResolver

SOP_CREDIT = "cancellation_and_service_credit_sop_v4::§2"
SOP_APPROVAL = "cancellation_and_service_credit_sop_v4::§3"
LUMENWORKS_CREDIT = "lumenworks_service_agreement::§3"
NORTHSTAR_CREDIT = "northstar_logistics_enterprise_agreement::§3"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture(scope="module")
def db_path():
    return get_settings().db_path


@pytest.fixture
def chain(db_path):
    stores: dict[str, tuple] = {}

    def factory(persona_id: str, order_id: str | None = None, *, hypothetical=None):
        principal = persona(persona_id)
        if persona_id not in stores:
            ctx = open_evidence_store(run_id="run_1", principal=principal)
            stores[persona_id] = (ctx, ctx.__enter__())
        store = stores[persona_id][1]

        if order_id:
            with open_repository(principal, db_path) as repo:
                payload = repo.get_order(order_id).to_payload()
        else:
            payload = {"order_id": None, "account_id": principal.account_id, **(hypothetical or {})}
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, payload)

        with PolicyResolver.open(db_path) as resolver:
            resolution = resolver.resolve("failed_pickup_credit", principal)
            approval = resolver.resolve("credit_approval", principal)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
        )
        # The approval line lives on its own topic (SOP v4 section 3), so it
        # arrives as its own resolution rather than being assumed.
        approval_handle = store.mint(EvidenceKind.POLICY_RESOLUTION, approval.to_payload())
        return store, snapshot, handle, approval_handle

    yield factory
    for ctx, _ in stores.values():
        ctx.__exit__(None, None, None)


class TestTheRealOrder:
    def test_lumenworks_gets_the_agreement_amount_not_the_sop_amount(self, chain):
        # GS-007. The SOP would pay lower(500, 10% of 2400) = 240. The agreement
        # replaces threshold and amount both: over 4 hours, flat 300.
        store, snapshot, resolution, approval = chain("lumenworks_customer", "ORD-2002")
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )

        assert outcome.eligible is True
        assert outcome.credit_inr == 300
        assert outcome.delay_hours == pytest.approx(4.5)
        assert outcome.threshold_hours == 4
        assert outcome.governing_clause == LUMENWORKS_CREDIT
        assert outcome.overridden_clauses == (SOP_CREDIT,)
        assert outcome.requires_manager_approval is False

    def test_the_sop_amount_is_reported_alongside_so_the_override_is_legible(self, chain):
        store, snapshot, resolution, approval = chain("lumenworks_customer", "ORD-2002")
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        # Surfacing the override means showing what it replaced, not just
        # naming the clause.
        assert outcome.default_credit_inr == 240
        assert outcome.default_threshold_hours == 2


class TestThresholdReplacementCutsBothWays:
    def test_at_three_hours_the_agreement_makes_lumenworks_ineligible(self, chain):
        # GS-008, the brief's own question. The override is against the
        # customer here, and must be surfaced exactly as loudly.
        store, snapshot, resolution, approval = chain(
            "lumenworks_customer",
            hypothetical={
                "delay_hours": 3.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 2400.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )

        assert outcome.eligible is False
        assert outcome.credit_inr is None
        assert outcome.threshold_hours == 4
        assert outcome.governing_clause == LUMENWORKS_CREDIT
        assert outcome.overridden_clauses == (SOP_CREDIT,)
        # Would have been paid under the default. That fact belongs in the answer.
        assert outcome.eligible_under_default is True

    def test_at_three_hours_an_account_with_no_agreement_is_eligible(self, chain):
        # GS-009. Word for word the same question, opposite answer.
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 3.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 1200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )

        assert outcome.eligible is True
        assert outcome.threshold_hours == 2
        assert outcome.governing_clause == SOP_CREDIT
        assert outcome.credit_inr == 120  # lower(500, 10% of 1200)


class TestTheAmountRule:
    def test_lower_of_is_a_cap_not_a_choice(self, chain):
        # GS-010. Ten percent of 15,000 is 1,500; the flat cap is 500. Reading
        # "lower of" as "either" both overpays and triggers a manager approval
        # that the correct reading never reaches.
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 15000.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )

        assert outcome.credit_inr == 500
        assert outcome.rate_basis == "flat_cap"
        assert outcome.requires_manager_approval is False

    def test_the_percentage_wins_when_it_is_the_smaller_number(self, chain):
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 1200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.credit_inr == 120
        assert outcome.rate_basis == "percentage_of_fee"

    def test_an_unknown_shipment_fee_yields_no_amount_rather_than_zero(self, chain):
        # GS-009's real shape: the question names no order, so there is no fee
        # to take ten percent of. A number here would be a hallucination
        # dressed as arithmetic.
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 3.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": None,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )

        assert outcome.eligible is True
        assert outcome.credit_inr is None
        assert outcome.amount_formula
        assert "shipment fee" in " ".join(outcome.warnings).lower()

    def test_a_flat_agreement_amount_needs_no_shipment_fee(self, chain):
        # LumenWorks pays a flat 300, so the same missing fee is not a problem.
        store, snapshot, resolution, approval = chain(
            "lumenworks_customer",
            hypothetical={
                "delay_hours": 5.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": None,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.credit_inr == 300
        assert outcome.rate_basis == "agreement_flat"


class TestEligibilityPreconditions:
    def test_no_carrier_fault_means_no_credit(self, chain):
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": False,
                "customer_fault": False,
                "shipment_fee_inr": 1200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.eligible is False
        assert outcome.ineligible_reason == "carrier_fault_not_established"
        assert outcome.credit_inr is None

    def test_customer_fault_disqualifies_even_with_carrier_fault(self, chain):
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": True,
                "customer_fault": True,
                "shipment_fee_inr": 1200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.eligible is False
        assert outcome.ineligible_reason == "customer_fault"

    def test_below_the_threshold_is_reported_as_such(self, chain):
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 1.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 1200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.eligible is False
        assert outcome.ineligible_reason == "below_threshold"


class TestApprovalAndCaps:
    def test_a_credit_above_the_line_requires_manager_approval(self, chain):
        # SOP v4 section 3. No natural case reaches it - the SOP caps at 500 -
        # so it is exercised with an explicit amount.
        store, snapshot, resolution, approval = chain(
            "beacon_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 1200.0,
                "goodwill_credit_inr": 2000.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
            proposed_credit_inr=2000.0,
        )
        assert outcome.credit_inr == 2000
        assert outcome.requires_manager_approval is True
        assert outcome.approval_source == SOP_APPROVAL

    def test_northstar_carries_its_monthly_cap_without_applying_it_per_claim(self, chain):
        # The cap is monthly and this calculator sees one claim, so it reports
        # the cap rather than silently enforcing it against a single credit.
        store, snapshot, resolution, approval = chain(
            "northstar_customer",
            hypothetical={
                "delay_hours": 6.0,
                "carrier_fault": True,
                "customer_fault": False,
                "shipment_fee_inr": 4200.0,
            },
        )
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert outcome.governing_clause == SOP_CREDIT
        assert outcome.monthly_cap_inr == 5000
        assert outcome.deferred_clauses == (NORTHSTAR_CREDIT,)


class TestEvidenceDiscipline:
    def test_a_resolution_for_the_wrong_topic_is_refused(self, chain, db_path):
        store, snapshot, _, _approval = chain("lumenworks_customer", "ORD-2002")
        principal = persona("lumenworks_customer")
        with PolicyResolver.open(db_path) as resolver:
            other = resolver.resolve("cancellation_fee", principal)
        handle = store.mint(EvidenceKind.POLICY_RESOLUTION, other.to_payload())
        with pytest.raises(WrongEvidence, match="failed_pickup_credit"):
            compute_service_credit(store, snapshot_id=snapshot, resolution_id=handle)

    def test_a_bare_order_id_is_refused(self, chain):
        store, _, resolution, _approval = chain("lumenworks_customer", "ORD-2002")
        # The refusal that forces the chain: without it a model can compute a
        # credit having never consulted precedence.
        with pytest.raises(EvidenceNotFound, match="ORD-2002"):
            compute_service_credit(store, snapshot_id="ORD-2002", resolution_id=resolution)

    def test_a_resolution_with_no_basis_is_refused(self, chain):
        store, snapshot, _, _approval = chain("lumenworks_customer", "ORD-2002")
        empty = store.mint(
            EvidenceKind.POLICY_RESOLUTION,
            {
                "topic": "failed_pickup_credit",
                "account_id": "ACCT-002",
                "governing": None,
                "overridden": [],
                "deferred": [],
                "supporting": [],
                "excluded": [],
                "unresolved_conflict": None,
            },
        )
        with pytest.raises(NoBasis):
            compute_service_credit(store, snapshot_id=snapshot, resolution_id=empty)


class TestOutcomeShape:
    def test_the_outcome_mints_a_calc_handle_with_provenance(self, chain):
        store, snapshot, resolution, approval = chain("lumenworks_customer", "ORD-2002")
        outcome = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        )
        assert isinstance(outcome, ServiceCreditOutcome)
        # All three, because the approval resolution genuinely informed the
        # answer. A chain that omitted it would let the grounding gate
        # accept an approval claim with nothing behind it.
        assert store.provenance(outcome.calc_id) == (
            snapshot.evidence_id,
            resolution.evidence_id,
            approval.evidence_id,
        )

    def test_the_outcome_serialises(self, chain):
        import json

        store, snapshot, resolution, approval = chain("lumenworks_customer", "ORD-2002")
        payload = compute_service_credit(
            store,
            snapshot_id=snapshot,
            resolution_id=resolution,
            approval_resolution_id=approval,
        ).to_payload()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["credit_inr"] == 300
