"""The fact block (D15a): the part of an answer the model cannot edit.

Every figure a user sees is rendered here, in Python, from evidence handles. The
model writes the sentences around it. That is what makes a confidently wrong
number structurally impossible rather than merely unlikely - and it is why the
tests below care so much about how *absence* renders. A block that prints
"INR 0" where it means "nobody recorded a price" has laundered a missing value
into a fact, and the prose around it will be fluent and wrong.
"""

from __future__ import annotations

import json
import re

import pytest

from src.agent.facts import FactBlock, FactRow, compose

CANCELLATION_OVERRIDE = {
    "order_id": "ORD-1001",
    "order_status": "BOOKED",
    "cancellable": True,
    "fee_inr": 0.0,
    "fee_basis": "the agreement waives the fee on any BOOKED shipment before pickup",
    "fee_source": "agreement_waiver",
    "minutes_since_booking": 120,
    "free_window_minutes": None,
    "governing_clause": "northstar_logistics_enterprise_agreement::§2",
    "overridden_clauses": ("cancellation_and_service_credit_sop_v4::§1",),
    "deferred_clauses": (),
}

RESOLUTION = {
    "topic": "cancellation_fee",
    "account_id": "ACCT-001",
    "governing": {
        "clause_id": "northstar_logistics_enterprise_agreement::§2",
        "citation": "ParcelPilot - Northstar Logistics Enterprise Agreement §2",
        "tier": 1,
        "title": "Cancellation",
        "params": {"fee_inr": 0, "waiver": True, "overrides": True},
    },
    "overridden": [
        {
            "clause_id": "cancellation_and_service_credit_sop_v4::§1",
            "citation": "ParcelPilot Cancellation & Service Credit SOP v4 §1",
            "tier": 2,
            "title": "Order cancellation",
            "params": {"fee_after_window_inr": 250, "free_window_minutes": 30},
        }
    ],
    "deferred": [],
    "supporting": [],
    "excluded": [
        {
            "clause_id": "support_policy_v2_deprecated::§-",
            "citation": "ParcelPilot Support Policy v2 (DEPRECATED) §-",
            "tier": 4,
            "title": "Deprecated policy",
            "params": {},
            "reason": "deprecated_tier",
        }
    ],
}

CONFLICT = {
    "subject_id": "ORD-1001",
    "blocking": True,
    "conflicts": [
        {
            "conflict_class": "stale_status",
            "severity": "blocking",
            "detail": "ORD-1001 is recorded as BOOKED with no pickup confirmation, "
            "but TKT-504 reports the parcel was collected.",
            "sources": ["ORD-1001", "TKT-504", "product_operations_guide_and_known_issues::KI-211"],
            "confidence": 0.8,
            "instruction": "Before telling a customer that a pickup did not occur, "
            "verify the carrier status or wait through the known delay window.",
            "inference_note": "TKT-504 names SwiftShip and does not name an order.",
        }
    ],
}


@pytest.fixture
def block() -> FactBlock:
    return compose(calculation=CANCELLATION_OVERRIDE, resolution=RESOLUTION, conflicts=CONFLICT)


def row(block: FactBlock, label: str) -> FactRow | None:
    return next((r for r in block.rows if r.label == label), None)


class TestTheRowsOfTheArchitecture:
    def test_the_verdict_comes_first(self, block):
        assert block.rows[0].label == "Verdict"
        assert "no cancellation fee" in block.rows[0].value.lower()

    def test_the_amount_is_rendered_from_the_calculation(self, block):
        assert row(block, "Amount").value == "INR 0"

    def test_the_governing_clause_carries_its_tier_and_kind(self, block):
        value = row(block, "Governing").value
        assert "Northstar Logistics Enterprise Agreement §2" in value
        assert "tier 1" in value

    def test_the_overridden_clause_says_what_it_would_have_charged(self, block):
        # "Overridden: SOP v4 §1" alone tells the reader nothing about what
        # changed, and the whole point of surfacing an override is the delta.
        value = row(block, "Overridden").value
        assert "SOP v4 §1" in value
        assert "250" in value

    def test_the_deprecated_clause_is_excluded_and_never_governing(self, block):
        assert "Support Policy v2" in row(block, "Excluded").value
        assert "Support Policy v2" not in row(block, "Governing").value

    def test_the_basis_shows_the_inputs_the_number_came_from(self, block):
        value = row(block, "Basis").value
        assert "120" in value
        assert "BOOKED" in value

    def test_a_blocking_conflict_becomes_a_caution(self, block):
        assert row(block, "Caution") is not None
        assert "TKT-504" in row(block, "Caution").value

    def test_the_caution_carries_the_corpus_instruction(self, block):
        # KI-211 says what to do, and a model rewriting it can drop the part
        # that matters.
        assert "verify the carrier status" in row(block, "Caution").value.lower()

    def test_an_inference_is_labelled_as_one(self, block):
        assert "inferred" in row(block, "Caution").value.lower()


class TestAbsenceIsNotZero:
    def test_a_null_amount_renders_as_unknown(self):
        block = compose(
            calculation={**CANCELLATION_OVERRIDE, "fee_inr": None, "cancellable": False},
            resolution=RESOLUTION,
        )
        value = row(block, "Amount").value
        assert "0" not in value
        assert "unknown" in value.lower() or "not" in value.lower()

    def test_a_genuine_zero_still_renders_as_zero(self, block):
        assert row(block, "Amount").value == "INR 0"

    def test_a_missing_window_is_not_printed_as_zero_minutes(self):
        # Northstar's agreement states no free window at all - the waiver is
        # unconditional - so `free_window_minutes` is None, and rendering it as
        # "0 minutes" would say the opposite of what the clause says.
        #
        # Asserted with a word boundary, because "30 minutes" contains
        # "0 minutes" and a substring check fails on correct output. That is the
        # same mistake the grounding gate is being built to avoid, arriving here
        # first.
        block = compose(calculation=CANCELLATION_OVERRIDE, resolution=RESOLUTION)
        assert not re.search(r"\b0 minutes\b", json.dumps(block.to_payload()))

    def test_no_conflicts_means_no_caution_row_rather_than_an_empty_one(self):
        block = compose(calculation=CANCELLATION_OVERRIDE, resolution=RESOLUTION)
        assert row(block, "Caution") is None


class TestWhatTheGateWillUse:
    def test_every_figure_in_the_block_is_collectable(self, block):
        # The gate compares prose figures against this set, so it has to be a
        # property of the block rather than re-parsed from the rendered text.
        assert 0 in block.figures
        assert 250 in block.figures
        assert 120 in block.figures

    def test_the_cited_clauses_are_collectable(self, block):
        assert "northstar_logistics_enterprise_agreement::§2" in block.citable
        assert "cancellation_and_service_credit_sop_v4::§1" in block.citable

    def test_an_excluded_clause_is_not_citable(self, block):
        assert "support_policy_v2_deprecated::§-" not in block.citable

    def test_the_block_serialises_for_the_sse_event(self, block):
        payload = block.to_payload()
        assert json.loads(json.dumps(payload))
        assert payload["rows"][0]["label"] == "Verdict"

    def test_it_renders_as_text_for_the_model(self, block):
        text = block.render()
        assert "Verdict" in text and "Governing" in text
        assert "INR 0" in text

    def test_the_rendered_text_is_stable(self, block):
        assert block.render() == block.render()


class TestComposingWithoutACalculation:
    """Not every question has a number. GS-016 asks about a target in the
    abstract, and GS-018 compares two policies."""

    def test_a_resolution_alone_still_produces_a_block(self):
        block = compose(resolution=RESOLUTION)
        assert row(block, "Governing") is not None
        assert row(block, "Amount") is None

    def test_nothing_at_all_is_an_empty_block_not_a_crash(self):
        block = compose()
        assert block.rows == ()
        assert block.is_empty

    def test_an_empty_block_grounds_no_figures(self):
        assert compose().figures == frozenset()
