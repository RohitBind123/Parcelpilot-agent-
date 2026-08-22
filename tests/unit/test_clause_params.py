"""Typed parameters extracted from clause prose.

`params` is the bridge from prose to arithmetic. Calculators read it and never
the text, so a wrong value here is a wrong answer carrying a correct-looking
citation - the single worst failure this system can produce, because it is
invisible to a reader who checks the citation.

The values asserted below are the discriminating ones from
docs/01_DATA_PACK_FINDINGS.md sections 4 and 10: the numbers that make
ORD-1001 and ORD-2001 come out differently.
"""

from __future__ import annotations

import pytest

from src.knowledge.clause_parser import parse_all
from src.knowledge.params import Duration, extract_params, score_confidence


@pytest.fixture(scope="module")
def params() -> dict[str, dict]:
    return {c.clause_id: extract_params(c) for doc in parse_all() for c in doc.clauses}


def p(params, clause_id: str) -> dict:
    assert clause_id in params, f"{clause_id} missing"
    return params[clause_id]


SOP_CANCEL = "cancellation_and_service_credit_sop_v4::§1"
SOP_CREDIT = "cancellation_and_service_credit_sop_v4::§2"
SOP_APPROVAL = "cancellation_and_service_credit_sop_v4::§3"
NORTHSTAR_SLA = "northstar_logistics_enterprise_agreement::§1"
NORTHSTAR_CANCEL = "northstar_logistics_enterprise_agreement::§2"
NORTHSTAR_CREDIT = "northstar_logistics_enterprise_agreement::§3"
LUMEN_SLA = "lumenworks_service_agreement::§1"
LUMEN_CANCEL = "lumenworks_service_agreement::§2"
LUMEN_CREDIT = "lumenworks_service_agreement::§3"
POLICY_V3_TARGETS = "support_policy_v3_current::§3"
POLICY_V2_TARGETS = "support_policy_v2_deprecated::§-"
KI_176 = "product_operations_guide_and_known_issues::KI-176"
KI_208 = "product_operations_guide_and_known_issues::KI-208"
KI_211 = "product_operations_guide_and_known_issues::KI-211"
GUIDE_PLANS = "product_operations_guide_and_known_issues::§1"


class TestCancellationParams:
    def test_the_default_window_and_fee(self, params):
        # "No fee within 30 minutes of booking. After 30 minutes, charge INR 250"
        got = p(params, SOP_CANCEL)
        assert got["free_window_minutes"] == 30
        assert got["fee_after_window_inr"] == 250

    def test_the_default_says_it_can_be_waived_by_agreement(self, params):
        # This is the hook the Northstar override hangs on.
        assert p(params, SOP_CANCEL)["waivable_by_agreement"] is True

    def test_the_default_declares_itself_the_default(self, params):
        # `overrides: None` means "I am the baseline", distinct from an
        # agreement that declines to override (`False`).
        assert p(params, SOP_CANCEL)["overrides"] is None

    def test_status_rules_are_extracted(self, params):
        rules = p(params, SOP_CANCEL)["status_rules"]
        assert rules["DRAFT"] == "free"
        assert rules["BOOKED"] == "conditional"
        assert rules["PICKED_UP"] == "return_to_origin"
        assert rules["DELIVERED"] == "not_cancellable"

    def test_northstar_waives_the_fee_entirely(self, params):
        got = p(params, NORTHSTAR_CANCEL)
        assert got["overrides"] is True
        assert got["waiver"] is True
        assert got["fee_inr"] == 0

    def test_northstar_has_no_time_window_at_all(self, params):
        # "regardless of how long ago the shipment was booked" - the absence of
        # a window is the whole point, so it must be an explicit None rather
        # than a missing key a caller could read as zero.
        got = p(params, NORTHSTAR_CANCEL)
        assert "window_minutes" in got
        assert got["window_minutes"] is None

    def test_northstar_applies_only_before_pickup(self, params):
        assert p(params, NORTHSTAR_CANCEL)["applies_to_status"] == ["BOOKED"]

    def test_lumenworks_explicitly_declines_to_override(self, params):
        # "No special cancellation-fee waiver applies." A Tier 1 clause
        # existing is not a Tier 1 clause winning, and this is the case that
        # proves the resolver reads the agreement instead of assuming.
        got = p(params, LUMEN_CANCEL)
        assert got["overrides"] is False
        assert got.get("waiver") is not True


class TestCreditParams:
    def test_the_default_threshold_and_amounts(self, params):
        # "more than 2 hours past the end ... lower of INR 500 or 10%"
        got = p(params, SOP_CREDIT)
        assert got["threshold_hours"] == 2
        assert got["credit_flat_inr"] == 500
        assert got["credit_percent"] == 10
        assert got["credit_rule"] == "lower_of"

    def test_the_default_requires_carrier_fault_and_no_customer_fault(self, params):
        got = p(params, SOP_CREDIT)
        assert got["requires_carrier_fault"] is True
        assert got["requires_no_customer_fault"] is True

    def test_lumenworks_replaces_both_the_threshold_and_the_amount(self, params):
        # 4 hours instead of 2, and a flat INR 300 instead of lower(500, 10%).
        # At a 3-hour delay this flips the answer from eligible to not.
        got = p(params, LUMEN_CREDIT)
        assert got["overrides"] is True
        assert got["threshold_hours"] == 4
        assert got["credit_inr"] == 300

    def test_northstar_caps_the_month_but_keeps_the_sop(self, params):
        # "Unless this agreement states otherwise, the current SOP applies."
        got = p(params, NORTHSTAR_CREDIT)
        assert got["monthly_cap_inr"] == 5000
        assert got["overrides"] is False

    def test_manager_approval_threshold(self, params):
        assert p(params, SOP_APPROVAL)["manager_approval_above_inr"] == 1000


class TestResponseTargetGrids:
    def test_the_current_policy_grid_is_complete(self, params):
        grid = p(params, POLICY_V3_TARGETS)["targets"]
        assert set(grid) == {"Enterprise", "Growth", "Standard"}
        assert all(set(row) == {"P1", "P2", "P3"} for row in grid.values())

    def test_enterprise_p1_is_thirty_minutes_around_the_clock(self, params):
        cell = Duration.from_dict(p(params, POLICY_V3_TARGETS)["targets"]["Enterprise"]["P1"])
        assert (cell.amount, cell.unit) == (30, "minutes")
        assert cell.always_on is True
        assert cell.business is False

    def test_business_hours_are_distinguished_from_calendar_hours(self, params):
        # Growth P1 is "2 business hours" while Enterprise P2 is "2 hours".
        # Conflating them is a two-day error when the snapshot is a Sunday.
        grid = p(params, POLICY_V3_TARGETS)["targets"]
        growth = Duration.from_dict(grid["Growth"]["P1"])
        enterprise = Duration.from_dict(grid["Enterprise"]["P2"])
        assert (growth.amount, growth.unit, growth.business) == (2, "hours", True)
        assert (enterprise.amount, enterprise.unit, enterprise.business) == (2, "hours", False)

    def test_the_deprecated_grid_is_also_extracted(self, params):
        # Tier 4 stays parsed so "what changed in v3?" can be answered, and so
        # a leak into the citable set is detectable rather than invisible.
        grid = p(params, POLICY_V2_TARGETS)["targets"]
        assert Duration.from_dict(grid["Enterprise"]["P1"]).amount == 1
        assert Duration.from_dict(grid["Enterprise"]["P1"]).unit == "hours"

    def test_every_cell_differs_between_the_two_policy_versions(self, params):
        # Findings section 5: all nine differ. If one ever matches, the
        # tier-leakage test set has lost a case.
        new = p(params, POLICY_V3_TARGETS)["targets"]
        old = p(params, POLICY_V2_TARGETS)["targets"]
        for plan in new:
            for severity in new[plan]:
                assert new[plan][severity] != old[plan][severity], f"{plan} {severity}"

    def test_northstar_targets_override_the_grid(self, params):
        targets = p(params, NORTHSTAR_SLA)["targets"]
        p1 = Duration.from_dict(targets["P1"])
        assert (p1.amount, p1.unit, p1.always_on) == (15, "minutes", True)
        assert Duration.from_dict(targets["P2"]).amount == 1
        assert Duration.from_dict(targets["P3"]).business is True

    def test_lumenworks_has_no_weekend_coverage(self, params):
        got = p(params, LUMEN_SLA)
        assert got["weekend_coverage"] is False
        assert Duration.from_dict(got["targets"]["P1"]).business is True


class TestProductParams:
    def test_the_supported_row_limit(self, params):
        got = p(params, GUIDE_PLANS)
        assert got["supported_rows"] == 5000
        assert set(got["plans_included"]) == {"Growth", "Enterprise"}

    def test_a_resolved_issue_says_so_rather_than_staying_silent(self, params):
        # KI-176 carries "Resolved 18 July 2026" in prose but no "Status:" line,
        # so it used to extract to {}. That made "is this issue current?" a
        # question answered by the absence of a key, and absence is not a fact -
        # it is equally what a failed regex looks like. A consistency check that
        # corroborates a conflict with a known issue has to be able to tell the
        # two apart.
        assert p(params, KI_176)["issue_status"] == "Resolved"

    def test_the_known_issue_threshold_is_not_the_product_limit(self, params):
        # KI-208 fails around 3,000 while the supported limit stays 5,000.
        # Conflating them is exactly the wrong answer TKT-451 recorded.
        assert p(params, KI_208)["failure_threshold_rows"] == 3000
        assert p(params, GUIDE_PLANS)["supported_rows"] == 5000

    def test_the_webhook_delay_window(self, params):
        got = p(params, KI_211)
        assert got["delay_minutes"] == 20
        assert got["carrier"] == "SwiftShip"


class TestConfidence:
    def test_a_clause_with_all_its_numbers_scores_full(self, params):
        assert score_confidence(SOP_CANCEL, p(params, SOP_CANCEL)).score == 1.0

    def test_a_clause_that_should_have_numbers_but_has_none_is_flagged(self):
        flag = score_confidence(SOP_CANCEL, {})
        assert flag.score < 1.0
        assert "no_params_extracted" in flag.reasons

    def test_a_partial_grid_is_flagged(self):
        flag = score_confidence(POLICY_V3_TARGETS, {"targets": {"Enterprise": {"P1": {}}}})
        assert "incomplete_grid" in flag.reasons

    def test_prose_only_clauses_are_not_penalised(self, params):
        # Northstar §4 is "Dedicated CSM: Priya Mehta." There is nothing to
        # extract, and flagging it would train a reviewer to ignore flags.
        flag = score_confidence("northstar_logistics_enterprise_agreement::§4", {})
        assert flag.score == 1.0


class TestDuration:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("30 minutes, 24x7", Duration(30, "minutes", business=False, always_on=True)),
            ("2 hours", Duration(2, "hours", business=False, always_on=False)),
            ("1 business day", Duration(1, "days", business=True, always_on=False)),
            ("4 business hours", Duration(4, "hours", business=True, always_on=False)),
            ("2 business days", Duration(2, "days", business=True, always_on=False)),
            ("15 minutes, 24x7", Duration(15, "minutes", business=False, always_on=True)),
            ("8 business hours", Duration(8, "hours", business=True, always_on=False)),
        ],
    )
    def test_parses_every_duration_shape_in_the_corpus(self, text, expected):
        assert Duration.parse(text) == expected

    def test_round_trips_through_a_dict(self):
        original = Duration(30, "minutes", business=False, always_on=True)
        assert Duration.from_dict(original.to_dict()) == original

    def test_an_unparseable_duration_returns_none_rather_than_guessing(self):
        assert Duration.parse("as soon as possible") is None


class TestConfidenceTargeting:
    """A flag a reviewer learns to ignore is worse than no flag.

    "Does this clause state a value?" is asked of the text directly rather than
    inferred from its topics. Policy v3 §1 and §4 both mention response targets
    without stating one, and tagging them first_response_target is correct for
    retrieval - it should not also mean "expected to carry numbers".
    """

    @pytest.mark.parametrize(
        "clause_id",
        [
            "support_policy_v3_current::§1",  # "defines ... response targets", states none
            "support_policy_v3_current::§4",  # "if a response target is already breached"
            "support_policy_v3_current::§2",  # severity definitions, prose only
            "product_operations_guide_and_known_issues::KI-176",  # a resolution date
            "northstar_logistics_enterprise_agreement::§4",  # "Dedicated CSM: Priya Mehta."
        ],
    )
    def test_clauses_that_state_no_value_are_not_flagged(self, params, clause_id):
        assert score_confidence(clause_id, p(params, clause_id)).score == 1.0

    @pytest.mark.parametrize(
        "clause_id",
        [SOP_CANCEL, SOP_CREDIT, SOP_APPROVAL, LUMEN_CREDIT, NORTHSTAR_SLA, KI_211],
    )
    def test_clauses_that_do_state_values_are_flagged_when_empty(self, clause_id):
        assert "no_params_extracted" in score_confidence(clause_id, {}).reasons

    def test_the_whole_corpus_is_clean(self, params):
        # The end state of the review gate: nothing left to look at.
        flagged = {
            cid: score_confidence(cid, values).reasons
            for cid, values in params.items()
            if score_confidence(cid, values).score < 1.0
        }
        assert flagged == {}
