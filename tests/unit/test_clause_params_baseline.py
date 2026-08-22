"""The baseline is the validator (D24).

The extraction pipeline is real, but a regex that reads the corpus correctly
today can read it differently after any change to normalisation, segmentation
or a pattern. `clause_params_baseline.yaml` is reviewed once by a human against
the PDFs; this file asserts that extraction still produces exactly that.

The failure it prevents is the quiet one. A threshold that shifts from 4 hours
to 2 produces an answer that is wrong, plausible, and carries a citation to the
right clause - so nobody checking the citation would catch it. Here it is a red
test instead.

If a value in the baseline is wrong, the extractor is wrong. Do not edit the
YAML to make this pass; fix the extractor and regenerate with
`scripts/review_params.py --write`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.knowledge.clause_parser import parse_all
from src.knowledge.params import extract_params, score_confidence

BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "knowledge" / "clause_params_baseline.yaml"
)


@pytest.fixture(scope="module")
def baseline() -> dict[str, dict]:
    return yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def extracted() -> dict[str, dict]:
    return {c.clause_id: extract_params(c) for doc in parse_all() for c in doc.clauses}


class TestNoDrift:
    def test_the_baseline_exists_and_is_populated(self, baseline):
        assert baseline, f"{BASELINE_PATH} is empty; run scripts/review_params.py --write"

    def test_the_same_clauses_are_covered(self, baseline, extracted):
        assert set(baseline) == set(extracted)

    def test_every_clause_extracts_exactly_what_was_reviewed(self, baseline, extracted):
        # Compared per clause so a failure names the clause that drifted
        # rather than dumping the whole registry.
        drifted = {
            clause_id: {"reviewed": baseline[clause_id], "extracted": extracted[clause_id]}
            for clause_id in sorted(baseline)
            if baseline[clause_id] != extracted[clause_id]
        }
        assert not drifted, f"extraction drifted from the reviewed baseline: {drifted}"

    def test_nothing_in_the_baseline_is_flagged_for_review(self, baseline):
        # The review gate closed with zero flags. If a change reopens one, the
        # baseline needs re-reviewing before it is trusted again.
        flagged = {
            clause_id: score_confidence(clause_id, values).reasons
            for clause_id, values in baseline.items()
            if score_confidence(clause_id, values).score < 1.0
        }
        assert flagged == {}


class TestBaselineContent:
    """Spot checks that the reviewed file says what the PDFs say.

    Redundant with the drift test by design: if someone regenerates the
    baseline from a broken extractor, the drift test passes and these fail.
    """

    def test_the_discriminating_cancellation_numbers(self, baseline):
        sop = baseline["cancellation_and_service_credit_sop_v4::§1"]
        assert (sop["free_window_minutes"], sop["fee_after_window_inr"]) == (30, 250)

        northstar = baseline["northstar_logistics_enterprise_agreement::§2"]
        assert northstar["overrides"] is True and northstar["fee_inr"] == 0

        lumenworks = baseline["lumenworks_service_agreement::§2"]
        assert lumenworks["overrides"] is False

    def test_the_discriminating_credit_numbers(self, baseline):
        sop = baseline["cancellation_and_service_credit_sop_v4::§2"]
        assert (sop["threshold_hours"], sop["credit_flat_inr"], sop["credit_percent"]) == (
            2,
            500,
            10,
        )

        lumenworks = baseline["lumenworks_service_agreement::§3"]
        assert (lumenworks["threshold_hours"], lumenworks["credit_inr"]) == (4, 300)

    def test_the_supported_limit_is_not_the_failure_threshold(self, baseline):
        # TKT-451's recorded resolution conflated these two, and the whole
        # historical-contradiction case rests on them staying distinct.
        assert baseline["product_operations_guide_and_known_issues::§1"]["supported_rows"] == 5000
        assert (
            baseline["product_operations_guide_and_known_issues::KI-208"]["failure_threshold_rows"]
            == 3000
        )

    def test_the_current_and_deprecated_grids_are_both_present_and_differ(self, baseline):
        current = baseline["support_policy_v3_current::§3"]["targets"]
        deprecated = baseline["support_policy_v2_deprecated::§-"]["targets"]
        assert current["Enterprise"]["P1"]["amount"] == 30
        assert deprecated["Enterprise"]["P1"]["amount"] == 1
        assert current != deprecated
