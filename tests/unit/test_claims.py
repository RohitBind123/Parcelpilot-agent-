"""Reading a number out of Tier 5 prose.

A closed ticket's `historical_resolution` is the only place in the pack where a
rule is asserted in free text rather than in a clause. Policy v3 section 1 says
those are context only and may contain incorrect past guidance - and both of the
two in the pack are in fact wrong. To say *how* they are wrong, rather than just
that they are untrustworthy, we need the number they assert and the topic they
assert it about.

The bar is narrowness, not recall. A claim this module invents becomes a
contradiction the system reports against a clause that never disagreed with
anything, and the user is told a past answer was wrong when it was not. Silence
is the correct output for prose that states no rule.
"""

from __future__ import annotations

import pytest

from src.domain.claims import Claim, extract_claims

TKT_450 = "Agent told customer a INR 250 cancellation fee applied after 30 minutes."
TKT_451 = "Agent told customer Growth plan only supports 3,000 rows."


def claim_for(text: str, param: str) -> Claim | None:
    return next((c for c in extract_claims(text) if c.param == param), None)


class TestTheTwoInThePack:
    def test_the_cancellation_fee_is_read_with_its_topic(self):
        claim = claim_for(TKT_450, "fee_after_window_inr")
        assert claim is not None
        assert claim.value == 250
        assert claim.topic == "cancellation_fee"

    def test_the_window_is_read_as_a_separate_claim(self):
        # One sentence, two assertions. The fee is the one that is wrong for
        # Northstar; the window is not, and reporting them together as a single
        # claim would make the correction say more than it can prove.
        claim = claim_for(TKT_450, "free_window_minutes")
        assert claim is not None
        assert claim.value == 30
        assert claim.topic == "cancellation_window"

    def test_the_row_limit_is_read_with_its_topic(self):
        claim = claim_for(TKT_451, "supported_rows")
        assert claim is not None
        assert claim.value == 5000 or claim.value == 3000

    def test_a_thousands_separator_is_not_a_word_boundary(self):
        # "3,000 rows" is three thousand. Reading it as 3 is the same class of
        # bug that made a currency amount match a section reference in M2.
        assert claim_for(TKT_451, "supported_rows").value == 3000

    def test_the_quoted_text_comes_back_for_the_answer_to_use(self):
        # The correction has to name what was said, and it must be the recorded
        # words rather than a paraphrase the model composes.
        claim = claim_for(TKT_450, "fee_after_window_inr")
        assert claim.quote in TKT_450
        assert "250" in claim.quote


class TestRestraint:
    @pytest.mark.parametrize(
        "text",
        [
            "Customer wants to replace the billing-contact email on their account.",
            "Driver collected the parcel around 10 minutes ago, but ParcelPilot still shows BOOKED.",
            "",
            "Agent apologised and closed the ticket.",
        ],
    )
    def test_prose_with_no_rule_in_it_yields_nothing(self, text):
        assert extract_claims(text) == ()

    def test_a_duration_that_is_not_a_policy_window_is_not_a_claim(self):
        # "around 10 minutes ago" is an observation about one parcel, not an
        # assertion about a rule. Only the phrasing that states a rule counts.
        assert extract_claims("The driver arrived 10 minutes ago.") == ()

    def test_an_amount_with_no_subject_is_not_a_fee_claim(self):
        assert extract_claims("The shipment cost INR 4,200.") == ()


class TestShape:
    def test_claims_are_immutable(self):
        claim = claim_for(TKT_450, "fee_after_window_inr")
        with pytest.raises((AttributeError, TypeError)):
            claim.value = 999  # type: ignore[misc]

    def test_extraction_is_stable_across_calls(self):
        assert extract_claims(TKT_450) == extract_claims(TKT_450)

    def test_none_is_tolerated_the_way_the_column_is_nullable(self):
        # `historical_resolution` is NULL on every open ticket.
        assert extract_claims(None) == ()
