"""The grounding gate (D16), and the mistake it is built not to make.

Three times now a substring filter over evidence has flagged a *correct* answer:

  GS-019  forbids asserting "pickup did not occur", and KI-211's instruction
          reads "Before telling a customer that a pickup did not occur, verify
          the carrier status" - the forbidden sentence as a prohibition of itself
  GS-027  forbids leaking "waive" to Beacon, and SOP v4 §1 - general policy every
          customer may read - says "unless a customer agreement explicitly waives
          the cancellation fee"
  M7      a test of mine asserting "0 minutes" never appears, which "30 minutes"
          contains

So the gate grades *asserted claims*, not strings. Numbers are the exception and
are checked deterministically, because "the prose contains a figure the evidence
does not" needs no judgement and must not be subject to any.
"""

from __future__ import annotations

import pytest

from src.agent.facts import compose
from src.agent.grounding import (
    Claim,
    GateOutcome,
    Verdict,
    check_figures,
    ground,
    unquoted_figures,
)

RESOLUTION = {
    "topic": "cancellation_fee",
    "account_id": "ACCT-001",
    "governing": {
        "clause_id": "northstar_logistics_enterprise_agreement::§2",
        "citation": "ParcelPilot - Northstar Logistics Enterprise Agreement §2",
        "tier": 1,
        "title": "Cancellation",
        "params": {"fee_inr": 0, "waiver": True},
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
    "excluded": [],
}
CALCULATION = {
    "order_id": "ORD-1001",
    "order_status": "BOOKED",
    "cancellable": True,
    "fee_inr": 0.0,
    "minutes_since_booking": 120,
    "governing_clause": "northstar_logistics_enterprise_agreement::§2",
}

SOURCES = {
    "northstar_logistics_enterprise_agreement::§2": (
        "Northstar may cancel any BOOKED shipment before pickup with no cancellation fee, "
        "regardless of how long ago the shipment was booked."
    ),
    "cancellation_and_service_credit_sop_v4::§1": (
        "No fee within 30 minutes of booking. After 30 minutes, charge INR 250 unless a "
        "customer agreement explicitly waives the cancellation fee."
    ),
    # Verbatim from the pack, because a truncated fixture would make the
    # support check pass or fail for reasons the real clause does not have.
    "product_operations_guide_and_known_issues::KI-211": (
        "SwiftShip pickup confirmation webhooks can arrive up to 20 minutes late. A parcel "
        "may physically be collected while ParcelPilot still shows BOOKED. Before telling a "
        "customer that a pickup did not occur, verify the carrier status or wait through "
        "the known delay window."
    ),
}


@pytest.fixture
def block():
    return compose(calculation=CALCULATION, resolution=RESOLUTION)


class StubExtractor:
    """Stands in for the cheap model that splits prose into atomic claims."""

    def __init__(self, *claims: str, raises: Exception | None = None):
        self.claims = list(claims)
        self.raises = raises
        self.seen: list[str] = []

    def extract(self, prose: str) -> list[str]:
        self.seen.append(prose)
        if self.raises is not None:
            raise self.raises
        return list(self.claims)


class TestFiguresAreCheckedWithoutAModel:
    def test_a_figure_in_the_block_passes(self, block):
        assert check_figures("There is no fee: INR 0 applies.", block, SOURCES) == ()

    def test_an_invented_figure_fails(self, block):
        # The failure the whole design exists to prevent.
        assert check_figures("A fee of INR 175 applies.", block, SOURCES) == ((175.0, "inr"),)

    def test_a_figure_quoted_from_a_cited_clause_passes(self, block):
        # "the standard INR 250 charge does not apply" is correct and necessary;
        # 250 is in the overridden clause the block already names.
        assert check_figures("The standard INR 250 charge does not apply.", block, SOURCES) == ()

    def test_a_figure_from_a_source_the_answer_read_passes(self, block):
        # KI-211's 20 minutes is not in the block, and the answer is entitled to
        # it because the clause is in the evidence.
        assert check_figures("Webhooks can lag 20 minutes.", block, SOURCES) == ()

    def test_a_figure_from_a_source_nobody_read_fails(self, block):
        assert check_figures("The cap is INR 5000.", block, {}) == ((5000.0, "inr"),)

    def test_thousands_separators_are_one_number(self, block):
        sources = {"x": "the supported limit remains 5,000 rows"}
        assert check_figures("The limit is 5000 rows.", block, sources) == ()

    @pytest.mark.parametrize(
        "prose",
        [
            "See ORD-1001 for details.",
            "Under SOP v4 §1 the fee applies.",
            "Ticket TKT-504 reports a collection.",
            "Account ACCT-001 holds the agreement.",
            "KI-211 describes the lag.",
        ],
    )
    def test_identifiers_and_references_are_not_figures(self, block, prose):
        # They contain digits and assert nothing. Treating them as claims would
        # fail every answer that cites anything.
        assert check_figures(prose, block, SOURCES) == ()

    def test_a_year_in_a_quoted_date_is_not_invented(self, block):
        sources = {"x": "Opened: 12 August 2026 Status: Monitoring"}
        assert check_figures("The issue was opened on 12 August 2026.", block, sources) == ()

    def test_several_inventions_are_all_reported(self, block):
        assert set(check_figures("INR 175 or INR 900.", block, SOURCES)) == {
            (175.0, "inr"),
            (900.0, "inr"),
        }


class TestTheSubstringMistake:
    """The three cases that made a naive filter fail on correct answers."""

    def test_the_correct_answer_to_gs_019_passes_the_gate(self, block):
        # It carries KI-211's instruction verbatim, so it contains the sentence
        # the golden set forbids asserting. The gate grades claims against
        # evidence, so this is simply a well-grounded answer.
        prose = (
            "Before telling you that a pickup did not occur, we verify the carrier status. "
            "A parcel may physically be collected while ParcelPilot still shows BOOKED."
        )
        outcome = ground(
            prose,
            block=block,
            sources=SOURCES,
            extractor=StubExtractor(
                "the carrier status is verified before saying a pickup did not occur",
                "a parcel may be collected while ParcelPilot still shows BOOKED",
            ),
        )
        assert outcome.verdict is Verdict.PASSED, outcome.failures

    def test_general_policy_mentioning_waivers_is_not_a_private_leak(self, block):
        # GS-027. SOP v4 §1 says "unless a customer agreement explicitly waives
        # the cancellation fee" and every customer may read it.
        outcome = ground(
            "After 30 minutes a fee of INR 250 applies unless a customer agreement "
            "explicitly waives it.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the SOP charges INR 250 after 30 minutes"),
        )
        assert outcome.verdict is Verdict.PASSED

    def test_there_is_no_deterministic_forbidden_assertion_check(self):
        # `must_not_assert` is a semantic judgement. Every Python version tried
        # during M7 flagged the correct answer: content-word containment reads
        # "the carrier status is verified before saying a pickup did not occur"
        # as asserting that the pickup did not occur, because all the words are
        # present. The first version of this test passed only because it used a
        # forbidden string with no vocabulary overlap - a weakened test hiding
        # an unsound checker.
        #
        # It belongs in the LLM-judged layer (M11). This test exists so nobody
        # adds it back here without reading why it was removed.
        import src.agent.grounding as grounding

        assert not hasattr(grounding, "asserts")


class TestClaimsMustMapToEvidence:
    def test_a_claim_supported_by_a_cited_clause_passes(self, block):
        outcome = ground(
            "Your agreement waives the fee.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the agreement waives the cancellation fee"),
        )
        assert outcome.verdict is Verdict.PASSED

    def test_an_unsupported_claim_is_reported_with_its_text(self, block):
        outcome = ground(
            "Refunds are processed within five working days.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("refunds are processed within five working days"),
        )
        assert outcome.verdict is not Verdict.PASSED
        assert any("refunds" in f.claim.text.lower() for f in outcome.failures)

    def test_every_claim_is_graded_not_just_the_first(self, block):
        outcome = ground(
            "text",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor(
                "the agreement waives the cancellation fee",
                "refunds take five working days",
                "we also deliver on Sundays",
            ),
        )
        assert len(outcome.failures) == 2

    def test_an_extractor_that_fails_does_not_silently_pass_the_prose(self, block):
        # A gate that treats its own outage as a clean bill of health is worse
        # than no gate: it is a clean bill of health nobody checked.
        outcome = ground(
            "Anything at all.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor(raises=RuntimeError("provider down")),
        )
        assert outcome.verdict is Verdict.UNCHECKED
        assert outcome.verdict is not Verdict.PASSED

    def test_an_answer_that_asserts_nothing_has_nothing_to_ground(self, block):
        """A successful extraction of zero claims is a pass, not an outage.

        This test asserted UNCHECKED until a greeting escalated to a human in
        the browser. The two states it was conflating are now separated at the
        boundary: `_to_claims` raises when the extractor fails, so reaching
        here with an empty list means the answer genuinely asserted nothing
        about policies or records.
        """
        outcome = ground(
            "Hello, how can I help?", block=block, sources=SOURCES, extractor=StubExtractor()
        )
        assert outcome.verdict is Verdict.PASSED
        assert outcome.claims == ()

    def test_asserting_nothing_still_fails_on_an_invented_figure(self, block):
        # A number the sources do not contain is an assertion however the
        # sentence around it is phrased, so the figure check is not skipped.
        outcome = ground(
            "I can help with lots of things, about 9999 of them.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor(),
        )
        assert outcome.verdict is Verdict.FAILED


class TestSufficiencyIsStructural:
    """The model may say the evidence is insufficient. It may never be the
    thing that says the evidence is sufficient - given three attempts an LLM
    grading its own evidence quietly lowers its bar."""

    def test_no_governing_clause_cannot_be_talked_into_a_pass(self):
        empty = compose()
        outcome = ground(
            "The retention period is seven years.",
            block=empty,
            sources={},
            extractor=StubExtractor("the retention period is seven years"),
        )
        assert outcome.verdict is Verdict.NO_BASIS

    def test_no_basis_needs_a_claim_to_have_no_basis_for(self):
        """The extractor now runs first, and the ordering had to invert.

        Deciding NO_BASIS on "nothing was retrieved" alone cannot tell an
        answer that needed a source and has none from an answer that needed no
        source. The first must be declined; the second is a greeting. So the
        question "does this answer assert anything?" is asked before the
        question "is there anything to assert it against?".
        """
        extractor = StubExtractor("the retention period is seven years")
        outcome = ground("prose", block=compose(), sources={}, extractor=extractor)
        assert outcome.verdict is Verdict.NO_BASIS
        assert extractor.seen == ["prose"]

    def test_an_answer_with_no_claims_and_no_evidence_is_not_declined(self):
        # A greeting in a conversation that touched no tool at all.
        outcome = ground("Hello.", block=compose(), sources={}, extractor=StubExtractor())
        assert outcome.verdict is Verdict.PASSED

    def test_a_block_with_a_governing_clause_has_a_basis(self, block):
        outcome = ground(
            "Your agreement waives the fee.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the agreement waives the cancellation fee"),
        )
        assert outcome.verdict is Verdict.PASSED


class TestTheOutcome:
    def test_a_pass_carries_the_prose_unchanged(self, block):
        prose = "Your agreement waives the fee."
        outcome = ground(
            prose,
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the agreement waives the cancellation fee"),
        )
        assert outcome.prose == prose

    def test_a_failure_names_what_was_unsupported(self, block):
        outcome = ground(
            "Refunds take five days.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("refunds take five days"),
        )
        assert isinstance(outcome, GateOutcome)
        assert isinstance(outcome.failures[0].claim, Claim)
        assert outcome.failures[0].reason

    def test_an_invented_figure_is_reported_separately_from_a_claim(self, block):
        outcome = ground(
            "A fee of INR 175 applies.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the fee is INR 175"),
        )
        assert outcome.invented_figures == ((175.0, "inr"),)

    def test_an_invented_figure_is_never_recoverable_by_re_retrieval(self, block):
        # A number the evidence does not contain will not be found by searching
        # for it; the honest exit is to drop the prose.
        outcome = ground(
            "A fee of INR 175 applies.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor("the fee is INR 175"),
        )
        assert outcome.repairable is False


class TestUnquotedFigures:
    def test_it_ignores_text_inside_quotation_marks(self):
        # A verbatim quotation is grounded by the clause it came from, and the
        # clause was already checked when it entered the evidence set.
        assert unquoted_figures('The clause says "charge INR 250" here.') == set()

    def test_it_still_sees_figures_outside_the_quotation(self):
        assert unquoted_figures('It says "charge INR 250" so you owe INR 400.') == {(400.0, "inr")}


class TestTheLlmExtractorBoundary:
    """`_to_claims` is where a provider response becomes claims.

    It must distinguish two things that used to look identical. An unusable
    response *raises*, so a broken extractor stops an answer. A well-formed
    empty list *returns empty*, so an answer that asserts nothing about
    policies or records is not treated as an extractor failure.

    Collapsing those two into "return []" is what made "tell me what you can
    do" escalate to a human: the gate could not tell an answer with nothing to
    ground from a gate that had failed to ground it.
    """

    @pytest.fixture
    def to_claims(self):
        from src.agent.claims_llm import _to_claims

        return _to_claims

    def test_a_clean_response_becomes_claims(self, to_claims):
        assert to_claims({"claims": ["the fee is waived", "the order is BOOKED"]}) == [
            "the fee is waived",
            "the order is BOOKED",
        ]

    def test_json_arriving_as_a_string_is_parsed(self, to_claims):
        assert to_claims('{"claims": ["a"]}') == ["a"]

    @pytest.mark.parametrize("raw", ["not json", {"other": []}, None, [], {"claims": None}])
    def test_anything_unusable_raises_rather_than_looking_like_silence(self, to_claims, raw):
        from src.agent.claims_llm import ExtractionError

        # `ground` catches this and returns UNCHECKED, which is not a pass.
        with pytest.raises(ExtractionError):
            to_claims(raw)

    def test_a_well_formed_empty_list_is_not_an_error(self, to_claims):
        # The extractor saying "this answer asserts nothing" is a real answer.
        assert to_claims({"claims": []}) == []

    def test_blank_claims_are_dropped(self, to_claims):
        assert to_claims({"claims": ["real", "  ", ""]}) == ["real"]

    def test_the_list_is_capped(self, to_claims):
        from src.agent.claims_llm import MAX_CLAIMS

        assert len(to_claims({"claims": [f"c{i}" for i in range(100)]})) == MAX_CLAIMS

    def test_the_prompt_shows_what_not_to_extract(self):
        """Self-description is not a claim about the corpus.

        Asked "are you hallucinating?", the model explains how it works, and
        the extractor read "all figures are calculated from system records"
        as an assertion needing a clause. No clause could ever support it, so
        the answer escalated. The prompt now carries worked examples, which
        steer an extraction far more reliably than an abstract rule.
        """
        from src.agent.claims_llm import _SYSTEM

        assert "Do NOT extract" in _SYSTEM
        assert "DO extract" in _SYSTEM
        # The exact shape that failed, given verbatim.
        assert "calculated from system records" in _SYSTEM

    def test_the_prompt_tells_the_model_a_prohibition_is_not_a_claim(self):
        # The linguistic judgement the gate depends on, asked for explicitly.
        from src.agent.claims_llm import _SYSTEM

        assert "not to conclude" in _SYSTEM or "warns against" in _SYSTEM

    def test_a_broken_extractor_still_does_not_pass_the_gate(self, block):
        # The property the old empty-is-unchecked rule was protecting, kept -
        # now carried by the exception rather than by the empty list.
        outcome = ground(
            "Some prose.",
            block=block,
            sources=SOURCES,
            extractor=StubExtractor(raises=RuntimeError("bad response")),
        )
        assert outcome.verdict is Verdict.UNCHECKED


class TestLayoutIsNotAClaim:
    """Markdown structure is not an assertion.

    A numbered list was read as one unsupported figure per item, so any answer
    that enumerated its points failed the gate. It was found on a capability
    answer - "1. Order lookups 2. Cancellation ..." - but nothing about the
    bug was specific to those: a domain answer listing three findings would
    have been rejected for its own bullet numbers.
    """

    LISTED = (
        "Here is what I can help with:\n\n"
        "1. **Order lookups** - the status of any shipment.\n"
        "2. **Cancellation** - whether an order can be cancelled.\n"
        "3. **Credits** - whether a delay qualifies.\n"
    )

    def test_ordered_list_markers_are_not_figures(self):
        assert unquoted_figures(self.LISTED) == set()

    def test_a_bracketed_marker_is_also_layout(self):
        assert unquoted_figures("1) first\n2) second\n") == set()

    def test_a_two_digit_marker_is_layout(self):
        assert unquoted_figures("10. tenth item\n") == set()

    def test_real_figures_in_a_list_are_still_read(self):
        # Only the marker is layout. What the item says is still a claim.
        found = unquoted_figures("1. The fee is INR 250.\n2. The window is 30 minutes.\n")
        assert found == {(250.0, "inr"), (30.0, "minutes")}

    def test_a_number_mid_sentence_is_not_treated_as_a_marker(self):
        # The marker rule is anchored to the start of a line, so an ordinary
        # sentence beginning with a figure keeps it.
        assert (250.0, "inr") in unquoted_figures("The charge is INR 250. It applies after.")

    def test_a_listed_answer_passes_the_gate(self, block):
        outcome = ground(self.LISTED, block=block, sources=SOURCES, extractor=StubExtractor())
        assert outcome.verdict is Verdict.PASSED


class TestIdentifiersAreNotQuantities:
    """Digits that identify rather than count. Every one of these appeared in a
    real answer during M7, and treating any of them as a claim fails a correct
    response for a figure nobody stated."""

    @pytest.mark.parametrize(
        ("prose", "expected"),
        [
            ("A P1 ticket has a 30 minute target.", {(30.0, "minutes")}),
            ("Under Support Policy v3 the target is 30 minutes.", {(30.0, "minutes")}),
            ("See ORD-1001, TKT-504 and KI-211.", set()),
            ("Section §3.1 applies.", set()),
            ("The fee is INR 250.", {(250.0, "inr")}),
            ("P2 tickets get 2 business hours.", {(2.0, "hours")}),
        ],
    )
    def test_only_quantities_are_figures(self, prose, expected):
        assert unquoted_figures(prose) == expected

    def test_the_gate_and_the_block_extract_figures_the_same_way(self):
        # One function, not two patterns kept in step. A divergence here shows
        # up as the gate failing an answer for a figure the block put in it.
        import src.agent.grounding as gate
        from src.agent.facts import figures_in

        assert gate.figures_in is figures_in
        assert figures_in("INR 250 after 30 minutes") == {(250.0, "inr"), (30.0, "minutes")}


class TestTheUnitIsPartOfTheFigure:
    """Policy v3 §3 says Enterprise P1 is "30 minutes, 24x7" and Standard P2 is
    "1 business day". The bare number 1 is therefore grounded by that grid - so
    a gate checking bare numbers accepts "the target is 1 hour", which is the
    deprecated v2 answer GS-017 exists to catch."""

    @pytest.fixture
    def grid(self):
        return {
            "support_policy_v3_current::§3": (
                "Plan P1 P2 P3\nEnterprise 30 minutes, 24x7 2 hours 1 business day\n"
                "Growth 2 business hours 4 business hours 2 business days"
            )
        }

    def test_the_right_number_with_the_wrong_unit_is_caught(self, block, grid):
        assert check_figures("The target is 1 hour.", block, grid) == ((1.0, "hours"),)

    def test_the_number_in_its_own_unit_passes(self, block, grid):
        assert check_figures("The target is 1 business day.", block, grid) == ()

    def test_the_correct_answer_passes(self, block, grid):
        assert check_figures("The target is 30 minutes, 24x7.", block, grid) == ()

    def test_a_figure_with_no_unit_is_grounded_by_the_same_value_anywhere(self, block, grid):
        # Prose that states no unit makes no claim about units, and demanding
        # one would fail ordinary writing.
        assert check_figures("There are 2 listed above yours.", block, grid) == ()
        assert check_figures("Exactly 30 of them apply.", block, grid) == ()

    def test_a_value_that_appears_nowhere_is_still_caught(self, block, grid):
        assert check_figures("There are 77 of them.", block, grid) == ((77.0, None),)


class TestInflectionDoesNotBreakSupport:
    """The support check compares stems, not similarity scores.

    The first version used difflib at a 0.85 cutoff and rejected "the fee is
    waived" against a clause that says "waives" - 0.833. A correct claim
    reported unsupported means a correct answer dropped, which is the failure
    mode a grounding gate is least allowed to have. The demo caught it.
    """

    @pytest.mark.parametrize(
        "claim",
        [
            "the fee is waived",
            "the agreement waives the fee",
            "no cancellation fees apply",
            "the shipment was booked 120 minutes ago",
            "cancelling is permitted before pickup",
        ],
    )
    def test_a_paraphrase_of_the_clause_is_supported(self, block, claim):
        outcome = ground(claim, block=block, sources=SOURCES, extractor=StubExtractor(claim))
        assert outcome.verdict is Verdict.PASSED, outcome.failures

    @pytest.mark.parametrize(
        "claim",
        [
            "refunds are processed within five working days",
            "we deliver on Sundays",
            "the retention period is seven years",
        ],
    )
    def test_something_the_clause_does_not_say_is_still_unsupported(self, block, claim):
        # Or the test above just proves the check accepts everything.
        outcome = ground(claim, block=block, sources=SOURCES, extractor=StubExtractor(claim))
        assert outcome.verdict is Verdict.FAILED

    def test_the_stems_that_mattered(self):
        from src.agent.grounding import _stem

        assert _stem("waived") == _stem("waives") == _stem("waive")
        assert _stem("fees") == _stem("fee")
        assert _stem("minutes") == _stem("minute")
        assert _stem("cancelled") == _stem("cancel")
