"""Assembling a finished answer, and declining rather than degrading.

The order is the design: collect what the tools returned, render the facts in
Python, grade the prose against them, and on failure drop the prose rather than
shorten it until it passes. A system that trims an unsupported answer has not
become more truthful, it has become vaguer - and vagueness is harder to catch
than a wrong number.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from src.agent.answer import (
    Answer,
    assemble,
    evidence_from,
    is_new_evidence,
    repair_queries,
)
from src.agent.escalation import DeclineReason
from src.agent.grounding import Claim, Failure, GateOutcome, Verdict
from src.auth.personas import get_persona, to_principal

RESOLUTION = {
    "topic": "cancellation_fee",
    "governing": {
        "clause_id": "northstar_logistics_enterprise_agreement::§2",
        "citation": "ParcelPilot - Northstar Logistics Enterprise Agreement §2",
        "tier": 1,
        "title": "Cancellation",
        "params": {"fee_inr": 0},
    },
    "overridden": [
        {
            "clause_id": "cancellation_and_service_credit_sop_v4::§1",
            "citation": "ParcelPilot Cancellation & Service Credit SOP v4 §1",
            "tier": 2,
            "title": "Order cancellation",
            "params": {"fee_after_window_inr": 250},
        }
    ],
    "excluded": [],
}


def tool_message(name: str, payload: dict) -> dict:
    return {"role": "tool", "name": name, "tool_call_id": "c", "content": json.dumps(payload)}


CONVERSATION = [
    {"role": "user", "content": "can I cancel ORD-1001?"},
    tool_message(
        "get_order", {"snapshot_id": "snap_1", "order_id": "ORD-1001", "status": "BOOKED"}
    ),
    tool_message(
        "resolve_policy",
        {
            "resolution_id": "res_1",
            "governing_clause": "northstar_logistics_enterprise_agreement::§2",
        },
    ),
    tool_message(
        "compute_cancellation_fee",
        {
            "calc_id": "calc_1",
            "order_id": "ORD-1001",
            "order_status": "BOOKED",
            "cancellable": True,
            "fee_inr": 0.0,
            "minutes_since_booking": 120,
        },
    ),
    tool_message(
        "search_policy",
        {
            "clauses": [
                {
                    "clause_id": "northstar_logistics_enterprise_agreement::§2",
                    "citation": "ParcelPilot - Northstar Logistics Enterprise Agreement §2",
                    "tier": 1,
                    "title": "Cancellation",
                    "text": "Northstar may cancel any BOOKED shipment before pickup with no "
                    "cancellation fee, regardless of how long ago the shipment was booked.",
                    "citable": True,
                },
                {
                    "clause_id": "support_policy_v2_deprecated::§-",
                    "citation": "ParcelPilot Support Policy v2 (DEPRECATED) §-",
                    "tier": 4,
                    "title": "Deprecated",
                    "text": "Enterprise P1 first response: 1 hour.",
                    "citable": False,
                },
            ]
        },
    ),
]


class StubExtractor:
    def __init__(self, *claims, raises=None):
        self.claims, self.raises = list(claims), raises

    def extract(self, prose):
        if self.raises:
            raise self.raises
        return list(self.claims)


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


def run(prose, extractor, *, messages=None, resolution=RESOLUTION, subject="this question"):
    return assemble(
        prose,
        messages=messages if messages is not None else CONVERSATION,
        resolution=resolution,
        principal=persona("northstar_customer"),
        thread_id="t1",
        question="can I cancel ORD-1001?",
        extractor=extractor,
        subject=subject,
    )


class TestReadingTheEvidenceBackOutOfTheConversation:
    def test_the_calculation_is_found(self):
        assert evidence_from(CONVERSATION)["calculation"]["fee_inr"] == 0.0

    def test_only_citable_clauses_become_sources(self):
        # A tier-4 clause reaching the source set would let the gate accept
        # "1 hour" as a grounded figure - the exact leak GS-017 tests for.
        sources = evidence_from(CONVERSATION)["sources"]
        assert "northstar_logistics_enterprise_agreement::§2" in sources
        assert "support_policy_v2_deprecated::§-" not in sources

    def test_a_denied_call_contributes_nothing(self):
        messages = [*CONVERSATION, tool_message("get_order", {"denied": True, "reason": "x"})]
        assert evidence_from(messages)["sources"] == evidence_from(CONVERSATION)["sources"]

    def test_an_errored_call_contributes_nothing(self):
        messages = [*CONVERSATION, tool_message("compute_cancellation_fee", {"error": True})]
        assert evidence_from(messages)["calculation"]["fee_inr"] == 0.0

    def test_unparseable_content_is_skipped_rather_than_fatal(self):
        messages = [*CONVERSATION, {"role": "tool", "name": "x", "content": "not json"}]
        assert evidence_from(messages)["calculation"]

    def test_it_works_from_messages_alone(self):
        # The checkpointer persists the conversation and nothing else, so a run
        # resumed in another process must rebuild the same block.
        assert evidence_from(list(CONVERSATION)) == evidence_from(CONVERSATION)


class TestAPassingAnswer:
    def test_the_prose_survives_unchanged(self):
        prose = "Your agreement waives the cancellation fee, so no fee applies."
        answer = run(prose, StubExtractor("the agreement waives the cancellation fee"))
        assert answer.prose == prose
        assert not answer.declined

    def test_the_block_is_rendered_beside_it(self):
        answer = run("No fee applies.", StubExtractor("no cancellation fee applies"))
        assert "Governing" in answer.block.render()
        assert "INR 0" in answer.block.render()

    def test_it_serialises_for_the_sse_events(self):
        answer = run("No fee applies.", StubExtractor("no cancellation fee applies"))
        assert json.loads(json.dumps(answer.to_payload()))


class TestDecliningRatherThanDegrading:
    def test_an_unsupported_claim_drops_the_prose_entirely(self):
        answer = run(
            "Refunds take five working days.", StubExtractor("refunds take five working days")
        )
        assert answer.prose == ""
        assert answer.declined

    def test_the_facts_survive_the_decline(self):
        # The block was computed in Python and is not in doubt. Throwing it
        # away with the prose would punish the user for the model's failure.
        answer = run("Refunds take five days.", StubExtractor("refunds take five days"))
        assert not answer.block.is_empty
        assert "INR 0" in answer.block.render()

    def test_the_escalation_names_why(self):
        answer = run("Refunds take five days.", StubExtractor("refunds take five days"))
        assert answer.escalation.reason is DeclineReason.UNSUPPORTED_CLAIM

    def test_a_gate_outage_declines_rather_than_passes(self):
        answer = run("Anything.", StubExtractor(raises=RuntimeError("provider down")))
        assert answer.declined
        assert answer.escalation.reason is DeclineReason.UNGROUNDED

    def test_no_evidence_at_all_declines_for_no_source(self):
        answer = run(
            "The retention period is seven years.",
            StubExtractor("the retention period is seven years"),
            messages=[{"role": "user", "content": "what is your retention period?"}],
            resolution=None,
        )
        assert answer.escalation.reason is DeclineReason.NO_CITABLE_SOURCE

    def test_the_escalation_carries_the_handles_the_run_minted(self):
        answer = run("Refunds take five days.", StubExtractor("refunds take five days"))
        assert set(answer.escalation.evidence_chain) >= {"snap_1", "res_1", "calc_1"}

    def test_the_escalation_lists_the_sources_that_were_read(self):
        answer = run("Refunds take five days.", StubExtractor("refunds take five days"))
        assert "northstar_logistics_enterprise_agreement::§2" in answer.escalation.sources_consulted


class TestRepairIsBoundedAndTargeted:
    def failed(self, *claims, invented=()):
        return GateOutcome(
            verdict=Verdict.FAILED,
            prose="x",
            failures=tuple(Failure(Claim(c), "no source supports this") for c in claims),
            invented_figures=tuple(invented),
        )

    def test_the_query_is_the_failing_claim_not_the_question(self):
        # A named gap makes a far better query than a rewritten question.
        queries = repair_queries(self.failed("refunds are processed within five working days"), ())
        assert queries == ["refunds are processed within five working days"]

    def test_a_query_already_attempted_is_not_repeated(self):
        gate = self.failed("refunds take five days")
        assert repair_queries(gate, ["Refunds take five days"]) == []

    def test_an_invented_figure_is_not_worth_re_retrieving(self):
        # A number the sources do not contain will not be found by searching
        # for it. Trying looks like diligence and spends half the budget.
        assert repair_queries(self.failed("the fee is INR 175", invented=(175.0,)), []) == []

    def test_every_failing_claim_becomes_a_query(self):
        assert len(repair_queries(self.failed("claim one", "claim two"), [])) == 2

    def test_a_rewrite_returning_nothing_new_stops_immediately(self):
        held = ["clause_a", "clause_b"]
        assert is_new_evidence(["clause_a"], held) is False
        assert is_new_evidence(["clause_a", "clause_c"], held) is True

    def test_an_empty_retrieval_is_not_new_evidence(self):
        assert is_new_evidence([], ["clause_a"]) is False


class TestTheAnswerShape:
    def test_it_is_immutable(self):
        answer = run("No fee.", StubExtractor("no fee applies"))
        assert isinstance(answer, Answer)
        with pytest.raises((AttributeError, TypeError)):
            answer.prose = "something else"


class TestEscalationIsEvidenceDrivenNotOnlyGateDriven:
    """A correct answer can still be somebody's job.

    Escalation used to fire only when the gate declined, which made "the prose
    is unsupported" and "this needs a person" the same condition. They are not:
    TKT-503's billing-contact question produces a *correct* answer - "no clause
    covers this" - and D27 still requires a drafted record. Tying the record to
    the gate meant the answer being right suppressed it.
    """

    MISSING_SOURCE: ClassVar[dict] = {
        "blocking": True,
        "conflicts": [
            {
                "conflict_class": "missing_source",
                "severity": "blocking",
                "detail": "no citable clause covers 'account_contact' for this account",
            }
        ],
    }

    def _messages(self, prose: str, conflicts: dict) -> list[dict]:
        import json as _json

        return [
            {"role": "user", "content": "how do I change the billing contact?"},
            {
                "role": "tool",
                "name": "check_data_consistency",
                "tool_call_id": "c1",
                "content": _json.dumps(conflicts),
            },
            {"role": "assistant", "content": prose},
        ]

    def _assemble(self, prose: str, conflicts: dict, claims=()):
        from src.agent.answer import assemble
        from src.auth.personas import get_persona, to_principal

        class Extractor:
            def extract(self, _prose):
                return list(claims)

        return assemble(
            prose,
            messages=self._messages(prose, conflicts),
            principal=to_principal(get_persona("maya_agent")),
            thread_id="t1",
            question="how do I change the billing contact?",
            extractor=Extractor(),
            subject="changing the billing contact",
        )

    def test_a_blocking_gap_drafts_a_record_even_when_the_gate_passes(self):
        answer = self._assemble("There is no citable clause covering this.", self.MISSING_SOURCE)
        assert answer.gate.verdict is Verdict.PASSED
        assert answer.escalation is not None
        assert answer.escalation.reason is DeclineReason.NO_CITABLE_SOURCE

    def test_the_answer_is_still_delivered(self):
        # The person gets the honest answer; a human gets the record. Dropping
        # the prose here would punish the model for being right.
        prose = "There is no citable clause covering this."
        answer = self._assemble(prose, self.MISSING_SOURCE)
        assert answer.prose == prose

    def test_a_conflict_from_an_earlier_turn_does_not_escalate_again(self):
        """A blocking conflict is a fact about the turn that found it.

        Conflicts stay in the transcript, so reading the whole conversation
        drafted a fresh escalation on every later turn - including "what else
        can you do", which had nothing to do with the order. Three turns in, a
        chat about one stale order had raised three escalations.
        """
        from src.agent.answer import assemble

        class Extractor:
            def extract(self, _prose):
                return []

        earlier = self._messages("Yes, no fee applies.", self.MISSING_SOURCE)
        # A later question, asked after the conflict was already on file.
        messages = [*earlier, {"role": "user", "content": "what else can you do?"}]
        answer = assemble(
            "I can look things up for you.",
            messages=messages,
            principal=to_principal(get_persona("maya_agent")),
            thread_id="t1",
            question="what else can you do?",
            extractor=Extractor(),
            subject="what else can you do",
        )
        assert answer.escalation is None

    def test_the_turn_that_finds_it_still_escalates(self):
        # The scoping must not switch the signal off entirely.
        assert self._assemble("No clause covers this.", self.MISSING_SOURCE).escalation is not None

    def test_an_advisory_conflict_does_not_draft_a_record(self):
        advisory = {
            "blocking": False,
            "conflicts": [
                {"conflict_class": "stale_status", "severity": "advisory", "detail": "x"}
            ],
        }
        assert self._assemble("All fine.", advisory).escalation is None

    def test_no_conflicts_means_no_record(self):
        assert self._assemble("All fine.", {"blocking": False, "conflicts": []}).escalation is None

    def test_an_unrecognised_blocking_class_still_escalates(self):
        # A new conflict class must not silently stop being somebody's job.
        unknown = {
            "blocking": True,
            "conflicts": [
                {"conflict_class": "something_new", "severity": "blocking", "detail": "x"}
            ],
        }
        answer = self._assemble("All fine.", unknown)
        assert answer.escalation is not None
        assert answer.escalation.reason is DeclineReason.UNRESOLVED_CONFLICT
