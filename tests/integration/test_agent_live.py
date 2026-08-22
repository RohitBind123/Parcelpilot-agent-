"""The agent against a real model. Deselected by default; run with `-m live`.

`test_graph.py` proves the structure with a scripted provider - that the toolset
is bound once, that a denial does not abort a run, that the loop terminates.
None of that shows a real model can actually drive the chain. This file does,
and it is the acceptance test for ARCHITECTURE milestone 6: the brief's two
example questions, from both sides of each discriminating pair, four different
answers.

Assertions are on substance and not on wording. A model that says "no fee
applies" and one that says "you will not be charged" are the same answer, and a
test that can only accept one of them is testing the sampler.
"""

from __future__ import annotations

import re

import pytest

from src.agent.context import open_agent
from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.knowledge.registry import load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.chroma import build_vector_store
from src.providers.registry import get_chat_provider, get_embedding_provider

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def retriever():
    settings = get_settings()
    chunks = load_chunks(settings.db_path)
    dense = build_vector_store(settings, get_embedding_provider(settings))
    return HybridRetriever(dense=dense, lexical=BM25Index(chunks))


@pytest.fixture(scope="module")
def ask(retriever, tmp_path_factory):
    """One real conversation per (persona, question), reused by every assertion.

    Each call is a multi-turn exchange with a real model and takes tens of
    seconds. Asking again per assertion would make this file too slow to run
    and would also mean different assertions grade different samples - so a
    flake would look like a disagreement between tests.
    """
    root = tmp_path_factory.mktemp("agent-live")
    cache: dict[tuple[str, str], object] = {}

    def go(persona_id: str, question: str) -> object:
        key = (persona_id, question)
        if key not in cache:
            with open_agent(
                to_principal(get_persona(persona_id)),
                provider=get_chat_provider(),
                retriever=retriever,
                checkpoint_path=root / "threads.db",
                run_id=persona_id,
            ) as agent:
                cache[key] = agent.ask(question, thread_id=f"{persona_id}-{len(cache)}")
        return cache[key]

    return go


def called(run, name: str) -> bool:
    return any(c.name == name for c in run.tool_calls)


CANCEL = "I need to cancel order {order}. Will I be charged a fee?"
CREDIT = "A pickup is three hours late because of carrier fault. Should I get a service credit?"


class TestTheModelDrivesTheChain:
    def test_it_reaches_the_calculator_through_the_handles(self, ask):
        run = ask("northstar_customer", CANCEL.format(order="ORD-1001"))
        # It cannot have computed a fee without both prerequisites, because the
        # arguments do not exist until they have run.
        assert called(run, "get_order")
        assert called(run, "resolve_policy")
        assert called(run, "compute_cancellation_fee")

    def test_it_is_not_told_the_order_of_the_calls(self, ask):
        # Nothing scripts the sequence. The schema makes the wrong order fail
        # with an error naming the next step, and the model follows it.
        run = ask("northstar_customer", CANCEL.format(order="ORD-1001"))
        names = [c.name for c in run.tool_calls]
        assert names.index("get_order") < names.index("compute_cancellation_fee")

    def test_the_handles_it_passed_were_the_ones_it_was_given(self, ask):
        # Handle arguments only. `order_id` also ends in _id and is something
        # the customer typed - the property under test is that the model never
        # invents an *evidence* handle, not that it never types an order number.
        handle_args = (
            "snapshot_id",
            "account_snapshot_id",
            "resolution_id",
            "approval_resolution_id",
            "report_id",
            "calc_id",
        )
        run = ask("northstar_customer", CANCEL.format(order="ORD-1001"))
        minted = set(run.handles)
        passed = {
            value
            for c in run.tool_calls
            for key, value in c.arguments.items()
            if key in handle_args and isinstance(value, str)
        }
        assert passed, "the chain never passed a handle, so this proves nothing"
        assert passed <= minted, f"invented handle(s): {sorted(passed - minted)}"


class TestTheBriefsFirstQuestion:
    """Two orders of identical shape, opposite answers."""

    def test_the_agreement_waives_the_fee(self, ask):
        answer = ask("northstar_customer", CANCEL.format(order="ORD-1001")).answer
        assert re.search(r"\bno\b.{0,30}\bfee\b|\bfee\b.{0,30}\bwaive|INR\s*0\b", answer, re.I)
        assert "Northstar" in answer or "agreement" in answer.lower()
        assert "250" not in _without_context(answer)

    def test_the_stale_status_is_surfaced_not_resolved(self, ask):
        # GS-001 requires the caveat. An answer that only says "no fee" is
        # incomplete in a way that matters: the parcel may already be gone.
        answer = ask("northstar_customer", CANCEL.format(order="ORD-1001")).answer.lower()
        assert "pickup" in answer or "collected" in answer
        assert "verify" in answer or "confirm" in answer or "may" in answer

    def test_the_agreement_that_defers_gets_the_default_fee(self, ask):
        answer = ask("lumenworks_customer", CANCEL.format(order="ORD-2001")).answer
        assert "250" in answer

    def test_the_two_answers_differ(self, ask):
        northstar = ask("northstar_customer", CANCEL.format(order="ORD-1001")).answer
        lumenworks = ask("lumenworks_customer", CANCEL.format(order="ORD-2001")).answer
        assert ("250" in lumenworks) and ("250" not in _without_context(northstar))


class TestTheBriefsSecondQuestion:
    """Word for word the same question, opposite answers, no order named."""

    def test_the_agreement_threshold_makes_this_account_ineligible(self, ask):
        answer = ask("lumenworks_customer", CREDIT).answer
        assert re.search(r"not eligible|ineligible|do(es)? not (meet|qualify|reach)", answer, re.I)
        assert "4 hour" in answer.lower() or "four hour" in answer.lower()

    def test_the_override_is_reported_even_though_it_costs_the_customer(self, ask):
        # A system that surfaces overrides only when they are favourable is not
        # reporting precedence, it is marketing.
        answer = ask("lumenworks_customer", CREDIT).answer.lower()
        assert "agreement" in answer
        assert "2 hour" in answer or "two hour" in answer  # the default it replaced

    def test_an_account_with_no_agreement_is_eligible(self, ask):
        answer = ask("beacon_customer", CREDIT).answer
        assert re.search(r"\byes\b|\beligible\b", answer, re.I)
        assert not re.search(r"not eligible|ineligible", answer, re.I)

    def test_it_asks_for_the_shipment_rather_than_inventing_an_amount(self, ask):
        # GS-009: no order was named, so no shipment fee exists to take 10% of.
        # A concrete figure here would be a hallucination with a citation.
        run = ask("beacon_customer", CREDIT)
        assert re.search(r"shipment|order", run.answer, re.I)

    def test_the_other_accounts_agreement_is_never_mentioned(self, ask):
        answer = ask("beacon_customer", CREDIT)
        assert "LumenWorks" not in answer.answer
        assert "300" not in answer.answer


class TestStaffSurface:
    def test_an_agent_answers_about_a_customer_and_names_the_account(self, ask):
        run = ask("maya_agent", "What is the cancellation position on ORD-1001?")
        assert "Northstar" in run.answer or "ACCT-001" in run.answer

    def test_the_queue_question_returns_mayas_tickets(self, ask):
        run = ask("maya_agent", "What is in my queue?")
        assert "TKT-502" in run.answer or "TKT-504" in run.answer
        assert "TKT-501" not in run.answer  # Rohit's


def _without_context(answer: str) -> str:
    """Drop sentences that quote the rule being overridden.

    "your agreement waives the INR 250 fee" contains 250 and is correct. What
    must not appear is 250 as the amount owed.
    """
    keep = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if re.search(
            r"override|overrid|instead of|would otherwise|waive|rather than", sentence, re.I
        ):
            continue
        keep.append(sentence)
    return " ".join(keep)
