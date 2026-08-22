"""The agent graph: the model plans, the graph enforces.

Driven by a scripted provider rather than a real one. That is not a compromise -
the properties under test are structural (does the toolset get rebound, does a
denial abort a run, does the loop terminate, do two threads stay apart) and a
sampled model would make them intermittent rather than testable. The live tests
cover whether a real model can actually drive it.

The one thing this file must not do is prove access control. That is the
projection's job, tested in `test_tool_projection.py`, and re-testing it here
against a scripted model would suggest the graph is where it lives.
"""

from __future__ import annotations

import json

import pytest

from src.agent.context import open_agent
from src.agent.graph import MAX_TOOL_TURNS, AgentRun, build_graph
from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import build_toolset
from src.auth.personas import get_persona, to_principal
from src.providers.base import Completion, ToolCall


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


class ScriptedProvider:
    """Returns the completions it was handed, in order.

    Records what it was asked, so a test can assert on the schema the model was
    shown as well as on what it did.
    """

    name = "scripted"

    def __init__(self, *completions: Completion):
        self.script = list(completions)
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=None, tier="strong", **kwargs):
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if not self.script:
            return say("(the script ran out)")
        return self.script.pop(0)

    def complete_structured(self, messages, *, schema, schema_name, tier="cheap"):
        raise NotImplementedError

    def to_assistant_message(self, completion):
        # Mirrors the real providers, arguments included. Serialising them as
        # "{}" here would make every tool see an empty call and quietly turn
        # these tests into a test of the missing-argument path.
        message = {"role": "assistant", "content": completion.text}
        if completion.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))},
                }
                for c in completion.tool_calls
            ]
        return message


MODEL = "scripted/test"


def say(text: str) -> Completion:
    return Completion(text=text, model=MODEL, tool_calls=())


def call(name: str, call_id: str = "c1", **arguments) -> Completion:
    return Completion(
        text="", model=MODEL, tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),)
    )


def calls(*specs: tuple[str, dict]) -> Completion:
    return Completion(
        text="",
        model=MODEL,
        tool_calls=tuple(
            ToolCall(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(specs)
        ),
    )


@pytest.fixture
def run(tmp_path):
    """One agent per persona, over the committed data and a scripted model."""
    opened = []

    def go(persona_id: str, provider, question: str, thread_id: str = "t1", **kwargs):
        cm = open_agent(
            persona(persona_id),
            provider=provider,
            checkpoint_path=tmp_path / "checkpoints.db",
            **kwargs,
        )
        agent = cm.__enter__()
        opened.append(cm)
        return agent.ask(question, thread_id=thread_id)

    yield go
    for cm in opened:
        cm.__exit__(None, None, None)


class TestTheToolsetIsBoundOnce:
    def test_the_schema_reaches_the_model(self, run):
        provider = ScriptedProvider(say("Hello."))
        run("northstar_customer", provider, "hello")
        offered = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert "get_order" in offered
        assert "my_queue" not in offered

    def test_it_is_the_same_schema_on_every_turn(self, run):
        # A toolset that could change mid-run would be a toolset the
        # conversation can influence.
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("Done."))
        run("northstar_customer", provider, "status of ORD-1001?")
        assert len(provider.calls) == 2
        assert provider.calls[0]["tools"] == provider.calls[1]["tools"]

    def test_a_staff_session_is_shown_more(self, run):
        provider = ScriptedProvider(say("Hello."))
        run("maya_agent", provider, "hello")
        offered = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert {"my_queue", "query_tickets", "sla_first_response_status"} <= offered


class TestTheLoop:
    def test_a_plain_answer_needs_one_turn(self, run):
        result = run("northstar_customer", ScriptedProvider(say("No fee applies.")), "hi")
        assert isinstance(result, AgentRun)
        assert result.answer == "No fee applies."
        assert result.tool_calls == ()

    def test_a_tool_call_round_trips(self, run):
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("It is BOOKED."))
        result = run("northstar_customer", provider, "status of ORD-1001?")
        assert result.answer == "It is BOOKED."
        assert [c.name for c in result.tool_calls] == ["get_order"]

    def test_the_tool_result_is_shown_to_the_model(self, run):
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("done"))
        run("northstar_customer", provider, "status?")
        second_turn = provider.calls[1]["messages"]
        tool_messages = [m for m in second_turn if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "BOOKED" in tool_messages[0]["content"]

    def test_several_calls_in_one_turn_all_run(self, run):
        provider = ScriptedProvider(
            calls(("get_order", {"order_id": "ORD-1001"}), ("get_account", {})),
            say("done"),
        )
        result = run("northstar_customer", provider, "everything")
        assert [c.name for c in result.tool_calls] == ["get_order", "get_account"]
        replies = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"]
        assert len(replies) == 2

    def test_every_call_gets_a_reply_keyed_by_its_id(self, run):
        # A tool_call with no matching tool message is a malformed conversation
        # and providers reject it.
        provider = ScriptedProvider(
            calls(("get_order", {"order_id": "ORD-1001"}), ("get_account", {})), say("done")
        )
        run("northstar_customer", provider, "everything")
        replies = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"]
        assert {m["tool_call_id"] for m in replies} == {"c0", "c1"}


class TestFailuresDoNotAbortTheRun:
    def test_an_unknown_tool_name_comes_back_as_an_error(self, run):
        provider = ScriptedProvider(call("scan_support_health"), say("I cannot do that."))
        result = run("northstar_customer", provider, "run the scan")
        reply = next(m for m in provider.calls[1]["messages"] if m.get("role") == "tool")
        assert "scan_support_health" in reply["content"]
        assert result.answer == "I cannot do that."

    def test_a_denial_is_a_result_not_a_crash(self, run):
        provider = ScriptedProvider(
            call("get_order", order_id="ORD-1001"), say("That order is not on your account.")
        )
        result = run("lumenworks_customer", provider, "status of ORD-1001?")
        reply = next(m for m in provider.calls[1]["messages"] if m.get("role") == "tool")
        assert "denied" in reply["content"]
        assert result.answer

    def test_a_denial_is_recorded_on_the_run_for_the_trace(self, run):
        # Denials are a demo asset (ARCHITECTURE 4.4) and drive a tool.denied
        # event in M8, so the run has to carry them rather than only the model.
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("no"))
        result = run("lumenworks_customer", provider, "status?")
        assert result.denials
        assert result.denials[0].reason == "out_of_scope"

    def test_bad_arguments_come_back_as_a_correctable_error(self, run):
        provider = ScriptedProvider(
            call("compute_cancellation_fee", order_id="ORD-1001"), say("Let me try again.")
        )
        run("northstar_customer", provider, "fee?")
        reply = next(m for m in provider.calls[1]["messages"] if m.get("role") == "tool")
        assert "order_id" in reply["content"]


class TestTermination:
    def test_a_model_that_only_calls_tools_still_stops(self, run):
        # Without a bound this is an unbounded spend, and the failure mode is a
        # bill rather than an exception.
        provider = ScriptedProvider(*[call("get_account", call_id=f"c{i}") for i in range(50)])
        result = run("northstar_customer", provider, "loop")
        assert len(provider.calls) <= MAX_TOOL_TURNS + 1
        assert result.stopped_early is True

    def test_stopping_early_still_returns_something_sayable(self, run):
        provider = ScriptedProvider(*[call("get_account", call_id=f"c{i}") for i in range(50)])
        result = run("northstar_customer", provider, "loop")
        assert result.answer

    def test_a_normal_run_is_not_marked_as_stopped(self, run):
        result = run("northstar_customer", ScriptedProvider(say("done")), "hi")
        assert result.stopped_early is False


class TestThreads:
    def test_a_thread_remembers_the_previous_turn(self, run, tmp_path):
        provider = ScriptedProvider(say("It is BOOKED."), say("I said BOOKED."))
        with open_agent(
            persona("northstar_customer"),
            provider=provider,
            checkpoint_path=tmp_path / "cp.db",
        ) as agent:
            agent.ask("status of ORD-1001?", thread_id="alpha")
            agent.ask("what did you say?", thread_id="alpha")
        history = provider.calls[1]["messages"]
        assert any(m.get("content") == "status of ORD-1001?" for m in history)

    def test_two_threads_do_not_see_each_other(self, run, tmp_path):
        provider = ScriptedProvider(say("first"), say("second"))
        with open_agent(
            persona("northstar_customer"),
            provider=provider,
            checkpoint_path=tmp_path / "cp.db",
        ) as agent:
            agent.ask("about ORD-1001", thread_id="alpha")
            agent.ask("about something else", thread_id="beta")
        second = provider.calls[1]["messages"]
        assert not any(m.get("content") == "about ORD-1001" for m in second)

    def test_a_thread_survives_the_process(self, tmp_path):
        path = tmp_path / "durable.db"
        first = ScriptedProvider(say("noted"))
        with open_agent(
            persona("northstar_customer"), provider=first, checkpoint_path=path
        ) as agent:
            agent.ask("remember ORD-1001", thread_id="gamma")

        second = ScriptedProvider(say("still here"))
        with open_agent(
            persona("northstar_customer"), provider=second, checkpoint_path=path
        ) as agent:
            agent.ask("still there?", thread_id="gamma")
        assert any(m.get("content") == "remember ORD-1001" for m in second.calls[0]["messages"])


class TestTheRunIsInspectable:
    def test_it_carries_the_evidence_handles_that_were_minted(self, run):
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("done"))
        result = run("northstar_customer", provider, "status?")
        assert result.handles
        assert result.handles[0].startswith("snap_")

    def test_it_carries_the_arguments_each_tool_was_called_with(self, run):
        provider = ScriptedProvider(call("get_order", order_id="ORD-1001"), say("done"))
        result = run("northstar_customer", provider, "status?")
        assert result.tool_calls[0].arguments == {"order_id": "ORD-1001"}

    def test_the_graph_can_be_built_without_being_run(self):
        # M8 compiles it once at startup; a build that needs a question would
        # make that impossible.
        with open_tool_context(persona("maya_agent")) as context:
            graph = build_graph(build_toolset(context), ScriptedProvider())
        assert graph is not None
