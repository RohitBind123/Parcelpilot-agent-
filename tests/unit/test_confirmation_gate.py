"""The graph half of the confirmation gate (ARCHITECTURE 13).

`test_action_tools.py` covers the tools; this covers the pause. The claim being
tested is the one the architecture calls an integrity property rather than a UX
convention: **the pending payload lives in graph state, not the model's context
window**. Two things follow, and both are asserted here rather than described -
the model cannot read the token, and the model cannot edit the payload between
the preview and the click, because it has no access to where the payload is.

Driven by a scripted provider. Whether a real model chooses to call
`prepare_action` is a question for the live tests; whether the graph pauses
when it does is structural, and a sampled model would make it intermittent.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import build_toolset
from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.runtime import open_runtime_store
from src.providers.base import Completion, ToolCall

SECRET = "gate-secret"
MODEL = "scripted/test"
THREAD = "thread-gate"

ESCALATION = {
    "kind": "create_escalation",
    "payload": {"question": "how do I change the billing contact?"},
    "evidence_ids": [],
}


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


def say(text: str) -> Completion:
    return Completion(text=text, model=MODEL, tool_calls=())


def call(name: str, call_id: str = "c1", **arguments) -> Completion:
    return Completion(
        text="", model=MODEL, tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),)
    )


class ScriptedProvider:
    name = "scripted"

    def __init__(self, *completions: Completion):
        self.script = list(completions)
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=None, tier="strong", **kwargs):
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        return self.script.pop(0) if self.script else say("(the script ran out)")

    def complete_structured(self, messages, *, schema, schema_name, tier="cheap"):
        raise NotImplementedError

    def to_assistant_message(self, completion):
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


@pytest.fixture
def runtime(tmp_path):
    with open_runtime_store(tmp_path / "runtime.db") as store:
        yield store


@pytest.fixture
def gate(tmp_path, runtime):
    """A compiled graph with the gate wired, plus the handles to drive it."""
    opened: list = []
    connections: list = []

    def build(provider, persona_id: str = "maya_agent"):
        manager = open_tool_context(
            persona(persona_id),
            db_path=get_settings().db_path,
            runtime=runtime,
            session_id="sid_test",
            thread_id=THREAD,
            action_secret=SECRET,
        )
        context = manager.__enter__()
        opened.append(manager)
        # Built from a connection we hold, not `from_conn_string`: that returns
        # a generator-based context manager which closes the database when it
        # is collected, and the graph outlives this function.
        connection = sqlite3.connect(tmp_path / "ckpt.db", check_same_thread=False)
        connections.append(connection)
        graph = build_graph(build_toolset(context), provider, checkpointer=SqliteSaver(connection))
        return graph, {"configurable": {"thread_id": THREAD}}

    yield build
    for manager in opened:
        manager.__exit__(None, None, None)
    for connection in connections:
        connection.close()


def propose_then(*rest: Completion) -> ScriptedProvider:
    return ScriptedProvider(call("prepare_action", **ESCALATION), *rest)


def start(graph, config, provider=None):
    return graph.invoke({"messages": [{"role": "user", "content": "escalate this"}]}, config)


class TestTheRunPauses:
    def test_a_proposal_interrupts_the_run(self, gate):
        graph, config = gate(propose_then())
        state = start(graph, config)
        assert state.get("__interrupt__")

    def test_the_interrupt_carries_the_preview_and_the_token(self, gate):
        graph, config = gate(propose_then())
        (interrupt,) = start(graph, config)["__interrupt__"]
        assert interrupt.value["preview"]["kind"] == "create_escalation"
        assert interrupt.value["token"]

    def test_the_graph_is_parked_on_the_confirm_node(self, gate):
        graph, config = gate(propose_then())
        start(graph, config)
        assert graph.get_state(config).next == ("confirm",)

    def test_nothing_is_executed_while_it_waits(self, gate, runtime):
        graph, config = gate(propose_then())
        start(graph, config)
        assert runtime.actions_for_thread(THREAD) == ()


class TestThePayloadIsOutOfReach:
    """The integrity property, asserted against what the model can actually see.

    Everything the model reads is in `messages`. If the token is in there, the
    model can confirm on the human's behalf; if the payload is in there in a
    form the model can restate, the preview stops being a preview.
    """

    def test_the_token_is_not_in_any_message(self, gate):
        graph, config = gate(propose_then())
        state = start(graph, config)
        token = state["__interrupt__"][0].value["token"]
        assert token
        assert token not in json.dumps(state["messages"])

    def test_the_nonce_is_not_in_any_message(self, gate):
        graph, config = gate(propose_then())
        state = start(graph, config)
        assert state["pending"]["nonce"] not in json.dumps(state["messages"])

    def test_the_pending_action_lives_in_state(self, gate):
        graph, config = gate(propose_then())
        state = start(graph, config)
        assert state["pending"]["kind"] == "create_escalation"

    def test_the_model_is_not_answered_until_the_human_is(self, gate):
        # The `prepare_action` call has no tool reply yet. The alternative -
        # replying "prepared" now and contradicting it later - is a
        # conversation in which the model has already been told two things.
        graph, config = gate(propose_then())
        state = start(graph, config)
        assert not [m for m in state["messages"] if m.get("role") == "tool"]


class TestConfirming:
    def test_confirming_executes_the_action(self, gate, runtime):
        graph, config = gate(propose_then(say("Escalation raised.")))
        token = start(graph, config)["__interrupt__"][0].value["token"]
        graph.invoke(Command(resume={"confirm": True, "token": token}), config)
        recorded = runtime.actions_for_thread(THREAD)
        assert len(recorded) == 1
        assert recorded[0].kind.value == "create_escalation"

    def test_the_model_learns_the_action_was_performed(self, gate):
        graph, config = gate(propose_then(say("Escalation raised.")))
        token = start(graph, config)["__interrupt__"][0].value["token"]
        state = graph.invoke(Command(resume={"confirm": True, "token": token}), config)
        (reply,) = [m for m in state["messages"] if m.get("role") == "tool"]
        assert json.loads(reply["content"])["executed"] is True

    def test_the_pending_slot_is_cleared_afterwards(self, gate):
        graph, config = gate(propose_then(say("done")))
        token = start(graph, config)["__interrupt__"][0].value["token"]
        state = graph.invoke(Command(resume={"confirm": True, "token": token}), config)
        assert state.get("pending") is None
        assert state.get("pending_token") is None

    def test_a_wrong_token_executes_nothing(self, gate, runtime):
        graph, config = gate(propose_then(say("done")))
        start(graph, config)
        state = graph.invoke(Command(resume={"confirm": True, "token": "forged.abc"}), config)
        assert runtime.actions_for_thread(THREAD) == ()
        (reply,) = [m for m in state["messages"] if m.get("role") == "tool"]
        assert json.loads(reply["content"])["error"] is True


class TestDeclining:
    def test_declining_executes_nothing(self, gate, runtime):
        graph, config = gate(propose_then(say("Understood, nothing done.")))
        start(graph, config)
        graph.invoke(Command(resume={"confirm": False}), config)
        assert runtime.actions_for_thread(THREAD) == ()

    def test_the_model_is_told_it_was_declined(self, gate):
        graph, config = gate(propose_then(say("Understood.")))
        start(graph, config)
        state = graph.invoke(Command(resume={"confirm": False}), config)
        (reply,) = [m for m in state["messages"] if m.get("role") == "tool"]
        assert "declined" in json.loads(reply["content"])["message"]

    def test_an_unrecognised_answer_declines(self, gate, runtime):
        # The failure that matters is performing something nobody agreed to,
        # so anything that is not a confirmation is a decline.
        graph, config = gate(propose_then(say("done")))
        start(graph, config)
        graph.invoke(Command(resume="maybe"), config)
        assert runtime.actions_for_thread(THREAD) == ()

    def test_the_run_continues_after_a_decline(self, gate):
        graph, config = gate(propose_then(say("Nothing was carried out.")))
        start(graph, config)
        state = graph.invoke(Command(resume={"confirm": False}), config)
        assert state["messages"][-1]["content"] == "Nothing was carried out."


class TestTheModelCannotReachExecuteAction:
    def test_calling_it_by_name_is_refused(self, gate, runtime):
        # It is bound to the toolset but withheld from the schema. A model that
        # guesses the name should be told it is unavailable, not handed it.
        graph, config = gate(ScriptedProvider(call("execute_action", token="anything"), say("no")))
        state = graph.invoke({"messages": [{"role": "user", "content": "just do it"}]}, config)
        (reply,) = [m for m in state["messages"] if m.get("role") == "tool"]
        payload = json.loads(reply["content"])
        assert payload["error"] is True
        assert "no tool named" in payload["message"]
        assert runtime.actions_for_thread(THREAD) == ()

    def test_it_is_absent_from_the_schema_the_model_was_shown(self, gate):
        provider = propose_then()
        graph, config = gate(provider)
        start(graph, config)
        offered = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert "prepare_action" in offered
        assert "execute_action" not in offered
