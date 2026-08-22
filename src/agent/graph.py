"""The agent graph: the model plans, the graph enforces.

One graph for both audiences (D26). What differs between a customer session and
a staff session is the toolset bound into it before the first call, not a branch
inside it and certainly not a sentence in a prompt.

**Why this is a `StateGraph` and not `create_react_agent`.** The prebuilt takes
a LangChain chat model, and `langchain-openai` against Gemini fails on the
second turn of any tool-using conversation:

    400 Function call is missing a thought_signature in functionCall parts.

Gemini 3.x returns an opaque per-tool-call signature and rejects the next
request unless it is echoed back. Our provider captures and replays it
(`to_assistant_message`); `ChatOpenAI` has nowhere to keep it. Gemini carries
dev, tests and the demo (D9a), so the prebuilt is unusable. The deviation is
narrow - this file is the model node the prebuilt would have supplied - and the
checkpointer, `thread_id` and `interrupt()` that M8 needs all still come from
LangGraph. `tests/integration/test_langchain_constraint_live.py` fails if the
constraint ever goes away.

Messages are kept in the provider's own wire shape rather than converted to
LangChain objects. One format end to end means nothing has to reconstruct an
assistant turn by hand, which is exactly how the signature gets dropped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.tools.base import Tool, ToolError
from src.providers.base import ChatProvider, ToolCall

logger = logging.getLogger(__name__)

#: How many times the model may call tools before the graph stops asking. Six is
#: comfortably above the longest chain the pack needs (order, account, resolve,
#: resolve, compute, check) and low enough that a loop costs pennies rather than
#: a bill. Exceeding it is reported, never hidden.
MAX_TOOL_TURNS = 8

_STOPPED_EARLY = (
    "I could not finish working this out within the steps available. "
    "Here is what I established before stopping."
)

#: Keys whose value is an evidence handle, collected for the trace.
_HANDLE_KEYS = ("snapshot_id", "account_snapshot_id", "resolution_id", "report_id", "calc_id")


def _extend(left: list, right: list) -> list:
    return [*left, *right]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], _extend]
    turns: int
    stopped_early: bool


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One call, as the trace will render it.

    Read back out of the messages rather than recorded as it happens, because
    the messages are what the checkpointer persists - a run resumed in another
    process has the conversation and nothing else, and the trace must not be
    poorer for it.
    """

    name: str
    arguments: dict[str, Any]
    outcome: str  # "result" | "denied" | "error"
    handle: str | None = None
    #: Present on a denial. A string rather than the enum, because this came
    #: back through JSON and pretending otherwise would be a lie about where
    #: the value has been.
    reason: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class AgentRun:
    """What a turn did, for the trace panel and the CLI."""

    answer: str
    tool_calls: tuple[ToolInvocation, ...] = ()
    handles: tuple[str, ...] = ()
    stopped_early: bool = False
    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    #: The gate's verdict and the fact block, when a gate ran. None means it
    #: did not - deliberately not a passing verdict, so nothing downstream can
    #: mistake "not checked" for "checked and fine".
    grounding: Any | None = None

    @property
    def denials(self) -> tuple[ToolInvocation, ...]:
        """Every refusal in the run. A demo asset (ARCHITECTURE 4.4), and the
        source of the `tool.denied` events M8 emits."""
        return tuple(c for c in self.tool_calls if c.outcome == "denied")


def build_graph(
    tools: Sequence[Tool],
    provider: ChatProvider,
    *,
    checkpointer: Any | None = None,
    max_tool_turns: int = MAX_TOOL_TURNS,
) -> Any:
    """Compile the graph for one Principal.

    The toolset is closed over here, before any message is read. Nothing in the
    conversation can reach it, which is what makes the projection a containment
    mechanism rather than a convention.
    """
    by_name = {tool.name: tool for tool in tools}
    schemas = [tool.to_schema() for tool in tools]

    def model(state: AgentState) -> AgentState:
        completion = provider.complete(state["messages"], tools=schemas, tier="strong")
        # Rebuilt by the provider, never by hand: this is where Gemini's
        # per-tool-call signature is put back.
        return {"messages": [provider.to_assistant_message(completion)]}

    def run_tools(state: AgentState) -> AgentState:
        assistant = state["messages"][-1]
        replies = [_run_one(by_name, _to_call(raw)) for raw in assistant.get("tool_calls", ())]
        return {"messages": replies, "turns": state.get("turns", 0) + 1}

    def route(state: AgentState) -> str:
        assistant = state["messages"][-1]
        if not assistant.get("tool_calls"):
            return END
        if state.get("turns", 0) >= max_tool_turns:
            # Stopping is reported rather than silently truncating: a run that
            # ran out of steps and one that finished must not look alike.
            logger.warning("tool turn budget exhausted after %s turns", state.get("turns"))
            return "exhausted"
        return "tools"

    def exhausted(_state: AgentState) -> AgentState:
        return {"stopped_early": True}

    graph = StateGraph(AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", run_tools)
    graph.add_node("exhausted", exhausted)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model", route, {"tools": "tools", "exhausted": "exhausted", END: END}
    )
    graph.add_edge("tools", "model")
    graph.add_edge("exhausted", END)
    return graph.compile(checkpointer=checkpointer)


def _to_call(raw: Any) -> ToolCall:
    """A tool call from graph state, which may have been through JSON."""
    if isinstance(raw, ToolCall):
        return raw
    function = raw.get("function", {})
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(id=raw.get("id", ""), name=function.get("name", ""), arguments=arguments)


def _run_one(by_name: dict[str, Tool], call: ToolCall) -> dict[str, Any]:
    """Execute one call and shape its reply for the next request.

    A tool the model invented is answered rather than raised. It has already
    happened by the time we see it, and the only useful response is to say the
    name is not available and let the model choose again.
    """
    tool = by_name.get(call.name)
    if tool is None:
        outcome: Any = ToolError(
            f"no tool named {call.name!r} is available in this session; "
            f"available: {sorted(by_name)}"
        )
    else:
        outcome = tool(**dict(call.arguments))

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(outcome.to_payload()),
    }


def summarise(messages: Sequence[dict[str, Any]], *, stopped_early: bool) -> AgentRun:
    """Turn a finished conversation into something a trace can render."""
    invocations: list[ToolInvocation] = []
    handles: list[str] = []

    pending: dict[str, ToolCall] = {}
    for message in messages:
        for raw in message.get("tool_calls", ()) or ():
            call = _to_call(raw)
            pending[call.id] = call
        if message.get("role") != "tool":
            continue
        call = pending.get(message.get("tool_call_id", ""))
        payload = json.loads(message["content"])
        if payload.get("denied"):
            outcome = "denied"
        elif payload.get("error"):
            outcome = "error"
        else:
            outcome = "result"
        handle = next((payload[k] for k in _HANDLE_KEYS if payload.get(k)), None)
        if handle:
            handles.append(handle)
        invocations.append(
            ToolInvocation(
                name=message.get("name") or (call.name if call else ""),
                arguments=dict(call.arguments) if call else {},
                outcome=outcome,
                handle=handle,
                reason=payload.get("reason"),
                message=str(payload.get("message", "")),
            )
        )

    answer = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            answer = str(message["content"])
            break
    if stopped_early and not answer:
        answer = _STOPPED_EARLY

    return AgentRun(
        answer=answer,
        tool_calls=tuple(invocations),
        handles=tuple(handles),
        stopped_early=stopped_early,
        messages=tuple(messages),
    )
