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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from src.agent.tools.base import Tool, ToolError, ToolPending
from src.agent.tools.registry import MODEL_INVISIBLE, to_schemas
from src.domain.action_tokens import PendingAction
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

_DECLINED = (
    "the person declined this action, so nothing was done. Do not prepare it again "
    "unless they ask; tell them it was not carried out."
)

_NO_PERFORMER = (
    "the action could not be carried out because this session cannot perform actions; "
    "nothing was changed"
)

#: Keys whose value is an evidence handle, collected for the trace.
_HANDLE_KEYS = ("snapshot_id", "account_snapshot_id", "resolution_id", "report_id", "calc_id")


def _extend(left: list, right: list) -> list:
    return [*left, *right]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], _extend]
    turns: int
    stopped_early: bool
    #: The action awaiting confirmation, as primitives (ARCHITECTURE 13).
    #:
    #: Here rather than in the conversation, which is the integrity property
    #: and not a UX convention: the model cannot edit graph state, so the thing
    #: that executes is the thing the human was shown. The tool reply for the
    #: `prepare_action` call is deliberately withheld until the answer comes
    #: back, so the model reads one coherent outcome instead of "prepared"
    #: followed later by something contradicting it.
    pending: dict[str, Any] | None
    pending_token: str | None
    pending_call_id: str | None


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
    # Through `to_schemas`, not a comprehension over `tools`: that helper is
    # what applies `MODEL_INVISIBLE`, and building the list here independently
    # is how a withheld tool ends up in the schema anyway.
    schemas = to_schemas(tools)

    def model(state: AgentState) -> AgentState:
        completion = provider.complete(state["messages"], tools=schemas, tier="strong")
        # Rebuilt by the provider, never by hand: this is where Gemini's
        # per-tool-call signature is put back.
        return {"messages": [provider.to_assistant_message(completion)]}

    def run_tools(state: AgentState) -> AgentState:
        assistant = state["messages"][-1]
        replies: list[dict[str, Any]] = []
        parked: AgentState = {}
        for raw in assistant.get("tool_calls", ()):
            call = _to_call(raw)
            outcome = _invoke(by_name, call)
            if isinstance(outcome, ToolPending) and not parked:
                # No reply for this call yet. It is answered by the confirm
                # node once the human has, so the conversation never contains
                # "prepared" without the outcome that followed it.
                parked = {
                    "pending": outcome.pending.to_state(),
                    "pending_token": outcome.token,
                    "pending_call_id": call.id,
                }
                continue
            if isinstance(outcome, ToolPending):
                # A second proposal in one turn. Only one can be confirmed at
                # a time, and silently dropping it would leave the model
                # believing both are waiting.
                outcome = ToolError(
                    "only one action can await confirmation at a time; "
                    "propose this one again after the first is answered"
                )
            replies.append(_reply(call, outcome))
        return {"messages": replies, "turns": state.get("turns", 0) + 1, **parked}

    def confirm(state: AgentState) -> AgentState:
        """Pause for a human, then perform or cancel.

        Nothing happens before `interrupt()`, and that is load-bearing rather
        than tidy: LangGraph re-executes the interrupting node from its start
        when the run resumes, so any side effect above this line would happen
        twice. It is why this is a node of its own instead of a pause inside
        `run_tools`.
        """
        stored = state.get("pending") or {}
        answer = interrupt(
            {
                "preview": PendingAction.from_state(stored).to_preview(),
                "token": state.get("pending_token"),
            }
        )

        call = ToolCall(id=state.get("pending_call_id") or "", name="prepare_action", arguments={})
        cleared: AgentState = {"pending": None, "pending_token": None, "pending_call_id": None}

        confirmed, token = _read_answer(answer)
        if not confirmed:
            return {"messages": [_reply(call, ToolError(_DECLINED, recoverable=False))], **cleared}

        performer = by_name.get("execute_action")
        if performer is None:
            logger.error("a confirmation arrived but no execute_action tool is bound")
            return {
                "messages": [_reply(call, ToolError(_NO_PERFORMER, recoverable=False))],
                **cleared,
            }

        outcome = performer(token=token, pending=PendingAction.from_state(stored))
        return {"messages": [_reply(call, outcome)], **cleared}

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

    def after_tools(state: AgentState) -> str:
        return "confirm" if state.get("pending") else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", run_tools)
    graph.add_node("confirm", confirm)
    graph.add_node("exhausted", exhausted)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model", route, {"tools": "tools", "exhausted": "exhausted", END: END}
    )
    graph.add_conditional_edges("tools", after_tools, {"confirm": "confirm", "model": "model"})
    graph.add_edge("confirm", "model")
    graph.add_edge("exhausted", END)
    return graph.compile(checkpointer=checkpointer)


def _read_answer(answer: Any) -> tuple[bool, str]:
    """What the client sent back on resume.

    A bare `True` is accepted as confirmation for the CLI and for tests; the
    API always sends the mapping. Anything unrecognised is a decline, because
    the failure that matters here is performing something nobody agreed to.
    """
    if answer is True:
        return True, ""
    if isinstance(answer, Mapping):
        return bool(answer.get("confirm")), str(answer.get("token") or "")
    return False, ""


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


def _invoke(by_name: dict[str, Tool], call: ToolCall) -> Any:
    """Execute one call.

    A tool the model invented is answered rather than raised. It has already
    happened by the time we see it, and the only useful response is to say the
    name is not available and let the model choose again.

    A tool withheld from the schema is answered the same way. `execute_action`
    is bound but nameless to the model (`MODEL_INVISIBLE`), and a model that
    guesses the name should be told it is unavailable, not handed it.
    """
    if call.name in MODEL_INVISIBLE:
        return ToolError(f"no tool named {call.name!r} is available in this session")
    tool = by_name.get(call.name)
    if tool is None:
        return ToolError(
            f"no tool named {call.name!r} is available in this session; "
            f"available: {sorted(n for n in by_name if n not in MODEL_INVISIBLE)}"
        )
    return tool(**dict(call.arguments))


def _reply(call: ToolCall, outcome: Any) -> dict[str, Any]:
    """Shape one outcome as the tool message the next request carries."""
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
