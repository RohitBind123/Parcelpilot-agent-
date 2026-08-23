"""One agent, opened for one Principal.

Holds the four things a session needs to answer a question - the tool context,
the toolset built from it, the compiled graph, and a durable checkpointer - and
makes the order they are created in unavailable to callers. That order is the
whole access-control guarantee: tools are bound before the graph is compiled,
and the graph is compiled before a message exists.

`thread_id` is the conversation. The checkpointer is a SQLite file, so a thread
survives a restart and the Streamlit client in M9 can list and reopen threads
without holding anything in memory.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.answer import assemble
from src.agent.graph import MAX_TOOL_TURNS, AgentRun, build_graph, summarise
from src.agent.prompts import system_prompt
from src.agent.tools.base import Tool
from src.agent.tools.context import ToolContext, open_tool_context
from src.auth.principal import Principal
from src.domain.evidence import EvidenceError, EvidenceKind
from src.providers.base import ChatProvider

#: Beside the committed database, not inside it: `parcelpilot.db` is rebuilt
#: from source by `scripts/build_db.py` and opened read-only.
DEFAULT_CHECKPOINT_PATH = Path("data/threads.db")


@dataclass(frozen=True, slots=True)
class Agent:
    principal: Principal
    tools: Sequence[Tool]
    graph: Any
    tool_context: ToolContext
    #: The cheap-model half of the grounding gate. Absent means the gate does
    #: not run - and `AgentRun.answer` is None rather than a pass, so nothing
    #: downstream can mistake "not checked" for "checked and fine".
    extractor: Any | None = None

    def ask(self, question: str, *, thread_id: str = "default") -> AgentRun:
        """One turn: the model plans, the tools run, and the gate grades."""
        state = self.graph.invoke(
            {"messages": self._opening(question, thread_id)},
            config={"configurable": {"thread_id": thread_id}},
        )
        run = summarise(state["messages"], stopped_early=bool(state.get("stopped_early")))
        if self.extractor is None:
            return run

        answer = assemble(
            run.answer,
            messages=state["messages"],
            resolution=self._last_resolution(state["messages"]),
            principal=self.principal,
            thread_id=thread_id,
            question=question,
            extractor=self.extractor,
            subject=question.rstrip("?").strip(),
        )
        # The gate's verdict replaces the prose, so a declined answer says what
        # it could not establish instead of saying the thing it could not
        # support. The facts stand either way; they were computed in Python.
        return replace(
            run,
            answer=answer.prose or answer.escalation.summary,
            grounding=answer,
        )

    def _last_resolution(self, messages: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        """The full resolver payload behind the last resolution the run minted.

        The tool result carries clause ids without their text, which is enough
        to cite and not enough to render the delta an override turns on. The
        store has the whole thing, under a handle the run already holds.
        """
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("name") != "resolve_policy":
                continue
            try:
                handle = json.loads(message["content"]).get("resolution_id")
            except (json.JSONDecodeError, TypeError):
                continue
            if not handle:
                continue
            try:
                return self.tool_context.store.read(handle, expect=EvidenceKind.POLICY_RESOLUTION)
            except EvidenceError:
                return None
        return None

    def history(self, thread_id: str = "default") -> list[dict[str, Any]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(snapshot.values.get("messages", [])) if snapshot.values else []

    def _opening(self, question: str, thread_id: str) -> list[dict[str, Any]]:
        """The messages this turn adds.

        The system prompt goes in once per thread, and whether it has been
        written is read from the checkpointer rather than remembered in the
        process. A module-level "threads I have seen" set looks equivalent and
        is not: after a restart it is empty, so a resumed conversation gets a
        second system prompt inserted halfway through - and under the M8 server
        two users on the same thread label would share it.
        """
        user = {"role": "user", "content": question}
        if self.history(thread_id):
            return [user]
        return [{"role": "system", "content": system_prompt(self.principal)}, user]


@contextmanager
def open_agent(
    principal: Principal,
    *,
    provider: ChatProvider,
    db_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    retriever: Any | None = None,
    severity_classifier: Any | None = None,
    extractor: Any | None = None,
    run_id: str = "run",
    max_tool_turns: int = MAX_TOOL_TURNS,
    runtime: Any | None = None,
    session_id: str = "session",
    thread_id: str = "default",
    action_secret: str | None = None,
) -> Iterator[Agent]:
    from src.agent.tools.registry import build_toolset

    path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)

    with open_tool_context(
        principal,
        run_id=run_id,
        db_path=db_path,
        retriever=retriever,
        severity_classifier=severity_classifier,
        runtime=runtime,
        session_id=session_id,
        thread_id=thread_id,
        action_secret=action_secret,
    ) as tool_context:
        tools = build_toolset(tool_context)
        graph = build_graph(tools, provider, checkpointer=saver, max_tool_turns=max_tool_turns)
        agent = Agent(
            principal=principal,
            tools=tools,
            graph=graph,
            tool_context=tool_context,
            extractor=extractor,
        )
        try:
            yield agent
        finally:
            connection.close()
