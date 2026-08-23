"""Driving one run and narrating it as events (ARCHITECTURE 16).

The graph is stepped with `stream(stream_mode="updates")` rather than invoked,
so each node's output can be turned into an event as it happens instead of
reconstructed afterwards. Tool calls, denials and the confirmation interrupt
all reach the client while the run is still going.

**Why `token.delta` chunks a finished answer rather than streaming the model.**
The grounding gate grades claims against the fact block, and on failure the
prose is dropped rather than shortened (D16, M7). Streaming raw model tokens
would put text in front of a person that the gate might then retract, which is
worse than waiting: a retracted sentence has already been read. So the model
call completes, the gate runs, and only a graded answer is emitted - split into
deltas so the client can render it progressively. The fact block goes first,
whole, because the numbers in it were computed in Python and were never in
doubt.

The run executes in a worker thread. `RunBus.emit` is therefore called from off
the event loop, which is why the bus hands events to subscriber queues through
`call_soon_threadsafe`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

from langgraph.types import Command

from src.agent.answer import assemble
from src.agent.graph import summarise
from src.api.events import RunBus
from src.datastore.runtime import RuntimeStore

logger = logging.getLogger(__name__)

#: Characters per `token.delta`. Small enough to look progressive, large enough
#: that a long answer is not thousands of rows in `run_events`.
DELTA_SIZE: Final = 48

_HANDLE_KEYS: Final = (
    "snapshot_id",
    "account_snapshot_id",
    "resolution_id",
    "report_id",
    "calc_id",
)


class RunExecutor:
    """One agent, one thread, narrated to the bus."""

    def __init__(
        self,
        *,
        agent: Any,
        bus: RunBus,
        store: RuntimeStore,
    ) -> None:
        self._agent = agent
        self._bus = bus
        self._store = store

    # -- entry points -------------------------------------------------------

    def start(self, *, run_id: str, thread_id: str, question: str) -> None:
        opening = {"messages": self._agent._opening(question, thread_id)}
        self._bus.emit(run_id, "run.started", {"run_id": run_id, "thread_id": thread_id})
        self._drive(run_id=run_id, thread_id=thread_id, question=question, payload=opening)

    def resume(self, *, run_id: str, thread_id: str, question: str, answer: Mapping[str, Any]):
        self._drive(
            run_id=run_id,
            thread_id=thread_id,
            question=question,
            payload=Command(resume=dict(answer)),
        )

    # -- the loop -----------------------------------------------------------

    def _drive(self, *, run_id: str, thread_id: str, question: str, payload: Any) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            for update in self._agent.graph.stream(payload, config=config, stream_mode="updates"):
                self._narrate(run_id, update)

            snapshot = self._agent.graph.get_state(config)
            if _is_paused(snapshot):
                # Parked on the confirmation node. The run is not finished and
                # must not be reported as such, or the client stops listening
                # and the person never sees the card.
                self._store.set_run_status(run_id, "awaiting_confirmation")
                return

            self._finish(run_id, thread_id, question, snapshot.values)
        except Exception as exc:
            logger.exception("run %s failed", run_id)
            self._store.set_run_status(run_id, "failed")
            self._bus.emit(run_id, "run.failed", {"error": f"{type(exc).__name__}: {exc}"})

    def _narrate(self, run_id: str, update: Mapping[str, Any]) -> None:
        for node, delta in update.items():
            if node == "__interrupt__":
                self._interrupt(run_id, delta)
                continue
            if not isinstance(delta, Mapping):
                continue
            if node == "model":
                self._model_step(run_id, delta)
            for message in delta.get("messages", ()) or ():
                if message.get("role") == "tool":
                    self._tool_event(run_id, message)

    def _model_step(self, run_id: str, delta: Mapping[str, Any]) -> None:
        assistant = (delta.get("messages") or [{}])[-1]
        planned = assistant.get("tool_calls") or ()
        self._bus.emit(run_id, "model.step", {"tool_calls": len(planned)})
        for raw in planned:
            function = raw.get("function", {}) if isinstance(raw, Mapping) else {}
            self._bus.emit(
                run_id,
                "tool.started",
                {
                    "call_id": raw.get("id", "") if isinstance(raw, Mapping) else "",
                    "name": function.get("name", ""),
                    # Public arguments only. A tool's arguments can name another
                    # account's record, and the trace panel is a thing people
                    # screenshot.
                    "args_public": _public_arguments(function.get("arguments")),
                },
            )

    def _tool_event(self, run_id: str, message: Mapping[str, Any]) -> None:
        try:
            body = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            body = {}
        common = {"call_id": message.get("tool_call_id", ""), "name": message.get("name", "")}

        if body.get("denied"):
            # A demo asset (ARCHITECTURE 4.4), and deliberately carries the
            # reason without the subject.
            self._bus.emit(run_id, "tool.denied", {**common, "reason": body.get("reason")})
            return
        if body.get("error"):
            self._bus.emit(run_id, "tool.error", {**common, "error": body.get("message")})
            return

        handle = next((body[k] for k in _HANDLE_KEYS if body.get(k)), None)
        self._bus.emit(
            run_id,
            "tool.finished",
            {**common, "evidence_id": handle, "summary": _summarise_result(body)},
        )
        if message.get("name") == "resolve_policy":
            self._bus.emit(
                run_id,
                "policy.resolved",
                {
                    "topic": body.get("topic"),
                    "governing": body.get("governing"),
                    "overridden": body.get("overridden", []),
                    "excluded": body.get("excluded", []),
                },
            )
        for conflict in body.get("conflicts", ()) or ():
            self._bus.emit(run_id, "conflict.detected", conflict)

    def _interrupt(self, run_id: str, delta: Any) -> None:
        for item in delta if isinstance(delta, Sequence) else [delta]:
            value = getattr(item, "value", item)
            if not isinstance(value, Mapping):
                continue
            preview = value.get("preview") or {}
            self._bus.emit(
                run_id,
                "interrupt.await_confirm",
                {
                    "preview": preview,
                    "token": value.get("token"),
                    "blocking_conflicts": [],
                    "advisories": preview.get("advisories", []),
                },
            )

    # -- the end ------------------------------------------------------------

    def _finish(
        self, run_id: str, thread_id: str, question: str, values: Mapping[str, Any]
    ) -> None:
        messages = list(values.get("messages", ()))
        run = summarise(messages, stopped_early=bool(values.get("stopped_early")))

        answer = None
        if self._agent.extractor is not None:
            answer = assemble(
                run.answer,
                messages=messages,
                resolution=self._agent._last_resolution(messages),
                principal=self._agent.principal,
                thread_id=thread_id,
                question=question,
                extractor=self._agent.extractor,
                subject=question.rstrip("?").strip(),
            )

        block = getattr(answer, "block", None)
        if block is not None:
            # Whole, and before any prose. The figures were computed in Python;
            # nothing about them is provisional.
            self._bus.emit(run_id, "facts.block", block.to_payload())

        prose = (answer.prose if answer else run.answer) or ""
        escalation = getattr(answer, "escalation", None)
        if escalation is not None:
            # `declined` is escalation-is-not-None, so this covers the case the
            # gate refused as well as the case nothing citable was found.
            self._bus.emit(
                run_id,
                "run.escalated",
                {
                    "reason": escalation.reason.value,
                    "escalation_preview": escalation.summary,
                },
            )
            prose = prose or escalation.summary

        for chunk in _chunks(prose):
            self._bus.emit(run_id, "token.delta", {"text": chunk})

        gate = getattr(answer, "gate", None)
        if gate is not None:
            self._bus.emit(
                run_id,
                "grounding.checked",
                {
                    "verdict": gate.verdict.value,
                    "claims_total": len(gate.claims),
                    # The claim text, not the Failure object: this is what a
                    # person reads in the trace when an answer was declined.
                    "unsupported": [failure.claim.text for failure in gate.failures],
                },
            )

        self._store.set_run_status(run_id, "completed")
        self._bus.emit(
            run_id,
            "run.completed",
            {
                "run_id": run_id,
                "stopped_early": run.stopped_early,
                "citations": sorted(_citations(messages)),
            },
        )


# -- helpers ----------------------------------------------------------------


def _is_paused(snapshot: Any) -> bool:
    return bool(getattr(snapshot, "next", ()))


def _chunks(text: str, size: int = DELTA_SIZE) -> Iterator[str]:
    for start in range(0, len(text), size):
        yield text[start : start + size]


def _public_arguments(raw: Any) -> dict[str, Any]:
    """Argument names, and scalar values that are safe to show.

    Identifiers the caller typed are echoed; anything structured is reduced to
    its shape. The trace is for showing what the agent did, not for
    reproducing the contents of a record in a screenshot.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        key: value if isinstance(value, str | int | float | bool) else f"<{type(value).__name__}>"
        for key, value in raw.items()
    }


def _summarise_result(body: Mapping[str, Any]) -> str:
    for key in ("verdict", "status", "governing", "severity"):
        if body.get(key):
            return f"{key}={body[key]}"
    return f"{len(body)} field(s)"


def _citations(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    found: set[str] = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            body = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        for key in ("governing", "target_clause", "basis_clause"):
            if isinstance(body.get(key), str):
                found.add(body[key])
        for key in ("overridden", "supporting", "clause_ids"):
            for value in body.get(key, ()) or ():
                if isinstance(value, str):
                    found.add(value)
    return found


__all__ = ["DELTA_SIZE", "RunExecutor"]
