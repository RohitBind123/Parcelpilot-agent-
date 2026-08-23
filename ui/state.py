"""The event reducer: SSE events in, one view model out.

Separated from the Streamlit app because this is where the display logic
actually is, and Streamlit is a poor place to test anything. Everything the
client renders comes from folding the run's events into a `RunView`, so the
questions worth asking - does the fact block arrive before the prose, does a
denial show up, does the card appear only while a confirmation is pending -
are answered here by a function with no browser and no server.

Purely additive and immutable. `apply` returns a new view, and a run replayed
from `?from_seq=0` folds to exactly the same value as one watched live; that is
what makes reattaching after a refresh indistinguishable from never having left.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final

#: Run states the client distinguishes. `awaiting` is not a failure and not a
#: completion: the graph is parked and the person is the next actor.
RUNNING: Final = "running"
AWAITING: Final = "awaiting_confirmation"
COMPLETED: Final = "completed"
FAILED: Final = "failed"


@dataclass(frozen=True, slots=True)
class ToolStep:
    """One tool call as the trace panel shows it."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    outcome: str = "running"  # running | ok | denied | error
    evidence_id: str | None = None
    summary: str = ""
    detail: str = ""

    @property
    def is_finished(self) -> bool:
        return self.outcome != "running"


@dataclass(frozen=True, slots=True)
class RunView:
    """Everything one run has said so far."""

    run_id: str = ""
    status: str = RUNNING
    prose: str = ""
    facts: Mapping[str, Any] | None = None
    steps: tuple[ToolStep, ...] = ()
    conflicts: tuple[Mapping[str, Any], ...] = ()
    resolutions: tuple[Mapping[str, Any], ...] = ()
    citations: tuple[str, ...] = ()
    grounding: Mapping[str, Any] | None = None
    escalation: Mapping[str, Any] | None = None
    #: The confirmation card, while one is pending. Cleared once answered.
    pending: Mapping[str, Any] | None = None
    confirm_token: str | None = None
    error: str | None = None
    #: The highest sequence number folded in. What a reattach asks to continue
    #: from, so it is the server's number rather than a count of what arrived.
    last_seq: int = 0

    @property
    def denials(self) -> tuple[ToolStep, ...]:
        """Refusals. A demo asset (ARCHITECTURE 4.4), shown rather than hidden."""
        return tuple(step for step in self.steps if step.outcome == "denied")

    @property
    def has_conflict(self) -> bool:
        """Whether the badge should be loud.

        An override counts as well as a conflict: a governing clause that
        displaced another is the single most legible demonstration of
        precedence, and it is invisible if only conflicts light the badge up.
        """
        return bool(self.conflicts) or any(r.get("overridden") for r in self.resolutions)

    @property
    def is_finished(self) -> bool:
        return self.status in {COMPLETED, FAILED}

    @property
    def awaiting_confirmation(self) -> bool:
        return self.pending is not None


def apply(view: RunView, seq: int, event: str, data: Mapping[str, Any]) -> RunView:
    """Fold one event into the view. Never mutates; always returns a new one."""
    handler = _HANDLERS.get(event)
    updated = handler(view, data) if handler else {}
    # `max` rather than assignment: a replay that overlaps a live stream can
    # deliver an older sequence, and the reattach point must never go
    # backwards or the client would ask for events it already has.
    return replace(view, **updated, last_seq=max(view.last_seq, seq))


def fold(events: Iterable[tuple[int, str, Mapping[str, Any]]], view: RunView | None = None):
    """Apply a whole stream. Used for replay and by the tests."""
    result = view or RunView()
    for seq, event, data in events:
        result = apply(result, seq, event, data)
    return result


# -- per-event handlers -----------------------------------------------------


def _started(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": data.get("run_id", view.run_id), "status": RUNNING}


def _tool_started(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    step = ToolStep(
        call_id=data.get("call_id", ""),
        name=data.get("name", ""),
        arguments=dict(data.get("args_public") or {}),
    )
    return {"steps": (*view.steps, step)}


def _tool_finished(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "steps": _close(
            view,
            data,
            outcome="ok",
            evidence_id=data.get("evidence_id"),
            summary=str(data.get("summary") or ""),
        )
    }


def _tool_denied(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": _close(view, data, outcome="denied", detail=str(data.get("reason") or ""))}


def _tool_error(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": _close(view, data, outcome="error", detail=str(data.get("error") or ""))}


def _close(view: RunView, data: Mapping[str, Any], **changes: Any) -> tuple[ToolStep, ...]:
    """Complete the matching open step, or record one that was never started.

    The fallback is not defensive padding. A replay that begins part-way
    through can deliver a `tool.finished` whose `tool.started` is behind
    `from_seq`, and dropping it would make the trace quietly shorter than the
    run it describes.
    """
    call_id = data.get("call_id", "")
    for index, step in enumerate(view.steps):
        if step.call_id == call_id and not step.is_finished:
            return (*view.steps[:index], replace(step, **changes), *view.steps[index + 1 :])
    return (*view.steps, replace(ToolStep(call_id=call_id, name=data.get("name", "")), **changes))


def _facts(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"facts": dict(data)}


def _delta(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"prose": view.prose + str(data.get("text", ""))}


def _resolved(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"resolutions": (*view.resolutions, dict(data))}


def _conflict(view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"conflicts": (*view.conflicts, dict(data))}


def _grounding(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"grounding": dict(data)}


def _escalated(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {"escalation": dict(data)}


def _await_confirm(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pending": dict(data.get("preview") or {}),
        "confirm_token": data.get("token"),
        "status": AWAITING,
    }


def _completed(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": COMPLETED,
        "citations": tuple(data.get("citations") or ()),
        # A run cannot finish with a card still up. Leaving one would offer a
        # Confirm button for a graph that is no longer parked.
        "pending": None,
        "confirm_token": None,
    }


def _failed(_view: RunView, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": FAILED,
        "error": str(data.get("error") or "the run failed"),
        "pending": None,
        "confirm_token": None,
    }


_HANDLERS: Final[dict[str, Any]] = {
    "run.started": _started,
    "tool.started": _tool_started,
    "tool.finished": _tool_finished,
    "tool.denied": _tool_denied,
    "tool.error": _tool_error,
    "facts.block": _facts,
    "token.delta": _delta,
    "policy.resolved": _resolved,
    "conflict.detected": _conflict,
    "grounding.checked": _grounding,
    "run.escalated": _escalated,
    "interrupt.await_confirm": _await_confirm,
    "run.completed": _completed,
    "run.failed": _failed,
}


__all__ = [
    "AWAITING",
    "COMPLETED",
    "FAILED",
    "RUNNING",
    "RunView",
    "ToolStep",
    "apply",
    "fold",
]
