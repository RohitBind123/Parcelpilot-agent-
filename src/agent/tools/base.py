"""Tool primitives: the schema a model sees, and the shape a failure takes.

Framework-agnostic on purpose. The provider layer speaks OpenAI-compatible
function calling and D9a requires the agent not be welded to one provider, so a
tool here is a plain object that can render itself as a function schema. The
LangChain adapter belongs next to the graph in M6, not underneath the tools.

Two decisions are load-bearing.

**A failure is a value, not an exception.** The model is the caller. It cannot
catch anything, and a traceback in its context window is noise it will either
ignore or invent around. A returned `ToolError` naming the missing prerequisite
is something it can read and act on - and that is what makes the multi-step
chain a mechanic rather than a hope: call a calculator without a resolution and
the error tells you to resolve first.

**A denial says nothing about what was denied.** Refusal text goes into model
context, which is what a summariser reads and a prompt injection reaches. A
denial that quotes the order it is refusing has not refused it. It must also be
indistinguishable from "no such record", or the tool is an existence oracle for
other accounts' identifiers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

logger = logging.getLogger(__name__)

JsonType = str  # "string" | "integer" | "number" | "boolean" | "array"


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    type: JsonType
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    items: JsonType | None = None
    #: The tool call that mints this value, when it is an evidence handle
    #: rather than something the user typed. Rendered into the error when the
    #: argument is missing, which is what turns a validation failure into an
    #: instruction the model can follow.
    produced_by: str | None = None

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.type == "array":
            schema["items"] = {"type": self.items or "string"}
        return schema


class DenialReason(StrEnum):
    #: The record exists somewhere, or does not exist at all. Same message.
    OUT_OF_SCOPE = "out_of_scope"
    #: The role does not carry the scope this tool needs.
    INSUFFICIENT_SCOPE = "insufficient_scope"


@dataclass(frozen=True, slots=True)
class ToolResult:
    data: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True, slots=True)
class ToolError:
    """A failure the model is expected to read and correct."""

    message: str
    #: Whether trying something different could succeed. False for an internal
    #: fault, so the model escalates instead of looping.
    recoverable: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {"error": True, "message": self.message, "recoverable": self.recoverable}


@dataclass(frozen=True, slots=True)
class ToolDenied:
    """A refusal. Carries a reason and, deliberately, nothing else."""

    reason: DenialReason
    #: What kind of thing was asked for, in words a customer would use.
    subject_kind: str
    #: The identifier the caller typed. Echoing it back leaks nothing - it came
    #: from them - and without it the refusal is unreadable.
    identifier: str | None = None

    @property
    def message(self) -> str:
        if self.reason is DenialReason.INSUFFICIENT_SCOPE:
            return f"your role does not have access to {self.subject_kind}"
        named = f" {self.identifier}" if self.identifier else ""
        # One sentence for both "not yours" and "no such thing". If they
        # differed, the difference would answer a question nobody may ask.
        return f"no {self.subject_kind}{named} is available on your account"

    def to_payload(self) -> dict[str, Any]:
        return {
            "denied": True,
            "reason": self.reason.value,
            "message": self.message,
            "subject_kind": self.subject_kind,
        }


Outcome = ToolResult | ToolError | ToolDenied

#: What a tool body returns. Denials and prerequisite errors are values it
#: constructs; anything raised is a fault, not an answer.
Runner = Callable[..., Outcome]


def _missing_message(tool_name: str, missing: Sequence[Param]) -> str:
    """Name what is absent, and where it comes from when that is knowable.

    A handle the model has not obtained yet is a different problem from an
    identifier the user has not supplied. Telling it to "call get_order first"
    to obtain an order id would send it round a loop; telling it to call
    resolve_policy to obtain a resolution_id is the instruction that makes the
    chain work.
    """
    names = [p.name for p in missing]
    head = f"missing required argument(s) {names} for {tool_name}"
    steps = [f"{p.name} comes from {p.produced_by}" for p in missing if p.produced_by]
    return f"{head}. " + "; ".join(steps) + "." if steps else head


_INTERNAL_FAULT: Final = (
    "the tool failed for an internal reason and the failure has been logged; do not retry this call"
)


@dataclass(frozen=True, slots=True)
class Tool:
    """One callable, with the schema that describes it to a model."""

    name: str
    description: str
    params: Sequence[Param]
    run: Runner
    #: Scope the Principal must hold. Advisory here - the real enforcement is
    #: that an unauthorised tool is never built (see `registry.py`).
    requires_scope: str | None = field(default=None, compare=False)

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.to_schema() for p in self.params},
                    "required": [p.name for p in self.params if p.required],
                    "additionalProperties": False,
                },
            },
        }

    def __call__(self, **arguments: Any) -> Outcome:
        known = {p.name for p in self.params}
        unknown = sorted(set(arguments) - known)
        if unknown:
            # Dropping one runs a different query from the one that was asked
            # for, which on an ACL-adjacent read is the wrong way to fail.
            return ToolError(
                f"unknown argument(s) {unknown} for {self.name}; "
                f"expected a subset of {sorted(known)}"
            )

        missing = [p for p in self.params if p.required and p.name not in arguments]
        if missing:
            return ToolError(_missing_message(self.name, missing))

        try:
            return self.run(**arguments)
        except Exception:
            # The message reaches model context, so it carries no detail from
            # the exception - connection strings and row contents both live
            # there. The log keeps the traceback.
            logger.exception("tool %s failed", self.name)
            return ToolError(_INTERNAL_FAULT, recoverable=False)
