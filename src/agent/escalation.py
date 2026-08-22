"""Escalation as a record, not a sentence (D27).

When the system declines to answer, "a human will follow up" is not an outcome -
nothing was created, nobody was told, and there is no artifact anyone can act
on. So a decline drafts a record naming the specific gap, carrying the evidence
chain that was assembled before the gap was found.

The wording of `what_is_unresolved` is the part that matters. "No source found"
is a shrug. "No clause in the corpus documents how to change a billing contact"
is a sentence someone can act on, and it is also a claim about the corpus that
can be checked - TKT-503 is in the pack precisely because every SaaS product has
a settings page and a model will happily invent the path to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.auth.principal import Principal


class DeclineReason(StrEnum):
    NO_CITABLE_SOURCE = "no_citable_source"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    UNDETERMINED_SEVERITY = "undetermined_severity"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    #: The gate could not run. Distinct from a failed check, because nobody has
    #: established that anything is wrong - only that nothing was verified.
    UNGROUNDED = "ungrounded"


_GAP_WORDING = {
    DeclineReason.NO_CITABLE_SOURCE: "no clause in the corpus covers {subject}",
    DeclineReason.UNRESOLVED_CONFLICT: (
        "two sources of equal authority disagree about {subject} and neither takes precedence"
    ),
    DeclineReason.UNDETERMINED_SEVERITY: (
        "the severity of {subject} could not be determined from the policy definitions"
    ),
    DeclineReason.UNSUPPORTED_CLAIM: (
        "a drafted answer about {subject} made claims the sources do not support"
    ),
    DeclineReason.UNGROUNDED: (
        "the answer about {subject} could not be checked against its sources"
    ),
}


@dataclass(frozen=True, slots=True)
class Escalation:
    """A drafted record. Nothing is created until the confirmation gate (M8)."""

    account_id: str | None
    thread_id: str
    question: str
    reason: DeclineReason
    what_is_unresolved: str
    severity: str | None = None
    severity_note: str = ""
    evidence_chain: tuple[str, ...] = field(default_factory=tuple)
    sources_consulted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> str:
        """What the user is told while the record awaits confirmation."""
        return (
            f"I do not have a source that settles this: {self.what_is_unresolved}. "
            "I have drafted an escalation for a human to pick up, including the question "
            "as you asked it and everything I checked."
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "create_escalation",
            "account_id": self.account_id,
            "thread_id": self.thread_id,
            "question": self.question,
            "reason": self.reason.value,
            "what_is_unresolved": self.what_is_unresolved,
            "severity": self.severity or "undetermined",
            "severity_note": self.severity_note,
            "evidence_chain": list(self.evidence_chain),
            "sources_consulted": list(self.sources_consulted),
        }


def draft(
    *,
    principal: Principal,
    thread_id: str,
    question: str,
    reason: DeclineReason,
    subject: str,
    evidence_chain: Sequence[str] = (),
    sources_consulted: Sequence[str] = (),
    severity: str | None = None,
    severity_note: str = "",
) -> Escalation:
    """Draft the record for a decline.

    `subject` is what could not be established, in the words of the question -
    "how to change a billing contact", not "account_contact". The record is read
    by a person, and an internal topic key tells them nothing.
    """
    return Escalation(
        account_id=principal.account_id,
        thread_id=thread_id,
        question=question,
        reason=reason,
        what_is_unresolved=_GAP_WORDING[reason].format(subject=subject),
        severity=severity,
        severity_note=severity_note or ("" if severity else "no severity could be derived"),
        evidence_chain=tuple(evidence_chain),
        sources_consulted=tuple(sources_consulted),
    )
