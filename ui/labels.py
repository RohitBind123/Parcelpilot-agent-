"""What the agent is doing, in words a customer can read.

Tool names are internal identifiers. `check_data_consistency` and
`sla_first_response_status` are precise, and showing either to somebody asking
about a parcel is the same mistake as rendering a snake_case enum into a page -
the reader is handed a database column and asked to infer what happened.

So every tool has a phrase, and a test asserts the map covers the projection
matrix. A tool added without one would otherwise appear in the status line
under its own name the first time a customer triggered it, which is exactly the
sort of thing nobody notices until a demo.

Arguments are folded in where they make the line concrete - "Looking up order
ORD-1001" beats "Looking up an order" - and only from `args_public`, which the
server already narrowed to scalars the caller supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

#: Present participle, no trailing ellipsis - the caller adds one. Written from
#: the reader's side: "Checking your response-time target", not "Running the
#: SLA calculator".
TOOL_PHRASES: Final[dict[str, str]] = {
    "search_policy": "Searching the policy documents",
    "resolve_policy": "Working out which policy applies",
    "get_order": "Looking up the order",
    "get_ticket": "Reading the ticket",
    "get_account": "Checking the account",
    "query_tickets": "Searching tickets",
    "my_queue": "Reading the queue",
    "compute_cancellation_fee": "Working out the cancellation fee",
    "compute_service_credit": "Working out the service credit",
    "sla_first_response_status": "Checking the response-time target",
    "check_data_consistency": "Cross-checking the records",
    "scan_support_health": "Scanning for emerging issues",
    "explain_finding": "Explaining the finding",
    "prepare_action": "Preparing an action for you to confirm",
    "execute_action": "Carrying out the action",
    "approve_credit": "Preparing a credit approval",
}

#: The identifier in a tool's arguments worth naming, per tool. One key each:
#: a status line is a sentence, not a dump of the call.
_SUBJECT_KEY: Final[dict[str, str]] = {
    "get_order": "order_id",
    "get_ticket": "ticket_id",
    "get_account": "account_id",
    "resolve_policy": "topic",
    "search_policy": "query",
    "compute_cancellation_fee": "order_id",
    "approve_credit": "account_id",
}

#: Phases that are not tool calls. `model.step` deliberately says "Thinking"
#: rather than naming the model: which model answered is a trace-panel fact,
#: not something a person waiting for a parcel update needs.
PHASE_PHRASES: Final[dict[str, str]] = {
    "run.started": "Starting",
    "model.step": "Thinking",
    "policy.resolved": "Applying the governing policy",
    "conflict.detected": "Found a conflict in the records",
    "facts.block": "Composing the answer",
    "grounding.checked": "Checking the answer against its sources",
    "interrupt.await_confirm": "Waiting for you to confirm",
    "run.escalated": "Drafting an escalation",
    "run.completed": "Done",
    "run.failed": "Something went wrong",
}

#: Long enough to read, short enough not to wrap in a status line.
_MAX_SUBJECT: Final = 48


def describe(event: str, data: Mapping[str, Any] | None = None) -> str | None:
    """One line of status for an event, or None if it deserves no line.

    None rather than a fallback string: an event with nothing useful to say
    should leave the previous line standing, because replacing "Looking up
    order ORD-1001" with "Working" is a downgrade the reader notices.
    """
    payload = data or {}
    if event in {"tool.started", "tool.finished"}:
        return tool_phrase(str(payload.get("name", "")), payload.get("args_public"))
    if event == "tool.denied":
        return "That is not available on this account"
    if event == "tool.error":
        return None
    return PHASE_PHRASES.get(event)


def tool_phrase(name: str, arguments: Mapping[str, Any] | None = None) -> str | None:
    """A tool call as a sentence, with its subject when there is one."""
    phrase = TOOL_PHRASES.get(name)
    if phrase is None:
        # Deliberately not the raw name. An unmapped tool is a gap in this
        # module, and showing `scan_support_health` to a customer is worse
        # than showing nothing - the test below is what keeps it from
        # happening rather than this fallback.
        return None
    subject = _subject(name, arguments or {})
    return f"{phrase} {subject}" if subject else phrase


def _subject(name: str, arguments: Mapping[str, Any]) -> str:
    key = _SUBJECT_KEY.get(name)
    if not key:
        return ""
    value = arguments.get(key)
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Topics are internal slugs; a reader should see "cancellation fee", not
    # "cancellation_fee".
    if key == "topic":
        text = text.replace("_", " ")
    if len(text) > _MAX_SUBJECT:
        text = text[: _MAX_SUBJECT - 1].rstrip() + "…"
    return f'"{text}"' if key in {"query", "topic"} else text


#: Escalation reasons, humanised. `unresolved_conflict` is a database value;
#: showing it to a customer is the same mistake as printing a snake_case
#: column into a page, and it appeared verbatim in the UI until this existed.
ESCALATION_REASONS: Final[dict[str, str]] = {
    "no_citable_source": "Nothing in our documented policies covers this",
    "unresolved_conflict": "Our records disagree and a person needs to settle it",
    "undetermined_severity": "How urgent this is could not be judged automatically",
    "unsupported_claim": "The draft answer went beyond what the sources support",
    "ungrounded": "The answer could not be checked against its sources",
}

#: Denial reasons, likewise. The wording says what the reader can do about it,
#: which for both of these is "nothing, and that is deliberate".
DENIAL_REASONS: Final[dict[str, str]] = {
    "out_of_scope": "that record is not on this account",
    "insufficient_scope": "this role does not have access to that",
}


#: Action kinds as a reader would name them.
ACTION_KINDS: Final[dict[str, str]] = {
    "create_escalation": "Escalation raised",
    "update_ticket_status": "Ticket status updated",
    "create_followup_task": "Follow-up task created",
    "request_carrier_verification": "Carrier verification requested",
    "approve_credit": "Credit approved",
}

#: Keys inside an action payload. The preview is the last thing somebody reads
#: before authorising a change, so it must not be a dump of a JSON object.
PAYLOAD_LABELS: Final[dict[str, str]] = {
    "question": "Their question",
    "unresolved": "What is unresolved",
    "sources": "Sources consulted",
    "evidence_chain": "Evidence",
    "account_id": "Account",
    "order_id": "Order",
    "ticket_id": "Ticket",
    "amount_inr": "Amount (INR)",
    "reason": "Reason",
    "carrier": "Carrier",
    "details": "Details",
    "severity": "Severity",
    "status": "New status",
}


def action_kind(kind: str | None) -> str:
    """What was done, past tense."""
    return ACTION_KINDS.get(str(kind or ""), "Action carried out")


def payload_label(key: str) -> str:
    """A payload key as a heading. Never the raw key."""
    return PAYLOAD_LABELS.get(key, key.replace("_", " ").capitalize())


def escalation_reason(code: str | None) -> str:
    """Why a person was brought in, in a sentence."""
    return ESCALATION_REASONS.get(str(code or ""), "This needs a person to look at it")


def denial_reason(code: str | None) -> str:
    return DENIAL_REASONS.get(str(code or ""), "that is not available here")


def elapsed_label(seconds: float) -> str:
    """A short, non-alarming duration. Whole seconds under a minute."""
    if seconds < 1:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


__all__ = [
    "ACTION_KINDS",
    "DENIAL_REASONS",
    "ESCALATION_REASONS",
    "PAYLOAD_LABELS",
    "PHASE_PHRASES",
    "TOOL_PHRASES",
    "action_kind",
    "denial_reason",
    "describe",
    "elapsed_label",
    "escalation_reason",
    "payload_label",
    "tool_phrase",
]
