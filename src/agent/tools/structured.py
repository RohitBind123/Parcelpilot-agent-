"""Structured reads: orders, tickets, accounts, queues.

Every one of these mints an evidence handle rather than returning a row. That
is not ceremony - the calculators take handles and nothing else, so a snapshot
is the only currency in which a lookup can be spent. It also means the trace
shows exactly which record an answer was computed from, at the moment it was
read.

The account scoping is done twice on purpose, and neither is redundant. The
schema a customer sees has no `account_id` parameter, so a cross-account lookup
cannot be expressed; and the repository reads through account-scoped SQL views,
so a bug in a tool body still cannot reach another account's rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agent.tools.base import DenialReason, Param, Tool, ToolDenied, ToolResult
from src.auth.principal import SCOPE_AGGREGATE_TICKETS, SCOPE_OWN_QUEUE
from src.datastore.repo import AccessDenied, NotFound
from src.domain.evidence import EvidenceKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

#: Enough for any queue in the pack, and bounded so no read is open-ended.
QUEUE_LIMIT = 50


def _denied(kind: str, identifier: str | None) -> ToolDenied:
    return ToolDenied(DenialReason.OUT_OF_SCOPE, kind, identifier)


def _summary(payload: dict, keys: tuple[str, ...]) -> dict:
    """The few fields worth putting in model context beside the handle.

    Not the whole row. The handle is what the calculators consume, and a full
    row in context is a row the model can quote from without having computed
    anything.
    """
    return {key: payload.get(key) for key in keys}


ORDER_SUMMARY = ("order_id", "account_id", "status", "carrier", "booked_at", "pickup_actual_at")
TICKET_SUMMARY = ("ticket_id", "account_id", "status", "subject", "created_at", "assigned_to")
ACCOUNT_SUMMARY = ("account_id", "account_name", "plan", "premium_support")


def get_order(context: ToolContext) -> Tool:
    staff = context.principal.is_staff

    def run(order_id: str, account_id: str | None = None) -> ToolResult | ToolDenied:
        try:
            order = context.repository.get_order(order_id)
        except (AccessDenied, NotFound):
            return _denied("order", order_id)
        if account_id and order.account_id != account_id:
            # Staff may name an account; naming the wrong one is a mistake
            # worth reporting rather than quietly ignoring.
            return _denied("order", order_id)
        payload = order.to_payload()
        handle = context.store.mint(EvidenceKind.ORDER_SNAPSHOT, payload)
        return ToolResult({"snapshot_id": handle.evidence_id, **_summary(payload, ORDER_SUMMARY)})

    params = [Param("order_id", "string", "Order identifier, for example ORD-1001.")]
    if staff:
        params.append(Param("account_id", "string", "Restrict to this account.", required=False))
    return Tool(
        name="get_order",
        description=(
            "Look up an order and take a snapshot of it. Returns a snapshot_id that the "
            "cancellation and credit calculators require."
            if not staff
            else "Look up any order and take a snapshot of it. Returns a snapshot_id."
        ),
        params=tuple(params),
        run=run,
    )


def get_ticket(context: ToolContext) -> Tool:
    staff = context.principal.is_staff

    def run(ticket_id: str, account_id: str | None = None) -> ToolResult | ToolDenied:
        try:
            ticket = context.repository.get_ticket(ticket_id)
        except (AccessDenied, NotFound):
            return _denied("ticket", ticket_id)
        if account_id and ticket.account_id != account_id:
            return _denied("ticket", ticket_id)
        payload = ticket.to_payload()
        handle = context.store.mint(EvidenceKind.TICKET_SNAPSHOT, payload)
        summary = _summary(payload, TICKET_SUMMARY)
        if staff:
            # A recorded resolution is Tier 5 and may be wrong; it is shown to
            # staff so `check_data_consistency` has something to be asked
            # about, and withheld from customers so it cannot be repeated.
            summary["has_recorded_resolution"] = payload.get("historical_resolution") is not None
        return ToolResult({"snapshot_id": handle.evidence_id, **summary})

    params = [Param("ticket_id", "string", "Ticket identifier, for example TKT-501.")]
    if staff:
        params.append(Param("account_id", "string", "Restrict to this account.", required=False))
    return Tool(
        name="get_ticket",
        description="Look up a ticket and take a snapshot of it. Returns a snapshot_id.",
        params=tuple(params),
        run=run,
    )


def get_account(context: ToolContext) -> Tool:
    staff = context.principal.is_staff

    def run(account_id: str | None = None) -> ToolResult | ToolDenied:
        try:
            account = context.repository.get_account(account_id)
        except (AccessDenied, NotFound):
            return _denied("account", account_id)
        payload = account.to_payload()
        handle = context.store.mint(EvidenceKind.ACCOUNT_SNAPSHOT, payload)
        return ToolResult(
            {"account_snapshot_id": handle.evidence_id, **_summary(payload, ACCOUNT_SUMMARY)}
        )

    params = (
        (Param("account_id", "string", "Account identifier, for example ACCT-001."),)
        if staff
        else ()
    )
    return Tool(
        name="get_account",
        description=(
            "Your account and plan. Returns an account_snapshot_id, which the SLA "
            "calculator requires."
            if not staff
            else "Any account and its plan. Returns an account_snapshot_id."
        ),
        params=params,
        run=run,
    )


def query_tickets(context: ToolContext) -> Tool:
    def run(
        status: str | None = None,
        account_id: str | None = None,
        assigned_to: str | None = None,
        limit: int = QUEUE_LIMIT,
    ) -> ToolResult:
        tickets = context.repository.query_tickets(
            status=status, account_id=account_id, assigned_to=assigned_to, limit=limit
        )
        return ToolResult(
            {
                "count": len(tickets),
                "tickets": [_summary(t.to_payload(), TICKET_SUMMARY) for t in tickets],
            }
        )

    return Tool(
        name="query_tickets",
        description="Search tickets by status, account or assignee. Staff only.",
        params=(
            Param("status", "string", "For example open or closed.", required=False),
            Param("account_id", "string", "Restrict to one account.", required=False),
            Param("assigned_to", "string", "Agent name, for example Maya.", required=False),
            Param("limit", "integer", f"Maximum rows, capped at {QUEUE_LIMIT}.", required=False),
        ),
        run=run,
        requires_scope=SCOPE_AGGREGATE_TICKETS,
    )


def my_queue(context: ToolContext) -> Tool:
    def run() -> ToolResult:
        tickets = context.repository.my_queue(limit=QUEUE_LIMIT)
        return ToolResult(
            {
                "count": len(tickets),
                "tickets": [_summary(t.to_payload(), TICKET_SUMMARY) for t in tickets],
            }
        )

    return Tool(
        name="my_queue",
        description="The tickets assigned to you. Takes no arguments.",
        params=(),
        run=run,
        requires_scope=SCOPE_OWN_QUEUE,
    )
