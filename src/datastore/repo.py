"""Principal-bound read access to the structured store.

This is the second ACL layer. The first is the tool schema: a customer's
`get_order` has no `account_id` parameter, so a cross-account query is not
expressible by the model. This layer exists for when that one has a bug.

Scoping is done by **binding account-scoped temp views at connect time**, so a
customer's connection cannot see another account's rows at all. The queries
below name `my_orders`, `my_tickets` and `my_accounts` and are identical for
every role; what differs is what those views resolve to. That is deliberate:
a WHERE clause is something a future query can forget, a view definition is not.

The connection is then set read-only. State changes go through the action path
(M8), never through here.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from src.auth.principal import (
    SCOPE_AGGREGATE_TICKETS,
    SCOPE_OWN_QUEUE,
    Principal,
)
from src.datastore.models import Account, Order, Ticket

logger = logging.getLogger(__name__)

#: The account id is interpolated into a view definition, so it is validated
#: even though it only ever originates from the server-side persona table.
_ACCOUNT_ID = re.compile(r"^ACCT-\d{3}$")

#: Every list read is bounded. No endpoint returns an unbounded collection.
DEFAULT_LIMIT: Final = 50
MAX_LIMIT: Final = 200

_TICKET_FILTERS: Final = frozenset({"status", "account_id", "assigned_to"})

#: Same message for "does not exist" and "not yours". The distinction is
#: recorded in the log and on the exception, never in the text a user sees.
_DENIAL_TEXT: Final = "not available on this account"


class RepositoryError(RuntimeError):
    """Base class for read failures."""


class NotFound(RepositoryError):
    """No such record. Raised for staff, who already read every account."""


class AccessDenied(RepositoryError):
    """The caller may not have this record.

    `reason` distinguishes `not_found` from `out_of_scope` for logging and for
    the trace panel. The rendered message does not, so a denial cannot be used
    to probe which identifiers exist.
    """

    def __init__(self, detail: str, *, reason: str) -> None:
        super().__init__(detail)
        self.reason = reason


class Repository:
    """Read surface for one Principal."""

    def __init__(self, connection: sqlite3.Connection, principal: Principal) -> None:
        self.connection = connection
        self.principal = principal

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, principal: Principal, db_path: Path | str | None = None) -> Repository:
        if db_path is None:
            from src.config import get_settings

            db_path = get_settings().db_path

        connection = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _bind_scoped_views(connection, principal)
        # Views are in place; nothing else on this connection may write.
        connection.execute("PRAGMA query_only = ON")
        return cls(connection, principal)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Repository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- single-record reads ----------------------------------------------

    def get_account(self, account_id: str | None = None) -> Account:
        account_id = self._resolve_account(account_id)
        sql = "SELECT * FROM my_accounts"
        params: tuple[str, ...] = ()
        if account_id:
            sql += " WHERE account_id = ?"
            params = (account_id,)
        row = self.connection.execute(sql, params).fetchone()
        if row is None:
            raise self._miss("accounts", "account_id", account_id or "<own>")
        return Account.from_row(row)

    def get_order(self, order_id: str) -> Order:
        row = self.connection.execute(
            "SELECT * FROM my_orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            raise self._miss("orders", "order_id", order_id)
        return Order.from_row(row)

    def get_ticket(self, ticket_id: str) -> Ticket:
        row = self.connection.execute(
            "SELECT * FROM my_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise self._miss("tickets", "ticket_id", ticket_id)
        return Ticket.from_row(row)

    # -- collection reads --------------------------------------------------

    def get_orders(self, order_ids: Sequence[str]) -> dict[str, Order]:
        """Batch fetch. One round-trip for the whole set, never one per id."""
        ids = list(dict.fromkeys(order_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.connection.execute(
            f"SELECT * FROM my_orders WHERE order_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["order_id"]: Order.from_row(row) for row in rows}

    def list_orders(
        self, account_id: str | None = None, *, limit: int = DEFAULT_LIMIT
    ) -> list[Order]:
        account_id = self._resolve_account(account_id)
        sql = "SELECT * FROM my_orders"
        params: list[object] = []
        if account_id:
            sql += " WHERE account_id = ?"
            params.append(account_id)
        sql += " ORDER BY booked_at DESC LIMIT ?"
        params.append(_bounded(limit))
        return [Order.from_row(r) for r in self.connection.execute(sql, params)]

    def list_tickets(
        self, account_id: str | None = None, *, limit: int = DEFAULT_LIMIT
    ) -> list[Ticket]:
        account_id = self._resolve_account(account_id)
        sql = "SELECT * FROM my_tickets"
        params: list[object] = []
        if account_id:
            sql += " WHERE account_id = ?"
            params.append(account_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(_bounded(limit))
        return [Ticket.from_row(r) for r in self.connection.execute(sql, params)]

    def query_tickets(self, *, limit: int = DEFAULT_LIMIT, **filters: object) -> list[Ticket]:
        """Parameterised ticket search. Staff only.

        A builder rather than free-text SQL: the model chooses filters, never
        a query. An unrecognised filter raises instead of being dropped -
        silently ignoring one returns a wider result set than the caller asked
        for, which on an ACL-adjacent read is the wrong way to fail.
        """
        self._require(SCOPE_AGGREGATE_TICKETS)

        unknown = set(filters) - _TICKET_FILTERS
        if unknown:
            raise ValueError(
                f"unsupported ticket filter(s): {sorted(unknown)}; "
                f"expected a subset of {sorted(_TICKET_FILTERS)}"
            )

        clauses = [f"{name} = ?" for name, value in filters.items() if value is not None]
        params: list[object] = [v for v in filters.values() if v is not None]
        sql = "SELECT * FROM my_tickets"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(_bounded(limit))
        return [Ticket.from_row(r) for r in self.connection.execute(sql, params)]

    def my_queue(self, *, limit: int = DEFAULT_LIMIT) -> list[Ticket]:
        """Tickets assigned to this staff member, via `tickets.assigned_to`."""
        self._require(SCOPE_OWN_QUEUE)
        if not self.principal.queue_key:
            return []
        return [
            Ticket.from_row(r)
            for r in self.connection.execute(
                "SELECT * FROM my_tickets WHERE assigned_to = ? ORDER BY created_at DESC LIMIT ?",
                (self.principal.queue_key, _bounded(limit)),
            )
        ]

    # -- internals ---------------------------------------------------------

    def _require(self, scope: str) -> None:
        try:
            self.principal.require(scope)
        except PermissionError as exc:
            logger.warning("scope denied: %s", exc)
            raise AccessDenied(str(exc), reason="out_of_scope") from exc

    def _resolve_account(self, requested: str | None) -> str | None:
        """Narrow a requested account to what this Principal may read.

        A customer passing an account_id cannot widen scope; a customer passing
        nothing gets its own. Staff get whatever they asked for, or everything.
        """
        if self.principal.is_staff:
            return requested
        if requested and requested != self.principal.account_id:
            logger.warning(
                "cross-account read refused: user=%s own=%s requested=%s",
                self.principal.user_id,
                self.principal.account_id,
                requested,
            )
            raise AccessDenied(f"account {requested} is {_DENIAL_TEXT}", reason="out_of_scope")
        return self.principal.account_id

    def _miss(self, table: str, column: str, value: str) -> RepositoryError:
        """Turn an empty scoped read into the right error.

        For staff the answer is simply "no such record". For a customer the
        record may exist on another account, and the two cases must be
        indistinguishable from outside - so both raise the same message, and
        only the log and `reason` say which happened.
        """
        if self.principal.is_staff:
            return NotFound(f"no {table[:-1]} {value}")

        exists = self.connection.execute(
            f"SELECT 1 FROM main.{table} WHERE {column} = ? LIMIT 1",
            (value,),
        ).fetchone()
        reason = "out_of_scope" if exists else "not_found"
        logger.warning(
            "scoped read miss: user=%s account=%s %s=%s reason=%s",
            self.principal.user_id,
            self.principal.account_id,
            column,
            value,
            reason,
        )
        return AccessDenied(f"{value} is {_DENIAL_TEXT}", reason=reason)


def _bounded(limit: int) -> int:
    """Clamp a caller-supplied limit. No read is unbounded."""
    return max(1, min(int(limit), MAX_LIMIT))


def _bind_scoped_views(connection: sqlite3.Connection, principal: Principal) -> None:
    """Create the temp views this Principal reads through.

    For staff these alias the base tables. For a customer they carry the
    account predicate, so the connection has no way to observe another
    account's rows even through a hand-written query.
    """
    if principal.is_staff:
        predicate = ""
    else:
        account_id = principal.account_id or ""
        if not _ACCOUNT_ID.match(account_id):
            raise ValueError(f"refusing to bind views for malformed account_id {account_id!r}")
        predicate = f" WHERE account_id = '{account_id}'"

    for view, table in (
        ("my_accounts", "accounts"),
        ("my_orders", "orders"),
        ("my_tickets", "tickets"),
    ):
        connection.execute(f"CREATE TEMP VIEW {view} AS SELECT * FROM main.{table}{predicate}")


def open_repository(principal: Principal, db_path: Path | str | None = None) -> Repository:
    """Convenience alias mirroring the tool-layer naming."""
    return Repository.open(principal, db_path)


__all__ = [
    "AccessDenied",
    "Account",
    "NotFound",
    "Order",
    "Repository",
    "RepositoryError",
    "Ticket",
    "open_repository",
]
