"""Typed evidence handles, minted server-side and scoped to a run (D13a).

This is the mechanism that makes the multi-step chain a property of the
signatures rather than a hope about tool ordering.
`compute_cancellation_fee` takes a `resolution_id`, not an `order_id`, so the
model cannot skip the precedence resolver: the argument it needs does not exist
until the resolver has produced one. The chain the brief asks for is guaranteed
without any of it being scripted.

The guarantee is only as good as the handle. A handle appears in model context,
and model context is what a prompt injection reaches, so four things have to
hold: it cannot be guessed, it cannot be invented, it cannot be replayed into
another run, and it cannot be consumed by a Principal other than the one it was
minted for. The last is the important one - staff are refused too, not because
they lack authority over the data, but because provenance is the point. An
agent must re-fetch under their own identity so the trace records who looked.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from src.auth.principal import Principal
from src.clock import wall_now

#: Bytes of entropy per handle. Guessing is not a strategy at this width, which
#: matters because a wrong guess is otherwise indistinguishable from a typo.
_ID_BYTES: Final = 18


class EvidenceKind(StrEnum):
    ORDER_SNAPSHOT = "order_snapshot"
    TICKET_SNAPSHOT = "ticket_snapshot"
    ACCOUNT_SNAPSHOT = "account_snapshot"
    POLICY_RESOLUTION = "policy_resolution"
    CALC_RESULT = "calc_result"
    CONSISTENCY_REPORT = "consistency_report"


#: Short prefixes so a handle is legible in a trace and in model context.
#: Decoration only - the kind is always enforced from the stored row.
_PREFIX: Final[dict[EvidenceKind, str]] = {
    EvidenceKind.ORDER_SNAPSHOT: "snap",
    EvidenceKind.TICKET_SNAPSHOT: "tsnap",
    EvidenceKind.ACCOUNT_SNAPSHOT: "asnap",
    EvidenceKind.POLICY_RESOLUTION: "res",
    EvidenceKind.CALC_RESULT: "calc",
    EvidenceKind.CONSISTENCY_REPORT: "rep",
}

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id     TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    principal_hash  TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    derived_from    TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_evidence_run ON evidence (run_id);
"""


class EvidenceError(RuntimeError):
    """A handle could not be used."""


class EvidenceNotFound(EvidenceError):
    """No such handle. Almost always a handle the model invented."""


class EvidenceScopeError(EvidenceError):
    """The handle exists but belongs to another run or another Principal.

    Deliberately carries no payload detail: the message is written into a tool
    result the model can see, and a denial that quotes what it is denying has
    not denied anything.
    """


class EvidenceKindError(EvidenceError):
    """The handle is of the wrong kind for where it was passed."""


@dataclass(frozen=True, slots=True)
class Handle:
    """A minted reference to stored evidence."""

    evidence_id: str
    kind: EvidenceKind

    def __str__(self) -> str:
        return self.evidence_id


def principal_fingerprint(principal: Principal) -> str:
    """A stable digest of who this is.

    Compared for equality on every read, never reversed, so a plain digest is
    sufficient - there is no secret here to protect, only an identity to match.
    """
    identity = f"{principal.user_id}|{principal.role}|{principal.account_id or ''}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class EvidenceStore:
    """Run-scoped, Principal-scoped evidence."""

    def __init__(
        self,
        *,
        run_id: str,
        principal: Principal,
        connection: sqlite3.Connection,
    ) -> None:
        self.run_id = run_id
        self.connection = connection
        self._fingerprint = principal_fingerprint(principal)
        self.connection.executescript(SCHEMA)

    # -- writing ------------------------------------------------------------

    def mint(
        self,
        kind: EvidenceKind,
        payload: Mapping[str, Any],
        *,
        derived_from: Sequence[Handle | str] = (),
    ) -> Handle:
        """Store a payload and return a handle to it.

        There is no `evidence_id` parameter on purpose. If the caller could
        choose the id, a model could pass one it invented and the store would
        create it, which is the whole attack.
        """
        # Serialise before writing anything, so a bad value fails here rather
        # than on read, when whatever produced it is long gone.
        payload_json = json.dumps(dict(payload), sort_keys=True)

        parents = [self._resolve_id(h) for h in derived_from]
        for parent in parents:
            # Reading it is the check: a parent from another run or Principal
            # raises, and an evidence chain must not span identities.
            self._row(parent)

        handle = Handle(f"{_PREFIX[kind]}_{secrets.token_urlsafe(_ID_BYTES)}", kind)
        self.connection.execute(
            "INSERT INTO evidence "
            "(evidence_id, kind, run_id, principal_hash, payload_json, derived_from, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                handle.evidence_id,
                str(kind),
                self.run_id,
                self._fingerprint,
                payload_json,
                json.dumps(parents),
                wall_now().isoformat(),
            ),
        )
        self.connection.commit()
        return handle

    # -- reading ------------------------------------------------------------

    def read(self, handle: Handle | str, *, expect: EvidenceKind) -> dict[str, Any]:
        """The payload behind a handle, or an exception.

        `expect` is required rather than optional. An optional type check is one
        a caller forgets, and the forgotten case is a resolution handle arriving
        where a snapshot belongs - which computes something plausible from the
        wrong evidence.
        """
        row = self._row(self._resolve_id(handle))
        if row["kind"] != str(expect):
            raise EvidenceKindError(f"handle is a {row['kind']}, but a {expect} was required here")
        return json.loads(row["payload_json"])

    def kind_of(self, handle: Handle | str) -> EvidenceKind:
        return EvidenceKind(self._row(self._resolve_id(handle))["kind"])

    def provenance(self, handle: Handle | str) -> tuple[str, ...]:
        """The handles this one was derived from, in the order given.

        Walked by the grounding gate and rendered by the trace panel. It has to
        be recorded at mint time: afterwards nothing knows what produced what.
        """
        return tuple(json.loads(self._row(self._resolve_id(handle))["derived_from"]))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _resolve_id(handle: Handle | str) -> str:
        # Tool arguments arrive from the model as strings, so a string has to be
        # accepted. It is never trusted - it is looked up like any other.
        return handle.evidence_id if isinstance(handle, Handle) else str(handle)

    def _row(self, evidence_id: str) -> sqlite3.Row:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            "SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise EvidenceNotFound(f"no evidence handle {evidence_id!r}")
        if row["run_id"] != self.run_id or row["principal_hash"] != self._fingerprint:
            raise EvidenceScopeError(
                f"evidence handle {evidence_id!r} does not belong to this session"
            )
        return row

    def close(self) -> None:
        self.connection.close()


@contextmanager
def open_evidence_store(
    *,
    run_id: str,
    principal: Principal,
    path: Path | str | None = None,
    connection: sqlite3.Connection | None = None,
):
    """Open a store, defaulting to an in-process database.

    Evidence is run-scoped working state, not part of the committed dataset, so
    it deliberately does not live in `parcelpilot.db` - that file is opened
    read-only and rebuilt from source.
    """
    owned = connection is None
    conn = connection or sqlite3.connect(str(path) if path else ":memory:")
    store = EvidenceStore(run_id=run_id, principal=principal, connection=conn)
    try:
        yield store
    finally:
        if owned:
            conn.close()
