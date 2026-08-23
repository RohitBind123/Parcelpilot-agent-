"""Runtime state: sessions, the action log, and the run event stream.

Deliberately not in `parcelpilot.db`. That file is committed, opened read-only
by every repository, and rebuilt from source rather than migrated - so anything
runtime wrote into it would be deleted by the next `build_db.py`. It is also
not in `threads.db`, which LangGraph's checkpointer owns and whose schema is
not ours to share.

Three tables, three different reasons:

`sessions` maps an opaque token to a persona (D17). Nothing in a request body
names a role or an account, so this table is the only thing that can widen a
principal, and it is server-side.

`actions` is the audit trail. ARCHITECTURE 13 calls it immutable, and that is
enforced by triggers rather than by the absence of an `update_action` method -
a method that does not exist stops nobody who can open a connection.

`run_events` is what makes `?from_seq=` work. Every event is persisted with a
monotonic sequence number before it is streamed, so a client that drops mid-run
can ask for everything after the last number it saw.

Time is recorded twice on an action and once everywhere else, and the split is
not incidental. `occurred_at` is domain time, frozen at AS_OF, so an escalation
drafted during the demo is dated alongside the ticket it is about instead of a
week later. `recorded_at` is wall time, so the log can be ordered in the real
world. Conflating them loses one or the other.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from src.clock import as_of, wall_now

logger = logging.getLogger(__name__)

#: Beside the read-only source database, and gitignored. See the module
#: docstring for why it is not inside it.
DEFAULT_RUNTIME_PATH: Final = Path("data/runtime.db")


class ActionKind(StrEnum):
    """The mocked action kinds from ARCHITECTURE 13.

    A closed set rather than a free string: the kind selects the preview
    renderer and the authorisation check, and an unrecognised kind reaching
    either of those is a silent no-op rather than a refusal.
    """

    CREATE_ESCALATION = "create_escalation"
    UPDATE_TICKET_STATUS = "update_ticket_status"
    CREATE_FOLLOWUP_TASK = "create_followup_task"
    REQUEST_CARRIER_VERIFICATION = "request_carrier_verification"
    APPROVE_CREDIT = "approve_credit"


class RuntimeStoreError(RuntimeError):
    """The runtime store refused a write."""


class ImmutableLogError(RuntimeStoreError):
    """Something tried to rewrite history."""


_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS sessions (
    token_id    TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id           TEXT NOT NULL UNIQUE,
    -- The confirmation nonce this action executed under. UNIQUE, so replaying
    -- a token is refused by the same constraint that makes the log immutable,
    -- atomically with the effect rather than by separate bookkeeping that can
    -- drift. NULL is allowed and repeatable in SQLite, for the rare action
    -- appended outside the gate.
    nonce               TEXT UNIQUE,
    kind                TEXT NOT NULL,
    payload_json        TEXT NOT NULL,
    evidence_chain_json TEXT NOT NULL,
    principal_id        TEXT NOT NULL,
    thread_id           TEXT NOT NULL,
    occurred_at         TEXT NOT NULL,
    recorded_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_actions_thread ON actions (thread_id, seq);

-- The immutability from ARCHITECTURE 13, enforced where it cannot be
-- forgotten. RAISE(ABORT) surfaces as sqlite3.IntegrityError.
CREATE TRIGGER IF NOT EXISTS actions_are_append_only_update
BEFORE UPDATE ON actions
BEGIN
    SELECT RAISE(ABORT, 'actions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS actions_are_append_only_delete
BEFORE DELETE ON actions
BEGIN
    SELECT RAISE(ABORT, 'actions is append-only');
END;

CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    thread_id  TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    status     TEXT NOT NULL,
    question   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_runs_persona ON runs (persona_id, updated_at);

CREATE TABLE IF NOT EXISTS run_events (
    run_id       TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    event        TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    ts           TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


@dataclass(frozen=True, slots=True)
class Session:
    token_id: str
    persona_id: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ActionRecord:
    seq: int
    action_id: str
    kind: ActionKind
    payload: Mapping[str, Any]
    evidence_chain: tuple[str, ...]
    principal_id: str
    thread_id: str
    #: The confirmation nonce this executed under, when it went through the
    #: gate. None for an action appended outside it.
    nonce: str | None
    #: Domain time. Frozen at AS_OF, so the record is dated in the world the
    #: answer describes.
    occurred_at: datetime
    #: Wall time. When the row was actually written.
    recorded_at: datetime

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "payload": dict(self.payload),
            "evidence_chain": list(self.evidence_chain),
            "thread_id": self.thread_id,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class Thread:
    thread_id: str
    persona_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    def to_payload(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    thread_id: str
    persona_id: str
    status: str
    question: str
    created_at: datetime
    updated_at: datetime

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "question": self.question,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    seq: int
    event: str
    payload: Mapping[str, Any]
    ts: datetime

    def to_payload(self) -> dict[str, Any]:
        return {"seq": self.seq, "event": self.event, "data": dict(self.payload)}


class RuntimeStore:
    """Read-write runtime state. One connection, created on open."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    @classmethod
    def open(cls, path: Path | str | None = None) -> RuntimeStore:
        resolved = Path(path) if path is not None else _default_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # WAL so the SSE reader and the run writer are not serialised against
        # each other; a stream that blocks on the run it is streaming would
        # deadlock the confirmation gate.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(_SCHEMA)
        connection.commit()
        return cls(connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RuntimeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- sessions -----------------------------------------------------------

    def create_session(self, *, token_id: str, persona_id: str, ttl_seconds: int) -> Session:
        created = wall_now()
        session = Session(
            token_id=token_id,
            persona_id=persona_id,
            created_at=created,
            expires_at=created + timedelta(seconds=ttl_seconds),
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO sessions (token_id, persona_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (
                session.token_id,
                session.persona_id,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
            ),
        )
        self.connection.commit()
        return session

    def get_session(self, token_id: str) -> Session | None:
        """The live session for this token, or None.

        None covers unknown, expired and malformed alike. The caller turns all
        three into the same 401, and distinguishing them in the return type
        would only tempt a route into saying which.
        """
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE token_id = ?", (token_id,)
        ).fetchone()
        if row is None:
            return None

        session = Session(
            token_id=row["token_id"],
            persona_id=row["persona_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )
        if session.expires_at <= wall_now():
            # Deleted rather than merely filtered, so an abandoned session does
            # not sit in the table indefinitely holding a persona binding.
            self.delete_session(token_id)
            return None
        return session

    def delete_session(self, token_id: str) -> None:
        self.connection.execute("DELETE FROM sessions WHERE token_id = ?", (token_id,))
        self.connection.commit()

    # -- the action log -----------------------------------------------------

    def append_action(
        self,
        *,
        kind: ActionKind,
        payload: Mapping[str, Any],
        evidence_chain: Sequence[str],
        principal_id: str,
        thread_id: str,
        action_id: str | None = None,
        nonce: str | None = None,
    ) -> ActionRecord:
        """Write one action. There is no update and no delete, by design.

        Passing the confirmation `nonce` makes the write single-use: a replayed
        token hits the UNIQUE constraint and is refused here, atomically with
        the effect it was trying to repeat.
        """
        resolved_id = action_id or f"act_{_new_id()}"
        occurred, recorded = as_of(), wall_now()
        try:
            cursor = self.connection.execute(
                "INSERT INTO actions (action_id, nonce, kind, payload_json, "
                "evidence_chain_json, principal_id, thread_id, occurred_at, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resolved_id,
                    nonce,
                    kind.value,
                    json.dumps(dict(payload), sort_keys=True),
                    json.dumps(list(evidence_chain)),
                    principal_id,
                    thread_id,
                    occurred.isoformat(),
                    recorded.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            # A replay. Raised rather than ignored: an executed action that
            # silently does nothing the second time is indistinguishable from
            # one that worked.
            if nonce is not None and "nonce" in str(exc):
                raise ImmutableLogError("this confirmation has already been executed") from exc
            raise ImmutableLogError(f"action {resolved_id} is already recorded") from exc
        self.connection.commit()

        return ActionRecord(
            seq=int(cursor.lastrowid or 0),
            action_id=resolved_id,
            kind=kind,
            payload=dict(payload),
            evidence_chain=tuple(evidence_chain),
            principal_id=principal_id,
            thread_id=thread_id,
            nonce=nonce,
            occurred_at=occurred,
            recorded_at=recorded,
        )

    def get_action(self, action_id: str) -> ActionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM actions WHERE action_id = ?", (action_id,)
        ).fetchone()
        return None if row is None else _to_action(row)

    def actions_for_thread(self, thread_id: str) -> tuple[ActionRecord, ...]:
        rows = self.connection.execute(
            "SELECT * FROM actions WHERE thread_id = ? ORDER BY seq", (thread_id,)
        ).fetchall()
        return tuple(_to_action(row) for row in rows)

    # -- threads and runs ---------------------------------------------------

    def upsert_thread(self, *, thread_id: str, persona_id: str, title: str) -> Thread:
        """Create a thread, or touch an existing one.

        The title is set once and then left alone: it comes from the opening
        question, and rewriting it on every message would make the sidebar
        rename itself as a conversation goes on.
        """
        now = wall_now()
        existing = self.get_thread(thread_id)
        if existing is None:
            record = Thread(
                thread_id=thread_id,
                persona_id=persona_id,
                title=title,
                created_at=now,
                updated_at=now,
            )
            self.connection.execute(
                "INSERT INTO threads (thread_id, persona_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, persona_id, title, now.isoformat(), now.isoformat()),
            )
        else:
            record = replace(existing, updated_at=now)
            self.connection.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now.isoformat(), thread_id),
            )
        self.connection.commit()
        return record

    def get_thread(self, thread_id: str) -> Thread | None:
        row = self.connection.execute(
            "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return None if row is None else _to_thread(row)

    def threads_for(self, persona_id: str, limit: int = 50) -> tuple[Thread, ...]:
        rows = self.connection.execute(
            "SELECT * FROM threads WHERE persona_id = ? ORDER BY updated_at DESC LIMIT ?",
            (persona_id, min(limit, 200)),
        ).fetchall()
        return tuple(_to_thread(row) for row in rows)

    def delete_thread(self, thread_id: str) -> None:
        """Forget a conversation. The action log is untouched, deliberately:
        deleting a chat must not delete the record of what it did."""
        self.connection.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        self.connection.commit()

    def create_run(
        self, *, run_id: str, thread_id: str, persona_id: str, question: str
    ) -> RunRecord:
        now = wall_now()
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            persona_id=persona_id,
            status="running",
            question=question,
            created_at=now,
            updated_at=now,
        )
        self.connection.execute(
            "INSERT INTO runs (run_id, thread_id, persona_id, status, question, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, thread_id, persona_id, "running", question, now.isoformat(), now.isoformat()),
        )
        self.connection.commit()
        return record

    def set_run_status(self, run_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, wall_now().isoformat(), run_id),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _to_run(row)

    def active_run_for(self, persona_id: str) -> RunRecord | None:
        """The most recent resumable run, for the reattach flow.

        Only `running` and `awaiting_confirmation` qualify. A completed run has
        nothing to reattach to, and a failed one should not put the client back
        on a stream that will never move.
        """
        row = self.connection.execute(
            "SELECT * FROM runs WHERE persona_id = ? AND status IN "
            "('running', 'awaiting_confirmation') ORDER BY updated_at DESC LIMIT 1",
            (persona_id,),
        ).fetchone()
        return None if row is None else _to_run(row)

    # -- the run event stream -----------------------------------------------

    def append_event(self, *, run_id: str, event: str, payload: Mapping[str, Any]) -> int:
        """Persist one event and return its sequence number.

        Called before the event is streamed, never after. That order is the
        whole reason `?from_seq=` can reattach: an event the client saw is by
        construction already on disk.
        """
        with self.connection:  # one transaction, so the read and write agree
            row = self.connection.execute(
                "SELECT max(seq) FROM run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
            seq = int(row[0] or 0) + 1
            self.connection.execute(
                "INSERT INTO run_events (run_id, seq, event, payload_json, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, event, json.dumps(dict(payload)), wall_now().isoformat()),
            )
        return seq

    def events_since(self, run_id: str, from_seq: int = 0) -> tuple[RunEvent, ...]:
        rows = self.connection.execute(
            "SELECT * FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq",
            (run_id, from_seq),
        ).fetchall()
        return tuple(
            RunEvent(
                run_id=row["run_id"],
                seq=int(row["seq"]),
                event=row["event"],
                payload=json.loads(row["payload_json"]),
                ts=datetime.fromisoformat(row["ts"]),
            )
            for row in rows
        )


@contextmanager
def open_runtime_store(path: Path | str | None = None) -> Iterator[RuntimeStore]:
    store = RuntimeStore.open(path)
    try:
        yield store
    finally:
        store.close()


# -- helpers ----------------------------------------------------------------


def _default_path() -> Path:
    from src.config import get_settings

    return get_settings().runtime_db_path


def _new_id() -> str:
    import secrets

    return secrets.token_hex(8)


def _to_thread(row: sqlite3.Row) -> Thread:
    return Thread(
        thread_id=row["thread_id"],
        persona_id=row["persona_id"],
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        thread_id=row["thread_id"],
        persona_id=row["persona_id"],
        status=row["status"],
        question=row["question"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _to_action(row: sqlite3.Row) -> ActionRecord:
    return ActionRecord(
        seq=int(row["seq"]),
        action_id=row["action_id"],
        kind=ActionKind(row["kind"]),
        payload=json.loads(row["payload_json"]),
        evidence_chain=tuple(json.loads(row["evidence_chain_json"])),
        principal_id=row["principal_id"],
        thread_id=row["thread_id"],
        nonce=row["nonce"],
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
    )


__all__ = [
    "DEFAULT_RUNTIME_PATH",
    "ActionKind",
    "ActionRecord",
    "ImmutableLogError",
    "RunEvent",
    "RunRecord",
    "RuntimeStore",
    "RuntimeStoreError",
    "Session",
    "Thread",
    "open_runtime_store",
]
