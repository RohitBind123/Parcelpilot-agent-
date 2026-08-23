"""The runtime store: sessions, the action log, and the run event stream.

Three tables that exist for three different reasons, in a file deliberately
separate from `parcelpilot.db`. That database is committed, opened read-only,
and rebuilt from source rather than migrated - so the next `build_db.py` run
would delete anything runtime wrote into it.

The property worth most here is that the action log cannot be rewritten.
"Executed actions append to an immutable table" (ARCHITECTURE 13) is only worth
saying if something enforces it, and a class with no `update` method enforces
nothing: the next caller opens a connection. The triggers below are the
enforcement, so these tests go around the Python API and issue the SQL directly.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.clock import as_of, wall_now
from src.datastore.runtime import (
    ActionKind,
    ImmutableLogError,
    RuntimeStore,
    open_runtime_store,
)

EVIDENCE = ("ev_snapshot_1", "ev_resolution_2")


@pytest.fixture
def store(tmp_path):
    with open_runtime_store(tmp_path / "runtime.db") as opened:
        yield opened


def append(store: RuntimeStore, **overrides):
    payload = {
        "kind": ActionKind.CREATE_ESCALATION,
        "payload": {"question": "how do I change the billing contact?"},
        "evidence_chain": EVIDENCE,
        "principal_id": "maya_agent",
        "thread_id": "thread-1",
    }
    payload.update(overrides)
    return store.append_action(**payload)


class TestOpening:
    def test_the_schema_is_created_on_first_open(self, tmp_path):
        with open_runtime_store(tmp_path / "runtime.db") as store:
            names = {
                row[0]
                for row in store.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        assert {"sessions", "actions", "run_events"} <= names

    def test_reopening_an_existing_file_keeps_what_is_in_it(self, tmp_path):
        path = tmp_path / "runtime.db"
        with open_runtime_store(path) as store:
            recorded = append(store)
        with open_runtime_store(path) as reopened:
            assert reopened.get_action(recorded.action_id) is not None

    def test_the_parent_directory_is_created(self, tmp_path):
        with open_runtime_store(tmp_path / "nested" / "deeper" / "runtime.db") as store:
            assert store.connection is not None


class TestSessions:
    def test_a_session_round_trips_to_its_persona(self, store):
        store.create_session(token_id="tok_1", persona_id="maya_agent", ttl_seconds=3600)
        found = store.get_session("tok_1")
        assert found is not None
        assert found.persona_id == "maya_agent"

    def test_an_unknown_token_is_none_rather_than_an_error(self, store):
        # The caller turns this into a 401. An exception here would have every
        # route wrapping it in a try block to say the same thing.
        assert store.get_session("tok_nope") is None

    def test_an_expired_session_is_not_returned(self, store):
        store.create_session(token_id="tok_2", persona_id="maya_agent", ttl_seconds=-1)
        assert store.get_session("tok_2") is None

    def test_an_expired_session_is_gone_rather_than_hidden(self, store):
        # Filtered on read *and* deleted, so an abandoned session does not sit
        # in the table forever holding a persona binding.
        store.create_session(token_id="tok_3", persona_id="maya_agent", ttl_seconds=-1)
        store.get_session("tok_3")
        remaining = store.connection.execute(
            "SELECT count(*) FROM sessions WHERE token_id = ?", ("tok_3",)
        ).fetchone()[0]
        assert remaining == 0

    def test_logging_out_removes_the_session(self, store):
        store.create_session(token_id="tok_4", persona_id="priya_manager", ttl_seconds=3600)
        store.delete_session("tok_4")
        assert store.get_session("tok_4") is None

    def test_sessions_use_wall_time_not_the_frozen_clock(self, store):
        # A session expiring is an infrastructure fact. Anchoring it to AS_OF
        # would make every session in the demo already six days stale.
        store.create_session(token_id="tok_5", persona_id="maya_agent", ttl_seconds=3600)
        found = store.get_session("tok_5")
        assert found is not None
        assert abs((found.created_at - wall_now()).total_seconds()) < 60


class TestTheActionLog:
    def test_an_action_round_trips_with_its_evidence_chain(self, store):
        recorded = append(store)
        found = store.get_action(recorded.action_id)
        assert found is not None
        assert found.evidence_chain == EVIDENCE
        assert found.kind is ActionKind.CREATE_ESCALATION

    def test_the_payload_survives_the_round_trip(self, store):
        recorded = append(store, payload={"amount": 500, "nested": {"tier": "P2"}})
        found = store.get_action(recorded.action_id)
        assert found is not None
        assert found.payload == {"amount": 500, "nested": {"tier": "P2"}}

    def test_actions_carry_both_domain_time_and_wall_time(self, store):
        # Two different facts. `occurred_at` is when this happened in the world
        # the answer describes, which is frozen; `recorded_at` is when the row
        # was written. Conflating them would either date the escalation a week
        # after its ticket or make the audit trail unorderable in real time.
        recorded = append(store)
        assert recorded.occurred_at == as_of()
        assert abs((recorded.recorded_at - wall_now()).total_seconds()) < 60

    def test_actions_for_a_thread_come_back_in_the_order_written(self, store):
        first = append(store, payload={"n": 1})
        second = append(store, payload={"n": 2})
        third = append(store, payload={"n": 3}, thread_id="thread-2")
        found = store.actions_for_thread("thread-1")
        assert [a.action_id for a in found] == [first.action_id, second.action_id]
        assert third.action_id not in {a.action_id for a in found}

    def test_ordering_does_not_depend_on_the_frozen_timestamp(self, store):
        # Every action in a demo run shares one `occurred_at`, so ordering by
        # it would be arbitrary. The sequence is what orders the log.
        appended = [append(store, payload={"n": n}) for n in range(5)]
        assert [a.seq for a in appended] == sorted(a.seq for a in appended)
        assert len({a.seq for a in appended}) == 5


class TestTheActionLogIsImmutable:
    """Enforced in the database, not in the Python API.

    A class that simply has no `update_action` method is a convention. These
    tests bypass the class entirely, because the next caller can too.
    """

    def test_an_action_cannot_be_updated(self, store):
        recorded = append(store)
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "UPDATE actions SET payload_json = ? WHERE action_id = ?",
                ('{"tampered": true}', recorded.action_id),
            )

    def test_an_action_cannot_be_deleted(self, store):
        recorded = append(store)
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "DELETE FROM actions WHERE action_id = ?", (recorded.action_id,)
            )

    def test_the_row_really_is_still_there_afterwards(self, store):
        recorded = append(store)
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("DELETE FROM actions")
        assert store.get_action(recorded.action_id) is not None

    def test_appending_the_same_action_id_twice_is_refused(self, store):
        recorded = append(store)
        with pytest.raises(ImmutableLogError):
            store.append_action(
                action_id=recorded.action_id,
                kind=ActionKind.UPDATE_TICKET_STATUS,
                payload={},
                evidence_chain=(),
                principal_id="maya_agent",
                thread_id="thread-1",
            )


class TestTheRunEventStream:
    def test_the_first_event_of_a_run_is_seq_one(self, store):
        assert store.append_event(run_id="run-1", event="run.started", payload={}) == 1

    def test_sequence_numbers_are_monotonic_within_a_run(self, store):
        seqs = [
            store.append_event(run_id="run-1", event="token.delta", payload={"text": str(n)})
            for n in range(4)
        ]
        assert seqs == [1, 2, 3, 4]

    def test_two_runs_number_independently(self, store):
        store.append_event(run_id="run-1", event="run.started", payload={})
        store.append_event(run_id="run-1", event="model.step", payload={})
        assert store.append_event(run_id="run-2", event="run.started", payload={}) == 1

    def test_events_since_returns_only_what_came_after(self, store):
        for n in range(5):
            store.append_event(run_id="run-1", event="token.delta", payload={"text": str(n)})
        found = store.events_since("run-1", from_seq=3)
        assert [e.seq for e in found] == [4, 5]

    def test_from_seq_zero_replays_the_whole_run(self, store):
        for n in range(3):
            store.append_event(run_id="run-1", event="token.delta", payload={"text": str(n)})
        assert len(store.events_since("run-1", from_seq=0)) == 3

    def test_an_unknown_run_replays_as_empty(self, store):
        assert store.events_since("run-nope", from_seq=0) == ()

    def test_the_payload_survives_the_round_trip(self, store):
        store.append_event(
            run_id="run-1", event="tool.finished", payload={"name": "get_order", "ms": 12}
        )
        (event,) = store.events_since("run-1", from_seq=0)
        assert event.payload == {"name": "get_order", "ms": 12}
        assert event.event == "tool.finished"

    def test_a_duplicate_sequence_number_is_impossible(self, store):
        store.append_event(run_id="run-1", event="run.started", payload={})
        # `?from_seq=` reattachment is only correct if seq is unique per run.
        # The uniqueness is a constraint, not an artefact of how the helper
        # happens to count.
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO run_events (run_id, seq, event, payload_json, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                ("run-1", 1, "run.started", "{}", wall_now().isoformat()),
            )
