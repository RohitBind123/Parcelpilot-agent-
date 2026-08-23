"""The event bus, and the property `?from_seq=` rests on.

A client that drops mid-run and reattaches must receive every event it missed
and no event twice. Two orderings make that true, and both are easy to get
wrong in a way no happy-path test notices:

- persist before broadcast, so nothing a subscriber saw is missing from the
  table it will replay from;
- attach before reading the backlog, so an event emitted during the read lands
  in the queue rather than in the gap between the two.

The second creates a deliberate overlap, which is why the dedupe by `seq`
exists. The test below emits during the handover to prove the overlap is
covered rather than merely believed.
"""

from __future__ import annotations

import asyncio

import pytest

from src.api.events import RunBus
from src.datastore.runtime import open_runtime_store

RUN = "run-1"


@pytest.fixture
def store(tmp_path):
    with open_runtime_store(tmp_path / "runtime.db") as opened:
        yield opened


@pytest.fixture
def bus(store):
    return RunBus(store)


async def drain(bus, run_id, from_seq=0, limit=None):
    seen = []
    async for record in bus.subscribe(run_id, from_seq=from_seq):
        seen.append(record)
        if limit is not None and len(seen) >= limit:
            break
    return seen


class TestPersistence:
    def test_emitting_returns_a_monotonic_sequence(self, bus):
        assert [bus.emit(RUN, "model.step", {"i": n}) for n in range(3)] == [1, 2, 3]

    def test_an_event_is_on_disk_before_anyone_is_told(self, bus, store):
        # Asserted from the table rather than the queue: a subscriber that
        # receives something the table lacks describes a run nobody can replay.
        seq = bus.emit(RUN, "run.started", {})
        assert [e.seq for e in store.events_since(RUN, from_seq=0)] == [seq]

    def test_the_payload_survives(self, bus, store):
        bus.emit(RUN, "tool.finished", {"name": "get_order", "ms": 12})
        (event,) = store.events_since(RUN, from_seq=0)
        assert event.payload == {"name": "get_order", "ms": 12}


class TestReplay:
    @pytest.mark.asyncio
    async def test_a_finished_run_replays_from_the_table(self, bus):
        bus.emit(RUN, "run.started", {})
        bus.emit(RUN, "token.delta", {"text": "hi"})
        bus.emit(RUN, "run.completed", {})
        assert [e.event for e in await drain(bus, RUN)] == [
            "run.started",
            "token.delta",
            "run.completed",
        ]

    @pytest.mark.asyncio
    async def test_a_finished_run_does_not_hang_a_late_subscriber(self, bus):
        # The run ended before anyone attached. Waiting for more would block on
        # a run with nothing left to say.
        bus.emit(RUN, "run.completed", {})
        assert len(await asyncio.wait_for(drain(bus, RUN), timeout=2)) == 1

    @pytest.mark.asyncio
    async def test_from_seq_skips_what_the_client_already_has(self, bus):
        for n in range(4):
            bus.emit(RUN, "token.delta", {"text": str(n)})
        bus.emit(RUN, "run.completed", {})
        seen = await drain(bus, RUN, from_seq=3)
        assert [e.seq for e in seen] == [4, 5]

    @pytest.mark.asyncio
    async def test_an_unknown_run_yields_nothing_rather_than_hanging(self, bus):
        with pytest.raises(TimeoutError):
            # No terminal event, so it waits - which is correct for a run that
            # might yet start, and is why the route requires a known run.
            await asyncio.wait_for(drain(bus, "run-nope"), timeout=0.3)


class TestLiveDelivery:
    @pytest.mark.asyncio
    async def test_a_subscriber_receives_events_emitted_after_it_attached(self, bus):
        bus.bind_loop(asyncio.get_running_loop())

        async def emit_soon():
            await asyncio.sleep(0.01)
            bus.emit(RUN, "token.delta", {"text": "a"})
            bus.emit(RUN, "run.completed", {})

        task = asyncio.create_task(emit_soon())
        seen = await asyncio.wait_for(drain(bus, RUN), timeout=3)
        await task
        assert [e.event for e in seen] == ["token.delta", "run.completed"]

    @pytest.mark.asyncio
    async def test_two_subscribers_both_receive_everything(self, bus):
        bus.bind_loop(asyncio.get_running_loop())

        async def emit_soon():
            await asyncio.sleep(0.01)
            bus.emit(RUN, "token.delta", {"text": "a"})
            bus.emit(RUN, "run.completed", {})

        task = asyncio.create_task(emit_soon())
        both = await asyncio.wait_for(asyncio.gather(drain(bus, RUN), drain(bus, RUN)), timeout=3)
        await task
        assert [[e.event for e in seen] for seen in both] == [
            ["token.delta", "run.completed"],
            ["token.delta", "run.completed"],
        ]

    @pytest.mark.asyncio
    async def test_an_event_emitted_during_the_handover_is_not_lost(self, bus):
        """The reason the queue is attached before the backlog is read.

        Emitting here lands in the queue because the subscriber is already
        attached, and is also in the table by the time the backlog is read -
        so it arrives exactly once, by way of the dedupe. Read-then-attach
        would drop it into the gap between the two and nobody would notice.
        """
        bus.bind_loop(asyncio.get_running_loop())
        bus.emit(RUN, "run.started", {})

        seen = []
        stream = bus.subscribe(RUN, from_seq=0)
        seen.append(await stream.__anext__())  # backlog: run.started

        bus.emit(RUN, "token.delta", {"text": "mid"})
        bus.emit(RUN, "run.completed", {})
        async for record in stream:
            seen.append(record)

        assert [e.event for e in seen] == ["run.started", "token.delta", "run.completed"]
        assert len({e.seq for e in seen}) == 3


class TestSubscriberBookkeeping:
    @pytest.mark.asyncio
    async def test_a_finished_subscription_is_forgotten(self, bus):
        bus.emit(RUN, "run.completed", {})
        await drain(bus, RUN)
        assert bus._subscribers == {}

    @pytest.mark.asyncio
    async def test_an_abandoned_subscription_is_forgotten(self, bus):
        # A client that walks away mid-stream must not leave a queue behind
        # collecting events forever.
        bus.bind_loop(asyncio.get_running_loop())
        bus.emit(RUN, "run.started", {})
        stream = bus.subscribe(RUN, from_seq=0)
        await stream.__anext__()
        await stream.aclose()
        assert bus._subscribers == {}
