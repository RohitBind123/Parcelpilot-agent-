"""Run events: persisted, then broadcast (ARCHITECTURE 16).

Every event is written to `run_events` with a monotonic `seq` **before** any
subscriber sees it. That order is the whole reason `?from_seq=` works: an event
a client saw is on disk by construction, so a client that reconnects and asks
for everything after the last number it holds cannot be told about a gap that
was never recorded.

Live delivery is an in-process broadcast rather than polling the table, because
polling adds latency to every token of a streamed answer for no benefit when
the writer and the reader are in the same process. The table is what makes
reattachment correct; the queue is what makes the first attachment fast.

Subscribing attaches to the queue **before** reading the backlog, and drops
anything from the queue whose `seq` the backlog already covered. The other
order has a hole: an event emitted between the read and the attach is in
neither, and the client silently never receives it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, Final

from src.datastore.runtime import RunEvent, RuntimeStore

logger = logging.getLogger(__name__)

#: After one of these the run is over and emits nothing further.
TERMINAL: Final[frozenset[str]] = frozenset({"run.completed", "run.failed"})

#: After one of these a subscription ends, which is not the same as the run
#: ending. A run parked on `interrupt.await_confirm` is waiting for a person to
#: read a card and decide, and that is human think-time - possibly minutes. The
#: client already has the preview and the token by then, so holding a
#: connection open buys nothing; it confirms, then reattaches with `?from_seq=`
#: at the sequence it stopped on. That makes the reattach path the normal flow
#: rather than an error path nobody exercises until it breaks.
CLOSES_STREAM: Final[frozenset[str]] = TERMINAL | {"interrupt.await_confirm"}

#: Bounded so a subscriber that stops reading cannot grow without limit. A
#: dropped event is not a correctness problem - the client reattaches with
#: `?from_seq=` and the table still has it - but a queue that eats memory is.
_QUEUE_LIMIT: Final = 1000


class RunBus:
    """Persists run events and fans them out to live subscribers."""

    def __init__(self, store: RuntimeStore) -> None:
        self._store = store
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | None]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the loop that owns the subscriber queues.

        Runs execute in a worker thread, so `emit` is called from off the loop
        and cannot touch an `asyncio.Queue` directly.
        """
        self._loop = loop

    def emit(self, run_id: str, event: str, payload: Mapping[str, Any] | None = None) -> int:
        """Persist one event and hand it to any live subscriber.

        Returns its sequence number. Persistence first, always: a subscriber
        that receives something the table does not have would be describing a
        run that cannot be replayed.
        """
        body = dict(payload or {})
        seq = self._store.append_event(run_id=run_id, event=event, payload=body)
        record = RunEvent(
            run_id=run_id,
            seq=seq,
            event=event,
            payload=body,
            ts=self._store.events_since(run_id, from_seq=seq - 1)[0].ts,
        )
        self._publish(run_id, record)
        if event in CLOSES_STREAM:
            self._publish(run_id, None)
        return seq

    def _publish(self, run_id: str, record: RunEvent | None) -> None:
        queues = list(self._subscribers.get(run_id, ()))
        if not queues:
            return
        loop = self._loop
        for queue in queues:
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(_offer, queue, record)
            else:
                _offer(queue, record)

    async def subscribe(self, run_id: str, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        """Everything after `from_seq`, then everything that follows live."""
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=_QUEUE_LIMIT)
        self._subscribers.setdefault(run_id, []).append(queue)
        try:
            highest = from_seq
            finished = False
            for record in self._store.events_since(run_id, from_seq=from_seq):
                highest = record.seq
                yield record
                if record.event in CLOSES_STREAM:
                    # The run ended, or is parked waiting for a person, before
                    # this client attached. Either way the backlog is
                    # everything there is; waiting would hang on a run with
                    # nothing more to say until somebody acts.
                    finished = True
            if finished:
                return

            while True:
                record = await queue.get()
                if record is None:
                    return
                if record.seq <= highest:
                    # Already delivered from the backlog. This is the overlap
                    # the attach-then-read order deliberately creates.
                    continue
                highest = record.seq
                yield record
                if record.event in CLOSES_STREAM:
                    return
        finally:
            remaining = self._subscribers.get(run_id, [])
            if queue in remaining:
                remaining.remove(queue)
            if not remaining:
                self._subscribers.pop(run_id, None)


def _offer(queue: asyncio.Queue[RunEvent | None], record: RunEvent | None) -> None:
    try:
        queue.put_nowait(record)
    except asyncio.QueueFull:
        # The client is not keeping up. It can reattach with `?from_seq=` and
        # the table still holds everything, so this costs latency, not data.
        logger.warning("subscriber queue full; dropping a live event")


__all__ = ["CLOSES_STREAM", "TERMINAL", "RunBus"]
