"""The only time source in the system.

Two rules, both load-bearing (docs/ARCHITECTURE.md D6, D22):

1. **There is no "now".** Every temporal answer is measured against `AS_OF`,
   the dataset snapshot declared in the workbook README. `datetime.now()`,
   `date.today()` and `time.time()` are banned across `src/` and a test
   enforces it. Without this, every cancellation-window and SLA answer
   becomes quietly wrong the day after it was checked.

2. **Business hours are an assumption, not a fact.** The pack expresses
   Growth and Standard targets in "business hours" and "business days" and
   never defines either, while `AS_OF` itself falls on a Sunday. The window
   below is our stated assumption (A1) and any answer that relies on it has
   to say so.

Infrastructure still needs real elapsed time - message timestamps, token
expiry, SSE ordering - so `wall_now()` is the single sanctioned wall-clock
call site. Naming it makes every real use greppable, which is what lets the
ban on `datetime.now()` be mechanical.

This module imports nothing from the rest of the application.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Assumption A1. Stated in every answer that depends on it.
BUSINESS_TZ: Final = ZoneInfo("Asia/Kolkata")
BUSINESS_START: Final = time(9, 0)
BUSINESS_END: Final = time(18, 0)
BUSINESS_WEEKDAYS: Final[frozenset[int]] = frozenset({0, 1, 2, 3, 4})  # Mon-Fri

BUSINESS_ASSUMPTION: Final = (
    "Business hours are assumed to be Monday-Friday 09:00-18:00 Asia/Kolkata; "
    "public holidays are not modelled."
)

#: Loop guards. Real inputs need a handful of iterations; these exist so a
#: malformed window can never hang a request.
_MAX_DAY_ROLLS: Final = 8
_MAX_HOUR_SPANS: Final = 5_000


class ClockError(RuntimeError):
    """A time value is missing, ambiguous, or impossible.

    Deliberately not recoverable. Every alternative to raising here is a
    silent wrong answer.
    """


def parse_snapshot(raw: str | None) -> datetime:
    """Parse a configured snapshot instant into a timezone-aware datetime.

    Accepts the workbook README's own format, `2026-08-16 11:00 Asia/Kolkata`,
    and ISO 8601 with an offset. A naive value is rejected rather than assumed
    to be local: it would resolve differently on every host.
    """
    if raw is None or not raw.strip():
        raise ClockError(
            "no snapshot time configured; there is deliberately no wall-clock fallback"
        )

    text = raw.strip()

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return _parse_with_zone_suffix(text)

    if parsed.tzinfo is None:
        raise ClockError(f"snapshot time {text!r} has no timezone; a naive value is ambiguous")
    return parsed


def _parse_with_zone_suffix(text: str) -> datetime:
    """Parse `<naive datetime> <IANA zone>`, the workbook's own format."""
    head, _, zone_name = text.rpartition(" ")
    if not head:
        raise ClockError(f"cannot parse snapshot time {text!r}")

    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ClockError(f"unknown timezone {zone_name!r} in snapshot time {text!r}") from exc

    try:
        naive = datetime.fromisoformat(head)
    except ValueError as exc:
        raise ClockError(f"cannot parse snapshot time {text!r}") from exc

    if naive.tzinfo is not None:
        raise ClockError(f"snapshot time {text!r} declares a timezone twice")
    return naive.replace(tzinfo=zone)


@lru_cache(maxsize=1)
def as_of() -> datetime:
    """The frozen snapshot instant. The only "now" this system has."""
    from src.config import get_settings

    try:
        return parse_snapshot(get_settings().as_of)
    except ClockError as exc:
        raise ClockError(f"AS_OF is not usable: {exc}") from exc


def elapsed_since(moment: datetime) -> timedelta:
    """How long ago `moment` was, measured against `AS_OF`."""
    return as_of() - _ensure_aware(moment)


def is_business_time(moment: datetime) -> bool:
    """Whether the business clock is running at this instant."""
    local = _to_local(moment)
    return local.weekday() in BUSINESS_WEEKDAYS and BUSINESS_START <= local.time() < BUSINESS_END


def next_business_start(moment: datetime) -> datetime:
    """The first instant at or after `moment` when the business clock runs.

    Returns `moment` itself if the clock is already running. This is what
    makes "raised on Sunday" resolve to "clock starts Monday 09:00".
    """
    local = _to_local(moment)

    for _ in range(_MAX_DAY_ROLLS):
        if local.weekday() in BUSINESS_WEEKDAYS:
            if local.time() < BUSINESS_START:
                return _at(local, BUSINESS_START)
            if local.time() < BUSINESS_END:
                return local
        local = _at(local + timedelta(days=1), BUSINESS_START)

    raise ClockError(f"no business window found within {_MAX_DAY_ROLLS} days of {moment}")


def business_hours_between(start: datetime, end: datetime) -> float:
    """Business hours elapsed between two instants, clipped to the window."""
    lo = _to_local(start)
    hi = _to_local(end)
    if hi < lo:
        raise ClockError(f"interval runs backwards: {start} .. {end}")

    seconds = 0.0
    day = lo.date()
    while day <= hi.date():
        if day.weekday() in BUSINESS_WEEKDAYS:
            opens = datetime.combine(day, BUSINESS_START, tzinfo=BUSINESS_TZ)
            closes = datetime.combine(day, BUSINESS_END, tzinfo=BUSINESS_TZ)
            overlap_start = max(lo, opens)
            overlap_end = min(hi, closes)
            if overlap_end > overlap_start:
                seconds += (overlap_end - overlap_start).total_seconds()
        day += timedelta(days=1)

    return seconds / 3600.0


def add_business_hours(start: datetime, hours: float) -> datetime:
    """Advance `start` by `hours` of business time.

    `start` is first normalised to the next business opening, so a target set
    on a Sunday begins accruing on Monday morning rather than immediately.
    Landing exactly on the close of business rolls forward to the next
    opening; a due time of 18:00 would be outside the window that produced it.
    """
    if hours < 0:
        raise ClockError(f"cannot add negative business hours: {hours}")

    cursor = next_business_start(start)
    remaining = float(hours)

    for _ in range(_MAX_HOUR_SPANS):
        if remaining <= 0:
            return cursor
        closes = datetime.combine(cursor.date(), BUSINESS_END, tzinfo=BUSINESS_TZ)
        available = (closes - cursor).total_seconds() / 3600.0
        if remaining < available:
            return cursor + timedelta(hours=remaining)
        remaining -= available
        cursor = next_business_start(closes)

    raise ClockError(f"could not resolve {hours} business hours from {start}")


def add_business_days(start: datetime, days: int) -> datetime:
    """Advance `start` by whole business days, preserving time of day."""
    if days < 0:
        raise ClockError(f"cannot add negative business days: {days}")

    cursor = next_business_start(start)
    day = cursor.date()
    for _ in range(days):
        day += timedelta(days=1)
        while day.weekday() not in BUSINESS_WEEKDAYS:
            day += timedelta(days=1)

    return datetime.combine(day, cursor.time(), tzinfo=BUSINESS_TZ)


def wall_now() -> datetime:
    """Real wall-clock time, in UTC. **Never** for domain reasoning.

    Domain time is `as_of()`. This exists only for infrastructure that must
    track real elapsed time regardless of the dataset snapshot: message and
    run timestamps, session-token expiry, log lines, SSE event ordering.

    It is defined here, and only here, so that the ban on `datetime.now()`
    across the codebase has exactly one sanctioned exception and every real
    use of the wall clock is greppable by name.
    """
    return datetime.now(tz=UTC)


def _ensure_aware(moment: datetime) -> datetime:
    if not isinstance(moment, datetime):
        raise ClockError(f"expected a datetime, got {type(moment).__name__}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ClockError(f"naive datetime {moment!r}; every instant must carry a timezone")
    return moment


def _to_local(moment: datetime) -> datetime:
    return _ensure_aware(moment).astimezone(BUSINESS_TZ)


def _at(moment: datetime, at: time) -> datetime:
    return datetime.combine(moment.date(), at, tzinfo=BUSINESS_TZ)
