"""The clock is the one piece of infrastructure that can make every answer
silently wrong. AS_OF is a Sunday, and most SLA targets in the pack are
expressed in business hours, so the weekend arithmetic below is not an edge
case — it is the common path (docs/01_DATA_PACK_FINDINGS.md §2).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.clock import (
    BUSINESS_END,
    BUSINESS_START,
    BUSINESS_TZ,
    ClockError,
    add_business_days,
    add_business_hours,
    business_hours_between,
    is_business_time,
    next_business_start,
    parse_snapshot,
)

IST = ZoneInfo("Asia/Kolkata")


def ist(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=IST)


# 2026-08-16 is a Sunday; 08-17 Monday, 08-14 Friday, 08-15 Saturday.
SNAPSHOT = ist(2026, 8, 16, 11, 0)


class TestParseSnapshot:
    def test_parses_the_workbook_readme_format(self):
        assert parse_snapshot("2026-08-16 11:00 Asia/Kolkata") == SNAPSHOT

    def test_parses_iso_with_offset(self):
        assert parse_snapshot("2026-08-16T11:00:00+05:30") == SNAPSHOT

    def test_the_snapshot_is_a_sunday(self):
        # The single most consequential fact about this dataset.
        assert parse_snapshot("2026-08-16 11:00 Asia/Kolkata").weekday() == 6

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_is_an_error_not_a_wall_clock_fallback(self, raw):
        with pytest.raises(ClockError):
            parse_snapshot(raw)

    def test_naive_datetime_string_is_rejected(self):
        # A naive AS_OF would be interpreted differently on every host.
        with pytest.raises(ClockError):
            parse_snapshot("2026-08-16 11:00")

    def test_unparseable_is_an_error(self):
        with pytest.raises(ClockError):
            parse_snapshot("last Tuesday")

    def test_unknown_timezone_is_an_error(self):
        with pytest.raises(ClockError):
            parse_snapshot("2026-08-16 11:00 Mars/Olympus")


class TestBusinessWindow:
    def test_business_window_is_mon_to_fri_nine_to_six(self):
        assert BUSINESS_START.hour == 9
        assert BUSINESS_END.hour == 18
        assert BUSINESS_TZ.key == "Asia/Kolkata"

    @pytest.mark.parametrize(
        "moment,expected",
        [
            (ist(2026, 8, 17, 9, 0), True),  # Monday opening
            (ist(2026, 8, 17, 13, 0), True),  # Monday midday
            (ist(2026, 8, 17, 17, 59), True),  # Monday just before close
            (ist(2026, 8, 17, 18, 0), False),  # closing instant is outside
            (ist(2026, 8, 17, 8, 59), False),  # before opening
            (ist(2026, 8, 15, 11, 0), False),  # Saturday
            (ist(2026, 8, 16, 11, 0), False),  # Sunday — the snapshot itself
        ],
    )
    def test_is_business_time(self, moment, expected):
        assert is_business_time(moment) is expected

    def test_naive_input_is_rejected_everywhere(self):
        naive = datetime(2026, 8, 17, 10, 0)
        for fn in (is_business_time, next_business_start):
            with pytest.raises(ClockError):
                fn(naive)


class TestNextBusinessStart:
    def test_inside_hours_returns_the_moment_itself(self):
        moment = ist(2026, 8, 17, 10, 30)
        assert next_business_start(moment) == moment

    def test_the_snapshot_sunday_rolls_to_monday_opening(self):
        # LumenWorks TKT-502 was raised at Sunday 09:45; its business-hours
        # clock does not start until Monday morning.
        assert next_business_start(SNAPSHOT) == ist(2026, 8, 17, 9, 0)

    def test_before_opening_rolls_to_the_same_day(self):
        assert next_business_start(ist(2026, 8, 17, 6, 0)) == ist(2026, 8, 17, 9, 0)

    def test_after_close_rolls_to_the_next_day(self):
        assert next_business_start(ist(2026, 8, 17, 19, 0)) == ist(2026, 8, 18, 9, 0)

    def test_friday_evening_rolls_across_the_weekend(self):
        assert next_business_start(ist(2026, 8, 14, 19, 0)) == ist(2026, 8, 17, 9, 0)

    def test_converts_a_foreign_timezone_before_deciding(self):
        utc_sunday = datetime(2026, 8, 16, 5, 30, tzinfo=ZoneInfo("UTC"))  # 11:00 IST
        assert next_business_start(utc_sunday) == ist(2026, 8, 17, 9, 0)


class TestBusinessHoursBetween:
    def test_a_full_working_day_is_nine_hours(self):
        assert business_hours_between(ist(2026, 8, 17, 9, 0), ist(2026, 8, 17, 18, 0)) == 9.0

    def test_a_whole_weekend_counts_zero(self):
        assert business_hours_between(ist(2026, 8, 15), ist(2026, 8, 17)) == 0.0

    def test_spans_a_weekend_correctly(self):
        # Friday 17:00 -> Monday 10:00 is 1h Friday + 1h Monday.
        assert business_hours_between(ist(2026, 8, 14, 17, 0), ist(2026, 8, 17, 10, 0)) == 2.0

    def test_clips_to_the_working_window(self):
        # 06:00 -> 20:00 on a Monday is the full 9-hour window, not 14 hours.
        assert business_hours_between(ist(2026, 8, 17, 6, 0), ist(2026, 8, 17, 20, 0)) == 9.0

    def test_equal_instants_are_zero(self):
        assert business_hours_between(SNAPSHOT, SNAPSHOT) == 0.0

    def test_reversed_interval_is_an_error(self):
        with pytest.raises(ClockError):
            business_hours_between(ist(2026, 8, 18), ist(2026, 8, 17))

    def test_partial_hours_are_preserved(self):
        assert business_hours_between(
            ist(2026, 8, 17, 9, 0), ist(2026, 8, 17, 9, 30)
        ) == pytest.approx(0.5)


class TestAddBusinessHours:
    def test_zero_hours_normalises_to_the_next_opening(self):
        assert add_business_hours(SNAPSHOT, 0) == ist(2026, 8, 17, 9, 0)

    def test_from_the_sunday_snapshot(self):
        # LumenWorks P2 is 4 business hours (agreement §1) and TKT-502 was
        # raised on a Sunday, so the target lands Monday lunchtime.
        assert add_business_hours(SNAPSHOT, 4) == ist(2026, 8, 17, 13, 0)

    def test_within_a_single_day(self):
        assert add_business_hours(ist(2026, 8, 17, 10, 0), 3) == ist(2026, 8, 17, 13, 0)

    def test_rolls_over_the_close_of_business(self):
        # 17:00 + 2h = 1h Monday + 1h Tuesday.
        assert add_business_hours(ist(2026, 8, 17, 17, 0), 2) == ist(2026, 8, 18, 10, 0)

    def test_rolls_across_the_weekend(self):
        assert add_business_hours(ist(2026, 8, 14, 17, 0), 2) == ist(2026, 8, 17, 10, 0)

    def test_exactly_one_working_day_lands_on_the_next_opening(self):
        # Reaching the close instant is normalised forward, never left at 18:00.
        assert add_business_hours(ist(2026, 8, 17, 9, 0), 9) == ist(2026, 8, 18, 9, 0)

    def test_fractional_hours(self):
        assert add_business_hours(ist(2026, 8, 17, 9, 0), 0.5) == ist(2026, 8, 17, 9, 30)

    def test_negative_is_an_error(self):
        with pytest.raises(ClockError):
            add_business_hours(SNAPSHOT, -1)

    def test_is_the_inverse_of_business_hours_between(self):
        start = ist(2026, 8, 14, 15, 30)
        for hours in (0.25, 1, 3.75, 9, 20):
            assert business_hours_between(
                next_business_start(start), add_business_hours(start, hours)
            ) == pytest.approx(hours)


class TestAddBusinessDays:
    def test_from_the_sunday_snapshot(self):
        # Beacon Retail TKT-503 is P3 = 2 business days, raised Sunday 10:05.
        assert add_business_days(SNAPSHOT, 2) == ist(2026, 8, 19, 9, 0)

    def test_preserves_time_of_day_inside_hours(self):
        assert add_business_days(ist(2026, 8, 17, 14, 0), 1) == ist(2026, 8, 18, 14, 0)

    def test_skips_the_weekend(self):
        assert add_business_days(ist(2026, 8, 14, 14, 0), 1) == ist(2026, 8, 17, 14, 0)

    def test_zero_days_normalises_only(self):
        assert add_business_days(SNAPSHOT, 0) == ist(2026, 8, 17, 9, 0)

    def test_negative_is_an_error(self):
        with pytest.raises(ClockError):
            add_business_days(SNAPSHOT, -1)


class TestAsOf:
    def test_reads_the_configured_snapshot(self, as_of_configured):
        from src.clock import as_of

        assert as_of() == SNAPSHOT

    def test_raises_when_unconfigured_rather_than_falling_back(self, as_of_unset):
        from src.clock import as_of

        with pytest.raises(ClockError, match="AS_OF"):
            as_of()

    def test_elapsed_since_is_measured_against_as_of(self, as_of_configured):
        from src.clock import elapsed_since

        created = SNAPSHOT - timedelta(minutes=30)
        assert elapsed_since(created) == timedelta(minutes=30)
