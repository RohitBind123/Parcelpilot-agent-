"""A frozen clock is only frozen if nothing bypasses it.

The failure this prevents is not hypothetical: `AS_OF` is 2026-08-16, so any
module that reaches for the real wall clock would compute cancellation windows
and SLA targets against a date years away from the dataset and be confidently,
silently wrong (docs/ARCHITECTURE.md D6).

`src/clock.py` is the one sanctioned exception. It defines `as_of()` for domain
time and `wall_now()` for infrastructure time, so every real use of the wall
clock is greppable by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src", "app")
SANCTIONED = {REPO_ROOT / "src" / "clock.py"}

BANNED = {
    "datetime.now(": re.compile(r"\bdatetime\.now\s*\("),
    "datetime.utcnow(": re.compile(r"\bdatetime\.utcnow\s*\("),
    "date.today(": re.compile(r"\bdate\.today\s*\("),
    "time.time(": re.compile(r"\btime\.time\s*\("),
}


def _python_files() -> list[Path]:
    return sorted(
        path
        for root in SCANNED_ROOTS
        for path in (REPO_ROOT / root).rglob("*.py")
        if path not in SANCTIONED and "__pycache__" not in path.parts
    )


def test_there_are_files_to_scan():
    # Guards against the scan silently passing because it found nothing.
    assert _python_files(), "time-discipline scan found no source files"


@pytest.mark.parametrize("label,pattern", sorted(BANNED.items()))
def test_wall_clock_is_not_read_outside_the_clock_module(label: str, pattern: re.Pattern[str]):
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in _python_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line) and "noqa: wall-clock" not in line
    ]
    assert not offenders, (
        f"{label} is banned outside src/clock.py — use as_of() for domain time or "
        f"wall_now() for infrastructure time. Found at: {', '.join(offenders)}"
    )


def test_the_sanctioned_module_actually_exists():
    for path in SANCTIONED:
        assert path.is_file(), f"sanctioned exception {path} is missing"


def test_wall_now_is_timezone_aware_and_distinct_from_as_of(as_of_configured):
    from src.clock import as_of, wall_now

    now = wall_now()
    assert now.tzinfo is not None
    # Not an equality check on purpose: the point is that they are different
    # clocks, and the domain one does not drift.
    assert as_of() != now
