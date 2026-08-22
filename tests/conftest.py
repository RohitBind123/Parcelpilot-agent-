"""Shared fixtures.

Every fixture that touches configuration clears the settings cache on the way
in and on the way out, so a test that changes the environment cannot leak into
the next one.
"""

from __future__ import annotations

import pytest

SNAPSHOT_RAW = "2026-08-16 11:00 Asia/Kolkata"


def _reset_caches() -> None:
    from src import clock, config

    config.get_settings.cache_clear()
    clock.as_of.cache_clear()


@pytest.fixture
def as_of_configured(monkeypatch: pytest.MonkeyPatch):
    """AS_OF set to the workbook snapshot."""
    monkeypatch.setenv("AS_OF", SNAPSHOT_RAW)
    _reset_caches()
    yield SNAPSHOT_RAW
    _reset_caches()


@pytest.fixture
def as_of_unset(monkeypatch: pytest.MonkeyPatch):
    """AS_OF absent. There is deliberately no wall-clock fallback (D6)."""
    monkeypatch.setenv("AS_OF", "")
    _reset_caches()
    yield
    _reset_caches()
