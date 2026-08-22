"""Verify every external dependency before trusting a run.

    uv run python scripts/preflight.py

Exits non-zero if anything is dead, so it can gate CI and a deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, get_settings
from src.providers.preflight import run_preflight


def main() -> int:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2

    print(
        f"provider: chat={settings.llm_provider} embeddings={settings.embedding_provider} "
        f"store={settings.vector_store}"
    )
    report = run_preflight(settings)
    print(report.text)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
