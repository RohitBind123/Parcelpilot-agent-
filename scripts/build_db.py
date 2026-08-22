"""Rebuild the structured store from the supplied workbook.

    uv run python scripts/build_db.py
    uv run python scripts/build_db.py --target /tmp/scratch.db

The resulting database is committed, so the hosted app parses nothing at
startup and any reviewer gets byte-identical data. The build is idempotent:
running it twice produces the same rows.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datastore.etl import WORKBOOK_PATH, EtlError, build_database

TABLES = ("accounts", "orders", "tickets")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK_PATH)
    parser.add_argument("--target", type=Path, default=None, help="defaults to SQLITE_PATH")
    args = parser.parse_args()

    try:
        path = build_database(args.workbook, args.target)
    except EtlError as exc:
        print(f"build failed: {exc}")
        return 1

    conn = sqlite3.connect(path)
    try:
        counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
        as_of = conn.execute("SELECT value FROM meta WHERE key='as_of'").fetchone()[0]
    finally:
        conn.close()

    print(f"built {path}")
    print("  " + "  ".join(f"{t}={n}" for t, n in counts.items()))
    print(f"  as_of={as_of}  (set AS_OF in .env to this value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
