"""Build-time ingest: PDFs to the clause registry.

Runs once, offline, as part of `scripts/build_db.py`. The resulting database is
committed (D10), so the hosted app parses no PDFs at startup and any reviewer
gets byte-identical clauses.

The registry is written into the same SQLite file as the workbook data, because
the resolver needs both in one query: "the governing clause for this topic,
visible to the account that owns this order".
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Final

from src.knowledge.clause_parser import Clause, parse_all
from src.knowledge.params import extract_params, score_confidence

logger = logging.getLogger(__name__)

_CLAUSE_COLUMNS: Final = (
    "clause_id",
    "doc_id",
    "doc_title",
    "clause_ref",
    "title",
    "tier",
    "account_id",
    "status",
    "effective_from",
    "effective_to",
    "superseded_by",
    "params",
    "text",
)


class IngestError(RuntimeError):
    """The corpus did not produce a usable registry."""


def build_registry(db_path: Path | str) -> int:
    """Parse every document and write the clause registry. Idempotent.

    Returns the number of clauses written.
    """
    documents = parse_all()
    clauses = [clause for document in documents for clause in document.clauses]
    if not clauses:
        raise IngestError("no clauses parsed from the corpus")

    flagged = _flagged(clauses)
    if flagged:
        # Not fatal: the reviewed baseline is the real gate, and a flag only
        # means "a human should look". Loud enough to notice in a build log.
        logger.warning("clauses flagged for parameter review: %s", flagged)

    conn = sqlite3.connect(Path(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # Rewritten wholesale rather than upserted: the registry is derived
        # data, and a partial refresh could leave a clause behind that no
        # document produces any more.
        conn.execute("DELETE FROM clause_topics")
        conn.execute("DELETE FROM clauses")
        conn.executemany(_insert_clause(), [_clause_row(c) for c in clauses])
        conn.executemany(
            "INSERT INTO clause_topics (clause_id, topic) VALUES (?, ?)",
            [(c.clause_id, topic) for c in clauses for topic in c.topics],
        )
        _assert_every_clause_is_reachable(conn, len(clauses))
        conn.commit()
    finally:
        conn.close()

    return len(clauses)


def _insert_clause() -> str:
    placeholders = ", ".join("?" * len(_CLAUSE_COLUMNS))
    return f"INSERT INTO clauses ({', '.join(_CLAUSE_COLUMNS)}) VALUES ({placeholders})"


def _clause_row(clause: Clause) -> tuple:
    return (
        clause.clause_id,
        clause.doc_id,
        clause.doc_title,
        clause.clause_ref,
        clause.title,
        clause.tier,
        clause.account_id,
        clause.status,
        clause.effective_from.isoformat() if clause.effective_from else None,
        clause.effective_to.isoformat() if clause.effective_to else None,
        clause.superseded_by,
        json.dumps(extract_params(clause), sort_keys=True),
        clause.text,
    )


def _flagged(clauses: list[Clause]) -> dict[str, tuple[str, ...]]:
    return {
        clause.clause_id: flag.reasons
        for clause in clauses
        if (flag := score_confidence(clause.clause_id, extract_params(clause))).score < 1.0
    }


def _assert_every_clause_is_reachable(conn: sqlite3.Connection, expected: int) -> None:
    """A clause with no topic is invisible to the resolver, which groups by topic."""
    written = conn.execute("SELECT count(*) FROM clauses").fetchone()[0]
    if written != expected:
        raise IngestError(f"wrote {written} clauses, expected {expected}")

    orphans = conn.execute(
        "SELECT clause_id FROM clauses WHERE clause_id NOT IN (SELECT clause_id FROM clause_topics)"
    ).fetchall()
    if orphans:
        raise IngestError(f"clauses with no topic are unreachable by the resolver: {orphans}")
