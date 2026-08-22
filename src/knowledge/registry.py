"""Reading the clause registry back out of SQLite.

Both retrievers are fed from here, which is deliberate: the BM25 index and the
Chroma collection must be built from the same rows, or a clause can be lexically
findable and semantically invisible (or worse, the reverse). One loader means
one definition of what is in the corpus.

Reads are unscoped. This is the *source* of the corpus, not a user-facing query
path - the ACL is applied by the retrievers, which is the layer that knows who
is asking. Callers of this module are build scripts and startup code.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from src.knowledge.vectorstore.base import Chunk

_QUERY: Final = """
    SELECT
        cl.clause_id, cl.doc_id, cl.doc_title, cl.clause_ref, cl.title,
        cl.tier, cl.account_id, cl.status,
        cl.text,
        (
            SELECT group_concat(t.topic, '|')
            FROM (
                SELECT topic FROM clause_topics
                WHERE clause_id = cl.clause_id
                ORDER BY topic
            ) AS t
        ) AS topics
    FROM clauses cl
    ORDER BY cl.tier, cl.clause_id
"""


class RegistryError(RuntimeError):
    """The registry is missing or empty."""


def load_chunks(db_path: Path | str) -> tuple[Chunk, ...]:
    """Every clause in the registry, ordered by tier then id.

    Deterministic ordering matters more than it looks: it is what makes the
    BM25 index and the committed collection reproducible run to run.
    """
    path = Path(db_path)
    if not path.exists():
        raise RegistryError(f"no database at {path}; run scripts/build_db.py first")

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(_QUERY).fetchall()
    except sqlite3.OperationalError as exc:
        raise RegistryError(f"registry tables missing from {path}: {exc}") from exc
    finally:
        connection.close()

    if not rows:
        raise RegistryError(f"no clauses in {path}; run scripts/build_db.py")

    return tuple(_to_chunk(row) for row in rows)


def _to_chunk(row: sqlite3.Row) -> Chunk:
    topics = row["topics"] or ""
    return Chunk(
        clause_id=row["clause_id"],
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        clause_ref=row["clause_ref"],
        title=row["title"],
        tier=int(row["tier"]),
        account_id=row["account_id"],
        status=row["status"],
        topics=tuple(topic for topic in topics.split("|") if topic),
        text=row["text"],
    )


def group_by_topic(chunks: Sequence[Chunk]) -> dict[str, tuple[Chunk, ...]]:
    """Clauses keyed by topic. A clause with several topics appears under each."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        for topic in chunk.topics:
            grouped.setdefault(topic, []).append(chunk)
    return {topic: tuple(items) for topic, items in sorted(grouped.items())}
