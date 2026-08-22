"""Query-embedding cache.

Embedding the query sits on the hot path of every retrieval, so it is the one
place where a provider outage or a rate limit turns into a broken conversation
rather than a slow one. Embeddings are deterministic per (model, text), so
caching them is free correctness-wise and removes that dependency for any query
asked twice - which, in a demo, is most of them.

The cache lives in its own SQLite file rather than in `parcelpilot.db`, because
rebuilding the application database must not discard it, and because it is
derived data that should never be committed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path

from src.clock import wall_now
from src.providers.base import Vector

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    identity   TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (identity, text_hash)
) WITHOUT ROWID;
"""

DEFAULT_CACHE_PATH = Path("data/embedding_cache.db")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SqliteEmbeddingCache:
    """Content-addressed vector cache keyed by (embedding identity, text).

    Keying on identity as well as text matters: the same words under a
    different embedding model are a different vector, and sharing a key would
    poison every similarity score computed afterwards.
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, identity: str, text: str) -> Vector | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector FROM embedding_cache WHERE identity = ? AND text_hash = ?",
                (identity, _digest(text)),
            ).fetchone()
        if row is None:
            return None
        buffer = array("f")
        buffer.frombytes(row[0])
        return list(buffer)

    def get_many(self, identity: str, texts: Sequence[str]) -> dict[str, Vector]:
        """One round-trip for a whole batch, rather than one per text."""
        if not texts:
            return {}
        by_hash = {_digest(t): t for t in texts}
        placeholders = ",".join("?" * len(by_hash))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT text_hash, vector FROM embedding_cache "
                f"WHERE identity = ? AND text_hash IN ({placeholders})",
                (identity, *by_hash),
            ).fetchall()

        found: dict[str, Vector] = {}
        for text_hash, blob in rows:
            buffer = array("f")
            buffer.frombytes(blob)
            found[by_hash[text_hash]] = list(buffer)
        return found

    def put(self, identity: str, text: str, vector: Vector) -> None:
        self.put_many(identity, {text: vector})

    def put_many(self, identity: str, vectors: dict[str, Vector]) -> None:
        if not vectors:
            return
        stamp = wall_now().isoformat()
        rows = [
            (identity, _digest(text), array("f", vector).tobytes(), stamp)
            for text, vector in vectors.items()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO embedding_cache (identity, text_hash, vector, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(identity, text_hash) DO UPDATE SET "
                "vector = excluded.vector, created_at = excluded.created_at",
                rows,
            )


class CachedEmbeddings:
    """Decorator that memoises an embedding provider through a cache."""

    def __init__(self, inner, cache: SqliteEmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self.identity: str = inner.identity
        self.dimensions: int = inner.dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        cached = self._cache.get_many(self.identity, texts)

        # Deduplicate before calling out: a batch that repeats a text should
        # pay for it once. dict preserves insertion order.
        misses = list(dict.fromkeys(t for t in texts if t not in cached))
        if misses:
            fresh = dict(zip(misses, self._inner.embed_documents(misses), strict=True))
            self._cache.put_many(self.identity, fresh)
            cached |= fresh

        return [cached[t] for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
