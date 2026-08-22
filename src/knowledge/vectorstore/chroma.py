"""Chroma-backed dense retrieval, local and hosted (D20).

Local and Cloud differ by one line - which client object is constructed - so
they share an implementation rather than duplicating the query logic twice and
letting the two copies drift. The distinction that matters is operational:
tests and the eval harness must never need a network, and the demo should run
against the same code path the hosted app uses.

Access control is applied as a Chroma `where` predicate, evaluated before
ranking rather than after (D3). Post-filtering a top-k is both a leak risk and
quietly lossy: drop three of eight results and the caller gets five, with no
signal that the good ones were removed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from src.auth.principal import Principal
from src.config import Settings
from src.knowledge.vectorstore.base import (
    CITABLE_TIERS,
    DEFAULT_K,
    GLOBAL_SCOPE,
    Chunk,
    visible_scopes,
)
from src.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

#: Chroma metadata values must be scalars, so the topic list is stored as a
#: delimited string. The delimiter cannot occur in a Topic value.
_TOPIC_SEPARATOR: Final = "|"


class VectorStoreError(RuntimeError):
    """The collection could not be reached or written."""


class ChromaStore:
    """Shared behaviour. Instantiate `ChromaLocalStore` or `ChromaCloudStore`."""

    def __init__(self, *, embeddings: EmbeddingProvider, client: Any) -> None:
        self._embeddings = embeddings
        self._client = client
        self.identity = embeddings.identity
        self.collection_name = collection_name_for(embeddings.identity)

    # -- collection ---------------------------------------------------------

    @property
    def _collection(self) -> Any:
        # Fetched per use rather than cached: `upsert` deletes and recreates,
        # and a cached handle to a deleted collection fails obscurely.
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_identity": self.identity},
        )

    def count(self) -> int:
        return int(self._collection.count())

    # -- writing ------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Replace the collection wholesale. Returns the number stored.

        Wholesale, not incremental, because the registry is rebuilt from the
        PDFs in one shot. An incremental upsert would leave a vector behind
        for a clause that no longer exists, and a stale vector is worse than a
        missing one: it is citable, and it points at nothing.
        """
        if not chunks:
            raise VectorStoreError("refusing to index an empty corpus")

        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # absent is the normal first-run case
            logger.debug("no existing collection %s to drop", self.collection_name)

        vectors = self._embeddings.embed_documents([chunk.searchable_text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise VectorStoreError(
                f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        self._collection.add(
            ids=[chunk.clause_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_to_metadata(chunk) for chunk in chunks],
        )
        logger.info("indexed %d clauses into %s", len(chunks), self.collection_name)
        return len(chunks)

    # -- reading ------------------------------------------------------------

    def query(
        self,
        text: str,
        *,
        principal: Principal,
        tiers: frozenset[int] | set[int] = CITABLE_TIERS,
        topics: Sequence[str] | None = None,
        k: int = DEFAULT_K,
    ) -> list[Chunk]:
        """Nearest citable clauses this principal may see.

        There is deliberately no way to pass a filter. The predicate is built
        here from the Principal, so an unauthorised read is unavailable rather
        than merely discouraged.
        """
        if self.count() == 0:
            return []

        where = _acl_predicate(principal, tiers, topics)
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(text)],
            n_results=min(k, self.count()),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        return _to_chunks(result)


class ChromaLocalStore(ChromaStore):
    """On-disk Chroma. Used by tests, the eval harness and offline work."""

    def __init__(self, *, embeddings: EmbeddingProvider, persist_dir: Path | str) -> None:
        import chromadb

        path = Path(persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        super().__init__(embeddings=embeddings, client=chromadb.PersistentClient(path=str(path)))


class ChromaCloudStore(ChromaStore):
    """Hosted Chroma. Used in dev and prod."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        api_key: str,
        tenant: str,
        database: str,
    ) -> None:
        import chromadb

        if not api_key:
            raise VectorStoreError("CHROMA_API_KEY is required for the hosted store")
        super().__init__(
            embeddings=embeddings,
            client=chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key),
        )


def build_vector_store(settings: Settings, embeddings: EmbeddingProvider) -> ChromaStore:
    """Select the store the configuration asks for."""
    if settings.vector_store == "chroma_cloud":
        return ChromaCloudStore(
            embeddings=embeddings,
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
    return ChromaLocalStore(embeddings=embeddings, persist_dir=settings.chroma_dir)


# -- helpers ----------------------------------------------------------------


def collection_name_for(identity: str) -> str:
    """`provider/model/dim` to a legal Chroma collection name.

    Chroma accepts 3-512 characters of `[a-zA-Z0-9._-]`, starting and ending
    alphanumeric.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", identity).strip("_-")
    return f"pp_clauses_{slug}"


def _acl_predicate(
    principal: Principal,
    tiers: frozenset[int] | set[int],
    topics: Sequence[str] | None,
) -> dict[str, Any]:
    if not tiers:
        raise ValueError("refusing to query with an empty tier set")

    clauses: list[dict[str, Any]] = [{"tier": {"$in": sorted(tiers)}}]

    scopes = visible_scopes(principal)
    if scopes is not None:
        clauses.append({"account_scope": {"$in": scopes}})

    if topics:
        # Stored as one boolean key per topic, because Chroma cannot filter
        # inside a delimited string and cannot hold a list.
        clauses.append(_any_of([{_topic_key(topic): True} for topic in topics]))

    return _all_of(clauses)


def _all_of(predicates: list[dict[str, Any]]) -> dict[str, Any]:
    """Chroma rejects `$and`/`$or` with fewer than two operands, so single
    predicates are passed through bare rather than wrapped."""
    return predicates[0] if len(predicates) == 1 else {"$and": predicates}


def _any_of(predicates: list[dict[str, Any]]) -> dict[str, Any]:
    return predicates[0] if len(predicates) == 1 else {"$or": predicates}


def _topic_key(topic: str) -> str:
    return f"topic_{topic}"


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "clause_id": chunk.clause_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "clause_ref": chunk.clause_ref,
        "title": chunk.title,
        "tier": chunk.tier,
        "account_scope": chunk.account_id or GLOBAL_SCOPE,
        "status": chunk.status,
        "topics": _TOPIC_SEPARATOR.join(chunk.topics),
    }
    metadata.update({_topic_key(topic): True for topic in chunk.topics})
    return metadata


def _to_chunks(result: dict[str, Any]) -> list[Chunk]:
    metadatas = (result.get("metadatas") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks = []
    for metadata, document, distance in zip(metadatas, documents, distances, strict=True):
        scope = metadata.get("account_scope", GLOBAL_SCOPE)
        topics = metadata.get("topics") or ""
        chunks.append(
            Chunk(
                clause_id=metadata["clause_id"],
                doc_id=metadata["doc_id"],
                doc_title=metadata["doc_title"],
                clause_ref=metadata["clause_ref"],
                title=metadata["title"],
                tier=int(metadata["tier"]),
                account_id=None if scope == GLOBAL_SCOPE else scope,
                status=metadata["status"],
                topics=tuple(t for t in topics.split(_TOPIC_SEPARATOR) if t),
                text=document,
                # Chroma returns a distance; callers rank on score, where
                # higher is better regardless of which retriever produced it.
                score=1.0 / (1.0 + float(distance)),
            )
        )
    return chunks
