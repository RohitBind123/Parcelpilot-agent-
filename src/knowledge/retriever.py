"""Hybrid retrieval over the clause registry (D4).

Two retrievers, because the questions are not all one shape. "Will I be charged
for cancelling?" is semantic. "What does SOP v4 §1 say?" is a lookup, and an
exact reference is close to the worst case for dense retrieval - "§1" has
almost no meaning to embed. BM25 covers that, and at 19 clauses it is free:
the index is rebuilt in memory from SQLite at startup, so there is nothing to
provision and nothing that can drift from the registry.

The two are combined with reciprocal rank fusion rather than a weighted score,
because a cosine similarity and a BM25 score are not the same unit. Fusing the
numbers would hand the answer to whichever retriever emits larger ones.

Access control is applied on both paths. This is the whole risk of adding a
second retriever to a store whose ACL you already trust: forget the predicate
on the new path and the leak is invisible, because the path you were watching
is still correct.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Final

from src.auth.principal import Principal
from src.knowledge.vectorstore.base import (
    CITABLE_TIERS,
    DEFAULT_K,
    Chunk,
    VectorStore,
    visible_scopes,
)

logger = logging.getLogger(__name__)

#: The constant from the original RRF paper. Large relative to the ranks in
#: play, which is what stops a single first place from dominating agreement.
RRF_K: Final = 60

#: Widen each retriever before fusing. Fusing two top-8 lists and then cutting
#: to 8 throws away the agreement signal at the boundary.
_CANDIDATE_MULTIPLIER: Final = 3

_TOKEN: Final = re.compile(r"[a-z0-9_]+")

#: A section reference in either spelling: "§2" or "Section 2".
_SECTION: Final = re.compile(r"(?:§\s*|\bsection\s+)([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)

#: The thousands separator inside a number. "INR 1,000" must not become the
#: two tokens "1" and "000" - the stray "1" then matches a query for "§1".
_THOUSANDS: Final = re.compile(r"(?<=\d),(?=\d)")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with section references kept atomic.

    Two things here are not incidental.

    A bare `§` is dropped by any plain tokenizer, which would make "SOP v4 §1"
    and "SOP v4" the same query. Splitting it into `section` + `1` is no better:
    every clause has a section, so `section` matches everything and the answer
    turns on the digit alone. `§1` therefore becomes the single token
    `section_1`, which matches the clause it names and nothing else. "Section
    1" written out in prose normalises to the same token.

    Thousands separators are stripped first. Without that, "INR 1,000" yields
    the tokens `1` and `000`, and the stray `1` is indistinguishable from a
    reference to §1 - which is exactly how "SOP v4 §1" came back pointing at
    §3, the shortest clause in the document and the only other one containing
    a bare 1.
    """
    text = _THOUSANDS.sub("", text)
    text = _SECTION.sub(lambda match: f" section_{match.group(1)} ", text)
    return _TOKEN.findall(text.replace("§", " ").lower())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Chunk]],
    *,
    k: int = RRF_K,
) -> list[Chunk]:
    """Fuse ranked lists by rank, not by score.

    score(d) = sum over lists of 1 / (k + rank(d)), rank counted from 1.

    Ties break on clause_id so the order is identical across runs. A golden set
    that flaps because two equal scores swapped is a test that fails for
    reasons unrelated to the answer.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.clause_id] = scores.get(chunk.clause_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(chunk.clause_id, chunk)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [seen[clause_id].with_score(score) for clause_id, score in ordered]


class BM25Index:
    """Lexical search, built in memory from the clause registry.

    The indexed document is title + reference + body, not body alone, because
    people cite clauses by their reference and search for them by their name.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = tuple(chunks)
        self._bm25 = (
            BM25Okapi([tokenize(c.searchable_text) for c in self._chunks]) if chunks else None
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(
        self,
        text: str,
        *,
        principal: Principal,
        tiers: frozenset[int] | set[int] = CITABLE_TIERS,
        topics: Sequence[str] | None = None,
        k: int = DEFAULT_K,
    ) -> list[Chunk]:
        """Best lexical matches this principal may see.

        Filtered before ranking, like the dense path, so k means k rather than
        "k minus whatever was removed afterwards".
        """
        if self._bm25 is None:
            return []

        scopes = visible_scopes(principal)
        wanted = set(topics or ())

        allowed = [
            index
            for index, chunk in enumerate(self._chunks)
            if chunk.tier in tiers
            and (scopes is None or (chunk.account_id or "*") in scopes)
            and (not wanted or wanted & set(chunk.topics))
        ]
        if not allowed:
            return []

        scores = self._bm25.get_scores(tokenize(text))
        # A zero score means no query term occurs in the document. Returning it
        # anyway would pad the list with noise that later looks like evidence.
        hits = [(scores[i], self._chunks[i]) for i in allowed if scores[i] > 0.0]
        hits.sort(key=lambda pair: (-pair[0], pair[1].clause_id))
        return [chunk.with_score(float(score)) for score, chunk in hits[:k]]


class HybridRetriever:
    """Dense plus lexical, fused by rank."""

    def __init__(
        self,
        *,
        dense: VectorStore,
        lexical: BM25Index,
        rrf_k: int = RRF_K,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k

    def retrieve(
        self,
        text: str,
        *,
        principal: Principal,
        tiers: frozenset[int] | set[int] = CITABLE_TIERS,
        topics: Sequence[str] | None = None,
        k: int = DEFAULT_K,
    ) -> list[Chunk]:
        candidates = max(k * _CANDIDATE_MULTIPLIER, k)
        rankings = [
            self._safe_dense(text, principal=principal, tiers=tiers, topics=topics, k=candidates),
            self._lexical.search(
                text, principal=principal, tiers=tiers, topics=topics, k=candidates
            ),
        ]
        return reciprocal_rank_fusion(rankings, k=self._rrf_k)[:k]

    def _safe_dense(self, text: str, **kwargs) -> list[Chunk]:
        """Dense retrieval, degraded rather than fatal.

        An unbuilt collection or an unreachable host should cost recall, not
        the whole answer: lexical search alone still finds the clause for most
        of this corpus.
        """
        try:
            return self._dense.query(text, **kwargs)
        except ValueError:
            # A malformed Principal is a bug, not a transient fault. Never
            # swallowed - degrading here would turn a refused query into a
            # silently unfiltered one.
            raise
        except Exception:
            logger.warning("dense retrieval unavailable; falling back to lexical", exc_info=True)
            return []


def build_lexical_index(chunks: Iterable[Chunk]) -> BM25Index:
    return BM25Index(tuple(chunks))
