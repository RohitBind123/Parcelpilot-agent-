"""A deterministic embedding provider, so retrieval is testable offline (D20).

Hashed bag-of-words. Every token is hashed into a bucket and the vector is L2
normalised, which makes cosine similarity a real function of shared vocabulary:
a query mentioning "cancellation fee" genuinely lands nearer a clause that uses
those words than one about bulk uploads.

It shares the retriever's tokenizer rather than rolling its own, so the stub
cannot exhibit a lexical flaw the real pipeline has already fixed - a test
double that is wrong in its own private way tests nothing useful.

That is weaker than a trained embedding and deliberately so. What these tests
assert is the plumbing - that the ACL predicate holds, that fusion prefers what
both retrievers agree on, that a switch of embedding identity selects a
different collection. None of that should depend on a network call, and none of
it should silently start passing because a hosted model got better.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Final

from src.knowledge.retriever import tokenize

DIMENSIONS: Final = 256


def _bucket(token: str, dimensions: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions


class HashingEmbeddings:
    """Satisfies `EmbeddingProvider` without leaving the process."""

    def __init__(self, *, dimensions: int = DIMENSIONS, tag: str = "hashing") -> None:
        self.dimensions = dimensions
        self.identity = f"test/{tag}/{dimensions}"
        #: Counts calls so a test can assert the cache or a batch actually saved work.
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            vector[_bucket(token, self.dimensions)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An all-zero vector has no direction, so cosine distance to it is
            # undefined. One fixed non-zero bucket keeps the space well formed.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._embed(text)
