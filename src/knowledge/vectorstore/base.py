"""What a vector store is, and what it is not allowed to be.

The contract's single unusual property is that `query` takes a `Principal` and
does not take a filter. Access control at this layer cannot be a parameter,
because the corpus contains one customer's contract alongside another's: a
caller who forgets the predicate does not get a slow query, they get a leak.
Making the predicate unpassable means the mistake is unavailable.

The same reasoning puts tier here. A deprecated policy is indexed - "what
changed in v3?" is a fair question - but it is excluded by default, so
reaching it takes saying so.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol, runtime_checkable

from src.auth.principal import Principal

#: Chroma metadata cannot hold NULL, so an unscoped clause needs a value that
#: means "applies to every account". Never returned to a caller: `Chunk`
#: converts it back to None on the way out.
GLOBAL_SCOPE: Final = "*"

#: Tiers 0-3 are citable. Tier 4 is deprecated and tier 5 is a historical
#: ticket resolution; both may inform an answer but neither may support one.
CITABLE_TIERS: Final[frozenset[int]] = frozenset({0, 1, 2, 3})

DEFAULT_K: Final = 8

_ACCOUNT_ID: Final = re.compile(r"^ACCT-\d{3}$")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable clause, carrying everything a citation needs.

    The clause is the chunk. At a median of 44 words there is nothing to split,
    and splitting would only separate a rule from the numbers that qualify it.
    """

    clause_id: str
    doc_id: str
    doc_title: str
    clause_ref: str
    title: str
    tier: int
    account_id: str | None
    status: str
    topics: tuple[str, ...]
    text: str
    #: Higher is better, whatever the retriever. Comparable only within one
    #: result list: a cosine similarity and an RRF score are not the same unit.
    score: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.doc_title} {self.clause_ref}".strip()

    @property
    def searchable_text(self) -> str:
        """What both retrievers index: the clause in the context that names it.

        Body alone is not enough. "Cancellation within 30 minutes is free"
        carries no trace of which document it came from or which section it is,
        so a lookup like "SOP v4 §1" has nothing to match and a dense vector
        has no idea the clause belongs to the SOP at all.

        Defined once, here, because the lexical and dense indexes must be built
        from the same string. If they drift, a clause becomes findable one way
        and invisible the other, which shows up as an intermittently missing
        citation rather than as an error.
        """
        return f"{self.doc_title} {self.clause_ref} {self.title}\n{self.text}"

    @property
    def is_citable(self) -> bool:
        return self.tier in CITABLE_TIERS

    def with_score(self, score: float) -> Chunk:
        return replace(self, score=score)


def visible_scopes(principal: Principal) -> list[str] | None:
    """The account scopes this principal may retrieve, or None for unrestricted.

    Staff read every account by design (D26); a customer reads its own plus
    everything unscoped. The account id is validated before it reaches a query
    because `Principal` is a plain frozen dataclass - constructing one through
    `build_principal` guarantees the shape, and nothing else should be trusted
    to have done so.
    """
    if principal.is_staff:
        return None
    account_id = principal.account_id or ""
    if not _ACCOUNT_ID.match(account_id):
        raise ValueError(f"refusing to query with malformed account_id {account_id!r}")
    return [account_id, GLOBAL_SCOPE]


@runtime_checkable
class VectorStore(Protocol):
    """Dense retrieval over the clause registry."""

    #: `provider/model/dimensions`. Namespaces the collection (D20).
    identity: str
    collection_name: str

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Replace the collection's contents with `chunks`. Returns the count."""
        ...

    def query(
        self,
        text: str,
        *,
        principal: Principal,
        tiers: frozenset[int] | set[int] = ...,
        topics: Sequence[str] | None = ...,
        k: int = ...,
    ) -> list[Chunk]: ...

    def count(self) -> int: ...
