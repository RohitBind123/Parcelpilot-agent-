"""Everything a toolset needs, opened once per run and scoped to one Principal.

The Principal lives here rather than beside it in `build_toolset`, so a toolset
cannot be built for one identity over a repository opened for another. That
pairing is the entire access-control guarantee; making the mismatch
unexpressible costs nothing and removes a whole class of bug from review.

Opened per run because two of the four dependencies are session state: the
evidence store is keyed by run and Principal, and the repository's scoped views
are bound from the Principal at connect time.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.auth.principal import Principal
from src.config import get_settings
from src.datastore.repo import Repository, open_repository
from src.domain.evidence import EvidenceStore
from src.domain.resolver import PolicyResolver

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class ToolContext:
    principal: Principal
    store: EvidenceStore
    repository: Repository
    resolver: PolicyResolver
    #: Hybrid retrieval. Optional: the structured and calculation tools do not
    #: need it, and a unit test of those should not have to build an index.
    retriever: Any | None = None

    def close(self) -> None:
        self.repository.close()
        self.store.close()


@contextmanager
def open_tool_context(
    principal: Principal,
    *,
    run_id: str = "run",
    db_path: Path | str | None = None,
    retriever: Any | None = None,
    evidence_connection: sqlite3.Connection | None = None,
) -> Iterator[ToolContext]:
    path = Path(db_path or get_settings().db_path)
    repository = open_repository(principal, path)
    evidence = evidence_connection or sqlite3.connect(":memory:")
    store = EvidenceStore(run_id=run_id, principal=principal, connection=evidence)
    context = ToolContext(
        principal=principal,
        store=store,
        repository=repository,
        # Shares the repository's read-only connection: the clause registry
        # lives in the same file, and a second handle would be a second
        # snapshot of a database that is rebuilt rather than migrated.
        resolver=PolicyResolver(repository.connection),
        retriever=retriever,
    )
    try:
        yield context
    finally:
        repository.close()
        if evidence_connection is None:
            evidence.close()
