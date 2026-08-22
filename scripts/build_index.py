"""Embed the clause registry into the configured vector store.

    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --local        # force on-disk Chroma
    uv run python scripts/build_index.py --dry-run      # report, embed nothing

Separate from `build_db.py` on purpose. Building the database is offline,
deterministic and free; building the index costs embedding calls against a
hosted model. Keeping them apart means a schema fix does not spend tokens, and
a re-index does not rebuild the database.

The collection is namespaced by embedding identity (D20), so running this after
switching provider writes a new collection rather than mixing vector spaces.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.knowledge.registry import RegistryError, load_chunks
from src.knowledge.vectorstore.base import GLOBAL_SCOPE
from src.knowledge.vectorstore.chroma import (
    ChromaLocalStore,
    VectorStoreError,
    build_vector_store,
)
from src.providers.registry import get_embedding_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="defaults to SQLITE_PATH")
    parser.add_argument("--local", action="store_true", help="force the on-disk store")
    parser.add_argument("--dry-run", action="store_true", help="report without embedding")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s  %(message)s",
    )

    settings = get_settings()
    db_path = args.db or settings.db_path

    try:
        chunks = load_chunks(db_path)
    except RegistryError as exc:
        print(f"cannot read the registry: {exc}")
        return 1

    embeddings = get_embedding_provider(settings)
    scoped = sum(1 for chunk in chunks if chunk.account_id)

    print(f"registry   {db_path}")
    print(f"clauses    {len(chunks)}  ({scoped} account-scoped, {len(chunks) - scoped} general)")
    print(f"embedding  {embeddings.identity}")

    if args.local:
        store = ChromaLocalStore(embeddings=embeddings, persist_dir=settings.chroma_dir)
    else:
        store = build_vector_store(settings, embeddings)

    print(f"store      {type(store).__name__} -> {store.collection_name}")

    if args.dry_run:
        by_tier: dict[int, int] = {}
        for chunk in chunks:
            by_tier[chunk.tier] = by_tier.get(chunk.tier, 0) + 1
        print(f"tiers      {dict(sorted(by_tier.items()))}")
        print("dry run; nothing embedded")
        return 0

    try:
        written = store.upsert(chunks)
    except VectorStoreError as exc:
        print(f"indexing failed: {exc}")
        return 1

    print(f"indexed    {written} clauses  (collection now holds {store.count()})")
    print(f"scopes     {sorted({c.account_id or GLOBAL_SCOPE for c in chunks})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
