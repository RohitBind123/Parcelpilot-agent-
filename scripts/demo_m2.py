"""Run the M2 pipeline by hand: registry, index, retrieval, per persona.

    uv run python scripts/demo_m2.py
    uv run python scripts/demo_m2.py --query "can I cancel without a fee?"
    uv run python scripts/demo_m2.py --persona northstar_customer --topic cancellation_fee
    uv run python scripts/demo_m2.py --offline    # deterministic stub embeddings

What it is for is seeing the access control work rather than reading a test
that asserts it. The default run asks every persona the same question, so the
interesting column is which clauses each one is allowed to reach: Northstar
sees its own waiver, LumenWorks sees the clause that declines to waive, Beacon
and Axis see only the general SOP, and staff see all of it.

Nothing here is part of the app. It exists so the pipeline is inspectable
without a UI in front of it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.personas import get_persona, list_personas, to_principal
from src.config import get_settings
from src.knowledge.registry import RegistryError, load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.chroma import ChromaLocalStore, build_vector_store
from src.providers.registry import get_embedding_provider

DEFAULT_QUERY = "will I be charged a fee if I cancel this booking?"
WIDTH = 96


def rule(title: str = "") -> None:
    print(f"\n{title}\n{'=' * WIDTH}" if title else "=" * WIDTH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--persona", default=None, help="one persona id; default is all")
    parser.add_argument("--topic", default=None, help="restrict to one topic")
    parser.add_argument("--tier", type=int, action="append", help="tiers to allow; repeatable")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="stub embeddings, no network")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s  %(message)s",
    )

    settings = get_settings()
    try:
        chunks = load_chunks(settings.db_path)
    except RegistryError as exc:
        print(f"cannot read the registry: {exc}")
        return 1

    if args.offline:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.support.embeddings import HashingEmbeddings

        embeddings = HashingEmbeddings()
        store = ChromaLocalStore(embeddings=embeddings, persist_dir=settings.chroma_dir / "demo")
        store.upsert(chunks)
    else:
        embeddings = get_embedding_provider(settings)
        store = build_vector_store(settings, embeddings)
        # `count()` promises not to raise for a collection that was never
        # built, but that promise rests on the hosted client reporting absence
        # as NotFoundError - a transport detail, not a language guarantee (see
        # TestTheHostedAbsentCollectionContract). A demo script should print
        # the next command to run, not a traceback, if that ever changes.
        try:
            held = store.count()
        except Exception as exc:
            print(f"cannot reach {store.collection_name} ({type(exc).__name__}: {exc})")
            return 1
        if held == 0:
            print(f"{store.collection_name} is empty; run scripts/build_index.py first")
            return 1

    retriever = HybridRetriever(dense=store, lexical=BM25Index(chunks))

    rule("corpus")
    print(f"  {len(chunks)} clauses from {len({c.doc_id for c in chunks})} documents")
    print(f"  embedding  {embeddings.identity}")
    print(f"  store      {type(store).__name__} -> {store.collection_name} ({store.count()})")

    personas = [get_persona(args.persona)] if args.persona else list(list_personas())
    tiers = frozenset(args.tier) if args.tier else None

    rule(f'query: "{args.query}"' + (f"   topic={args.topic}" if args.topic else ""))

    for persona in personas:
        principal = to_principal(persona)
        scope = principal.account_id or principal.role
        print(f"\n  {persona.display_name}  ({scope})")

        kwargs = {"principal": principal, "k": args.k}
        if args.topic:
            kwargs["topics"] = [args.topic]
        if tiers:
            kwargs["tiers"] = tiers

        found = retriever.retrieve(args.query, **kwargs)
        if not found:
            print("      (nothing retrievable)")
            continue
        for rank, chunk in enumerate(found, start=1):
            owner = chunk.account_id or "all accounts"
            print(f"      {rank}. t{chunk.tier}  {chunk.citation}  [{owner}]")
            print(f"         {chunk.title}")

    rule()
    print("Note which clauses never appear for which persona. That is the ACL,")
    print("enforced inside the store and the lexical index rather than by a prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
