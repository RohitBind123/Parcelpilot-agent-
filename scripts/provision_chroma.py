"""Create and inspect the hosted Chroma database (D20).

    uv run python scripts/provision_chroma.py --check       # identity + collections
    uv run python scripts/provision_chroma.py --create      # create the database
    uv run python scripts/provision_chroma.py --drop-stale  # remove foreign collections

The tenant starts with zero databases, so this runs once before the first
hosted index build. It is deliberately not wired into `build_index.py`:
creating remote state should be something a person typed.

Tenant discovery and database creation go through the v2 REST API rather than
the Python client. `chromadb.CloudClient` needs a database that already exists
in order to construct, so it cannot be the thing that creates one, and the
`AdminClient` route only reaches the cloud sysdb through a private attribute.
The two REST calls are stable, documented, and do not depend on client
internals that a patch release can move.

`--drop-stale` removes collections whose name does not match the current
embedding identity - dead weight against a free-tier quota after a provider
switch. It is destructive, so it asks first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, get_settings
from src.knowledge.vectorstore.chroma import collection_name_for
from src.providers.registry import get_embedding_provider

CLOUD_API = "https://api.trychroma.com"
TIMEOUT = 30.0


def _headers(settings: Settings) -> dict[str, str]:
    return {"x-chroma-token": settings.chroma_api_key}


def identity(settings: Settings) -> dict[str, Any]:
    """Who this API key is. The tenant id is not otherwise discoverable."""
    response = httpx.get(
        f"{CLOUD_API}/api/v2/auth/identity", headers=_headers(settings), timeout=TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def list_databases(settings: Settings, tenant: str) -> list[dict[str, Any]]:
    response = httpx.get(
        f"{CLOUD_API}/api/v2/tenants/{tenant}/databases",
        headers=_headers(settings),
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def create_database(settings: Settings, tenant: str, name: str) -> bool:
    """Create the database. Returns False when it already existed."""
    response = httpx.post(
        f"{CLOUD_API}/api/v2/tenants/{tenant}/databases",
        headers=_headers(settings),
        json={"name": name},
        timeout=TIMEOUT,
    )
    if response.status_code in (409, 500) and "exist" in response.text.lower():
        return False
    response.raise_for_status()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report identity and collections")
    parser.add_argument("--create", action="store_true", help="create the database")
    parser.add_argument("--drop-stale", action="store_true", help="remove foreign collections")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    if not (args.check or args.create or args.drop_stale):
        parser.error("choose --check, --create or --drop-stale")

    settings = get_settings()
    if not settings.chroma_api_key:
        print("CHROMA_API_KEY is not set")
        return 1

    try:
        who = identity(settings)
    except httpx.HTTPError as exc:
        print(f"cannot reach Chroma Cloud: {type(exc).__name__}: {exc}")
        return 1

    tenant = settings.chroma_tenant or who["tenant"]
    expected = collection_name_for(get_embedding_provider(settings).identity)

    print(f"tenant      {tenant}")
    if not settings.chroma_tenant:
        print("            (not in .env; set CHROMA_TENANT to pin it)")
    print(f"database    {settings.chroma_database}")
    print(f"collection  {expected}")

    if args.create:
        created = create_database(settings, tenant, settings.chroma_database)
        print(f"\n{'created' if created else 'already present'}  {settings.chroma_database}")

    databases = [db["name"] for db in list_databases(settings, tenant)]
    print(f"\ndatabases ({len(databases)}): {databases or '(none)'}")
    if settings.chroma_database not in databases:
        print(f"  {settings.chroma_database!r} does not exist; run --create")
        return 1

    import chromadb

    client = chromadb.CloudClient(
        tenant=tenant, database=settings.chroma_database, api_key=settings.chroma_api_key
    )
    collections = [c.name for c in client.list_collections()]

    print(f"\ncollections ({len(collections)}):")
    for name in collections:
        print(f"  {name}{'  <- current' if name == expected else ''}")
    if not collections:
        print("  (none; run scripts/build_index.py)")
    elif expected in collections:
        print(f"\n{client.get_collection(expected).count()} vectors in {expected}")

    if args.drop_stale:
        stale = [name for name in collections if name != expected]
        if not stale:
            print("\nnothing stale to drop")
            return 0
        print(f"\nwould drop: {stale}")
        if not args.yes and input("drop them? [y/N] ").strip().lower() != "y":
            print("left alone")
            return 0
        for name in stale:
            client.delete_collection(name)
            print(f"dropped   {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
