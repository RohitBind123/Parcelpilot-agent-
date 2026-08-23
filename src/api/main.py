"""ASGI entrypoint.

Separate from `app.py` so that importing the app factory does not build a real
service - opening the runtime store, resolving providers and reading settings -
as a side effect of an import. Tests import `create_app`; uvicorn imports this.

    uv run uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from src.api.app import create_app

app = create_app()

__all__ = ["app"]
