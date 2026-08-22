"""Config to provider instances.

The rest of the system asks for "the chat provider" and never branches on a
provider name. Adding a third provider is a new module plus one entry here.
"""

from __future__ import annotations

from typing import Any, Final

from src.config import Settings, get_settings
from src.providers import gemini, openrouter
from src.providers.base import ProviderError
from src.providers.cache import CachedEmbeddings, SqliteEmbeddingCache

_CHAT_BUILDERS: Final = {"gemini": gemini.build_chat, "openrouter": openrouter.build_chat}
_EMBED_BUILDERS: Final = {
    "gemini": gemini.build_embeddings,
    "openrouter": openrouter.build_embeddings,
}

_chat_cache: dict[int, Any] = {}
_embed_cache: dict[int, Any] = {}


def reset_providers() -> None:
    """Drop memoised providers. Used by tests and by a config reload."""
    _chat_cache.clear()
    _embed_cache.clear()


def get_chat_provider(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    key = id(settings)
    if key not in _chat_cache:
        config = settings.chat_config()
        builder = _CHAT_BUILDERS.get(config.provider)
        if builder is None:
            raise ProviderError(f"no chat provider registered for {config.provider!r}")
        _chat_cache[key] = builder(config, timeout=settings.request_timeout_seconds)
    return _chat_cache[key]


def get_embedding_provider(settings: Settings | None = None) -> Any:
    """The embedding provider, wrapped in its cache.

    Wrapping happens here rather than at each call site so nothing can reach an
    uncached provider by accident and put a network call on the retrieval path.
    """
    settings = settings or get_settings()
    key = id(settings)
    if key not in _embed_cache:
        config = settings.embedding_config()
        builder = _EMBED_BUILDERS.get(config.provider)
        if builder is None:
            raise ProviderError(f"no embedding provider registered for {config.provider!r}")
        inner = builder(config, timeout=settings.request_timeout_seconds)
        cache = SqliteEmbeddingCache(settings.db_path.with_name("embedding_cache.db"))
        _embed_cache[key] = CachedEmbeddings(inner, cache)
    return _embed_cache[key]
