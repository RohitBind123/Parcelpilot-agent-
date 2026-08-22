"""Configuration must fail loudly at startup, not halfway through a demo.

Two model slugs died in a single afternoon while the data pack was being read
(docs/01_DATA_PACK_FINDINGS.md §11), so slugs live here and never in code, and
selecting a provider without its key is an error rather than a 401 later.
"""

from __future__ import annotations

import pytest

from src.config import ConfigError, Settings, get_settings

CLEARED = (
    "LLM_PROVIDER",
    "EMBEDDING_PROVIDER",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_MODEL_STRONG",
    "GEMINI_EMBEDDING_DIM",
    "VECTOR_STORE",
    "CHROMA_API_KEY",
    "SESSION_SECRET",
    "AS_OF",
)


def build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """Settings built from an explicit environment, ignoring the real .env."""
    for key in CLEARED:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key.upper(), value)
    return Settings(_env_file=None)


def offline(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """The posture tests and the eval harness run in: local store, no network."""
    env.setdefault("gemini_api_key", "k")
    env.setdefault("vector_store", "chroma_local")
    return build(monkeypatch, **env)


class TestProviderSelection:
    def test_defaults_to_gemini(self, monkeypatch):
        # D9a: Gemini carries dev, tests and the demo; OpenRouter is unfunded.
        s = offline(monkeypatch)
        assert s.llm_provider == "gemini"
        assert s.embedding_provider == "gemini"

    def test_unknown_provider_is_rejected(self, monkeypatch):
        with pytest.raises(ConfigError, match="llama_farm"):
            offline(monkeypatch, llm_provider="llama_farm")

    def test_selecting_a_provider_without_its_key_names_the_variable(self, monkeypatch):
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            offline(monkeypatch, llm_provider="openrouter")

    def test_an_unselected_provider_may_be_unconfigured(self, monkeypatch):
        # OpenRouter stays implemented but unfunded, so its key is optional
        # until someone actually selects it.
        s = offline(monkeypatch)
        assert s.openrouter_api_key == ""

    def test_llm_and_embedding_providers_may_differ(self, monkeypatch):
        s = offline(
            monkeypatch,
            llm_provider="gemini",
            embedding_provider="openrouter",
            openrouter_api_key="o",
        )
        assert s.llm_provider == "gemini"
        assert s.embedding_provider == "openrouter"


class TestProviderConfig:
    def test_resolves_key_base_url_and_slugs(self, monkeypatch):
        s = offline(monkeypatch)
        cfg = s.chat_config()
        assert cfg.api_key == "k"
        assert cfg.base_url.startswith("https://")
        assert cfg.cheap_model and cfg.strong_model

    def test_slugs_are_overridable_from_the_environment(self, monkeypatch):
        s = offline(monkeypatch, gemini_model_strong="gemini-9-ultra")
        assert s.chat_config().strong_model == "gemini-9-ultra"

    def test_openrouter_config_resolves_when_selected(self, monkeypatch):
        s = offline(monkeypatch, llm_provider="openrouter", openrouter_api_key="o")
        cfg = s.chat_config()
        assert cfg.api_key == "o"
        assert "openrouter.ai" in cfg.base_url


class TestEmbeddingIdentity:
    def test_identity_names_provider_model_and_dimension(self, monkeypatch):
        s = offline(monkeypatch)
        cfg = s.embedding_config()
        assert cfg.identity == f"gemini/{cfg.model}/{cfg.dimensions}"

    def test_collection_name_is_namespaced_by_identity(self, monkeypatch):
        # Swapping embedding provider must select a different collection, not
        # silently compare incompatible vectors (D20).
        gem = offline(monkeypatch).embedding_config().collection_name
        opr = (
            offline(monkeypatch, embedding_provider="openrouter", openrouter_api_key="o")
            .embedding_config()
            .collection_name
        )
        assert gem != opr

    def test_collection_name_is_legal_for_chroma(self, monkeypatch):
        name = offline(monkeypatch).embedding_config().collection_name
        assert 3 <= len(name) <= 512
        assert name[0].isalnum() and name[-1].isalnum()
        assert all(c.isalnum() or c in "_-" for c in name)

    def test_dimension_must_be_positive(self, monkeypatch):
        with pytest.raises(ConfigError, match="dimension"):
            offline(monkeypatch, gemini_embedding_dim="0")


class TestVectorStore:
    def test_defaults_to_chroma_cloud(self, monkeypatch):
        assert (
            build(monkeypatch, gemini_api_key="k", chroma_api_key="c").vector_store
            == "chroma_cloud"
        )

    def test_cloud_without_a_key_is_rejected(self, monkeypatch):
        with pytest.raises(ConfigError, match="CHROMA_API_KEY"):
            build(monkeypatch, gemini_api_key="k")

    def test_local_needs_no_key(self, monkeypatch):
        # Tests and the eval harness run offline against a local store.
        s = build(monkeypatch, gemini_api_key="k", vector_store="chroma_local")
        assert s.vector_store == "chroma_local"

    def test_unknown_store_is_rejected(self, monkeypatch):
        with pytest.raises(ConfigError, match="pinecone"):
            offline(monkeypatch, vector_store="pinecone")


class TestPaths:
    def test_relative_paths_resolve_against_the_repo_root(self, monkeypatch):
        s = offline(monkeypatch)
        assert s.db_path.is_absolute()
        assert s.chroma_dir.is_absolute()
        assert s.db_path.name == "parcelpilot.db"


class TestSessionSecret:
    def test_uses_the_configured_secret(self, monkeypatch):
        s = offline(monkeypatch, session_secret="s3cret")
        assert s.session_secret == "s3cret"

    def test_generates_a_random_secret_rather_than_shipping_a_default(self, monkeypatch):
        # A hardcoded fallback secret is a forgeable session token. Random
        # per-process means sessions do not survive a restart, which is the
        # correct trade for a mocked login.
        a = offline(monkeypatch).session_secret
        b = offline(monkeypatch).session_secret
        assert a and b and a != b
        assert len(a) >= 32


class TestCaching:
    def test_get_settings_is_cached(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
