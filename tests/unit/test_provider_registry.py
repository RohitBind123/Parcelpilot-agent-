"""Registry, embedding cache, and startup preflight.

The cache exists because query embedding sits on the hot path of every
retrieval, and the preflight exists because two model slugs died in one
afternoon. Both are about the demo not falling over.
"""

from __future__ import annotations

import pytest
from src.config import EmbeddingConfig, Settings
from src.providers.base import ProviderError
from src.providers.cache import CachedEmbeddings, SqliteEmbeddingCache
from src.providers.registry import get_chat_provider, get_embedding_provider, reset_providers

EMBED_CFG = EmbeddingConfig(
    provider="gemini", api_key="k", base_url="https://x.invalid/", model="embed-1", dimensions=3
)


class RecordingEmbeddings:
    """Counts calls so cache hits are observable."""

    identity = EMBED_CFG.identity
    dimensions = EMBED_CFG.dimensions

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), 0.5, 0.25] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def cache(tmp_path):
    return SqliteEmbeddingCache(tmp_path / "cache.db")


class TestSqliteEmbeddingCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("id/a/3", "unseen") is None

    def test_round_trips_a_vector(self, cache):
        cache.put("id/a/3", "cancellation fee", [0.1, 0.2, 0.3])
        assert cache.get("id/a/3", "cancellation fee") == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)

    def test_is_keyed_by_identity_as_well_as_text(self, cache):
        # The same words under a different embedding model are a different
        # vector. Sharing the key would poison every similarity score.
        cache.put("gemini/a/3", "same words", [0.1, 0.2, 0.3])
        assert cache.get("openrouter/b/3", "same words") is None

    def test_writing_twice_is_idempotent(self, cache):
        cache.put("id/a/3", "t", [0.1, 0.2, 0.3])
        cache.put("id/a/3", "t", [0.9, 0.8, 0.7])
        assert cache.get("id/a/3", "t") == pytest.approx([0.9, 0.8, 0.7], abs=1e-6)

    def test_survives_reopening_the_file(self, tmp_path):
        path = tmp_path / "cache.db"
        SqliteEmbeddingCache(path).put("id/a/3", "t", [0.1, 0.2, 0.3])
        assert SqliteEmbeddingCache(path).get("id/a/3", "t") is not None

    def test_creates_its_parent_directory(self, tmp_path):
        cache = SqliteEmbeddingCache(tmp_path / "nested" / "deep" / "cache.db")
        cache.put("id/a/3", "t", [0.1, 0.2, 0.3])
        assert cache.get("id/a/3", "t") is not None


class TestCachedEmbeddings:
    def test_first_query_calls_through_and_second_does_not(self, cache):
        inner = RecordingEmbeddings()
        wrapped = CachedEmbeddings(inner, cache)
        assert wrapped.embed_query("fee") == wrapped.embed_query("fee")
        assert len(inner.batches) == 1

    def test_preserves_identity_and_dimensions(self, cache):
        wrapped = CachedEmbeddings(RecordingEmbeddings(), cache)
        assert wrapped.identity == EMBED_CFG.identity
        assert wrapped.dimensions == 3

    def test_a_partially_cached_batch_only_embeds_the_misses(self, cache):
        # One request for the misses, not one request per miss.
        inner = RecordingEmbeddings()
        wrapped = CachedEmbeddings(inner, cache)
        wrapped.embed_documents(["a", "bb"])
        inner.batches.clear()
        wrapped.embed_documents(["a", "bb", "ccc"])
        assert inner.batches == [["ccc"]]

    def test_batch_order_is_preserved_across_a_partial_hit(self, cache):
        inner = RecordingEmbeddings()
        wrapped = CachedEmbeddings(inner, cache)
        wrapped.embed_documents(["bb"])
        out = wrapped.embed_documents(["a", "bb", "ccc"])
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]

    def test_duplicate_texts_in_one_batch_are_embedded_once(self, cache):
        inner = RecordingEmbeddings()
        CachedEmbeddings(inner, cache).embed_documents(["a", "a", "bb"])
        assert inner.batches == [["a", "bb"]]

    def test_empty_batch_touches_nothing(self, cache):
        inner = RecordingEmbeddings()
        assert CachedEmbeddings(inner, cache).embed_documents([]) == []
        assert inner.batches == []


class TestRegistry:
    def setup_method(self):
        reset_providers()

    def teardown_method(self):
        reset_providers()

    def _settings(self, **env) -> Settings:
        base = {
            "gemini_api_key": "k",
            "vector_store": "chroma_local",
            "sqlite_path": "data/parcelpilot.db",
        }
        return Settings(_env_file=None, **{**base, **env})

    def test_builds_a_gemini_chat_provider(self):
        provider = get_chat_provider(self._settings())
        assert provider.name == "gemini"
        assert provider.model_for("strong") == "gemini-3.6-flash"

    def test_builds_an_openrouter_chat_provider_with_attribution_headers(self):
        # OpenRouter asks for HTTP-Referer and X-Title so usage is attributable.
        provider = get_chat_provider(
            self._settings(llm_provider="openrouter", openrouter_api_key="o")
        )
        assert provider.name == "openrouter"
        assert "X-Title" in provider._extra_headers

    def test_embedding_provider_is_cache_wrapped(self):
        provider = get_embedding_provider(self._settings())
        assert isinstance(provider, CachedEmbeddings)
        assert provider.identity.startswith("gemini/")

    def test_providers_are_memoised_per_settings(self):
        settings = self._settings()
        assert get_chat_provider(settings) is get_chat_provider(settings)

    def test_reset_clears_the_memo(self):
        settings = self._settings()
        first = get_chat_provider(settings)
        reset_providers()
        assert get_chat_provider(settings) is not first


class TestPreflight:
    def test_reports_a_failure_per_check_rather_than_raising(self):
        from src.providers.preflight import PreflightResult, summarise

        results = [
            PreflightResult("chat:cheap", ok=True, detail="ok"),
            PreflightResult("chat:strong", ok=False, detail="404 model not found"),
        ]
        report = summarise(results)
        assert report.ok is False
        assert "chat:strong" in report.text
        assert "404" in report.text

    def test_all_green_summarises_as_ok(self):
        from src.providers.preflight import PreflightResult, summarise

        report = summarise([PreflightResult("chat:cheap", ok=True, detail="ok")])
        assert report.ok is True

    def test_an_empty_run_is_not_silently_ok(self):
        # A preflight that checked nothing must not read as a pass.
        from src.providers.preflight import summarise

        with pytest.raises(ProviderError, match="no checks"):
            summarise([])


class TestRunPreflight:
    """The orchestration, without network.

    The promise is that every check runs even when an earlier one fails: if
    three things are broken you want all three in one report, not a fix-rerun
    cycle per failure.
    """

    def _settings(self):
        return Settings(_env_file=None, gemini_api_key="k", vector_store="chroma_local")

    def _install(self, monkeypatch, chat, embeddings):
        import src.providers.preflight as pf

        monkeypatch.setattr("src.providers.registry.get_chat_provider", lambda s: chat)
        monkeypatch.setattr("src.providers.registry.get_embedding_provider", lambda s: embeddings)
        monkeypatch.setattr(pf, "_check_clock", lambda: "AS_OF stub (Sunday)")

    def test_all_checks_pass_with_a_healthy_provider(self, monkeypatch, as_of_configured):
        from src.providers.preflight import run_preflight

        self._install(monkeypatch, HealthyChat(), HealthyEmbeddings())
        report = run_preflight(self._settings())
        assert report.ok, report.text
        assert {r.name for r in report.results} >= {
            "chat:cheap",
            "chat:strong",
            "chat:tool_calling",
            "chat:structured_output",
            "embeddings",
            "clock",
        }

    def test_one_broken_check_does_not_stop_the_others(self, monkeypatch, as_of_configured):
        from src.providers.preflight import run_preflight

        self._install(monkeypatch, HealthyChat(tool_calls=False), HealthyEmbeddings())
        report = run_preflight(self._settings())
        assert report.ok is False
        failed = {r.name for r in report.results if not r.ok}
        assert failed == {"chat:tool_calling"}
        assert len(report.results) >= 6

    def test_an_empty_reply_fails_rather_than_passing_quietly(self, monkeypatch, as_of_configured):
        # A thinking model on a tight token budget spends it all on reasoning
        # and returns "". A check that accepts that is not a check.
        from src.providers.preflight import run_preflight

        self._install(monkeypatch, HealthyChat(text=""), HealthyEmbeddings())
        report = run_preflight(self._settings())
        assert {r.name for r in report.results if not r.ok} >= {"chat:cheap", "chat:strong"}

    def test_a_wrong_width_vector_is_caught(self, monkeypatch, as_of_configured):
        from src.providers.preflight import run_preflight

        self._install(monkeypatch, HealthyChat(), HealthyEmbeddings(width=8))
        report = run_preflight(self._settings())
        assert not report.ok
        (embed_result,) = [r for r in report.results if r.name == "embeddings"]
        assert "8" in embed_result.detail


class HealthyChat:
    name = "stub"

    def __init__(self, *, text="ok", tool_calls=True):
        self._text = text
        self._tool_calls = tool_calls

    def model_for(self, tier):
        return f"stub-{tier}"

    def complete(self, messages, *, tools=None, tier="strong", max_tokens=None, **_):
        from src.providers.base import Completion, ToolCall

        calls = ()
        if tools and self._tool_calls:
            calls = (ToolCall(id="c1", name="get_order", arguments={"order_id": "ORD-1001"}),)
        return Completion(text=self._text, model=self.model_for(tier), tool_calls=calls)

    def complete_structured(self, messages, *, schema, schema_name, tier="cheap"):
        return {"ok": True}


class HealthyEmbeddings:
    identity = "stub/embed/3"

    def __init__(self, *, width=3):
        self.dimensions = 3
        self._width = width

    def embed_query(self, text):
        return [0.1] * self._width
