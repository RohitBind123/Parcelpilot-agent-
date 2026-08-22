"""Both providers speak the OpenAI wire format, so there is one implementation
and two configurations (D9a). These tests pin that contract with a fake client:
no network, no keys, no quota.

The live counterparts are in tests/integration/test_providers_live.py and are
deselected by default.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config import ChatConfig, EmbeddingConfig
from src.providers.base import ProviderError, Tier
from src.providers.openai_compatible import OpenAICompatibleChat, OpenAICompatibleEmbeddings

CHAT_CFG = ChatConfig(
    provider="gemini",
    api_key="k",
    base_url="https://example.invalid/v1/",
    cheap_model="cheap-1",
    strong_model="strong-1",
)

EMBED_CFG = EmbeddingConfig(
    provider="gemini",
    api_key="k",
    base_url="https://example.invalid/v1/",
    model="embed-1",
    dimensions=4,
)


# --------------------------------------------------------------------------
# Fakes


def _message(content="ok", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(content="ok", tool_calls=None, finish_reason="stop", model="strong-1"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(message=_message(content, tool_calls), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )


def _tool_call(call_id="call_1", name="get_order", arguments='{"order_id": "ORD-1001"}'):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments)
    )


class FakeChatClient:
    def __init__(self, response=None, error=None):
        self._response = response or _response()
        self._error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


class FakeEmbeddingClient:
    def __init__(self, vectors=None, error=None):
        self._vectors = vectors or [[0.1, 0.2, 0.3, 0.4]]
        self._error = error
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        n = len(kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=self._vectors[i % len(self._vectors)]) for i in range(n)
            ]
        )


# --------------------------------------------------------------------------


class TestTierRouting:
    def test_strong_is_the_default_tier(self):
        client = FakeChatClient()
        OpenAICompatibleChat(CHAT_CFG, client=client).complete([{"role": "user", "content": "hi"}])
        assert client.calls[0]["model"] == "strong-1"

    def test_cheap_tier_selects_the_cheap_slug(self):
        # Claim extraction, severity inference and query rewriting are high
        # volume and structurally simple; synthesis is where wrongness costs.
        client = FakeChatClient()
        OpenAICompatibleChat(CHAT_CFG, client=client).complete(
            [{"role": "user", "content": "hi"}], tier="cheap"
        )
        assert client.calls[0]["model"] == "cheap-1"

    def test_unknown_tier_is_rejected(self):
        with pytest.raises(ProviderError, match="tier"):
            OpenAICompatibleChat(CHAT_CFG, client=FakeChatClient()).complete(
                [{"role": "user", "content": "hi"}],
                tier="luxury",  # type: ignore[arg-type]
            )

    def test_every_tier_in_the_enum_resolves(self):
        client = FakeChatClient()
        chat = OpenAICompatibleChat(CHAT_CFG, client=client)
        for tier in Tier.__args__:  # type: ignore[attr-defined]
            chat.complete([{"role": "user", "content": "hi"}], tier=tier)
        assert {c["model"] for c in client.calls} == {"cheap-1", "strong-1"}


class TestCompletionShape:
    def test_returns_text_model_and_usage(self):
        chat = OpenAICompatibleChat(CHAT_CFG, client=FakeChatClient())
        result = chat.complete([{"role": "user", "content": "hi"}])
        assert result.text == "ok"
        assert result.model == "strong-1"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3
        assert result.finish_reason == "stop"
        assert not result.has_tool_calls

    def test_null_content_becomes_empty_text_not_none(self):
        # A tool-calling turn has content=None; downstream code should not
        # have to guard against None on every access.
        client = FakeChatClient(_response(content=None, tool_calls=[_tool_call()]))
        assert OpenAICompatibleChat(CHAT_CFG, client=client).complete([]).text == ""

    def test_parses_tool_calls_into_typed_arguments(self):
        client = FakeChatClient(_response(content=None, tool_calls=[_tool_call()]))
        result = OpenAICompatibleChat(CHAT_CFG, client=client).complete([])
        assert result.has_tool_calls
        (call,) = result.tool_calls
        assert call.id == "call_1"
        assert call.name == "get_order"
        assert call.arguments == {"order_id": "ORD-1001"}

    def test_malformed_tool_arguments_raise_rather_than_reaching_a_tool(self):
        # A tool receiving half-parsed arguments is how an ACL check gets
        # skipped. Fail loudly instead.
        client = FakeChatClient(
            _response(content=None, tool_calls=[_tool_call(arguments="{not json")])
        )
        with pytest.raises(ProviderError, match="arguments"):
            OpenAICompatibleChat(CHAT_CFG, client=client).complete([])

    def test_empty_tool_arguments_parse_to_an_empty_mapping(self):
        client = FakeChatClient(_response(content=None, tool_calls=[_tool_call(arguments="")]))
        (call,) = OpenAICompatibleChat(CHAT_CFG, client=client).complete([]).tool_calls
        assert call.arguments == {}


class TestRequestConstruction:
    def test_tools_and_response_format_are_passed_through(self):
        client = FakeChatClient()
        tools = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]
        fmt = {"type": "json_schema", "json_schema": {"name": "r", "schema": {}}}
        OpenAICompatibleChat(CHAT_CFG, client=client).complete([], tools=tools, response_format=fmt)
        assert client.calls[0]["tools"] == tools
        assert client.calls[0]["response_format"] == fmt

    def test_absent_options_are_omitted_not_sent_as_none(self):
        # Gemini's OpenAI-compatible endpoint rejects some explicit nulls.
        client = FakeChatClient()
        OpenAICompatibleChat(CHAT_CFG, client=client).complete([])
        assert "tools" not in client.calls[0]
        assert "response_format" not in client.calls[0]

    def test_a_max_token_cap_is_always_sent(self):
        # OpenRouter reserves max_tokens against the balance before running,
        # so an uncapped request 402s on a low balance even for a short reply.
        # Observed live on 2026-08-22: "you requested up to 65535 tokens, but
        # can only afford 15998".
        client = FakeChatClient()
        OpenAICompatibleChat(CHAT_CFG, client=client).complete([])
        assert client.calls[0]["max_tokens"] == CHAT_CFG.max_output_tokens

    def test_an_explicit_max_token_value_wins(self):
        client = FakeChatClient()
        OpenAICompatibleChat(CHAT_CFG, client=client).complete([], max_tokens=16)
        assert client.calls[0]["max_tokens"] == 16

    def test_extra_headers_from_the_provider_are_sent(self):
        client = FakeChatClient()
        chat = OpenAICompatibleChat(
            CHAT_CFG, client=client, extra_headers={"X-Title": "ParcelPilot"}
        )
        chat.complete([])
        assert client.calls[0]["extra_headers"]["X-Title"] == "ParcelPilot"


class TestStructuredOutput:
    def test_returns_parsed_json(self):
        client = FakeChatClient(_response(content='{"severity": "P1", "confidence": 0.92}'))
        out = OpenAICompatibleChat(CHAT_CFG, client=client).complete_structured(
            [], schema={"type": "object"}, schema_name="severity"
        )
        assert out == {"severity": "P1", "confidence": 0.92}

    def test_requests_a_strict_schema(self):
        client = FakeChatClient(_response(content="{}"))
        OpenAICompatibleChat(CHAT_CFG, client=client).complete_structured(
            [], schema={"type": "object"}, schema_name="severity"
        )
        fmt = client.calls[0]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "severity"
        assert fmt["json_schema"]["strict"] is True

    def test_unparseable_output_raises_rather_than_returning_a_guess(self):
        client = FakeChatClient(_response(content="I think it is P1"))
        with pytest.raises(ProviderError, match="JSON"):
            OpenAICompatibleChat(CHAT_CFG, client=client).complete_structured(
                [], schema={"type": "object"}, schema_name="severity"
            )

    def test_tolerates_a_fenced_code_block(self):
        fenced = "```json\n" + json.dumps({"severity": "P2"}) + "\n```"
        client = FakeChatClient(_response(content=fenced))
        out = OpenAICompatibleChat(CHAT_CFG, client=client).complete_structured(
            [], schema={"type": "object"}, schema_name="severity"
        )
        assert out == {"severity": "P2"}


class TestErrorWrapping:
    def test_provider_errors_name_the_provider_and_model(self):
        client = FakeChatClient(error=RuntimeError("upstream exploded"))
        with pytest.raises(ProviderError) as exc:
            OpenAICompatibleChat(CHAT_CFG, client=client).complete([])
        assert "gemini" in str(exc.value)
        assert "strong-1" in str(exc.value)


class TestEmbeddings:
    def test_identity_and_dimensions_come_from_config(self):
        emb = OpenAICompatibleEmbeddings(EMBED_CFG, client=FakeEmbeddingClient())
        assert emb.identity == "gemini/embed-1/4"
        assert emb.dimensions == 4

    def test_embeds_a_batch_of_documents(self):
        client = FakeEmbeddingClient()
        vectors = OpenAICompatibleEmbeddings(EMBED_CFG, client=client).embed_documents(["a", "b"])
        assert len(vectors) == 2
        assert client.calls[0]["model"] == "embed-1"

    def test_requests_the_configured_dimension(self):
        client = FakeEmbeddingClient()
        OpenAICompatibleEmbeddings(EMBED_CFG, client=client).embed_documents(["a"])
        assert client.calls[0]["dimensions"] == 4

    def test_rejects_a_vector_of_the_wrong_width(self):
        # A silently truncated vector poisons the collection and every later
        # similarity score with it.
        client = FakeEmbeddingClient(vectors=[[0.1, 0.2]])
        with pytest.raises(ProviderError, match="dimension"):
            OpenAICompatibleEmbeddings(EMBED_CFG, client=client).embed_documents(["a"])

    def test_empty_batch_makes_no_request(self):
        client = FakeEmbeddingClient()
        assert OpenAICompatibleEmbeddings(EMBED_CFG, client=client).embed_documents([]) == []
        assert client.calls == []

    def test_embed_query_returns_a_single_vector(self):
        emb = OpenAICompatibleEmbeddings(EMBED_CFG, client=FakeEmbeddingClient())
        assert len(emb.embed_query("cancellation fee")) == 4


class TestProviderMetadataRoundTrip:
    """Gemini 3.x attaches a thought_signature to each tool call and returns
    400 on the next turn if it is not echoed back verbatim:

        "Function call is missing a thought_signature in functionCall parts."

    Verified live on 2026-08-22 against gemini-3.6-flash and
    gemini-3.5-flash-lite. Dropping this metadata breaks every multi-step
    conversation on the primary provider, so it is carried opaquely rather
    than interpreted.
    """

    def _signed_call(self):
        call = _tool_call()
        call.extra_content = {"google": {"thought_signature": "SIG-123"}}
        return call

    def test_provider_metadata_is_captured_from_the_tool_call(self):
        client = FakeChatClient(_response(content=None, tool_calls=[self._signed_call()]))
        (call,) = OpenAICompatibleChat(CHAT_CFG, client=client).complete([]).tool_calls
        assert call.provider_meta == {"google": {"thought_signature": "SIG-123"}}

    def test_absent_metadata_is_none_not_an_empty_dict(self):
        client = FakeChatClient(_response(content=None, tool_calls=[_tool_call()]))
        (call,) = OpenAICompatibleChat(CHAT_CFG, client=client).complete([]).tool_calls
        assert call.provider_meta is None

    def test_assistant_message_echoes_the_metadata_verbatim(self):
        client = FakeChatClient(_response(content=None, tool_calls=[self._signed_call()]))
        chat = OpenAICompatibleChat(CHAT_CFG, client=client)
        message = chat.to_assistant_message(chat.complete([]))
        assert message["role"] == "assistant"
        (echoed,) = message["tool_calls"]
        assert echoed["extra_content"] == {"google": {"thought_signature": "SIG-123"}}

    def test_assistant_message_reserialises_arguments_as_json(self):
        client = FakeChatClient(_response(content=None, tool_calls=[_tool_call()]))
        chat = OpenAICompatibleChat(CHAT_CFG, client=client)
        (echoed,) = chat.to_assistant_message(chat.complete([]))["tool_calls"]
        assert json.loads(echoed["function"]["arguments"]) == {"order_id": "ORD-1001"}
        assert "extra_content" not in echoed

    def test_a_text_only_turn_has_no_tool_calls_key(self):
        chat = OpenAICompatibleChat(CHAT_CFG, client=FakeChatClient())
        message = chat.to_assistant_message(chat.complete([]))
        assert message == {"role": "assistant", "content": "ok"}

    def test_round_trip_survives_being_fed_back_as_a_request(self):
        # The failure this guards is a 400 on turn two of every multi-step
        # conversation, which is every conversation that matters here.
        client = FakeChatClient(_response(content=None, tool_calls=[self._signed_call()]))
        chat = OpenAICompatibleChat(CHAT_CFG, client=client)
        history = [chat.to_assistant_message(chat.complete([]))]
        chat.complete([*history, {"role": "tool", "tool_call_id": "call_1", "content": "{}"}])
        sent = client.calls[-1]["messages"][0]
        assert sent["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG-123"
