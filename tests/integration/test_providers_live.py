"""Live provider checks. Deselected by default; run with `-m live`.

These exist because the unit suite can only prove we handle the wire format we
think we are getting. Two slugs died in one afternoon, and Gemini's tool-call
contract has a requirement no amount of mocking would have revealed.
"""

from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.providers.registry import get_chat_provider, get_embedding_provider, reset_providers

pytestmark = pytest.mark.live


def _settings(provider: str) -> Settings:
    settings = Settings(vector_store="chroma_local")
    if not getattr(settings, f"{provider}_api_key", ""):
        pytest.skip(f"{provider.upper()}_API_KEY not configured")
    return Settings(vector_store="chroma_local", llm_provider=provider, embedding_provider=provider)


@pytest.fixture(autouse=True)
def _clean():
    reset_providers()
    yield
    reset_providers()


@pytest.fixture(params=["gemini", "openrouter"])
def provider_settings(request) -> Settings:
    if request.param == "openrouter":
        # Implemented and switchable, but the account is deliberately unfunded
        # (D9a). Opt in explicitly rather than failing a suite on a balance.
        pytest.importorskip("openai")
        settings = Settings(vector_store="chroma_local")
        if not settings.openrouter_api_key:
            pytest.skip("OPENROUTER_API_KEY not configured")
    return _settings(request.param)


def test_chat_answers_on_both_tiers(provider_settings):
    chat = get_chat_provider(provider_settings)
    for tier in ("cheap", "strong"):
        completion = chat.complete(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            tier=tier,
            max_tokens=256,
        )
        assert completion.text.strip(), f"{chat.model_for(tier)} returned no text"


def test_structured_output_is_constrained(provider_settings):
    chat = get_chat_provider(provider_settings)
    out = chat.complete_structured(
        [{"role": "user", "content": "Classify the severity of a total production outage."}],
        schema={
            "type": "object",
            "properties": {"severity": {"type": "string", "enum": ["P1", "P2", "P3"]}},
            "required": ["severity"],
            "additionalProperties": False,
        },
        schema_name="severity",
    )
    assert out["severity"] in {"P1", "P2", "P3"}


def test_embeddings_return_the_configured_width(provider_settings):
    embeddings = get_embedding_provider(provider_settings)
    vector = embeddings.embed_query("cancellation fee waiver before pickup")
    assert len(vector) == embeddings.dimensions


def test_multi_step_tool_chain_survives_the_provider_metadata_round_trip(provider_settings):
    """The regression that matters most.

    Gemini 3.x attaches a thought_signature to each tool call and rejects the
    next request without it, so every conversation past the first tool call
    would 400. This walks the real ORD-1001 chain: get_order, then
    resolve_policy keyed on the snapshot, then the calculator keyed on both.

    It also demonstrates D13a: nothing scripts the order. The typed handles
    make skipping a step impossible, and the model finds the chain itself.
    """
    chat = get_chat_provider(provider_settings)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Look up an order on your account. Returns a snapshot_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_policy",
                "description": "Resolve the governing clause. Requires a snapshot_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "enum": ["cancellation_fee"]},
                        "snapshot_id": {"type": "string"},
                    },
                    "required": ["topic", "snapshot_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_cancellation_fee",
                "description": "Compute the fee. Requires a snapshot_id AND a resolution_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "snapshot_id": {"type": "string"},
                        "resolution_id": {"type": "string"},
                    },
                    "required": ["snapshot_id", "resolution_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    stubs = {
        "get_order": {"snapshot_id": "snap_a1", "status": "BOOKED", "account_id": "ACCT-001"},
        "resolve_policy": {"resolution_id": "res_b2", "governing": "Northstar Agreement s2"},
        "compute_cancellation_fee": {"fee_inr": 0, "eligible": True},
    }

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are ParcelPilot's support agent. Use tools for every factual claim. "
                "Never state a fee you did not compute with a tool."
            ),
        },
        {"role": "user", "content": "Can Northstar cancel ORD-1001 without a cancellation fee?"},
    ]

    called: list[str] = []
    for _ in range(6):
        completion = chat.complete(messages, tools=tools)
        if not completion.has_tool_calls:
            assert completion.text.strip()
            break
        messages.append(chat.to_assistant_message(completion))
        for call in completion.tool_calls:
            called.append(call.name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(stubs.get(call.name, {})),
                }
            )
    else:
        pytest.fail(f"tool loop did not terminate; calls were {called}")

    assert called[0] == "get_order", f"expected the chain to start at get_order, got {called}"
    assert "compute_cancellation_fee" in called, f"chain never reached the calculator: {called}"
    assert called.index("resolve_policy") < called.index("compute_cancellation_fee")
