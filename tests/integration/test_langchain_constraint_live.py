"""Why the graph does not use `create_react_agent`. Run with `-m live`.

ARCHITECTURE section 11 specifies LangGraph's prebuilt ReAct agent. That
prebuilt takes a LangChain chat model, and `langchain-openai` against Gemini
fails on the second turn of any tool-using conversation:

    400 Function call is missing a thought_signature in functionCall parts.

Gemini 3.x returns an opaque per-tool-call signature and rejects the next
request unless it comes back. Our provider captures and replays it
(`to_assistant_message`, closed in M0); `ChatOpenAI` has nowhere to put it, so
it is dropped. Since Gemini carries dev, tests and the demo (D9a), the prebuilt
is unusable and the graph drives our own provider instead.

This file exists so that deviation is a measured fact with a reproduction rather
than a claim in a commit message - and so it expires by itself. **If this test
starts failing, langchain-openai has fixed the round trip and `create_react_agent`
should be reconsidered.**
"""

from __future__ import annotations

import pytest

from src.config import get_settings

pytestmark = pytest.mark.live

GET_ORDER = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "Look up an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1001"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


@pytest.fixture
def gemini_chat():
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not configured")
    return ChatOpenAI(
        model=settings.gemini_model_cheap,
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        max_tokens=300,
    )


class TestTheConstraintThatForcedTheDeviation:
    def test_langchain_still_drops_the_thought_signature(self, gemini_chat):
        from langchain_core.messages import HumanMessage, ToolMessage

        bound = gemini_chat.bind_tools([GET_ORDER])
        first = bound.invoke([HumanMessage("What is the status of order ORD-1001?")])
        if not first.tool_calls:
            pytest.skip("the model did not call a tool; nothing to round-trip")

        # Nowhere in the LangChain message carries the signature Gemini needs.
        carried = {**first.additional_kwargs, **first.response_metadata}
        assert not any("thought" in key.lower() for key in carried)

        messages = [
            HumanMessage("What is the status of order ORD-1001?"),
            first,
            ToolMessage(content='{"status": "BOOKED"}', tool_call_id=first.tool_calls[0]["id"]),
        ]
        with pytest.raises(Exception) as failure:
            bound.invoke(messages)
        assert "thought_signature" in str(failure.value), (
            "langchain-openai now round-trips Gemini tool calls. The reason the graph "
            "avoids create_react_agent has gone away; reconsider ARCHITECTURE section 11."
        )

    def test_our_provider_does_round_trip_the_same_conversation(self):
        # The other half of the argument: the constraint is LangChain's, not
        # Gemini's, and our provider layer already clears it.
        from src.providers.registry import get_chat_provider

        provider = get_chat_provider()
        messages = [{"role": "user", "content": "What is the status of order ORD-1001?"}]
        first = provider.complete(messages, tools=[GET_ORDER], tier="cheap", max_tokens=300)
        if not first.has_tool_calls:
            pytest.skip("the model did not call a tool; nothing to round-trip")

        call = first.tool_calls[0]
        second = provider.complete(
            [
                *messages,
                provider.to_assistant_message(first),
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"status": "BOOKED"}',
                },
            ],
            tools=[GET_ORDER],
            tier="cheap",
            max_tokens=300,
        )
        assert second.text
