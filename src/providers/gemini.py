"""Gemini provider wiring.

Primary for dev, tests and the demo (D9a). Reached through its
OpenAI-compatible endpoint at `/v1beta/openai/`, so there is no separate
client - only the quirks recorded here.

Verified live on 2026-08-22:
  - `dimensions` is honoured on embeddings (1536 requested, 1536 returned;
    the model's native width is 3072).
  - Batched embedding input works.
  - Tool calling works, **provided** each tool call's `thought_signature` is
    echoed back on the next turn. See `OpenAICompatibleChat.to_assistant_message`.
"""

from __future__ import annotations

from src.config import ChatConfig, EmbeddingConfig
from src.providers.openai_compatible import OpenAICompatibleChat, OpenAICompatibleEmbeddings


def build_chat(config: ChatConfig, *, timeout: float = 30.0) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(config, timeout=timeout)


def build_embeddings(
    config: EmbeddingConfig, *, timeout: float = 30.0
) -> OpenAICompatibleEmbeddings:
    return OpenAICompatibleEmbeddings(config, supports_dimensions=True, timeout=timeout)
