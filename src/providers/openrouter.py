"""OpenRouter provider wiring.

Implemented and switchable, deliberately unfunded and off the demo path (D9a):
the account holds no credits and its `:free` slugs are actively being delisted.
Selecting it is one environment variable.

OpenRouter asks callers to send `HTTP-Referer` and `X-Title` so requests are
attributable in its dashboard; neither affects routing.
"""

from __future__ import annotations

from typing import Final

from src.config import ChatConfig, EmbeddingConfig
from src.providers.openai_compatible import OpenAICompatibleChat, OpenAICompatibleEmbeddings

ATTRIBUTION_HEADERS: Final[dict[str, str]] = {
    "HTTP-Referer": "https://github.com/RohitBind123/Parcelpilot-agent-",
    "X-Title": "ParcelPilot AI Support",
}


def build_chat(config: ChatConfig, *, timeout: float = 30.0) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(config, extra_headers=ATTRIBUTION_HEADERS, timeout=timeout)


def build_embeddings(
    config: EmbeddingConfig, *, timeout: float = 30.0
) -> OpenAICompatibleEmbeddings:
    return OpenAICompatibleEmbeddings(config, supports_dimensions=True, timeout=timeout)
