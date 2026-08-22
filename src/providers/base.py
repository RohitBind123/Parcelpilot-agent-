"""Provider-agnostic contracts for chat and embeddings.

Gemini and OpenRouter both expose an OpenAI-compatible endpoint, so
provider-agnosticism here costs three settings rather than an abstraction
layer (D9a). These protocols exist so the rest of the system never branches on
a provider name, and so a third provider would be one new module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

#: Quality tier. Cheap handles high-volume, structurally simple work (claim
#: extraction, severity inference, query rewriting); strong handles the output
#: a human acts on. See ARCHITECTURE.md §15.
Tier = Literal["cheap", "strong"]

Vector = list[float]


class ProviderError(RuntimeError):
    """A model provider call failed, or returned something unusable.

    Deliberately raised rather than swallowed: a half-parsed tool call or a
    truncated vector is worse than an error, because it reaches a tool or a
    collection and corrupts everything downstream of it.
    """


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model, with arguments parsed."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    #: Opaque provider metadata that must be echoed back verbatim on the next
    #: turn. Gemini 3.x attaches a `thought_signature` here and returns 400 on
    #: the following request without it, which would break every multi-step
    #: conversation on the primary provider. Carried, never interpreted.
    provider_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """One model turn."""

    text: str
    model: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class ChatProvider(Protocol):
    """A chat model that can call tools and emit constrained JSON."""

    name: str

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = ...,
        response_format: Mapping[str, Any] | None = ...,
        tier: Tier = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
    ) -> Completion: ...

    def complete_structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        tier: Tier = ...,
    ) -> Mapping[str, Any]: ...

    def to_assistant_message(self, completion: Completion) -> Mapping[str, Any]:
        """Rebuild the assistant turn for the next request.

        On the protocol because the wire shape is the provider's business:
        this is where opaque per-provider metadata is put back.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """An embedding model, pinned by identity.

    `identity` is `provider/model/dimensions`. It namespaces the vector
    collection, so switching provider selects a different collection rather
    than silently comparing vectors that were never in the same space.
    """

    identity: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...
