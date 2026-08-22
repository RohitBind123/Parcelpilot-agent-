"""One implementation of both provider protocols, over the OpenAI wire format.

Gemini exposes `/v1beta/openai/` and OpenRouter exposes `/api/v1`, both
OpenAI-compatible, so there is no reason for two clients. Provider-specific
quirks are injected as constructor arguments (extra headers, dimension
support), not as subclass branches on a provider name.

Retries are delegated to the OpenAI SDK, which already backs off exponentially
with jitter and honours `retry-after`. Rolling our own on top would double the
backoff and double the quota burn. What we add is the part the SDK cannot know:
a dead slug or a bad key must fail immediately and loudly, because retrying a
404 just delays the moment someone notices the model was delisted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.config import ChatConfig, EmbeddingConfig
from src.providers.base import Completion, ProviderError, Tier, ToolCall, Vector

logger = logging.getLogger(__name__)

_TIERS: frozenset[str] = frozenset({"cheap", "strong"})

#: The SDK retries transient failures; everything else surfaces at once.
_MAX_SDK_RETRIES = 3


def _build_client(api_key: str, base_url: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=_MAX_SDK_RETRIES,
    )


class OpenAICompatibleChat:
    """Chat provider for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ChatConfig,
        *,
        client: Any | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = config.provider
        self._config = config
        self._extra_headers = dict(extra_headers or {})
        self._client = client or _build_client(config.api_key, config.base_url, timeout)

    def model_for(self, tier: Tier) -> str:
        if tier not in _TIERS:
            raise ProviderError(f"unknown tier {tier!r}; expected one of {sorted(_TIERS)}")
        return self._config.cheap_model if tier == "cheap" else self._config.strong_model

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        tier: Tier = "strong",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        model = self.model_for(tier)

        # Absent options are omitted rather than sent as null: Gemini's
        # compatibility layer rejects some explicit nulls that OpenAI ignores.
        request: dict[str, Any] = {"model": model, "messages": list(messages)}
        if tools:
            request["tools"] = list(tools)
        if response_format is not None:
            request["response_format"] = dict(response_format)
        if temperature is not None:
            request["temperature"] = temperature
        # Always send a cap. OpenRouter reserves max_tokens against the
        # account balance before running the request, so relying on a
        # provider default of 65535 fails with 402 on a low balance even when
        # the reply is two words.
        request["max_tokens"] = max_tokens or self._config.max_output_tokens
        if self._extra_headers:
            request["extra_headers"] = dict(self._extra_headers)

        try:
            response = self._client.chat.completions.create(**request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name} chat call failed for model {model!r}: {exc}") from exc

        return self._to_completion(response, model)

    def complete_structured(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        tier: Tier = "cheap",
    ) -> Mapping[str, Any]:
        """Constrained JSON output, parsed.

        Used wherever a downstream branch depends on the answer - severity,
        claim extraction, conflict classification. An unparseable response
        raises instead of degrading to a guess.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": dict(schema)},
        }
        completion = self.complete(messages, response_format=response_format, tier=tier)
        return _parse_json(completion.text, context=f"{self.name}/{schema_name}")

    def to_assistant_message(self, completion: Completion) -> dict[str, Any]:
        """Rebuild the assistant turn, preserving per-tool-call provider metadata.

        Gemini 3.x rejects the next request if a tool call's
        `thought_signature` is missing, so the metadata captured in
        `_to_tool_calls` has to come back out here unchanged.
        """
        message: dict[str, Any] = {"role": "assistant", "content": completion.text}
        if not completion.tool_calls:
            return message

        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                **({"extra_content": dict(call.provider_meta)} if call.provider_meta else {}),
            }
            for call in completion.tool_calls
        ]
        return message

    def _to_completion(self, response: Any, model: str) -> Completion:
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as exc:
            raise ProviderError(f"{self.name} returned no choices for model {model!r}") from exc

        usage = getattr(response, "usage", None)
        return Completion(
            # content is None on a pure tool-calling turn; normalise so callers
            # never have to guard against None on every access.
            text=getattr(message, "content", None) or "",
            model=getattr(response, "model", model),
            tool_calls=self._to_tool_calls(getattr(message, "tool_calls", None)),
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _to_tool_calls(self, raw: Any) -> tuple[ToolCall, ...]:
        if not raw:
            return ()
        parsed: list[ToolCall] = []
        for call in raw:
            function = call.function
            arguments = (function.arguments or "").strip()
            if not arguments:
                decoded: Mapping[str, Any] = {}
            else:
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    # A tool receiving half-parsed arguments is how an ACL
                    # check gets skipped. Never guess at a repair.
                    raise ProviderError(
                        f"{self.name} produced malformed arguments for tool "
                        f"{function.name!r}: {arguments!r}"
                    ) from exc
            extra = getattr(call, "extra_content", None)
            parsed.append(
                ToolCall(
                    id=call.id,
                    name=function.name,
                    arguments=decoded,
                    provider_meta=dict(extra) if extra else None,
                )
            )
        return tuple(parsed)


class OpenAICompatibleEmbeddings:
    """Embedding provider for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        client: Any | None = None,
        supports_dimensions: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.name = config.provider
        self.identity = config.identity
        self.dimensions = config.dimensions
        self._config = config
        self._supports_dimensions = supports_dimensions
        self._client = client or _build_client(config.api_key, config.base_url, timeout)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        request: dict[str, Any] = {"model": self._config.model, "input": list(texts)}
        if self._supports_dimensions:
            request["dimensions"] = self.dimensions

        try:
            response = self._client.embeddings.create(**request)
        except Exception as exc:
            raise ProviderError(
                f"{self.name} embedding call failed for model {self._config.model!r}: {exc}"
            ) from exc

        vectors = [list(item.embedding) for item in response.data]
        for vector in vectors:
            if len(vector) != self.dimensions:
                # A silently truncated vector poisons the collection and every
                # similarity score computed against it afterwards.
                raise ProviderError(
                    f"{self.identity} returned dimension {len(vector)}, expected {self.dimensions}"
                )
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


def _parse_json(text: str, *, context: str) -> Mapping[str, Any]:
    """Parse model JSON, tolerating a fenced block but nothing more.

    Tolerance stops here on purpose. Repairing arbitrary malformed JSON means
    guessing what the model meant, and a guessed severity or a guessed claim
    list is exactly the confident wrongness this system exists to avoid.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{context} did not return JSON: {text[:200]!r}") from exc

    if not isinstance(parsed, Mapping):
        raise ProviderError(f"{context} returned JSON {type(parsed).__name__}, expected an object")
    return parsed
