"""Startup verification for everything that can be dead by tomorrow.

Model slugs churn. In a single afternoon while the data pack was being read,
an OpenRouter `:free` endpoint was delisted and `gemini-2.5-flash` began
returning 404 for new accounts. A configuration that was correct last week is
not evidence that it is correct now.

So the preflight makes one cheap real call per capability the system depends
on, and reports every result rather than stopping at the first failure - if
three things are broken, you want to know all three before you start fixing.
It is run by `scripts/preflight.py` and by the API on startup.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from src.config import Settings, get_settings
from src.providers.base import ProviderError

#: A tool the model has no plausible reason to refuse, used to prove that tool
#: calling works end to end on the configured slug.
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "Look up a shipment order by its identifier.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class PreflightResult:
    name: str
    ok: bool
    detail: str
    latency_ms: int = 0

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        timing = f" [{self.latency_ms} ms]" if self.latency_ms else ""
        return f"  {mark}  {self.name}{timing}  {self.detail}"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    ok: bool
    results: tuple[PreflightResult, ...]

    @property
    def text(self) -> str:
        lines = [r.render() for r in self.results]
        failures = [r for r in self.results if not r.ok]
        header = "preflight passed" if self.ok else f"preflight FAILED ({len(failures)} check(s))"
        return "\n".join([header, *lines])


def summarise(results: Sequence[PreflightResult]) -> PreflightReport:
    """Fold results into a report.

    An empty run raises rather than reporting success: a preflight that
    checked nothing must never read as a pass.
    """
    if not results:
        raise ProviderError("preflight ran no checks; refusing to report success")
    return PreflightReport(ok=all(r.ok for r in results), results=tuple(results))


def _timed(name: str, fn) -> PreflightResult:
    started = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return PreflightResult(
            name, ok=False, detail=f"{type(exc).__name__}: {exc}"[:300], latency_ms=elapsed
        )
    elapsed = int((time.perf_counter() - started) * 1000)
    return PreflightResult(name, ok=True, detail=detail, latency_ms=elapsed)


def run_preflight(settings: Settings | None = None) -> PreflightReport:
    """Make one real call per capability the system depends on."""
    settings = settings or get_settings()
    from src.providers.registry import get_chat_provider, get_embedding_provider

    chat = get_chat_provider(settings)
    embeddings = get_embedding_provider(settings)
    ping = [{"role": "user", "content": "Reply with the single word: ok"}]
    results: list[PreflightResult] = []

    def _ping(tier: str) -> str:
        # Generous max_tokens on purpose: the strong tier is a thinking model,
        # and a tight budget is spent on reasoning before any text is emitted.
        # A check that passes on empty output is not a check.
        completion = chat.complete(ping, tier=tier, max_tokens=256)
        text = completion.text.strip()
        if not text:
            raise ProviderError(
                f"{chat.model_for(tier)} returned no text "
                f"(finish_reason={completion.finish_reason}, "
                f"completion_tokens={completion.completion_tokens})"
            )
        return f"{chat.model_for(tier)} -> {text[:30]!r}"

    for tier in ("cheap", "strong"):
        results.append(_timed(f"chat:{tier}", lambda t=tier: _ping(t)))

    def _tool_calling() -> str:
        # The ACL and confirmation designs assume well-formed tool calls, so a
        # provider that cannot produce one is unusable regardless of prose.
        completion = chat.complete(
            [{"role": "user", "content": "Look up order ORD-1001."}],
            tools=[_PROBE_TOOL],
            tier="cheap",
        )
        if not completion.has_tool_calls:
            raise ProviderError(f"no tool call produced; model said {completion.text[:80]!r}")
        call = completion.tool_calls[0]
        signed = "signed" if call.provider_meta else "unsigned"
        return f"{call.name}({dict(call.arguments)}) [{signed}]"

    results.append(_timed("chat:tool_calling", _tool_calling))

    results.append(
        _timed(
            "chat:structured_output",
            lambda: str(
                chat.complete_structured(
                    [{"role": "user", "content": "Return ok=true."}],
                    schema=_PROBE_SCHEMA,
                    schema_name="probe",
                )
            ),
        )
    )

    def _embeddings() -> str:
        vector = embeddings.embed_query("cancellation fee waiver")
        if len(vector) != embeddings.dimensions:
            raise ProviderError(f"expected {embeddings.dimensions} dims, got {len(vector)}")
        return f"{embeddings.identity} -> {len(vector)} dims"

    results.append(_timed("embeddings", _embeddings))
    results.append(_timed("clock", _check_clock))

    return summarise(results)


def _check_clock() -> str:
    from src.clock import as_of

    moment = as_of()
    return f"AS_OF {moment.isoformat()} ({moment:%A})"
