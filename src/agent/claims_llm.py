"""Atomic claim extraction, the cheap-model half of the grounding gate (D16).

Separate from `grounding.py` for the same reason `severity_llm.py` is separate
from `severity.py`: the checking logic must have no import path to a provider,
so the unit tests grade it offline and a deployment with no model still runs the
deterministic half.

The prompt asks for what the answer *asserts in its own voice*. That distinction
is the whole reason the gate is claim-level rather than string-level - an answer
carrying KI-211's instruction not to say something must not be read as saying
it - and it is a linguistic judgement, which is what the model is for. What the
model never decides is whether the claims are supported; that comparison happens
in Python.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from src.providers.base import ChatProvider

logger = logging.getLogger(__name__)

MAX_CLAIMS: Final = 20

_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each factual assertion the text makes, one per item.",
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_SYSTEM: Final = (
    "You split a support answer into the atomic factual claims it asserts "
    "about ParcelPilot's policies, records, or the customer's account. "
    "Include only what the text states as fact in its own voice. "
    "A conditional, a hedge, an instruction, a question, and a report of what "
    "someone else said are not claims - and a sentence telling the reader NOT "
    "to conclude something asserts the opposite of that thing, so do not list "
    "the thing it warns against. Keep each claim to one assertion, in the "
    "text's own words as far as possible, and resolve pronouns.\n\n"
    "Statements about the assistant itself are NOT claims of this kind. A "
    "greeting, an offer to help, a description of what the assistant can do, "
    "and an explanation of how it works all assert nothing about policies or "
    "records - no clause could support or contradict them, so listing one is "
    "an error.\n\n"
    "Do NOT extract, even when they mention a record by name:\n"
    '  "I can look up orders, check cancellation eligibility, and calculate '
    'service credits."\n'
    '  "All figures I report are calculated from system records rather than '
    'written by me."\n'
    '  "The conflict was identified by an automated check comparing ORD-1001 '
    'and TKT-504."\n'
    '  "Hello - how can I help with your shipments today?"\n\n'
    "DO extract, because each states something about the world the sources "
    "describe:\n"
    '  "ORD-1001 can be cancelled with no fee."\n'
    '  "Your agreement waives the cancellation fee."\n'
    '  "The first-response target is four business hours."\n\n'
    "Return an empty list for an answer that makes no assertion about "
    "ParcelPilot's policies, records or accounts."
)


class LlmClaimExtractor:
    """Splits drafted prose into the claims it asserts."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def extract(self, prose: str) -> list[str]:
        raw = self._provider.complete_structured(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prose},
            ],
            schema=_SCHEMA,
            schema_name="atomic_claims",
            tier="cheap",
        )
        return _to_claims(raw)


class ExtractionError(RuntimeError):
    """The extractor answered with something that is not an extraction."""


def _to_claims(raw: Any) -> list[str]:
    """Whatever came back, as a list of claims.

    Raises rather than returning [] when the response is unusable, and that
    distinction is the whole point of this function. "The extractor broke" and
    "the answer asserts nothing about policies or records" are different facts
    with opposite consequences - the first must stop an answer, the second must
    let a greeting through - and returning [] for both made them the same fact.

    They were the same fact for a while, and the symptom was that "tell me what
    you can do" escalated to a human: the gate had no way to tell an answer with
    nothing to ground from an extractor that had failed to ground it.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExtractionError("claim extractor returned text that is not JSON") from exc
    try:
        claims = [str(c).strip() for c in raw["claims"] if str(c).strip()]
    except (KeyError, TypeError) as exc:
        raise ExtractionError(f"claim extractor response was unusable: {exc}") from exc
    return claims[:MAX_CLAIMS]
