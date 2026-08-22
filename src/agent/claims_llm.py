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
    "You split a support answer into the atomic factual claims it asserts. "
    "Include only what the text states as fact in its own voice. "
    "A conditional, a hedge, an instruction, a question, and a report of what "
    "someone else said are not claims - and a sentence telling the reader NOT "
    "to conclude something asserts the opposite of that thing, so do not list "
    "the thing it warns against. Keep each claim to one assertion, in the "
    "text's own words as far as possible, and resolve pronouns."
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


def _to_claims(raw: Any) -> list[str]:
    """Whatever came back, as a list of claims or as nothing.

    Returning [] on a malformed response is safe here and only here: the gate
    reads an empty extraction as UNCHECKED, never as a pass, so a broken
    extractor stops an answer rather than waving it through.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("claim extractor returned text that is not JSON")
            return []
    try:
        claims = [str(c).strip() for c in raw["claims"] if str(c).strip()]
    except (KeyError, TypeError) as exc:
        logger.warning("claim extractor response was unusable: %s", exc)
        return []
    return claims[:MAX_CLAIMS]
