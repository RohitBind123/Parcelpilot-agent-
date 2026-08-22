"""The model half of severity inference (D23).

Kept apart from `severity.py` so the guards, the definitions and the validation
have no import path to a provider. That matters twice: the unit tests grade
inference offline against a stub, and a deployment with no model configured
still pages for an outage because the guards do not depend on this file.

The prompt carries the definitions read from the clause registry rather than a
copy written here, and asks for the span the model matched. Neither is
politeness: the span is checked against the clause in `infer_severity`, and a
model that cannot quote the definition it claims to have applied is one whose
grading is not accepted.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from src.domain.severity import ClassifierVerdict, SeverityDefinitions
from src.providers.base import ChatProvider

logger = logging.getLogger(__name__)

#: Small on purpose. The answer is three fields, and an uncapped request
#: reserves the whole budget against the account balance on OpenRouter.
MAX_TOKENS: Final = 300

_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["P1", "P2", "P3"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "matched_span": {
            "type": "string",
            "description": "The exact words from the definition you applied, copied verbatim.",
        },
    },
    "required": ["severity", "confidence", "matched_span"],
    "additionalProperties": False,
}

_SYSTEM: Final = (
    "You grade support tickets against one company's written severity policy. "
    "Apply only the definitions given to you; do not use severity conventions "
    "from anywhere else. Copy `matched_span` verbatim from the definition you "
    "applied - it is checked against the policy text, and an approximation is "
    "treated as a failure to ground. Report `confidence` as your real belief: "
    "a low value routes the ticket to a human, which is a good outcome when the "
    "definitions genuinely do not settle it."
)


class LlmSeverityClassifier:
    """Grades a ticket against the definition spans it is handed."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def classify(
        self, subject: str, description: str, definitions: SeverityDefinitions
    ) -> ClassifierVerdict | None:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _prompt(subject, description, definitions)},
        ]
        raw = self._provider.complete_structured(
            messages, schema=_SCHEMA, schema_name="severity_grading", tier="cheap"
        )
        return _to_verdict(raw)


def _prompt(subject: str, description: str, definitions: SeverityDefinitions) -> str:
    spans = "\n".join(f"{sev}: {definitions.spans[sev]}" for sev in sorted(definitions.spans))
    return (
        f"Severity definitions from {definitions.clause_id}:\n{spans}\n\n"
        f"Ticket subject: {subject}\n"
        f"Ticket description: {description}\n\n"
        "Return the severity, your confidence, and the verbatim span you matched."
    )


def _to_verdict(raw: Any) -> ClassifierVerdict | None:
    """Whatever came back, as a verdict or as nothing.

    A malformed response is not an approximation of a grading. Returning None
    lets `infer_severity` record "the classifier returned no verdict", which is
    the honest thing to put in a trace.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("severity classifier returned text that is not JSON")
            return None
    try:
        return ClassifierVerdict(
            severity=str(raw["severity"]),
            confidence=float(raw["confidence"]),
            matched_span=str(raw["matched_span"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("severity classifier response was unusable: %s", exc)
        return None
