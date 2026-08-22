"""Severity, so far as it can be decided without a model (D23).

Policy v3 section 2 names two P1 triggers in words specific enough to match
deterministically: a complete production outage preventing all shipment
creation, and a confirmed security incident or suspected credential exposure.
Those two never reach model inference. A P1 that a sampled classifier
occasionally reads as P2 is an outage nobody was paged for, and the cost is not
symmetric with the cost of over-triaging.

Everything else is inferred in M4. This module holds only the guard, plus the
vocabulary both halves share.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

SEVERITIES: Final = ("P1", "P2", "P3")

#: Below this, severity is not trusted enough to quote a target from (D25).
CONFIDENCE_THRESHOLD: Final = 0.7

_SEVERITY_CLAUSE: Final = "support_policy_v3_current::§2"

#: Complete outage of shipment creation. Requires both the totality and the
#: subject: "creating one shipment failed" is not this.
_OUTAGE: Final = re.compile(
    r"(?=.*\b(all|every|any|complete|entire)\b)"
    r"(?=.*\b(shipment|order)s?\b)"
    r"(?=.*\b(creat\w+|book\w+)\b)"
    r"(?=.*\b(fail\w*|error|down|outage|500|unavailable|cannot|can't|unable)\b)",
    re.IGNORECASE | re.DOTALL,
)

#: Credential exposure. "Suspected" is enough - the clause says so, and waiting
#: for confirmation is the failure it is written against.
_CREDENTIAL: Final = re.compile(
    r"(?=.*\b(api[ _-]?key|secret|token|credential|password)s?\b)"
    r"(?=.*\b(expos\w+|leak\w+|post\w+|shar\w+|publish\w+|public|breach\w*|compromis\w+)\b)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class SeverityVerdict:
    severity: str | None
    confidence: float
    basis_clause: str | None
    basis: str
    deterministic: bool = False

    @property
    def is_trusted(self) -> bool:
        return self.severity is not None and self.confidence >= CONFIDENCE_THRESHOLD


def deterministic_severity(subject: str, description: str) -> SeverityVerdict | None:
    """P1 by guard, or None if no guard matches.

    Returns None rather than a low-confidence guess: "the guard did not fire"
    and "the guard says P3" are different statements, and only inference may
    make the second.
    """
    text = f"{subject}\n{description}"
    if _CREDENTIAL.search(text):
        return SeverityVerdict(
            severity="P1",
            confidence=1.0,
            basis_clause=_SEVERITY_CLAUSE,
            basis="suspected credential exposure",
            deterministic=True,
        )
    if _OUTAGE.search(text):
        return SeverityVerdict(
            severity="P1",
            confidence=1.0,
            basis_clause=_SEVERITY_CLAUSE,
            basis="complete outage preventing all shipment creation",
            deterministic=True,
        )
    return None
