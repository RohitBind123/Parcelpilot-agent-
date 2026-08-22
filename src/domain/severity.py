"""Severity, so far as it can be decided without a model (D23).

Policy v3 section 2 names two P1 triggers in words specific enough to match
deterministically: a complete production outage preventing all shipment
creation, and a confirmed security incident or suspected credential exposure.
Those two never reach model inference. A P1 that a sampled classifier
occasionally reads as P2 is an outage nobody was paged for, and the cost is not
symmetric with the cost of over-triaging.

Everything else is judgment, so it goes to a model - and a model is not trusted
to grade itself. Three things are checked in Python before a sampled verdict is
believed: that the severity is one Policy v3 actually defines, that the
definition span the model quotes is really in the clause, and that the guards
above cannot be reached at all once they have fired. A classifier that invents a
citation is a worse failure than one that is unsure, because unsure is visible.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

logger = logging.getLogger(__name__)

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
    #: Why a verdict is not to be trusted, in the words the answer can use.
    warnings: tuple[str, ...] = field(default_factory=tuple)

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


# ---------------------------------------------------------------------------
# The definitions, read from the clause rather than retyped here.

#: "- P1 - Critical: ..." up to the next bullet. Written to tolerate the leading
#: dash being absent, because normalisation of the PDF bullets is not this
#: module's business.
_DEFINITION: Final = re.compile(
    r"^\s*-?\s*(P[123])\s*[-\u2013]\s*\w+\s*:\s*(.+?)(?=\n\s*-?\s*P[123]\s*[-\u2013]|\Z)",
    re.MULTILINE | re.DOTALL,
)

_WHITESPACE: Final = re.compile(r"\s+")

_DEFINITIONS_QUERY: Final = """
    SELECT c.clause_id, c.text
    FROM clauses c
    JOIN clause_topics t ON t.clause_id = c.clause_id
    WHERE t.topic = 'severity_definition' AND c.tier < 4
    ORDER BY c.tier
    LIMIT 1
"""


class SeverityError(RuntimeError):
    """The severity definitions could not be read."""


@dataclass(frozen=True, slots=True)
class SeverityDefinitions:
    """The three definition spans, and the clause that states them."""

    clause_id: str
    spans: Mapping[str, str]

    def contains(self, severity: str, span: str) -> bool:
        """Is this quotation really in the definition it claims to come from?

        Compared on collapsed whitespace and case, because a model quoting
        across a line break has still quoted the policy - the check exists to
        catch invention, not formatting.
        """
        definition = self.spans.get(severity)
        if not definition or not span.strip():
            return False
        return _flatten(span) in _flatten(definition)


def _flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def load_severity_definitions(connection: sqlite3.Connection) -> SeverityDefinitions:
    """The current policy's severity definitions.

    The tier predicate is what keeps the deprecated Policy v2 out. v2 defines
    the same three severities against different response targets, so grading a
    ticket by it produces a severity that is defensible in isolation and wrong
    in every answer that quotes a target next to it.
    """
    row = connection.execute(_DEFINITIONS_QUERY).fetchone()
    if row is None:
        raise SeverityError("no citable clause defines severity")

    clause_id, text = row[0], row[1]
    spans = {match.group(1): match.group(2).strip() for match in _DEFINITION.finditer(text)}
    missing = [s for s in SEVERITIES if s not in spans]
    if missing:
        raise SeverityError(f"{clause_id} does not define {', '.join(missing)}")
    return SeverityDefinitions(clause_id=clause_id, spans=spans)


# ---------------------------------------------------------------------------
# Inference, for everything the guards decline to decide.


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    """What a classifier returns before any of it is believed."""

    severity: str
    confidence: float
    #: The words from the definition the classifier says it matched. Checked
    #: against the clause; this is the field that catches an invented citation.
    matched_span: str


class SeverityClassifier(Protocol):
    def classify(
        self, subject: str, description: str, definitions: SeverityDefinitions
    ) -> ClassifierVerdict | None: ...


def infer_severity(
    subject: str,
    description: str,
    *,
    definitions: SeverityDefinitions,
    classifier: SeverityClassifier | None,
) -> SeverityVerdict:
    """Severity by guard where the policy is explicit, by model where it is not.

    Never returns a severity the caller can mistake for a certainty. The three
    ways a verdict can be untrustworthy - no classifier, a classifier that
    failed, a classifier that cited something the policy does not say - all come
    back as a verdict below the threshold with a warning saying which, rather
    than as an exception the caller has to remember to catch.
    """
    if guard := deterministic_severity(subject, description):
        return guard

    if classifier is None:
        return _undetermined("no severity classifier is configured", definitions)

    try:
        raw = classifier.classify(subject, description, definitions)
    except Exception as exc:  # a provider failure is not evidence of a P3
        logger.warning("severity classification failed: %s", exc)
        return _undetermined(f"severity classification failed ({type(exc).__name__})", definitions)

    if raw is None:
        return _undetermined("the classifier returned no verdict", definitions)

    if raw.severity not in SEVERITIES:
        return _undetermined(
            f"the classifier returned severity {raw.severity!r}, "
            f"which {definitions.clause_id} does not define",
            definitions,
        )

    confidence = min(1.0, max(0.0, float(raw.confidence)))
    if not definitions.contains(raw.severity, raw.matched_span):
        # The quotation is the audit trail. Without it there is nothing to
        # check, so the verdict is kept and its confidence is not.
        return SeverityVerdict(
            severity=raw.severity,
            confidence=0.0,
            basis_clause=definitions.clause_id,
            basis="",
            warnings=(
                f"the quoted {raw.severity} definition was not found in "
                f"{definitions.clause_id}, so the grading was not accepted",
            ),
        )

    return SeverityVerdict(
        severity=raw.severity,
        confidence=confidence,
        basis_clause=definitions.clause_id,
        basis=raw.matched_span.strip(),
    )


def _undetermined(reason: str, definitions: SeverityDefinitions) -> SeverityVerdict:
    return SeverityVerdict(
        severity=None,
        confidence=0.0,
        basis_clause=definitions.clause_id,
        basis="",
        warnings=(reason,),
    )
