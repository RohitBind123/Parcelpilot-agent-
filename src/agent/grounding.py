"""The grounding gate (D16), and the mistake it is built not to make.

The prose is model output, so every claim in it must map to a tool result or a
citable clause before anyone reads it. Two mechanisms, deliberately different.

**Figures are checked in Python.** "This answer states a number the evidence
does not contain" needs no judgement and must not be subject to any. The allowed
set is the fact block plus the text of every source the answer actually read - a
figure quoted from a clause the answer cites is grounded by that clause, and a
figure from nowhere is a hard failure that no amount of re-retrieval can rescue.

**Claims go to a cheap model, and are graded against evidence, not strings.**
Three times in this build a substring filter over evidence flagged a *correct*
answer:

  GS-019  forbids asserting "pickup did not occur"; KI-211's instruction reads
          "Before telling a customer that a pickup did not occur, verify the
          carrier status" - the forbidden sentence as a prohibition of itself
  GS-027  forbids leaking "waive" to Beacon; SOP v4 §1, general policy every
          customer may read, says "unless a customer agreement explicitly
          waives the cancellation fee"
  M7      a test asserting "0 minutes" never appears, which "30 minutes"
          contains

So a forbidden assertion is compared against the *extracted claims*, never
against the prose.

**Sufficiency is structural.** An LLM grading its own evidence inside a repair
loop rationalises: given three attempts it quietly lowers its bar. Whether a
governing clause exists is a boolean, and it is checked before the extractor is
consulted. The model may declare evidence insufficient; it is never the thing
that declares it sufficient.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol

from src.agent.facts import FactBlock

logger = logging.getLogger(__name__)

#: How much of a claim's wording must appear in a source for it to count as
#: supported. Deliberately generous: the check is "is this claim about
#: something the evidence says", and a paraphrase of a clause is still grounded
#: in it. Inventions fail this easily because they share almost no vocabulary
#: with anything retrieved.
SUPPORT_RATIO: Final = 0.55

_IDENTIFIER: Final = re.compile(r"\b(?:ORD|TKT|ACCT|KI)-\d+\b|§\s*[\d.]+|\bv\d+\b", re.IGNORECASE)
_NUMBER: Final = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTED: Final = re.compile(r"[\"\u201c\u2018]([^\"\u201d\u2019]*)[\"\u201d\u2019]")
_WORD: Final = re.compile(r"[a-z0-9]+")

#: Words that carry no evidential weight, so overlap on them means nothing.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "does",
        "do",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "should",
        "than",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "under",
        "was",
        "were",
        "what",
        "when",
        "which",
        "will",
        "with",
        "would",
        "you",
        "your",
        "our",
        "we",
        "us",
    ]
)


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    #: The gate could not run. Never treated as a pass: a gate that reads its
    #: own outage as a clean bill of health is worse than no gate.
    UNCHECKED = "unchecked"
    #: No citable clause governs the topic. Structural, decided before the
    #: model is asked anything.
    NO_BASIS = "no_basis"


@dataclass(frozen=True, slots=True)
class Claim:
    text: str


@dataclass(frozen=True, slots=True)
class Failure:
    claim: Claim
    reason: str


class ClaimExtractor(Protocol):
    def extract(self, prose: str) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class GateOutcome:
    verdict: Verdict
    prose: str
    claims: tuple[Claim, ...] = ()
    failures: tuple[Failure, ...] = ()
    invented_figures: tuple[float, ...] = ()

    @property
    def repairable(self) -> bool:
        """Whether another retrieval could plausibly fix this.

        An unsupported claim might be: the evidence for it may exist and not
        have been fetched. An invented figure cannot be - a number the sources
        do not contain will not be found by searching for it, and trying looks
        like diligence while being a way to spend budget on a foregone
        conclusion.
        """
        return self.verdict is Verdict.FAILED and not self.invented_figures

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "claims": [c.text for c in self.claims],
            "failures": [{"claim": f.claim.text, "reason": f.reason} for f in self.failures],
            "invented_figures": list(self.invented_figures),
        }


def ground(
    prose: str,
    *,
    block: FactBlock,
    sources: Mapping[str, str],
    extractor: ClaimExtractor,
) -> GateOutcome:
    """Grade a draft against the evidence it was supposed to be written from."""
    # Structural first, and before the extractor is consulted. If nothing
    # governs the topic there is no answer to grade, only prose to discard.
    if block.is_empty and not sources:
        return GateOutcome(verdict=Verdict.NO_BASIS, prose=prose)

    invented = check_figures(prose, block, sources)

    try:
        extracted = extractor.extract(prose)
    except Exception as exc:  # a gate outage is not a pass
        logger.warning("claim extraction failed: %s", exc)
        return GateOutcome(verdict=Verdict.UNCHECKED, prose=prose, invented_figures=invented)

    if not extracted:
        logger.warning("claim extraction returned nothing for %d characters of prose", len(prose))
        return GateOutcome(verdict=Verdict.UNCHECKED, prose=prose, invented_figures=invented)

    claims = tuple(Claim(text) for text in extracted)
    supported_by = _evidence_text(block, sources)
    failures = [
        Failure(claim, "no source supports this")
        for claim in claims
        if not _supported(claim.text, supported_by)
    ]
    verdict = Verdict.FAILED if (failures or invented) else Verdict.PASSED
    return GateOutcome(
        verdict=verdict,
        prose=prose,
        claims=claims,
        failures=tuple(failures),
        invented_figures=invented,
    )


def check_figures(prose: str, block: FactBlock, sources: Mapping[str, str]) -> tuple[float, ...]:
    """Figures in the prose that no evidence contains.

    Allowed: anything in the fact block, and anything in the text of a source
    the answer read. The second is not a loophole - a number quoted from a
    clause the answer cites is grounded by that clause, and refusing it would
    fail every answer that explains what rule it is applying.
    """
    allowed = set(block.figures)
    for text in sources.values():
        allowed |= _figures_in(text)
    return tuple(sorted(f for f in unquoted_figures(prose) if f not in allowed))


def unquoted_figures(prose: str) -> set[float]:
    """Figures the model asserted in its own voice.

    Text inside quotation marks is a verbatim citation, and was checked when the
    clause entered the evidence set. Re-checking it here would flag an answer
    for accurately quoting its source.
    """
    return _figures_in(_QUOTED.sub(" ", prose))


# -- internals --------------------------------------------------------------


def _figures_in(text: str) -> set[float]:
    stripped = _IDENTIFIER.sub(" ", text)
    return {float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(stripped)}


def _evidence_text(block: FactBlock, sources: Mapping[str, str]) -> str:
    return " ".join([block.render(), *sources.values()]).lower()


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _supported(claim: str, evidence: str) -> bool:
    """Does the evidence say something about this?

    Content-word overlap rather than a similarity score over whole sentences:
    a claim and the clause it paraphrases share their nouns and numbers and
    almost nothing else, and stopword overlap would make every sentence look
    supported.
    """
    words = _content_words(claim)
    if not words:
        return True  # nothing asserted, nothing to support
    evidence_words = _content_words(evidence)
    hits = sum(1 for w in words if w in evidence_words or _close(w, evidence_words))
    return hits / len(words) >= SUPPORT_RATIO


def _close(word: str, vocabulary: set[str]) -> bool:
    """Tolerate inflection: 'waives' against 'waive', 'fees' against 'fee'."""
    return bool(difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.85))


# There is deliberately no `asserts(claims, forbidden)` helper here.
#
# The golden set's `must_not_assert` needs one - GS-019 forbids telling a
# customer their parcel was not collected - and every deterministic version of
# it tried during M7 flagged the *correct* answer. Content-word containment is
# the obvious approach and it reports
#
#     "the carrier status is verified before saying a pickup did not occur"
#
# as an assertion that the pickup did not occur, because the words are all
# there. That is the same failure this module's docstring lists three times,
# reappearing one level up.
#
# Whether a sentence asserts something or forbids asserting it is a semantic
# judgement, so it belongs in the LLM-judged evaluation layer (D21 layer 3,
# M11) rather than in a Python helper that looks sound and is not. What M7 can
# check about GS-019 deterministically - that the conflict is surfaced, that
# neither side is chosen, that the instruction survives verbatim - it does.
