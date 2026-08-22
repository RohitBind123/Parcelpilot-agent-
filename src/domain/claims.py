"""Typed claims read out of Tier 5 free text.

`tickets.historical_resolution` is the one place in the pack where a rule is
asserted in prose rather than in a clause, and Policy v3 section 1 says such
text is context only and may contain incorrect past guidance. Both of the two
recorded resolutions in the pack are in fact wrong, and they are there so that a
retrieval system with no tier discipline will repeat one of them.

Saying *how* a past answer was wrong - rather than only that its source is not
citable - needs the number it asserted and the topic it asserted it about. That
is all this module does.

The design bar is narrowness rather than recall, because the two failures are
not symmetric. A claim missed means the report says nothing about a sentence it
could not read, and the answer falls back to "historical tickets are context
only", which is true. A claim invented means the system contradicts a clause
that never disagreed with anything, and tells the customer a past answer was
wrong when it was right. So every pattern here requires the words that make the
sentence a statement about a rule: "supports N rows", not "N rows"; "a fee of
INR N", not "INR N".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: Digits with optional thousands separators. Stripped before int() - "3,000
#: rows" is three thousand, and reading it as 3 is the bug that made a currency
#: amount match a section reference during M2 retrieval.
_INT: Final = r"(\d[\d,]*)"


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion about a rule, lifted from prose.

    `param` names the key on a clause's `params` that would settle it, which is
    what lets a contradiction be checked arithmetically instead of by reading.
    """

    topic: str
    param: str
    value: int
    quote: str


#: (topic, param, pattern). Each pattern must carry the words that turn a number
#: into an assertion about policy; a bare amount or duration never matches.
_PATTERNS: Final[tuple[tuple[str, str, re.Pattern[str]], ...]] = (
    (
        "cancellation_fee",
        "fee_after_window_inr",
        re.compile(rf"INR\s*{_INT}\s+cancellation\s+fee", re.IGNORECASE),
    ),
    (
        "cancellation_fee",
        "fee_after_window_inr",
        re.compile(rf"cancellation\s+fee\s+of\s+INR\s*{_INT}", re.IGNORECASE),
    ),
    (
        "cancellation_window",
        "free_window_minutes",
        re.compile(rf"(?:applied|applies|charged)\s+after\s+{_INT}\s+minutes", re.IGNORECASE),
    ),
    (
        "bulk_upload_limit",
        "supported_rows",
        re.compile(rf"(?:only\s+)?supports?\s+(?:up\s+to\s+)?{_INT}\s+rows", re.IGNORECASE),
    ),
    (
        "bulk_upload_limit",
        "supported_rows",
        re.compile(rf"limit\s+(?:is|of)\s+{_INT}\s+rows", re.IGNORECASE),
    ),
)


def extract_claims(text: str | None) -> tuple[Claim, ...]:
    """Every rule this prose asserts, or an empty tuple.

    `None` is tolerated because the column is nullable and is NULL on every open
    ticket - the caller should not have to guard a read it cannot avoid making.
    """
    if not text:
        return ()

    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for topic, param, pattern in _PATTERNS:
        match = pattern.search(text)
        if match is None or (topic, param) in seen:
            continue
        seen.add((topic, param))
        claims.append(
            Claim(
                topic=topic,
                param=param,
                value=int(match.group(1).replace(",", "")),
                quote=match.group(0),
            )
        )
    return tuple(claims)
