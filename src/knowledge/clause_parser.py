"""PDF to typed clauses: the authority spine.

Precedence in this system is a sort on `tier`, so tier had better be a reading
of the corpus rather than an assertion about it. Everything structural here -
tier, account scope, status, effective dates - comes out of each document's own
header block. A table keyed on filename would give the same answers today and
be wrong the moment a seventh document arrives.

Segmentation handles three shapes present in the pack:

  numbered sections   "1. Order cancellation"      -> §1
  known-issue entries "KI-208 - Bulk Upload ..."   -> KI-208
  unnumbered body     Support Policy v2            -> §-

The third exists because the deprecated policy has no section numbers and still
has to be a citable unit, so "what changed in v3?" can quote it.

Parsing is deterministic end to end. The registry is committed (D10), so two
builds on two machines must produce identical clauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from src.knowledge.pdftext import load_document_text, unwrap
from src.knowledge.sources import SOURCE_FILES, SourceFile
from src.knowledge.topics import tag

#: Header keys that appear across the six documents.
_HEADER_KEYS: Final[tuple[str, ...]] = (
    "Status",
    "Effective",
    "Updated",
    "Supersedes",
    "Superseded by",
    "Account",
    "Customer",
    "Plan",
    "Term",
)
_KEY_ALTERNATION: Final = "|".join(re.escape(k) for k in _HEADER_KEYS)
_HEADER_FIELD: Final = re.compile(
    rf"\b({_KEY_ALTERNATION})\s*:\s*(.*?)(?=\s+\b(?:{_KEY_ALTERNATION})\s*:|$)",
    re.DOTALL,
)
_FIRST_KEY: Final = re.compile(rf"\b(?:{_KEY_ALTERNATION})\s*:")

_NUMBERED_HEADING: Final = re.compile(r"^(\d{1,2})\.\s+(.+?)\s*$", re.MULTILINE)
# The separator after a known-issue id is a hyphen, an en dash or a colon.
_KI_HEADING: Final = re.compile(r"^(KI-\d+)\s*[-\u2013:]\s*(.+?)\s*$", re.MULTILINE)
_ACCOUNT_ID: Final = re.compile(r"\bACCT-\d{3}\b")
_DATE: Final = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
_TERM_RANGE: Final = re.compile(r"(.+?)\s+to\s+(.+)", re.DOTALL)

#: Tier is decided by what the header says, in this order.
_POLICY_TITLE: Final = re.compile(r"\b(policy|SOP)\b", re.IGNORECASE)
_GUIDE_TITLE: Final = re.compile(r"\b(guide|operations|handbook)\b", re.IGNORECASE)

TIER_AGREEMENT: Final = 1
TIER_POLICY: Final = 2
TIER_GUIDE: Final = 3
TIER_DEPRECATED: Final = 4

#: A block whose body adds this little beyond its heading is a stub - the
#: leftover of a section whose content was split out into its own clauses.
_MIN_BODY_CHARS: Final = 10


class ClauseParseError(RuntimeError):
    """A document did not look the way the parser expects."""


@dataclass(frozen=True, slots=True)
class Clause:
    """One citable unit of authority."""

    clause_id: str
    doc_id: str
    doc_title: str
    clause_ref: str
    title: str
    tier: int
    account_id: str | None
    status: str
    effective_from: date | None
    effective_to: date | None
    superseded_by: str | None
    topics: tuple[str, ...]
    text: str

    @property
    def is_current(self) -> bool:
        return self.status != "DEPRECATED"

    @property
    def citation(self) -> str:
        """How this clause is named to a user."""
        return f"{self.doc_title} {self.clause_ref}".replace(" §-", "")


@dataclass(frozen=True, slots=True)
class Document:
    """One source document and the clauses parsed from it."""

    doc_id: str
    title: str
    tier: int
    account_id: str | None
    status: str
    effective_from: date | None
    effective_to: date | None
    superseded_by: str | None
    clauses: tuple[Clause, ...]

    @property
    def is_current(self) -> bool:
        return self.status != "DEPRECATED"


def parse_all() -> tuple[Document, ...]:
    """Parse every supplied document."""
    return tuple(parse_document(source) for source in SOURCE_FILES)


def parse_document(source: SourceFile) -> Document:
    text = load_document_text(source.path)
    header, body = _split_header(text)
    fields = _header_fields(header)

    title = _title(header)
    account_id = _account_id(fields, header)
    status = _status(fields)
    effective_from, effective_to = _effective_dates(fields)

    document_scope = {
        "doc_id": source.doc_id,
        "doc_title": title,
        "tier": _tier(title, status, account_id),
        "account_id": account_id,
        "status": status,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "superseded_by": fields.get("Superseded by"),
    }

    clauses = _segment(body, document_scope)
    if not clauses:
        raise ClauseParseError(f"{source.doc_id} produced no clauses")

    return Document(
        clauses=clauses,
        title=title,
        **{k: v for k, v in document_scope.items() if k != "doc_title"},
    )


# -- header -----------------------------------------------------------------


def _split_header(text: str) -> tuple[str, str]:
    """Separate the header block from the body.

    Normalisation puts a blank line before every numbered heading, so for most
    documents the header is simply the first block. Policy v2 has no numbered
    headings and therefore no blank line, so its header is taken to be the
    leading lines that carry header keys.
    """
    blocks = text.split("\n\n")
    if len(blocks) > 1:
        return blocks[0], "\n\n".join(blocks[1:])

    lines = text.splitlines()
    cut = 1  # the title line always belongs to the header
    while cut < len(lines) and _FIRST_KEY.search(lines[cut]):
        cut += 1
    return "\n".join(lines[:cut]), "\n".join(lines[cut:])


def _header_fields(header: str) -> dict[str, str]:
    flat = " ".join(header.split())
    return {key: value.strip() for key, value in _HEADER_FIELD.findall(flat)}


def _title(header: str) -> str:
    """The document title, as a citation will show it.

    The Northstar agreement wraps its title across two lines, leaving the word
    "Agreement" stranded at the start of the header line. Anything before the
    first header key on that line belongs to the title.
    """
    lines = header.splitlines()
    title = lines[0].strip()
    if len(lines) > 1:
        match = _FIRST_KEY.search(lines[1])
        stranded = (lines[1][: match.start()] if match else lines[1]).strip()
        if stranded:
            title = f"{title} {stranded}"
    return title


def _account_id(fields: dict[str, str], header: str) -> str | None:
    """The account this document is private to, if any.

    Presence of an account is what makes a document a signed agreement, which
    is simultaneously its ACL predicate and its tier.
    """
    match = _ACCOUNT_ID.search(fields.get("Account", "")) or _ACCOUNT_ID.search(header)
    return match.group(0) if match else None


def _status(fields: dict[str, str]) -> str:
    """First word of the declared status: CURRENT, DEPRECATED or ACTIVE."""
    raw = fields.get("Status", "").strip()
    return raw.split()[0].upper().rstrip("-") if raw else "UNKNOWN"


def _effective_dates(fields: dict[str, str]) -> tuple[date | None, date | None]:
    if term := fields.get("Term"):
        if match := _TERM_RANGE.match(term):
            return _parse_date(match.group(1)), _parse_date(match.group(2))
        return _parse_date(term), None
    for key in ("Effective", "Updated"):
        if value := fields.get(key):
            return _parse_date(value), None
    return None, None


def _parse_date(text: str) -> date | None:
    match = _DATE.search(text)
    if not match:
        return None
    day, month, year = match.groups()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{day} {month} {year}", fmt).date()  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _tier(title: str, status: str, account_id: str | None) -> int:
    """Authority tier, from what the document says about itself.

    Order matters. A signed agreement outranks policy whatever it is called,
    and a deprecated document is demoted whatever else it is.
    """
    if account_id:
        return TIER_AGREEMENT
    if status == "DEPRECATED":
        return TIER_DEPRECATED
    if _GUIDE_TITLE.search(title):
        return TIER_GUIDE
    if _POLICY_TITLE.search(title):
        return TIER_POLICY
    return TIER_GUIDE


# -- segmentation -----------------------------------------------------------


def _segment(body: str, scope: dict) -> tuple[Clause, ...]:
    clauses: list[Clause] = []
    for block in (b.strip() for b in body.split("\n\n")):
        if not block:
            continue
        ref, title = _block_heading(block)
        if ref is None:
            continue
        # A section whose content was split into its own clauses leaves a
        # heading-only stub behind. Keeping it would only dilute retrieval.
        if len(block) - len(title) < _MIN_BODY_CHARS:
            continue
        clauses.append(_build(ref, title, _clause_text(block), scope))

    if not clauses and body.strip():
        # No numbered headings anywhere: the whole body is one citable unit.
        lines = body.strip().splitlines()
        clauses.append(_build("§-", lines[0].strip(), _clause_text(body.strip()), scope))

    return tuple(clauses)


def _clause_text(block: str) -> str:
    """The clause as it will be quoted: heading intact, soft wraps joined."""
    return unwrap(block)


def _block_heading(block: str) -> tuple[str | None, str]:
    first_line = block.splitlines()[0]
    if match := _NUMBERED_HEADING.match(first_line):
        return f"§{match.group(1)}", match.group(2).strip()
    if match := _KI_HEADING.match(first_line):
        return match.group(1), _known_issue_title(match.group(2))
    return None, ""


def _known_issue_title(raw: str) -> str:
    """The subject of a known issue, without the prose that follows it.

    A known-issue heading and its body share one line, so the title has to be
    cut rather than taken to end of line: KI-176 reads "Address validation:
    Resolved 18 July 2026. Do not use this resolved issue..." and only the
    first two words name the issue. Titles are cited, so they stay short.
    """
    title = raw.split(":", 1)[0]
    return title.split(". ", 1)[0].strip()


def _build(ref: str, title: str, text: str, scope: dict) -> Clause:
    return Clause(
        clause_id=f"{scope['doc_id']}::{ref}",
        clause_ref=ref,
        title=title,
        text=text,
        topics=tag(text),
        **scope,
    )
