"""PDF text extraction and normalisation.

pypdf renders the supplied files with two artifacts that defeat naive parsing.
Every space is doubled, and a wrapped line is emitted one word per line:

    charge\\n \\nINR\\n \\n250\\n \\nunless\\n \\na\\n \\ncustomer

Neither a clause segmenter nor a number extractor can work with that, so
normalisation is a real pipeline stage rather than tidying. It is also
idempotent: ingest runs it once, but the committed clause text must not depend
on how many times it ran.

Everything here is deterministic. The index is committed (D10), so a build must
produce byte-identical output on any machine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: Word-per-line runs: a newline, **at least one** blank, another newline.
#:
#: The quantifier is `+` rather than `*` deliberately, and that is what makes
#: normalisation idempotent. pypdf separates word runs with `\n \n`, never a
#: bare `\n\n` - verified across all six files - while the paragraph breaks
#: this module *inserts* are bare. So a second pass leaves its own output
#: alone instead of collapsing the block structure it just built.
_WORD_RUN: Final = re.compile(r"\n[ \t]+\n")
_HORIZONTAL_SPACE: Final = re.compile(r"[ \t]+")
_AROUND_NEWLINE: Final = re.compile(r" *\n *")
_BULLET: Final = re.compile(r"\s*[●•▪·]\s*")

#: A numbered heading that begins a line.
_HEADING_AT_LINE_START: Final = re.compile(r"\n(?=\d{1,2}\.[ \t]+[A-Z])")

#: A numbered heading that follows a sentence on the same visual line, as in
#: "...the standard return-to-origin process applies. 3. Service credits".
#: The lookbehind on sentence punctuation is what stops "INR 5,000. Unless"
#: from being read as a heading.
_HEADING_MID_LINE: Final = re.compile(r"(?<=[.:])[ \t]+(?=\d{1,2}\.[ \t]+[A-Z])")

#: Known-issue entries are separately citable authorities - detection matches a
#: ticket against one issue, not against a section - so each starts its own
#: block. They appear both mid-line after a sentence and at a line start
#: directly under their section heading.
_KNOWN_ISSUE: Final = re.compile(r"(?<=[.:])[ \t]+(?=KI-\d+\b)")
_KNOWN_ISSUE_AT_LINE_START: Final = re.compile(r"\n(?=KI-\d+\b)")

_BLANK_RUN: Final = re.compile(r"\n{3,}")

#: Lines that are structure rather than a wrapped continuation: a numbered
#: heading, a known-issue entry, or a response-target table row.
_STRUCTURAL_LINE: Final = re.compile(
    r"^(?:\d{1,2}\.\s+[A-Z]|KI-\d+\b|(?:Plan|Enterprise|Growth|Standard)\s)"
)

_SOFT_HYPHEN: Final = "­"
_NBSP: Final = " "
_NARROW_NBSP: Final = " "


def extract_text(path: Path | str) -> str:
    """Raw text of every page, joined. Raises if the file is absent."""
    from pypdf import PdfReader

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"source document not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def normalise(raw: str) -> str:
    """Repair extraction artifacts and impose block structure.

    Order matters. Word runs are rejoined before spaces are collapsed, and
    headings are split only after the text is on predictable lines.
    """
    text = raw.replace(_NBSP, " ").replace(_NARROW_NBSP, " ").replace(_SOFT_HYPHEN, "")
    text = _WORD_RUN.sub(" ", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _AROUND_NEWLINE.sub("\n", text)
    text = _BULLET.sub("\n- ", text)
    text = _HEADING_MID_LINE.sub("\n\n", text)
    text = _KNOWN_ISSUE.sub("\n\n", text)
    text = _KNOWN_ISSUE_AT_LINE_START.sub("\n\n", text)
    text = _HEADING_AT_LINE_START.sub("\n\n", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def unwrap(block: str) -> str:
    """Join the PDF's soft line wraps inside one block.

    A page break in the middle of "After 30 minutes,\\ncharge INR 250" is a
    typesetting artifact, not structure. Leaving it in means every downstream
    pattern has to spell `\\s*` where it means a space, and quoted clause text
    carries breaks the author never wrote. Bullets, headings and table rows
    keep their own lines because those breaks are real.

    Called by the clause parser on a clause body, deliberately not by
    `normalise`. Unwrapping the whole document first would merge the header
    into the body and destroy the boundary the parser needs to find.
    """
    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        starts_new_line = stripped.startswith("- ") or _STRUCTURAL_LINE.match(stripped)
        # A heading also *ends* its line: the sentence after it is body text,
        # not a continuation of the title.
        follows_heading = bool(lines) and bool(_STRUCTURAL_LINE.match(lines[-1]))
        if lines and not starts_new_line and not follows_heading:
            lines[-1] = f"{lines[-1]} {stripped}"
        else:
            lines.append(stripped)
    return "\n".join(lines)


def load_document_text(path: Path | str) -> str:
    """Extract and normalise in one step."""
    return normalise(extract_text(path))
