"""The six supplied documents, and the identity we give each one.

`doc_id` is derived from the filename with the ordering prefix stripped, so it
is stable across rebuilds without being a hand-maintained mapping. Citations
never show it - they show the parsed title - so a long identifier costs nothing
and a derived one cannot drift from the file it names.

Nothing about tier, account scope or status lives here. All of that is read
from each document's own header (`clause_parser`), because a tier table keyed
on filename is an assertion about the corpus rather than a reading of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
RAW_DIR: Final = REPO_ROOT / "data" / "raw"

_ORDER_PREFIX: Final = re.compile(r"^\d+_")


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One supplied PDF."""

    doc_id: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


def _source(filename: str) -> SourceFile:
    doc_id = _ORDER_PREFIX.sub("", Path(filename).stem).lower()
    return SourceFile(doc_id=doc_id, path=RAW_DIR / filename)


SOURCE_FILES: Final[tuple[SourceFile, ...]] = tuple(
    _source(name)
    for name in (
        "01_Support_Policy_v3_CURRENT.pdf",
        "02_Support_Policy_v2_DEPRECATED.pdf",
        "03_Cancellation_and_Service_Credit_SOP_v4.pdf",
        "04_Product_Operations_Guide_and_Known_Issues.pdf",
        "05_Northstar_Logistics_Enterprise_Agreement.pdf",
        "06_LumenWorks_Service_Agreement.pdf",
    )
)

_BY_ID: Final[dict[str, SourceFile]] = {s.doc_id: s for s in SOURCE_FILES}


def get_source(doc_id: str) -> SourceFile:
    try:
        return _BY_ID[doc_id]
    except KeyError as exc:
        raise LookupError(f"unknown document: {doc_id!r}") from exc
