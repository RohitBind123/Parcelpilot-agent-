"""The fact block (D15a): the part of an answer the model cannot edit.

Python renders every figure a user will see, from evidence the tools minted. The
model writes the sentences around it and is checked against it afterwards. That
is the difference between a confidently wrong number being unlikely and being
structurally impossible.

Two things this module is careful about.

**Absence.** A block that prints "INR 0" where it means "nobody recorded a
price" has laundered a missing value into a fact, and the prose written around
it will be fluent and wrong. Null renders as unknown, everywhere.

**The delta.** "Overridden: SOP v4 §1" tells a reader nothing. What makes an
override worth surfacing is what the losing clause would have said, so the row
carries it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

#: Digits that identify rather than count. Clause references, document
#: versions, record ids and severity labels all contain them and assert no
#: quantity. Shared with the grounding gate, which must draw the same line -
#: two copies of this pattern would drift, and the drift would show up as a
#: correct answer being failed for a figure nobody stated.
#:
#: The general rule is a letter immediately followed by digits: P1, v3, KI-211.
#: A real quantity always has a space or a symbol between them ("INR 250",
#: "30 minutes"), so nothing countable is caught.
IDENTIFIER: Final = re.compile(
    r"\b[A-Za-z]{1,4}-\d+(?:\.\d+)*\b"  # ORD-1001, TKT-504, KI-211
    r"|\b[A-Za-z]{1,2}\d+\b"  # P1, v3
    r"|§\s*[\d.]+",  # §2, §3.1
    re.IGNORECASE,
)
#: A quantity and the unit that gives it meaning. The unit matters: Policy v3
#: §3 says Enterprise P1 is "30 minutes, 24x7" and Standard P2 is "1 business
#: day", so the bare number 1 is grounded by that grid - and "the target is
#: 1 hour", the deprecated v2 answer GS-017 exists to catch, would inherit its
#: support from a row about days. Pairing the value with its unit is what makes
#: the check mean what it appears to mean.
_QUANTITY: Final = re.compile(
    r"(?P<currency>INR|Rs\.?|₹)?\s*(?P<value>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?:business\s+)?(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|"
    r"months?|years?|rows?|%|percent))?",
    re.IGNORECASE,
)

_UNIT_ALIASES: Final = {
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
    "month": "months",
    "year": "years",
    "row": "rows",
    "percent": "%",
}


@dataclass(frozen=True, slots=True)
class FactRow:
    label: str
    value: str

    def to_payload(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True, slots=True)
class FactBlock:
    rows: tuple[FactRow, ...] = ()
    #: Every quantity the prose may state without further support, paired with
    #: its unit. See `_QUANTITY` for why the unit is not optional.
    figures: frozenset[tuple[float, str | None]] = field(default_factory=frozenset)
    #: Clause ids the prose may cite. Excluded clauses are deliberately absent.
    citable: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def render(self) -> str:
        """The block as the model sees it, above its own draft."""
        if not self.rows:
            return ""
        width = max(len(r.label) for r in self.rows)
        return "\n".join(f"{r.label:<{width}}  {r.value}" for r in self.rows)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rows": [r.to_payload() for r in self.rows],
            "figures": sorted((v, u or "") for v, u in self.figures),
            "citable": sorted(self.citable),
        }


def compose(
    *,
    calculation: Mapping[str, Any] | None = None,
    resolution: Mapping[str, Any] | None = None,
    conflicts: Mapping[str, Any] | None = None,
    sla: Mapping[str, Any] | None = None,
) -> FactBlock:
    """Render the block from whatever evidence this answer actually has.

    Every argument is optional because not every question has a number. GS-016
    asks about a response target in the abstract; GS-018 compares two policies.
    A block with a Governing row and no Amount is a complete answer to those.
    """
    rows: list[FactRow] = []
    figures: set[float] = set()
    citable: set[str] = set()

    if calculation:
        rows.extend(_verdict_and_amount(calculation, figures))
    if sla:
        rows.extend(_sla_rows(sla, figures))
    if resolution:
        rows.extend(_authority_rows(resolution, figures, citable))
    if calculation or sla:
        basis = _basis(calculation or {}, sla or {})
        if basis:
            rows.append(FactRow("Basis", basis))
            figures.update(figures_in(basis))
    if conflicts:
        caution = _caution(conflicts)
        if caution:
            rows.append(FactRow("Caution", caution))
            figures.update(figures_in(caution))

    return FactBlock(rows=tuple(rows), figures=frozenset(figures), citable=frozenset(citable))


# -- rows -------------------------------------------------------------------


def _verdict_and_amount(calc: Mapping[str, Any], figures: set[float]) -> list[FactRow]:
    rows = [FactRow("Verdict", _verdict(calc))]
    amount = _amount(calc)
    if amount is not None:
        rows.append(FactRow("Amount", amount))
        figures.update(figures_in(amount))
    return rows


def _verdict(calc: Mapping[str, Any]) -> str:
    if "cancellable" in calc:
        if not calc.get("cancellable"):
            return f"Not cancellable in status {calc.get('order_status')}"
        fee = calc.get("fee_inr")
        if fee == 0:
            return "Cancellable, no cancellation fee"
        if fee is None:
            return "Cancellable; the fee could not be determined"
        return "Cancellable, cancellation fee applies"
    if "eligible" in calc:
        return (
            "Eligible for a service credit"
            if calc.get("eligible")
            else "Not eligible for a service credit"
        )
    return "See below"


def _amount(calc: Mapping[str, Any]) -> str | None:
    """Money, or the honest absence of it.

    `None` is not zero. A tier with no recorded price is unknown, and printing
    INR 0 turns a gap in the data into a promise.
    """
    for key in ("fee_inr", "credit_inr"):
        if key not in calc:
            continue
        value = calc[key]
        if value is None:
            formula = calc.get("amount_formula")
            return f"unknown{f' ({formula})' if formula else ''}"
        return f"INR {value:g}"
    return None


def _sla_rows(sla: Mapping[str, Any], figures: set[float]) -> list[FactRow]:
    rows = []
    severity = sla.get("severity")
    rows.append(
        FactRow("Severity", f"{severity}{' (inferred)' if sla.get('severity_inferred') else ''}")
        if severity
        else FactRow("Severity", "undetermined")
    )
    if sla.get("target"):
        rows.append(FactRow("Target", f"{sla['target']} ({sla.get('clock_type', 'calendar')})"))
        figures.update(figures_in(str(sla["target"])))
    if sla.get("due_at"):
        rows.append(FactRow("Due", str(sla["due_at"])))
    # A4: there is no first_response_at column, so a breach is not measurable.
    if sla.get("measurable") is False:
        rows.append(FactRow("Measurable", "no - ParcelPilot does not record first-reply times"))
    return rows


def _authority_rows(
    resolution: Mapping[str, Any], figures: set[float], citable: set[str]
) -> list[FactRow]:
    rows = []
    governing = resolution.get("governing")
    if governing:
        citable.add(governing["clause_id"])
        rows.append(FactRow("Governing", _describe(governing)))
        figures.update(_params_figures(governing))

    for label, key in (("Overridden", "overridden"), ("Deferred", "deferred")):
        entries = resolution.get(key) or []
        if not entries:
            continue
        for entry in entries:
            citable.add(entry["clause_id"])
            figures.update(_params_figures(entry))
        rows.append(FactRow(label, "; ".join(_describe(e, with_params=True) for e in entries)))

    supporting = resolution.get("supporting") or []
    if supporting:
        for entry in supporting:
            citable.add(entry["clause_id"])
            figures.update(_params_figures(entry))
        rows.append(FactRow("Supporting", "; ".join(_describe(e) for e in supporting)))

    excluded = resolution.get("excluded") or []
    if excluded:
        # Named, and deliberately not added to `citable`. A reader should see
        # that the deprecated policy was considered and set aside.
        rows.append(
            FactRow(
                "Excluded",
                "; ".join(f"{e['citation']} ({e.get('reason', 'excluded')})" for e in excluded),
            )
        )
    return rows


def _describe(entry: Mapping[str, Any], *, with_params: bool = False) -> str:
    kind = _kind(entry)
    described = f"{entry['citation']} (tier {entry['tier']}, {kind})"
    if not with_params:
        return described
    stated = _state_params(entry.get("params") or {})
    return f"{described}{f' - {stated}' if stated else ''}"


def _kind(entry: Mapping[str, Any]) -> str:
    return {1: "agreement", 2: "policy", 3: "product doc", 4: "deprecated", 5: "historical"}.get(
        int(entry.get("tier", 0)), "source"
    )


def _state_params(params: Mapping[str, Any]) -> str:
    """What a losing clause would have said, in the reader's units."""
    said = []
    for key, value in sorted(params.items()):
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key.endswith("_inr"):
            said.append(f"INR {value:g}")
        elif key.endswith("_minutes"):
            said.append(f"{value:g} minutes")
        elif key.endswith("_hours"):
            said.append(f"{value:g} hours")
        elif key.endswith("_rows"):
            said.append(f"{value:g} rows")
        else:
            said.append(f"{key.replace('_', ' ')} {value:g}")
    return ", ".join(said)


def _basis(calc: Mapping[str, Any], sla: Mapping[str, Any]) -> str:
    parts = []
    if calc.get("minutes_since_booking") is not None:
        parts.append(f"{calc['minutes_since_booking']} minutes since booking")
    if calc.get("delay_hours") is not None:
        parts.append(f"{calc['delay_hours']:g} hours past the pickup window")
    if calc.get("order_status"):
        parts.append(f"status {calc['order_status']}")
    if sla.get("elapsed_minutes") is not None:
        parts.append(f"{sla['elapsed_minutes']} minutes elapsed")
    return ", ".join(parts)


def _caution(report: Mapping[str, Any]) -> str:
    """Conflicts, in the corpus's own words.

    The instruction is carried verbatim: KI-211 does not merely describe the
    webhook lag, it says what to do about it, and a model paraphrasing can drop
    the part that matters.
    """
    said = []
    for conflict in report.get("conflicts") or ():
        line = conflict.get("detail", "")
        if conflict.get("inference_note"):
            line += f" (the link is inferred, not recorded: {conflict['inference_note']})"
        if conflict.get("instruction"):
            line += f" {conflict['instruction']}"
        said.append(line)
    return " ".join(said)


# -- figures ----------------------------------------------------------------


def _params_figures(entry: Mapping[str, Any]) -> set[tuple[float, str | None]]:
    """Quantities from a clause's typed params, with the unit its key implies.

    `fee_after_window_inr` is 250 rupees, not the bare number 250, and the key
    already says so - which is more reliable than re-reading it out of prose.
    """
    found = set()
    for key, value in (entry.get("params") or {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        unit = next((u for suffix, u in _KEY_UNITS.items() if key.endswith(suffix)), None)
        found.add((float(value), unit))
    return found


_KEY_UNITS: Final = {
    "_inr": "inr",
    "_minutes": "minutes",
    "_hours": "hours",
    "_days": "days",
    "_rows": "rows",
}


def figures_in(text: str) -> set[tuple[float, str | None]]:
    """Quantities in text, excluding anything that identifies rather than counts."""
    stripped = IDENTIFIER.sub(" ", text)
    found = set()
    for match in _QUANTITY.finditer(stripped):
        raw_unit = (match.group("unit") or "").lower().rstrip()
        unit = _UNIT_ALIASES.get(raw_unit, raw_unit) or None
        if match.group("currency"):
            unit = "inr"
        found.add((float(match.group("value").replace(",", "")), unit))
    return found
