"""First-response target status for a ticket.

Acceptance is GS-011 through GS-016.

`measurable` is always False, and that is the point of the module rather than a
caveat on it. The tickets table has no `first_response_at` column (findings
section 9), so a real breach cannot be measured - only elapsed time against a
target. Asserting a breach would be asserting data that does not exist, and it
is the kind of claim a support organisation gets held to.

The clock is the other difficulty. AS_OF is a Sunday, so a business-hours target
does not start until Monday 09:00 while a 24x7 target runs immediately. Two
tickets created ninety minutes apart on the same Sunday can therefore be one
past its target and the other not yet started.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal

from src.clock import add_business_days, add_business_hours, as_of, next_business_start
from src.domain.calculators.base import attribution, load_resolution, mint_result
from src.domain.calculators.errors import NoBasis
from src.domain.calculators.params import lookup
from src.domain.evidence import EvidenceKind, EvidenceStore, Handle
from src.domain.severity import SEVERITIES, SeverityVerdict

TOPIC: Final = "first_response_target"

Surface = Literal["ops", "customer"]

MEASURABILITY_NOTE: Final = (
    "ParcelPilot does not record when a ticket was first replied to, so this is "
    "elapsed time against the target, not a confirmed breach."
)


@dataclass(frozen=True, slots=True)
class SlaOutcome:
    ticket_id: str
    severity: str | None
    severity_basis_clause: str | None
    severity_confidence: float
    severity_inferred: bool
    target: str | None
    target_clause: str | None
    clock_type: str | None
    clock_starts_at: str | None
    due_at: str | None
    elapsed_minutes: int
    past_target_by_minutes: int | None
    #: Always False. See the module docstring - this is load-bearing.
    measurable: bool
    measurability_note: str
    escalate: bool
    governing_clause: str
    overridden_clauses: tuple[str, ...]
    deferred_clauses: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    calc_id: str | None = None

    @property
    def past_target(self) -> bool | None:
        return None if self.past_target_by_minutes is None else self.past_target_by_minutes > 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "severity": self.severity,
            "severity_basis_clause": self.severity_basis_clause,
            "severity_confidence": self.severity_confidence,
            "severity_inferred": self.severity_inferred,
            "target": self.target,
            "target_clause": self.target_clause,
            "clock_type": self.clock_type,
            "clock_starts_at": self.clock_starts_at,
            "due_at": self.due_at,
            "elapsed_minutes": self.elapsed_minutes,
            "past_target_by_minutes": self.past_target_by_minutes,
            "past_target": self.past_target,
            "measurable": self.measurable,
            "measurability_note": self.measurability_note,
            "escalate": self.escalate,
            "governing_clause": self.governing_clause,
            "overridden_clauses": list(self.overridden_clauses),
            "deferred_clauses": list(self.deferred_clauses),
            "warnings": list(self.warnings),
        }


def sla_first_response_status(
    store: EvidenceStore,
    *,
    snapshot_id: Handle | str,
    account_snapshot_id: Handle | str,
    resolution_id: Handle | str,
    severity: SeverityVerdict,
    surface: Surface = "ops",
) -> SlaOutcome:
    """Elapsed time against the first-response target for a ticket.

    Takes an account snapshot as well as a ticket one, because the default
    target grid is keyed by plan and the ticket does not carry it. Passing the
    plan as a bare string would be the one place in the chain where a caller
    could assert a fact without evidence behind it.
    """
    ticket = store.read(snapshot_id, expect=EvidenceKind.TICKET_SNAPSHOT)
    account = store.read(account_snapshot_id, expect=EvidenceKind.ACCOUNT_SNAPSHOT)
    resolution = load_resolution(
        store, resolution_id, topic=TOPIC, account_id=ticket.get("account_id")
    )

    if account.get("account_id") != ticket.get("account_id"):
        from src.domain.calculators.errors import WrongEvidence

        raise WrongEvidence("ticket and account snapshots describe different accounts")

    warnings: list[str] = []
    created = datetime.fromisoformat(ticket["created_at"])
    now = as_of()
    elapsed = int((now - created).total_seconds() // 60)

    effective, inferred, escalate = _apply_confidence(severity, surface, warnings)

    base = {
        "ticket_id": ticket["ticket_id"],
        "severity": effective,
        "severity_basis_clause": severity.basis_clause,
        "severity_confidence": severity.confidence,
        "severity_inferred": inferred,
        "elapsed_minutes": elapsed,
        "measurable": False,
        "measurability_note": MEASURABILITY_NOTE,
        "deferred_clauses": tuple(c.clause_id for c in resolution.deferred),
    }

    if effective is None:
        # D25 on the customer surface: quoting a target derived from a severity
        # we do not trust is a promise ParcelPilot may not keep.
        outcome = SlaOutcome(
            target=None,
            target_clause=None,
            clock_type=None,
            clock_starts_at=None,
            due_at=None,
            past_target_by_minutes=None,
            escalate=escalate,
            warnings=tuple(warnings),
            **base,
            **attribution(resolution, None),
        )
        return mint_result(store, outcome, snapshot_id, account_snapshot_id, resolution_id)

    target, source = _target(resolution, account.get("plan"), effective)
    clock_type, starts_at, due_at = _deadline(created, target)
    past_by = int((now - due_at).total_seconds() // 60)

    if clock_type == "calendar":
        warnings.append(
            f"the clause states {_describe(target)} without saying whether business "
            "hours apply; elapsed time is treated as continuous"
        )
    if starts_at != created:
        warnings.append(
            f"created outside business hours; the response clock starts {starts_at:%A %d %b %H:%M}"
        )

    outcome = SlaOutcome(
        target=_describe(target),
        target_clause=source,
        clock_type=clock_type,
        clock_starts_at=starts_at.isoformat(),
        due_at=due_at.isoformat(),
        past_target_by_minutes=past_by,
        escalate=escalate or past_by > 0,
        warnings=tuple(warnings),
        **base,
        **attribution(resolution, source),
    )
    return mint_result(store, outcome, snapshot_id, account_snapshot_id, resolution_id)


def _apply_confidence(
    severity: SeverityVerdict, surface: Surface, warnings: list[str]
) -> tuple[str | None, bool, bool]:
    """D25: the asymmetry is by surface, because the cost of being wrong is.

    Ops triage rounds up - an over-prioritised ticket costs an analyst two
    minutes, a missed P1 costs an outage. The customer surface declines to
    quote anything and escalates instead.
    """
    if severity.is_trusted:
        return severity.severity, not severity.deterministic, False

    if surface == "customer":
        warnings.append(
            "severity could not be determined with confidence, so no response target is quoted"
        )
        return None, True, True

    escalated = _more_severe(severity.severity)
    warnings.append(
        f"severity inferred with low confidence ({severity.confidence:.2f}); "
        f"triaged up to {escalated} for review"
    )
    return escalated, True, False


def _more_severe(severity: str | None) -> str:
    if severity not in SEVERITIES:
        return SEVERITIES[0]
    return SEVERITIES[max(0, SEVERITIES.index(severity) - 1)]


def _target(resolution, plan: str | None, severity: str) -> tuple[dict[str, Any], str | None]:
    """The target for this plan and severity, from whichever clause holds it.

    Agreement grids are keyed by severity alone; the default policy grid is
    keyed by plan and then severity. Both shapes are read here rather than
    normalised at ingest, because the difference is real: an agreement applies
    to one account regardless of what plan it is on.
    """
    found = lookup(resolution, "targets")
    if found.missing:
        raise NoBasis(f"no response targets in any clause resolved for {TOPIC!r}")

    grid = found.value
    if severity not in grid:
        if plan is None:
            raise NoBasis("response targets are keyed by plan, but no plan is on the snapshot")
        if plan not in grid:
            raise NoBasis(f"no response targets for plan {plan!r}")
        grid = grid[plan]

    if severity not in grid:
        raise NoBasis(f"no response target for severity {severity!r}")
    return grid[severity], found.source


def _deadline(created: datetime, target: dict[str, Any]) -> tuple[str, datetime, datetime]:
    amount, unit = target["amount"], target["unit"]
    if target.get("always_on"):
        return "24x7", created, created + _delta(amount, unit)
    if target.get("business"):
        starts = next_business_start(created)
        due = (
            add_business_hours(created, amount)
            if unit == "hours"
            else add_business_days(created, amount)
        )
        return "business_hours", starts, due
    return "calendar", created, created + _delta(amount, unit)


def _delta(amount: int, unit: str) -> timedelta:
    if unit == "minutes":
        return timedelta(minutes=amount)
    if unit == "hours":
        return timedelta(hours=amount)
    if unit == "days":
        return timedelta(days=amount)
    raise NoBasis(f"unknown target unit {unit!r}")


def _describe(target: dict[str, Any]) -> str:
    amount, unit = target["amount"], target["unit"]
    unit = unit.rstrip("s") if amount == 1 else unit
    business = " business" if target.get("business") else ""
    always = ", 24x7" if target.get("always_on") else ""
    return f"{amount}{business} {unit}{always}"
