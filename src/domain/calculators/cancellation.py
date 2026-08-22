"""What it costs to cancel a shipment, and whether it can be cancelled at all.

Acceptance is golden-set GS-001 to GS-006. The order of operations is the whole
design: status is decided before money. Northstar's waiver applies to BOOKED
shipments only, and a calculator that applies the waiver first and checks status
second happily reports INR 0 for a parcel already in transit.

The second load-bearing detail is that a shipment which cannot be cancelled
reports `fee_inr: None`, not 0. "Cancelled free of charge" and "cannot be
cancelled" are opposite answers, and zero renders as the wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from src.domain.calculators.base import attribution, load_resolution, mint_result
from src.domain.calculators.errors import NoBasis
from src.domain.calculators.params import lookup
from src.domain.evidence import EvidenceKind, EvidenceStore, Handle
from src.domain.resolver import ClauseRef, PolicyResolution

TOPIC: Final = "cancellation_fee"

#: What a status rule means. Read from `status_rules` in the registry; this maps
#: the registry's vocabulary onto behaviour, and an unknown value is refused
#: rather than guessed.
_FREE: Final = "free"
_CONDITIONAL: Final = "conditional"
_RETURN_TO_ORIGIN: Final = "return_to_origin"
_NOT_CANCELLABLE: Final = "not_cancellable"


@dataclass(frozen=True, slots=True)
class CancellationOutcome:
    """Everything the fact block needs, with a clause behind every number."""

    order_id: str
    order_status: str
    cancellable: bool
    #: None when no cancellation can happen. Never 0 in that case.
    fee_inr: float | None
    fee_basis: str | None
    fee_source: str | None
    minutes_since_booking: int | None
    free_window_minutes: int | None
    window_source: str | None
    within_free_window: bool | None
    status_rule: str | None
    status_rule_source: str | None
    next_action: str | None
    governing_clause: str
    overridden_clauses: tuple[str, ...]
    deferred_clauses: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    calc_id: str | None = None

    @property
    def is_override(self) -> bool:
        return bool(self.overridden_clauses)

    def to_payload(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "order_status": self.order_status,
            "cancellable": self.cancellable,
            "fee_inr": self.fee_inr,
            "fee_basis": self.fee_basis,
            "fee_source": self.fee_source,
            "minutes_since_booking": self.minutes_since_booking,
            "free_window_minutes": self.free_window_minutes,
            "window_source": self.window_source,
            "within_free_window": self.within_free_window,
            "status_rule": self.status_rule,
            "status_rule_source": self.status_rule_source,
            "next_action": self.next_action,
            "governing_clause": self.governing_clause,
            "overridden_clauses": list(self.overridden_clauses),
            "deferred_clauses": list(self.deferred_clauses),
            "is_override": self.is_override,
            "warnings": list(self.warnings),
        }


def compute_cancellation_fee(
    store: EvidenceStore,
    *,
    snapshot_id: Handle | str,
    resolution_id: Handle | str,
) -> CancellationOutcome:
    """The fee for cancelling the shipment in `snapshot_id`.

    Takes handles, not an order id. That is the constraint that forces the
    model through the resolver: there is no argument here it could satisfy by
    naming an order.
    """
    order = store.read(snapshot_id, expect=EvidenceKind.ORDER_SNAPSHOT)
    resolution = load_resolution(
        store, resolution_id, topic=TOPIC, account_id=order.get("account_id")
    )

    status = order["status"]
    rule = lookup(resolution, "status_rules")
    status_rules = rule.value if isinstance(rule.value, dict) else {}
    disposition = status_rules.get(status)

    warnings: list[str] = []
    if disposition is None:
        # An unrecognised status is not a licence to guess a fee.
        raise NoBasis(f"no status rule for {status!r} in {rule.source or 'any resolved clause'}")

    _ref(resolution.governing)
    common = {
        "order_id": order["order_id"],
        "order_status": status,
        "status_rule": disposition,
        "status_rule_source": rule.source,
        "deferred_clauses": tuple(c.clause_id for c in resolution.deferred),
    }

    if disposition in (_RETURN_TO_ORIGIN, _NOT_CANCELLABLE):
        outcome = CancellationOutcome(
            cancellable=False,
            **attribution(resolution, rule.source),
            fee_inr=None,
            fee_basis=None,
            fee_source=None,
            minutes_since_booking=_elapsed_minutes(order),
            free_window_minutes=None,
            window_source=None,
            within_free_window=None,
            next_action=_RETURN_TO_ORIGIN if disposition == _RETURN_TO_ORIGIN else None,
            warnings=tuple(warnings),
            **common,
        )
        return mint_result(store, outcome, snapshot_id, resolution_id)

    elapsed = _elapsed_minutes(order)
    window = lookup(resolution, "free_window_minutes")
    window_minutes = window.value if isinstance(window.value, int) else None
    within = None if elapsed is None or window_minutes is None else elapsed <= window_minutes

    fee, basis, source = _fee(resolution, disposition, status, within, warnings, rule.source)

    if elapsed is None:
        warnings.append("no cancellation request time recorded; elapsed time could not be computed")

    outcome = CancellationOutcome(
        cancellable=True,
        **attribution(resolution, source),
        fee_inr=fee,
        fee_basis=basis,
        fee_source=source,
        minutes_since_booking=elapsed,
        free_window_minutes=window_minutes,
        window_source=window.source,
        within_free_window=within,
        next_action=None,
        warnings=tuple(warnings),
        **common,
    )
    return mint_result(store, outcome, snapshot_id, resolution_id)


def _fee(
    resolution: PolicyResolution,
    disposition: str,
    status: str,
    within_window: bool | None,
    warnings: list[str],
    status_rule_source: str | None,
) -> tuple[float | None, str, str | None]:
    """The prioritised chain that decides the amount.

    Every branch is explicit and names its source. There is no final `else:
    return 0` - an unreachable case raises instead, because a silent zero here
    is a free cancellation nobody authorised.
    """
    if disposition == _FREE:
        return 0.0, "status_is_free", status_rule_source

    waiver = lookup(resolution, "waiver")
    if waiver.value is True:
        applies_to = lookup(resolution, "applies_to_status")
        scoped = applies_to.value if isinstance(applies_to.value, list) else None
        if scoped is None or status in scoped:
            fee = lookup(resolution, "fee_inr")
            # A waiver clause that states no amount is a parse failure, not a
            # free cancellation.
            if fee.missing:
                raise NoBasis(f"{waiver.source} waives the fee but states no amount")
            return float(fee.value), "agreement_waiver", fee.source

    if within_window is True:
        window = lookup(resolution, "free_window_minutes")
        return 0.0, "inside_free_window", window.source

    charged = lookup(resolution, "fee_after_window_inr")
    if charged.missing:
        raise NoBasis(f"no cancellation fee amount in any clause resolved for {resolution.topic}")
    if within_window is None:
        warnings.append("free window could not be evaluated; standard fee assumed")
    return float(charged.value), "after_free_window", charged.source


def _ref(governing: ClauseRef | None) -> ClauseRef:
    if governing is None:  # pragma: no cover - has_basis already guaranteed it
        raise NoBasis("resolution has no governing clause")
    return governing


def _elapsed_minutes(order: dict[str, Any]) -> int | None:
    booked, requested = order.get("booked_at"), order.get("cancellation_requested_at")
    if not booked or not requested:
        # No request recorded. None, not zero: "cancelled the instant it was
        # booked" is a different claim from "we do not know when they asked".
        return None
    delta = datetime.fromisoformat(requested) - datetime.fromisoformat(booked)
    return int(delta.total_seconds() // 60)
