"""Failed-pickup service credits.

Acceptance is GS-007 through GS-010.

The property that makes this harder than it looks is that an agreement can
replace the *threshold* as well as the amount. LumenWorks section 3 pays a flat
INR 300 but only past four hours, where the SOP pays past two - so at a
three-hour delay the override makes the customer worse off. Both directions have
to be surfaced with equal prominence; a system that mentions overrides only when
they are favourable is not reporting precedence.

Two amounts are therefore computed: what the governing rule pays, and what the
default would have paid. The second is not decoration - it is what makes an
override legible to the person reading the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from src.clock import as_of
from src.domain.calculators.base import attribution, load_resolution, mint_result
from src.domain.calculators.errors import NoBasis
from src.domain.calculators.params import clauses_in_priority, lookup
from src.domain.evidence import EvidenceKind, EvidenceStore, Handle
from src.domain.resolver import PolicyResolution

TOPIC: Final = "failed_pickup_credit"
APPROVAL_TOPIC: Final = "credit_approval"


@dataclass(frozen=True, slots=True)
class ServiceCreditOutcome:
    """A credit decision, with the clause behind every number."""

    order_id: str | None
    eligible: bool
    ineligible_reason: str | None
    #: None when no credit is owed, and also when one is owed but the amount
    #: cannot be computed. Those are different states; `amount_formula` and
    #: `warnings` distinguish them.
    credit_inr: float | None
    amount_formula: str | None
    rate_basis: str | None
    delay_hours: float | None
    threshold_hours: float | None
    threshold_source: str | None
    #: What the default policy would have done. Present so an override is
    #: legible rather than merely cited.
    default_threshold_hours: float | None
    default_credit_inr: float | None
    eligible_under_default: bool | None
    requires_manager_approval: bool | None
    approval_threshold_inr: float | None
    approval_source: str | None
    monthly_cap_inr: float | None
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
            "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason,
            "credit_inr": self.credit_inr,
            "amount_formula": self.amount_formula,
            "rate_basis": self.rate_basis,
            "delay_hours": self.delay_hours,
            "threshold_hours": self.threshold_hours,
            "threshold_source": self.threshold_source,
            "default_threshold_hours": self.default_threshold_hours,
            "default_credit_inr": self.default_credit_inr,
            "eligible_under_default": self.eligible_under_default,
            "requires_manager_approval": self.requires_manager_approval,
            "approval_threshold_inr": self.approval_threshold_inr,
            "approval_source": self.approval_source,
            "monthly_cap_inr": self.monthly_cap_inr,
            "governing_clause": self.governing_clause,
            "overridden_clauses": list(self.overridden_clauses),
            "deferred_clauses": list(self.deferred_clauses),
            "is_override": self.is_override,
            "warnings": list(self.warnings),
        }


def compute_service_credit(
    store: EvidenceStore,
    *,
    snapshot_id: Handle | str,
    resolution_id: Handle | str,
    approval_resolution_id: Handle | str | None = None,
    proposed_credit_inr: float | None = None,
) -> ServiceCreditOutcome:
    """Whether a failed pickup earns a credit, and how much.

    `approval_resolution_id` is separate because the manager-approval line lives
    on a different topic (SOP v4 section 3 is `credit_approval`, not
    `failed_pickup_credit`). Without it the approval question is reported as
    unknown rather than answered as False - a missing threshold is not a
    licence to settle.
    """
    order = store.read(snapshot_id, expect=EvidenceKind.ORDER_SNAPSHOT)
    resolution = load_resolution(
        store, resolution_id, topic=TOPIC, account_id=order.get("account_id")
    )

    warnings: list[str] = []
    delay = _delay_hours(order, warnings)

    threshold = lookup(resolution, "threshold_hours")
    if threshold.missing:
        raise NoBasis(f"no delay threshold in any clause resolved for {TOPIC!r}")

    eligible, reason = _eligibility(order, delay, float(threshold.value), warnings)

    baseline = _baseline(resolution)
    default_threshold = baseline.get("threshold_hours") if baseline else None
    default_amount = _default_amount(baseline, order.get("shipment_fee_inr"))
    eligible_under_default = (
        None
        if delay is None or default_threshold is None
        else _preconditions_met(order) and delay > float(default_threshold)
    )

    credit, basis, formula, source = (
        _amount(resolution, order, warnings, proposed_credit_inr)
        if eligible
        else (None, None, None, None)
    )

    approval_threshold, approval_source = _approval(store, approval_resolution_id, order, warnings)
    requires_approval = (
        None if approval_threshold is None or credit is None else credit > approval_threshold
    )

    cap = lookup(resolution, "monthly_cap_inr")

    outcome = ServiceCreditOutcome(
        order_id=order.get("order_id"),
        eligible=eligible,
        ineligible_reason=reason,
        credit_inr=credit,
        amount_formula=formula,
        rate_basis=basis,
        delay_hours=delay,
        threshold_hours=float(threshold.value),
        threshold_source=threshold.source,
        default_threshold_hours=default_threshold,
        default_credit_inr=default_amount,
        eligible_under_default=eligible_under_default,
        requires_manager_approval=requires_approval,
        approval_threshold_inr=approval_threshold,
        approval_source=approval_source,
        monthly_cap_inr=cap.value if not cap.missing else None,
        deferred_clauses=tuple(c.clause_id for c in resolution.deferred),
        warnings=tuple(warnings),
        **attribution(resolution, source or threshold.source),
    )
    sources = [snapshot_id, resolution_id]
    if approval_resolution_id is not None:
        sources.append(approval_resolution_id)
    return mint_result(store, outcome, *sources)


# -- eligibility -------------------------------------------------------------


def _preconditions_met(order: dict[str, Any]) -> bool:
    return bool(order.get("carrier_fault")) and not order.get("customer_fault")


def _eligibility(
    order: dict[str, Any], delay: float | None, threshold: float, warnings: list[str]
) -> tuple[bool, str | None]:
    """Explicit branches, named reasons. No silent False."""
    if not order.get("carrier_fault"):
        return False, "carrier_fault_not_established"
    if order.get("customer_fault"):
        return False, "customer_fault"
    if delay is None:
        warnings.append("pickup delay could not be computed; eligibility undetermined")
        return False, "delay_unknown"
    if delay <= threshold:
        return False, "below_threshold"
    return True, None


def _delay_hours(order: dict[str, Any], warnings: list[str]) -> float | None:
    """Hours past the end of the scheduled pickup window.

    A hypothetical question carries `delay_hours` directly; a real order is
    measured against the frozen clock when nothing was ever collected.
    """
    if order.get("delay_hours") is not None:
        return float(order["delay_hours"])

    window_end = order.get("pickup_window_end")
    if not window_end:
        warnings.append("no scheduled pickup window recorded")
        return None

    end = datetime.fromisoformat(window_end)
    actual = order.get("pickup_actual_at")
    reference = datetime.fromisoformat(actual) if actual else as_of()
    return round((reference - end).total_seconds() / 3600, 4)


# -- amount ------------------------------------------------------------------


def _amount(
    resolution: PolicyResolution,
    order: dict[str, Any],
    warnings: list[str],
    proposed: float | None,
) -> tuple[float | None, str, str | None, str | None]:
    """The prioritised chain that decides the credit."""
    if proposed is not None:
        # A goodwill amount an agent typed. Still measured against the approval
        # line rather than accepted silently.
        return float(proposed), "proposed", f"proposed {proposed}", None

    flat = lookup(resolution, "credit_inr")
    if not flat.missing:
        return float(flat.value), "agreement_flat", f"flat {flat.value}", flat.source

    cap = lookup(resolution, "credit_flat_inr")
    percent = lookup(resolution, "credit_percent")
    if cap.missing or percent.missing:
        raise NoBasis(f"no credit amount in any clause resolved for {TOPIC!r}")

    formula = f"lower of INR {cap.value} or {percent.value}% of the shipment fee"
    fee = order.get("shipment_fee_inr")
    if fee is None:
        # Eligible, but the amount cannot be computed. Reporting a number here
        # would be a hallucination dressed as arithmetic; reporting zero would
        # be worse.
        warnings.append(
            "shipment fee is unknown, so the credit amount cannot be computed; "
            "the rule is stated instead"
        )
        return None, "undetermined", formula, cap.source

    proportional = float(fee) * float(percent.value) / 100
    if proportional < float(cap.value):
        return round(proportional, 2), "percentage_of_fee", formula, percent.source
    # "Lower of" is a cap, not a choice. Reading it as "either" both overpays
    # and manufactures a manager approval the correct reading never reaches.
    return float(cap.value), "flat_cap", formula, cap.source


def _baseline(resolution: PolicyResolution) -> dict[str, Any] | None:
    """The default-policy clause, whether or not it governs.

    Identified by carrying `credit_rule`, which only the SOP does. Used to say
    what the default would have paid, which is what makes an override legible.
    """
    for ref in clauses_in_priority(resolution):
        if "credit_rule" in ref.params:
            return dict(ref.params)
    return None


def _default_amount(baseline: dict[str, Any] | None, fee: float | None) -> float | None:
    if not baseline or fee is None:
        return None
    flat, percent = baseline.get("credit_flat_inr"), baseline.get("credit_percent")
    if flat is None or percent is None:
        return None
    return round(min(float(flat), float(fee) * float(percent) / 100), 2)


def _approval(
    store: EvidenceStore,
    approval_resolution_id: Handle | str | None,
    order: dict[str, Any],
    warnings: list[str],
) -> tuple[float | None, str | None]:
    if approval_resolution_id is None:
        warnings.append(
            "no credit-approval resolution supplied; whether manager approval "
            "is required was not determined"
        )
        return None, None
    approval = load_resolution(
        store,
        approval_resolution_id,
        topic=APPROVAL_TOPIC,
        account_id=order.get("account_id"),
    )
    found = lookup(approval, "manager_approval_above_inr")
    return (None if found.missing else float(found.value)), found.source
