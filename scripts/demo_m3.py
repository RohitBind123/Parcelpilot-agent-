"""Run the M3 chain by hand: snapshot, resolve, compute.

    uv run python scripts/demo_m3.py
    uv run python scripts/demo_m3.py --order ORD-2002
    uv run python scripts/demo_m3.py --ticket TKT-501

Shows the discriminating pair side by side, because that is the case the whole
design is arranged around: two shipments of identical shape and opposite
answers, with the difference visible in the clause rather than asserted.

Each line is produced by the same code the API will call. Nothing here is
illustrative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.cancellation import compute_cancellation_fee
from src.domain.calculators.credit import compute_service_credit
from src.domain.calculators.sla import sla_first_response_status
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import SeverityVerdict, deterministic_severity

WIDTH = 92
OWNER = {
    "ORD-1001": ("northstar_customer", "ACCT-001"),
    "ORD-1002": ("northstar_customer", "ACCT-001"),
    "ORD-2001": ("lumenworks_customer", "ACCT-002"),
    "ORD-2002": ("lumenworks_customer", "ACCT-002"),
    "ORD-3001": ("beacon_customer", "ACCT-003"),
    "ORD-4001": ("axis_customer", "ACCT-004"),
    "TKT-501": ("maya_agent", "ACCT-001"),
    "TKT-502": ("maya_agent", "ACCT-002"),
    "TKT-503": ("rohit_agent", "ACCT-003"),
    "TKT-504": ("maya_agent", "ACCT-001"),
    "TKT-505": ("rohit_agent", "ACCT-004"),
}
PINNED_SEVERITY = {"TKT-502": "P2", "TKT-503": "P3", "TKT-504": "P3"}


def money(value: float | None) -> str:
    return "not applicable" if value is None else f"INR {value:,.0f}"


def rule(title: str) -> None:
    print(f"\n{'=' * WIDTH}\n{title}\n{'=' * WIDTH}")


def cancellation(db, order_id: str) -> None:
    persona_id, account = OWNER[order_id]
    principal = to_principal(get_persona(persona_id))
    with open_evidence_store(run_id="demo", principal=principal) as store:
        with open_repository(principal, db) as repo:
            order = repo.get_order(order_id)
        snap = store.mint(EvidenceKind.ORDER_SNAPSHOT, order.to_payload())
        with PolicyResolver.open(db) as resolver:
            resolution = resolver.resolve("cancellation_fee", principal)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snap]
        )
        out = compute_cancellation_fee(store, snapshot_id=snap, resolution_id=handle)

    print(f"\n  {order_id}  {account}  status={out.order_status}")
    if out.minutes_since_booking is not None:
        print(f"      elapsed        {out.minutes_since_booking} min since booking")
    print(f"      verdict        {'cancellable' if out.cancellable else 'NOT cancellable'}")
    print(f"      fee            {money(out.fee_inr)}   ({out.fee_basis or 'no fee rule applied'})")
    print(f"      governing      {out.governing_clause}")
    for clause in out.overridden_clauses:
        print(f"      OVERRIDES      {clause}")
    for clause in out.deferred_clauses:
        print(f"      defers to SOP  {clause}  (agreement present, declines to override)")
    if out.next_action:
        print(f"      next action    {out.next_action}")
    for warning in out.warnings:
        print(f"      warning        {warning}")


def credit(db, order_id: str) -> None:
    persona_id, account = OWNER[order_id]
    principal = to_principal(get_persona(persona_id))
    with open_evidence_store(run_id="demo", principal=principal) as store:
        with open_repository(principal, db) as repo:
            order = repo.get_order(order_id)
        snap = store.mint(EvidenceKind.ORDER_SNAPSHOT, order.to_payload())
        with PolicyResolver.open(db) as resolver:
            resolution = resolver.resolve("failed_pickup_credit", principal)
            approval = resolver.resolve("credit_approval", principal)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snap]
        )
        approval_handle = store.mint(EvidenceKind.POLICY_RESOLUTION, approval.to_payload())
        out = compute_service_credit(
            store,
            snapshot_id=snap,
            resolution_id=handle,
            approval_resolution_id=approval_handle,
        )

    print(f"\n  {order_id}  {account}")
    print(f"      delay          {out.delay_hours} h past the pickup window")
    print(f"      threshold      {out.threshold_hours} h   ({out.threshold_source})")
    print(
        f"      verdict        {'ELIGIBLE' if out.eligible else 'not eligible'}"
        f"{'' if out.eligible else f'  ({out.ineligible_reason})'}"
    )
    print(f"      credit         {money(out.credit_inr)}   ({out.rate_basis})")
    if out.default_credit_inr is not None:
        print(
            f"      default policy would have paid {money(out.default_credit_inr)} "
            f"past {out.default_threshold_hours} h"
        )
    print(f"      governing      {out.governing_clause}")
    for clause in out.overridden_clauses:
        print(f"      OVERRIDES      {clause}")
    print(
        f"      approval       {'required' if out.requires_manager_approval else 'not required'}"
        f"  (above {money(out.approval_threshold_inr)})"
    )


def sla(db, ticket_id: str) -> None:
    persona_id, account_id = OWNER[ticket_id]
    principal = to_principal(get_persona(persona_id))
    with open_evidence_store(run_id="demo", principal=principal) as store:
        with open_repository(principal, db) as repo:
            ticket = repo.get_ticket(ticket_id)
            account = repo.get_account(account_id)
        tsnap = store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())
        asnap = store.mint(EvidenceKind.ACCOUNT_SNAPSHOT, account.to_payload())
        with PolicyResolver.open(db) as resolver:
            resolution = resolver.resolve("first_response_target", principal, account_id=account_id)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[tsnap]
        )
        verdict = deterministic_severity(
            ticket.subject, ticket.description or ""
        ) or SeverityVerdict(
            severity=PINNED_SEVERITY.get(ticket_id, "P3"),
            confidence=1.0,
            basis_clause="support_policy_v3_current::§2",
            basis="pinned pending M4 inference",
        )
        out = sla_first_response_status(
            store,
            snapshot_id=tsnap,
            account_snapshot_id=asnap,
            resolution_id=handle,
            severity=verdict,
            surface="ops",
        )

    flag = "guard" if not out.severity_inferred else "inferred"
    print(f"\n  {ticket_id}  {account_id}  {account.plan}")
    print(f"      severity       {out.severity}  ({flag}, {verdict.basis})")
    print(f"      target         {out.target}   ({out.clock_type})")
    print(f"      clock starts   {out.clock_starts_at}")
    print(f"      due            {out.due_at}")
    past = out.past_target_by_minutes
    standing = f"past target by {past} min" if past > 0 else f"due in {-past} min"
    print(f"      elapsed        {out.elapsed_minutes} min   {standing}")
    print(f"      governing      {out.target_clause}")
    for clause in out.overridden_clauses:
        print(f"      OVERRIDES      {clause}")
    print(f"      measurable     {out.measurable}  - {out.measurability_note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default=None)
    parser.add_argument("--ticket", default=None)
    args = parser.parse_args()

    db = get_settings().db_path

    if args.order:
        rule(f"cancellation and credit: {args.order}")
        cancellation(db, args.order)
        credit(db, args.order)
        return 0
    if args.ticket:
        rule(f"first-response status: {args.ticket}")
        sla(db, args.ticket)
        return 0

    rule("The discriminating pair - same shape, opposite answers")
    cancellation(db, "ORD-1001")
    cancellation(db, "ORD-2001")
    print("\n  Both BOOKED, both past the 30-minute window. The difference is the agreement:")
    print("  Northstar's overrides the SOP; LumenWorks' exists and declines to.")

    rule("Cancellation across the rest of the pack")
    for order_id in ("ORD-3001", "ORD-1002", "ORD-4001"):
        cancellation(db, order_id)

    rule("Service credit - the threshold replacement")
    credit(db, "ORD-2002")

    rule("First-response targets - AS_OF is a Sunday")
    for ticket_id in ("TKT-501", "TKT-502", "TKT-505"):
        sla(db, ticket_id)
    print("\n  24x7 targets run through the weekend; business-hours targets wait for Monday.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
