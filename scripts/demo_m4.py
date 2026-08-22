"""Run the M4 checks by hand: the three conflicts, and severity.

    uv run python scripts/demo_m4.py
    uv run python scripts/demo_m4.py --live      # use the real classifier

Three disagreements the pack contains on purpose, and what the system says
about each. Every line comes from the code the API will call; nothing here is
illustrative.

Without --live, severity shows only what the guards decide. That is not a
degraded mode - it is what a deployment with no model reachable would do, and
the two P1 triggers still fire.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.consistency import ConsistencyChecker
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import (
    CONFIDENCE_THRESHOLD,
    infer_severity,
    load_severity_definitions,
)

WIDTH = 92

SUBJECTS = (
    ("ORD-1001", "northstar_customer", "the app says BOOKED; a ticket says the driver has been"),
    ("ORD-2001", "lumenworks_customer", "the control: same shape, nobody says otherwise"),
    ("TKT-450", "maya_agent", "a closed ticket records a fee this account never owed"),
    ("TKT-451", "rohit_agent", "a closed ticket records a bug threshold as a plan limit"),
)


def rule(title: str = "") -> None:
    print(f"\n{title}\n{'-' * WIDTH}" if title else "-" * WIDTH)


def wrap(text: str, indent: str = "    ") -> str:
    import textwrap

    return textwrap.fill(
        text, width=WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent
    )


def show_conflicts(db_path: Path) -> None:
    rule("CONSISTENCY (D19)")
    for subject, persona_id, note in SUBJECTS:
        principal = to_principal(get_persona(persona_id))
        with (
            open_evidence_store(run_id="demo_m4", principal=principal) as store,
            open_repository(principal, db_path) as repo,
        ):
            if subject.startswith("ORD-"):
                kind = EvidenceKind.ORDER_SNAPSHOT
                payload = repo.get_order(subject).to_payload()
            else:
                kind = EvidenceKind.TICKET_SNAPSHOT
                payload = repo.get_ticket(subject).to_payload()
            snapshot = store.mint(kind, payload)
            checker = ConsistencyChecker(
                store=store, repository=repo, resolver=PolicyResolver(repo.connection)
            )
            report = checker.check(snapshot_id=snapshot)

        print(f"\n{subject}  ({persona_id})  - {note}")
        if not report.conflicts:
            print(f"    no conflict; checked {', '.join(c.value for c in report.checked)}")
            continue
        for conflict in report.conflicts:
            print(
                f"    {conflict.conflict_class.value.upper()}  [{conflict.severity.value}]"
                f"  confidence {conflict.confidence}"
            )
            print(wrap(conflict.detail, "      "))
            if conflict.claimed_value is not None:
                print(
                    f"      recorded: {conflict.claimed_value}   "
                    f"governing: {conflict.current_value}  ({conflict.basis_clause})"
                )
            if conflict.inference_note:
                print(wrap(f"INFERRED: {conflict.inference_note}", "      "))
            if conflict.instruction:
                print(wrap(f"INSTRUCTION: {conflict.instruction}", "      "))
            print(f"      sources: {', '.join(conflict.sources)}")
        print(
            f"    blocking: {report.blocking}"
            f"{'  - prepare_action would refuse' if report.blocking else ''}"
        )


def show_severity(db_path: Path, *, live: bool) -> None:
    rule(f"SEVERITY (D23)   threshold {CONFIDENCE_THRESHOLD}")
    connection = sqlite3.connect(db_path)
    try:
        definitions = load_severity_definitions(connection)
    finally:
        connection.close()

    classifier = None
    if live:
        from src.domain.severity_llm import LlmSeverityClassifier
        from src.providers.registry import get_chat_provider

        classifier = LlmSeverityClassifier(get_chat_provider())
    else:
        print("    (no classifier; guards only - what a deployment with no model would do)")

    principal = to_principal(get_persona("priya_manager"))
    with open_repository(principal, db_path) as repo:
        tickets = sorted(repo.query_tickets(status="open"), key=lambda t: t.ticket_id)

    for ticket in tickets:
        verdict = infer_severity(
            ticket.subject,
            ticket.description or "",
            definitions=definitions,
            classifier=classifier,
        )
        source = "guard" if verdict.deterministic else "inferred"
        trusted = "trusted" if verdict.is_trusted else "NOT TRUSTED"
        print(
            f"\n{ticket.ticket_id}  {verdict.severity or 'undetermined':>12}  "
            f"({source}, confidence {verdict.confidence:.2f}, {trusted})"
        )
        print(f"    {ticket.subject}")
        if verdict.basis:
            print(wrap(f'matched: "{verdict.basis}"  [{verdict.basis_clause}]', "    "))
        for warning in verdict.warnings:
            print(wrap(f"warning: {warning}", "    "))
        if not verdict.is_trusted:
            print("    D25: ops rounds up and flags; the customer surface declines and escalates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use the real severity classifier")
    args = parser.parse_args()

    db_path = get_settings().db_path
    print("=" * WIDTH)
    print("ParcelPilot M4 - conflicts found before the model writes a word")
    print("=" * WIDTH)
    show_conflicts(db_path)
    show_severity(db_path, live=args.live)
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
