"""Pick the severity confidence threshold from data, not from taste.

    uv run python scripts/calibrate_severity.py
    uv run python scripts/calibrate_severity.py --samples 5

ARCHITECTURE open item 5: D25 settles the *behaviour* below the threshold - ops
rounds up, the customer surface declines and escalates - and leaves the number
open, to be calibrated against the five open tickets in Milestone 4.

This runs the real classifier over every open ticket several times and prints
what came back. What we are looking for is separation: the tickets whose
severity the definitions plainly settle should score well clear of the ones
where they genuinely do not, and the threshold belongs in the gap. If there is
no gap, the threshold is arbitrary and the honest thing is to say so rather than
to pick a number that looks decisive.

Needs a working chat provider. The two P1 guards are printed for contrast but
are not part of the calibration - they never reach inference.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.severity import (
    CONFIDENCE_THRESHOLD,
    deterministic_severity,
    infer_severity,
    load_severity_definitions,
)
from src.domain.severity_llm import LlmSeverityClassifier
from src.providers.registry import get_chat_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3, help="runs per ticket")
    args = parser.parse_args()

    settings = get_settings()
    connection = sqlite3.connect(settings.db_path)
    try:
        definitions = load_severity_definitions(connection)
    finally:
        connection.close()

    classifier = LlmSeverityClassifier(get_chat_provider())
    principal = to_principal(get_persona("priya_manager"))

    with open_repository(principal, settings.db_path) as repo:
        tickets = sorted(repo.query_tickets(status="open"), key=lambda t: t.ticket_id)

    print(f"severity calibration over {len(tickets)} open tickets, {args.samples} samples each")
    print(f"current threshold: {CONFIDENCE_THRESHOLD}\n")

    inferred: list[tuple[str, list[float], set[str]]] = []
    for ticket in tickets:
        description = ticket.description or ""
        if guard := deterministic_severity(ticket.subject, description):
            print(f"{ticket.ticket_id}  {guard.severity}  by guard ({guard.basis})")
            continue

        confidences, severities, ungrounded = [], set(), 0
        for _ in range(args.samples):
            verdict = infer_severity(
                ticket.subject, description, definitions=definitions, classifier=classifier
            )
            confidences.append(verdict.confidence)
            severities.add(verdict.severity)
            ungrounded += bool(verdict.warnings)

        inferred.append((ticket.ticket_id, confidences, severities))
        spread = f"{min(confidences):.2f}-{max(confidences):.2f}"
        note = f"  [{ungrounded} ungrounded]" if ungrounded else ""
        print(
            f"{ticket.ticket_id}  {'/'.join(sorted(s or 'undetermined' for s in severities))}"
            f"  mean {statistics.fmean(confidences):.2f}  range {spread}"
            f"  {'stable' if len(severities) == 1 else 'UNSTABLE'}{note}"
        )

    _report_gap(inferred)
    return 0


def _report_gap(inferred: list[tuple[str, list[float], set[str]]]) -> None:
    """Where the threshold could sit, and whether the data supports one at all."""
    if not inferred:
        print("\nnothing reached inference; every open ticket matched a guard")
        return

    stable = [min(c) for _, c, s in inferred if len(s) == 1]
    unstable = [max(c) for _, c, s in inferred if len(s) != 1]

    print()
    if not unstable:
        floor = min(stable)
        print(f"every inferred ticket graded stably; lowest confidence seen was {floor:.2f}")
        print(f"a threshold anywhere below {floor:.2f} accepts all of them")
        print("no ticket in the pack exercises the low-confidence path, so the number is")
        print("chosen for the tickets that are not in the pack - keep it conservative")
        return

    ceiling, floor = max(unstable), min(stable)
    if ceiling < floor:
        print(
            f"clean gap: unstable tickets top out at {ceiling:.2f}, stable ones start at {floor:.2f}"
        )
        print(f"the threshold belongs between them; midpoint {(ceiling + floor) / 2:.2f}")
    else:
        print(f"no clean gap: unstable reaches {ceiling:.2f} and stable falls to {floor:.2f}")
        print("confidence does not separate the two here; the threshold is a judgement call")
        print("and should be documented as one rather than presented as calibrated")


if __name__ == "__main__":
    raise SystemExit(main())
