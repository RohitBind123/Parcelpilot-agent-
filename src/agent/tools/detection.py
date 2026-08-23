"""Proactive detection, as tools (ARCHITECTURE 14).

Manager-only, and that is the whole of the access story: `scan_support_health`
reads every account's tickets, so it is absent from a support agent's schema
and from a customer's. GS-031 and its pair exist to prove the capability is
withheld by role rather than simply unbuilt.

The scanner itself takes no evidence handles - it is a read over the whole
queue, not a calculation about one record. The first-response signal does need
them, because the target depends on the account's agreement, so the closure
below mints the snapshots and resolves the policy per ticket. That keeps the
evidence discipline where the numbers are without demanding a handle for a
dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from src.agent.tools.base import Param, Tool, ToolError, ToolResult
from src.auth.principal import SCOPE_OPS_DETECTION
from src.domain.calculators.sla import sla_first_response_status
from src.domain.detection import HealthScanner
from src.domain.evidence import EvidenceKind
from src.knowledge.topics import Topic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

logger = logging.getLogger(__name__)

#: How many findings a single response carries. The pack produces six; the cap
#: is here so a larger deployment cannot return an unbounded list.
MAX_FINDINGS = 50


def _sla_for(context: ToolContext):
    """A per-ticket first-response outcome, or None.

    Returns a plain mapping rather than the dataclass so the scanner stays free
    of calculator imports - it ranks findings, it does not compute targets.
    """

    def outcome(ticket: Any, verdict: Any) -> dict[str, Any] | None:
        try:
            account = context.repository.get_account(ticket.account_id)
            snapshot = context.store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())
            account_snapshot = context.store.mint(
                EvidenceKind.ACCOUNT_SNAPSHOT, account.to_payload()
            )
            resolution = context.resolver.resolve(
                Topic.FIRST_RESPONSE_TARGET.value,
                context.principal,
                account_id=ticket.account_id,
            )
            handle = context.store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
            )
            computed = sla_first_response_status(
                context.store,
                snapshot_id=snapshot,
                account_snapshot_id=account_snapshot,
                resolution_id=handle,
                severity=verdict,
                surface="ops",
            )
        except Exception as exc:
            # One ticket without a resolvable target must not take the scan
            # down; the finding is simply absent and the signal still reports.
            logger.warning("no first-response target for %s: %s", ticket.ticket_id, exc)
            return None
        return asdict(computed) if hasattr(computed, "__dataclass_fields__") else dict(computed)

    return outcome


def _scanner(context: ToolContext) -> HealthScanner:
    return HealthScanner(
        repository=context.repository,
        resolver=context.resolver,
        severity_classifier=context.severity_classifier,
        sla=_sla_for(context),
    )


def scan_support_health(context: ToolContext) -> Tool:
    def run(limit: int = 20) -> ToolResult | ToolError:
        if not isinstance(limit, int) or limit < 1:
            return ToolError("limit must be a positive whole number")
        report = _scanner(context).scan()
        payload = report.to_payload()
        payload["findings"] = payload["findings"][: min(limit, MAX_FINDINGS)]
        payload["total_findings"] = len(report.findings)
        return ToolResult(payload)

    return Tool(
        name="scan_support_health",
        description=(
            "Scan the whole support queue for issues worth attention: tickets matching a "
            "known issue, tickets past their response target, severity clustered on one "
            "account, and high-severity tickets that match no known issue. Returns ranked "
            "findings, worst first."
        ),
        params=[
            Param(
                name="limit",
                type="integer",
                description="How many findings to return, worst first.",
                required=False,
            )
        ],
        run=run,
        requires_scope=SCOPE_OPS_DETECTION,
    )


def explain_finding(context: ToolContext) -> Tool:
    def run(finding_id: str) -> ToolResult | ToolError:
        report = _scanner(context).scan()
        finding = report.find(finding_id)
        if finding is None:
            known = [f.finding_id for f in report.findings]
            return ToolError(f"no finding {finding_id!r} in the current scan; available: {known}")

        payload = finding.to_payload()
        # The subject as the records hold it, so the drill-down does not send
        # the model back for another lookup it already had the answer to.
        if finding.subject_kind == "ticket":
            try:
                payload["ticket"] = context.repository.get_ticket(finding.subject_id).to_payload()
            except Exception:
                logger.warning("finding %s names an unreadable ticket", finding_id)
        payload["as_of"] = report.as_of
        return ToolResult(payload)

    return Tool(
        name="explain_finding",
        description=(
            "Explain one finding from the support-health scan: what was detected, which "
            "records and clauses it rests on, and what the corpus says to do about it."
        ),
        params=[
            Param(
                name="finding_id",
                type="string",
                description="The finding to explain.",
                produced_by="scan_support_health",
            )
        ],
        run=run,
        requires_scope=SCOPE_OPS_DETECTION,
    )


__all__ = ["MAX_FINDINGS", "explain_finding", "scan_support_health"]
