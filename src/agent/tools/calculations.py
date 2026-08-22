"""Calculators and the consistency check, as tools.

The signatures are the point. `compute_cancellation_fee` has no `order_id`
parameter, so the model cannot skip the snapshot; it has no `topic` parameter,
so it cannot skip the resolver. The argument it needs does not exist until the
prerequisite has run, which is what turns "the brief requires multi-step" from a
hope about tool ordering into a property of the schema.

When a prerequisite is missing the model gets a `ToolError` naming the tool that
mints it, not a validation failure. Being told what to do next is the whole
difference between a constraint the model can satisfy and one it thrashes
against.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from src.agent.tools.base import Param, Tool, ToolError, ToolResult
from src.auth.principal import SCOPE_SLA_STATUS
from src.domain.calculators.cancellation import compute_cancellation_fee as _cancellation
from src.domain.calculators.credit import compute_service_credit as _credit
from src.domain.calculators.errors import CalculationError, NoBasis, WrongEvidence
from src.domain.calculators.sla import sla_first_response_status as _sla
from src.domain.consistency import ConsistencyChecker
from src.domain.evidence import EvidenceError, EvidenceKind
from src.domain.severity import infer_severity, load_severity_definitions
from src.knowledge.topics import Topic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

TOPICS = tuple(sorted(t.value for t in Topic))


def _outcome(value: Any) -> dict[str, Any]:
    return asdict(value) if is_dataclass(value) else dict(value)


def _handle_failure(exc: Exception, *, needs: str, produced_by: str, topic: str) -> ToolError:
    """Turn a domain refusal into something the model can act on.

    `WrongEvidence` means the handle exists and belongs elsewhere - a resolution
    for another topic or another account. Naming the call that would produce
    the right one is what lets the model fix it instead of retrying the same
    arguments.

    `NoBasis` is not recoverable by trying differently: nothing citable governs
    the topic for this account, and the correct next move is to say so and
    escalate, not to search again.
    """
    if isinstance(exc, WrongEvidence):
        return ToolError(
            f"{exc}. Call {produced_by}(topic={topic!r}) and pass the handle as {needs}."
        )
    return ToolError(str(exc), recoverable=not isinstance(exc, NoBasis))


def compute_cancellation_fee(context: ToolContext) -> Tool:
    def run(snapshot_id: str, resolution_id: str) -> ToolResult | ToolError:
        try:
            outcome = _cancellation(
                context.store, snapshot_id=snapshot_id, resolution_id=resolution_id
            )
        except (CalculationError, EvidenceError) as exc:
            return _handle_failure(
                exc, needs="resolution_id", produced_by="resolve_policy", topic="cancellation_fee"
            )
        return ToolResult(_outcome(outcome))

    return Tool(
        name="compute_cancellation_fee",
        description=(
            "Whether an order can be cancelled and what it costs. Requires a snapshot_id "
            "from get_order and a resolution_id from resolve_policy(topic='cancellation_fee'). "
            "There is no order_id parameter: take the snapshot first."
        ),
        params=(
            Param("snapshot_id", "string", "From get_order.", produced_by="get_order"),
            Param(
                "resolution_id",
                "string",
                "From resolve_policy(topic='cancellation_fee').",
                produced_by="resolve_policy(topic='cancellation_fee')",
            ),
        ),
        run=run,
    )


def compute_service_credit(context: ToolContext) -> Tool:
    def run(
        snapshot_id: str,
        resolution_id: str,
        approval_resolution_id: str | None = None,
        proposed_credit_inr: float | None = None,
    ) -> ToolResult | ToolError:
        try:
            outcome = _credit(
                context.store,
                snapshot_id=snapshot_id,
                resolution_id=resolution_id,
                approval_resolution_id=approval_resolution_id,
                proposed_credit_inr=proposed_credit_inr,
            )
        except (CalculationError, EvidenceError) as exc:
            return _handle_failure(
                exc,
                needs="resolution_id",
                produced_by="resolve_policy",
                topic="failed_pickup_credit",
            )
        return ToolResult(_outcome(outcome))

    return Tool(
        name="compute_service_credit",
        description=(
            "Service credit for a failed or late pickup. Requires a snapshot_id from "
            "get_order and a resolution_id from "
            "resolve_policy(topic='failed_pickup_credit'). Pass approval_resolution_id "
            "from resolve_policy(topic='credit_approval') to learn whether manager "
            "approval is needed."
        ),
        params=(
            Param("snapshot_id", "string", "From get_order.", produced_by="get_order"),
            Param(
                "resolution_id",
                "string",
                "From resolve_policy(topic='failed_pickup_credit').",
                produced_by="resolve_policy(topic='failed_pickup_credit')",
            ),
            Param(
                "approval_resolution_id",
                "string",
                "From resolve_policy(topic='credit_approval').",
                required=False,
            ),
            Param(
                "proposed_credit_inr",
                "number",
                "A specific amount to test against the approval threshold.",
                required=False,
            ),
        ),
        run=run,
    )


def sla_first_response_status(context: ToolContext) -> Tool:
    def run(
        snapshot_id: str, account_snapshot_id: str, resolution_id: str
    ) -> ToolResult | ToolError:
        try:
            ticket = context.store.read(snapshot_id, expect=EvidenceKind.TICKET_SNAPSHOT)
        except EvidenceError as exc:
            return ToolError(f"{exc}. Call get_ticket and pass its snapshot_id.")

        severity = infer_severity(
            str(ticket.get("subject", "")),
            str(ticket.get("description") or ""),
            definitions=load_severity_definitions(context.repository.connection),
            classifier=context.severity_classifier,
        )
        try:
            outcome = _sla(
                context.store,
                snapshot_id=snapshot_id,
                account_snapshot_id=account_snapshot_id,
                resolution_id=resolution_id,
                severity=severity,
                surface="ops",
            )
        except (CalculationError, EvidenceError) as exc:
            return _handle_failure(
                exc,
                needs="resolution_id",
                produced_by="resolve_policy",
                topic="first_response_target",
            )
        return ToolResult(_outcome(outcome))

    return Tool(
        name="sla_first_response_status",
        description=(
            "Elapsed time against the first-response target for a ticket. Severity is "
            "derived from the policy, never supplied. Requires a snapshot_id from "
            "get_ticket, an account_snapshot_id from get_account, and a resolution_id "
            "from resolve_policy(topic='first_response_target'). Note that ParcelPilot "
            "does not record first-reply times, so this is never a confirmed breach."
        ),
        params=(
            Param("snapshot_id", "string", "From get_ticket.", produced_by="get_ticket"),
            Param(
                "account_snapshot_id",
                "string",
                "From get_account.",
                produced_by="get_account",
            ),
            Param(
                "resolution_id",
                "string",
                "From resolve_policy(topic='first_response_target').",
                produced_by="resolve_policy(topic='first_response_target')",
            ),
        ),
        run=run,
        requires_scope=SCOPE_SLA_STATUS,
    )


def check_data_consistency(context: ToolContext) -> Tool:
    def run(snapshot_id: str, topics: list[str] | None = None) -> ToolResult | ToolError:
        named = tuple(topics or ())
        unknown = sorted(set(named) - set(TOPICS))
        if unknown:
            return ToolError(f"unknown topic(s) {unknown}; expected a subset of {list(TOPICS)}")

        checker = ConsistencyChecker(
            store=context.store, repository=context.repository, resolver=context.resolver
        )
        try:
            report = checker.check(snapshot_id=snapshot_id, topics=named)
        except EvidenceError as exc:
            return ToolError(f"{exc}. Call get_order or get_ticket and pass its snapshot_id.")
        return ToolResult(
            {
                "report_id": report.report_id.evidence_id,
                "blocking": report.blocking,
                "checked": [c.value for c in report.checked],
                "conflicts": [c.to_payload() for c in report.conflicts],
            }
        )

    return Tool(
        name="check_data_consistency",
        description=(
            "Cross-check a snapshot against open tickets, current known issues and the "
            "clauses that govern it. Required before any state-changing action: SOP v4 "
            "section 3 says a conflict must be identified and verified first. Pass the "
            "topics under discussion to also check that a citable rule exists for them."
        ),
        params=(
            Param(
                "snapshot_id",
                "string",
                "From get_order or get_ticket.",
                produced_by="get_order or get_ticket",
            ),
            Param(
                "topics",
                "array",
                "Topics the question is about, checked for a citable source.",
                required=False,
                items="string",
            ),
        ),
        run=run,
    )
