"""The confirmation gate, as tools (ARCHITECTURE 13).

Three tools, and the shape of each is the argument for it.

`prepare_action` takes what to do and returns a proposal. It never performs
anything, so a model that stops here has changed nothing.

`execute_action` takes a token and the pending action the graph is holding. It
takes no payload, so there is nothing for a model to supply: the only thing it
can do is authorise the description a human already saw.

`approve_credit` exists as a separate tool rather than a `kind` because the
projection matrix is what keeps it out of a support agent's schema - "Maya's
has no `approve_credit` to be talked into calling". That containment is worth
nothing if `prepare_action(kind="approve_credit")` works, since the kind is a
string the model writes, so the generic path refuses that kind outright. For a
manager too: managers approve credit through the tool that applies the SOP v4
threshold, and letting the generic path through would skip it.

D19 gates the proposal. A blocking conflict refuses before a token is minted,
because a token is a promise that the description is safe to execute. An
advisory conflict travels with the preview so the card can show it, which is
the difference between "we cannot do this" and "you should know this first".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from src.agent.tools.base import Param, Tool, ToolError, ToolPending, ToolResult
from src.auth.principal import SCOPE_APPROVE_CREDIT, SCOPE_PREPARE_ACTION
from src.datastore.runtime import ActionKind, ImmutableLogError
from src.domain.action_tokens import PendingAction, TokenError, build_pending, mint_token
from src.domain.consistency import ConsistencyChecker
from src.domain.evidence import EvidenceError, EvidenceKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

logger = logging.getLogger(__name__)

#: Kinds the generic `prepare_action` will propose. `approve_credit` is absent
#: on purpose - see the module docstring.
PROPOSABLE_KINDS: Final[tuple[ActionKind, ...]] = (
    ActionKind.CREATE_ESCALATION,
    ActionKind.UPDATE_TICKET_STATUS,
    ActionKind.CREATE_FOLLOWUP_TASK,
    ActionKind.REQUEST_CARRIER_VERIFICATION,
)

_KIND_NAMES: Final = tuple(k.value for k in PROPOSABLE_KINDS)

#: Handle kinds a consistency check is defined on. A resolution or a
#: calculation is not a subject for one; skipping those is the definition of
#: the check, not a gap in it.
_CHECKABLE: Final = frozenset(
    {
        EvidenceKind.ORDER_SNAPSHOT,
        EvidenceKind.TICKET_SNAPSHOT,
        EvidenceKind.ACCOUNT_SNAPSHOT,
    }
)


def prepare_action(context: ToolContext) -> Tool:
    def run(
        kind: str, payload: Mapping[str, Any], evidence_ids: Sequence[str] | None = None
    ) -> ToolPending | ToolError:
        if kind == ActionKind.APPROVE_CREDIT.value:
            # Routing, not permission. Named explicitly so the refusal reads as
            # "use the other tool" rather than "no".
            return ToolError(
                "credit approval does not go through prepare_action; it has its own tool, "
                "which applies the SOP v4 threshold and is available to ops managers only"
            )
        try:
            resolved = ActionKind(kind)
        except ValueError:
            return ToolError(f"unknown action kind {kind!r}; expected one of {list(_KIND_NAMES)}")
        if resolved not in PROPOSABLE_KINDS:
            return ToolError(f"action kind {kind!r} cannot be proposed here")
        if not isinstance(payload, Mapping):
            return ToolError("payload must be an object describing the action")

        return _propose(
            context, kind=resolved, payload=dict(payload), evidence_ids=evidence_ids or ()
        )

    return Tool(
        name="prepare_action",
        description=(
            "Propose an action and pause for the person to confirm it. Nothing happens until "
            "they do. Use this to draft an escalation, update a ticket status, create a "
            "follow-up, or ask for carrier verification."
        ),
        params=[
            Param(
                name="kind",
                type="string",
                description="What kind of action to propose.",
                enum=_KIND_NAMES,
            ),
            Param(
                name="payload",
                type="object",
                description=(
                    "The details of the action, as an object. For an escalation include the "
                    "question verbatim, what is unresolved, and the sources consulted."
                ),
            ),
            Param(
                name="evidence_ids",
                type="array",
                description="Evidence handles that justify this action.",
                required=False,
                items="string",
            ),
        ],
        run=run,
        requires_scope=SCOPE_PREPARE_ACTION,
    )


def approve_credit(context: ToolContext) -> Tool:
    def run(
        account_id: str,
        amount_inr: float,
        reason: str,
        evidence_ids: Sequence[str] | None = None,
    ) -> ToolPending | ToolError:
        if not isinstance(amount_inr, int | float) or amount_inr <= 0:
            # Zero is not a credit and a negative one is a charge.
            return ToolError("amount_inr must be a positive number of rupees")
        if not reason.strip():
            return ToolError("a credit approval needs a reason; it goes on the audit record")

        return _propose(
            context,
            kind=ActionKind.APPROVE_CREDIT,
            payload={
                "account_id": account_id,
                "amount_inr": amount_inr,
                "reason": reason.strip(),
            },
            evidence_ids=evidence_ids or (),
        )

    return Tool(
        name="approve_credit",
        description=(
            "Propose a service credit for approval. Pauses for confirmation. Credits above "
            "INR 1,000 require an ops manager under SOP v4 section 3."
        ),
        params=[
            Param(name="account_id", type="string", description="Account to credit."),
            Param(name="amount_inr", type="number", description="Credit amount in rupees."),
            Param(name="reason", type="string", description="Why the credit is owed."),
            Param(
                name="evidence_ids",
                type="array",
                description="Evidence handles that justify the amount.",
                required=False,
                items="string",
            ),
        ],
        run=run,
        requires_scope=SCOPE_APPROVE_CREDIT,
    )


def execute_action(context: ToolContext) -> Tool:
    """Perform an action a human has confirmed.

    Takes no payload. The description comes from `pending`, which the graph
    holds in state, so there is nothing here for a model to substitute - the
    only thing a caller can do is authorise what was already previewed.
    """

    def run(token: str, pending: PendingAction | None = None) -> ToolResult | ToolError:
        if pending is None:
            return ToolError("there is no prepared action waiting for confirmation")
        try:
            from src.domain.action_tokens import verify_token

            verify_token(token, pending, secret=context.action_secret)
        except TokenError as exc:
            logger.warning("confirmation refused for %s: %s", pending.kind.value, exc)
            return ToolError(str(exc), recoverable=False)

        try:
            record = context.runtime.append_action(
                kind=pending.kind,
                payload=pending.payload,
                evidence_chain=pending.evidence_chain,
                principal_id=context.principal.user_id,
                thread_id=pending.thread_id,
                nonce=pending.nonce,
            )
        except ImmutableLogError as exc:
            # A replay. The effect already happened once; saying so is more
            # useful than either repeating it or reporting a generic failure.
            return ToolError(str(exc), recoverable=False)

        return ToolResult(
            {
                "executed": True,
                "action_id": record.action_id,
                "kind": record.kind.value,
                "occurred_at": record.occurred_at.isoformat(),
                "evidence_chain": list(record.evidence_chain),
            }
        )

    return Tool(
        name="execute_action",
        description="Perform an action the person has confirmed.",
        params=[
            Param(
                name="token",
                type="string",
                description="The confirmation token.",
                produced_by="the person confirming the action",
            )
        ],
        run=run,
        requires_scope=SCOPE_PREPARE_ACTION,
        injected=frozenset({"pending"}),
    )


# -- shared -----------------------------------------------------------------


def _propose(
    context: ToolContext,
    *,
    kind: ActionKind,
    payload: dict[str, Any],
    evidence_ids: Sequence[str],
) -> ToolPending | ToolError:
    """Check consistency, then mint. Blocking conflicts never get a token."""
    blocking, advisories = _consistency(context, evidence_ids)
    if blocking:
        return ToolError(
            "refusing to prepare this action: the records it rests on disagree. "
            + "; ".join(blocking)
            + ". Resolve the conflict, or say what is inconsistent and escalate.",
            recoverable=False,
        )

    pending = build_pending(
        kind=kind,
        payload=payload,
        evidence_chain=tuple(evidence_ids),
        session_id=context.session_id,
        thread_id=context.thread_id,
        advisories=advisories,
    )
    return ToolPending(pending=pending, token=mint_token(pending, secret=context.action_secret))


def _consistency(
    context: ToolContext, evidence_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Blocking and advisory conflict details across the evidence chain (D19)."""
    if not evidence_ids or context.runtime is None:
        return (), ()

    checker = ConsistencyChecker(
        store=context.store, repository=context.repository, resolver=context.resolver
    )
    blocking: list[str] = []
    advisory: list[str] = []
    for handle in evidence_ids:
        try:
            if context.store.kind_of(handle) not in _CHECKABLE:
                continue
            report = checker.check(snapshot_id=handle)
        except EvidenceError:
            # A handle this run does not hold. Not a conflict, and not
            # something to prepare an action on either.
            logger.warning("evidence handle %s is not readable in this run", handle)
            continue
        for conflict in report.conflicts:
            bucket = blocking if conflict.severity.value == "blocking" else advisory
            bucket.append(f"{conflict.conflict_class.value}: {conflict.detail}")
    return tuple(blocking), tuple(advisory)


__all__ = ["PROPOSABLE_KINDS", "approve_credit", "execute_action", "prepare_action"]
