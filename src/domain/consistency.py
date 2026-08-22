"""`check_data_consistency` (D19): the conflicts, found before the model writes.

SOP v4 section 3, verbatim: "When data conflicts, identify the conflict and
request verification before a state-changing action." That sentence is why this
is a tool producing a handle rather than a warning string on a calculator. A
warning is advice the model may or may not repeat; a `blocking` conflict on a
report is something `prepare_action` can refuse to mint a token against.

Four classes, three of which the pack contains for real. Their severities differ
because what they threaten differs:

  stale_status            blocking  - the stored fact may be false, so acting on
                                      it is acting on something unverified
  unresolved_same_tier    blocking  - no rule has been established, so there is
                                      nothing to act under
  missing_source          blocking  - same, by absence rather than by tie
  historical_contradiction advisory - the current data is not in doubt; a past
                                      answer was wrong and must be corrected

The hard part is not detection, it is restraint. ORD-2001 is BOOKED, carried by
the same carrier, with the same missing pickup timestamp as ORD-1001, and it is
not stale - nobody has said a driver came. A detector that cannot tell those two
apart makes every answer arrive hedged, and a hedge that is always present
carries no information at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from src.datastore.repo import Repository
from src.domain.claims import Claim, extract_claims
from src.domain.evidence import EvidenceKind, EvidenceKindError, EvidenceStore, Handle
from src.domain.resolver import ClauseRef, PolicyResolution, PolicyResolver

#: A known issue can corroborate a conflict only while it is open. KI-176 is
#: resolved and says so in terms: "Do not use this resolved issue to explain new
#: incidents." Anything else - including a missing status, which is equally what
#: a broken extractor produces - cannot corroborate.
_ACTIVE_ISSUE_STATUSES: Final[frozenset[str]] = frozenset({"investigating", "monitoring"})

#: A ticket claiming the parcel has physically moved. Deliberately narrow: it
#: has to be an assertion about collection, not a mention of the word pickup.
_COLLECTION_CLAIMED: Final = re.compile(
    r"\b(driver\s+(collected|picked|has\s+been)|collected\s+the\s+parcel|"
    r"after\s+driver\s+pickup|parcel\s+(was|has\s+been)\s+(collected|picked\s+up)|"
    r"picked\s+up\s+(the|our)\s+parcel)\b",
    re.IGNORECASE,
)

#: Statuses where a late pickup webhook could still be in flight. A DELIVERED or
#: CANCELLED order has moved past the point KI-211 describes.
_AWAITING_PICKUP: Final[frozenset[str]] = frozenset({"BOOKED"})

_KNOWN_ISSUE_TOPIC: Final = "known_issue"


class ConflictClass(StrEnum):
    STALE_STATUS = "stale_status"
    HISTORICAL_CONTRADICTION = "historical_contradiction"
    UNRESOLVED_SAME_TIER = "unresolved_same_tier"
    MISSING_SOURCE = "missing_source"


class ConflictSeverity(StrEnum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class Conflict:
    conflict_class: ConflictClass
    severity: ConflictSeverity
    detail: str
    #: Every record and clause that took part, so the trace can show the whole
    #: disagreement rather than the conclusion drawn from it.
    sources: tuple[str, ...]
    confidence: float
    #: The clause the correction rests on. Never a Tier 5 source.
    basis_clause: str | None = None
    #: What a closed ticket asserted, and what the governing clause says now.
    claimed_value: int | None = None
    current_value: int | None = None
    #: Present when a link between records was inferred rather than read (A3).
    inference_note: str | None = None
    #: What the corpus says to do about it, in the corpus's own words.
    instruction: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "conflict_class": self.conflict_class.value,
            "severity": self.severity.value,
            "detail": self.detail,
            "sources": list(self.sources),
            "confidence": self.confidence,
            "basis_clause": self.basis_clause,
            "claimed_value": self.claimed_value,
            "current_value": self.current_value,
            "inference_note": self.inference_note,
            "instruction": self.instruction,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    report_id: Handle
    subject_id: str
    subject_kind: str
    conflicts: tuple[Conflict, ...]
    #: What was looked for. A report saying "no conflicts" means nothing without
    #: it: silence from a check that never ran reads exactly the same.
    checked: tuple[ConflictClass, ...] = field(default_factory=tuple)

    @property
    def blocking(self) -> bool:
        return any(c.severity is ConflictSeverity.BLOCKING for c in self.conflicts)

    def to_payload(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "blocking": self.blocking,
            "checked": [c.value for c in self.checked],
            "conflicts": [c.to_payload() for c in self.conflicts],
        }


class ConsistencyChecker:
    """Cross-references one snapshot against the rest of what is known."""

    def __init__(
        self, *, store: EvidenceStore, repository: Repository, resolver: PolicyResolver
    ) -> None:
        self._store = store
        self._repo = repository
        self._resolver = resolver

    def check(self, *, snapshot_id: Handle | str, topics: Sequence[str] = ()) -> ConsistencyReport:
        """Every conflict this snapshot is party to.

        `topics` is a documented extension of the tool signature in
        ARCHITECTURE section 10. `missing_source` and `unresolved_same_tier` are
        properties of a *question*, not of a row, and inferring the question
        from a ticket subject with a keyword map is the guessing this design
        refuses everywhere else. The caller names the topics under discussion;
        with none named, only the snapshot-intrinsic checks run.
        """
        kind = self._store.kind_of(snapshot_id)
        payload = self._store.read(snapshot_id, expect=kind)

        conflicts: list[Conflict] = []
        checked: list[ConflictClass] = []

        if kind is EvidenceKind.ORDER_SNAPSHOT:
            subject_id = str(payload["order_id"])
            checked.append(ConflictClass.STALE_STATUS)
            conflicts.extend(self._stale_status(payload))
        elif kind is EvidenceKind.TICKET_SNAPSHOT:
            subject_id = str(payload["ticket_id"])
            checked.append(ConflictClass.HISTORICAL_CONTRADICTION)
            conflicts.extend(self._historical_contradiction(payload))
        else:
            # EvidenceKindError rather than TypeError so the tool layer's one
            # handler turns it into a structured ToolError the model can read
            # and correct, like every other wrong-handle mistake.
            raise EvidenceKindError(
                f"consistency is checked against an order or ticket snapshot, not {kind.value}"
            )

        if topics:
            checked.extend((ConflictClass.MISSING_SOURCE, ConflictClass.UNRESOLVED_SAME_TIER))
            conflicts.extend(self._topic_conflicts(payload, topics))

        report_id = self._store.mint(
            EvidenceKind.CONSISTENCY_REPORT,
            {
                "subject_id": subject_id,
                "subject_kind": kind.value,
                "blocking": any(c.severity is ConflictSeverity.BLOCKING for c in conflicts),
                "checked": [c.value for c in checked],
                "conflicts": [c.to_payload() for c in conflicts],
            },
            derived_from=[snapshot_id],
        )
        return ConsistencyReport(
            report_id=report_id,
            subject_id=subject_id,
            subject_kind=kind.value,
            conflicts=tuple(conflicts),
            checked=tuple(checked),
        )

    # -- stale status ------------------------------------------------------

    def _stale_status(self, order: dict[str, Any]) -> list[Conflict]:
        """ORD-1001: the record, a ticket, and a known issue, none agreeing.

        All three legs are required. The status must be one a late webhook could
        still change, some open ticket must actually assert the parcel moved,
        and a current known issue must explain how both can be true at once.
        Drop any leg and the detector fires on ordinary orders.
        """
        if order.get("status") not in _AWAITING_PICKUP or order.get("pickup_actual_at"):
            return []

        issue = self._active_issue_for(str(order.get("carrier", "")))
        if issue is None:
            return []

        witness = self._collection_witness(str(order["account_id"]))
        if witness is None:
            return []

        ticket, note, confidence = witness
        return [
            Conflict(
                conflict_class=ConflictClass.STALE_STATUS,
                severity=ConflictSeverity.BLOCKING,
                detail=(
                    f"{order['order_id']} is recorded as {order['status']} with no pickup "
                    f"confirmation, but {ticket['ticket_id']} reports the parcel was collected. "
                    f"{issue.clause_ref} says {order['carrier']} pickup webhooks can arrive up to "
                    f"{issue.params.get('delay_minutes')} minutes late, so the stored status may "
                    "be stale. Which is true cannot be confirmed from ParcelPilot's data."
                ),
                sources=(str(order["order_id"]), str(ticket["ticket_id"]), issue.clause_id),
                confidence=confidence,
                basis_clause=issue.clause_id,
                inference_note=note,
                instruction=_instruction(issue),
            )
        ]

    def _active_issue_for(self, carrier: str) -> ClauseRef | None:
        """A current known issue about this carrier's pickup reporting."""
        if not carrier:
            return None
        resolution = self._resolver.resolve(
            _KNOWN_ISSUE_TOPIC, self._repo.principal, account_id=self._scope()
        )
        for ref in resolution.all_clauses:
            status = str(ref.params.get("issue_status", "")).lower()
            if status in _ACTIVE_ISSUE_STATUSES and ref.params.get("carrier") == carrier:
                return ref
        return None

    def _collection_witness(
        self, account_id: str
    ) -> tuple[dict[str, Any], str | None, float] | None:
        """An open ticket on this account claiming the parcel was collected.

        The link back to a specific order is an inference, because the tickets
        table has no `order_id` (assumption A3). It is returned with the reason
        it was drawn and a confidence below 1.0, so an answer can state it as
        the inference it is. Reporting it as fact is how a caveat silently
        becomes a claim.
        """
        for ticket in self._repo.list_tickets(account_id):
            text = f"{ticket.subject}\n{ticket.description or ''}"
            if ticket.status.lower() == "closed" or not _COLLECTION_CLAIMED.search(text):
                continue
            payload = {"ticket_id": ticket.ticket_id, "subject": ticket.subject}
            return payload, self._link_note(account_id, ticket.ticket_id, text), 0.8
        return None

    def _link_note(self, account_id: str, ticket_id: str, text: str) -> str | None:
        """Why this ticket is thought to be about this order."""
        carriers = {
            order.carrier
            for order in self._repo.list_orders(account_id)
            if order.status in _AWAITING_PICKUP and order.pickup_actual_at is None
        }
        named = [c for c in carriers if re.search(re.escape(c), text, re.IGNORECASE)]
        if len(named) == 1:
            matching = [
                order.order_id
                for order in self._repo.list_orders(account_id)
                if order.carrier == named[0]
                and order.status in _AWAITING_PICKUP
                and order.pickup_actual_at is None
            ]
            if len(matching) == 1:
                return (
                    f"{ticket_id} names {named[0]} and does not name an order; tickets carry no "
                    f"order reference. {matching[0]} is the only {named[0]} shipment on this "
                    "account still awaiting pickup confirmation, so the two are taken to be the "
                    "same shipment. That link is inferred, not recorded."
                )
        return (
            f"{ticket_id} carries no order reference, so the shipment it describes is inferred "
            "from the account alone."
        )

    # -- historical contradiction -----------------------------------------

    def _historical_contradiction(self, ticket: dict[str, Any]) -> list[Conflict]:
        """A Tier 5 recorded answer against the clause that governs today.

        Account-relative on purpose. "A INR 250 cancellation fee applied" is
        wrong for Northstar, whose agreement waives it, and right for an account
        with no agreement, where the SOP charges exactly that. A checker
        comparing against "the policy" rather than "this account's governing
        clause" gets the discriminating pair backwards.
        """
        recorded = ticket.get("historical_resolution")
        if not recorded:
            return []

        account_id = str(ticket["account_id"])
        conflicts = []
        for claim in extract_claims(recorded):
            resolution = self._resolver.resolve(
                claim.topic, self._repo.principal, account_id=account_id
            )
            governing = resolution.governing
            if governing is None:
                continue
            current = _current_value(governing, claim)
            if current is None or current == claim.value:
                continue
            explains = self._issue_explaining(claim, resolution)
            conflicts.append(
                Conflict(
                    conflict_class=ConflictClass.HISTORICAL_CONTRADICTION,
                    severity=ConflictSeverity.ADVISORY,
                    detail=(
                        f'{ticket["ticket_id"]} is closed and records "{claim.quote}", but '
                        f"{governing.citation} gives {claim.param.replace('_', ' ')} as {current} "
                        f"for this account. A closed ticket is context only and may contain "
                        "incorrect past guidance, so the recorded answer was wrong."
                    ),
                    sources=tuple(
                        filter(
                            None,
                            (
                                str(ticket["ticket_id"]),
                                governing.clause_id,
                                explains.clause_id if explains else None,
                            ),
                        )
                    ),
                    confidence=1.0,
                    basis_clause=governing.clause_id,
                    claimed_value=claim.value,
                    current_value=current,
                    instruction=_instruction(explains) if explains else None,
                )
            )
        return conflicts

    def _issue_explaining(self, claim: Claim, resolution: PolicyResolution) -> ClauseRef | None:
        """A current known issue whose own threshold is the number that was quoted.

        TKT-451 is the case. 3,000 is real - it is where uploads start failing -
        and it is not the plan limit. Correcting the number without saying where
        it came from tells the customer their 3,500-row upload should work.
        """
        for ref in resolution.all_clauses:
            if str(ref.params.get("issue_status", "")).lower() not in _ACTIVE_ISSUE_STATUSES:
                continue
            if claim.value in {v for v in ref.params.values() if isinstance(v, int)}:
                return ref
        return None

    # -- topic checks ------------------------------------------------------

    def _topic_conflicts(self, payload: dict[str, Any], topics: Sequence[str]) -> list[Conflict]:
        account_id = str(payload["account_id"])
        conflicts = []
        for topic in topics:
            resolution = self._resolver.resolve(topic, self._repo.principal, account_id=account_id)
            if resolution.unresolved_conflict is not None:
                clash = resolution.unresolved_conflict
                conflicts.append(
                    Conflict(
                        conflict_class=ConflictClass.UNRESOLVED_SAME_TIER,
                        severity=ConflictSeverity.BLOCKING,
                        detail=(
                            f"two tier-{clash.tier} clauses disagree about {topic!r} and neither "
                            "takes precedence, so no rule is established"
                        ),
                        sources=tuple(c.clause_id for c in clash.clauses),
                        confidence=1.0,
                    )
                )
            elif not resolution.has_basis:
                conflicts.append(
                    Conflict(
                        conflict_class=ConflictClass.MISSING_SOURCE,
                        severity=ConflictSeverity.BLOCKING,
                        detail=(
                            f"no citable clause covers {topic!r} for this account, so there is "
                            "no basis to answer from or to act under"
                        ),
                        sources=tuple(c.clause_id for c in resolution.excluded),
                        confidence=1.0,
                    )
                )
        return conflicts

    def _scope(self) -> str | None:
        """Known issues apply to everyone, so staff need not name an account."""
        from src.domain.resolver import GENERAL_POLICY

        principal = self._repo.principal
        return GENERAL_POLICY if principal.is_staff else principal.account_id


def _current_value(governing: ClauseRef, claim: Claim) -> int | None:
    """What the governing clause says, for the key the claim asserted.

    Two clauses can state the same fact under different names: the SOP calls it
    `fee_after_window_inr` because it has a window, and Northstar's waiver calls
    it `fee_inr` because it does not. Both answer "what is the fee".
    """
    for key in _EQUIVALENT.get(claim.param, (claim.param,)):
        value = governing.params.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


#: Parameter names that answer the same question on different clauses.
_EQUIVALENT: Final[dict[str, tuple[str, ...]]] = {
    "fee_after_window_inr": ("fee_after_window_inr", "fee_inr"),
    "free_window_minutes": ("free_window_minutes", "window_minutes"),
    "supported_rows": ("supported_rows",),
}


def _instruction(ref: ClauseRef | None) -> str | None:
    """The corpus's own words about what to do, not a paraphrase of them.

    KI-211 does not merely describe the webhook lag, it says to verify carrier
    status before telling a customer a pickup did not happen. KI-208 gives a
    workaround. Those sentences are the answer; a model rewriting them can drop
    the part that matters.
    """
    if ref is None or not ref.text:
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ref.text) if s.strip()]
    actionable = [
        s
        for s in sentences
        if re.search(r"\b(verify|workaround|split|do not|before telling|wait)\b", s, re.IGNORECASE)
    ]
    return " ".join(actionable) or None
