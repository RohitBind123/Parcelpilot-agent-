"""Shared plumbing for the calculators.

Every calculator does the same three things before it computes anything: read a
snapshot, read a resolution and check it belongs with that snapshot, then mint a
result handle recording both as provenance. Doing it in one place means the
topic and account checks cannot be present in one calculator and forgotten in
the next, which is the only way that class of bug ever happens.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Protocol

from src.domain.calculators.errors import NoBasis, WrongEvidence
from src.domain.evidence import EvidenceKind, EvidenceStore, Handle
from src.domain.resolver import ClauseRef, PolicyResolution, UnresolvedConflict


class _HasPayload(Protocol):
    def to_payload(self) -> dict[str, Any]: ...


def load_resolution(
    store: EvidenceStore,
    resolution_id: Handle | str,
    *,
    topic: str,
    account_id: str | None,
    require_basis: bool = True,
) -> PolicyResolution:
    """A resolution that belongs with this snapshot, or an exception.

    The two checks are cheap and both guard silent-wrong-answer bugs. A
    resolution for the wrong topic computes something plausible from the wrong
    rule; one for the wrong account applies another customer's contract. In
    both cases the arithmetic succeeds, which is what makes them dangerous.
    """
    payload = store.read(resolution_id, expect=EvidenceKind.POLICY_RESOLUTION)

    if payload.get("topic") != topic:
        raise WrongEvidence(
            f"resolution is for {payload.get('topic')!r}, but {topic!r} is required here"
        )
    if payload.get("account_id") != account_id:
        raise WrongEvidence(
            "resolution and snapshot describe different accounts "
            f"({payload.get('account_id')} vs {account_id})"
        )

    resolution = rehydrate(payload)
    if require_basis and not resolution.has_basis:
        raise NoBasis(
            f"nothing governs {topic!r} for this account"
            if resolution.unresolved_conflict is None
            else f"unresolved conflict at tier {resolution.unresolved_conflict.tier}"
        )
    return resolution


def rehydrate(payload: dict[str, Any]) -> PolicyResolution:
    """Rebuild a resolution from its stored payload.

    Round-tripping through the evidence store rather than passing the object
    means a calculator can only see what was actually recorded. A field missing
    from the payload is missing here too, instead of quietly working in-process
    and failing the first time the API serialises it.
    """

    def ref(entry: dict[str, Any] | None) -> ClauseRef | None:
        if entry is None:
            return None
        doc_title, _, clause_ref = entry["citation"].rpartition(" ")
        return ClauseRef(
            clause_id=entry["clause_id"],
            doc_id=entry["clause_id"].split("::")[0],
            doc_title=doc_title,
            clause_ref=clause_ref,
            title=entry.get("title", ""),
            tier=entry["tier"],
            account_id=entry.get("account_id"),
            status="",
            params=entry.get("params", {}),
            reason=entry.get("reason"),
        )

    conflict = payload.get("unresolved_conflict")
    return PolicyResolution(
        topic=payload["topic"],
        account_id=payload.get("account_id"),
        governing=ref(payload.get("governing")),
        overridden=tuple(ref(e) for e in payload.get("overridden", [])),
        deferred=tuple(ref(e) for e in payload.get("deferred", [])),
        supporting=tuple(ref(e) for e in payload.get("supporting", [])),
        excluded=tuple(ref(e) for e in payload.get("excluded", [])),
        unresolved_conflict=(
            UnresolvedConflict(
                tier=conflict["tier"],
                clauses=tuple(ref(e) for e in conflict["clauses"]),
                differing_params=tuple(conflict["differing_params"]),
            )
            if conflict
            else None
        ),
    )


def attribution(resolution: PolicyResolution, operative: str | None) -> dict[str, Any]:
    """Which clause decided *this* answer, and what it actually displaced.

    Not the same as the topic-level winner. Northstar section 2 governs the
    cancellation fee, but its waiver is scoped to BOOKED shipments - so for a
    PICKED_UP order the agreement decides nothing and the SOP's status rule
    decides everything. Claiming an override there would be false in the
    direction that reads as generous.

    An override is therefore only reported when the clause that supplied the
    operative value is the one the resolver ranked first.
    """
    governing = resolution.governing
    decided_by_governing = (
        operative is not None and governing is not None and operative == governing.clause_id
    )
    return {
        "governing_clause": operative or (governing.clause_id if governing else ""),
        "overridden_clauses": (
            tuple(c.clause_id for c in resolution.overridden) if decided_by_governing else ()
        ),
    }


def mint_result(
    store: EvidenceStore,
    outcome: _HasPayload,
    *sources: Handle | str,
) -> Any:
    """Record the result and stamp its handle onto the returned outcome."""
    handle = store.mint(
        EvidenceKind.CALC_RESULT,
        json.loads(json.dumps(outcome.to_payload())),
        derived_from=list(sources),
    )
    return replace(outcome, calc_id=handle.evidence_id)
