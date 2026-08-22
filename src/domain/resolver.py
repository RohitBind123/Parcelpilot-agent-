"""Deterministic precedence over the clause registry (ARCHITECTURE section 6).

Retrieval decides what is *relevant*. This decides what is *binding*. Keeping
them apart is the point: a system that answers from the best-matching clause
will quote a general policy at a customer whose contract says otherwise, and the
citation will look impeccable.

The rule is quoted, not invented - Support Policy v3 section 1 says to use the
signed customer agreement first, then the current support policy, then current
product documentation, and that historical tickets are context only. The tier
ladder in the registry is that sentence, made queryable.

Three things here are less obvious than the ladder:

**A Tier 1 clause existing is not a Tier 1 clause winning.** LumenWorks section
2 is an agreement clause that says to use the standard SOP. It carries
`overrides: false` and must not govern - but it must still be cited, because
"your agreement was checked and defers" is a different answer from "you have no
agreement".

**`overrides: null` is not `overrides: false`.** Null means "I am the baseline".
Reading the two as the same leaves nothing governing at all.

**A clause can share a topic without stating a rule.** Three Tier 2 clauses
carry `first_response_target`; only one holds the grid. The other two are the
precedence preamble and the escalation duty. Treating every same-tier clause as
a rival authority reports an unresolved conflict on every SLA question in the
pack.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from src.auth.principal import Principal
from src.clock import as_of

#: Tiers that may support an answer. 4 is deprecated, 5 is a historical ticket
#: resolution; both may inform an answer and neither may justify one.
CITABLE_TIERS: Final[frozenset[int]] = frozenset({0, 1, 2, 3})


class _GeneralPolicy:
    """Sentinel: resolve against unscoped policy only, with no account.

    Staff must say which account they are resolving for, because precedence
    depends on whose agreement applies and a wrong guess returns general policy
    that looks authoritative. But "what changed between policy v2 and v3?" is a
    real question with no account behind it, so the account-less case has to be
    reachable - just never by omission.
    """

    def __repr__(self) -> str:
        return "GENERAL_POLICY"


GENERAL_POLICY: Final = _GeneralPolicy()

#: Keys that describe how a clause relates to others rather than what it says.
#: Two clauses differing only in these are not in conflict about anything.
_RELATIONAL_KEYS: Final[frozenset[str]] = frozenset({"overrides", "waivable_by_agreement"})

#: The topic under which a known issue is itself the subject rather than a
#: qualification of someone else's rule.
KNOWN_ISSUE_TOPIC: Final = "known_issue"

_QUERY: Final = """
    SELECT DISTINCT
        cl.clause_id, cl.doc_id, cl.doc_title, cl.clause_ref, cl.title,
        cl.tier, cl.account_id, cl.status,
        cl.effective_from, cl.effective_to, cl.superseded_by,
        cl.params, cl.text,
        EXISTS (
            SELECT 1 FROM clause_topics k
            WHERE k.clause_id = cl.clause_id AND k.topic = :known_issue
        ) AS is_known_issue
    FROM clauses cl
    JOIN clause_topics t ON t.clause_id = cl.clause_id
    WHERE t.topic = :topic
      AND (cl.account_id IS NULL OR cl.account_id = :account)
    ORDER BY cl.tier, cl.clause_id
"""


class ResolverError(RuntimeError):
    """The registry could not be read."""


@dataclass(frozen=True, slots=True)
class ClauseRef:
    """A clause, with everything needed to cite it and to compute from it."""

    clause_id: str
    doc_id: str
    doc_title: str
    clause_ref: str
    title: str
    tier: int
    account_id: str | None
    status: str
    params: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    #: Tagged `known_issue`. A defect report describes a deviation from a rule
    #: and is never the rule; see `_states_the_rule`.
    is_known_issue: bool = False
    #: Why this clause is in the bucket it is in. Set on exclusions.
    reason: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.doc_title} {self.clause_ref}".strip()

    @property
    def states_a_rule(self) -> bool:
        """Whether the clause is an authority on its topic, or only speaks about it.

        Two ways to be on a topic without being its rule.

        A clause whose only params are relational says nothing that could
        conflict with another clause. Policy v3 section 1 is on the
        `first_response_target` topic and states no target.

        And a known issue reports a defect - a departure from the rule that is
        expected to be repaired - so it can qualify an answer but never be its
        basis. KI-208 says as much itself: uploads above roughly 3,000 rows fail
        intermittently "even though the supported product limit remains 5,000
        rows". Treating it as a rival authority on `bulk_upload_limit` handed
        the topic to whichever clause id sorted first, which was the defect
        report, and turned an open bug into a contractual entitlement. It stays
        reachable as a supporting clause, which is where an answer wants it.
        """
        if self.is_known_issue:
            return False
        return any(key not in _RELATIONAL_KEYS for key in self.params)

    @property
    def declines_to_override(self) -> bool:
        """`overrides: false` - present, current, and deliberately not governing."""
        return self.params.get("overrides") is False

    def with_reason(self, reason: str) -> ClauseRef:
        from dataclasses import replace

        return replace(self, reason=reason)

    def to_payload(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "citation": self.citation,
            "title": self.title,
            "tier": self.tier,
            "account_id": self.account_id,
            "params": dict(self.params),
            **({"reason": self.reason} if self.reason else {}),
        }


@dataclass(frozen=True, slots=True)
class UnresolvedConflict:
    """Two clauses of equal authority stating different things.

    No such case exists in this pack (findings section 8). The branch exists
    because a resolver that cannot represent a conflict will silently pick a
    side, and silently picking is the failure the whole design avoids.
    """

    tier: int
    clauses: tuple[ClauseRef, ...]
    differing_params: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "clauses": [c.to_payload() for c in self.clauses],
            "differing_params": list(self.differing_params),
        }


@dataclass(frozen=True, slots=True)
class PolicyResolution:
    """What governs this topic for this account, and what it displaced."""

    topic: str
    account_id: str | None
    governing: ClauseRef | None = None
    #: Weaker-tier rules the governing clause displaced. Surfacing these is the
    #: Problem 2 deliverable; never let the model pick a winner silently.
    overridden: tuple[ClauseRef, ...] = ()
    #: Stronger-tier clauses that exist and explicitly decline to override.
    #: Not in the architecture's original sketch - added because GS-002 needs
    #: LumenWorks section 2 cited while not governing, and no other bucket
    #: means that. `context_only` would file a live agreement beside a
    #: deprecated policy.
    deferred: tuple[ClauseRef, ...] = ()
    #: On-topic, current, citable, but stating no competing value.
    supporting: tuple[ClauseRef, ...] = ()
    #: Deprecated, superseded, or outside its effective window.
    excluded: tuple[ClauseRef, ...] = ()
    unresolved_conflict: UnresolvedConflict | None = None
    #: The tiers this resolution was built with. `citable` filters on these
    #: rather than on a constant, so a caller who deliberately asked for the
    #: deprecated policy gets it back - and one who did not, cannot.
    included_tiers: frozenset[int] = CITABLE_TIERS

    @property
    def has_basis(self) -> bool:
        """Whether an answer may be computed at all.

        False when nothing governs, and false when a conflict stands - a
        calculator must refuse both rather than guess.
        """
        return self.governing is not None and self.unresolved_conflict is None

    @property
    def is_override(self) -> bool:
        return bool(self.overridden)

    @property
    def citable(self) -> tuple[ClauseRef, ...]:
        refs = [*self.deferred, *self.supporting, *self.overridden]
        if self.governing is not None:
            refs.insert(0, self.governing)
        return tuple(r for r in refs if r.tier in self.included_tiers)

    @property
    def all_clauses(self) -> tuple[ClauseRef, ...]:
        conflicted = self.unresolved_conflict.clauses if self.unresolved_conflict else ()
        return tuple(
            [
                *([self.governing] if self.governing is not None else []),
                *self.overridden,
                *self.deferred,
                *self.supporting,
                *self.excluded,
                *conflicted,
            ]
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "account_id": self.account_id,
            "governing": self.governing.to_payload() if self.governing else None,
            "overridden": [c.to_payload() for c in self.overridden],
            "deferred": [c.to_payload() for c in self.deferred],
            "supporting": [c.to_payload() for c in self.supporting],
            "excluded": [c.to_payload() for c in self.excluded],
            "unresolved_conflict": (
                self.unresolved_conflict.to_payload() if self.unresolved_conflict else None
            ),
            "has_basis": self.has_basis,
            "is_override": self.is_override,
        }


class PolicyResolver:
    """Resolves a topic to its governing clause for one account."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    @contextmanager
    def open(cls, db_path: Path | str) -> Iterator[PolicyResolver]:
        path = Path(db_path)
        if not path.exists():
            raise ResolverError(f"no database at {path}; run scripts/build_db.py")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            yield cls(connection)
        finally:
            connection.close()

    def resolve(
        self,
        topic: str,
        principal: Principal,
        *,
        account_id: str | _GeneralPolicy | None = None,
        include_tiers: Sequence[int] | frozenset[int] | set[int] = CITABLE_TIERS,
        moment: datetime | None = None,
    ) -> PolicyResolution:
        """The governing clause for `topic`, as it applies to one account."""
        scope = self._scope(principal, account_id)
        now = (moment or as_of()).date()
        tiers = frozenset(include_tiers)

        rows = self.connection.execute(
            _QUERY, {"topic": topic, "account": scope, "known_issue": KNOWN_ISSUE_TOPIC}
        ).fetchall()
        candidates, excluded = [], []
        for row in rows:
            ref = _to_ref(row)
            if ref.tier not in tiers:
                excluded.append(ref.with_reason(_exclusion_reason(ref)))
            elif not _effective(row, now):
                excluded.append(ref.with_reason("not_effective"))
            else:
                candidates.append(ref)

        return _decide(topic, scope, candidates, tuple(excluded), tiers)

    @staticmethod
    def _scope(principal: Principal, account_id: str | _GeneralPolicy | None) -> str | None:
        if account_id is GENERAL_POLICY:
            return None
        if principal.is_staff:
            if not account_id:
                # Staff read every account, so "which account" is not implied.
                # Guessing would silently return general policy and drop every
                # agreement - a wrong answer that looks like a right one. Pass
                # GENERAL_POLICY to say that is what you meant.
                raise ValueError(
                    "resolving policy as staff requires an explicit account_id "
                    "(or GENERAL_POLICY): precedence depends on whose agreement applies"
                )
            return str(account_id)
        if account_id and account_id != principal.account_id:
            raise PermissionError(
                f"principal scoped to {principal.account_id} cannot resolve for {account_id}"
            )
        return principal.account_id


# -- decision ---------------------------------------------------------------


def _decide(
    topic: str,
    account_id: str | None,
    candidates: list[ClauseRef],
    excluded: tuple[ClauseRef, ...],
    tiers: frozenset[int],
) -> PolicyResolution:
    """Walk the tiers from strongest, and stop at the first that asserts a rule."""
    rules = [c for c in candidates if c.states_a_rule]
    supporting = [c for c in candidates if not c.states_a_rule]
    deferred: list[ClauseRef] = []

    for tier in sorted({c.tier for c in rules}):
        group = [c for c in rules if c.tier == tier]

        asserting = [c for c in group if not c.declines_to_override]
        deferred.extend(c for c in group if c.declines_to_override)
        if not asserting:
            # Every clause at this tier defers. Keep walking; the SOP below is
            # what they are deferring to.
            continue

        conflict = _conflict(tier, asserting)
        if conflict is not None:
            return PolicyResolution(
                topic=topic,
                account_id=account_id,
                deferred=tuple(deferred),
                supporting=tuple(supporting),
                excluded=excluded,
                unresolved_conflict=conflict,
                included_tiers=tiers,
            )

        governing, *twins = sorted(asserting, key=lambda c: c.clause_id)
        # Twins say the same thing, so citing one is complete - but dropping
        # the others would lose a clause the customer may have read.
        supporting.extend(twins)

        return PolicyResolution(
            topic=topic,
            account_id=account_id,
            governing=governing,
            overridden=tuple(c for c in rules if c.tier > tier and not c.declines_to_override),
            deferred=tuple(deferred),
            supporting=tuple(supporting),
            excluded=excluded,
            included_tiers=tiers,
        )

    return PolicyResolution(
        topic=topic,
        account_id=account_id,
        deferred=tuple(deferred),
        supporting=tuple(supporting),
        excluded=excluded,
        included_tiers=tiers,
    )


def _conflict(tier: int, group: list[ClauseRef]) -> UnresolvedConflict | None:
    if len(group) < 2:
        return None
    # Only keys that *every* clause in the group states can disagree. A clause
    # silent on a key is not asserting a different value for it; it is asserting
    # nothing, and `params.get(key)` returning None made silence look like
    # dissent. That is the missing-data-is-not-zero rule arriving in the
    # precedence layer, and it left `bulk_upload_limit` with no governing clause
    # at all - from a corpus that states the limit in plain words.
    shared = {k for k in group[0].params if k not in _RELATIONAL_KEYS}
    for clause in group[1:]:
        shared &= set(clause.params)
    differing = sorted(
        key for key in shared if len({json.dumps(c.params[key], sort_keys=True) for c in group}) > 1
    )
    if not differing:
        return None
    return UnresolvedConflict(
        tier=tier,
        clauses=tuple(sorted(group, key=lambda c: c.clause_id)),
        differing_params=tuple(differing),
    )


# -- helpers ----------------------------------------------------------------


def _to_ref(row: sqlite3.Row) -> ClauseRef:
    return ClauseRef(
        clause_id=row["clause_id"],
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        clause_ref=row["clause_ref"],
        title=row["title"],
        tier=int(row["tier"]),
        account_id=row["account_id"],
        status=row["status"],
        params=json.loads(row["params"] or "{}"),
        text=row["text"],
        is_known_issue=bool(row["is_known_issue"]),
    )


def _exclusion_reason(ref: ClauseRef) -> str:
    if ref.status.upper() == "DEPRECATED":
        return "deprecated"
    if ref.tier >= 5:
        return "historical"
    return "out_of_requested_tiers"


def _effective(row: sqlite3.Row, on: date) -> bool:
    start, end = row["effective_from"], row["effective_to"]
    if start and date.fromisoformat(start[:10]) > on:
        return False
    return not (end and date.fromisoformat(end[:10]) < on)
