"""Document search and policy resolution.

`search_policy` finds clauses; `resolve_policy` decides which of them governs.
Keeping them separate matters: retrieval is a guess ranked by similarity, and
precedence is a sort over the registry. Collapsing the two would make an
override contingent on recall, which is exactly the failure the clause registry
exists to prevent.

Tier 4 is excluded from both by default. It stays reachable for staff through
an explicit flag, because "what changed between v2 and v3?" is a real question
(GS-018) and answering it requires reading a superseded document on purpose -
which is a different act from citing one by accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.tools.base import Param, Tool, ToolError, ToolResult
from src.domain.evidence import EvidenceKind
from src.domain.resolver import GENERAL_POLICY
from src.knowledge.topics import Topic
from src.knowledge.vectorstore.base import CITABLE_TIERS, DEFAULT_K

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

#: Includes tier 4. Only reachable through the staff-only flag.
_WITH_DEPRECATED = frozenset(CITABLE_TIERS | {4})

TOPICS = tuple(sorted(t.value for t in Topic))


def search_policy(context: ToolContext) -> Tool:
    staff = context.principal.is_staff

    def run(
        query: str,
        topic: str | None = None,
        include_deprecated: bool = False,
        k: int = DEFAULT_K,
    ) -> ToolResult | ToolError:
        # Argument validation first, and the order matters. A bad topic is the
        # caller's mistake and is actionable; "search is unavailable" is not
        # recoverable, so reporting it first would stop the model correcting an
        # error it actually made.
        if topic is not None and topic not in TOPICS:
            return ToolError(f"unknown topic {topic!r}; expected one of {list(TOPICS)}")
        if context.retriever is None:
            return ToolError(
                "policy search is unavailable in this session; "
                "use resolve_policy if you know the topic",
                recoverable=False,
            )

        tiers = _WITH_DEPRECATED if (staff and include_deprecated) else CITABLE_TIERS
        # Passed into the search rather than applied to its output: the
        # retriever filters before ranking, so k means k instead of "k minus
        # whatever was dropped afterwards".
        chunks = context.retriever.retrieve(
            query,
            principal=context.principal,
            tiers=tiers,
            topics=(topic,) if topic else (),
            k=k,
        )
        return ToolResult(
            {
                "count": len(chunks),
                "clauses": [
                    {
                        "clause_id": c.clause_id,
                        "citation": f"{c.doc_title} {c.clause_ref}",
                        "tier": c.tier,
                        "title": c.title,
                        "text": c.text,
                        # Stated rather than implied, so a model quoting a
                        # superseded clause has been told it is superseded.
                        "citable": c.tier in CITABLE_TIERS,
                    }
                    for c in chunks
                ],
            }
        )

    params = [
        Param("query", "string", "What to search for, in the customer's own words."),
        Param("topic", "string", "Narrow to one topic.", required=False, enum=TOPICS),
        Param("k", "integer", f"How many clauses to return (default {DEFAULT_K}).", required=False),
    ]
    if staff:
        params.append(
            Param(
                "include_deprecated",
                "boolean",
                "Also return superseded documents. They are context only and "
                "must never be cited as current.",
                required=False,
            )
        )
    return Tool(
        name="search_policy",
        description=(
            "Search policy documents, agreements and product guides. Returns clauses "
            "with their tier; only citable clauses may be used as the basis of an answer."
        ),
        params=tuple(params),
        run=run,
    )


def resolve_policy(context: ToolContext) -> Tool:
    staff = context.principal.is_staff

    def run(
        topic: str,
        account_id: str | None = None,
        snapshot_id: str | None = None,
        include_deprecated: bool = False,
    ) -> ToolResult | ToolError:
        if topic not in TOPICS:
            return ToolError(f"unknown topic {topic!r}; expected one of {list(TOPICS)}")

        scope: Any = account_id
        if staff and account_id is None:
            if snapshot_id is None:
                return ToolError(
                    "resolving policy as staff needs an account: pass account_id, or pass "
                    "the snapshot_id of an order or ticket and the account will be read "
                    "from it. For a question about no particular customer, "
                    "pass account_id='GENERAL'."
                )
            resolved = _account_from_snapshot(context, snapshot_id)
            if resolved is None:
                return ToolError(f"{snapshot_id!r} is not a snapshot this session can read")
            scope = resolved
        elif staff and account_id == "GENERAL":
            scope = GENERAL_POLICY

        tiers = _WITH_DEPRECATED if (staff and include_deprecated) else CITABLE_TIERS
        resolution = context.resolver.resolve(
            topic, context.principal, account_id=scope, include_tiers=tiers
        )
        payload = resolution.to_payload()
        handle = context.store.mint(
            EvidenceKind.POLICY_RESOLUTION,
            payload,
            derived_from=[snapshot_id] if snapshot_id else [],
        )
        governing = resolution.governing
        return ToolResult(
            {
                "resolution_id": handle.evidence_id,
                "topic": topic,
                "governing_clause": governing.clause_id if governing else None,
                "governing_citation": governing.citation if governing else None,
                "is_override": resolution.is_override,
                "overridden": [c.clause_id for c in resolution.overridden],
                "deferred": [c.clause_id for c in resolution.deferred],
                "supporting": [c.clause_id for c in resolution.supporting],
                "unresolved_conflict": resolution.unresolved_conflict is not None,
                "has_basis": resolution.has_basis,
            }
        )

    params = [
        Param("topic", "string", "The subject to resolve precedence for.", enum=TOPICS),
        Param(
            "snapshot_id",
            "string",
            "A snapshot this resolution is for, recorded as provenance.",
            required=False,
        ),
    ]
    if staff:
        params.insert(
            1,
            Param(
                "account_id",
                "string",
                "Whose agreement applies. Use 'GENERAL' for a question about no "
                "particular customer.",
                required=False,
            ),
        )
        params.append(
            Param(
                "include_deprecated",
                "boolean",
                "Include superseded documents as context. Never citable.",
                required=False,
            )
        )
    return Tool(
        name="resolve_policy",
        description=(
            "Decide which clause governs a topic, applying the precedence ladder. "
            "Returns a resolution_id that the calculators require."
        ),
        params=tuple(params),
        run=run,
    )


def _account_from_snapshot(context: ToolContext, snapshot_id: str) -> str | None:
    """Read the account off a snapshot the caller already holds.

    Saves the model a guess it has no business making, and refuses rather than
    falling back: a resolution against the wrong account applies another
    customer's contract and the arithmetic still succeeds.
    """
    for kind in (
        EvidenceKind.ORDER_SNAPSHOT,
        EvidenceKind.TICKET_SNAPSHOT,
        EvidenceKind.ACCOUNT_SNAPSHOT,
    ):
        try:
            payload = context.store.read(snapshot_id, expect=kind)
        except Exception:  # a wrong kind or a foreign handle: try the next
            continue
        return payload.get("account_id")
    return None
