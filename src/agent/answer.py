"""Assembling a finished answer: facts, then prose, then the gate.

The pieces exist separately - `facts.py` renders, `grounding.py` grades,
`escalation.py` drafts - and this is where the order between them is fixed.
That order is the design:

1. Collect what the tools actually returned, from the conversation. Not from
   the model's summary of it, which is the thing being checked.
2. Render the fact block in Python.
3. The model's prose is graded against the block and the sources.
4. A failure that could plausibly be fixed by more evidence gets one targeted
   re-retrieval. A failure that could not - an invented figure - does not.
5. Out of budget, or nothing to answer from, and the prose is dropped. The exit
   is the fact block plus a drafted escalation, never a degraded answer.

Step 5 is the one worth defending. A system that shortens an unsupported answer
until it passes has not become more truthful, it has become vaguer, and vagueness
is harder to catch than a wrong number.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agent.escalation import DeclineReason, Escalation, draft
from src.agent.facts import FactBlock, compose
from src.agent.grounding import ClaimExtractor, GateOutcome, Verdict, ground
from src.auth.principal import Principal

logger = logging.getLogger(__name__)

#: One targeted re-retrieval, then a second, then stop (D16). The exit is
#: escalation; a third attempt only spends money on the same conclusion.
REPAIR_BUDGET = 2

_DECLINE_FOR = {
    Verdict.NO_BASIS: DeclineReason.NO_CITABLE_SOURCE,
    Verdict.UNCHECKED: DeclineReason.UNGROUNDED,
    Verdict.FAILED: DeclineReason.UNSUPPORTED_CLAIM,
}


@dataclass(frozen=True, slots=True)
class Answer:
    block: FactBlock
    prose: str
    gate: GateOutcome | None = None
    escalation: Escalation | None = None
    repairs: int = 0
    #: Queries already tried, so none is repeated (D16).
    attempted_queries: tuple[str, ...] = field(default_factory=tuple)

    @property
    def declined(self) -> bool:
        return self.escalation is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "facts": self.block.to_payload(),
            "prose": self.prose,
            "gate": self.gate.to_payload() if self.gate else None,
            "escalation": self.escalation.to_payload() if self.escalation else None,
            "repairs": self.repairs,
        }


def evidence_from(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the tools returned, read back out of the conversation.

    Read from the messages rather than recorded as the tools ran, because the
    messages are what the checkpointer persists. A run resumed in another
    process has the conversation and nothing else, and it must be able to
    rebuild the same fact block.
    """
    found: dict[str, Any] = {"sources": {}}
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("denied") or payload.get("error"):
            continue

        name = message.get("name", "")
        if name == "search_policy":
            for clause in payload.get("clauses", []):
                if clause.get("citable"):
                    found["sources"][clause["clause_id"]] = clause.get("text", "")
        elif name == "resolve_policy":
            found["resolution_summary"] = payload
        elif name.startswith("compute_"):
            found["calculation"] = payload
        elif name == "sla_first_response_status":
            found["sla"] = payload
        elif name == "check_data_consistency":
            found["conflicts"] = payload
    return found


def assemble(
    prose: str,
    *,
    messages: Sequence[Mapping[str, Any]],
    resolution: Mapping[str, Any] | None = None,
    principal: Principal,
    thread_id: str,
    question: str,
    extractor: ClaimExtractor,
    subject: str = "this question",
) -> Answer:
    """Render the facts, grade the prose, and decline rather than degrade.

    `resolution` is the full resolver payload when the caller has it; the tool
    result alone carries clause ids without their text, which is enough to cite
    and not enough to render the delta an override turns on.
    """
    evidence = evidence_from(messages)
    block = compose(
        calculation=evidence.get("calculation"),
        resolution=resolution,
        conflicts=evidence.get("conflicts"),
        sla=evidence.get("sla"),
    )
    sources = evidence["sources"]

    gate = ground(prose, block=block, sources=sources, extractor=extractor)
    if gate.verdict is Verdict.PASSED:
        return Answer(block=block, prose=prose, gate=gate)

    return Answer(
        block=block,
        prose="",
        gate=gate,
        escalation=draft(
            principal=principal,
            thread_id=thread_id,
            question=question,
            reason=_DECLINE_FOR[gate.verdict],
            subject=subject,
            evidence_chain=tuple(_handles(messages)),
            sources_consulted=tuple(sources),
        ),
    )


def repair_queries(gate: GateOutcome, attempted: Sequence[str]) -> list[str]:
    """What to search for next.

    A named gap makes a far better query than a rewritten question: "refunds are
    processed within five working days" says exactly what is missing, where
    rephrasing the user's words asks the same thing again.

    Nothing already attempted is returned - a query that found nothing the first
    time will find nothing the second, and the budget is small enough that one
    wasted attempt is half of it.
    """
    if not gate.repairable:
        return []
    seen = {q.lower() for q in attempted}
    return [f.claim.text for f in gate.failures if f.claim.text.lower() not in seen]


def is_new_evidence(retrieved: Sequence[str], already_held: Sequence[str]) -> bool:
    """The subset check (D16).

    If a rewrite returns only clauses already in hand, stop immediately. No new
    information cannot become support, and further rewrites are a way of
    spending budget while looking diligent.
    """
    return bool(set(retrieved) - set(already_held))


def _handles(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = ("snapshot_id", "account_snapshot_id", "resolution_id", "report_id", "calc_id")
    found = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        found.extend(str(payload[k]) for k in keys if payload.get(k))
    return found
