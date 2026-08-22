"""The fact block, the gate, and what happens when prose outruns its evidence.

    uv run python scripts/demo_m7.py
    uv run python scripts/demo_m7.py --live     # grade real model prose

Four drafts against one set of evidence. Three are wrong in ways that are easy
to miss when reading fluent prose, and each is caught by a different part of the
gate. Nothing here is illustrative: the block and the verdicts come from the
code the API will call.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.escalation import DeclineReason, draft
from src.agent.facts import compose
from src.agent.grounding import Verdict, check_figures, ground
from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import build_toolset
from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.domain.evidence import EvidenceKind
from src.knowledge.registry import load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.chroma import build_vector_store
from src.providers.registry import get_chat_provider, get_embedding_provider

WIDTH = 92

DRAFTS = (
    (
        "grounded",
        "Your agreement waives the cancellation fee on any BOOKED shipment before pickup, "
        "so the standard INR 250 charge after 30 minutes does not apply.",
        ("the agreement waives the cancellation fee on a BOOKED shipment before pickup",),
    ),
    (
        "invented figure",
        "A reduced cancellation fee of INR 175 applies to your order.",
        ("a reduced cancellation fee of INR 175 applies",),
    ),
    (
        "figure from a document nobody read",
        "The monthly service-credit cap on your account is INR 8,000.",
        ("the monthly service credit cap is INR 8,000",),
    ),
    (
        "unsupported claim",
        "The fee is waived, and refunds are processed within five working days.",
        ("the fee is waived", "refunds are processed within five working days"),
    ),
)


class Scripted:
    """Stands in for the cheap model that splits prose into claims."""

    def __init__(self, claims):
        self.claims = list(claims)

    def extract(self, _prose):
        return list(self.claims)


def wrap(text: str, indent: str = "      ") -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


def rule(title: str = "") -> None:
    print(f"\n{title}\n{'=' * WIDTH}" if title else "=" * WIDTH)


def gather(db_path: Path):
    """Walk the real chain for ORD-1001 and keep what it produced."""
    principal = to_principal(get_persona("northstar_customer"))
    settings = get_settings()
    chunks = load_chunks(db_path)
    retriever = HybridRetriever(
        dense=build_vector_store(settings, get_embedding_provider(settings)),
        lexical=BM25Index(chunks),
    )
    with open_tool_context(
        principal, run_id="demo_m7", db_path=db_path, retriever=retriever
    ) as context:
        tools = {t.name: t for t in build_toolset(context)}
        snapshot = tools["get_order"](order_id="ORD-1001").data["snapshot_id"]
        resolution_id = tools["resolve_policy"](
            topic="cancellation_fee", snapshot_id=snapshot
        ).data["resolution_id"]
        calculation = tools["compute_cancellation_fee"](
            snapshot_id=snapshot, resolution_id=resolution_id
        ).data
        conflicts = tools["check_data_consistency"](snapshot_id=snapshot).data
        found = tools["search_policy"](
            query="cancellation fee waiver first response target", topic="cancellation_fee"
        ).data
        sla_clauses = tools["search_policy"](
            query="first response target", topic="first_response_target"
        ).data
        resolution = context.store.read(resolution_id, expect=EvidenceKind.POLICY_RESOLUTION)
        sources = {
            c["clause_id"]: c["text"]
            for c in [*found["clauses"], *sla_clauses["clauses"]]
            if c["citable"]
        }
    return calculation, resolution, conflicts, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="grade prose a real model wrote")
    args = parser.parse_args()

    db_path = get_settings().db_path
    calculation, resolution, conflicts, sources = gather(db_path)
    block = compose(calculation=calculation, resolution=resolution, conflicts=conflicts)

    rule()
    print("ParcelPilot M7 - Python states the facts, the model writes around them")
    rule("THE FACT BLOCK (rendered by Python, the model cannot edit it)")
    print()
    for line in block.render().splitlines():
        print(f"  {line}")
    print(f"\n  grounded quantities: {sorted((v, u or '-') for v, u in block.figures)}")
    print(f"  citable clauses:     {len(block.citable)}")
    print(f"  citable sources read: {len(sources)}")

    rule("FOUR DRAFTS AGAINST THAT EVIDENCE")
    extractor_cls = Scripted
    if args.live:
        from src.agent.claims_llm import LlmClaimExtractor

        print("\n  (--live: claims extracted by the real model)")

    for label, prose, claims in DRAFTS:
        extractor = LlmClaimExtractor(get_chat_provider()) if args.live else extractor_cls(claims)
        outcome = ground(prose, block=block, sources=sources, extractor=extractor)
        print(f"\n  {label.upper()}")
        print(wrap(prose))
        print(f"      -> {outcome.verdict.value}")
        for figure, unit in outcome.invented_figures:
            print(f"         figure not in evidence: {figure:g} {unit or '(no unit)'}")
        for failure in outcome.failures:
            print(f"         unsupported: {failure.claim.text}")
        if outcome.verdict is not Verdict.PASSED:
            print("         -> the prose is dropped; the block above still stands")

    rule("WHY THE UNIT MATTERS")
    print()
    print(wrap('Policy v3 §3 says Enterprise P1 is "30 minutes, 24x7" and, in the same', "  "))
    print(wrap('grid, "1 business day". So the bare number 1 IS grounded - and a gate', "  "))
    print(wrap('checking bare numbers accepts "the target is 1 hour", the deprecated v2', "  "))
    print(wrap("answer GS-017 exists to catch. Pairing each value with its unit:", "  "))
    print()
    print(wrap("(Checked against the v3 grid alone, as an account with no agreement", "  "))
    print(wrap("would see it. Northstar's own agreement does say 1 hour, so for that", "  "))
    print(wrap("session the figure is genuinely grounded - which is the point.)", "  "))
    print()
    grid = {k: v for k, v in sources.items() if "policy_v3" in k}
    for prose in (
        "The target is 1 hour.",
        "The target is 1 business day.",
        "The target is 30 minutes.",
    ):
        bad = check_figures(prose, compose(), grid)
        verdict = "REJECTED " + str([f"{v:g} {u}" for v, u in bad]) if bad else "grounded"
        print(f"    {prose:<34} {verdict}")

    rule("A QUESTION WITH NO SOURCE")
    record = draft(
        principal=to_principal(get_persona("beacon_customer")),
        thread_id="demo",
        question="How do we change the billing contact on our account?",
        reason=DeclineReason.NO_CITABLE_SOURCE,
        subject="how to change a billing contact",
        sources_consulted=tuple(sources),
    )
    print()
    print(wrap(record.summary, "  "))
    print(f"\n  drafted record: {record.to_payload()['kind']}")
    print(f"  gap:            {record.what_is_unresolved}")
    print(f"  sources read:   {len(record.sources_consulted)}")
    print("\n  Nothing is created until the confirmation gate. The record names the")
    print("  gap rather than inventing a settings page, which is the failure this")
    print("  question exists to catch.")
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
