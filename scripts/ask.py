"""Ask the agent a question from a persona. No UI, no server.

    uv run python scripts/ask.py --persona northstar_customer \\
        --question "I need to cancel this shipment. Will I be charged a fee?"

    uv run python scripts/ask.py --demo          # the brief's two questions, all four sides
    uv run python scripts/ask.py --persona maya_agent --thread triage --trace

The trace is on by default because the tool chain is the interesting part: which
tools the model chose, what handles came back, and what it was refused. `--quiet`
turns it off when only the answer is wanted.

Exits non-zero if a run produced no answer, so this is usable as a smoke check.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.context import open_agent
from src.auth.personas import PERSONAS, get_persona, to_principal
from src.config import get_settings
from src.knowledge.registry import load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.chroma import build_vector_store
from src.providers.registry import get_chat_provider, get_embedding_provider

WIDTH = 92

#: The brief's two example questions, each from both sides of its
#: discriminating pair. Four answers; a system that hard-codes any of them
#: fails at least one.
DEMO = (
    (
        "northstar_customer",
        "I need to cancel order ORD-1001. Will I be charged a fee?",
        "agreement waives the fee; the BOOKED status may also be stale",
    ),
    (
        "lumenworks_customer",
        "I need to cancel order ORD-2001. Will I be charged a fee?",
        "same shape, but the agreement defers - INR 250 under the SOP",
    ),
    (
        "lumenworks_customer",
        "A pickup is three hours late because of carrier fault. Should I get a service credit?",
        "ineligible: the agreement replaces the 2-hour threshold with 4",
    ),
    (
        "beacon_customer",
        "A pickup is three hours late because of carrier fault. Should I get a service credit?",
        "eligible: no agreement, so the SOP's 2-hour threshold applies",
    ),
)


def wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(line, width=WIDTH, initial_indent=indent, subsequent_indent=indent)
        if line.strip()
        else ""
        for line in text.splitlines()
    )


def build_retriever(db_path: Path) -> HybridRetriever | None:
    """The configured store, the same one the API will use.

    Not a stand-in. `search_policy` reporting itself unavailable would send the
    model looking for a way around a tool that is merely unconfigured, and the
    answers it produced would not be the answers the product gives.
    """
    settings = get_settings()
    try:
        chunks = load_chunks(db_path)
        dense = build_vector_store(settings, get_embedding_provider(settings))
        if not dense.count():
            print("warning: the vector collection is empty; run scripts/build_index.py")
        return HybridRetriever(dense=dense, lexical=BM25Index(chunks))
    except Exception as exc:
        print(f"warning: policy search unavailable ({type(exc).__name__}: {exc})")
        return None


def ask(persona_id: str, question: str, *, thread: str, trace: bool, retriever) -> bool:
    principal = to_principal(get_persona(persona_id))
    provider = get_chat_provider()

    with open_agent(principal, provider=provider, retriever=retriever, run_id=thread) as agent:
        run = agent.ask(question, thread_id=thread)

    print(f"\n{'=' * WIDTH}")
    print(f"{persona_id}  ({principal.role})")
    print(f"{'-' * WIDTH}")
    print(wrap(question, "> "))

    if trace and run.tool_calls:
        print("\n  tools")
        for step in run.tool_calls:
            arguments = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items()) or ""
            marker = {"result": " ", "denied": "!", "error": "?"}[step.outcome]
            print(f"   {marker} {step.name}({arguments})")
            if step.handle:
                print(f"       -> {step.handle}")
            elif step.outcome != "result":
                print(f"       -> {step.message}")

    if run.stopped_early:
        print("\n  (stopped early: the tool-turn budget was exhausted)")

    print()
    print(wrap(run.answer or "(no answer)"))
    return bool(run.answer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona", default="northstar_customer", choices=[p.persona_id for p in PERSONAS]
    )
    parser.add_argument("--question")
    parser.add_argument("--thread", default="cli")
    parser.add_argument(
        "--demo", action="store_true", help="the brief's two questions, from all four sides"
    )
    parser.add_argument("--quiet", action="store_true", help="answer only, no tool trace")
    args = parser.parse_args()

    if not args.demo and not args.question:
        parser.error("pass --question or --demo")

    db_path = get_settings().db_path
    retriever = build_retriever(db_path)

    answered = True
    if args.demo:
        for index, (persona_id, question, expected) in enumerate(DEMO):
            answered &= ask(
                persona_id,
                question,
                thread=f"demo-{index}",
                trace=not args.quiet,
                retriever=retriever,
            )
            print(f"\n  expected: {expected}")
    else:
        answered = ask(
            args.persona,
            args.question,
            thread=args.thread,
            trace=not args.quiet,
            retriever=retriever,
        )

    print(f"\n{'=' * WIDTH}")
    return 0 if answered else 1


if __name__ == "__main__":
    raise SystemExit(main())
