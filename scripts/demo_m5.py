"""Three roles, three schemas, one agent.

    uv run python scripts/demo_m5.py
    uv run python scripts/demo_m5.py --chain      # also walk a full tool chain

Prints what each persona's model can see. The containment claim is that an
unauthorised query is not refused - it is absent, because tools are curried
with the Principal before the first LLM call. This is that claim, rendered.

Nothing here is illustrative: the schemas printed are the ones sent on the wire.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import PROJECTION, UNIMPLEMENTED, build_toolset
from src.auth.personas import get_persona, to_principal
from src.config import get_settings

WIDTH = 92
SHOWCASE = ("northstar_customer", "maya_agent", "priya_manager")


def rule(title: str = "") -> None:
    print(f"\n{title}\n{'=' * WIDTH}" if title else "=" * WIDTH)


def show_schemas(db_path: Path) -> None:
    schemas = {}
    for persona_id in SHOWCASE:
        principal = to_principal(get_persona(persona_id))
        with open_tool_context(principal, run_id=persona_id, db_path=db_path) as context:
            schemas[persona_id] = {t.name: t for t in build_toolset(context)}

    rule("WHAT EACH ROLE'S MODEL CAN SEE")
    every = sorted(PROJECTION)
    header = f"{'tool':<30}" + "".join(f"{p.split('_')[0]:>14}" for p in SHOWCASE)
    print(header)
    print("-" * WIDTH)
    for name in every:
        marks = ""
        for persona_id in SHOWCASE:
            if name in schemas[persona_id]:
                marks += f"{'yes':>14}"
            elif name in UNIMPLEMENTED:
                role = to_principal(get_persona(persona_id)).role
                marks += f"{('(' + UNIMPLEMENTED[name].split('(')[1]) if role in PROJECTION[name] else '-':>14}"
            else:
                marks += f"{'-':>14}"
        print(f"{name:<30}{marks}")
    print("-" * WIDTH)
    print("'-' means the tool is not in the schema at all. There is nothing to refuse.")
    print("A bracketed milestone means the role is entitled to it and it is not built yet.")

    rule("THE SAME TOOL, DIFFERENT SHAPE")
    for name in ("get_order", "search_policy", "resolve_policy"):
        print(f"\n{name}")
        for persona_id in SHOWCASE:
            tool = schemas[persona_id].get(name)
            if tool is None:
                print(f"    {persona_id:<20} absent")
                continue
            params = (
                ", ".join(f"{p.name}{'' if p.required else '?'}" for p in tool.params) or "(none)"
            )
            print(f"    {persona_id:<20} ({params})")
    print("\nA customer's get_order has no account_id. The cross-account lookup is not")
    print("refused - it cannot be expressed.")


def walk_chain(db_path: Path) -> None:
    rule("A FULL CHAIN, AS A MODEL WOULD DRIVE IT")
    for persona_id, order_id in (
        ("northstar_customer", "ORD-1001"),
        ("lumenworks_customer", "ORD-2001"),
    ):
        principal = to_principal(get_persona(persona_id))
        with open_tool_context(principal, run_id=persona_id, db_path=db_path) as context:
            tools = {t.name: t for t in build_toolset(context)}

            print(f"\n{persona_id}  asks about {order_id}")
            skipped = tools["compute_cancellation_fee"](order_id=order_id)
            print(f"  compute_cancellation_fee(order_id=...)  -> {skipped.message}")

            snapshot = tools["get_order"](order_id=order_id).data["snapshot_id"]
            print(f"  get_order                               -> {snapshot}")

            bare = tools["compute_cancellation_fee"](snapshot_id=snapshot)
            print(f"  compute_cancellation_fee(snapshot only) -> {bare.message}")

            resolution = tools["resolve_policy"](
                topic="cancellation_fee", snapshot_id=snapshot
            ).data
            print(f"  resolve_policy                          -> {resolution['resolution_id']}")
            print(f"      governing {resolution['governing_citation']}")
            if resolution["overridden"]:
                print(f"      overrides {', '.join(resolution['overridden'])}")

            fee = tools["compute_cancellation_fee"](
                snapshot_id=snapshot, resolution_id=resolution["resolution_id"]
            ).data
            print(f"  compute_cancellation_fee                -> INR {fee['fee_inr']}")

            report = tools["check_data_consistency"](snapshot_id=snapshot).data
            state = "BLOCKING conflict" if report["blocking"] else "no conflict"
            print(f"  check_data_consistency                  -> {state}")

    rule("A PROBE THAT IS NOT REFUSED SO MUCH AS UNAVAILABLE")
    principal = to_principal(get_persona("lumenworks_customer"))
    with open_tool_context(principal, run_id="probe", db_path=db_path) as context:
        tools = {t.name: t for t in build_toolset(context)}
        denial = tools["get_order"](order_id="ORD-1001")
        print("\nlumenworks_customer  get_order(ORD-1001)")
        print(f"  {denial.to_payload()}")
        print("\nNothing in that payload says whose order it is, whether it exists, or")
        print("what state it is in. 'Not yours' and 'no such order' read identically.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", action="store_true", help="also walk a full tool chain")
    args = parser.parse_args()

    db_path = get_settings().db_path
    rule()
    print("ParcelPilot M5 - the projection is the access control")
    show_schemas(db_path)
    if args.chain:
        walk_chain(db_path)
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
