"""The projection matrix, and the build that applies it (D26, ARCHITECTURE 4.3).

This is the containment mechanism. An unauthorised query is not refused at
runtime - it is absent from the schema the model is handed, because tools are
curried with the Principal before the first LLM call. A customer's model has no
vocabulary for a cross-account lookup; Maya's has no `approve_credit` to be
talked into calling. Nothing in a prompt, a ticket description or a summarised
document can widen a toolset that was fixed before any of that text was read.

The matrix is data rather than a series of `if principal.is_staff` branches
scattered through the builders, for two reasons. It can be diffed against the
architecture table by a test, and a tool that has not been written yet can still
have a row - so "absent from Maya's schema" is a claim measured against
something, rather than a test that passes because nobody ever wrote the tool.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Final

from src.agent.tools import actions, authority, calculations, detection, structured
from src.agent.tools.base import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.agent.tools.context import ToolContext

CUSTOMER: Final = "customer"
SUPPORT_AGENT: Final = "support_agent"
OPS_MANAGER: Final = "ops_manager"

_ALL: Final = frozenset({CUSTOMER, SUPPORT_AGENT, OPS_MANAGER})
_STAFF: Final = frozenset({SUPPORT_AGENT, OPS_MANAGER})
_MANAGER: Final = frozenset({OPS_MANAGER})

#: ARCHITECTURE 4.3, verbatim. A test diffs it row by row.
PROJECTION: Final[dict[str, frozenset[str]]] = {
    "search_policy": _ALL,
    "get_order": _ALL,
    "get_ticket": _ALL,
    "get_account": _ALL,
    "query_tickets": _STAFF,
    "my_queue": _STAFF,
    "resolve_policy": _ALL,
    "compute_cancellation_fee": _ALL,
    "compute_service_credit": _ALL,
    "sla_first_response_status": _STAFF,
    "check_data_consistency": _ALL,
    "scan_support_health": _MANAGER,
    "explain_finding": _MANAGER,
    "prepare_action": _ALL,
    "execute_action": _ALL,
    "approve_credit": _MANAGER,
}

#: Named individually with the milestone that adds each, so the gap is a to-do
#: rather than a shrug - and so a reviewer can tell "not built" from "not
#: allowed" by reading one dict.
#: Nothing. Every row of the projection matrix is built.
UNIMPLEMENTED: Final[dict[str, str]] = {}

_BUILDERS: Final[dict[str, Callable[[ToolContext], Tool]]] = {
    "search_policy": authority.search_policy,
    "resolve_policy": authority.resolve_policy,
    "get_order": structured.get_order,
    "get_ticket": structured.get_ticket,
    "get_account": structured.get_account,
    "query_tickets": structured.query_tickets,
    "my_queue": structured.my_queue,
    "compute_cancellation_fee": calculations.compute_cancellation_fee,
    "compute_service_credit": calculations.compute_service_credit,
    "sla_first_response_status": calculations.sla_first_response_status,
    "check_data_consistency": calculations.check_data_consistency,
    "prepare_action": actions.prepare_action,
    "execute_action": actions.execute_action,
    "approve_credit": actions.approve_credit,
    "scan_support_health": detection.scan_support_health,
    "explain_finding": detection.explain_finding,
}

#: Built into the toolset but withheld from the schema the model is shown.
#:
#: `execute_action` is driven by the graph after a human confirms, with the
#: token the client sent. Offering it to the model would give it vocabulary for
#: performing an action without one - the same containment argument as the
#: projection matrix itself, one level in. It stays a real tool so that it is
#: still subject to the matrix and testable on its own.
MODEL_INVISIBLE: Final[frozenset[str]] = frozenset({"execute_action"})


class ProjectionError(RuntimeError):
    """The matrix and the builders disagree about what exists."""


def build_toolset(context: ToolContext) -> list[Tool]:
    """The tools this Principal may use, built and closed over their identity.

    Takes the context alone rather than a Principal beside it: the context
    already carries the Principal its repository and evidence store were opened
    for, and passing both would allow a toolset for one identity over a
    repository scoped to another.
    """
    role = context.principal.role
    built = [
        _BUILDERS[name](context)
        for name, roles in PROJECTION.items()
        if role in roles and name in _BUILDERS
    ]

    # A builder returning a tool under a different name would put it in the
    # schema without the matrix ever having allowed it.
    for tool in built:
        if tool.name not in PROJECTION:
            raise ProjectionError(f"{tool.name!r} is not in the projection matrix")
        if role not in PROJECTION[tool.name]:
            raise ProjectionError(f"{tool.name!r} is not permitted for {role!r}")
        if tool.requires_scope and tool.requires_scope not in context.principal.scopes:
            raise ProjectionError(
                f"{tool.name!r} needs {tool.requires_scope!r}, which {role!r} does not hold"
            )
    return built


def tool_names(tools: Iterable[Tool]) -> set[str]:
    return {tool.name: None for tool in tools}.keys() | set()


def to_schemas(tools: Sequence[Tool]) -> list[dict]:
    """The `tools` argument for a chat completion.

    Filters `MODEL_INVISIBLE`, so a tool can exist in the toolset without the
    model having a name for it.
    """
    return [tool.to_schema() for tool in tools if tool.name not in MODEL_INVISIBLE]


def _startup_check() -> None:
    """Every builder is in the matrix, and every matrix row is built or named."""
    unknown = sorted(set(_BUILDERS) - set(PROJECTION))
    if unknown:
        raise ProjectionError(f"builders with no matrix row: {unknown}")
    missing = sorted(set(PROJECTION) - set(_BUILDERS) - set(UNIMPLEMENTED))
    if missing:
        raise ProjectionError(f"matrix rows with neither a builder nor a deferral: {missing}")


_startup_check()
