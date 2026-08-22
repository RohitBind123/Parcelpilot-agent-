"""The session identity. Immutable, and never a model-supplied argument.

The Principal is created at login and closed over by the tool projection
(ARCHITECTURE.md §4.2). Because tools are curried with it at graph-build
time, an unauthorised query is not merely refused - it is inexpressible in
the schema the model sees.

Nothing in the agent graph may construct or mutate a Principal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Role = Literal["customer", "support_agent", "ops_manager"]

STAFF_ROLES: Final[frozenset[Role]] = frozenset({"support_agent", "ops_manager"})

# Capability scopes. Route entitlement (agent.intents) is checked against these.
SCOPE_READ_OWN_ACCOUNT: Final = "read:own_account"
SCOPE_READ_ANY_ACCOUNT: Final = "read:any_account"
SCOPE_AGGREGATE_TICKETS: Final = "read:ticket_aggregates"
SCOPE_OPS_DETECTION: Final = "read:ops_detection"
SCOPE_PREPARE_ACTION: Final = "write:prepare_action"

_ROLE_SCOPES: Final[dict[Role, frozenset[str]]] = {
    "customer": frozenset({SCOPE_READ_OWN_ACCOUNT, SCOPE_PREPARE_ACTION}),
    "support_agent": frozenset(
        {
            SCOPE_READ_ANY_ACCOUNT,
            SCOPE_AGGREGATE_TICKETS,
            SCOPE_OPS_DETECTION,
            SCOPE_PREPARE_ACTION,
        }
    ),
    "ops_manager": frozenset(
        {
            SCOPE_READ_ANY_ACCOUNT,
            SCOPE_AGGREGATE_TICKETS,
            SCOPE_OPS_DETECTION,
            SCOPE_PREPARE_ACTION,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is asking. Fixed for the lifetime of a session."""

    user_id: str
    role: Role
    account_id: str | None
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        # A customer without an account has no scope to read; staff carrying an
        # account_id would silently narrow the staff read surface. Both are bugs
        # in the login path, not recoverable states.
        if self.role == "customer" and not self.account_id:
            raise ValueError("customer Principal requires an account_id")
        if self.role in STAFF_ROLES and self.account_id is not None:
            raise ValueError("staff Principal must not carry an account_id")

    @property
    def is_staff(self) -> bool:
        return self.role in STAFF_ROLES

    def has(self, scope: str) -> bool:
        return scope in self.scopes


def build_principal(user_id: str, role: Role, account_id: str | None = None) -> Principal:
    """Construct a Principal with the canonical scope set for its role.

    This is the only sanctioned constructor - it guarantees scopes and role
    cannot drift apart, which would defeat route entitlement checks.
    """
    if role not in _ROLE_SCOPES:
        raise ValueError(f"unknown role: {role!r}")
    return Principal(
        user_id=user_id,
        role=role,
        account_id=account_id,
        scopes=_ROLE_SCOPES[role],
    )
