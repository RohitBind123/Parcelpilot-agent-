"""Who is asking, and what that entitles them to.

The Principal is the containment boundary (docs/ARCHITECTURE.md §4). It is
created once at login from a signed session token, frozen for the session, and
**never** a model-supplied argument or a client-supplied field. Tools are
curried with it at graph-build time, so an unauthorised query is not refused at
runtime - it is absent from the schema the model can see.

Scopes are derived from the role by the one sanctioned constructor, so a
Principal whose scopes disagree with its role cannot be built. That is what
makes every downstream `require()` meaningful.

The three-role split is D26. Its point is that the roles differ on real rows:
a support agent reads every account but cannot run detection, and only a
manager can approve a credit above the SOP v4 §3 threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

Role = Literal["customer", "support_agent", "ops_manager"]

ROLES: Final[tuple[Role, ...]] = get_args(Role)
STAFF_ROLES: Final[frozenset[Role]] = frozenset({"support_agent", "ops_manager"})

# --- Capability scopes -----------------------------------------------------
# One scope per capability the tool layer gates on. Named as verb:noun so a
# denial reads clearly in a log line and in the UI.

SCOPE_READ_OWN_ACCOUNT: Final = "read:own_account"
SCOPE_READ_ANY_ACCOUNT: Final = "read:any_account"
SCOPE_AGGREGATE_TICKETS: Final = "read:ticket_aggregates"
SCOPE_OWN_QUEUE: Final = "read:own_queue"
SCOPE_SLA_STATUS: Final = "read:sla_status"
SCOPE_OPS_DETECTION: Final = "read:ops_detection"
SCOPE_PREPARE_ACTION: Final = "write:prepare_action"
SCOPE_APPROVE_CREDIT: Final = "write:approve_credit"

_CUSTOMER_SCOPES: Final = frozenset({SCOPE_READ_OWN_ACCOUNT, SCOPE_PREPARE_ACTION})

_SUPPORT_AGENT_SCOPES: Final = frozenset(
    {
        SCOPE_READ_ANY_ACCOUNT,
        SCOPE_AGGREGATE_TICKETS,
        SCOPE_OWN_QUEUE,
        SCOPE_SLA_STATUS,
        SCOPE_PREPARE_ACTION,
    }
)

# A strict superset of the agent's scopes: detection (the ops dashboard) and
# the SOP v4 §3 credit-approval gate are the manager's alone.
_OPS_MANAGER_SCOPES: Final = _SUPPORT_AGENT_SCOPES | {
    SCOPE_OPS_DETECTION,
    SCOPE_APPROVE_CREDIT,
}

_ROLE_SCOPES: Final[dict[Role, frozenset[str]]] = {
    "customer": _CUSTOMER_SCOPES,
    "support_agent": _SUPPORT_AGENT_SCOPES,
    "ops_manager": _OPS_MANAGER_SCOPES,
}


@dataclass(frozen=True, slots=True)
class Principal:
    """Session identity. Immutable for the lifetime of the session."""

    user_id: str
    role: Role
    account_id: str | None
    scopes: frozenset[str]
    #: Matches `tickets.assigned_to`, which is how `my_queue` is scoped.
    #: None for customers, who have no queue.
    queue_key: str | None = None
    display_name: str = ""

    def __post_init__(self) -> None:
        # These are bugs in the login path, not recoverable states, so they
        # fail at construction rather than producing a narrowed or widened
        # read surface that nobody notices.
        if self.role == "customer":
            if not self.account_id:
                raise ValueError("customer Principal requires an account_id")
            if self.queue_key is not None:
                raise ValueError("customer Principal must not carry a queue_key")
        elif self.role in STAFF_ROLES and self.account_id is not None:
            raise ValueError("staff Principal must not carry an account_id")

    @property
    def is_staff(self) -> bool:
        return self.role in STAFF_ROLES

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def require(self, scope: str) -> None:
        """Raise unless this Principal holds `scope`.

        Tools call this so a missing scope is a loud, loggable denial rather
        than a silently empty result. Denials are a demo asset (§4.3).
        """
        if scope not in self.scopes:
            raise PermissionError(f"{self.role} {self.user_id!r} lacks the scope {scope!r}")


def build_principal(
    user_id: str,
    role: Role,
    *,
    account_id: str | None = None,
    queue_key: str | None = None,
    display_name: str = "",
) -> Principal:
    """The only sanctioned way to construct a Principal.

    Scopes come from the role table, never from the caller, so role and scopes
    cannot drift apart.
    """
    if role not in _ROLE_SCOPES:
        raise ValueError(f"unknown role: {role!r}")
    return Principal(
        user_id=user_id,
        role=role,
        account_id=account_id,
        scopes=_ROLE_SCOPES[role],
        queue_key=queue_key,
        display_name=display_name or user_id,
    )
