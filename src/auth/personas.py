"""Seeded identities for the mocked login (D17).

Authentication is mocked, as the brief permits, but the *shape* is not: a
persona is selected, the server mints a signed token, and the Principal is
resolved server-side from that token. The client never states its own role.

The set below is chosen so that every distinct policy situation in the pack is
one click away, and so the D26 role split shows up on real rows:

  Northstar    agreement overrides the SOP outright (no cancellation fee)
  LumenWorks   agreement declines to override cancellation, but replaces the
               failed-pickup credit threshold and amount
  Beacon       no agreement at all; standard policy is the governing source
  Axis Labs    Enterprise plan without premium support; its only order is
               DELIVERED, and it owns the P1 credential-exposure ticket
  Maya, Rohit  the two names in tickets.assigned_to, so my_queue splits the
               real ticket set between them
  Priya Mehta  ops manager; the CSM named in the accounts sheet and in the
               Northstar agreement §4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from src.auth.principal import Principal, Role, build_principal


@dataclass(frozen=True, slots=True)
class Persona:
    """A selectable identity in the login picker."""

    persona_id: str
    display_name: str
    role: Role
    description: str
    account_id: str | None = None
    #: Matches `tickets.assigned_to`. Staff only.
    queue_key: str | None = None


PERSONAS: Final[tuple[Persona, ...]] = (
    Persona(
        persona_id="northstar_customer",
        display_name="Northstar Logistics",
        role="customer",
        account_id="ACCT-001",
        description="Enterprise customer whose agreement overrides the cancellation SOP.",
    ),
    Persona(
        persona_id="lumenworks_customer",
        display_name="LumenWorks",
        role="customer",
        account_id="ACCT-002",
        description="Growth customer whose agreement replaces the failed-pickup credit terms.",
    ),
    Persona(
        persona_id="beacon_customer",
        display_name="Beacon Retail",
        role="customer",
        account_id="ACCT-003",
        description="Standard customer with no agreement; general policy governs.",
    ),
    Persona(
        persona_id="axis_customer",
        display_name="Axis Labs",
        role="customer",
        account_id="ACCT-004",
        description="Enterprise plan without premium support; its only order is delivered.",
    ),
    Persona(
        persona_id="maya_agent",
        display_name="Maya",
        role="support_agent",
        queue_key="Maya",
        description="Support agent. Reads every account, but cannot run the ops scan.",
    ),
    Persona(
        persona_id="rohit_agent",
        display_name="Rohit",
        role="support_agent",
        queue_key="Rohit",
        description="Support agent. Holds both open P1 tickets in his queue.",
    ),
    Persona(
        persona_id="priya_manager",
        display_name="Priya Mehta",
        role="ops_manager",
        queue_key="Priya Mehta",
        description="Ops manager. The only role that can run detection or approve a credit.",
    ),
)

_BY_ID: Final[dict[str, Persona]] = {p.persona_id: p for p in PERSONAS}

_ROLE_ORDER: Final[dict[Role, int]] = {"customer": 0, "support_agent": 1, "ops_manager": 2}


def list_personas() -> tuple[Persona, ...]:
    """Personas in picker order: customers first, then staff by privilege."""
    return tuple(sorted(PERSONAS, key=lambda p: (_ROLE_ORDER[p.role], p.display_name)))


def get_persona(persona_id: str) -> Persona:
    """Look up a persona, or raise. Unknown ids are never coerced to a default."""
    try:
        return _BY_ID[persona_id]
    except KeyError as exc:
        raise LookupError(f"unknown persona: {persona_id!r}") from exc


def to_principal(persona: Persona) -> Principal:
    """Resolve a persona to the Principal the tool layer is bound with."""
    return build_principal(
        persona.persona_id,
        persona.role,
        account_id=persona.account_id,
        queue_key=persona.queue_key,
        display_name=persona.display_name,
    )
