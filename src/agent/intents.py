"""Route taxonomy, entitlement gating, and the fail-open parse.

The classifier proposes; the Principal disposes (ARCHITECTURE.md §9.4). A
classifier is an injection surface, so two properties are enforced here
rather than in a prompt:

1. `parse_route` accepts only members of the Route enum. Anything else -
   free text, a hallucinated route, a prompt-injection payload - becomes
   COMPLEX. Unparseable never means "pick the cheap path".
2. `is_entitled` is checked deterministically against the Principal's
   scopes after classification. A customer session that reaches
   OPS_INVESTIGATION is denied regardless of what the classifier said.

The bias throughout is fail-open (§9.5): routing a simple question to the
planner wastes a model call, while routing a complex one to a scripted
pipeline produces a confidently wrong answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from src.auth.principal import (
    SCOPE_OPS_DETECTION,
    SCOPE_PREPARE_ACTION,
    Principal,
)


class Route(StrEnum):
    """The recognised query shapes. See ARCHITECTURE.md §9.2."""

    CHITCHAT = "CHITCHAT"
    CAPABILITY = "CAPABILITY"
    POLICY_LOOKUP = "POLICY_LOOKUP"
    ACCOUNT_FACT = "ACCOUNT_FACT"
    ENTITLEMENT_DECISION = "ENTITLEMENT_DECISION"
    OPS_INVESTIGATION = "OPS_INVESTIGATION"
    ACTION_REQUEST = "ACTION_REQUEST"
    COMPLEX = "COMPLEX"
    UNSUPPORTED_EXCEPTION = "UNSUPPORTED_EXCEPTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


#: Where an unparseable or low-confidence classification lands. Never a
#: cheaper route than the one that was ambiguous.
FALLBACK_ROUTE: Final = Route.COMPLEX

#: Scope required to reach a route. Absent means every role may reach it.
ROUTE_SCOPE: Final[dict[Route, str]] = {
    Route.OPS_INVESTIGATION: SCOPE_OPS_DETECTION,
    Route.ACTION_REQUEST: SCOPE_PREPARE_ACTION,
}

#: Routes that may answer without touching retrieval, and only on high
#: confidence - the three where being wrong is cheap and obvious (§9.5).
EARLY_EXIT_ROUTES: Final[frozenset[Route]] = frozenset(
    {Route.CHITCHAT, Route.CAPABILITY, Route.OUT_OF_SCOPE}
)

#: Routes handled by a scripted pipeline rather than the freeform planner.
SCRIPTED_ROUTES: Final[frozenset[Route]] = frozenset(
    {Route.POLICY_LOOKUP, Route.ACCOUNT_FACT, Route.ENTITLEMENT_DECISION}
)

#: Below this, the classifier's answer is discarded for FALLBACK_ROUTE.
CONFIDENCE_THRESHOLD: Final = 0.6


def parse_route(raw: object) -> tuple[Route, bool]:
    """Coerce classifier output to a Route, failing open to COMPLEX.

    Returns `(route, was_parsed)`. `was_parsed is False` means the input was
    not a recognised route and FALLBACK_ROUTE was substituted - the caller
    should record that as `classifier_source="fallback"` so the trace shows
    it happened.
    """
    if isinstance(raw, Route):
        return raw, True
    if isinstance(raw, str):
        try:
            return Route(raw.strip().upper()), True
        except ValueError:
            return FALLBACK_ROUTE, False
    return FALLBACK_ROUTE, False


def apply_confidence(route: Route, confidence: float) -> tuple[Route, bool]:
    """Downgrade a low-confidence classification to the more capable path.

    Returns `(route, kept)`. Early-exit routes are held to the same bar -
    they terminate before retrieval, so a wrong one cannot be recovered.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK_ROUTE, False
    return route, True


def is_entitled(principal: Principal, route: Route) -> bool:
    """Whether this Principal may reach this route.

    Deterministic and independent of the classifier. This is the check that
    holds even if the classifier is fully subverted.
    """
    required = ROUTE_SCOPE.get(route)
    return required is None or principal.has(required)


def gate(principal: Principal, route: Route) -> tuple[Route, bool]:
    """Entitlement gate. Returns `(route, allowed)`.

    A denied route is not silently rewritten to something the caller may
    reach - the caller routes to `denied` and logs the attempt, because
    denials are a demo asset, not an embarrassment (§4.3).
    """
    return route, is_entitled(principal, route)
