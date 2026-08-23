"""The confirmation token that binds a preview to its execution (ARCHITECTURE 13).

`prepare_action` describes what it is about to do, mints a token over that
description, and pauses. A human reads the description and confirms.
`execute_action` recomputes the token from the description still held in graph
state and refuses if it does not match.

The point is not authentication - the session already did that. The point is
that **the thing confirmed is the thing executed**. Between the preview and the
click, the only actor in the loop is a language model, and a gate that asks
"confirm?" and then reads a payload the model can still edit has confirmed
nothing. The token makes any edit detectable, because every field that
describes the action is signed.

The token is a digest, not an envelope. It carries no payload, so it cannot be
decoded, cannot be used to smuggle a payload back into the model's context
window, and cannot be replayed against a different action - the nonce is signed
too. What the token proves is "the server minted this for exactly this action";
what the *state* holds is the action itself.

Single use is enforced elsewhere and deliberately: the nonce is written onto
the immutable action row under a UNIQUE constraint, so replay is refused by the
same mechanism that makes the log immutable, atomically with the effect. A
separate "seen nonces" table would be one more thing that can drift out of step
with what actually happened.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Final

from src.clock import wall_now
from src.datastore.runtime import ActionKind

#: Ten minutes. A confirmation card is a question being asked of someone who is
#: looking at it; one left open overnight should not still execute.
TOKEN_TTL_SECONDS: Final = 10 * 60

_DIGEST: Final = hashlib.sha256
_SEPARATOR: Final = "."
#: A byte that cannot occur in any of the signed fields, so concatenation is
#: unambiguous. Without it, ("ab", "c") and ("a", "bc") sign identically.
_FIELD_JOIN: Final = "\x1f"


class TokenError(RuntimeError):
    """The token does not authorise this action.

    One class for malformed, mismatched, wrong-secret and expired. The caller
    turns all of them into the same refusal, and naming which part failed tells
    whoever is probing how to get closer.
    """


def new_nonce() -> str:
    return secrets.token_urlsafe(18)


@dataclass(frozen=True, slots=True)
class PendingAction:
    """What was shown to the human, held in graph state until they answer.

    Frozen because it is the record of what was previewed. If it can be edited
    between preview and execution then the preview was not a preview.
    """

    nonce: str
    kind: ActionKind
    payload: Mapping[str, Any]
    evidence_chain: tuple[str, ...]
    session_id: str
    thread_id: str
    expires_at: datetime
    #: Blocking conflicts are refused before a token is ever minted (D19).
    #: Advisory ones travel with the preview so the card can show them.
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_preview(self) -> dict[str, Any]:
        """What the client renders. Deliberately without the token.

        The token goes to the client in its own field on the interrupt event,
        not inside the preview, so that a preview can be logged or echoed
        without carrying the proof along with it.
        """
        return {
            "kind": self.kind.value,
            "payload": dict(self.payload),
            "evidence_chain": list(self.evidence_chain),
            "expires_at": self.expires_at.isoformat(),
            "advisories": list(self.advisories),
        }

    def signing_material(self) -> str:
        """Every field that describes the action, canonically.

        Changing any of them changes the token. `payload` is dumped with sorted
        keys so that a round-trip through JSON does not invalidate a token
        nobody touched.
        """
        return _FIELD_JOIN.join(
            (
                self.nonce,
                self.kind.value,
                json.dumps(dict(self.payload), sort_keys=True, separators=(",", ":")),
                json.dumps(list(self.evidence_chain), separators=(",", ":")),
                self.session_id,
                self.thread_id,
                self.expires_at.isoformat(),
            )
        )

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= wall_now()


def build_pending(
    *,
    kind: ActionKind,
    payload: Mapping[str, Any],
    evidence_chain: Sequence[str],
    session_id: str,
    thread_id: str,
    advisories: Sequence[str] = (),
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> PendingAction:
    """A pending action expiring `ttl_seconds` from now."""
    return PendingAction(
        nonce=new_nonce(),
        kind=kind,
        payload=dict(payload),
        evidence_chain=tuple(evidence_chain),
        session_id=session_id,
        thread_id=thread_id,
        expires_at=wall_now() + timedelta(seconds=ttl_seconds),
        advisories=tuple(advisories),
    )


def mint_token(action: PendingAction, *, secret: str) -> str:
    """`nonce.signature`.

    The nonce travels in the clear so a mismatched token can be identified as
    belonging to a different action rather than merely failing; the signature
    is what makes it unforgeable.
    """
    return f"{action.nonce}{_SEPARATOR}{_sign(action, secret)}"


def verify_token(token: str, action: PendingAction, *, secret: str) -> None:
    """Raise `TokenError` unless this token authorises exactly this action."""
    if not isinstance(token, str) or not token.strip():
        raise TokenError("no confirmation token supplied")

    nonce, separator, signature = token.partition(_SEPARATOR)
    if not separator or not nonce or not signature:
        raise TokenError("confirmation token is malformed")

    # Compared before the signature so a token for a different pending action
    # is named as such. It is not a security distinction - both refuse - and
    # it is the difference between a debuggable failure and a shrug.
    if not hmac.compare_digest(nonce, action.nonce):
        raise TokenError("confirmation token is for a different action")

    if not hmac.compare_digest(signature, _sign(action, secret)):
        raise TokenError("confirmation token does not match the action to be executed")

    # Last, so that a stale token still gets the integrity check. An expired
    # token whose payload was also edited should not report only the expiry.
    if action.is_expired:
        raise TokenError("confirmation token has expired")


def _sign(action: PendingAction, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        action.signing_material().encode("utf-8"),
        _DIGEST,
    ).hexdigest()


__all__ = [
    "TOKEN_TTL_SECONDS",
    "PendingAction",
    "TokenError",
    "build_pending",
    "mint_token",
    "new_nonce",
    "verify_token",
]
