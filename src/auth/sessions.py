"""Opaque signed session tokens (D17, ARCHITECTURE 4.2).

`POST /auth/login {persona_id}` returns a token. Every later request carries
only that token. There is no `role` and no `account_id` anywhere in a request
body, which is what makes the access-control story hold: to be staff you must
forge a signature, not set a field.

The token carries a random session id and nothing else. Not the persona, not
the role, not the account. That is deliberate and slightly stronger than
signing a claims blob: a signed claim is still a claim the client holds, and
revoking it means maintaining a denylist. Here the binding lives in a
server-side table, so deleting the row revokes the token immediately even
though the signature still verifies.

Two failure modes deliberately look identical from outside. A bad signature and
a session that no longer exists both raise `InvalidToken`, because the
difference answers "is this a real session id" for anyone who asks enough times.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Final

from itsdangerous import BadSignature, URLSafeSerializer

from src.auth.personas import get_persona, to_principal
from src.auth.principal import Principal
from src.datastore.runtime import RuntimeStore

logger = logging.getLogger(__name__)

#: Eight hours: long enough for a working session, short enough that an
#: abandoned one is not a standing credential.
DEFAULT_TTL_SECONDS: Final = 8 * 3600

_SALT: Final = "parcelpilot.session"


class SessionError(RuntimeError):
    """Login or resolution failed."""


class InvalidToken(SessionError):
    """The token is unreadable, unsigned, expired, or revoked.

    One exception for all four. Distinguishing them would tell a caller
    which part of a forged token was wrong.
    """


class UnknownPersona(SessionError):
    """No such persona. Never coerced to a default."""


class SessionManager:
    """Mints and resolves session tokens against the runtime store."""

    def __init__(
        self,
        store: RuntimeStore,
        *,
        secret: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._serialiser = URLSafeSerializer(_resolve_secret(secret), salt=_SALT)

    def login(self, persona_id: str) -> str:
        """Bind a new session to a persona and return its token."""
        try:
            persona = get_persona(persona_id)
        except LookupError as exc:
            raise UnknownPersona(str(exc)) from exc

        session_id = f"sid_{secrets.token_urlsafe(24)}"
        self._store.create_session(
            token_id=session_id,
            persona_id=persona.persona_id,
            ttl_seconds=self._ttl_seconds,
        )
        # Only the id is signed. Everything about the user is looked up.
        return self._serialiser.dumps({"sid": session_id})

    def resolve(self, token: str) -> Principal:
        """The Principal this token identifies, or `InvalidToken`."""
        session_id = self.peek(token)["sid"]
        session = self._store.get_session(session_id)
        if session is None:
            # Correctly signed and no longer valid: logged out, expired, or
            # minted before a restart regenerated the secret.
            raise InvalidToken("session is not active")
        try:
            return to_principal(get_persona(session.persona_id))
        except LookupError as exc:
            # The persona table changed under a live session. Not the client's
            # fault and not something they can fix, but not a usable session.
            logger.warning("session %s names an unknown persona", session_id)
            raise InvalidToken("session persona no longer exists") from exc

    def peek(self, token: str) -> dict[str, Any]:
        """The signed payload, verified but not resolved.

        Exposed because the API layer and the tests both need the session id
        without a store round-trip. It verifies the signature first, so this
        is not a way to read an unsigned token.
        """
        if not isinstance(token, str) or not token.strip():
            raise InvalidToken("no token supplied")
        try:
            payload = self._serialiser.loads(token)
        except BadSignature as exc:
            raise InvalidToken("token signature is not valid") from exc
        except Exception as exc:  # malformed base64, wrong shape, anything else
            raise InvalidToken("token could not be read") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("sid"), str):
            raise InvalidToken("token payload is not a session reference")
        return payload

    def logout(self, token: str) -> None:
        """Revoke a session. Idempotent, and never raises on a bad token.

        A stale tab logging out twice, or logging out with a token from before
        a restart, is ordinary. Raising would turn it into a 500 for something
        that has already achieved what the caller wanted.
        """
        try:
            session_id = self.peek(token)["sid"]
        except InvalidToken:
            return
        self._store.delete_session(session_id)


def _resolve_secret(secret: str | None) -> str:
    """The signing secret, or a fresh random one.

    An empty secret is not an error and is not signed with. `.env.example`
    documents the trade: unset means a per-process random secret, so sessions
    do not survive a restart and there is no shipped default anybody could
    sign a staff token with.
    """
    if secret:
        return secret
    logger.info("no session secret configured; generating a per-process one")
    return secrets.token_urlsafe(32)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "InvalidToken",
    "SessionError",
    "SessionManager",
    "UnknownPersona",
]
