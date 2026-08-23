"""Session tokens (D17), and what a request is not allowed to say about itself.

The rule the whole access-control story rests on: no request body, query
parameter or header names a role or an account. A request carries one opaque
token, and the server resolves it to a Principal through a table it owns. That
is what makes "forging a staff session requires forging a signature" true
(ARCHITECTURE 4.2) rather than aspirational.

So the tests worth writing are mostly negative. A token that survives a secret
change, or that carries its own persona, or that still resolves after logout,
is a hole - and none of those would be caught by a test that only checks the
happy path round-trips.
"""

from __future__ import annotations

import pytest

from src.auth.sessions import (
    DEFAULT_TTL_SECONDS,
    InvalidToken,
    SessionManager,
    UnknownPersona,
)
from src.datastore.runtime import open_runtime_store


@pytest.fixture
def store(tmp_path):
    with open_runtime_store(tmp_path / "runtime.db") as opened:
        yield opened


@pytest.fixture
def sessions(store):
    return SessionManager(store, secret="test-secret-not-a-default")


class TestLoginAndResolve:
    def test_a_token_resolves_to_the_persona_it_was_minted_for(self, sessions):
        token = sessions.login("maya_agent")
        principal = sessions.resolve(token)
        assert principal.user_id == "maya_agent"
        assert principal.role == "support_agent"

    def test_a_customer_token_carries_the_account_the_server_knows_about(self, sessions):
        principal = sessions.resolve(sessions.login("northstar_customer"))
        assert principal.account_id == "ACCT-001"
        assert principal.role == "customer"

    def test_an_unknown_persona_is_refused_rather_than_defaulted(self, sessions):
        # Coercing an unknown id to a default is how a typo becomes a login.
        with pytest.raises(UnknownPersona):
            sessions.login("root")

    def test_two_logins_produce_different_tokens(self, sessions):
        assert sessions.login("maya_agent") != sessions.login("maya_agent")


class TestTheTokenSaysNothingAboutTheUser:
    """The containment property, tested against the token's own bytes.

    If a role or an account id is anywhere in the token, then the token is
    describing the user rather than identifying them, and the next question is
    what happens when someone edits it.
    """

    def test_the_token_does_not_contain_the_persona_id(self, sessions):
        assert "maya_agent" not in sessions.login("maya_agent")

    def test_the_token_does_not_contain_the_role(self, sessions):
        token = sessions.login("priya_manager")
        assert "ops_manager" not in token
        assert "manager" not in token.lower()

    def test_the_token_does_not_contain_the_account_id(self, sessions):
        assert "ACCT-001" not in sessions.login("northstar_customer")

    def test_the_unsigned_payload_is_only_an_opaque_id(self, sessions):
        # Decoded rather than inferred from the surface string: a base64 body
        # could hold the persona without it showing up in a substring check.
        token = sessions.login("northstar_customer")
        payload = sessions.peek(token)
        assert set(payload) == {"sid"}
        assert "ACCT" not in payload["sid"]

    def test_rebinding_the_session_changes_who_the_token_is(self, sessions, store):
        # The persona lives in the table, not the token. This is the same
        # property from the other side: the token is a key, and the server
        # decides what it opens.
        token = sessions.login("maya_agent")
        payload = sessions.peek(token)
        store.create_session(token_id=payload["sid"], persona_id="priya_manager", ttl_seconds=3600)
        assert sessions.resolve(token).role == "ops_manager"


class TestForgery:
    def test_a_token_signed_with_another_secret_is_rejected(self, store):
        attacker = SessionManager(store, secret="some-other-secret")
        forged = attacker.login("priya_manager")
        # Same store, so the session row genuinely exists. Only the signature
        # is wrong, which is the whole defence.
        legitimate = SessionManager(store, secret="test-secret-not-a-default")
        with pytest.raises(InvalidToken):
            legitimate.resolve(forged)

    def test_a_tampered_token_is_rejected(self, sessions):
        token = sessions.login("maya_agent")
        tampered = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
        with pytest.raises(InvalidToken):
            sessions.resolve(tampered)

    def test_a_well_formed_token_for_a_session_that_does_not_exist_is_rejected(
        self, sessions, store
    ):
        # A correct signature is necessary and not sufficient. Deleting the row
        # must revoke the token even though it still verifies.
        token = sessions.login("maya_agent")
        store.delete_session(sessions.peek(token)["sid"])
        with pytest.raises(InvalidToken):
            sessions.resolve(token)

    def test_garbage_is_rejected_as_a_token_not_as_a_crash(self, sessions):
        for candidate in ("", "   ", "not-a-token", "a.b.c"):
            with pytest.raises(InvalidToken):
                sessions.resolve(candidate)


class TestExpiryAndLogout:
    def test_logging_out_revokes_the_token(self, sessions):
        token = sessions.login("maya_agent")
        sessions.logout(token)
        with pytest.raises(InvalidToken):
            sessions.resolve(token)

    def test_logging_out_twice_is_not_an_error(self, sessions):
        token = sessions.login("maya_agent")
        sessions.logout(token)
        sessions.logout(token)

    def test_logging_out_with_garbage_is_not_an_error(self, sessions):
        # Logout is idempotent and unauthenticated in effect; raising here
        # would turn a stale browser tab into a 500.
        sessions.logout("not-a-token")

    def test_an_expired_session_no_longer_resolves(self, store):
        expiring = SessionManager(store, secret="s", ttl_seconds=-1)
        with pytest.raises(InvalidToken):
            expiring.resolve(expiring.login("maya_agent"))

    def test_the_default_lifetime_is_bounded(self):
        # A session that never expires is a credential, not a session.
        assert 0 < DEFAULT_TTL_SECONDS <= 24 * 3600


class TestTheSecret:
    def test_an_absent_secret_is_generated_per_process(self, store):
        # `.env.example` documents this: unset means a random secret, so
        # sessions do not survive a restart and there is no shipped default
        # for anyone to sign with.
        first = SessionManager(store, secret="")
        second = SessionManager(store, secret="")
        with pytest.raises(InvalidToken):
            second.resolve(first.login("maya_agent"))

    def test_a_generated_secret_still_works_within_the_process(self, store):
        manager = SessionManager(store, secret="")
        assert manager.resolve(manager.login("maya_agent")).user_id == "maya_agent"
