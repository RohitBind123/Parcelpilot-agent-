"""The confirmation token (ARCHITECTURE 13).

`prepare_action` mints a token over the payload, the session and a nonce;
`execute_action` recomputes it and refuses on mismatch, reuse or expiry. The
reason this is a token rather than a boolean is that the thing being confirmed
must be the thing that executes. A gate that asks "confirm?" and then reads a
payload the model can still edit has confirmed nothing.

So the load-bearing test here is not that a good token passes. It is that a
token stops verifying the moment any byte of what it authorised changes - and
that the payload cannot be recovered from the token, because the payload is
supposed to live in graph state where the model cannot reach it.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from src.clock import wall_now
from src.datastore.runtime import ActionKind
from src.domain.action_tokens import (
    TOKEN_TTL_SECONDS,
    PendingAction,
    TokenError,
    mint_token,
    verify_token,
)

SECRET = "signing-secret"

#: Fixed once, not recomputed per call, and that matters more than it looks.
#: With a fresh `wall_now()` in the helper, two `pending()` calls differ in
#: `expires_at` - so every tamper test below would pass whether or not the
#: field it claims to test was ever signed. Pinning it means each test varies
#: exactly the one thing it names. Far enough ahead to outlast the suite.
FIXED_EXPIRY = wall_now() + timedelta(seconds=TOKEN_TTL_SECONDS)


def pending(**overrides) -> PendingAction:
    fields = {
        "nonce": "nonce-1",
        "kind": ActionKind.CREATE_ESCALATION,
        "payload": {"ticket_id": "TKT-503", "severity": "P3"},
        "evidence_chain": ("ev_1", "ev_2"),
        "session_id": "sid_abc",
        "thread_id": "thread-1",
        "expires_at": FIXED_EXPIRY,
    }
    fields.update(overrides)
    return PendingAction(**fields)


class TestAGoodToken:
    def test_a_token_verifies_against_the_action_it_authorised(self):
        action = pending()
        verify_token(mint_token(action, secret=SECRET), action, secret=SECRET)

    def test_minting_is_deterministic_for_the_same_action(self):
        action = pending()
        assert mint_token(action, secret=SECRET) == mint_token(action, secret=SECRET)

    def test_two_different_payloads_get_different_tokens(self):
        first = mint_token(pending(payload={"a": 1}), secret=SECRET)
        second = mint_token(pending(payload={"a": 2}), secret=SECRET)
        assert first != second

    def test_key_order_in_the_payload_does_not_change_the_token(self):
        # Canonicalised before signing. Otherwise a round-trip through JSON
        # could invalidate a token nobody touched.
        first = mint_token(pending(payload={"a": 1, "b": 2}), secret=SECRET)
        second = mint_token(pending(payload={"b": 2, "a": 1}), secret=SECRET)
        assert first == second


class TestTheTokenDoesNotCarryThePayload:
    """The integrity property from ARCHITECTURE 13, from the token's side.

    "The pending payload lives in graph state, not the model's context window."
    A token that embedded the payload would put it back in the context window,
    and a token that could be decoded into a payload would let a caller supply
    its own.
    """

    def test_the_payload_values_are_not_present_in_the_token(self):
        token = mint_token(
            pending(payload={"ticket_id": "TKT-503", "amount_inr": 4200}), secret=SECRET
        )
        assert "TKT-503" not in token
        assert "4200" not in token

    def test_the_token_is_a_digest_not_an_envelope(self):
        # Fixed length regardless of how large the payload is: there is nowhere
        # for a payload to hide.
        small = mint_token(pending(payload={"a": 1}), secret=SECRET)
        large = mint_token(pending(payload={"a": "x" * 5000}), secret=SECRET)
        assert len(small) == len(large)


class TestTamperingIsRefused:
    def test_a_changed_payload_stops_verifying(self):
        token = mint_token(pending(payload={"amount_inr": 500}), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(token, pending(payload={"amount_inr": 50000}), secret=SECRET)

    def test_a_payload_gaining_a_field_stops_verifying(self):
        token = mint_token(pending(payload={"ticket_id": "TKT-503"}), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(
                token,
                pending(payload={"ticket_id": "TKT-503", "auto_approve": True}),
                secret=SECRET,
            )

    def test_a_changed_kind_stops_verifying(self):
        # Same payload, different action. Approving a credit is not escalating.
        token = mint_token(pending(kind=ActionKind.CREATE_ESCALATION), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(token, pending(kind=ActionKind.APPROVE_CREDIT), secret=SECRET)

    def test_a_changed_evidence_chain_stops_verifying(self):
        # The evidence is what justified the action; swapping it swaps the
        # justification the auditor will read.
        token = mint_token(pending(evidence_chain=("ev_1",)), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(token, pending(evidence_chain=("ev_9",)), secret=SECRET)

    def test_a_token_from_another_session_is_refused(self):
        token = mint_token(pending(session_id="sid_attacker"), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(token, pending(session_id="sid_victim"), secret=SECRET)

    def test_a_token_for_another_pending_action_is_refused(self):
        token = mint_token(pending(nonce="nonce-other"), secret=SECRET)
        with pytest.raises(TokenError):
            verify_token(token, pending(nonce="nonce-1"), secret=SECRET)

    def test_a_token_signed_with_another_secret_is_refused(self):
        token = mint_token(pending(), secret="not-the-server-secret")
        with pytest.raises(TokenError):
            verify_token(token, pending(), secret=SECRET)

    @pytest.mark.parametrize("candidate", ["", "   ", "garbage", "a.b.c", "nonce-1.", ".sig"])
    def test_a_malformed_token_is_refused_rather_than_crashing(self, candidate):
        with pytest.raises(TokenError):
            verify_token(candidate, pending(), secret=SECRET)

    def test_a_token_is_refused_when_the_expiry_it_was_signed_with_moves(self):
        action = pending()
        token = mint_token(action, secret=SECRET)
        later = pending(expires_at=action.expires_at + timedelta(hours=1))
        with pytest.raises(TokenError):
            verify_token(token, later, secret=SECRET)


class TestExpiry:
    def test_an_expired_token_is_refused_even_though_it_verifies(self):
        # The signature is correct. A confirmation card left open overnight
        # should still not execute in the morning.
        stale = pending(expires_at=wall_now() - timedelta(seconds=1))
        with pytest.raises(TokenError, match="expired"):
            verify_token(mint_token(stale, secret=SECRET), stale, secret=SECRET)

    def test_the_lifetime_is_short(self):
        assert 0 < TOKEN_TTL_SECONDS <= 30 * 60


class TestPendingActionShape:
    def test_a_pending_action_is_frozen(self):
        # It is the record of what was shown to the human. If it can be edited
        # between preview and execution, the preview was not a preview.
        with pytest.raises(FrozenInstanceError):
            pending().payload = {"tampered": True}  # type: ignore[misc]

    def test_a_nonce_is_generated_when_one_is_not_supplied(self):
        from src.domain.action_tokens import new_nonce

        assert new_nonce() != new_nonce()

    def test_the_preview_payload_carries_no_signature(self):
        # What goes to the client for rendering is the action, not the proof.
        # The token travels in its own field so it cannot be logged as part of
        # a preview dump.
        assert "signature" not in pending().to_preview()
        assert "token" not in pending().to_preview()
