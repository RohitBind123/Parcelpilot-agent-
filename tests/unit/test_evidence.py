"""Typed evidence handles (D13a).

The handles are what make the multi-step chain a property of the type system
rather than a hope about tool ordering. `compute_cancellation_fee` takes a
`resolution_id`, not an `order_id`, so the model cannot skip the resolver: the
argument it needs does not exist until the resolver has run.

That only holds if a handle cannot be forged, guessed, reused across runs, or
consumed by a different Principal. Those four are what this file attacks. The
threat is not an attacker in the usual sense - it is a model with a handle in
its context window, inventing a plausible-looking one, or replaying one it saw
in an earlier turn belonging to someone else.
"""

from __future__ import annotations

import inspect

import pytest

from src.auth.personas import get_persona, to_principal
from src.domain.evidence import (
    EvidenceKind,
    EvidenceKindError,
    EvidenceNotFound,
    EvidenceScopeError,
    EvidenceStore,
    Handle,
    open_evidence_store,
)


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


ORDER_PAYLOAD = {"order_id": "ORD-1001", "status": "BOOKED", "shipment_fee_inr": 4200.0}


@pytest.fixture
def store():
    with open_evidence_store(run_id="run_a", principal=persona("northstar_customer")) as opened:
        yield opened


class TestMinting:
    def test_a_handle_comes_back_typed(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        assert isinstance(handle, Handle)
        assert handle.kind is EvidenceKind.ORDER_SNAPSHOT

    def test_the_id_is_prefixed_by_its_kind(self, store):
        # Purely so a handle is legible in a trace and in model context. The
        # prefix is decoration; the kind is enforced from the stored row.
        handle = store.mint(EvidenceKind.POLICY_RESOLUTION, {"topic": "cancellation_fee"})
        assert handle.evidence_id.startswith("res_")

    def test_a_caller_cannot_choose_the_id(self):
        # If the id were an argument, a model could pass one it invented and the
        # store would happily create it.
        assert "evidence_id" not in inspect.signature(EvidenceStore.mint).parameters

    def test_ids_are_unpredictable_and_unique(self, store):
        minted = {
            store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD).evidence_id for _ in range(50)
        }
        assert len(minted) == 50
        # Long enough that guessing is not a strategy.
        assert all(len(i) > 20 for i in minted)

    def test_the_payload_round_trips_unchanged(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        assert store.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT) == ORDER_PAYLOAD

    def test_a_payload_that_cannot_be_serialised_is_refused_at_mint(self, store):
        # Better to fail where the bad value entered than to fail on read, when
        # the context that produced it is long gone.
        with pytest.raises(TypeError):
            store.mint(EvidenceKind.ORDER_SNAPSHOT, {"when": object()})


class TestKindEnforcement:
    def test_reading_with_the_wrong_expected_kind_is_refused(self, store):
        resolution = store.mint(EvidenceKind.POLICY_RESOLUTION, {"topic": "cancellation_fee"})
        # The exact confusion the design exists to prevent: a resolution handle
        # arriving where a snapshot belongs.
        with pytest.raises(EvidenceKindError):
            store.read(resolution, expect=EvidenceKind.ORDER_SNAPSHOT)

    def test_the_error_names_both_kinds(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        with pytest.raises(EvidenceKindError, match="policy_resolution"):
            store.read(handle, expect=EvidenceKind.POLICY_RESOLUTION)

    def test_a_bare_string_id_is_accepted_but_still_type_checked(self, store):
        # Tool arguments arrive from the model as strings, so the store has to
        # take one - it just must not trust it.
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        assert store.read(handle.evidence_id, expect=EvidenceKind.ORDER_SNAPSHOT) == ORDER_PAYLOAD
        with pytest.raises(EvidenceKindError):
            store.read(handle.evidence_id, expect=EvidenceKind.CALC_RESULT)


class TestForgeryAndReplay:
    def test_an_invented_handle_is_refused(self, store):
        with pytest.raises(EvidenceNotFound):
            store.read("snap_totally_made_up", expect=EvidenceKind.ORDER_SNAPSHOT)

    def test_a_handle_from_another_run_is_refused(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        with (
            open_evidence_store(
                run_id="run_b",
                principal=persona("northstar_customer"),
                connection=store.connection,
            ) as other_run,
            pytest.raises(EvidenceScopeError),
        ):
            # Same principal, same database, different run. Handles do not
            # survive a conversation boundary.
            other_run.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)

    def test_a_handle_minted_for_another_principal_is_refused_within_the_same_run(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        with (
            open_evidence_store(
                run_id="run_a",
                principal=persona("lumenworks_customer"),
                connection=store.connection,
            ) as other_principal,
            pytest.raises(EvidenceScopeError),
        ):
            # This is the leak that matters: the handle is in model context, and
            # the context is what a prompt injection can reach.
            other_principal.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)

    def test_staff_cannot_consume_a_customers_handle_either(self, store):
        # Staff read every account, but not every *handle*. Scope on read is
        # about provenance, not authorisation: an agent must re-fetch under
        # their own identity so the trace shows who looked.
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        with (
            open_evidence_store(
                run_id="run_a", principal=persona("maya_agent"), connection=store.connection
            ) as staff,
            pytest.raises(EvidenceScopeError),
        ):
            staff.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)

    def test_a_scope_failure_does_not_reveal_whether_the_handle_exists(self, store):
        handle = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        with (
            open_evidence_store(
                run_id="run_a",
                principal=persona("lumenworks_customer"),
                connection=store.connection,
            ) as other,
            pytest.raises(EvidenceScopeError) as denied,
        ):
            other.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)
        # The payload is the thing being protected; the message must not carry
        # any of it.
        assert "ORD-1001" not in str(denied.value)
        assert "BOOKED" not in str(denied.value)


class TestChaining:
    def test_a_handle_can_record_the_handles_it_was_derived_from(self, store):
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        resolution = store.mint(
            EvidenceKind.POLICY_RESOLUTION, {"topic": "cancellation_fee"}, derived_from=[snapshot]
        )
        # The evidence chain is what the trace panel renders and what the
        # grounding gate walks; it has to be recorded at mint time, because
        # afterwards nothing knows what produced what.
        assert store.provenance(resolution) == (snapshot.evidence_id,)

    def test_provenance_is_empty_for_a_root_handle(self, store):
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        assert store.provenance(snapshot) == ()

    def test_the_chain_survives_two_hops(self, store):
        snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, ORDER_PAYLOAD)
        resolution = store.mint(
            EvidenceKind.POLICY_RESOLUTION, {"topic": "cancellation_fee"}, derived_from=[snapshot]
        )
        calc = store.mint(
            EvidenceKind.CALC_RESULT, {"fee_inr": 0}, derived_from=[snapshot, resolution]
        )
        assert store.provenance(calc) == (snapshot.evidence_id, resolution.evidence_id)

    def test_derivation_from_a_foreign_handle_is_refused(self, store):
        with open_evidence_store(
            run_id="run_a",
            principal=persona("lumenworks_customer"),
            connection=store.connection,
        ) as other:
            foreign = other.mint(EvidenceKind.ORDER_SNAPSHOT, {"order_id": "ORD-2001"})
        with pytest.raises(EvidenceScopeError):
            store.mint(EvidenceKind.CALC_RESULT, {"fee_inr": 0}, derived_from=[foreign])
