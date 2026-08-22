"""The Principal is the containment boundary.

Access control in this system is a property of the tool schema the model is
handed, and that schema is derived from the Principal's scopes. So the scope
table is not documentation - it is the enforcement point, and D26's role split
has to be true here before it can be true anywhere else.
"""

from __future__ import annotations

import dataclasses

import pytest
from src.auth.principal import (
    SCOPE_AGGREGATE_TICKETS,
    SCOPE_APPROVE_CREDIT,
    SCOPE_OPS_DETECTION,
    SCOPE_OWN_QUEUE,
    SCOPE_PREPARE_ACTION,
    SCOPE_READ_ANY_ACCOUNT,
    SCOPE_READ_OWN_ACCOUNT,
    SCOPE_SLA_STATUS,
    Principal,
    build_principal,
)


def customer(account_id: str = "ACCT-001") -> Principal:
    return build_principal("u_test", "customer", account_id=account_id)


def agent(queue_key: str = "Maya") -> Principal:
    return build_principal("u_maya", "support_agent", queue_key=queue_key)


def manager() -> Principal:
    return build_principal("u_priya", "ops_manager", queue_key="Priya Mehta")


class TestConstruction:
    def test_customer_requires_an_account(self):
        with pytest.raises(ValueError, match="account_id"):
            build_principal("u", "customer", account_id=None)

    def test_staff_must_not_carry_an_account(self):
        # A staff Principal with an account_id would silently narrow every
        # staff read to one customer.
        with pytest.raises(ValueError, match="account_id"):
            build_principal("u", "ops_manager", account_id="ACCT-001")

    def test_customer_must_not_carry_a_queue(self):
        with pytest.raises(ValueError, match="queue"):
            build_principal("u", "customer", account_id="ACCT-001", queue_key="Maya")

    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError, match="role"):
            build_principal("u", "wizard")  # type: ignore[arg-type]

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            customer().account_id = "ACCT-002"  # type: ignore[misc]

    def test_scopes_are_derived_from_role_not_supplied(self):
        # The only sanctioned constructor guarantees scopes and role cannot
        # drift apart, which is what makes the entitlement check meaningful.
        assert build_principal("u", "customer", account_id="ACCT-001").scopes == customer().scopes


class TestRoleScopes:
    def test_customer_can_read_only_its_own_account(self):
        p = customer()
        assert p.has(SCOPE_READ_OWN_ACCOUNT)
        assert not p.has(SCOPE_READ_ANY_ACCOUNT)

    def test_customer_cannot_aggregate_or_investigate(self):
        p = customer()
        for scope in (SCOPE_AGGREGATE_TICKETS, SCOPE_OPS_DETECTION, SCOPE_SLA_STATUS):
            assert not p.has(scope), scope

    def test_every_role_may_prepare_an_action(self):
        # Preparing is always allowed; executing is gated by the confirmation
        # token, and approving a large credit by SCOPE_APPROVE_CREDIT.
        for p in (customer(), agent(), manager()):
            assert p.has(SCOPE_PREPARE_ACTION)

    def test_support_agent_reads_widely_but_cannot_run_detection(self):
        # D26: the ops dashboard is manager-only. This is the distinction that
        # makes three roles visible on real data rather than notional.
        p = agent()
        assert p.has(SCOPE_READ_ANY_ACCOUNT)
        assert p.has(SCOPE_AGGREGATE_TICKETS)
        assert p.has(SCOPE_SLA_STATUS)
        assert not p.has(SCOPE_OPS_DETECTION)

    def test_only_the_manager_runs_detection(self):
        assert manager().has(SCOPE_OPS_DETECTION)
        assert not agent().has(SCOPE_OPS_DETECTION)
        assert not customer().has(SCOPE_OPS_DETECTION)

    def test_only_the_manager_approves_a_large_credit(self):
        # SOP v4 s3: any individual credit above INR 1,000 needs manager
        # approval. Unreachable on the shipped data (credits cap at INR 500)
        # but the gate is real (assumption A10).
        assert manager().has(SCOPE_APPROVE_CREDIT)
        assert not agent().has(SCOPE_APPROVE_CREDIT)
        assert not customer().has(SCOPE_APPROVE_CREDIT)

    def test_staff_have_a_queue_and_customers_do_not(self):
        assert agent().has(SCOPE_OWN_QUEUE)
        assert manager().has(SCOPE_OWN_QUEUE)
        assert not customer().has(SCOPE_OWN_QUEUE)

    def test_manager_scopes_are_a_strict_superset_of_agent_scopes(self):
        assert agent().scopes < manager().scopes

    def test_customer_and_staff_scope_sets_are_disjoint_except_prepare(self):
        shared = customer().scopes & manager().scopes
        assert shared == {SCOPE_PREPARE_ACTION}


class TestHelpers:
    def test_is_staff(self):
        assert not customer().is_staff
        assert agent().is_staff
        assert manager().is_staff

    def test_queue_key_defaults_to_none(self):
        assert customer().queue_key is None
        assert agent("Rohit").queue_key == "Rohit"

    def test_require_raises_for_a_missing_scope(self):
        # Tools call require() so a missing scope is a loud denial that can be
        # logged and surfaced, never a silent empty result.
        with pytest.raises(PermissionError, match=SCOPE_OPS_DETECTION):
            customer().require(SCOPE_OPS_DETECTION)

    def test_require_is_silent_when_permitted(self):
        assert manager().require(SCOPE_OPS_DETECTION) is None
