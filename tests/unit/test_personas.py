"""Seeded personas are the mocked login surface (D17).

They are chosen so that every distinct policy situation in the pack is
reachable in one click, and so that the role split of D26 is visible on real
rows rather than only in the scope table.
"""

from __future__ import annotations

import pytest

from src.auth.personas import PERSONAS, get_persona, list_personas, to_principal
from src.auth.principal import (
    SCOPE_APPROVE_CREDIT,
    SCOPE_OPS_DETECTION,
    SCOPE_READ_ANY_ACCOUNT,
)

# The four accounts in the workbook. Cross-checked against the real database
# in the M1 integration suite; asserted here so a typo fails fast.
WORKBOOK_ACCOUNTS = {"ACCT-001", "ACCT-002", "ACCT-003", "ACCT-004"}

# The two names that appear in tickets.assigned_to.
WORKBOOK_ASSIGNEES = {"Maya", "Rohit"}


class TestRegistry:
    def test_persona_ids_are_unique(self):
        ids = [p.persona_id for p in PERSONAS]
        assert len(ids) == len(set(ids))

    def test_unknown_persona_is_rejected(self):
        with pytest.raises(LookupError, match="nobody"):
            get_persona("nobody")

    def test_list_personas_is_ordered_customers_first(self):
        roles = [p.role for p in list_personas()]
        assert roles == sorted(roles, key=lambda r: r != "customer")

    def test_every_persona_has_a_description_for_the_picker(self):
        for p in PERSONAS:
            assert p.description.strip(), p.persona_id


class TestCustomerPersonas:
    def test_covers_all_four_workbook_accounts(self):
        # Each account is a distinct policy situation: Northstar's agreement
        # overrides, LumenWorks' declines to override cancellation but
        # replaces the credit terms, Beacon has no agreement, and Axis Labs
        # is Enterprise without premium support.
        seen = {p.account_id for p in PERSONAS if p.role == "customer"}
        assert seen == WORKBOOK_ACCOUNTS

    def test_customers_are_scoped_to_their_own_account(self):
        for persona in (p for p in PERSONAS if p.role == "customer"):
            principal = to_principal(persona)
            assert principal.account_id == persona.account_id
            assert not principal.has(SCOPE_READ_ANY_ACCOUNT)

    def test_customers_have_no_queue(self):
        assert all(p.queue_key is None for p in PERSONAS if p.role == "customer")


class TestStaffPersonas:
    def test_support_agents_match_the_assignees_in_the_data(self):
        # my_queue is driven by tickets.assigned_to, so a persona whose
        # queue_key does not appear there would return an empty queue and
        # quietly look broken.
        keys = {p.queue_key for p in PERSONAS if p.role == "support_agent"}
        assert keys == WORKBOOK_ASSIGNEES

    def test_there_is_exactly_one_ops_manager(self):
        managers = [p for p in PERSONAS if p.role == "ops_manager"]
        assert len(managers) == 1
        assert managers[0].display_name == "Priya Mehta"

    def test_staff_carry_no_account_id(self):
        assert all(p.account_id is None for p in PERSONAS if p.role != "customer")

    def test_only_the_manager_can_run_detection_or_approve_credit(self):
        by_role = {p.role: to_principal(p) for p in PERSONAS}
        assert by_role["ops_manager"].has(SCOPE_OPS_DETECTION)
        assert by_role["ops_manager"].has(SCOPE_APPROVE_CREDIT)
        assert not by_role["support_agent"].has(SCOPE_OPS_DETECTION)
        assert not by_role["support_agent"].has(SCOPE_APPROVE_CREDIT)


class TestConversion:
    def test_every_persona_builds_a_valid_principal(self):
        for persona in PERSONAS:
            principal = to_principal(persona)
            assert principal.user_id == persona.persona_id
            assert principal.role == persona.role
            assert principal.display_name == persona.display_name

    def test_round_trips_by_id(self):
        for persona in PERSONAS:
            assert to_principal(get_persona(persona.persona_id)) == to_principal(persona)
