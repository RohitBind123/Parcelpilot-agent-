"""What the prompts must not contain.

Non-negotiable 2: access control lives in the tool layer, never in a prompt.
A sentence like "never access another account" is worse than silence - it
implies the boundary depends on the model's cooperation, and it hands a prompt
injection something to argue with. These tests are the enforcement.
"""

from __future__ import annotations

import pytest

from src.agent.prompts import system_prompt
from src.auth.personas import PERSONAS, get_persona, to_principal

ALL = [p.persona_id for p in PERSONAS]


def prompt(persona_id: str) -> str:
    return system_prompt(to_principal(get_persona(persona_id)))


class TestAccessControlIsNotInHere:
    @pytest.mark.parametrize("persona_id", ALL)
    def test_no_prompt_names_an_account_id(self, persona_id):
        # A prompt carrying ACCT-001 would make the boundary a string the model
        # is trusted to compare against.
        assert "ACCT-" not in prompt(persona_id)

    @pytest.mark.parametrize("persona_id", ALL)
    def test_no_prompt_tells_the_model_to_refuse_anything(self, persona_id):
        text = prompt(persona_id).lower()
        for word in ("refuse", "deny", "not allowed", "forbidden", "unauthorised", "permission"):
            assert word not in text

    @pytest.mark.parametrize("persona_id", ALL)
    def test_no_prompt_enumerates_tools(self, persona_id):
        # The schema already says which tools exist. A list here would go stale
        # and, worse, would describe tools the projection withheld.
        text = prompt(persona_id)
        for tool in ("my_queue", "scan_support_health", "approve_credit", "query_tickets"):
            assert tool not in text

    @pytest.mark.parametrize("persona_id", ALL)
    def test_no_prompt_mentions_another_account_by_name(self, persona_id):
        text = prompt(persona_id)
        for name in ("Northstar", "LumenWorks", "Beacon", "Axis"):
            assert name not in text


class TestWhatTheSchemaCannotSay:
    @pytest.mark.parametrize("persona_id", ALL)
    def test_every_prompt_requires_a_citation(self, persona_id):
        assert "citation" in prompt(persona_id).lower()

    @pytest.mark.parametrize("persona_id", ALL)
    def test_every_prompt_forbids_doing_the_arithmetic(self, persona_id):
        assert "do not calculate" in prompt(persona_id).lower()

    @pytest.mark.parametrize("persona_id", ALL)
    def test_every_prompt_says_missing_is_not_zero(self, persona_id):
        assert "not zero" in prompt(persona_id).lower()

    @pytest.mark.parametrize("persona_id", ALL)
    def test_every_prompt_permits_saying_i_do_not_know(self, persona_id):
        assert "do not have a source" in prompt(persona_id).lower()

    @pytest.mark.parametrize("persona_id", ALL)
    def test_every_prompt_explains_the_handle_discipline(self, persona_id):
        text = prompt(persona_id)
        assert "snapshot_id" in text and "resolution_id" in text


class TestTheSurfacesDiffer:
    def test_a_customer_and_an_agent_get_different_instructions(self):
        assert prompt("northstar_customer") != prompt("maya_agent")

    def test_only_staff_are_told_about_tier_five_history(self):
        assert "Historical tickets" in prompt("maya_agent")
        assert "Historical tickets" not in prompt("northstar_customer")

    def test_only_staff_are_warned_about_the_unmeasurable_breach(self):
        # A4. A customer never sees an SLA calculation, so the caveat would be
        # noise; an agent does, and would otherwise report a breach.
        assert "confirmed breach" in prompt("rohit_agent")
        assert "confirmed breach" not in prompt("axis_customer")

    def test_a_manager_is_a_superset_of_an_agent(self):
        agent, manager = prompt("maya_agent"), prompt("priya_manager")
        assert manager.startswith(agent)
        assert len(manager) > len(agent)

    def test_both_agents_get_the_same_prompt(self):
        assert prompt("maya_agent") == prompt("rohit_agent")

    def test_all_customers_get_the_same_prompt(self):
        prompts = {prompt(p) for p in ("northstar_customer", "beacon_customer", "axis_customer")}
        assert len(prompts) == 1
