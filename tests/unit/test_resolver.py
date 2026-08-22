"""The precedence resolver (ARCHITECTURE section 6).

Deterministic, and operating on the clause registry rather than on whatever
retrieval happened to return. Retrieval decides what is *relevant*; this decides
what is *binding*, and conflating the two is how a system ends up quoting the
best-matching clause instead of the governing one.

The hard cases are not the overrides. They are the clauses that look like
overrides and are not: a Tier 1 agreement that explicitly defers to the SOP, and
a Tier 2 prose clause that shares a topic while stating no rule at all. Both
exist in this pack and both break a naive implementation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.domain.resolver import GENERAL_POLICY, PolicyResolver, UnresolvedConflict

NORTHSTAR_CANCEL = "northstar_logistics_enterprise_agreement::§2"
NORTHSTAR_SLA = "northstar_logistics_enterprise_agreement::§1"
NORTHSTAR_CREDIT = "northstar_logistics_enterprise_agreement::§3"
LUMENWORKS_CANCEL = "lumenworks_service_agreement::§2"
LUMENWORKS_CREDIT = "lumenworks_service_agreement::§3"
SOP_CANCEL = "cancellation_and_service_credit_sop_v4::§1"
SOP_CREDIT = "cancellation_and_service_credit_sop_v4::§2"
V3_TARGETS = "support_policy_v3_current::§3"
V3_SCOPE = "support_policy_v3_current::§1"
V2 = "support_policy_v2_deprecated::§-"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture(scope="module")
def resolver():
    with PolicyResolver.open(get_settings().db_path) as opened:
        yield opened


def ids(refs) -> set[str]:
    return {r.clause_id for r in refs}


class TestTheDiscriminatingPair:
    def test_an_overriding_agreement_governs_and_the_loser_is_recorded(self, resolver):
        resolution = resolver.resolve("cancellation_fee", persona("northstar_customer"))

        assert resolution.governing.clause_id == NORTHSTAR_CANCEL
        assert resolution.governing.tier == 1
        # Recorded, not discarded. The customer is getting an answer that
        # differs from published policy and is entitled to know which rule was
        # displaced - that is the Problem 2 deliverable.
        assert ids(resolution.overridden) == {SOP_CANCEL}
        assert resolution.is_override is True

    def test_a_tier_one_clause_that_declines_does_not_govern(self, resolver):
        resolution = resolver.resolve("cancellation_fee", persona("lumenworks_customer"))

        # The trap. LumenWorks section 2 is Tier 1 and still loses to a Tier 2
        # SOP, because it says to use the SOP.
        assert resolution.governing.clause_id == SOP_CANCEL
        assert resolution.governing.tier == 2
        assert resolution.overridden == ()
        assert resolution.is_override is False

    def test_the_declining_clause_is_still_cited(self, resolver):
        resolution = resolver.resolve("cancellation_fee", persona("lumenworks_customer"))
        # "Your agreement was checked and defers to standard policy" is a
        # different answer from "you have no agreement", and dropping the clause
        # makes the two indistinguishable.
        assert ids(resolution.deferred) == {LUMENWORKS_CANCEL}

    def test_an_account_with_no_agreement_has_nothing_deferring(self, resolver):
        resolution = resolver.resolve("cancellation_fee", persona("beacon_customer"))
        assert resolution.governing.clause_id == SOP_CANCEL
        assert resolution.deferred == ()
        assert resolution.overridden == ()


class TestOverrideDirections:
    def test_an_agreement_can_override_in_the_customers_favour(self, resolver):
        resolution = resolver.resolve("failed_pickup_credit", persona("lumenworks_customer"))
        assert resolution.governing.clause_id == LUMENWORKS_CREDIT
        assert ids(resolution.overridden) == {SOP_CREDIT}

    def test_an_agreement_can_decline_on_one_topic_while_overriding_on_another(self, resolver):
        # LumenWorks defers on cancellation and overrides on credits. A resolver
        # that caches "this account has an agreement" gets one of the two wrong.
        cancellation = resolver.resolve("cancellation_fee", persona("lumenworks_customer"))
        credit = resolver.resolve("failed_pickup_credit", persona("lumenworks_customer"))
        assert cancellation.governing.tier == 2
        assert credit.governing.tier == 1

    def test_northstar_declines_on_credits_despite_overriding_on_cancellation(self, resolver):
        credit = resolver.resolve("failed_pickup_credit", persona("northstar_customer"))
        assert credit.governing.clause_id == SOP_CREDIT
        assert ids(credit.deferred) == {NORTHSTAR_CREDIT}

    def test_a_baseline_clause_is_not_mistaken_for_a_decliner(self, resolver):
        # overrides is null on the SOP, meaning "I am the baseline". Reading
        # null as false would leave nothing governing at all.
        resolution = resolver.resolve("cancellation_fee", persona("beacon_customer"))
        assert resolution.governing.params.get("overrides") is None
        assert resolution.governing.clause_id == SOP_CANCEL


class TestProseClausesAreNotRivals:
    """The failure a naive same-tier conflict check walks into.

    `first_response_target` is carried by three Tier 2 clauses: the targets
    grid, the scope-and-precedence preamble, and the escalation duty. Only one
    states a value. Treating all three as competing authorities reports an
    unresolved conflict on every SLA question in the pack.
    """

    def test_the_clause_stating_a_value_governs(self, resolver):
        resolution = resolver.resolve("first_response_target", persona("beacon_customer"))
        assert resolution.governing.clause_id == V3_TARGETS
        assert resolution.unresolved_conflict is None

    def test_the_prose_clauses_are_kept_as_supporting_not_dropped(self, resolver):
        resolution = resolver.resolve("first_response_target", persona("beacon_customer"))
        # Policy v3 section 1 is the clause that states the precedence rule the
        # whole system implements. It must remain citable.
        assert V3_SCOPE in ids(resolution.supporting)

    def test_an_agreement_still_overrides_only_the_clause_that_states_a_value(self, resolver):
        resolution = resolver.resolve("first_response_target", persona("northstar_customer"))
        assert resolution.governing.clause_id == NORTHSTAR_SLA
        assert ids(resolution.overridden) == {V3_TARGETS}
        assert V3_SCOPE in ids(resolution.supporting)


class TestTierDiscipline:
    def test_the_deprecated_policy_is_excluded_with_a_reason(self, resolver):
        resolution = resolver.resolve("first_response_target", persona("axis_customer"))
        excluded = {r.clause_id: r.reason for r in resolution.excluded}
        assert excluded == {V2: "deprecated"}

    def test_the_deprecated_policy_never_governs_for_any_persona(self, resolver):
        for persona_id in (
            "northstar_customer",
            "lumenworks_customer",
            "beacon_customer",
            "axis_customer",
            "maya_agent",
            "priya_manager",
        ):
            scope = {"maya_agent": "ACCT-001", "priya_manager": "ACCT-003"}.get(persona_id)
            for topic in ("first_response_target", "severity_definition"):
                resolution = resolver.resolve(topic, persona(persona_id), account_id=scope)
                assert resolution.governing is None or resolution.governing.tier < 4
                assert all(r.tier < 4 for r in resolution.citable)

    def test_the_deprecated_policy_is_reachable_when_asked_for_deliberately(self, resolver):
        # GS-018: "what changed between v2 and v3?" is a legitimate question.
        # Exclusion is a predicate that can be lifted, not an absent document.
        resolution = resolver.resolve(
            "first_response_target",
            persona("priya_manager"),
            account_id=GENERAL_POLICY,
            include_tiers={1, 2, 3, 4},
        )
        assert V2 in ids(resolution.citable)


class TestAccountScoping:
    def test_no_account_resolves_against_another_accounts_clause(self, resolver):
        for persona_id, account in (
            ("northstar_customer", "ACCT-001"),
            ("lumenworks_customer", "ACCT-002"),
            ("beacon_customer", "ACCT-003"),
            ("axis_customer", "ACCT-004"),
        ):
            for topic in ("cancellation_fee", "failed_pickup_credit", "first_response_target"):
                resolution = resolver.resolve(topic, persona(persona_id))
                for ref in resolution.all_clauses:
                    assert ref.account_id in (None, account), (
                        f"{persona_id} resolved against {ref.clause_id}"
                    )

    def test_staff_must_name_the_account_they_are_resolving_for(self, resolver):
        # A support agent reads every account, so "which account" is not
        # implied by the Principal and cannot be guessed. Resolving policy
        # without it would silently return general policy and quietly drop
        # every agreement.
        with pytest.raises(ValueError, match="account"):
            resolver.resolve("cancellation_fee", persona("maya_agent"))

    def test_staff_resolving_for_an_account_get_that_accounts_agreement(self, resolver):
        resolution = resolver.resolve(
            "cancellation_fee", persona("maya_agent"), account_id="ACCT-001"
        )
        assert resolution.governing.clause_id == NORTHSTAR_CANCEL

    def test_a_customer_cannot_widen_scope_by_naming_another_account(self, resolver):
        with pytest.raises(PermissionError):
            resolver.resolve(
                "cancellation_fee", persona("lumenworks_customer"), account_id="ACCT-001"
            )


class TestUnknownTopic:
    def test_a_topic_with_no_clauses_resolves_to_nothing_rather_than_raising(self, resolver):
        # GS-024 and GS-025: the no-source case. This must be a representable
        # answer, because "I have no basis for this" is the correct response and
        # an exception would be indistinguishable from a bug.
        resolution = resolver.resolve("data_retention", persona("northstar_customer"))
        assert resolution.governing is None
        assert resolution.has_basis is False
        assert resolution.citable == ()

    def test_a_resolution_with_a_governing_clause_has_a_basis(self, resolver):
        assert resolver.resolve("cancellation_fee", persona("beacon_customer")).has_basis


class TestSameTierConflict:
    """No such case exists in the pack (findings section 8).

    The branch exists because a resolver that cannot represent a conflict will
    silently pick one, and silently picking is the failure mode the whole
    design is arranged to avoid. Covered by a synthetic fixture, and labelled
    as synthetic rather than dressed up as a real case.
    """

    @pytest.fixture
    def conflicted(self, tmp_path: Path):
        db = tmp_path / "conflict.db"
        conn = sqlite3.connect(db)
        conn.executescript((Path("src/datastore/schema.sql")).read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO accounts (account_id, account_name, plan, status, premium_support) "
            "VALUES (?,?,?,?,?)",
            ("ACCT-001", "Northstar Logistics", "Enterprise", "active", 1),
        )
        for ref, params in (
            ("§A", {"fee_inr": 0, "overrides": True}),
            ("§B", {"fee_inr": 99, "overrides": True}),
        ):
            conn.execute(
                "INSERT INTO clauses (clause_id, doc_id, doc_title, clause_ref, title, tier, "
                "account_id, status, effective_from, params, text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"synthetic_agreement::{ref}",
                    "synthetic_agreement",
                    "Synthetic Agreement",
                    ref,
                    f"Clause {ref}",
                    1,
                    "ACCT-001",
                    "ACTIVE",
                    "2026-01-01",
                    json.dumps(params),
                    f"Synthetic clause {ref}.",
                ),
            )
            conn.execute(
                "INSERT INTO clause_topics (clause_id, topic) VALUES (?,?)",
                (f"synthetic_agreement::{ref}", "cancellation_fee"),
            )
        conn.commit()
        conn.close()
        with PolicyResolver.open(db) as opened:
            yield opened

    def test_two_same_tier_clauses_with_different_params_do_not_resolve(self, conflicted):
        resolution = conflicted.resolve("cancellation_fee", persona("northstar_customer"))
        assert resolution.governing is None
        assert isinstance(resolution.unresolved_conflict, UnresolvedConflict)

    def test_the_conflict_names_the_clauses_and_the_differing_keys(self, conflicted):
        conflict = conflicted.resolve(
            "cancellation_fee", persona("northstar_customer")
        ).unresolved_conflict
        assert ids(conflict.clauses) == {
            "synthetic_agreement::§A",
            "synthetic_agreement::§B",
        }
        assert conflict.differing_params == ("fee_inr",)
        assert conflict.tier == 1

    def test_a_conflicted_resolution_has_no_basis_to_answer_from(self, conflicted):
        resolution = conflicted.resolve("cancellation_fee", persona("northstar_customer"))
        # A calculator must refuse this rather than pick a side, so the flag has
        # to be unambiguous.
        assert resolution.has_basis is False

    def test_identical_params_at_the_same_tier_are_not_a_conflict(self, tmp_path):
        db = tmp_path / "twins.db"
        conn = sqlite3.connect(db)
        conn.executescript(Path("src/datastore/schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO accounts (account_id, account_name, plan, status, premium_support) "
            "VALUES (?,?,?,?,?)",
            ("ACCT-001", "Northstar Logistics", "Enterprise", "active", 1),
        )
        for ref in ("§A", "§B"):
            conn.execute(
                "INSERT INTO clauses (clause_id, doc_id, doc_title, clause_ref, title, tier, "
                "account_id, status, effective_from, params, text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"twin::{ref}",
                    "twin",
                    "Twin",
                    ref,
                    f"Clause {ref}",
                    1,
                    "ACCT-001",
                    "ACTIVE",
                    "2026-01-01",
                    json.dumps({"fee_inr": 0, "overrides": True}),
                    "Same rule, stated twice.",
                ),
            )
            conn.execute(
                "INSERT INTO clause_topics (clause_id, topic) VALUES (?,?)",
                (f"twin::{ref}", "cancellation_fee"),
            )
        conn.commit()
        conn.close()
        with PolicyResolver.open(db) as resolver:
            resolution = resolver.resolve("cancellation_fee", persona("northstar_customer"))
        # They say the same thing, so citing either is complete. Reporting a
        # conflict here would escalate a question that has one clear answer.
        assert resolution.unresolved_conflict is None
        assert resolution.governing.clause_id == "twin::§A"
        assert ids(resolution.supporting) == {"twin::§B"}


class TestEffectiveDates:
    def test_a_clause_outside_its_effective_window_is_excluded(self, tmp_path):
        db = tmp_path / "expired.db"
        conn = sqlite3.connect(db)
        conn.executescript(Path("src/datastore/schema.sql").read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO accounts (account_id, account_name, plan, status, premium_support) "
            "VALUES (?,?,?,?,?)",
            ("ACCT-001", "Northstar Logistics", "Enterprise", "active", 1),
        )
        conn.execute(
            "INSERT INTO clauses (clause_id, doc_id, doc_title, clause_ref, title, tier, "
            "account_id, status, effective_from, effective_to, params, text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "lapsed::§1",
                "lapsed",
                "Lapsed Agreement",
                "§1",
                "Cancellation",
                1,
                "ACCT-001",
                "ACTIVE",
                "2025-01-01",
                "2026-01-31",
                json.dumps({"fee_inr": 0, "overrides": True}),
                "Expired waiver.",
            ),
        )
        conn.execute(
            "INSERT INTO clause_topics (clause_id, topic) VALUES (?,?)",
            ("lapsed::§1", "cancellation_fee"),
        )
        conn.commit()
        conn.close()
        with PolicyResolver.open(db) as resolver:
            resolution = resolver.resolve("cancellation_fee", persona("northstar_customer"))
        assert resolution.governing is None
        assert {r.reason for r in resolution.excluded} == {"not_effective"}


class TestResolutionShape:
    def test_a_clause_ref_carries_what_a_citation_needs(self, resolver):
        governing = resolver.resolve("cancellation_fee", persona("northstar_customer")).governing
        assert governing.citation
        assert governing.doc_title and governing.clause_ref and governing.title
        assert governing.params["fee_inr"] == 0

    def test_every_clause_appears_in_exactly_one_bucket(self, resolver):
        for topic in ("cancellation_fee", "failed_pickup_credit", "first_response_target"):
            for persona_id in ("northstar_customer", "lumenworks_customer", "beacon_customer"):
                resolution = resolver.resolve(topic, persona(persona_id))
                seen = [r.clause_id for r in resolution.all_clauses]
                assert len(seen) == len(set(seen)), f"{topic}/{persona_id} double-counts a clause"

    def test_the_resolution_is_serialisable_for_an_evidence_payload(self, resolver):
        resolution = resolver.resolve("cancellation_fee", persona("northstar_customer"))
        payload = resolution.to_payload()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["governing"]["clause_id"] == NORTHSTAR_CANCEL
        assert payload["topic"] == "cancellation_fee"
