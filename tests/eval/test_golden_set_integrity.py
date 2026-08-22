"""The golden set is internally consistent with the data (D28).

This does NOT assert that the system produces these answers - nothing may
depend on the verdicts until they are signed off, and the components that would
produce them do not exist yet. What it asserts is that the file is a valid
target: the arithmetic matches the database, every clause reference resolves,
every persona exists, and the structure is uniform.

The reason to have it now rather than at M11 is that the golden set is about to
become the thing every later milestone is measured against. A typo in a clause
id makes an assertion vacuous - it passes, and it checks nothing. Better to
catch that here than to discover in the evaluation writeup that six tests were
green because they were asserting against a clause that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.review_golden_set import GOLDEN_PATH, Checker

from src.config import get_settings

VALID_SURFACE = {
    "override",
    "no_override",
    "stale_status_conflict",
    "historical_contradiction",
    "sla_breach",
    "not_measurable",
    "known_issue",
    "manager_approval",
    "inferred_subject",
    "escalate",
    "access_denied",
}


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return yaml.safe_load(Path(GOLDEN_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checked(entries) -> Checker:
    checker = Checker(get_settings().db_path)
    for entry in entries:
        checker.check_entry(entry)
    return checker


class TestArithmetic:
    def test_every_derived_quantity_matches_the_data(self, checked):
        assert checked.failures == []

    def test_the_checker_can_actually_fail(self, checked, entries):
        # Guards the test above from being vacuous. If `check` blocks stopped
        # being read, everything would pass and nothing would be verified.
        corrupted = dict(entries[0])
        corrupted["check"] = {"elapsed_minutes": -1}
        before = len(checked.failures)
        checked.check_entry(corrupted)
        assert len(checked.failures) > before
        del checked.failures[before:]


class TestStructure:
    def test_the_set_is_the_size_the_architecture_calls_for(self, entries):
        assert len(entries) >= 30

    def test_ids_are_unique_and_sequential(self, entries):
        ids = [e["id"] for e in entries]
        assert len(set(ids)) == len(ids)
        assert ids == [f"GS-{n:03d}" for n in range(1, len(ids) + 1)]

    @pytest.mark.parametrize(
        "field", ["id", "category", "question", "persona", "expect", "derivation"]
    )
    def test_every_entry_carries_the_required_fields(self, entries, field):
        missing = [e.get("id", "?") for e in entries if not e.get(field)]
        assert missing == [], f"missing {field}: {missing}"

    def test_every_entry_explains_itself(self, entries):
        # A verdict with no derivation cannot be reviewed, which makes the sign
        # -off meaningless for that row.
        thin = [e["id"] for e in entries if len((e.get("derivation") or "").split()) < 25]
        assert thin == []

    def test_surface_flags_come_from_the_closed_vocabulary(self, entries):
        unknown = {
            flag
            for entry in entries
            for flag in (entry["expect"].get("must_surface") or [])
            if flag not in VALID_SURFACE
        }
        assert unknown == set()

    def test_escalation_entries_say_so_in_both_places(self, entries):
        for entry in entries:
            expect = entry["expect"]
            surfaced = "escalate" in (expect.get("must_surface") or [])
            if surfaced:
                assert expect.get("escalate") is True, entry["id"]


class TestCoverage:
    """The set must contain the cases the architecture names (section 18)."""

    def test_the_discriminating_pair_is_present_and_opposite(self, entries):
        # Selected by id, not by subject: ORD-1001 is also the subject of the
        # staleness-conflict entry, which has no amount at all.
        by_id = {e["id"]: e for e in entries}
        northstar, lumenworks = by_id["GS-001"], by_id["GS-002"]
        assert northstar["expect"]["amount_inr"] == 0
        assert lumenworks["expect"]["amount_inr"] == 250
        # Same shape, opposite answers - that is the whole point of the pair.
        assert northstar["check"]["order_status"] == lumenworks["check"]["order_status"]

    def test_the_three_hour_credit_question_is_asked_from_both_sides(self, entries):
        asked = [e for e in entries if "three hours late" in e["question"]]
        assert len(asked) == 2
        verdicts = {e["persona"]: e["expect"]["verdict"] for e in asked}
        assert verdicts == {
            "lumenworks_customer": "not_eligible",
            "beacon_customer": "eligible",
        }

    def test_a_no_source_question_must_escalate(self, entries):
        no_source = [e for e in entries if e["expect"].get("verdict") == "no_source"]
        assert len(no_source) >= 2
        assert all(e["expect"]["escalate"] for e in no_source)
        assert all(not e["expect"]["governing"] for e in no_source)

    def test_the_deprecated_policy_is_forbidden_widely_and_permitted_once(self, entries):
        deprecated = "support_policy_v2_deprecated::§-"
        forbidden = [e for e in entries if deprecated in (e["expect"].get("must_not_cite") or [])]
        permitted = [e for e in entries if deprecated in (e["expect"].get("context_only") or [])]
        assert len(forbidden) >= 6
        assert len(permitted) == 1
        assert permitted[0]["expect"]["tier_4_read_permitted"] is True

    def test_cross_account_probes_are_denied(self, entries):
        denials = [e for e in entries if e["category"] == "access_control"]
        assert len(denials) >= 4
        assert all(
            e["expect"]["verdict"] in {"denied", "denied_or_general_only", "refused"}
            for e in denials
        )

    def test_no_customer_facing_denial_leaks_the_thing_it_denies(self, entries):
        # Scoped to customer sessions. A staff scope refusal - Maya being told
        # she cannot approve a credit - has nothing to leak: she may already see
        # the account, and the refusal reveals only her own permissions. The
        # leak risk is a customer learning that another account's order or
        # contract term exists.
        leaky = [
            e["id"]
            for e in entries
            if e["category"] == "access_control"
            and e["persona"].endswith("_customer")
            and not e["expect"].get("must_not_leak")
        ]
        assert leaky == []

    def test_missing_amounts_are_null_never_zero(self, entries):
        # "Missing data is not zero" applied to the expectations themselves: a
        # golden set that encodes INR 0 for "not applicable" teaches the system
        # to render the same lie.
        for entry in entries:
            expect = entry["expect"]
            if expect.get("verdict") in {"not_cancellable", "not_eligible"}:
                assert expect.get("amount_inr") is None, entry["id"]

    def test_every_persona_is_exercised(self, entries):
        from src.auth.personas import PERSONAS

        used = {e["persona"] for e in entries}
        assert used == {p.persona_id for p in PERSONAS}
