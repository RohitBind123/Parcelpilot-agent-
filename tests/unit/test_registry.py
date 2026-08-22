"""The clause registry in SQLite.

The point of persisting clauses relationally, rather than leaving precedence to
whatever retrieval returns, is that the resolver's central question becomes one
indexed query: every clause about this subject, visible to this account,
ordered by tier. This file tests that query directly, because M3's resolver is
a thin wrapper over it.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.datastore.etl import WORKBOOK_PATH, build_database
from src.knowledge.ingest import build_registry

RESOLVER_QUERY = """
    SELECT cl.clause_id, cl.tier, cl.account_id, cl.params
    FROM clauses cl
    JOIN clause_topics t ON t.clause_id = cl.clause_id
    WHERE t.topic = :topic
      AND (cl.account_id = :account OR cl.account_id IS NULL)
      AND cl.tier < 4
    ORDER BY cl.tier
"""


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("registry") / "parcelpilot.db"
    build_database(WORKBOOK_PATH, path)
    build_registry(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def resolve(db, topic: str, account: str | None) -> list[sqlite3.Row]:
    return db.execute(RESOLVER_QUERY, {"topic": topic, "account": account}).fetchall()


class TestPersistence:
    def test_every_parsed_clause_is_stored(self, db):
        assert db.execute("SELECT count(*) AS n FROM clauses").fetchone()["n"] == 19

    def test_tiers_are_distributed_as_the_corpus_dictates(self, db):
        counts = dict(db.execute("SELECT tier, count(*) FROM clauses GROUP BY tier").fetchall())
        assert counts == {1: 7, 2: 7, 3: 4, 4: 1}

    def test_no_clause_is_unreachable(self, db):
        # A clause with no topic can never be returned by the resolver.
        orphans = db.execute(
            "SELECT clause_id FROM clauses "
            "WHERE clause_id NOT IN (SELECT clause_id FROM clause_topics)"
        ).fetchall()
        assert orphans == []

    def test_params_round_trip_as_json(self, db):
        row = db.execute(
            "SELECT params FROM clauses WHERE clause_id = ?",
            ("cancellation_and_service_credit_sop_v4::§1",),
        ).fetchone()
        assert json.loads(row["params"])["fee_after_window_inr"] == 250

    def test_agreements_carry_their_account_and_policy_does_not(self, db):
        scoped = dict(
            db.execute("SELECT doc_id, account_id FROM clauses GROUP BY doc_id").fetchall()
        )
        assert scoped["northstar_logistics_enterprise_agreement"] == "ACCT-001"
        assert scoped["lumenworks_service_agreement"] == "ACCT-002"
        assert scoped["cancellation_and_service_credit_sop_v4"] is None

    def test_account_scope_references_a_real_account(self, db):
        db.execute("PRAGMA foreign_keys = ON")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    def test_rebuilding_replaces_rather_than_appends(self, tmp_path):
        path = tmp_path / "twice.db"
        build_database(WORKBOOK_PATH, path)
        assert build_registry(path) == build_registry(path) == 19
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("SELECT count(*) FROM clauses").fetchone()[0] == 19
        finally:
            conn.close()


class TestResolverQuery:
    """The discriminating pair, at the data layer.

    ORD-1001 and ORD-2001 are both BOOKED and both past the 30-minute window.
    Everything that makes their answers differ is visible here.
    """

    def test_northstar_sees_its_own_agreement_first(self, db):
        rows = resolve(db, "cancellation_fee", "ACCT-001")
        assert rows[0]["clause_id"] == "northstar_logistics_enterprise_agreement::§2"
        assert rows[0]["tier"] == 1
        assert json.loads(rows[0]["params"])["overrides"] is True

    def test_lumenworks_sees_a_tier_one_clause_that_declines_to_override(self, db):
        # The case that catches a resolver assuming "an agreement exists, so
        # it wins". The clause is Tier 1 and still not the governing rule.
        rows = resolve(db, "cancellation_fee", "ACCT-002")
        assert rows[0]["clause_id"] == "lumenworks_service_agreement::§2"
        assert json.loads(rows[0]["params"])["overrides"] is False

    def test_an_account_with_no_agreement_sees_only_general_policy(self, db):
        rows = resolve(db, "cancellation_fee", "ACCT-003")
        assert {r["tier"] for r in rows} == {2}
        assert all(r["account_id"] is None for r in rows)

    def test_no_account_ever_sees_another_accounts_agreement(self, db):
        for account in ("ACCT-001", "ACCT-002", "ACCT-003", "ACCT-004"):
            for topic in ("cancellation_fee", "failed_pickup_credit", "first_response_target"):
                foreign = [
                    r["clause_id"]
                    for r in resolve(db, topic, account)
                    if r["account_id"] not in (None, account)
                ]
                assert not foreign, f"{account} saw {foreign}"

    def test_the_deprecated_policy_is_stored_but_never_citable(self, db):
        stored = db.execute("SELECT count(*) AS n FROM clauses WHERE tier = 4").fetchone()["n"]
        assert stored == 1
        # It is excluded by the tier predicate, not by being absent - which is
        # what lets "what changed in v3?" still reach it deliberately.
        assert all(r["tier"] < 4 for r in resolve(db, "first_response_target", "ACCT-001"))

    def test_the_credit_override_resolves_per_account(self, db):
        lumenworks = resolve(db, "failed_pickup_credit", "ACCT-002")
        assert lumenworks[0]["clause_id"] == "lumenworks_service_agreement::§3"
        assert json.loads(lumenworks[0]["params"])["threshold_hours"] == 4

        beacon = resolve(db, "failed_pickup_credit", "ACCT-003")
        assert json.loads(beacon[0]["params"])["threshold_hours"] == 2

    def test_sla_targets_resolve_to_the_agreement_where_one_exists(self, db):
        northstar = resolve(db, "first_response_target", "ACCT-001")
        assert northstar[0]["account_id"] == "ACCT-001"

        axis = resolve(db, "first_response_target", "ACCT-004")
        assert all(r["account_id"] is None for r in axis)
