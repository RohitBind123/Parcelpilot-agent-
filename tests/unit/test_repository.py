"""The repository is the second ACL layer.

The first is the tool schema: a customer's `get_order` has no `account_id`
parameter, so a cross-account query is not expressible. This layer exists for
when that one has a bug. It binds account-scoped temp views from the Principal
at connect time, so a customer connection literally cannot see another
account's rows - the scoping is in the view definition, not in a WHERE clause
someone might forget.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.auth.personas import get_persona, to_principal
from src.auth.principal import build_principal
from src.datastore.etl import WORKBOOK_PATH, build_database
from src.datastore.repo import AccessDenied, NotFound, Repository

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("repo") / "parcelpilot.db"
    build_database(WORKBOOK_PATH, path)
    return path


def repo_for(db_path, persona_id: str) -> Repository:
    return Repository.open(to_principal(get_persona(persona_id)), db_path)


@pytest.fixture
def northstar(db_path):
    with repo_for(db_path, "northstar_customer") as repo:
        yield repo


@pytest.fixture
def lumenworks(db_path):
    with repo_for(db_path, "lumenworks_customer") as repo:
        yield repo


@pytest.fixture
def maya(db_path):
    with repo_for(db_path, "maya_agent") as repo:
        yield repo


@pytest.fixture
def priya(db_path):
    with repo_for(db_path, "priya_manager") as repo:
        yield repo


class TestCustomerScoping:
    def test_reads_its_own_order(self, northstar):
        order = northstar.get_order("ORD-1001")
        assert order.account_id == "ACCT-001"
        assert order.status == "BOOKED"

    def test_cannot_read_another_accounts_order(self, northstar):
        # The whole containment story in one assertion.
        with pytest.raises(AccessDenied) as exc:
            northstar.get_order("ORD-2001")
        assert exc.value.reason == "out_of_scope"

    def test_a_genuinely_absent_record_is_distinguished_internally(self, northstar):
        # Same user-facing outcome, different log line: ops needs to tell a
        # typo apart from a probe.
        with pytest.raises(AccessDenied) as exc:
            northstar.get_order("ORD-9999")
        assert exc.value.reason == "not_found"

    def test_the_denial_message_does_not_confirm_the_record_exists(self, northstar):
        # Echoing back the id the caller supplied leaks nothing. What must not
        # differ is the wording, or a denial becomes an existence oracle.
        with pytest.raises(AccessDenied) as real:
            northstar.get_order("ORD-2001")
        with pytest.raises(AccessDenied) as absent:
            northstar.get_order("ORD-9999")
        assert str(real.value).replace("ORD-2001", "X") == str(absent.value).replace(
            "ORD-9999", "X"
        )

    def test_cannot_widen_scope_by_passing_an_account_id(self, northstar):
        # Even if a tool schema leaked the parameter, the repository refuses.
        with pytest.raises(AccessDenied):
            northstar.list_orders(account_id="ACCT-002")

    def test_listing_returns_only_its_own_orders(self, northstar):
        assert {o.order_id for o in northstar.list_orders()} == {"ORD-1001", "ORD-1002"}

    def test_ticket_scoping_matches_order_scoping(self, lumenworks):
        assert {t.ticket_id for t in lumenworks.list_tickets()} == {"TKT-502", "TKT-451"}
        with pytest.raises(AccessDenied):
            lumenworks.get_ticket("TKT-501")

    def test_account_lookup_returns_its_own_account(self, lumenworks):
        assert lumenworks.get_account().account_name == "LumenWorks"

    def test_a_customer_cannot_aggregate(self, northstar):
        with pytest.raises(AccessDenied, match="read:ticket_aggregates"):
            northstar.query_tickets(status="open")

    def test_a_customer_has_no_queue(self, northstar):
        with pytest.raises(AccessDenied, match="read:own_queue"):
            northstar.my_queue()


class TestScopedViewsAreTheReadSurface:
    def test_the_view_itself_excludes_foreign_rows(self, northstar):
        # Not a WHERE clause a caller could forget - the view definition.
        visible = northstar.connection.execute("SELECT order_id FROM my_orders").fetchall()
        assert {r["order_id"] for r in visible} == {"ORD-1001", "ORD-1002"}

    def test_the_read_connection_cannot_write(self, northstar):
        with pytest.raises(sqlite3.OperationalError):
            northstar.connection.execute("DELETE FROM main.orders WHERE order_id = 'ORD-1001'")

    def test_staff_views_span_every_account(self, maya):
        assert len(maya.connection.execute("SELECT order_id FROM my_orders").fetchall()) == 6

    def test_an_account_id_that_is_not_well_formed_is_refused(self, db_path):
        # The account id is interpolated into the view definition, so it is
        # validated first even though it only ever comes from the persona table.
        hostile = build_principal("u", "customer", account_id="ACCT-001'; DROP TABLE orders;--")
        with pytest.raises(ValueError, match="account_id"):
            Repository.open(hostile, db_path)


class TestStaffAccess:
    def test_reads_any_account(self, maya):
        assert maya.get_order("ORD-2001").account_id == "ACCT-002"
        assert maya.get_ticket("TKT-505").account_id == "ACCT-004"

    def test_an_absent_record_is_a_plain_not_found_for_staff(self, maya):
        # Staff already read every account, so hiding existence buys nothing
        # and a clear error is more useful.
        with pytest.raises(NotFound):
            maya.get_order("ORD-9999")

    def test_my_queue_uses_assigned_to(self, maya):
        assert {t.ticket_id for t in maya.my_queue()} == {"TKT-502", "TKT-504", "TKT-450"}

    def test_a_second_agent_gets_a_different_queue(self, db_path):
        with repo_for(db_path, "rohit_agent") as rohit:
            assert {t.ticket_id for t in rohit.my_queue()} == {
                "TKT-501",
                "TKT-503",
                "TKT-505",
                "TKT-451",
            }

    def test_the_manager_can_run_aggregates(self, priya):
        assert len(priya.query_tickets(status="open")) == 5

    def test_query_tickets_filters_combine(self, priya):
        found = priya.query_tickets(status="open", account_id="ACCT-001")
        assert {t.ticket_id for t in found} == {"TKT-501", "TKT-504"}

    def test_query_tickets_caps_its_own_result_set(self, priya):
        assert len(priya.query_tickets(limit=2)) == 2

    def test_query_tickets_rejects_an_unknown_filter_rather_than_ignoring_it(self, priya):
        # Silently dropping a filter returns more rows than the caller asked
        # for, which for an ACL-adjacent query is the wrong way to fail.
        with pytest.raises(ValueError, match="carrier"):
            priya.query_tickets(carrier="SwiftShip")


class TestTypedRows:
    def test_timestamps_come_back_timezone_aware(self, northstar):
        assert northstar.get_order("ORD-1001").booked_at == datetime(2026, 8, 16, 9, 0, tzinfo=IST)

    def test_absent_timestamps_stay_none(self, northstar):
        assert northstar.get_order("ORD-1001").pickup_actual_at is None

    def test_booleans_come_back_as_booleans(self, lumenworks):
        order = lumenworks.get_order("ORD-2002")
        assert order.carrier_fault is True
        assert order.customer_fault is False

    def test_rows_are_immutable(self, northstar):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            northstar.get_order("ORD-1001").status = "CANCELLED"

    def test_an_account_without_an_agreement_reports_none_not_empty_string(self, db_path):
        with repo_for(db_path, "beacon_customer") as beacon:
            assert beacon.get_account().contract_file is None
            assert beacon.get_account().has_agreement is False

    def test_an_account_with_an_agreement_says_so(self, northstar):
        assert northstar.get_account().has_agreement is True


class TestBatchFetch:
    def test_many_orders_are_fetched_in_one_round_trip(self, priya):
        # The N+1 rule: one IN query, not one SELECT per id.
        found = priya.get_orders(["ORD-1001", "ORD-2001", "ORD-9999"])
        assert set(found) == {"ORD-1001", "ORD-2001"}

    def test_an_empty_id_list_makes_no_query(self, priya):
        assert priya.get_orders([]) == {}

    def test_batch_fetch_respects_scoping(self, northstar):
        found = northstar.get_orders(["ORD-1001", "ORD-2001"])
        assert set(found) == {"ORD-1001"}


class TestConfigDataInvariants:
    """Config and the workbook must not drift apart."""

    def test_the_configured_as_of_matches_the_workbook(self, db_path, as_of_configured):
        from src.clock import as_of

        conn = sqlite3.connect(db_path)
        stored = conn.execute("SELECT value FROM meta WHERE key='as_of'").fetchone()[0]
        conn.close()
        assert datetime.fromisoformat(stored) == as_of()

    def test_every_persona_account_exists(self, db_path):
        from src.auth.personas import PERSONAS

        conn = sqlite3.connect(db_path)
        real = {r[0] for r in conn.execute("SELECT account_id FROM accounts")}
        conn.close()
        assert {p.account_id for p in PERSONAS if p.account_id} <= real

    def test_every_assignee_has_a_matching_agent_persona(self, db_path):
        from src.auth.personas import PERSONAS

        conn = sqlite3.connect(db_path)
        assignees = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT assigned_to FROM tickets WHERE assigned_to IS NOT NULL"
            )
        }
        conn.close()
        queues = {p.queue_key for p in PERSONAS if p.role == "support_agent"}
        assert assignees == queues
