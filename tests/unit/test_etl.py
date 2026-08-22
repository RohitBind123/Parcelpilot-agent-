"""The workbook becomes a database exactly once, at build time.

Two failure modes are worth more attention than the rest. A naive timestamp
silently shifts every window calculation by the host's offset, and an absent
value coerced to zero turns "we do not know when this was picked up" into
"it was picked up at midnight 1970" - which is the misleading-zero failure the
brief warns about, one layer below the UI.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import pytest
from src.datastore.etl import WORKBOOK_PATH, build_database

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> sqlite3.Connection:
    path = tmp_path_factory.mktemp("etl") / "parcelpilot.db"
    build_database(WORKBOOK_PATH, path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def rows(db, sql, *args) -> list[sqlite3.Row]:
    return db.execute(sql, args).fetchall()


def one(db, sql, *args):
    return db.execute(sql, args).fetchone()


class TestRowCounts:
    @pytest.mark.parametrize("table,expected", [("accounts", 4), ("orders", 6), ("tickets", 7)])
    def test_every_row_is_loaded(self, db, table, expected):
        # Exact counts, not "at least": a silently dropped row is the kind of
        # bug that only shows up as a wrong answer months later.
        assert one(db, f"SELECT count(*) AS n FROM {table}")["n"] == expected


class TestSnapshotTime:
    def test_as_of_is_read_from_the_readme_sheet(self, db):
        stored = one(db, "SELECT value FROM meta WHERE key = 'as_of'")["value"]
        assert datetime.fromisoformat(stored) == datetime(2026, 8, 16, 11, 0, tzinfo=IST)

    def test_as_of_is_recorded_as_a_sunday(self, db):
        stored = one(db, "SELECT value FROM meta WHERE key = 'as_of'")["value"]
        assert datetime.fromisoformat(stored).weekday() == 6

    def test_currency_is_captured(self, db):
        assert one(db, "SELECT value FROM meta WHERE key = 'currency'")["value"] == "INR"


class TestTimestamps:
    TIMESTAMP_COLUMNS: ClassVar[dict[str, list[str]]] = {
        "orders": [
            "booked_at",
            "pickup_window_start",
            "pickup_window_end",
            "pickup_actual_at",
            "cancellation_requested_at",
        ],
        "tickets": ["created_at", "last_customer_message_at"],
    }

    def test_every_stored_timestamp_carries_an_offset(self, db):
        # A naive timestamp resolves differently on every host, which would
        # shift every cancellation window silently.
        for table, columns in self.TIMESTAMP_COLUMNS.items():
            for column in columns:
                for row in rows(
                    db, f"SELECT {column} AS v FROM {table} WHERE {column} IS NOT NULL"
                ):
                    assert datetime.fromisoformat(row["v"]).tzinfo is not None, f"{table}.{column}"

    def test_workbook_local_time_is_interpreted_as_asia_kolkata(self, db):
        booked = one(db, "SELECT booked_at FROM orders WHERE order_id = 'ORD-1001'")["booked_at"]
        assert datetime.fromisoformat(booked) == datetime(2026, 8, 16, 9, 0, tzinfo=IST)


class TestMissingDataStaysMissing:
    """Absent is not zero, and not the epoch, and not an empty string."""

    def test_an_unpicked_order_has_a_null_pickup_time(self, db):
        # ORD-1001 is BOOKED and was never collected as far as the data says.
        assert (
            one(db, "SELECT pickup_actual_at AS v FROM orders WHERE order_id='ORD-1001'")["v"]
            is None
        )

    def test_an_order_with_no_cancellation_request_is_null(self, db):
        # ORD-2002 is the failed-pickup credit case; nobody asked to cancel it.
        assert (
            one(db, "SELECT cancellation_requested_at AS v FROM orders WHERE order_id='ORD-2002'")[
                "v"
            ]
            is None
        )

    def test_an_account_without_an_agreement_has_a_null_contract_file(self, db):
        # Beacon Retail has no agreement in the pack. Empty string would read
        # as "a contract named ''".
        assert (
            one(db, "SELECT contract_file AS v FROM accounts WHERE account_id='ACCT-003'")["v"]
            is None
        )

    def test_open_tickets_have_no_historical_resolution(self, db):
        assert (
            one(
                db,
                "SELECT count(*) AS n FROM tickets WHERE status='open' AND historical_resolution IS NOT NULL",
            )["n"]
            == 0
        )

    def test_the_two_closed_tickets_do_have_one(self, db):
        # Both are wrong, which is the point; they must survive to be caught.
        assert (
            one(db, "SELECT count(*) AS n FROM tickets WHERE historical_resolution IS NOT NULL")[
                "n"
            ]
            == 2
        )


class TestBooleans:
    def test_premium_support_round_trips(self, db):
        flags = {
            r["account_id"]: r["premium_support"]
            for r in rows(db, "SELECT account_id, premium_support FROM accounts")
        }
        assert flags == {"ACCT-001": 1, "ACCT-002": 0, "ACCT-003": 0, "ACCT-004": 0}

    def test_fault_flags_round_trip(self, db):
        # ORD-2002 is the only carrier-fault order, and it drives the whole
        # service-credit path.
        faults = {
            r["order_id"]: (r["carrier_fault"], r["customer_fault"])
            for r in rows(db, "SELECT order_id, carrier_fault, customer_fault FROM orders")
        }
        assert faults["ORD-2002"] == (1, 0)
        assert all(v == (0, 0) for k, v in faults.items() if k != "ORD-2002")


class TestGroundTruth:
    """Spot checks against docs/01_DATA_PACK_FINDINGS.md §10."""

    @pytest.mark.parametrize(
        "order_id,account_id,status,fee",
        [
            ("ORD-1001", "ACCT-001", "BOOKED", 4200.0),
            ("ORD-1002", "ACCT-001", "PICKED_UP", 5100.0),
            ("ORD-2001", "ACCT-002", "BOOKED", 1800.0),
            ("ORD-2002", "ACCT-002", "BOOKED", 2400.0),
            ("ORD-3001", "ACCT-003", "BOOKED", 1200.0),
            ("ORD-4001", "ACCT-004", "DELIVERED", 3600.0),
        ],
    )
    def test_order_facts(self, db, order_id, account_id, status, fee):
        row = one(db, "SELECT * FROM orders WHERE order_id = ?", order_id)
        assert (row["account_id"], row["status"], row["shipment_fee_inr"]) == (
            account_id,
            status,
            fee,
        )

    def test_the_discriminating_pair_differs_only_by_account(self, db):
        # ORD-1001 and ORD-2001 are both BOOKED and both past 30 minutes.
        # Everything that makes their answers differ lives in the agreements.
        pair = {
            r["order_id"]: r
            for r in rows(db, "SELECT * FROM orders WHERE order_id IN ('ORD-1001','ORD-2001')")
        }
        assert pair["ORD-1001"]["status"] == pair["ORD-2001"]["status"] == "BOOKED"
        assert pair["ORD-1001"]["account_id"] != pair["ORD-2001"]["account_id"]

    def test_agreements_are_linked_to_the_two_accounts_that_have_them(self, db):
        linked = {
            r["account_id"]: r["contract_file"]
            for r in rows(
                db, "SELECT account_id, contract_file FROM accounts WHERE contract_file IS NOT NULL"
            )
        }
        assert set(linked) == {"ACCT-001", "ACCT-002"}


class TestSchemaIntegrity:
    def test_foreign_keys_are_declared(self, db):
        assert one(db, "SELECT count(*) AS n FROM pragma_foreign_key_list('orders')")["n"] >= 1
        assert one(db, "SELECT count(*) AS n FROM pragma_foreign_key_list('tickets')")["n"] >= 1

    def test_no_orphan_rows(self, db):
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []

    @pytest.mark.parametrize(
        "index_on",
        ["orders.account_id", "tickets.account_id", "tickets.assigned_to", "tickets.status"],
    )
    def test_query_shapes_are_indexed(self, db, index_on):
        table, column = index_on.split(".")
        indexed = {
            r["name"]
            for idx in rows(db, f"SELECT name FROM pragma_index_list('{table}')")
            for r in rows(db, f"SELECT name FROM pragma_index_info('{idx['name']}')")
        }
        assert column in indexed


class TestIdempotence:
    def test_rebuilding_produces_identical_rows(self, tmp_path):
        # The database is committed, so a rebuild that reorders or renumbers
        # rows would produce a noisy diff and an unreproducible artifact.
        first, second = tmp_path / "a.db", tmp_path / "b.db"
        build_database(WORKBOOK_PATH, first)
        build_database(WORKBOOK_PATH, second)
        for table in ("accounts", "orders", "tickets", "meta"):
            a = sqlite3.connect(first).execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            b = sqlite3.connect(second).execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            assert a == b, table

    def test_rebuilding_over_an_existing_file_does_not_duplicate(self, tmp_path):
        path = tmp_path / "a.db"
        build_database(WORKBOOK_PATH, path)
        build_database(WORKBOOK_PATH, path)
        assert sqlite3.connect(path).execute("SELECT count(*) FROM orders").fetchone()[0] == 6
