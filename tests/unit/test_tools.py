"""What the tools do, once the projection has decided which ones exist.

The projection tests prove an unauthorised query is inexpressible. These prove
the expressible ones behave: a lookup mints a handle, a calculator refuses a
call that skipped its prerequisite, a denial says nothing about what it denied,
and Maya's queue is Maya's.

Everything runs against the committed database and the real domain layer. The
only stand-in is the retriever, and only in the classes that do not test it.
"""

from __future__ import annotations

import json

import pytest

from src.agent.tools.base import ToolDenied, ToolError, ToolResult
from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import build_toolset
from src.auth.personas import get_persona, to_principal
from src.domain.evidence import EvidenceKind


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture
def session():
    """A toolset plus its context, as one run would have."""
    opened = []

    def build(persona_id: str, **kwargs):
        cm = open_tool_context(persona(persona_id), **kwargs)
        context = cm.__enter__()
        opened.append(cm)
        return {t.name: t for t in build_toolset(context)}, context

    yield build
    for cm in opened:
        cm.__exit__(None, None, None)


def tools(session, persona_id: str, **kwargs):
    return session(persona_id, **kwargs)[0]


class TestLookupsMintHandles:
    def test_get_order_returns_a_snapshot_id(self, session):
        result = tools(session, "northstar_customer")["get_order"](order_id="ORD-1001")
        assert isinstance(result, ToolResult)
        assert result.data["snapshot_id"].startswith("snap_")

    def test_the_handle_resolves_to_the_row(self, session):
        toolset, context = session("northstar_customer")
        handle = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        payload = context.store.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)
        assert payload["order_id"] == "ORD-1001"
        assert payload["shipment_fee_inr"] == 4200.0

    def test_the_summary_is_a_summary_not_the_row(self, session):
        # The handle is what the calculators consume. A whole row in context is
        # a row the model can quote from without having computed anything.
        data = tools(session, "northstar_customer")["get_order"](order_id="ORD-1001").data
        assert "shipment_fee_inr" not in data
        assert data["status"] == "BOOKED"

    def test_a_missing_pickup_time_stays_null_in_the_snapshot(self, session):
        toolset, context = session("northstar_customer")
        handle = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        payload = context.store.read(handle, expect=EvidenceKind.ORDER_SNAPSHOT)
        assert payload["pickup_actual_at"] is None

    def test_get_account_mints_the_handle_the_sla_calculator_wants(self, session):
        data = tools(session, "lumenworks_customer")["get_account"]().data
        assert data["account_snapshot_id"].startswith("asnap_")
        assert data["plan"] == "Growth"

    def test_every_result_is_json_serialisable(self, session):
        toolset = tools(session, "maya_agent")
        for call in (
            lambda: toolset["get_order"](order_id="ORD-1001"),
            lambda: toolset["get_ticket"](ticket_id="TKT-501"),
            lambda: toolset["my_queue"](),
        ):
            assert json.loads(json.dumps(call().to_payload()))


class TestDenialsLeakNothing:
    def test_a_customer_cannot_read_another_accounts_order(self, session):
        # GS-026, at the tool boundary.
        result = tools(session, "lumenworks_customer")["get_order"](order_id="ORD-1001")
        assert isinstance(result, ToolDenied)

    def test_the_denial_names_nothing_about_the_order(self, session):
        result = tools(session, "lumenworks_customer")["get_order"](order_id="ORD-1001")
        message = json.dumps(result.to_payload())
        for leaked in ("ACCT-001", "Northstar", "BOOKED", "SwiftShip", "4200"):
            assert leaked not in message

    def test_a_foreign_order_and_a_nonexistent_one_are_indistinguishable(self, session):
        toolset = tools(session, "lumenworks_customer")
        foreign = toolset["get_order"](order_id="ORD-1001").message
        missing = toolset["get_order"](order_id="ORD-9999").message
        assert foreign.replace("ORD-1001", "X") == missing.replace("ORD-9999", "X")

    def test_no_handle_is_minted_for_a_denied_read(self, session):
        # A snapshot of a row the caller may not see would be a handle they
        # could pass to a calculator.
        toolset, context = session("lumenworks_customer")
        toolset["get_order"](order_id="ORD-1001")
        assert (
            context.store.connection.execute(
                "SELECT count(*) FROM evidence WHERE kind = 'order_snapshot'"
            ).fetchone()[0]
            == 0
        )

    @pytest.mark.parametrize(
        "persona_id", ["northstar_customer", "lumenworks_customer", "beacon_customer"]
    )
    def test_every_customer_is_denied_someone_elses_ticket(self, session, persona_id):
        foreign = {"northstar_customer": "TKT-451"}.get(persona_id, "TKT-504")
        assert isinstance(tools(session, persona_id)["get_ticket"](ticket_id=foreign), ToolDenied)

    def test_staff_read_across_accounts(self, session):
        toolset = tools(session, "maya_agent")
        for ticket_id in ("TKT-501", "TKT-502", "TKT-503", "TKT-504", "TKT-505"):
            assert isinstance(toolset["get_ticket"](ticket_id=ticket_id), ToolResult)

    def test_staff_naming_the_wrong_account_is_told_so(self, session):
        result = tools(session, "maya_agent")["get_order"](
            order_id="ORD-1001", account_id="ACCT-002"
        )
        assert isinstance(result, ToolDenied)


class TestQueues:
    def test_my_queue_is_mine(self, session):
        # GS-030. Maya's three, and not Rohit's four.
        data = tools(session, "maya_agent")["my_queue"]().data
        assert {t["ticket_id"] for t in data["tickets"]} == {"TKT-450", "TKT-502", "TKT-504"}

    def test_the_other_agent_gets_the_complement(self, session):
        data = tools(session, "rohit_agent")["my_queue"]().data
        assert {t["ticket_id"] for t in data["tickets"]} == {
            "TKT-451",
            "TKT-501",
            "TKT-503",
            "TKT-505",
        }

    def test_query_tickets_refuses_an_unknown_filter(self, session):
        result = tools(session, "maya_agent")["query_tickets"](severity="P1")
        assert isinstance(result, ToolError)
        assert "severity" in result.message

    def test_query_tickets_filters_by_status(self, session):
        data = tools(session, "priya_manager")["query_tickets"](status="open").data
        assert data["count"] == 5

    def test_a_result_is_capped(self, session):
        data = tools(session, "priya_manager")["query_tickets"](limit=10_000).data
        assert data["count"] <= 50


class TestTheChainCannotBeSkipped:
    def test_the_calculator_has_no_order_id_parameter_at_all(self, session):
        result = tools(session, "northstar_customer")["compute_cancellation_fee"](
            order_id="ORD-1001"
        )
        assert isinstance(result, ToolError)
        assert "order_id" in result.message

    def test_calling_without_a_resolution_names_the_tool_that_mints_one(self, session):
        toolset = tools(session, "northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        result = toolset["compute_cancellation_fee"](snapshot_id=snapshot)
        assert isinstance(result, ToolError)
        assert "resolution_id" in result.message

    def test_a_resolution_for_the_wrong_topic_is_refused_with_a_next_step(self, session):
        toolset = tools(session, "northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        wrong = toolset["resolve_policy"](topic="failed_pickup_credit").data["resolution_id"]
        result = toolset["compute_cancellation_fee"](snapshot_id=snapshot, resolution_id=wrong)
        assert isinstance(result, ToolError)
        assert "cancellation_fee" in result.message
        assert "resolve_policy" in result.message

    def test_the_full_chain_produces_the_signed_off_answer(self, session):
        toolset = tools(session, "northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        resolution = toolset["resolve_policy"](topic="cancellation_fee", snapshot_id=snapshot).data[
            "resolution_id"
        ]
        outcome = toolset["compute_cancellation_fee"](
            snapshot_id=snapshot, resolution_id=resolution
        ).data
        assert outcome["fee_inr"] == 0
        assert outcome["governing_clause"] == "northstar_logistics_enterprise_agreement::§2"

    def test_the_same_chain_for_another_account_gives_the_other_answer(self, session):
        toolset = tools(session, "lumenworks_customer")
        snapshot = toolset["get_order"](order_id="ORD-2001").data["snapshot_id"]
        resolution = toolset["resolve_policy"](topic="cancellation_fee").data["resolution_id"]
        outcome = toolset["compute_cancellation_fee"](
            snapshot_id=snapshot, resolution_id=resolution
        ).data
        assert outcome["fee_inr"] == 250

    def test_a_handle_from_another_session_is_refused(self, session):
        mine, _ = session("northstar_customer")
        theirs, _ = session("lumenworks_customer")
        snapshot = mine["get_order"](order_id="ORD-1001").data["snapshot_id"]
        result = theirs["compute_cancellation_fee"](snapshot_id=snapshot, resolution_id="res_x")
        assert isinstance(result, ToolError)


class TestResolvePolicy:
    def test_a_customer_resolves_against_their_own_account_implicitly(self, session):
        data = tools(session, "northstar_customer")["resolve_policy"](topic="cancellation_fee").data
        assert data["governing_clause"] == "northstar_logistics_enterprise_agreement::§2"
        assert data["is_override"] is True

    def test_an_agreement_that_declines_to_override_is_still_reported(self, session):
        data = tools(session, "lumenworks_customer")["resolve_policy"](
            topic="cancellation_fee"
        ).data
        assert data["governing_clause"] == "cancellation_and_service_credit_sop_v4::§1"
        assert "lumenworks_service_agreement::§2" in data["deferred"]

    def test_staff_must_say_whose_agreement_applies(self, session):
        result = tools(session, "maya_agent")["resolve_policy"](topic="cancellation_fee")
        assert isinstance(result, ToolError)
        assert "account_id" in result.message

    def test_staff_may_read_the_account_off_a_snapshot(self, session):
        toolset = tools(session, "maya_agent")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        data = toolset["resolve_policy"](topic="cancellation_fee", snapshot_id=snapshot).data
        assert data["governing_clause"] == "northstar_logistics_enterprise_agreement::§2"

    def test_staff_may_ask_about_no_account_in_particular(self, session):
        data = tools(session, "priya_manager")["resolve_policy"](
            topic="cancellation_fee", account_id="GENERAL"
        ).data
        assert data["governing_clause"] == "cancellation_and_service_credit_sop_v4::§1"

    def test_an_unknown_topic_lists_the_known_ones(self, session):
        result = tools(session, "northstar_customer")["resolve_policy"](topic="refunds")
        assert isinstance(result, ToolError)
        assert "cancellation_fee" in result.message

    def test_the_resolution_records_the_snapshot_as_provenance(self, session):
        toolset, context = session("northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        resolution = toolset["resolve_policy"](topic="cancellation_fee", snapshot_id=snapshot).data[
            "resolution_id"
        ]
        assert context.store.provenance(resolution) == (snapshot,)


class TestConsistencyTool:
    def test_it_reports_the_staleness_conflict(self, session):
        toolset = tools(session, "northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        data = toolset["check_data_consistency"](snapshot_id=snapshot).data
        assert data["blocking"] is True
        assert data["conflicts"][0]["conflict_class"] == "stale_status"

    def test_it_returns_a_report_handle(self, session):
        toolset, context = session("northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        report = toolset["check_data_consistency"](snapshot_id=snapshot).data["report_id"]
        assert context.store.kind_of(report) is EvidenceKind.CONSISTENCY_REPORT

    def test_an_unknown_topic_is_refused_rather_than_ignored(self, session):
        toolset = tools(session, "northstar_customer")
        snapshot = toolset["get_order"](order_id="ORD-1001").data["snapshot_id"]
        result = toolset["check_data_consistency"](snapshot_id=snapshot, topics=["refunds"])
        assert isinstance(result, ToolError)

    def test_a_snapshot_it_cannot_read_gets_a_next_step(self, session):
        result = tools(session, "northstar_customer")["check_data_consistency"](
            snapshot_id="snap_invented"
        )
        assert isinstance(result, ToolError)
        assert "get_order" in result.message
