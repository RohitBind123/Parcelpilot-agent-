"""Tool primitives: the shape a model sees, and the shape a failure takes.

Two decisions are being tested here rather than assumed.

**A tool failure is a value, not an exception.** The model is the caller. It
cannot catch anything, and a traceback in its context window is noise it will
either ignore or hallucinate around. A returned `ToolError` naming the missing
prerequisite is something it can read and act on - which is what turns "the
brief requires multi-step" from a hope into a mechanic: call the calculator
without a resolution and the error tells you to resolve first.

**A denial says nothing about what was denied.** The refusal text goes into
model context, and model context is what a summariser reads and a prompt
injection reaches. A denial that quotes the order it is refusing has not
refused it.
"""

from __future__ import annotations

import json

import pytest

from src.agent.tools.base import (
    DenialReason,
    Param,
    Tool,
    ToolDenied,
    ToolError,
    ToolResult,
)


def echo(**kwargs):
    return ToolResult(data=dict(kwargs))


@pytest.fixture
def tool() -> Tool:
    return Tool(
        name="get_order",
        description="Look up an order on your account.",
        params=(
            Param("order_id", "string", "The order identifier, e.g. ORD-1001."),
            Param("verbose", "boolean", "Include notes.", required=False),
        ),
        run=echo,
    )


class TestTheSchemaAModelSees:
    def test_it_renders_as_an_openai_function(self, tool):
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_order"
        assert schema["function"]["description"].startswith("Look up an order")

    def test_parameters_carry_types_and_descriptions(self, tool):
        properties = tool.to_schema()["function"]["parameters"]["properties"]
        assert properties["order_id"]["type"] == "string"
        assert "ORD-1001" in properties["order_id"]["description"]

    def test_only_required_parameters_are_required(self, tool):
        assert tool.to_schema()["function"]["parameters"]["required"] == ["order_id"]

    def test_a_tool_with_no_parameters_still_renders_validly(self):
        # `my_queue` takes nothing. An empty properties object is valid JSON
        # Schema; omitting the key entirely is what some providers reject.
        bare = Tool(name="my_queue", description="Your tickets.", params=(), run=echo)
        parameters = bare.to_schema()["function"]["parameters"]
        assert parameters["properties"] == {}
        assert parameters["required"] == []

    def test_the_schema_is_json_serialisable(self, tool):
        assert json.loads(json.dumps(tool.to_schema()))


class TestCalling:
    def test_a_call_returns_a_result(self, tool):
        assert tool(order_id="ORD-1001").data == {"order_id": "ORD-1001"}

    def test_an_unknown_argument_is_refused_rather_than_dropped(self, tool):
        # Silently ignoring one runs a different query from the one the model
        # asked for, which on an ACL-adjacent read is the wrong way to fail.
        result = tool(order_id="ORD-1001", account_id="ACCT-002")
        assert isinstance(result, ToolError)
        assert "account_id" in result.message

    def test_a_missing_required_argument_names_it(self, tool):
        result = tool()
        assert isinstance(result, ToolError)
        assert "order_id" in result.message

    def test_an_optional_argument_may_be_omitted(self, tool):
        assert isinstance(tool(order_id="ORD-1001"), ToolResult)


class TestErrorsAreValuesNotExceptions:
    def test_a_tool_error_is_returned_not_raised(self, tool):
        # The whole point. The model is the caller; it cannot catch anything.
        assert isinstance(tool(), ToolError)

    def test_an_error_names_the_prerequisite_it_wants(self):
        error = ToolError.missing_prerequisite(
            "resolution_id", produced_by="resolve_policy", topic="cancellation_fee"
        )
        assert "resolution_id" in error.message
        assert "resolve_policy" in error.message
        assert error.recoverable is True

    def test_an_error_serialises_for_model_context(self):
        error = ToolError.missing_prerequisite("resolution_id", produced_by="resolve_policy")
        payload = error.to_payload()
        assert payload["error"] is True
        assert json.loads(json.dumps(payload))

    def test_an_unexpected_failure_inside_a_tool_becomes_an_error_value(self):
        def explodes(**kwargs):
            raise RuntimeError("the database fell over")

        boom = Tool(name="boom", description="", params=(), run=explodes)
        result = boom()
        assert isinstance(result, ToolError)
        # Not recoverable by the model retrying differently - it should say so
        # rather than invite a loop.
        assert result.recoverable is False

    def test_an_internal_failure_does_not_put_its_traceback_in_context(self):
        def explodes(**kwargs):
            raise RuntimeError("connection string postgres://user:hunter2@host/db")

        boom = Tool(name="boom", description="", params=(), run=explodes)
        assert "hunter2" not in boom().message


class TestDenialsSayNothingAboutWhatWasDenied:
    def test_a_denial_carries_a_reason_code(self):
        denial = ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-1001")
        assert denial.reason is DenialReason.OUT_OF_SCOPE

    def test_a_denial_names_neither_the_owner_nor_the_content(self):
        # GS-026: LumenWorks asking about ORD-1001 must learn nothing about it.
        denial = ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-1001")
        message = denial.message
        for leaked in ("ACCT-001", "Northstar", "BOOKED", "SwiftShip"):
            assert leaked not in message

    def test_a_denial_may_name_the_identifier_the_caller_already_typed(self):
        # It came from them. Withholding it makes the refusal unreadable
        # without protecting anything.
        assert "ORD-1001" in ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-1001").message

    def test_a_scope_denial_does_not_reveal_whether_the_record_exists(self):
        # "No such order" and "not yours" must be indistinguishable, or the
        # tool is an existence oracle for other accounts' identifiers.
        missing = ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-9999").message
        foreign = ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-1001").message
        assert missing.replace("ORD-9999", "X") == foreign.replace("ORD-1001", "X")

    def test_an_insufficient_scope_denial_is_a_different_reason(self):
        # GS-029: Maya asking to approve a credit is not an ACL breach, it is a
        # role that lacks the scope, and the answer should say which.
        denial = ToolDenied(DenialReason.INSUFFICIENT_SCOPE, "credit approval", None)
        assert denial.reason is DenialReason.INSUFFICIENT_SCOPE

    def test_a_denial_serialises_with_its_reason(self):
        payload = ToolDenied(DenialReason.OUT_OF_SCOPE, "order", "ORD-1001").to_payload()
        assert payload["denied"] is True
        assert payload["reason"] == "out_of_scope"
        assert json.loads(json.dumps(payload))
