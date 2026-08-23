"""The confirmation gate as three tools (ARCHITECTURE 13).

`prepare_action` proposes, a human confirms, `execute_action` performs. The
properties worth testing are the ones that would let something skip a step:

- the **token never enters model context**, because a model holding the token
  can confirm on the human's behalf;
- `prepare_action` **refuses `kind="approve_credit"`**, because otherwise the
  role projection that keeps `approve_credit` out of Maya's schema is bypassed
  by passing a string;
- a **blocking** consistency conflict refuses outright (D19), while an advisory
  one travels to the card so the human sees it;
- the payload that executes is the payload that was previewed, which is what
  the token is for.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from src.agent.tools.actions import approve_credit, execute_action, prepare_action
from src.agent.tools.base import ToolDenied, ToolError, ToolPending, ToolResult
from src.agent.tools.context import open_tool_context
from src.auth.personas import get_persona, to_principal
from src.clock import wall_now
from src.config import get_settings
from src.datastore.runtime import ActionKind, open_runtime_store
from src.domain.action_tokens import PendingAction, mint_token
from src.domain.evidence import EvidenceKind

SECRET = "gate-secret"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture
def runtime(tmp_path):
    with open_runtime_store(tmp_path / "runtime.db") as store:
        yield store


@pytest.fixture
def context_for(runtime):
    """A tool context bound to one persona, with the gate wired up."""
    opened = []

    def factory(persona_id: str, thread_id: str = "thread-1"):
        manager = open_tool_context(
            persona(persona_id),
            db_path=get_settings().db_path,
            evidence_connection=sqlite3.connect(":memory:"),
            runtime=runtime,
            session_id="sid_test",
            thread_id=thread_id,
            action_secret=SECRET,
        )
        context = manager.__enter__()
        opened.append(manager)
        return context

    yield factory
    for manager in opened:
        manager.__exit__(None, None, None)


def propose(context, **overrides):
    arguments = {
        "kind": "create_escalation",
        "payload": {"question": "how do I change the billing contact?"},
        "evidence_ids": [],
    }
    arguments.update(overrides)
    return prepare_action(context)(**arguments)


class TestPreparingAnAction:
    def test_a_proposal_comes_back_pending(self, context_for):
        outcome = propose(context_for("maya_agent"))
        assert isinstance(outcome, ToolPending)
        assert outcome.pending.kind is ActionKind.CREATE_ESCALATION

    def test_the_preview_describes_what_will_happen(self, context_for):
        outcome = propose(context_for("maya_agent"))
        preview = outcome.pending.to_preview()
        assert preview["kind"] == "create_escalation"
        assert preview["payload"]["question"].startswith("how do I change")

    def test_an_unknown_kind_is_refused(self, context_for):
        outcome = propose(context_for("maya_agent"), kind="delete_everything")
        assert isinstance(outcome, ToolError)
        assert "delete_everything" in outcome.message

    def test_the_evidence_chain_is_carried_onto_the_proposal(self, context_for):
        outcome = propose(context_for("maya_agent"), evidence_ids=["ev_1", "ev_2"])
        assert outcome.pending.evidence_chain == ("ev_1", "ev_2")

    def test_the_proposal_is_bound_to_this_session_and_thread(self, context_for):
        outcome = propose(context_for("maya_agent", thread_id="thread-9"))
        assert outcome.pending.session_id == "sid_test"
        assert outcome.pending.thread_id == "thread-9"


class TestTheTokenNeverReachesTheModel:
    """The reason the gate is a gate.

    A model that can read the token can call `execute_action` with it and the
    human is out of the loop. The token goes to the client on the interrupt
    event; what goes back into the conversation is the preview and the fact
    that something is waiting.
    """

    def test_the_model_visible_payload_does_not_contain_the_token(self, context_for):
        outcome = propose(context_for("maya_agent"))
        rendered = str(outcome.to_payload())
        assert outcome.token
        assert outcome.token not in rendered

    def test_the_model_visible_payload_does_not_contain_the_nonce(self, context_for):
        # The nonce is half of the token's identity and the single-use key.
        outcome = propose(context_for("maya_agent"))
        assert outcome.pending.nonce not in str(outcome.to_payload())

    def test_the_model_is_told_a_human_must_confirm(self, context_for):
        payload = propose(context_for("maya_agent")).to_payload()
        assert payload.get("awaiting_confirmation") is True


class TestKindCannotRouteAroundTheProjection:
    """`approve_credit` is a separate tool so Maya's schema does not contain it.

    That containment is worth nothing if `prepare_action(kind="approve_credit")`
    works, because the kind is a string the model writes.
    """

    def test_prepare_action_refuses_the_credit_kind_for_an_agent(self, context_for):
        outcome = propose(context_for("maya_agent"), kind="approve_credit")
        assert isinstance(outcome, ToolError | ToolDenied)

    def test_prepare_action_refuses_the_credit_kind_even_for_a_manager(self, context_for):
        # Not a permission check - a routing one. Managers approve credit
        # through `approve_credit`, which applies the SOP v4 threshold. Letting
        # the generic path through would skip it.
        outcome = propose(context_for("priya_manager"), kind="approve_credit")
        assert isinstance(outcome, ToolError | ToolDenied)

    def test_a_customer_can_still_propose_an_ordinary_action(self, context_for):
        assert isinstance(propose(context_for("northstar_customer")), ToolPending)


class TestConsistencyGatesTheProposal:
    """D19: a blocking conflict refuses; an advisory one is shown."""

    def test_a_blocking_conflict_refuses_the_proposal(self, context_for, monkeypatch):
        context = context_for("maya_agent")
        handle = _snapshot(context)
        _force_conflicts(monkeypatch, blocking=True)
        outcome = propose(context, evidence_ids=[str(handle)])
        assert isinstance(outcome, ToolError)
        assert "disagree" in outcome.message.lower()

    def test_a_blocking_conflict_means_no_token_was_minted(self, context_for, monkeypatch):
        # A token is a promise the description is safe to execute. Refusing but
        # handing one over anyway would leave the promise outstanding.
        context = context_for("maya_agent")
        handle = _snapshot(context)
        _force_conflicts(monkeypatch, blocking=True)
        assert not isinstance(propose(context, evidence_ids=[str(handle)]), ToolPending)

    def test_an_advisory_conflict_travels_to_the_card(self, context_for, monkeypatch):
        context = context_for("maya_agent")
        handle = _snapshot(context)
        _force_conflicts(monkeypatch, blocking=False)
        outcome = propose(context, evidence_ids=[str(handle)])
        assert isinstance(outcome, ToolPending)
        assert outcome.pending.advisories

    def test_no_conflict_means_no_advisories(self, context_for):
        outcome = propose(context_for("maya_agent"))
        assert outcome.pending.advisories == ()


class TestApproveCredit:
    def test_a_manager_can_propose_a_credit(self, context_for):
        outcome = approve_credit(context_for("priya_manager"))(
            account_id="ACCT-001", amount_inr=1500, reason="SLA miss", evidence_ids=["ev_1"]
        )
        assert isinstance(outcome, ToolPending)
        assert outcome.pending.kind is ActionKind.APPROVE_CREDIT

    def test_the_tool_declares_the_manager_scope(self, context_for):
        assert approve_credit(context_for("priya_manager")).requires_scope == "write:approve_credit"

    def test_a_non_positive_amount_is_refused(self, context_for):
        outcome = approve_credit(context_for("priya_manager"))(
            account_id="ACCT-001", amount_inr=0, reason="x", evidence_ids=[]
        )
        assert isinstance(outcome, ToolError)

    def test_the_amount_reaches_the_payload_unrounded(self, context_for):
        outcome = approve_credit(context_for("priya_manager"))(
            account_id="ACCT-001", amount_inr=1250, reason="SLA miss", evidence_ids=[]
        )
        assert outcome.pending.payload["amount_inr"] == 1250


class TestExecuting:
    def test_a_confirmed_action_is_appended_to_the_log(self, context_for, runtime):
        context = context_for("maya_agent")
        proposal = propose(context)
        outcome = execute_action(context)(token=proposal.token, pending=proposal.pending)
        assert isinstance(outcome, ToolResult)
        recorded = runtime.actions_for_thread("thread-1")
        assert len(recorded) == 1
        assert recorded[0].kind is ActionKind.CREATE_ESCALATION

    def test_the_executed_row_carries_the_evidence_chain(self, context_for, runtime):
        context = context_for("maya_agent")
        proposal = propose(context, evidence_ids=["ev_1", "ev_2"])
        execute_action(context)(token=proposal.token, pending=proposal.pending)
        assert runtime.actions_for_thread("thread-1")[0].evidence_chain == ("ev_1", "ev_2")

    def test_the_same_confirmation_cannot_execute_twice(self, context_for, runtime):
        context = context_for("maya_agent")
        proposal = propose(context)
        execute_action(context)(token=proposal.token, pending=proposal.pending)
        again = execute_action(context)(token=proposal.token, pending=proposal.pending)
        assert isinstance(again, ToolError)
        assert len(runtime.actions_for_thread("thread-1")) == 1

    def test_a_payload_edited_after_the_preview_is_refused(self, context_for, runtime):
        # The whole reason the token exists. The human confirmed an escalation
        # about a billing contact, not a credit for fifty thousand rupees.
        context = context_for("maya_agent")
        proposal = propose(context)
        tampered = PendingAction(
            nonce=proposal.pending.nonce,
            kind=proposal.pending.kind,
            payload={"amount_inr": 50000},
            evidence_chain=proposal.pending.evidence_chain,
            session_id=proposal.pending.session_id,
            thread_id=proposal.pending.thread_id,
            expires_at=proposal.pending.expires_at,
        )
        outcome = execute_action(context)(token=proposal.token, pending=tampered)
        assert isinstance(outcome, ToolError)
        assert runtime.actions_for_thread("thread-1") == ()

    def test_an_expired_confirmation_is_refused(self, context_for, runtime):
        context = context_for("maya_agent")
        proposal = propose(context)
        stale = PendingAction(
            nonce=proposal.pending.nonce,
            kind=proposal.pending.kind,
            payload=proposal.pending.payload,
            evidence_chain=proposal.pending.evidence_chain,
            session_id=proposal.pending.session_id,
            thread_id=proposal.pending.thread_id,
            expires_at=wall_now() - timedelta(seconds=1),
        )
        outcome = execute_action(context)(token=mint_token(stale, secret=SECRET), pending=stale)
        assert isinstance(outcome, ToolError)
        assert "expired" in outcome.message.lower()
        assert runtime.actions_for_thread("thread-1") == ()

    def test_a_token_from_another_session_is_refused(self, context_for, runtime):
        context = context_for("maya_agent")
        proposal = propose(context)
        foreign = PendingAction(
            nonce=proposal.pending.nonce,
            kind=proposal.pending.kind,
            payload=proposal.pending.payload,
            evidence_chain=proposal.pending.evidence_chain,
            session_id="sid_someone_else",
            thread_id=proposal.pending.thread_id,
            expires_at=proposal.pending.expires_at,
        )
        outcome = execute_action(context)(token=proposal.token, pending=foreign)
        assert isinstance(outcome, ToolError)
        assert runtime.actions_for_thread("thread-1") == ()

    def test_executing_without_a_pending_action_is_refused(self, context_for):
        # Nothing was prepared, so there is nothing a token could authorise.
        outcome = execute_action(context_for("maya_agent"))(token="anything", pending=None)
        assert isinstance(outcome, ToolError)


def _snapshot(context) -> object:
    """A real ticket-snapshot handle in this context's evidence store.

    Real rather than a stand-in string, because `prepare_action` decides what
    to check by asking the store what kind each handle is - a fake id would
    never reach the checker and the test would pass without exercising it.
    """
    ticket = context.repository.get_ticket("TKT-504")
    return context.store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())


def _force_conflicts(monkeypatch, *, blocking: bool):
    """Make the consistency check report a conflict of the given severity."""
    from src.domain import consistency as consistency_module

    severity = (
        consistency_module.ConflictSeverity.BLOCKING
        if blocking
        else consistency_module.ConflictSeverity.ADVISORY
    )
    conflict = consistency_module.Conflict(
        conflict_class=consistency_module.ConflictClass.STALE_STATUS,
        severity=severity,
        detail="carrier says delivered; order says in transit",
        sources=("ORD-1001",),
        confidence=0.9,
    )

    def fake_check(self, *, snapshot_id=None, topics=(), **_kwargs):
        return consistency_module.ConsistencyReport(
            report_id=None,
            subject_id="ORD-1001",
            subject_kind="order",
            conflicts=(conflict,),
            checked=(consistency_module.ConflictClass.STALE_STATUS,),
        )

    monkeypatch.setattr(consistency_module.ConsistencyChecker, "check", fake_check)
