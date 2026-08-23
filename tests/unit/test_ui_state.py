"""The event reducer, which is where the client's logic really lives.

Streamlit is a bad place to test anything, so the display decisions are made by
a pure function and asserted here: does the fact block arrive before the prose,
does a denial survive to be shown, is the card up only while a confirmation is
pending, does the badge fire on an override and not only on a conflict.

The property that makes reattaching work is here too. A run replayed from
`?from_seq=0` must fold to the same value as one watched live, and a replay
that overlaps a live stream must not drag the reattach point backwards.
"""

from __future__ import annotations

import pytest

from ui.state import AWAITING, COMPLETED, FAILED, RunView, ToolStep, apply, fold

RUN = "run-1"


def events(*specs):
    """Number a list of `(event, data)` pairs the way the server would."""
    return [(index + 1, name, data) for index, (name, data) in enumerate(specs)]


ANSWERED = events(
    ("run.started", {"run_id": RUN, "thread_id": "t1"}),
    ("model.step", {"tool_calls": 1}),
    ("tool.started", {"call_id": "c1", "name": "get_order", "args_public": {"order_id": "ORD-1"}}),
    ("tool.finished", {"call_id": "c1", "name": "get_order", "evidence_id": "ev_1"}),
    ("facts.block", {"rows": [{"label": "Verdict", "value": "No fee"}]}),
    ("token.delta", {"text": "No cancellation "}),
    ("token.delta", {"text": "fee applies."}),
    ("grounding.checked", {"verdict": "passed", "claims_total": 2, "unsupported": []}),
    ("run.completed", {"run_id": RUN, "citations": ["northstar::§2"]}),
)


class TestFoldingAWholeRun:
    def test_the_prose_is_the_deltas_in_order(self):
        assert fold(ANSWERED).prose == "No cancellation fee applies."

    def test_the_status_ends_completed(self):
        assert fold(ANSWERED).status == COMPLETED

    def test_the_citations_come_from_the_completion(self):
        assert fold(ANSWERED).citations == ("northstar::§2",)

    def test_the_last_sequence_is_the_servers_number(self):
        # Not a count of events folded. A replay starting mid-run would make
        # those differ, and the reattach point has to be the server's.
        assert fold(ANSWERED).last_seq == len(ANSWERED)

    def test_a_tool_call_is_recorded_with_its_arguments_and_handle(self):
        (step,) = fold(ANSWERED).steps
        assert step.name == "get_order"
        assert step.arguments == {"order_id": "ORD-1"}
        assert step.evidence_id == "ev_1"
        assert step.outcome == "ok"


class TestTheFactBlockPrecedesTheProse:
    def test_facts_are_present_before_the_first_delta_is_folded(self):
        """The ordering ARCHITECTURE 16 requires, checked where it shows.

        The server emits the block whole before any prose. If the client only
        had both by the end, a viewer would watch sentences appear against an
        empty card, which is the opposite of the intended reading order.
        """
        upto_facts = fold(ANSWERED[:5])
        assert upto_facts.facts is not None
        assert upto_facts.prose == ""

    def test_the_block_survives_the_rest_of_the_run(self):
        assert fold(ANSWERED).facts == {"rows": [{"label": "Verdict", "value": "No fee"}]}


class TestReplayEqualsLive:
    def test_replaying_from_zero_gives_the_same_view(self):
        live = fold(ANSWERED)
        replayed = fold(ANSWERED, RunView())
        assert live == replayed

    def test_a_reattach_continues_from_the_last_sequence(self):
        first = fold(ANSWERED[:4])
        whole = fold(ANSWERED[4:], first)
        assert whole == fold(ANSWERED)

    def test_an_overlapping_replay_does_not_move_the_point_backwards(self):
        # A live stream and a backlog can overlap by design. If `last_seq`
        # took the newest value rather than the highest, the next reattach
        # would ask for events the client already has.
        view = fold(ANSWERED)
        rewound = apply(view, 2, "model.step", {})
        assert rewound.last_seq == view.last_seq


class TestDenialsAreVisible:
    def test_a_denial_is_kept_and_reported(self):
        view = fold(
            events(
                ("tool.started", {"call_id": "c1", "name": "query_tickets"}),
                (
                    "tool.denied",
                    {"call_id": "c1", "name": "query_tickets", "reason": "out_of_scope"},
                ),
            )
        )
        (denial,) = view.denials
        assert denial.name == "query_tickets"
        assert denial.detail == "out_of_scope"

    def test_a_successful_call_is_not_a_denial(self):
        assert fold(ANSWERED).denials == ()

    def test_an_error_is_distinct_from_a_denial(self):
        # "Your role cannot" and "the tool broke" are different facts, and the
        # first is a demo asset while the second is a bug report.
        view = fold(
            events(
                ("tool.started", {"call_id": "c1", "name": "search_policy"}),
                ("tool.error", {"call_id": "c1", "name": "search_policy", "error": "unavailable"}),
            )
        )
        assert view.denials == ()
        assert view.steps[0].outcome == "error"


class TestTheConflictBadge:
    def test_a_conflict_lights_it(self):
        view = fold(events(("conflict.detected", {"severity": "blocking", "detail": "x"})))
        assert view.has_conflict

    def test_an_override_lights_it_too(self):
        # The single most legible demonstration of precedence. A badge that
        # only fires on conflicts would leave the override invisible.
        view = fold(
            events(("policy.resolved", {"governing": "northstar::§2", "overridden": ["sop::§1"]}))
        )
        assert view.has_conflict

    def test_a_resolution_with_nothing_overridden_does_not(self):
        view = fold(events(("policy.resolved", {"governing": "sop::§1", "overridden": []})))
        assert not view.has_conflict

    def test_a_plain_run_does_not(self):
        assert not fold(ANSWERED).has_conflict


class TestTheConfirmationCard:
    PROPOSAL = events(
        ("run.started", {"run_id": RUN}),
        ("tool.started", {"call_id": "c1", "name": "prepare_action"}),
        (
            "interrupt.await_confirm",
            {
                "preview": {"kind": "create_escalation", "payload": {"q": "?"}, "advisories": []},
                "token": "nonce.sig",
            },
        ),
    )

    def test_a_pending_action_raises_the_card(self):
        view = fold(self.PROPOSAL)
        assert view.awaiting_confirmation
        assert view.pending["kind"] == "create_escalation"
        assert view.status == AWAITING

    def test_the_token_is_held_for_the_resume_call(self):
        assert fold(self.PROPOSAL).confirm_token == "nonce.sig"

    def test_completing_the_run_takes_the_card_down(self):
        # A finished run cannot still be offering Confirm: the graph is no
        # longer parked, and the button would post a token for nothing.
        view = fold([*self.PROPOSAL, (4, "run.completed", {"run_id": RUN})])
        assert not view.awaiting_confirmation
        assert view.confirm_token is None

    def test_a_failed_run_takes_the_card_down_too(self):
        view = fold([*self.PROPOSAL, (4, "run.failed", {"error": "boom"})])
        assert not view.awaiting_confirmation
        assert view.status == FAILED
        assert view.error == "boom"


class TestPartialStreams:
    def test_a_finish_without_its_start_is_still_recorded(self):
        """A reattach can begin after the `tool.started` it needs.

        Dropping the orphan would make the trace shorter than the run it
        describes, which is worse than a step with no arguments.
        """
        view = fold(
            [(7, "tool.finished", {"call_id": "c9", "name": "resolve_policy", "evidence_id": "ev"})]
        )
        (step,) = view.steps
        assert step.name == "resolve_policy"
        assert step.outcome == "ok"

    def test_a_second_call_to_the_same_tool_gets_its_own_step(self):
        view = fold(
            events(
                ("tool.started", {"call_id": "c1", "name": "resolve_policy"}),
                ("tool.started", {"call_id": "c2", "name": "resolve_policy"}),
                ("tool.finished", {"call_id": "c1", "name": "resolve_policy"}),
            )
        )
        assert [s.outcome for s in view.steps] == ["ok", "running"]

    def test_an_unknown_event_is_ignored_rather_than_fatal(self):
        # The server may grow an event before the client learns to show it.
        view = apply(RunView(), 3, "ops.finding", {"whatever": True})
        assert view.last_seq == 3

    def test_a_run_in_flight_is_neither_finished_nor_awaiting(self):
        view = fold(ANSWERED[:4])
        assert not view.is_finished
        assert not view.awaiting_confirmation


class TestImmutability:
    def test_apply_returns_a_new_view(self):
        before = RunView()
        after = apply(before, 1, "token.delta", {"text": "hi"})
        assert before.prose == ""
        assert after.prose == "hi"

    def test_a_view_is_frozen(self):
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            RunView().prose = "edited"  # type: ignore[misc]

    def test_a_step_is_frozen(self):
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            ToolStep(call_id="c", name="n").outcome = "ok"  # type: ignore[misc]
