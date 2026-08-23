"""Words a reader can use, in place of the names a database uses.

The load-bearing test is the exhaustiveness one. `check_data_consistency` is a
precise name and a terrible thing to show somebody asking about a parcel, so
every tool in the projection matrix needs a phrase - and a tool added without
one would otherwise appear under its own name the first time a customer
triggered it, which is the sort of thing nobody notices until a demo.

`unresolved_conflict` reached the page verbatim before this module existed.
That is the same failure as printing a snake_case column into a UI, and the
maps here are the fix.
"""

from __future__ import annotations

import pytest

from src.agent.tools.registry import PROJECTION
from ui.labels import (
    DENIAL_REASONS,
    ESCALATION_REASONS,
    PHASE_PHRASES,
    TOOL_PHRASES,
    denial_reason,
    describe,
    elapsed_label,
    escalation_reason,
    tool_phrase,
)


class TestEveryToolHasWords:
    def test_the_map_covers_the_projection_matrix(self):
        missing = sorted(set(PROJECTION) - set(TOOL_PHRASES))
        assert missing == [], f"tools with no reader-facing phrase: {missing}"

    def test_no_phrase_is_a_tool_name(self):
        # A phrase that is just the identifier with spaces has not translated
        # anything. Every one of these must read as something a person said.
        for name, phrase in TOOL_PHRASES.items():
            assert phrase.lower() != name.replace("_", " ")

    def test_no_phrase_leaks_snake_case(self):
        for phrase in [*TOOL_PHRASES.values(), *PHASE_PHRASES.values()]:
            assert "_" not in phrase

    def test_an_unmapped_tool_gets_nothing_rather_than_its_name(self):
        # Silence beats showing `scan_support_health` to a customer. The test
        # above is what stops the case arising; this is what happens if it does.
        assert tool_phrase("some_future_tool") is None


class TestPhrasesNameTheirSubject:
    def test_an_order_lookup_names_the_order(self):
        assert tool_phrase("get_order", {"order_id": "ORD-1001"}) == "Looking up the order ORD-1001"

    def test_a_resolution_names_the_topic_without_the_underscores(self):
        phrase = tool_phrase("resolve_policy", {"topic": "cancellation_fee"})
        assert phrase == 'Working out which policy applies "cancellation fee"'

    def test_a_tool_with_no_subject_key_is_left_alone(self):
        assert tool_phrase("check_data_consistency", {"snapshot_id": "ev_1"}) == (
            "Cross-checking the records"
        )

    def test_a_missing_argument_does_not_produce_a_dangling_phrase(self):
        assert tool_phrase("get_order", {}) == "Looking up the order"

    def test_a_long_subject_is_trimmed(self):
        phrase = tool_phrase("search_policy", {"query": "x" * 200})
        assert phrase is not None
        assert len(phrase) < 120

    def test_a_non_scalar_argument_is_ignored(self):
        assert tool_phrase("get_order", {"order_id": {"nested": True}}) == "Looking up the order"

    def test_an_evidence_handle_is_never_shown_as_a_subject(self):
        # Handles are internal plumbing. They belong in the trace panel.
        assert "ev_" not in (tool_phrase("compute_cancellation_fee", {"snapshot_id": "ev_9"}) or "")


class TestDescribingEvents:
    def test_a_tool_event_describes_the_tool(self):
        said = describe(
            "tool.started", {"name": "get_ticket", "args_public": {"ticket_id": "TKT-1"}}
        )
        assert said == "Reading the ticket TKT-1"

    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            ("model.step", "Thinking"),
            ("facts.block", "Composing the answer"),
            ("run.completed", "Done"),
        ],
    )
    def test_a_phase_event_has_its_own_line(self, event, expected):
        assert describe(event) == expected

    def test_an_event_with_nothing_to_say_returns_none(self):
        # None leaves the previous line standing. Replacing "Looking up order
        # ORD-1001" with a filler string is a downgrade the reader notices.
        assert describe("tool.error", {"name": "search_policy"}) is None
        assert describe("some.unknown.event") is None

    def test_a_denial_is_described_without_naming_the_record(self):
        said = describe("tool.denied", {"name": "get_order", "reason": "out_of_scope"})
        assert said is not None
        assert "ORD" not in said


class TestReasonsAreSentences:
    def test_every_escalation_reason_is_covered(self):
        from src.agent.escalation import DeclineReason

        missing = sorted({r.value for r in DeclineReason} - set(ESCALATION_REASONS))
        assert missing == []

    def test_every_denial_reason_is_covered(self):
        from src.agent.tools.base import DenialReason

        missing = sorted({r.value for r in DenialReason} - set(DENIAL_REASONS))
        assert missing == []

    def test_no_reason_is_rendered_as_its_code(self):
        for code, wording in ESCALATION_REASONS.items():
            assert code not in wording
            assert "_" not in wording

    def test_an_unknown_code_still_reads_as_a_sentence(self):
        assert "_" not in escalation_reason("something_new")
        assert "_" not in denial_reason("something_new")

    def test_a_missing_code_does_not_render_none(self):
        assert "None" not in escalation_reason(None)
        assert "None" not in denial_reason(None)


class TestActionsAndPayloads:
    def test_every_action_kind_has_words(self):
        from src.datastore.runtime import ActionKind
        from ui.labels import ACTION_KINDS

        missing = sorted({k.value for k in ActionKind} - set(ACTION_KINDS))
        assert missing == []

    def test_an_action_reads_as_something_that_happened(self):
        from ui.labels import action_kind

        assert action_kind("create_escalation") == "Escalation raised"

    def test_an_unknown_kind_does_not_render_its_slug(self):
        from ui.labels import action_kind

        assert "_" not in action_kind("some_new_kind")

    def test_payload_keys_become_headings(self):
        from ui.labels import payload_label

        assert payload_label("unresolved") == "What is unresolved"

    def test_an_unmapped_payload_key_is_still_readable(self):
        # A payload the model composed can carry any key. None may reach the
        # page as snake_case - this is the last screen before a change is
        # authorised.
        from ui.labels import payload_label

        assert payload_label("some_new_field") == "Some new field"


class TestElapsed:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.4, ""), (1, "1s"), (45, "45s"), (60, "1m 00s"), (135, "2m 15s")],
    )
    def test_durations_read_naturally(self, seconds, expected):
        assert elapsed_label(seconds) == expected

    def test_a_sub_second_run_shows_nothing_rather_than_zero(self):
        # "0s" invites the reader to wonder what went wrong.
        assert elapsed_label(0.2) == ""
