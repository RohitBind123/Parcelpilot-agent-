"""Escalation as a record (D27).

"A human will follow up" is not an outcome: nothing was created, nobody was
told, and there is no artifact anyone can act on. A decline drafts a record
naming the specific gap, and the naming is the part that matters. "No source
found" is a shrug; "no clause in the corpus covers how to change a billing
contact" is a sentence someone can act on and a claim about the corpus that can
be checked.
"""

from __future__ import annotations

import json

import pytest

from src.agent.escalation import DeclineReason, Escalation, draft
from src.auth.personas import get_persona, to_principal


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture
def record() -> Escalation:
    return draft(
        principal=persona("beacon_customer"),
        thread_id="t1",
        question="How do we change the billing contact on our account?",
        reason=DeclineReason.NO_CITABLE_SOURCE,
        subject="how to change a billing contact",
        evidence_chain=("acct_abc", "res_def"),
        sources_consulted=("product_operations_guide_and_known_issues::§1",),
    )


class TestTheGapIsNamedSpecifically:
    def test_it_says_what_is_missing_not_that_something_is(self, record):
        assert "billing contact" in record.what_is_unresolved
        assert record.what_is_unresolved != "no source found"

    def test_it_names_the_corpus_as_the_thing_that_lacks_it(self, record):
        # A checkable claim: a reviewer can open the six documents and confirm.
        assert "corpus" in record.what_is_unresolved

    def test_the_question_is_carried_verbatim(self, record):
        # Paraphrasing it loses the thing the customer actually asked, which is
        # what the human picking this up needs.
        assert record.question == "How do we change the billing contact on our account?"

    def test_the_subject_is_in_the_words_of_the_question(self, record):
        # Not "account_contact". The record is read by a person, and an
        # internal topic key tells them nothing.
        assert "_" not in record.what_is_unresolved

    @pytest.mark.parametrize("reason", list(DeclineReason))
    def test_every_decline_reason_has_wording(self, reason):
        record = draft(
            principal=persona("northstar_customer"),
            thread_id="t",
            question="q",
            reason=reason,
            subject="the retention period",
        )
        assert "the retention period" in record.what_is_unresolved


class TestWhatTheRecordCarries:
    def test_the_evidence_chain_survives(self, record):
        assert record.evidence_chain == ("acct_abc", "res_def")

    def test_the_sources_actually_read_are_listed(self, record):
        # "We checked and found nothing" is only credible with the list.
        assert record.sources_consulted

    def test_the_account_comes_from_the_principal_not_the_caller(self, record):
        # A client-settable account on an escalation would be the same hole
        # D17 closes one layer up.
        assert record.account_id == "ACCT-003"

    def test_staff_have_no_account_and_the_record_says_so(self):
        record = draft(
            principal=persona("maya_agent"),
            thread_id="t",
            question="q",
            reason=DeclineReason.NO_CITABLE_SOURCE,
            subject="x",
        )
        assert record.account_id is None

    def test_an_underived_severity_is_undetermined_with_a_reason(self, record):
        assert record.to_payload()["severity"] == "undetermined"
        assert record.severity_note

    def test_a_derived_severity_is_carried(self):
        record = draft(
            principal=persona("maya_agent"),
            thread_id="t",
            question="q",
            reason=DeclineReason.UNDETERMINED_SEVERITY,
            subject="x",
            severity="P2",
        )
        assert record.to_payload()["severity"] == "P2"

    def test_it_serialises_for_the_confirmation_card(self, record):
        payload = record.to_payload()
        assert payload["kind"] == "create_escalation"
        assert json.loads(json.dumps(payload))


class TestWhatTheUserIsTold:
    def test_the_summary_admits_the_gap(self, record):
        assert "do not have a source" in record.summary

    def test_the_summary_says_a_record_was_drafted_not_created(self, record):
        # Nothing exists until the confirmation gate. Saying otherwise would be
        # the system reporting an action it has not taken.
        assert "drafted" in record.summary

    def test_the_summary_does_not_invent_a_procedure(self, record):
        # GS-024. Every SaaS product has a settings page and a model will
        # happily invent the path to it.
        text = record.summary.lower()
        for invented in ("settings", "billing portal", "dashboard", "profile page"):
            assert invented not in text

    def test_the_summary_repeats_the_specific_gap(self, record):
        assert "billing contact" in record.summary
