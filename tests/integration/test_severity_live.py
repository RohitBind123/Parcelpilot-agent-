"""The real severity classifier. Deselected by default; run with `-m live`.

The offline suite proves the validation around inference: a guard cannot be
talked down, an invented span is caught, a failure becomes undetermined rather
than P3. What it cannot prove is that a real model, handed the real definitions,
grades the real tickets the way the golden set says.

These are also the tests that would notice the calibration going stale. The
threshold in `severity.py` was chosen from the numbers this file asserts; if a
model update moves them, the number stops being justified and should be
recalibrated rather than nudged.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.severity import (
    CONFIDENCE_THRESHOLD,
    deterministic_severity,
    infer_severity,
    load_severity_definitions,
)
from src.domain.severity_llm import LlmSeverityClassifier
from src.providers.registry import get_chat_provider

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def definitions():
    connection = sqlite3.connect(get_settings().db_path)
    try:
        yield load_severity_definitions(connection)
    finally:
        connection.close()


@pytest.fixture(scope="module")
def classifier():
    return LlmSeverityClassifier(get_chat_provider())


@pytest.fixture(scope="module")
def tickets():
    principal = to_principal(get_persona("priya_manager"))
    with open_repository(principal, get_settings().db_path) as repo:
        return {t.ticket_id: t for t in repo.query_tickets(status="open")}


def grade(classifier, definitions, ticket):
    return infer_severity(
        ticket.subject, ticket.description or "", definitions=definitions, classifier=classifier
    )


class TestTheModelGradesThePackCorrectly:
    @pytest.mark.parametrize(("ticket_id", "expected"), [("TKT-502", "P2"), ("TKT-503", "P3")])
    def test_the_unambiguous_tickets_get_the_signed_off_severity(
        self, classifier, definitions, tickets, ticket_id, expected
    ):
        verdict = grade(classifier, definitions, tickets[ticket_id])
        assert verdict.severity == expected
        assert verdict.is_trusted, f"confidence {verdict.confidence} is below the threshold"

    def test_the_quoted_span_is_really_in_the_policy(self, classifier, definitions, tickets):
        # The check that makes the citation worth printing. If this fails the
        # model is paraphrasing, and a paraphrased authority is not one.
        verdict = grade(classifier, definitions, tickets["TKT-502"])
        assert verdict.basis
        assert definitions.contains(verdict.severity, verdict.basis)

    def test_the_genuinely_ambiguous_ticket_is_not_confidently_graded(
        self, classifier, definitions, tickets
    ):
        # TKT-504: a status display that lags is either a materially degraded
        # feature or a minor defect, and section 2 does not say which. The
        # system is supposed to notice that it does not know.
        #
        # Sampled three times, because the first version of this test asserted
        # one draw was below the threshold and flaked - which is the caveat in
        # `severity.py` arriving in practice rather than a separate problem.
        # The property is that the ticket does not get graded confidently and
        # consistently; either instability or low confidence establishes it,
        # and a single sample establishes neither.
        verdicts = [grade(classifier, definitions, tickets["TKT-504"]) for _ in range(3)]
        unstable = len({v.severity for v in verdicts}) > 1
        unsure = any(v.confidence < CONFIDENCE_THRESHOLD for v in verdicts)
        assert unstable or unsure, (
            f"TKT-504 graded {verdicts[0].severity} confidently three times running; "
            "the definitions may have changed, or the threshold needs recalibrating"
        )


class TestTheGuardsDoNotDependOnTheModel:
    @pytest.mark.parametrize("ticket_id", ["TKT-501", "TKT-505"])
    def test_the_two_p1_triggers_never_reach_the_classifier(
        self, classifier, definitions, tickets, ticket_id
    ):
        ticket = tickets[ticket_id]
        assert deterministic_severity(ticket.subject, ticket.description or "") is not None

        calls: list[str] = []

        class Counting:
            def classify(self, subject, description, defs):
                calls.append(subject)
                raise AssertionError("the classifier must not be consulted for a guard match")

        verdict = infer_severity(
            ticket.subject, ticket.description or "", definitions=definitions, classifier=Counting()
        )
        assert verdict.severity == "P1"
        assert verdict.deterministic is True
        assert calls == []


class TestStability:
    def test_a_stable_ticket_grades_the_same_three_times_running(
        self, classifier, definitions, tickets
    ):
        # The property the threshold is standing in for. If this starts
        # flapping, the calibration is stale.
        verdicts = [grade(classifier, definitions, tickets["TKT-502"]) for _ in range(3)]
        assert {v.severity for v in verdicts} == {"P2"}
