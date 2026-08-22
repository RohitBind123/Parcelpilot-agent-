"""Severity inference (D23), and the guards it must never be able to overrule.

Policy v3 section 2 names two P1 triggers in words specific enough to match in
Python: a complete production outage preventing all shipment creation, and a
confirmed security incident or suspected credential exposure. Those are matched
by guard and never sampled, because a P1 that a classifier occasionally reads as
P2 is an outage nobody was paged for.

The rest is judgment, so it goes to a model. That opens three failure modes this
file is written against: a classifier that talks a guard match down, one that
cites a definition span the policy does not contain, and one that fails in a way
the caller reads as a confident P3. All three end the same way - a ticket
carrying a severity the policy does not support - and only the first is
obviously a bug when you read the output.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.config import get_settings
from src.domain.severity import (
    CONFIDENCE_THRESHOLD,
    SEVERITIES,
    ClassifierVerdict,
    SeverityDefinitions,
    SeverityVerdict,
    deterministic_severity,
    infer_severity,
    load_severity_definitions,
)

OUTAGE = (
    "All shipment creation is failing",
    "Every user at Northstar gets HTTP 500 when creating any shipment. "
    "Existing shipments can still be viewed.",
)
CREDENTIAL = (
    "Possible API key exposure",
    "An employee accidentally posted a screenshot containing a production API key "
    "in a public channel. They are asking what to do.",
)
BULK_UPLOAD = (
    "Bulk upload fails for 4,200-row CSV",
    "The CSV reaches roughly 70% and fails. Creating shipments one-by-one still works.",
)
HOW_TO = (
    "How do we change the billing contact?",
    "Customer wants to replace the billing-contact email on their account.",
)


@pytest.fixture(scope="module")
def definitions() -> SeverityDefinitions:
    connection = sqlite3.connect(get_settings().db_path)
    try:
        yield load_severity_definitions(connection)
    finally:
        connection.close()


class Stub:
    """A classifier with a scripted answer, so inference is testable offline."""

    def __init__(self, verdict=None, *, raises: Exception | None = None):
        self.verdict = verdict
        self.raises = raises
        self.calls = 0

    def classify(self, subject, description, definitions):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.verdict


def verdict(severity="P2", confidence=1.0, span="Major feature unavailable"):
    return ClassifierVerdict(severity=severity, confidence=confidence, matched_span=span)


class TestDefinitions:
    def test_the_spans_are_read_from_the_clause_not_retyped(self, definitions):
        # If the policy is edited, inference must follow it. A copy in this
        # module would keep grading against a rule that no longer exists.
        assert set(definitions.spans) == {"P1", "P2", "P3"}
        assert "Complete production outage" in definitions.spans["P1"]
        assert "workaround exists" in definitions.spans["P2"]
        assert "how-to question" in definitions.spans["P3"]

    def test_the_definitions_carry_the_clause_that_states_them(self, definitions):
        assert definitions.clause_id == "support_policy_v3_current::§2"

    def test_the_deprecated_policy_is_not_the_source(self, definitions):
        # Policy v2 also defines severities, with different targets attached.
        assert "v2" not in definitions.clause_id

    def test_a_span_can_be_checked_for_membership(self, definitions):
        assert definitions.contains("P3", "how-to question")
        assert not definitions.contains("P3", "complete production outage")

    def test_membership_ignores_case_and_whitespace_noise(self, definitions):
        # A model quoting across a line break should not be called a liar for it.
        assert definitions.contains("P1", "complete   production\n outage")


class TestGuardsWin:
    def test_a_guard_match_never_reaches_the_classifier(self, definitions):
        stub = Stub(verdict(severity="P3", confidence=0.99))
        got = infer_severity(*OUTAGE, definitions=definitions, classifier=stub)
        assert got.severity == "P1"
        assert got.deterministic is True
        assert stub.calls == 0

    def test_credential_exposure_is_p1_however_calmly_it_is_worded(self, definitions):
        stub = Stub(verdict(severity="P3", confidence=0.99))
        got = infer_severity(*CREDENTIAL, definitions=definitions, classifier=stub)
        assert got.severity == "P1"
        assert got.deterministic is True

    def test_the_guard_cites_the_severity_clause(self, definitions):
        got = infer_severity(*OUTAGE, definitions=definitions, classifier=None)
        assert got.basis_clause == definitions.clause_id

    def test_the_guard_still_answers_with_no_classifier_configured(self, definitions):
        # The deployment where no model is reachable must still page for an
        # outage. It is the case where being right matters most.
        assert (
            infer_severity(*CREDENTIAL, definitions=definitions, classifier=None).severity == "P1"
        )


class TestInference:
    def test_a_trusted_verdict_passes_through(self, definitions):
        got = infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=Stub(verdict()))
        assert got.severity == "P2"
        assert got.deterministic is False
        assert got.is_trusted

    def test_the_matched_span_becomes_the_basis(self, definitions):
        got = infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=Stub(verdict()))
        assert got.basis == "Major feature unavailable"
        assert got.basis_clause == definitions.clause_id

    def test_a_how_to_question_can_be_graded_p3(self, definitions):
        stub = Stub(verdict(severity="P3", span="how-to question"))
        assert infer_severity(*HOW_TO, definitions=definitions, classifier=stub).severity == "P3"


class TestTheClassifierIsNotTrusted:
    def test_a_span_absent_from_the_policy_is_not_believed(self, definitions):
        # An invented citation is the failure the whole design is arranged
        # against, and it is invisible unless something checks.
        stub = Stub(verdict(span="P2 - anything the customer considers urgent"))
        got = infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=stub)
        assert not got.is_trusted
        assert any("not found" in w for w in got.warnings)

    def test_a_span_from_the_wrong_severity_is_not_believed(self, definitions):
        # Quoting the P1 definition while answering P2 means the model did not
        # read the definition it claims to have matched.
        stub = Stub(verdict(severity="P2", span="Complete production outage"))
        assert not infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=stub).is_trusted

    def test_a_severity_outside_the_policy_is_refused_outright(self, definitions):
        stub = Stub(verdict(severity="P0", confidence=0.99))
        got = infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=stub)
        assert got.severity is None
        assert not got.is_trusted

    @pytest.mark.parametrize("raw", [1.4, -0.2])
    def test_confidence_is_clamped_to_a_probability(self, definitions, raw):
        got = infer_severity(
            *BULK_UPLOAD, definitions=definitions, classifier=Stub(verdict(confidence=raw))
        )
        assert 0.0 <= got.confidence <= 1.0

    def test_a_classifier_that_raises_yields_undetermined_not_a_guess(self, definitions):
        got = infer_severity(
            *BULK_UPLOAD,
            definitions=definitions,
            classifier=Stub(raises=RuntimeError("provider down")),
        )
        assert got.severity is None
        assert got.confidence == 0.0
        assert not got.is_trusted

    def test_a_classifier_returning_nothing_yields_undetermined(self, definitions):
        got = infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=Stub(None))
        assert got.severity is None

    def test_no_classifier_and_no_guard_is_undetermined_rather_than_p3(self, definitions):
        # "Nothing matched" and "this is a minor issue" are different
        # statements, and only inference is entitled to make the second.
        got = infer_severity(*HOW_TO, definitions=definitions, classifier=None)
        assert got.severity is None
        assert not got.is_trusted


class TestTheThresholdIsCalibrated:
    def test_the_threshold_is_a_probability(self):
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_a_verdict_at_the_threshold_is_trusted(self, definitions):
        stub = Stub(verdict(confidence=CONFIDENCE_THRESHOLD))
        assert infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=stub).is_trusted

    def test_a_verdict_just_below_it_is_not(self, definitions):
        stub = Stub(verdict(confidence=CONFIDENCE_THRESHOLD - 0.01))
        assert not infer_severity(*BULK_UPLOAD, definitions=definitions, classifier=stub).is_trusted


class TestTheGuardsStillHoldAlone:
    """`deterministic_severity` is load-bearing on its own; M3 tests it through
    the SLA calculator, so its edges belong here."""

    def test_one_failed_shipment_is_not_an_outage(self):
        assert deterministic_severity("Shipment creation failed", "One shipment errored.") is None

    def test_a_bulk_upload_failure_is_not_an_outage(self):
        # It reaches 70% and single creation works, so nothing is complete
        # about it. A looser pattern pages someone for this every time.
        assert deterministic_severity(*BULK_UPLOAD) is None

    def test_asking_about_api_keys_is_not_an_exposure(self):
        assert deterministic_severity("Rotating our API key", "How do we rotate a key?") is None

    def test_the_verdict_is_frozen(self):
        got = deterministic_severity(*OUTAGE)
        assert isinstance(got, SeverityVerdict)
        with pytest.raises((AttributeError, TypeError)):
            got.severity = "P3"  # type: ignore[misc]


class TestTheClassifierHandlesAMisbehavingModel:
    """`_to_verdict` is the boundary where a provider response becomes a typed
    verdict. Every path out of it that is not a clean verdict must be None, so
    `infer_severity` records "the classifier returned no verdict" rather than
    building a grading out of whatever came back."""

    @pytest.fixture
    def to_verdict(self):
        from src.domain.severity_llm import _to_verdict

        return _to_verdict

    def test_a_clean_mapping_becomes_a_verdict(self, to_verdict):
        got = to_verdict({"severity": "P2", "confidence": 0.9, "matched_span": "Major feature"})
        assert (got.severity, got.confidence) == ("P2", 0.9)

    def test_json_arriving_as_a_string_is_parsed(self, to_verdict):
        # Providers differ on whether structured output comes back decoded.
        raw = '{"severity": "P3", "confidence": 1.0, "matched_span": "how-to question"}'
        assert to_verdict(raw).severity == "P3"

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            {"confidence": 0.9, "matched_span": "x"},
            {"severity": "P2", "matched_span": "x"},
            {"severity": "P2", "confidence": "very", "matched_span": "x"},
            None,
            [],
        ],
    )
    def test_anything_unusable_becomes_none_rather_than_a_partial_verdict(self, to_verdict, raw):
        assert to_verdict(raw) is None

    def test_an_unusable_response_reaches_infer_severity_as_undetermined(self, definitions):
        class Garbage:
            def classify(self, subject, description, defs):
                from src.domain.severity_llm import _to_verdict

                return _to_verdict("not json at all")

        got = infer_severity(
            "Bulk upload fails", "It fails.", definitions=definitions, classifier=Garbage()
        )
        assert got.severity is None
        assert not got.is_trusted

    def test_the_prompt_carries_the_definitions_and_asks_for_the_span(self, definitions):
        from src.domain.severity_llm import _prompt

        prompt = _prompt("Subject", "Description", definitions)
        assert definitions.spans["P1"] in prompt
        assert definitions.clause_id in prompt
        assert "verbatim" in prompt.lower()

    def test_the_schema_constrains_the_severity_to_the_three_the_policy_defines(self):
        # Belt to the validation's braces: the model is asked for an enum, and
        # the answer is checked against SEVERITIES anyway, because a provider
        # that ignores the schema must not be able to widen the vocabulary.
        from src.domain.severity_llm import _SCHEMA

        assert _SCHEMA["properties"]["severity"]["enum"] == list(SEVERITIES)
        assert _SCHEMA["additionalProperties"] is False
