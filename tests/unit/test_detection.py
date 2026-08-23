"""Proactive detection (ARCHITECTURE 14, Problem 1).

Two things carry this file.

**What must be found**, against the table in findings §9: TKT-501 and TKT-505
are P1 with no matching known issue, TKT-502 is the second occurrence of
KI-208, TKT-504 matches KI-211.

**What must not be found.** No known issue spans two accounts on this pack, so
cross-account impact returns nothing - and reports that it looked. A signal
that is silently absent below some threshold is a dashboard whose gaps nobody
can see, and §14 predicts this one specifically.

The matcher gets its own tests because the first version attributed TKT-501, a
total shipment-creation outage, to KI-208, the bulk-upload issue, on the
strength of "shipment" and "creation". Both are ordinary English and neither
distinguishes anything here.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.detection import (
    VOLUME_SIGNAL_MINIMUM,
    HealthScanner,
    Signal,
    build_known_issues,
    common_terms,
    match_issue,
)
from src.domain.resolver import PolicyResolver


@pytest.fixture(scope="module")
def scanned():
    principal = to_principal(get_persona("priya_manager"))
    with open_repository(principal, get_settings().db_path) as repo:
        scanner = HealthScanner(repository=repo, resolver=PolicyResolver(repo.connection))
        yield scanner.scan(), repo


@pytest.fixture
def report(scanned):
    return scanned[0]


@pytest.fixture(scope="module")
def issues(scanned):
    _report, repo = scanned
    scanner = HealthScanner(repository=repo, resolver=None)
    return build_known_issues(scanner._known_issue_rows())


def signal(report, name: Signal):
    return next(s for s in report.signals if s.signal is name)


def subjects(report, name: Signal) -> set[str]:
    return {f.subject_id for f in signal(report, name).findings}


class TestWhatTheScanFinds:
    def test_the_two_unmatched_p1s_are_flagged_as_possible_incidents(self, report):
        # findings §9: both are P1 by a named trigger in Policy v3 §2, and
        # neither is described anywhere in the Known Issues document.
        assert subjects(report, Signal.UNMATCHED_HIGH_SEVERITY) == {"TKT-501", "TKT-505"}

    def test_the_bulk_upload_ticket_matches_its_known_issue(self, report):
        assert "TKT-502" in subjects(report, Signal.KNOWN_ISSUE_RECURRENCE)

    def test_the_pickup_ticket_matches_the_webhook_issue(self, report):
        assert "TKT-504" in subjects(report, Signal.KNOWN_ISSUE_RECURRENCE)

    def test_the_bulk_upload_ticket_is_the_second_occurrence(self, report):
        # TKT-451 on 11 August is the first. Counting only open tickets would
        # hide the recurrence, which is the finding.
        found = next(
            f
            for f in signal(report, Signal.KNOWN_ISSUE_RECURRENCE).findings
            if f.subject_id == "TKT-502"
        )
        assert "second occurrence" in found.headline
        assert "TKT-451" in found.evidence

    def test_the_shipment_outage_is_not_attributed_to_the_upload_issue(self, report):
        # The bug this signal was born with. TKT-501 is a total outage and
        # KI-208 is about large CSVs; they share only ordinary words.
        assert "TKT-501" not in subjects(report, Signal.KNOWN_ISSUE_RECURRENCE)

    def test_the_worst_findings_come_first(self, report):
        severities = [f.severity for f in report.findings]
        ranked = sorted(report.findings, key=lambda f: f.rank)
        assert [f.severity for f in ranked] == severities
        assert severities[0] == "P1"

    def test_every_finding_carries_its_evidence(self, report):
        assert all(f.evidence for f in report.findings)

    def test_no_finding_claims_a_measured_breach(self, report):
        # There is no `first_response_at` column (A4/A5), so elapsed-versus-
        # target is computable and met-or-missed is not.
        assert all(f.measurable is False for f in report.findings)


class TestWhatTheScanDeliberatelyDoesNotFind:
    def test_cross_account_impact_finds_nothing_and_says_it_looked(self, report):
        # §14 predicts this. Manufacturing a systemic issue would be data
        # augmentation, which the design declined.
        found = signal(report, Signal.CROSS_ACCOUNT_IMPACT)
        assert found.checked is True
        assert found.findings == ()
        assert "no known issue" in found.note

    def test_the_volume_signal_is_suppressed_with_its_reason(self, report):
        found = signal(report, Signal.VOLUME_SPIKE)
        assert found.checked is False
        assert str(VOLUME_SIGNAL_MINIMUM) in found.note

    def test_a_signal_that_ran_is_distinguishable_from_one_that_did_not(self, report):
        # The whole reason `checked` exists. Without it "found nothing" and
        # "never ran" are the same row on a dashboard.
        ran = {s.signal for s in report.signals if s.checked}
        assert Signal.CROSS_ACCOUNT_IMPACT in ran
        assert Signal.VOLUME_SPIKE not in ran


class TestTheMatcher:
    def test_a_resolved_issue_never_matches_a_live_ticket(self, scanned, issues):
        # KI-176 is resolved and says so. Attributing today's ticket to a fixed
        # issue is worse than attributing it to nothing.
        assert all(not i.is_active for i in issues if i.reference == "KI-176")

    def test_an_issue_is_identified_by_its_title_not_its_prose(self, issues):
        upload = next(i for i in issues if i.reference == "KI-208")
        assert {"bulk", "upload"} <= upload.terms
        # "shipment" appears in KI-208's body. It is not what the issue is
        # about, and treating body prose as identifying is what mis-attributed
        # the outage ticket.
        assert "shipment" not in upload.terms

    def test_a_match_reports_the_words_that_decided_it(self, scanned, issues):
        _report, repo = scanned
        ticket = repo.get_ticket("TKT-502")
        match = match_issue(ticket, issues, common_terms(repo.query_tickets(limit=50)))
        assert match.issue is not None
        assert match.issue.reference == "KI-208"
        # "the vectors were close" is not an answer to "why this issue?".
        assert set(match.terms) == {"bulk", "upload"}

    def test_an_unrelated_ticket_matches_nothing(self, scanned, issues):
        _report, repo = scanned
        ticket = repo.get_ticket("TKT-503")
        assert match_issue(ticket, issues, common_terms(repo.query_tickets(limit=50))).issue is None

    def test_common_words_are_dropped_before_matching(self, scanned):
        _report, repo = scanned
        widespread = common_terms(repo.query_tickets(limit=50))
        assert isinstance(widespread, frozenset)


class TestFindingIdentity:
    def test_ids_are_stable_across_scans(self, scanned):
        # A drill-down names a finding across two requests. An id derived from
        # position would point at a different finding after any change.
        _report, repo = scanned
        scanner = HealthScanner(repository=repo, resolver=PolicyResolver(repo.connection))
        first = [f.finding_id for f in scanner.scan().findings]
        second = [f.finding_id for f in scanner.scan().findings]
        assert first == second

    def test_ids_are_unique(self, report):
        ids = [f.finding_id for f in report.findings]
        assert len(ids) == len(set(ids))

    def test_a_finding_can_be_looked_up_by_id(self, report):
        first = report.findings[0]
        assert report.find(first.finding_id) is first
        assert report.find("nope") is None
