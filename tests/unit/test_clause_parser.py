"""Clause segmentation and document metadata.

Tier, account scope, status and dates are read from each document's own header
rather than from a table keyed on filename. That distinction matters: a
filename table is an assertion about the corpus, while a header parse is a
reading of it, and only the second one survives a seventh document.
"""

from __future__ import annotations

from datetime import date

import pytest
from src.knowledge.clause_parser import Clause, parse_all, parse_document
from src.knowledge.sources import SOURCE_FILES, get_source
from src.knowledge.topics import ALL_TOPICS, Topic


@pytest.fixture(scope="module")
def documents():
    return {doc.doc_id: doc for doc in parse_all()}


@pytest.fixture(scope="module")
def clauses(documents) -> dict[str, Clause]:
    return {c.clause_id: c for doc in documents.values() for c in doc.clauses}


def clause(clauses, clause_id: str) -> Clause:
    assert clause_id in clauses, f"{clause_id} missing; have {sorted(clauses)}"
    return clauses[clause_id]


class TestDocumentMetadata:
    def test_all_six_documents_parse(self, documents):
        assert len(documents) == 6

    @pytest.mark.parametrize(
        "doc_id,tier",
        [
            ("support_policy_v3_current", 2),
            ("support_policy_v2_deprecated", 4),
            ("cancellation_and_service_credit_sop_v4", 2),
            ("product_operations_guide_and_known_issues", 3),
            ("northstar_logistics_enterprise_agreement", 1),
            ("lumenworks_service_agreement", 1),
        ],
    )
    def test_tier_is_derived_from_the_header(self, documents, doc_id, tier):
        assert documents[doc_id].tier == tier

    def test_agreements_are_account_scoped(self, documents):
        assert documents["northstar_logistics_enterprise_agreement"].account_id == "ACCT-001"
        assert documents["lumenworks_service_agreement"].account_id == "ACCT-002"

    def test_general_policy_is_scoped_to_nobody(self, documents):
        # NULL account means "applies to every account", which is what lets one
        # predicate serve both ACL and precedence.
        for doc_id in ("support_policy_v3_current", "cancellation_and_service_credit_sop_v4"):
            assert documents[doc_id].account_id is None

    def test_the_deprecated_policy_says_so(self, documents):
        v2 = documents["support_policy_v2_deprecated"]
        assert v2.status == "DEPRECATED"
        assert v2.is_current is False
        assert v2.superseded_by and "v3" in v2.superseded_by

    def test_current_documents_are_current(self, documents):
        assert documents["support_policy_v3_current"].is_current is True
        assert documents["northstar_logistics_enterprise_agreement"].is_current is True

    @pytest.mark.parametrize(
        "doc_id,effective_from",
        [
            ("support_policy_v3_current", date(2026, 5, 1)),
            ("support_policy_v2_deprecated", date(2025, 1, 1)),
            ("cancellation_and_service_credit_sop_v4", date(2026, 6, 15)),
            ("product_operations_guide_and_known_issues", date(2026, 8, 14)),
            ("northstar_logistics_enterprise_agreement", date(2026, 1, 1)),
            ("lumenworks_service_agreement", date(2026, 3, 1)),
        ],
    )
    def test_effective_dates_are_parsed(self, documents, doc_id, effective_from):
        assert documents[doc_id].effective_from == effective_from

    def test_agreement_terms_have_an_end_date(self, documents):
        assert documents["northstar_logistics_enterprise_agreement"].effective_to == date(
            2026, 12, 31
        )
        assert documents["lumenworks_service_agreement"].effective_to == date(2027, 2, 28)

    def test_a_policy_has_no_end_date(self, documents):
        assert documents["cancellation_and_service_credit_sop_v4"].effective_to is None

    def test_titles_are_readable_because_citations_show_them(self, documents):
        assert documents["northstar_logistics_enterprise_agreement"].title.endswith("Agreement")
        assert "Support Policy v3" in documents["support_policy_v3_current"].title


class TestSegmentation:
    @pytest.mark.parametrize(
        "doc_id,refs",
        [
            ("support_policy_v3_current", ["§1", "§2", "§3", "§4"]),
            ("cancellation_and_service_credit_sop_v4", ["§1", "§2", "§3"]),
            ("northstar_logistics_enterprise_agreement", ["§1", "§2", "§3", "§4"]),
            ("lumenworks_service_agreement", ["§1", "§2", "§3"]),
        ],
    )
    def test_numbered_clauses_are_found(self, documents, doc_id, refs):
        assert [c.clause_ref for c in documents[doc_id].clauses] == refs

    def test_a_document_without_numbered_headings_yields_one_clause(self, documents):
        # Policy v2 has an unnumbered heading. It still needs to be a citable
        # unit so "what changed in v3?" can quote it.
        v2 = documents["support_policy_v2_deprecated"]
        assert [c.clause_ref for c in v2.clauses] == ["§-"]
        assert "Severity" in v2.clauses[0].title

    def test_known_issues_become_their_own_clauses(self, documents):
        # KI-208 and KI-211 are separately citable authorities: detection
        # matches tickets against one issue, not against a section.
        refs = [
            c.clause_ref for c in documents["product_operations_guide_and_known_issues"].clauses
        ]
        assert {"KI-208", "KI-211", "KI-176"} <= set(refs)

    def test_heading_only_stubs_are_dropped(self, documents):
        # "2. Current known issues" carries no content once its KI entries are
        # split out; an empty clause would only pollute retrieval.
        for c in documents["product_operations_guide_and_known_issues"].clauses:
            assert len(c.text) > len(c.title) + 10, c.clause_ref

    def test_clause_ids_are_stable_and_namespaced(self, clauses):
        c = clause(clauses, "northstar_logistics_enterprise_agreement::§2")
        assert c.doc_id == "northstar_logistics_enterprise_agreement"
        assert c.clause_ref == "§2"

    def test_parsing_twice_yields_identical_clauses(self):
        # The registry is committed, so the build must be reproducible.
        first = {c.clause_id: c for doc in parse_all() for c in doc.clauses}
        second = {c.clause_id: c for doc in parse_all() for c in doc.clauses}
        assert first == second

    def test_clause_text_is_verbatim_from_the_document(self, clauses):
        text = clause(clauses, "cancellation_and_service_credit_sop_v4::§1").text
        assert "No fee within 30 minutes of booking" in text
        assert "After 30 minutes, charge INR 250" in text

    def test_a_clause_inherits_its_document_scope(self, clauses):
        c = clause(clauses, "lumenworks_service_agreement::§3")
        assert c.tier == 1
        assert c.account_id == "ACCT-002"


class TestTopicTagging:
    def test_no_clause_is_left_untagged(self, clauses):
        # An untagged clause is invisible to the resolver, which groups by tag.
        untagged = [cid for cid, c in clauses.items() if not c.topics]
        assert untagged == []

    def test_every_tag_is_from_the_closed_enum(self, clauses):
        for c in clauses.values():
            assert set(c.topics) <= ALL_TOPICS, c.clause_id

    @pytest.mark.parametrize(
        "clause_id,topic",
        [
            ("cancellation_and_service_credit_sop_v4::§1", Topic.CANCELLATION_FEE),
            ("northstar_logistics_enterprise_agreement::§2", Topic.CANCELLATION_FEE),
            ("lumenworks_service_agreement::§3", Topic.FAILED_PICKUP_CREDIT),
            ("cancellation_and_service_credit_sop_v4::§2", Topic.FAILED_PICKUP_CREDIT),
            ("cancellation_and_service_credit_sop_v4::§3", Topic.CREDIT_APPROVAL),
            ("support_policy_v3_current::§1", Topic.SOURCE_PRECEDENCE),
            ("support_policy_v3_current::§2", Topic.SEVERITY_DEFINITION),
            ("support_policy_v3_current::§3", Topic.FIRST_RESPONSE_TARGET),
            ("lumenworks_service_agreement::§1", Topic.WEEKEND_COVERAGE),
            ("product_operations_guide_and_known_issues::KI-208", Topic.KNOWN_ISSUE),
        ],
    )
    def test_the_load_bearing_tags_are_present(self, clauses, clause_id, topic):
        assert topic.value in clause(clauses, clause_id).topics

    def test_the_override_pair_shares_a_topic(self, clauses):
        # Precedence is only decidable between clauses about the same subject.
        # If this ever fails, the Northstar override silently stops firing.
        northstar = set(clause(clauses, "northstar_logistics_enterprise_agreement::§2").topics)
        sop = set(clause(clauses, "cancellation_and_service_credit_sop_v4::§1").topics)
        assert Topic.CANCELLATION_FEE.value in northstar & sop

    def test_the_credit_override_pair_shares_a_topic(self, clauses):
        lumen = set(clause(clauses, "lumenworks_service_agreement::§3").topics)
        sop = set(clause(clauses, "cancellation_and_service_credit_sop_v4::§2").topics)
        assert Topic.FAILED_PICKUP_CREDIT.value in lumen & sop


class TestSingleDocument:
    def test_parse_document_matches_parse_all(self, documents):
        source = get_source("lumenworks_service_agreement")
        assert parse_document(source) == documents["lumenworks_service_agreement"]

    def test_every_source_yields_at_least_one_clause(self):
        for source in SOURCE_FILES:
            assert parse_document(source).clauses, source.doc_id


class TestTaggingPrecision:
    """False positives here are not cosmetic.

    The resolver groups by topic, so a clause tagged with a subject it is not
    actually about becomes a competing authority on that subject - and two
    tier-1 clauses on one topic is exactly the shape that triggers an
    unresolved-conflict escalation.
    """

    def test_a_reference_to_the_sop_by_name_is_not_a_credit_clause(self, clauses):
        # LumenWorks §2 is about cancellation and says "use the current
        # ParcelPilot Cancellation & Service Credit SOP". Tagging it
        # failed_pickup_credit would put it up against its own §3.
        c = clause(clauses, "lumenworks_service_agreement::§2")
        assert Topic.FAILED_PICKUP_CREDIT.value not in c.topics
        assert Topic.CANCELLATION_FEE.value in c.topics

    def test_the_word_workaround_alone_is_not_a_known_issue(self, clauses):
        # Policy v3 §2 defines P2 as "a workaround exists". That is a severity
        # definition, not an operational known issue.
        c = clause(clauses, "support_policy_v3_current::§2")
        assert Topic.KNOWN_ISSUE.value not in c.topics
        assert Topic.SEVERITY_DEFINITION.value in c.topics

    def test_only_the_guide_carries_known_issue_clauses(self, clauses):
        owners = {c.doc_id for c in clauses.values() if Topic.KNOWN_ISSUE.value in c.topics}
        assert owners == {"product_operations_guide_and_known_issues"}

    def test_exactly_one_tier_one_clause_per_account_per_override_topic(self, clauses):
        # The invariant the resolver depends on: within one account, one
        # governing agreement clause per subject.
        from collections import Counter

        for account in ("ACCT-001", "ACCT-002"):
            counts = Counter(
                topic for c in clauses.values() if c.account_id == account for topic in c.topics
            )
            contested = {t: n for t, n in counts.items() if n > 1}
            assert not contested, f"{account} has competing tier-1 clauses: {contested}"


class TestKnownIssueTitles:
    def test_a_known_issue_title_stops_at_the_subject(self, clauses):
        # The heading and its body land on one line, so the title has to be
        # cut rather than taken to end of line.
        assert clause(clauses, "product_operations_guide_and_known_issues::KI-176").title == (
            "Address validation"
        )

    def test_known_issue_titles_stay_short_enough_to_cite(self, clauses):
        for c in clauses.values():
            if c.clause_ref.startswith("KI-"):
                assert len(c.title) < 60, (c.clause_ref, c.title)

    def test_the_body_is_still_intact_after_the_title_is_cut(self, clauses):
        c = clause(clauses, "product_operations_guide_and_known_issues::KI-176")
        assert "Resolved 18 July 2026" in c.text
