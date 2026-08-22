"""Hybrid retrieval: BM25, dense, and the fusion between them.

Two retrievers exist here because the queries are not all the same shape. "Am I
being charged for cancelling?" is a semantic question. "What does SOP v4 §1
say?" is a lookup, and a dense retriever is poor at lookups - an exact
reference carries almost no semantic content to match on. BM25 costs nothing at
19 clauses and covers the case dense is worst at.

The fusion is tested as a pure function over rankings, because that is what it
is. Testing it only through two live retrievers would leave its tie-breaking
and its rank arithmetic implicit.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.knowledge.retriever import (
    BM25Index,
    HybridRetriever,
    build_lexical_index,
    reciprocal_rank_fusion,
    tokenize,
)
from src.knowledge.vectorstore.base import Chunk
from src.knowledge.vectorstore.chroma import ChromaLocalStore
from tests.support.embeddings import HashingEmbeddings

NORTHSTAR = "northstar_logistics_enterprise_agreement::§2"
LUMENWORKS = "lumenworks_service_agreement::§2"
SOP_CANCELLATION = "cancellation_and_service_credit_sop_v4::§1"
SOP_CREDIT = "cancellation_and_service_credit_sop_v4::§2"
POLICY_V2 = "support_policy_v2_deprecated::§-"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


def chunk(clause_id, *, tier, account_id, text, title="", doc_title="", topics=()) -> Chunk:
    doc_id, _, ref = clause_id.partition("::")
    return Chunk(
        clause_id=clause_id,
        doc_id=doc_id,
        doc_title=doc_title or doc_id.replace("_", " ").title(),
        clause_ref=ref,
        title=title or f"title of {ref}",
        tier=tier,
        account_id=account_id,
        status="ACTIVE",
        topics=tuple(topics),
        text=text,
    )


CORPUS = (
    chunk(
        NORTHSTAR,
        tier=1,
        account_id="ACCT-001",
        doc_title="Northstar Logistics Enterprise Agreement",
        title="Cancellation",
        text="Cancellation fees are waived for Northstar on booked shipments.",
        topics=("cancellation_fee",),
    ),
    chunk(
        LUMENWORKS,
        tier=1,
        account_id="ACCT-002",
        doc_title="LumenWorks Service Agreement",
        title="Cancellation",
        text="No special cancellation fee waiver applies. Use the current SOP.",
        topics=("cancellation_fee",),
    ),
    chunk(
        SOP_CANCELLATION,
        tier=2,
        account_id=None,
        doc_title="Cancellation and Service Credit SOP v4",
        title="Cancellation window and fees",
        text="Cancellation within 30 minutes of booking is free; after that a fee of INR 250 applies.",
        topics=("cancellation_fee", "cancellation_window"),
    ),
    chunk(
        SOP_CREDIT,
        tier=2,
        account_id=None,
        doc_title="Cancellation and Service Credit SOP v4",
        title="Failed pickup service credit",
        text="A failed pickup more than 2 hours late earns the lower of INR 500 or 10 percent.",
        topics=("failed_pickup_credit",),
    ),
    chunk(
        POLICY_V2,
        tier=4,
        account_id=None,
        doc_title="Support Policy v2 (Deprecated)",
        title="Response targets",
        text="Deprecated first response targets for every plan tier.",
        topics=("first_response_target",),
    ),
)


@pytest.fixture(scope="module")
def lexical() -> BM25Index:
    return BM25Index(CORPUS)


@pytest.fixture
def hybrid(tmp_path) -> HybridRetriever:
    dense = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=tmp_path / "index")
    dense.upsert(CORPUS)
    return HybridRetriever(dense=dense, lexical=BM25Index(CORPUS))


def ids(chunks) -> list[str]:
    return [c.clause_id for c in chunks]


class TestReciprocalRankFusion:
    def test_a_clause_both_retrievers_rank_beats_one_that_only_one_ranks(self):
        a, b, c = CORPUS[0], CORPUS[1], CORPUS[2]
        fused = reciprocal_rank_fusion([[a, b], [c, a]])
        # `a` is 1st and 2nd; `b` and `c` are each ranked once. Agreement wins
        # even though `b` and `c` each hold a first place.
        assert ids(fused)[0] == a.clause_id

    def test_a_top_rank_in_one_list_can_still_win_without_agreement(self):
        a, b = CORPUS[0], CORPUS[1]
        fused = reciprocal_rank_fusion([[a, b], []])
        assert ids(fused) == [a.clause_id, b.clause_id]

    def test_the_score_is_the_sum_of_reciprocal_ranks(self):
        a = CORPUS[0]
        fused = reciprocal_rank_fusion([[a], [a]], k=60)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_rank_not_score_is_what_fuses(self):
        # The whole point of RRF: a cosine similarity and a BM25 score are not
        # on the same scale, so fusing the numbers directly would let whichever
        # retriever happens to emit larger values dominate.
        a, b = CORPUS[0], CORPUS[1]
        loud = [a.with_score(9_000.0), b.with_score(8_999.0)]
        quiet = [b.with_score(0.02), a.with_score(0.01)]
        assert ids(reciprocal_rank_fusion([loud, quiet])) == ids(
            reciprocal_rank_fusion([quiet, loud])
        )

    def test_empty_input_is_empty_output(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_results_are_ordered_by_descending_score(self):
        fused = reciprocal_rank_fusion([list(CORPUS[:3]), list(reversed(CORPUS[:3]))])
        assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)

    def test_ties_break_deterministically(self):
        a, b = CORPUS[0], CORPUS[1]
        # Perfectly symmetric input: both appear once at rank 1. Whatever order
        # comes out must be the same order every run, or the golden set becomes
        # flaky for reasons that have nothing to do with the answer.
        first = ids(reciprocal_rank_fusion([[a], [b]]))
        assert all(ids(reciprocal_rank_fusion([[a], [b]])) == first for _ in range(5))


class TestBM25:
    def test_an_exact_clause_reference_is_found(self, lexical):
        # The case dense retrieval is worst at: "§1" carries no semantics.
        found = lexical.search("SOP v4 §1", principal=persona("beacon_customer"), k=3)
        assert found[0].clause_id == SOP_CANCELLATION

    def test_the_document_title_is_searchable_not_just_the_body(self, lexical):
        found = lexical.search(
            "Northstar enterprise agreement", principal=persona("maya_agent"), k=3
        )
        assert found[0].clause_id == NORTHSTAR

    def test_a_rare_term_outranks_a_common_one(self, lexical):
        # "cancellation" appears in three clauses; "waived" in one.
        found = lexical.search("waived", principal=persona("maya_agent"), k=3)
        assert found[0].clause_id == NORTHSTAR

    def test_lexical_search_enforces_the_same_acl_as_dense(self, lexical):
        # The failure this guards against is the quiet one: bolt a second
        # retriever onto a store whose ACL you trust, forget the predicate on
        # the new path, and the leak is invisible because the dense path is
        # still correct.
        found = lexical.search(
            "cancellation fee waiver", principal=persona("lumenworks_customer"), k=10
        )
        assert NORTHSTAR not in ids(found)

    def test_lexical_search_excludes_deprecated_tiers_by_default(self, lexical):
        found = lexical.search("response targets", principal=persona("maya_agent"), k=10)
        assert POLICY_V2 not in ids(found)

    def test_a_no_match_query_returns_nothing_rather_than_noise(self, lexical):
        assert lexical.search("kubernetes", principal=persona("maya_agent"), k=5) == []

    def test_topics_narrow_the_search(self, lexical):
        found = lexical.search(
            "credit", principal=persona("maya_agent"), topics=["failed_pickup_credit"], k=5
        )
        assert ids(found) == [SOP_CREDIT]


class TestHybrid:
    def test_both_paths_contribute(self, hybrid):
        found = hybrid.retrieve("cancellation fee", principal=persona("maya_agent"), k=10)
        assert {NORTHSTAR, LUMENWORKS, SOP_CANCELLATION} <= set(ids(found))

    def test_the_acl_holds_through_fusion(self, hybrid):
        for who, forbidden in (
            ("northstar_customer", LUMENWORKS),
            ("lumenworks_customer", NORTHSTAR),
        ):
            found = hybrid.retrieve("cancellation fee waiver", principal=persona(who), k=10)
            assert forbidden not in ids(found)

    def test_an_exact_reference_survives_fusion(self, hybrid):
        found = hybrid.retrieve("SOP v4 §1", principal=persona("beacon_customer"), k=5)
        assert found[0].clause_id == SOP_CANCELLATION

    def test_k_bounds_the_fused_result(self, hybrid):
        assert len(hybrid.retrieve("cancellation", principal=persona("maya_agent"), k=2)) == 2

    def test_retrieval_survives_an_empty_dense_index(self, tmp_path):
        # Degradation, not failure: an unbuilt or unreachable collection should
        # leave lexical search working rather than take the whole answer down.
        empty = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=tmp_path / "empty")
        retriever = HybridRetriever(dense=empty, lexical=BM25Index(CORPUS))
        found = retriever.retrieve("cancellation fee", principal=persona("maya_agent"), k=5)
        assert SOP_CANCELLATION in ids(found)

    def test_deprecated_policy_is_reachable_only_on_request(self, hybrid):
        default = hybrid.retrieve("response targets", principal=persona("maya_agent"), k=10)
        assert POLICY_V2 not in ids(default)

        asked = hybrid.retrieve(
            "response targets", principal=persona("maya_agent"), tiers={4}, k=10
        )
        assert POLICY_V2 in ids(asked)


class TestTokenizer:
    """Regression: a section reference must not be matched by a stray digit.

    "SOP v4 §1" came back pointing at §3. `INR 1,000` tokenized to `1` and
    `000`, so §3 held a bare `1`; being the shortest clause in the document,
    BM25's length normalisation floated it to the top. The citation looked
    plausible and pointed at the wrong rule.
    """

    def test_a_thousands_separator_does_not_split_a_number(self):
        assert tokenize("INR 1,000") == ["inr", "1000"]

    def test_a_section_reference_is_one_token(self):
        assert tokenize("SOP v4 §1") == ["sop", "v4", "section_1"]

    def test_the_written_and_symbol_spellings_agree(self):
        assert tokenize("Section 2") == tokenize("§2") == ["section_2"]

    def test_a_section_reference_does_not_match_a_bare_digit(self):
        assert "1" not in tokenize("§1")
        assert "section_1" not in tokenize("credit above INR 1,000")

    def test_a_placeholder_reference_yields_no_spurious_token(self):
        # Policy v2 has no numbered sections; its ref is "§-".
        assert tokenize("§-") == []

    def test_known_issue_identifiers_survive(self):
        assert tokenize("KI-208") == ["ki", "208"]

    def test_the_exact_reference_outranks_the_clause_holding_a_stray_digit(self, lexical):
        found = lexical.search("SOP v4 §1", principal=persona("maya_agent"), k=5)
        assert found[0].clause_id == SOP_CANCELLATION


class TestSharedRepresentation:
    def test_both_retrievers_index_the_same_string(self, lexical, hybrid):
        # If these drifted, a clause could be findable lexically and invisible
        # semantically, which surfaces as a citation that appears at random.
        chunk = CORPUS[0]
        assert chunk.doc_title in chunk.searchable_text
        assert chunk.clause_ref in chunk.searchable_text
        assert chunk.title in chunk.searchable_text
        assert chunk.text in chunk.searchable_text

    def test_a_clause_is_findable_by_its_document_title_through_both_paths(self, hybrid):
        found = hybrid.retrieve(
            "Northstar Logistics Enterprise Agreement", principal=persona("maya_agent"), k=5
        )
        assert NORTHSTAR in ids(found)


class _BrokenDense:
    """A dense store that fails the way a hosted one does."""

    identity = "test/broken/1"
    collection_name = "broken"

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def upsert(self, chunks):
        return 0

    def count(self) -> int:
        return 0

    def query(self, text, **kwargs):
        self.calls += 1
        raise self._error


class TestDegradation:
    """What happens when the hosted half of retrieval is not there.

    The two cases have to be told apart. An unreachable Chroma is a transient
    infrastructure fault and should cost recall, not the answer. A malformed
    Principal is a bug in access control, and swallowing it would convert a
    query that should have been refused into one that silently ran unfiltered.
    """

    def test_a_transient_dense_failure_falls_back_to_lexical(self, caplog):
        broken = _BrokenDense(ConnectionError("chroma unreachable"))
        retriever = HybridRetriever(dense=broken, lexical=BM25Index(CORPUS))

        found = retriever.retrieve("cancellation fee", principal=persona("maya_agent"), k=5)

        assert broken.calls == 1
        assert SOP_CANCELLATION in ids(found)
        assert "falling back to lexical" in caplog.text

    def test_the_fallback_still_enforces_the_acl(self):
        # The dangerous version of degrading: losing the filter along with the
        # retriever it lived in.
        retriever = HybridRetriever(
            dense=_BrokenDense(ConnectionError("down")), lexical=BM25Index(CORPUS)
        )
        found = retriever.retrieve(
            "cancellation fee waiver", principal=persona("lumenworks_customer"), k=10
        )
        assert NORTHSTAR not in ids(found)
        assert found, "degraded retrieval should still return the clauses it may return"

    def test_a_malformed_principal_is_never_degraded_into_an_unfiltered_query(self):
        retriever = HybridRetriever(
            dense=_BrokenDense(ValueError("refusing to query with malformed account_id")),
            lexical=BM25Index(CORPUS),
        )
        with pytest.raises(ValueError, match="account_id"):
            retriever.retrieve("cancellation", principal=persona("northstar_customer"), k=5)


class TestConstruction:
    def test_build_lexical_index_accepts_any_iterable(self):
        index = build_lexical_index(chunk for chunk in CORPUS)
        assert len(index) == len(CORPUS)

    def test_an_empty_corpus_searches_without_raising(self):
        assert BM25Index([]).search("anything", principal=persona("maya_agent"), k=5) == []
