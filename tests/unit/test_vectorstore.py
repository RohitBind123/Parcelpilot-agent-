"""The vector store, and the access control that lives inside it.

The reason this layer exists at all is the ACL. Retrieval is the one place
where a customer's question is matched against a corpus that contains other
customers' contracts, so "which rows may this principal see" cannot be a
parameter a caller remembers to pass - it has to be a property of the store.
These tests attack that boundary rather than the ranking, which is why several
of them inspect the signature rather than the results.
"""

from __future__ import annotations

import inspect

import pytest

from src.auth.personas import get_persona, to_principal
from src.auth.principal import Principal, build_principal
from src.knowledge.vectorstore.base import GLOBAL_SCOPE, Chunk, VectorStore
from src.knowledge.vectorstore.chroma import ChromaLocalStore
from tests.support.embeddings import HashingEmbeddings


def persona(persona_id: str) -> Principal:
    return to_principal(get_persona(persona_id))


NORTHSTAR = "northstar_logistics_enterprise_agreement::§2"
LUMENWORKS = "lumenworks_service_agreement::§2"
SOP_CANCELLATION = "cancellation_and_service_credit_sop_v4::§1"
POLICY_V2 = "support_policy_v2_deprecated::§-"


def chunk(clause_id: str, *, tier: int, account_id: str | None, text: str, topics=()) -> Chunk:
    doc_id, _, ref = clause_id.partition("::")
    return Chunk(
        clause_id=clause_id,
        doc_id=doc_id,
        doc_title=doc_id.replace("_", " ").title(),
        clause_ref=ref,
        title=f"title of {ref}",
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
        text="Cancellation fees are waived for Northstar on booked shipments.",
        topics=("cancellation_fee",),
    ),
    chunk(
        LUMENWORKS,
        tier=1,
        account_id="ACCT-002",
        text="No special cancellation fee waiver applies. Use the current SOP.",
        topics=("cancellation_fee",),
    ),
    chunk(
        SOP_CANCELLATION,
        tier=2,
        account_id=None,
        text="Cancellation within 30 minutes is free; after that a fee of INR 250 applies.",
        topics=("cancellation_fee", "cancellation_window"),
    ),
    chunk(
        POLICY_V2,
        tier=4,
        account_id=None,
        text="Deprecated first response targets for every plan.",
        topics=("first_response_target",),
    ),
)


@pytest.fixture
def store(tmp_path):
    built = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=tmp_path / "index")
    built.upsert(CORPUS)
    return built


def ids(chunks) -> set[str]:
    return {c.clause_id for c in chunks}


class TestTheStoreOwnsAccessControl:
    def test_a_customer_never_retrieves_another_accounts_agreement(self, store):
        # The query is worded to make the foreign agreement the best lexical
        # match available, so a store that ranked before filtering would fail.
        found = store.query(
            "cancellation fee waiver", principal=persona("northstar_customer"), k=10
        )
        assert LUMENWORKS not in ids(found)

        found = store.query(
            "cancellation fee waiver", principal=persona("lumenworks_customer"), k=10
        )
        assert NORTHSTAR not in ids(found)

    def test_a_customer_sees_its_own_agreement_and_general_policy(self, store):
        found = ids(store.query("cancellation fee", principal=persona("northstar_customer"), k=10))
        assert NORTHSTAR in found
        assert SOP_CANCELLATION in found

    def test_an_account_with_no_agreement_sees_only_unscoped_clauses(self, store):
        found = store.query("cancellation fee", principal=persona("beacon_customer"), k=10)
        assert all(c.account_id is None for c in found)

    def test_staff_see_every_account(self, store):
        found = ids(store.query("cancellation fee waiver", principal=persona("maya_agent"), k=10))
        assert {NORTHSTAR, LUMENWORKS, SOP_CANCELLATION} <= found

    def test_the_caller_cannot_supply_a_filter(self):
        # If a `where` or `filter` argument existed, the ACL would be advice
        # rather than enforcement: one caller forgetting it leaks a contract.
        parameters = set(inspect.signature(ChromaLocalStore.query).parameters)
        assert not parameters & {"where", "filter", "metadata_filter", "account_id"}

    def test_the_principal_is_required_and_keyword_only(self):
        parameter = inspect.signature(ChromaLocalStore.query).parameters["principal"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    def test_a_malformed_account_id_is_refused_rather_than_interpolated(self, store):
        # Defence in depth, mirroring the repository. A Principal can only be
        # built through `build_principal`, so this is unreachable in normal
        # use - which is exactly why it should fail loudly if it is reached.
        rogue = build_principal("rogue", "customer", account_id="ACCT-001")
        object.__setattr__(rogue, "account_id", "ACCT-001' OR '1'='1")
        with pytest.raises(ValueError, match="account_id"):
            store.query("cancellation", principal=rogue, k=5)


class TestTierScoping:
    def test_deprecated_policy_is_absent_from_a_default_query(self, store):
        found = store.query("first response target", principal=persona("maya_agent"), k=10)
        assert POLICY_V2 not in ids(found)

    def test_deprecated_policy_is_reachable_when_asked_for_deliberately(self, store):
        # "What changed between v2 and v3?" is a legitimate question, so the
        # tier-4 clause is excluded by a predicate rather than never indexed.
        found = store.query(
            "first response target", principal=persona("maya_agent"), tiers={4}, k=10
        )
        assert POLICY_V2 in ids(found)


class TestCollectionIdentity:
    def test_the_collection_is_namespaced_by_embedding_identity(self, tmp_path):
        first = ChromaLocalStore(
            embeddings=HashingEmbeddings(tag="model-a"), persist_dir=tmp_path / "index"
        )
        second = ChromaLocalStore(
            embeddings=HashingEmbeddings(tag="model-b"), persist_dir=tmp_path / "index"
        )
        assert first.collection_name != second.collection_name

    def test_switching_identity_selects_an_empty_collection_not_stale_vectors(self, tmp_path):
        first = ChromaLocalStore(
            embeddings=HashingEmbeddings(tag="model-a"), persist_dir=tmp_path / "index"
        )
        first.upsert(CORPUS)
        second = ChromaLocalStore(
            embeddings=HashingEmbeddings(tag="model-b"), persist_dir=tmp_path / "index"
        )
        # The alternative - one shared collection - would compare vectors from
        # two different spaces and return confident nonsense.
        assert first.count() == len(CORPUS)
        assert second.count() == 0

    def test_a_dimension_change_alone_changes_the_collection(self, tmp_path):
        narrow = ChromaLocalStore(
            embeddings=HashingEmbeddings(dimensions=64), persist_dir=tmp_path / "index"
        )
        wide = ChromaLocalStore(
            embeddings=HashingEmbeddings(dimensions=128), persist_dir=tmp_path / "index"
        )
        assert narrow.collection_name != wide.collection_name


class TestIndexing:
    def test_upsert_is_idempotent(self, tmp_path):
        built = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=tmp_path / "index")
        assert built.upsert(CORPUS) == built.upsert(CORPUS) == len(CORPUS)
        assert built.count() == len(CORPUS)

    def test_a_removed_clause_does_not_survive_a_rebuild(self, tmp_path):
        built = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=tmp_path / "index")
        built.upsert(CORPUS)
        built.upsert(CORPUS[:2])
        # A stale vector is worse than a missing one: it is citable, and the
        # clause it points at no longer exists in the registry.
        assert built.count() == 2

    def test_documents_are_embedded_in_one_batch(self, tmp_path):
        embeddings = HashingEmbeddings()
        built = ChromaLocalStore(embeddings=embeddings, persist_dir=tmp_path / "index")
        built.upsert(CORPUS)
        assert embeddings.calls == 1

    def test_an_unscoped_clause_is_stored_under_the_global_sentinel(self, store):
        # Chroma metadata cannot hold NULL, so "applies to everyone" needs a
        # value. It must not round-trip back as a literal account id.
        found = store.query(
            "cancellation within 30 minutes", principal=persona("beacon_customer"), k=5
        )
        assert found and all(c.account_id is None for c in found)
        assert GLOBAL_SCOPE not in {c.account_id for c in found}

    def test_a_chunk_carries_everything_a_citation_needs(self, store):
        # Selected by id rather than taken from the top, because what is under
        # test is the metadata round-trip, not the ranking. The stub embedding
        # does not stem, so "fees"/"waived" miss "fee"/"waiver" and the SOP
        # clause outranks the agreement on this query.
        found = store.query(
            "cancellation fee waiver", principal=persona("northstar_customer"), k=10
        )
        cited = next(c for c in found if c.clause_id == NORTHSTAR)
        assert cited.doc_title and cited.clause_ref and cited.title
        assert cited.tier == 1
        assert cited.topics == ("cancellation_fee",)
        assert "Northstar" in cited.text

    def test_results_are_ordered_by_descending_score(self, store):
        found = store.query("cancellation fee", principal=persona("maya_agent"), k=10)
        assert [c.score for c in found] == sorted((c.score for c in found), reverse=True)

    def test_k_bounds_the_result_count(self, store):
        assert len(store.query("cancellation", principal=persona("maya_agent"), k=2)) <= 2


class TestProtocol:
    def test_the_local_store_satisfies_the_protocol(self, store):
        assert isinstance(store, VectorStore)


class TestFilterConstruction:
    """Regression: Chroma rejects `$and`/`$or` with fewer than two operands.

    Found by the end-to-end test, not by the unit tests above, because none of
    them passed a topic through to the dense path. Every topic-scoped query
    with exactly one topic raised - which is the single most common shape the
    resolver will issue.
    """

    def test_a_single_topic_filter_is_accepted(self, store):
        found = store.query(
            "cancellation", principal=persona("maya_agent"), topics=["cancellation_fee"], k=5
        )
        assert ids(found) == {NORTHSTAR, LUMENWORKS, SOP_CANCELLATION}

    def test_several_topics_are_accepted(self, store):
        found = store.query(
            "cancellation",
            principal=persona("maya_agent"),
            topics=["cancellation_fee", "cancellation_window"],
            k=10,
        )
        assert SOP_CANCELLATION in ids(found)

    def test_a_single_topic_for_a_customer_combines_with_the_acl(self, store):
        found = store.query(
            "cancellation",
            principal=persona("northstar_customer"),
            topics=["cancellation_fee"],
            k=10,
        )
        assert ids(found) == {NORTHSTAR, SOP_CANCELLATION}

    def test_a_topic_nothing_carries_returns_nothing(self, store):
        assert (
            store.query("anything", principal=persona("maya_agent"), topics=["credit_cap"], k=5)
            == []
        )

    def test_an_empty_tier_set_is_refused_rather_than_matching_everything(self, store):
        with pytest.raises(ValueError, match="tier"):
            store.query("cancellation", principal=persona("maya_agent"), tiers=set(), k=5)
