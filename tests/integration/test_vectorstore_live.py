"""The hosted path, against real Chroma Cloud and a real embedding model.

Deselected by default. Everything else in the suite runs against local Chroma
with a stub embedding, which proves the logic but not the parts that only fail
in the presence of a network: the tenant, the collection, the real vector
width, and whether a metadata filter behaves the same server-side as it does
in the embedded engine.

    uv run pytest -m live tests/integration/test_vectorstore_live.py

Needs CHROMA_API_KEY, CHROMA_TENANT and the selected embedding provider's key.
The collection is namespaced by embedding identity and rebuilt wholesale, so
running this replaces the demo index rather than corrupting it.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.knowledge.registry import load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.chroma import ChromaCloudStore, VectorStoreError
from src.providers.registry import get_embedding_provider

pytestmark = pytest.mark.live

NORTHSTAR = "northstar_logistics_enterprise_agreement::§2"
LUMENWORKS = "lumenworks_service_agreement::§2"
SOP_CANCELLATION = "cancellation_and_service_credit_sop_v4::§1"


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


@pytest.fixture(scope="module")
def cloud():
    settings = get_settings()
    if not (settings.chroma_api_key and settings.chroma_tenant):
        pytest.skip("CHROMA_API_KEY / CHROMA_TENANT not configured")

    embeddings = get_embedding_provider(settings)
    store = ChromaCloudStore(
        embeddings=embeddings,
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )
    chunks = load_chunks(settings.db_path)
    store.upsert(chunks)
    return {"store": store, "chunks": chunks, "embeddings": embeddings}


class _UnusedEmbeddings:
    """An embedding identity nothing has ever been indexed under.

    Both methods raise: an absent collection has to be discovered before any
    text is embedded, and this fixture must never write to the real tenant.
    """

    identity = "nobody/never-indexed-model/999"

    def embed_query(self, text):
        raise AssertionError("an absent collection must be found before embedding")

    def embed_documents(self, texts):
        raise AssertionError("this fixture must never write")


@pytest.fixture(scope="module")
def never_indexed():
    settings = get_settings()
    if not (settings.chroma_api_key and settings.chroma_tenant):
        pytest.skip("CHROMA_API_KEY / CHROMA_TENANT not configured")
    return ChromaCloudStore(
        embeddings=_UnusedEmbeddings(),
        api_key=settings.chroma_api_key,
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
    )


class TestTheHostedAbsentCollectionContract:
    """That the Cloud client reports a missing collection as `NotFoundError`.

    `_find_collection` turns exactly one exception into "not built", and every
    other exception is allowed to propagate as a real fault. That split is only
    as good as the error type, and on the hosted client the type is not a
    language-level guarantee: `BaseHTTPClient._raise_chroma_error` promotes an
    HTTP failure to a typed error only when the response body is JSON shaped
    like `{"error": "NotFoundError", ...}`, which is the OSS reference server's
    envelope. A gateway 404, a proxy error page, or a control plane that words
    its JSON differently all fall through to a bare `Exception`.

    Measured against the real tenant, Cloud does raise `NotFoundError` today.
    This test is here so that stays a fact rather than something I remember
    checking once. **If it starts failing, `_find_collection` is silently
    letting a missing collection propagate as a fault** - `count()` would raise
    where it promises not to, and `query()` would lose the specific
    `VectorStoreError` in favour of whatever the transport threw.

    No collection is created: `get_collection` is a read, and the identity is
    one that has never been indexed.
    """

    def test_the_hosted_client_raises_the_typed_not_found_error(self, never_indexed):
        from chromadb.errors import NotFoundError

        with pytest.raises(NotFoundError):
            never_indexed._client.get_collection(name=never_indexed.collection_name)

    def test_count_is_zero_rather_than_raising(self, never_indexed):
        # The contract `count()` advertises, exercised against the transport
        # that could break it.
        assert never_indexed.count() == 0

    def test_query_raises_the_specific_error_and_names_the_identity(self, never_indexed):
        with pytest.raises(VectorStoreError) as caught:
            never_indexed.query("cancellation fee", principal=persona("maya_agent"))
        assert never_indexed.collection_name in str(caught.value)


class TestHostedStore:
    def test_the_collection_holds_the_whole_registry(self, cloud):
        assert cloud["store"].count() == len(cloud["chunks"])

    def test_the_collection_is_named_for_the_embedding_identity(self, cloud):
        assert cloud["embeddings"].identity.split("/")[1] in cloud["store"].collection_name

    def test_vectors_have_the_configured_width(self, cloud):
        # A silently truncated or padded vector still stores and still queries;
        # it just returns nonsense. Worth one assertion against the real model.
        vector = cloud["embeddings"].embed_query("cancellation fee")
        assert len(vector) == get_settings().embedding_config().dimensions

    def test_the_acl_filter_behaves_the_same_server_side(self, cloud):
        # The point of this test. Local Chroma runs the predicate in-process;
        # Cloud runs it on their servers. Same predicate, different engine.
        found = {
            c.clause_id
            for c in cloud["store"].query(
                "cancellation fee waiver", principal=persona("northstar_customer"), k=10
            )
        }
        assert NORTHSTAR in found
        assert LUMENWORKS not in found

    def test_a_single_topic_filter_works_against_the_hosted_engine(self, cloud):
        found = cloud["store"].query(
            "cancellation",
            principal=persona("maya_agent"),
            topics=["cancellation_fee"],
            k=10,
        )
        assert found
        assert all("cancellation_fee" in c.topics for c in found)


class TestRealEmbeddingsRankSensibly:
    """The stub embedding cannot show whether the model understands the corpus.

    These are the only assertions in the suite that depend on a trained model,
    and they are kept to what a competent embedding must get right rather than
    to an exact ordering, which would be a hosted-model version bump away from
    failing for no real reason.
    """

    def test_a_paraphrased_question_finds_the_clause_that_answers_it(self, cloud):
        retriever = HybridRetriever(dense=cloud["store"], lexical=BM25Index(cloud["chunks"]))
        found = [
            c.clause_id
            for c in retriever.retrieve(
                "if I call off a pickup soon after booking, does it cost anything?",
                principal=persona("beacon_customer"),
                k=5,
            )
        ]
        # Nothing in that sentence shares vocabulary with the clause, so BM25
        # alone would miss it. This is what the dense half is buying.
        assert SOP_CANCELLATION in found

    def test_the_account_override_outranks_general_policy_for_its_owner(self, cloud):
        retriever = HybridRetriever(dense=cloud["store"], lexical=BM25Index(cloud["chunks"]))
        found = [
            c.clause_id
            for c in retriever.retrieve(
                "are cancellation fees waived for us?",
                principal=persona("northstar_customer"),
                topics=["cancellation_fee"],
                k=5,
            )
        ]
        assert NORTHSTAR in found
