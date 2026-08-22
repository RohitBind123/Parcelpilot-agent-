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
from src.knowledge.vectorstore.chroma import ChromaCloudStore
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
