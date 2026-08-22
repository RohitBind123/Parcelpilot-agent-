"""The whole pipeline, from the supplied files to retrieved evidence.

This file grows with each milestone. Today it covers M1 to M4: the workbook and
the six PDFs are built into SQLite, the registry is indexed into a real Chroma
collection and a real BM25 index, each persona asks a real question through the
real retriever, and the whole answering chain - snapshot, resolve, compute,
cross-check - runs on a database built in this test rather than the committed
one, so a stale artefact cannot make it pass.

Nothing here is mocked except the embedding model, which is a deterministic
hashing stand-in so the suite runs offline (D20). Every other layer is the one
that ships: the same ETL, the same parser, the same store, the same fusion.

What it is checking is the property the unit tests can only check in pieces -
that a customer's question reaches its own contract and the general policy, and
never reaches another customer's contract, through a stack where the ACL is
enforced in three separate places.
"""

from __future__ import annotations

import pytest

from src.auth.personas import get_persona, to_principal
from src.datastore.etl import WORKBOOK_PATH, build_database
from src.datastore.repo import AccessDenied, open_repository
from src.domain.calculators.cancellation import compute_cancellation_fee
from src.domain.consistency import ConflictClass, ConflictSeverity, ConsistencyChecker
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import deterministic_severity, infer_severity, load_severity_definitions
from src.knowledge.ingest import build_registry
from src.knowledge.registry import load_chunks
from src.knowledge.retriever import BM25Index, HybridRetriever
from src.knowledge.vectorstore.base import CITABLE_TIERS, Chunk
from src.knowledge.vectorstore.chroma import ChromaLocalStore
from tests.support.embeddings import HashingEmbeddings

NORTHSTAR_CANCELLATION = "northstar_logistics_enterprise_agreement::§2"
LUMENWORKS_CANCELLATION = "lumenworks_service_agreement::§2"
LUMENWORKS_CREDIT = "lumenworks_service_agreement::§3"
SOP_CANCELLATION = "cancellation_and_service_credit_sop_v4::§1"
SOP_CREDIT = "cancellation_and_service_credit_sop_v4::§2"
POLICY_V2 = "support_policy_v2_deprecated::§-"

CUSTOMERS = ("northstar_customer", "lumenworks_customer", "beacon_customer", "axis_customer")
AGREEMENT_OWNER = {"ACCT-001": "northstar", "ACCT-002": "lumenworks"}


def persona(persona_id: str):
    return to_principal(get_persona(persona_id))


def ids(chunks) -> list[str]:
    return [c.clause_id for c in chunks]


@pytest.fixture(scope="module")
def pipeline(
    tmp_path_factory,
):
    """Build everything from the supplied files, once."""
    root = tmp_path_factory.mktemp("e2e")
    db_path = root / "parcelpilot.db"

    build_database(WORKBOOK_PATH, db_path)
    clause_count = build_registry(db_path)

    chunks = load_chunks(db_path)
    assert len(chunks) == clause_count

    dense = ChromaLocalStore(embeddings=HashingEmbeddings(), persist_dir=root / "index")
    dense.upsert(chunks)

    return {
        "db_path": db_path,
        "chunks": chunks,
        "dense": dense,
        "retriever": HybridRetriever(dense=dense, lexical=BM25Index(chunks)),
    }


@pytest.fixture(scope="module")
def retriever(pipeline) -> HybridRetriever:
    return pipeline["retriever"]


class TestTheBuildProducesWhatTheCorpusContains:
    def test_the_structured_and_unstructured_halves_land_in_one_database(self, pipeline):
        # The resolver needs an order and the clauses that govern it in a
        # single query, so they cannot live in two files.
        with open_repository(persona("priya_manager"), pipeline["db_path"]) as repo:
            assert repo.get_order("ORD-1001").account_id == "ACCT-001"
        assert len(pipeline["chunks"]) == 19

    def test_every_clause_is_indexed(self, pipeline):
        assert pipeline["dense"].count() == len(pipeline["chunks"])

    def test_every_indexed_clause_carries_at_least_one_topic(self, pipeline):
        # An untagged clause is unreachable by topic-scoped retrieval, which is
        # how the resolver asks for candidates.
        assert all(chunk.topics for chunk in pipeline["chunks"])


class TestTheDiscriminatingPair:
    """ORD-1001 and ORD-2001 are both BOOKED and both past the free window.

    Everything that makes their answers differ has to be visible in what
    retrieval returns, or the model is being asked to guess.
    """

    def test_northstar_reaches_its_own_waiver_and_the_general_sop(self, retriever):
        found = ids(
            retriever.retrieve(
                "am I charged a fee for cancelling this booking?",
                principal=persona("northstar_customer"),
                topics=["cancellation_fee"],
                k=8,
            )
        )
        assert NORTHSTAR_CANCELLATION in found
        assert SOP_CANCELLATION in found
        assert LUMENWORKS_CANCELLATION not in found

    def test_lumenworks_reaches_its_own_clause_and_the_general_sop(self, retriever):
        found = ids(
            retriever.retrieve(
                "am I charged a fee for cancelling this booking?",
                principal=persona("lumenworks_customer"),
                topics=["cancellation_fee"],
                k=8,
            )
        )
        # LumenWorks §2 is Tier 1 and declines to override. Retrieval must
        # surface it anyway: "an agreement exists and says use the SOP" is a
        # different answer from "no agreement exists", and the resolver cannot
        # tell them apart if the clause never arrives.
        assert LUMENWORKS_CANCELLATION in found
        assert SOP_CANCELLATION in found
        assert NORTHSTAR_CANCELLATION not in found

    def test_the_credit_override_is_reachable_only_by_its_own_account(self, retriever):
        lumenworks = ids(
            retriever.retrieve(
                "failed pickup service credit",
                principal=persona("lumenworks_customer"),
                topics=["failed_pickup_credit"],
                k=8,
            )
        )
        assert LUMENWORKS_CREDIT in lumenworks

        beacon = ids(
            retriever.retrieve(
                "failed pickup service credit",
                principal=persona("beacon_customer"),
                topics=["failed_pickup_credit"],
                k=8,
            )
        )
        assert LUMENWORKS_CREDIT not in beacon
        assert SOP_CREDIT in beacon


class TestAccountScopingHoldsAcrossTheCorpus:
    @pytest.mark.parametrize("persona_id", CUSTOMERS)
    def test_no_customer_ever_retrieves_a_foreign_agreement(self, retriever, pipeline, persona_id):
        principal = persona(persona_id)
        queries = (
            "cancellation fee waiver",
            "service credit for a failed pickup",
            "first response time for a P1 issue",
            "weekend support coverage",
            "bulk upload limit",
            "enterprise agreement",
        )
        for query in queries:
            for chunk in retriever.retrieve(query, principal=principal, k=12):
                assert chunk.account_id in (None, principal.account_id), (
                    f"{persona_id} retrieved {chunk.clause_id} scoped to {chunk.account_id}"
                )

    @pytest.mark.parametrize("persona_id", CUSTOMERS)
    def test_a_customer_naming_another_company_still_gets_nothing(self, retriever, persona_id):
        # The adversarial version: the query is the foreign account's own name,
        # which is the strongest possible lexical signal toward its contract.
        principal = persona(persona_id)
        for company in ("Northstar Logistics enterprise agreement", "LumenWorks service agreement"):
            for chunk in retriever.retrieve(company, principal=principal, k=12):
                assert chunk.account_id in (None, principal.account_id)

    def test_staff_read_every_account(self, retriever):
        found = ids(
            retriever.retrieve("cancellation fee waiver", principal=persona("maya_agent"), k=12)
        )
        assert NORTHSTAR_CANCELLATION in found
        assert LUMENWORKS_CANCELLATION in found

    def test_retrieval_scoping_agrees_with_repository_scoping(self, pipeline, retriever):
        # Two independent enforcement points, one answer. If they disagreed,
        # a customer could see a clause about an order they cannot read.
        principal = persona("beacon_customer")
        with open_repository(principal, pipeline["db_path"]) as repo, pytest.raises(AccessDenied):
            repo.get_order("ORD-1001")
        found = retriever.retrieve("cancellation fee", principal=principal, k=12)
        assert all(c.account_id is None for c in found)


class TestTierDiscipline:
    def test_the_deprecated_policy_is_indexed_but_never_citable_by_default(
        self, pipeline, retriever
    ):
        assert any(c.clause_id == POLICY_V2 for c in pipeline["chunks"])

        for persona_id in (*CUSTOMERS, "maya_agent", "priya_manager"):
            found = retriever.retrieve(
                "first response target for a P1 issue",
                principal=persona(persona_id),
                k=12,
            )
            assert POLICY_V2 not in ids(found)
            assert all(chunk.is_citable for chunk in found)

    def test_the_deprecated_policy_is_reachable_when_named(self, retriever):
        # "What changed between v2 and v3?" is a real support question. The
        # clause is excluded by a predicate, not withheld from the index.
        found = retriever.retrieve(
            "deprecated support policy v2 response targets",
            principal=persona("priya_manager"),
            tiers={4},
            k=8,
        )
        assert POLICY_V2 in ids(found)

    def test_every_default_result_is_within_the_citable_tiers(self, retriever):
        found = retriever.retrieve("service credit", principal=persona("maya_agent"), k=12)
        assert {chunk.tier for chunk in found} <= CITABLE_TIERS


class TestLookupQueries:
    def test_an_exact_clause_reference_resolves(self, retriever):
        found = retriever.retrieve("SOP v4 §1", principal=persona("beacon_customer"), k=5)
        assert found[0].clause_id == SOP_CANCELLATION

    def test_a_known_issue_is_findable_by_its_identifier(self, retriever):
        found = retriever.retrieve("KI-208", principal=persona("maya_agent"), k=5)
        assert any("KI-208" in chunk.clause_id for chunk in found)

    def test_retrieved_chunks_can_be_rendered_as_citations(self, retriever):
        found = retriever.retrieve("cancellation fee", principal=persona("northstar_customer"), k=5)
        assert found
        for chunk in found:
            assert isinstance(chunk, Chunk)
            assert chunk.citation.strip()
            assert chunk.text.strip()


class TestTheAnsweringChainOnAFreshlyBuiltDatabase:
    """M3 and M4 end to end, on the database this test built.

    The unit suites run against the committed `parcelpilot.db`. That is the
    right default - it is what ships - but it means a build regression can hide
    behind an artefact that was correct when it was generated. Here the PDFs are
    parsed and the workbook is read in this process, and the same answers have
    to come out.
    """

    def test_the_discriminating_pair_survives_a_rebuild(self, pipeline):
        # INR 0 for Northstar, INR 250 for LumenWorks, from two orders that are
        # identical in every field the calculator reads except the account.
        assert _fee(pipeline, "northstar_customer", "ORD-1001") == (0.0, NORTHSTAR_CANCELLATION)
        assert _fee(pipeline, "lumenworks_customer", "ORD-2001") == (250.0, SOP_CANCELLATION)

    def test_the_staleness_conflict_is_found_through_the_whole_stack(self, pipeline):
        report = _consistency(pipeline, "northstar_customer", "ORD-1001")
        conflict = next(
            c for c in report.conflicts if c.conflict_class is ConflictClass.STALE_STATUS
        )
        assert report.blocking is True
        assert conflict.severity is ConflictSeverity.BLOCKING
        # All three sources, and the ticket link marked as inferred (A3).
        assert {"ORD-1001", "TKT-504"} <= set(conflict.sources)
        assert conflict.inference_note is not None

    def test_the_same_shaped_order_without_a_witness_is_clean(self, pipeline):
        # ORD-2001 is BOOKED, SwiftShip, no pickup timestamp - and nobody has
        # said a driver came. If this ever reports a conflict, every answer the
        # system gives starts arriving hedged.
        assert _consistency(pipeline, "lumenworks_customer", "ORD-2001").conflicts == ()

    def test_both_historical_contradictions_are_caught(self, pipeline):
        for persona_id, ticket_id, claimed, current in (
            ("maya_agent", "TKT-450", 250, 0),
            ("rohit_agent", "TKT-451", 3000, 5000),
        ):
            report = _consistency(pipeline, persona_id, ticket_id)
            conflict = next(
                c
                for c in report.conflicts
                if c.conflict_class is ConflictClass.HISTORICAL_CONTRADICTION
            )
            assert (conflict.claimed_value, conflict.current_value) == (claimed, current)
            # Advisory: a past answer was wrong, but nothing current is in doubt.
            assert report.blocking is False

    def test_the_p1_guards_fire_on_exactly_the_two_tickets_they_should(self, pipeline):
        with open_repository(persona("priya_manager"), pipeline["db_path"]) as repo:
            fired = {
                ticket.ticket_id
                for ticket in repo.query_tickets(status="open")
                if deterministic_severity(ticket.subject, ticket.description or "")
            }
        assert fired == {"TKT-501", "TKT-505"}

    def test_severity_definitions_come_from_the_rebuilt_registry(self, pipeline):
        import sqlite3

        connection = sqlite3.connect(pipeline["db_path"])
        try:
            definitions = load_severity_definitions(connection)
        finally:
            connection.close()
        assert definitions.clause_id == "support_policy_v3_current::§2"
        assert definitions.clause_id != POLICY_V2

    def test_an_unreachable_classifier_leaves_severity_undetermined_not_low(self, pipeline):
        # The deployment where the model is down must not silently grade every
        # ticket P3. It must say it does not know.
        import sqlite3

        connection = sqlite3.connect(pipeline["db_path"])
        try:
            definitions = load_severity_definitions(connection)
        finally:
            connection.close()
        verdict = infer_severity(
            "Bulk upload fails for 4,200-row CSV",
            "The CSV reaches roughly 70% and fails.",
            definitions=definitions,
            classifier=None,
        )
        assert verdict.severity is None
        assert not verdict.is_trusted

    def test_the_bulk_upload_limit_has_a_governing_clause_at_all(self, pipeline):
        # Regression: a defect report and the plan capability are both Tier 3
        # on this topic, and the resolver used to call them an unresolved
        # conflict - leaving the corpus's plainest statement unanswerable.
        with PolicyResolver.open(pipeline["db_path"]) as resolver:
            resolution = resolver.resolve("bulk_upload_limit", persona("lumenworks_customer"))
        assert resolution.governing.params["supported_rows"] == 5000
        assert any(c.clause_id.endswith("KI-208") for c in resolution.supporting)


def _fee(pipeline, persona_id: str, order_id: str) -> tuple[float | None, str]:
    principal = persona(persona_id)
    with open_evidence_store(run_id="e2e", principal=principal) as store:
        with open_repository(principal, pipeline["db_path"]) as repo:
            snapshot = store.mint(
                EvidenceKind.ORDER_SNAPSHOT, repo.get_order(order_id).to_payload()
            )
        with PolicyResolver.open(pipeline["db_path"]) as resolver:
            resolution = resolver.resolve("cancellation_fee", principal)
        handle = store.mint(
            EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
        )
        outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=handle)
    return outcome.fee_inr, outcome.governing_clause


def _consistency(pipeline, persona_id: str, subject: str):
    principal = persona(persona_id)
    with (
        open_evidence_store(run_id="e2e", principal=principal) as store,
        open_repository(principal, pipeline["db_path"]) as repo,
    ):
        if subject.startswith("ORD-"):
            kind = EvidenceKind.ORDER_SNAPSHOT
            payload = repo.get_order(subject).to_payload()
        else:
            kind = EvidenceKind.TICKET_SNAPSHOT
            payload = repo.get_ticket(subject).to_payload()
        snapshot = store.mint(kind, payload)
        checker = ConsistencyChecker(
            store=store, repository=repo, resolver=PolicyResolver(repo.connection)
        )
        return checker.check(snapshot_id=snapshot)
