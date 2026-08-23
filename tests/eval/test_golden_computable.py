"""The golden set, run through the real chain (D21 layer 2).

Every entry M3 can answer is driven from `golden_set.yaml` rather than restated
here, so the expectations in this file are the ones that were signed off. If a
verdict changes there, this fails - which is the whole reason the golden set was
written before the resolver.

Thirty-one of the thirty-two are computable today. The last one needs proactive
detection (M10), and the coverage test at the bottom asserts that the uncovered
set is exactly that and shrinks as milestones land - so this file cannot quietly
stop testing something.

Nothing is mocked. The database is the committed one, the clauses are the real
registry, and the arithmetic is the arithmetic that will ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.agent.answer import assemble
from src.agent.escalation import DeclineReason
from src.agent.escalation import draft as escalation_draft
from src.agent.facts import compose
from src.agent.grounding import check_figures
from src.agent.tools.base import ToolDenied
from src.agent.tools.context import open_tool_context
from src.agent.tools.registry import PROJECTION, build_toolset
from src.auth.personas import get_persona, to_principal
from src.config import get_settings
from src.datastore.repo import open_repository
from src.domain.calculators.cancellation import compute_cancellation_fee
from src.domain.calculators.credit import compute_service_credit
from src.domain.calculators.sla import sla_first_response_status
from src.domain.consistency import ConflictClass, ConsistencyChecker
from src.domain.evidence import EvidenceKind, open_evidence_store
from src.domain.resolver import PolicyResolver
from src.domain.severity import SeverityVerdict, deterministic_severity

GOLDEN = Path(__file__).resolve().parent / "golden_set.yaml"
POLICY_V2 = "support_policy_v2_deprecated::§-"

CANCELLATION = ["GS-001", "GS-002", "GS-003", "GS-004", "GS-005", "GS-006"]
CREDIT = ["GS-007", "GS-008", "GS-009", "GS-010"]
SLA = ["GS-011", "GS-012", "GS-013", "GS-014", "GS-015"]
CONFLICT = ["GS-019", "GS-020", "GS-021"]
#: Entries whose acceptance is a property of the tool layer: which tools exist,
#: what they refuse, and what a refusal is allowed to say.
#: Entries whose acceptance is a property of composition: what the fact block
#: renders, and what the gate refuses to let through.
COMPOSITION = ["GS-017", "GS-024", "GS-025"]
TOOLS = [
    "GS-016",
    "GS-018",
    "GS-022",
    "GS-023",
    "GS-026",
    "GS-027",
    "GS-028",
    "GS-029",
    "GS-030",
    "GS-032",
]

#: Entries whose acceptance needs a milestone that has not landed. Named
#: individually so the list is a to-do rather than a shrug.
NOT_YET_COMPUTABLE = {
    # M10 built `scan_support_health`, so this is computable now. The assertion
    # belongs with the rest of the evaluation work in M11 rather than here.
    "GS-031": "ops detection; tool built in M10, asserted with the eval layer (M11)",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, dict]:
    return {e["id"]: e for e in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def db_path():
    return get_settings().db_path


@pytest.fixture(scope="module")
def index(tmp_path_factory, db_path):
    """A real hybrid retriever over the committed registry.

    Offline: the embeddings are the deterministic hashing stand-in, so the
    lexical half does the work the dense half would. What is under test in
    these entries is the ACL predicate and the tier filter, and both sit
    underneath ranking.
    """
    from src.knowledge.registry import load_chunks
    from src.knowledge.retriever import BM25Index, HybridRetriever
    from src.knowledge.vectorstore.chroma import ChromaLocalStore
    from tests.support.embeddings import HashingEmbeddings

    chunks = load_chunks(db_path)
    dense = ChromaLocalStore(
        embeddings=HashingEmbeddings(), persist_dir=tmp_path_factory.mktemp("golden") / "index"
    )
    dense.upsert(chunks)
    return HybridRetriever(dense=dense, lexical=BM25Index(chunks))


@pytest.fixture
def toolset(db_path, index):
    """A live toolset per persona, over a context scoped to that persona."""
    opened = []

    def build(persona_id: str):
        cm = open_tool_context(
            to_principal(get_persona(persona_id)),
            run_id=persona_id,
            db_path=db_path,
            retriever=index,
        )
        context = cm.__enter__()
        opened.append(cm)
        tools = build_toolset(context)
        return _Toolset(tools)

    yield build
    for cm in opened:
        cm.__exit__(None, None, None)


class _Toolset:
    """Indexable by name, iterable for the absence assertions."""

    def __init__(self, tools):
        self._by_name = {t.name: t for t in tools}

    def __getitem__(self, name):
        return self._by_name[name]

    def __iter__(self):
        return iter(self._by_name.values())


def principal_for(entry: dict):
    return to_principal(get_persona(entry["persona"]))


def scope_for(entry: dict, principal):
    """Which account the question is about. Staff must say; customers cannot."""
    return None if principal.account_id else _subject_account(entry)


def _subject_account(entry: dict) -> str:
    return {
        "ORD-1001": "ACCT-001",
        "ORD-1002": "ACCT-001",
        "ORD-2001": "ACCT-002",
        "ORD-2002": "ACCT-002",
        "ORD-3001": "ACCT-003",
        "ORD-4001": "ACCT-004",
        "TKT-501": "ACCT-001",
        "TKT-502": "ACCT-002",
        "TKT-503": "ACCT-003",
        "TKT-504": "ACCT-001",
        "TKT-505": "ACCT-004",
        "TKT-450": "ACCT-001",
        "TKT-451": "ACCT-002",
    }[entry["subject"]]


class TestCancellation:
    @pytest.mark.parametrize("entry_id", CANCELLATION)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        principal = principal_for(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            with open_repository(principal, db_path) as repo:
                order = repo.get_order(entry["subject"])
            snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, order.to_payload())
            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "cancellation_fee", principal, account_id=scope_for(entry, principal)
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
            )
            outcome = compute_cancellation_fee(store, snapshot_id=snapshot, resolution_id=handle)

        assert outcome.fee_inr == expect["amount_inr"], entry["derivation"]
        assert outcome.governing_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        for forbidden in expect.get("must_not_cite", []):
            assert forbidden != outcome.governing_clause
            assert forbidden not in outcome.overridden_clauses

    def test_the_pair_is_genuinely_discriminating(self, golden):
        # Guards the parametrised test above from passing on a system that
        # hard-codes one of the two.
        assert (
            golden["GS-001"]["check"]["order_status"] == golden["GS-002"]["check"]["order_status"]
        )
        assert golden["GS-001"]["expect"]["amount_inr"] != golden["GS-002"]["expect"]["amount_inr"]


class TestServiceCredit:
    @pytest.mark.parametrize("entry_id", CREDIT)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        check = entry.get("check") or {}
        principal = principal_for(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            if entry.get("subject"):
                with open_repository(principal, db_path) as repo:
                    payload = repo.get_order(entry["subject"]).to_payload()
                account = _subject_account(entry)
            else:
                # A hypothetical: the question states the delay and the fee, so
                # the snapshot carries exactly what the question asserted and
                # nothing more.
                account = principal.account_id or "ACCT-003"
                payload = {
                    "order_id": None,
                    "account_id": account,
                    "delay_hours": check.get("hypothetical_delay_hours"),
                    "carrier_fault": True,
                    "customer_fault": False,
                    "shipment_fee_inr": check.get("hypothetical_fee_inr"),
                }
            snapshot = store.mint(EvidenceKind.ORDER_SNAPSHOT, payload)

            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "failed_pickup_credit",
                    principal,
                    account_id=None if principal.account_id else account,
                )
                approval = resolver.resolve(
                    "credit_approval",
                    principal,
                    account_id=None if principal.account_id else account,
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[snapshot]
            )
            approval_handle = store.mint(EvidenceKind.POLICY_RESOLUTION, approval.to_payload())
            outcome = compute_service_credit(
                store,
                snapshot_id=snapshot,
                resolution_id=handle,
                approval_resolution_id=approval_handle,
            )

        assert outcome.eligible == (expect["verdict"] == "eligible"), entry["derivation"]
        assert outcome.credit_inr == expect["amount_inr"]
        assert outcome.governing_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        if expect.get("manager_approval_required") is not None:
            assert outcome.requires_manager_approval == expect["manager_approval_required"]

    def test_the_same_question_gets_opposite_answers_by_account(self, golden):
        assert golden["GS-008"]["question"] == golden["GS-009"]["question"]
        assert golden["GS-008"]["expect"]["verdict"] != golden["GS-009"]["expect"]["verdict"]


class TestSlaTargets:
    @pytest.mark.parametrize("entry_id", SLA)
    def test_matches_the_signed_off_answer(self, golden, db_path, entry_id):
        entry = golden[entry_id]
        expect = entry["expect"]
        principal = principal_for(entry)
        account_id = _subject_account(entry)

        with open_evidence_store(run_id=entry_id, principal=principal) as store:
            with open_repository(principal, db_path) as repo:
                ticket = repo.get_ticket(entry["subject"])
                account = repo.get_account(account_id)
            tsnap = store.mint(EvidenceKind.TICKET_SNAPSHOT, ticket.to_payload())
            asnap = store.mint(EvidenceKind.ACCOUNT_SNAPSHOT, account.to_payload())

            with PolicyResolver.open(db_path) as resolver:
                resolution = resolver.resolve(
                    "first_response_target", principal, account_id=account_id
                )
            handle = store.mint(
                EvidenceKind.POLICY_RESOLUTION, resolution.to_payload(), derived_from=[tsnap]
            )

            # The guard where it fires; the signed-off severity otherwise,
            # because inference is M4's job and this is not the test for it.
            verdict = deterministic_severity(
                ticket.subject, ticket.description or ""
            ) or SeverityVerdict(
                severity=expect["severity"],
                confidence=1.0,
                basis_clause="support_policy_v3_current::§2",
                basis="pinned from the reviewed golden set pending M4",
            )
            outcome = sla_first_response_status(
                store,
                snapshot_id=tsnap,
                account_snapshot_id=asnap,
                resolution_id=handle,
                severity=verdict,
                surface="ops",
            )

        assert outcome.severity == expect["severity"], entry["derivation"]
        assert outcome.target == expect["target"]
        assert outcome.due_at == expect["target_due"]
        assert outcome.target_clause == expect["governing"][0]
        assert list(outcome.overridden_clauses) == expect["overridden"]
        assert outcome.measurable is False
        for forbidden in expect.get("must_not_cite", []):
            assert outcome.target_clause != forbidden

    def test_the_two_named_p1_triggers_are_deterministic(self, golden, db_path):
        # GS-011 and GS-015 must never depend on a model sample.
        for entry_id in ("GS-011", "GS-015"):
            entry = golden[entry_id]
            principal = principal_for(entry)
            with open_repository(principal, db_path) as repo:
                ticket = repo.get_ticket(entry["subject"])
            verdict = deterministic_severity(ticket.subject, ticket.description or "")
            assert verdict is not None and verdict.severity == "P1"
            assert verdict.deterministic is True

    def test_the_weekend_actually_moves_a_deadline(self, golden):
        # If AS_OF stopped being a Sunday this suite would still pass while
        # testing nothing interesting, so the property is asserted directly.
        assert golden["GS-012"]["expect"]["target_due"].startswith("2026-08-17")
        assert golden["GS-011"]["expect"]["target_due"].startswith("2026-08-16")


class TestConflicts:
    """GS-019 to GS-021: the three places the pack contradicts itself.

    Each is a trap for a different reflex. GS-019 punishes picking a side when
    the data does not support one. GS-020 punishes repeating a recorded answer
    because it is on file. GS-021 punishes correcting a number without saying
    where the wrong one came from.
    """

    def test_the_stale_status_conflict_is_found_and_not_resolved(self, golden, db_path):
        entry = golden["GS-019"]
        report = _consistency(entry, db_path)
        conflict = _one(report, ConflictClass.STALE_STATUS)

        assert report.blocking is True, entry["derivation"]
        # The clause the answer must follow, and the two records it reconciles.
        assert entry["expect"]["governing"][0] in conflict.sources
        assert "TKT-504" in conflict.sources
        # A3: the ticket names no order. Stating the link as fact is the failure.
        assert conflict.inference_note is not None
        assert conflict.confidence < 1.0

    def test_the_stale_status_finding_does_not_assert_the_parcel_is_still_there(
        self, golden, db_path
    ):
        # KI-211 exists precisely because that statement is often false, and the
        # golden set forbids it in `must_not_assert`.
        #
        # Checked against `detail` alone, and the exclusion is deliberate.
        # KI-211's instruction reads "Before telling a customer that a pickup
        # did not occur, verify the carrier status" - it contains the forbidden
        # sentence as a prohibition of it. A grounding gate in M7 that greps the
        # answer for forbidden phrases will flag the one answer that is right,
        # so the gate has to work on asserted claims, not on substrings.
        entry = golden["GS-019"]
        detail = _one(_consistency(entry, db_path), ConflictClass.STALE_STATUS).detail.lower()
        for forbidden in entry["expect"]["must_not_assert"]:
            assert forbidden.lower() not in detail
        assert "may be stale" in detail

    def test_the_known_issue_instruction_reaches_the_answer(self, golden, db_path):
        conflict = _one(_consistency(golden["GS-019"], db_path), ConflictClass.STALE_STATUS)
        assert "verify the carrier status" in conflict.instruction.lower()

    @pytest.mark.parametrize("entry_id", ["GS-020", "GS-021"])
    def test_a_recorded_answer_that_contradicts_the_current_rule_is_caught(
        self, golden, db_path, entry_id
    ):
        entry = golden[entry_id]
        conflict = _one(_consistency(entry, db_path), ConflictClass.HISTORICAL_CONTRADICTION)

        assert conflict.basis_clause == entry["expect"]["governing"][0], entry["derivation"]
        # Tier 5 is what is being corrected, never the authority for it.
        assert not conflict.basis_clause.startswith("TKT-")
        assert entry["subject"] in conflict.sources

    def test_the_fee_northstar_was_charged_was_waived_all_along(self, golden, db_path):
        conflict = _one(
            _consistency(golden["GS-020"], db_path), ConflictClass.HISTORICAL_CONTRADICTION
        )
        assert conflict.claimed_value == 250
        assert conflict.current_value == golden["GS-020"]["expect"]["amount_inr"] == 0

    def test_the_row_limit_correction_carries_the_defect_that_explains_it(self, golden, db_path):
        # The subtlest entry in the set. 3,000 is real and is not the limit;
        # an answer that only corrects the number sends the customer back to a
        # 3,500-row upload that will fail exactly as before.
        entry = golden["GS-021"]
        conflict = _one(_consistency(entry, db_path), ConflictClass.HISTORICAL_CONTRADICTION)
        assert conflict.claimed_value == entry["check"]["failure_threshold_rows"] == 3000
        assert conflict.current_value == entry["expect"]["supported_rows"] == 5000
        assert set(entry["expect"]["governing"]) <= set(conflict.sources)
        assert "split" in (conflict.instruction or "").lower()

    def test_a_contradiction_advises_and_a_stale_status_blocks(self, golden, db_path):
        # Different conflicts threaten different things: one says a past answer
        # was wrong, the other says a current fact may be.
        assert _consistency(golden["GS-020"], db_path).blocking is False
        assert _consistency(golden["GS-019"], db_path).blocking is True


def _one(report, conflict_class):
    matching = [c for c in report.conflicts if c.conflict_class == conflict_class]
    assert len(matching) == 1, f"expected one {conflict_class}, got {[c.detail for c in matching]}"
    return matching[0]


def _consistency(entry: dict, db_path):
    """The real chain: snapshot, then check, as the tool layer will call it."""
    principal = principal_for(entry)
    subject = entry["subject"]
    with (
        open_evidence_store(run_id=entry["id"], principal=principal) as store,
        open_repository(principal, db_path) as repo,
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


class TestTheToolLayer:
    """GS-016 to GS-032: the entries that turn on which tools a role is given.

    Nine of the ten are answered by the schema rather than by a tool body. That
    is the claim the design makes - containment is a property of what exists,
    not of what refuses - and these are where it is cashed.
    """

    def test_an_abstract_sla_question_needs_no_ticket(self, golden, toolset):
        # GS-016. LumenWorks asks about their guarantee in general, so there is
        # nothing to snapshot; resolve_policy alone answers it.
        entry = golden["GS-016"]
        data = toolset("lumenworks_customer")["resolve_policy"](topic="first_response_target").data
        assert data["governing_clause"] == entry["expect"]["governing"][0], entry["derivation"]
        assert data["overridden"] == entry["expect"]["overridden"]
        assert data["is_override"] is True

    def test_the_deprecated_policy_is_reachable_on_purpose_and_only_by_staff(self, golden, toolset):
        # GS-018. "What changed between v2 and v3?" cannot be answered without
        # reading a superseded document, which is a different act from citing
        # one by accident - so it takes a flag, and the flag is staff-only.
        entry = golden["GS-018"]
        staff = toolset("priya_manager")["search_policy"]
        default = staff(query="first response target", topic="first_response_target").data
        assert POLICY_V2 not in {c["clause_id"] for c in default["clauses"]}

        deliberate = staff(
            query="first response target",
            topic="first_response_target",
            include_deprecated=True,
        ).data
        found = {c["clause_id"]: c for c in deliberate["clauses"]}
        assert POLICY_V2 in found, entry["derivation"]
        # Reachable, and marked as what it is.
        assert found[POLICY_V2]["citable"] is False
        assert found[entry["expect"]["governing"][0]]["citable"] is True

    def test_a_customer_has_no_way_to_ask_for_it(self, golden, toolset):
        params = {p.name for p in _tool(toolset("lumenworks_customer"), "search_policy").params}
        assert "include_deprecated" not in params

    def test_the_bulk_upload_failure_resolves_to_the_known_issue(self, golden, toolset):
        # GS-022. 4,200 rows is inside the supported 5,000 and above KI-208's
        # ~3,000 threshold, so this is the open defect and not a plan limit.
        entry = golden["GS-022"]
        data = toolset("lumenworks_customer")["resolve_policy"](topic="bulk_upload_limit").data
        assert data["governing_clause"] == "product_operations_guide_and_known_issues::§1"
        assert "product_operations_guide_and_known_issues::KI-208" in data["supporting"]

        clauses = toolset("lumenworks_customer")["search_policy"](
            query=entry["question"], topic="bulk_upload_limit"
        ).data["clauses"]
        by_id = {c["clause_id"]: c for c in clauses}
        assert set(entry["expect"]["governing"]) <= set(by_id), entry["derivation"]
        assert (
            entry["expect"]["workaround"]
            in by_id["product_operations_guide_and_known_issues::KI-208"]["text"]
        )

    def test_the_plan_that_lacks_the_feature_gets_the_capability_clause(self, golden, toolset):
        # GS-023. Beacon is on Standard, where Bulk Upload is not included.
        # Relevance is account-scoped, not just topical: explaining why large
        # files fail answers a question this customer cannot yet ask.
        entry = golden["GS-023"]
        data = toolset("beacon_customer")["resolve_policy"](topic="plan_capability").data
        assert data["governing_clause"] == entry["expect"]["governing"][0], entry["derivation"]
        account = toolset("beacon_customer")["get_account"]().data
        assert account["plan"] == entry["check"]["plan"]

    def test_a_cross_account_lookup_is_denied_and_leaks_nothing(self, golden, toolset):
        # GS-026.
        entry = golden["GS-026"]
        result = toolset("lumenworks_customer")["get_order"](order_id=entry["subject"])
        assert isinstance(result, ToolDenied)
        assert result.reason.value == entry["expect"]["reason"]
        rendered = json.dumps(result.to_payload())
        for leaked in entry["expect"]["must_not_leak"]:
            assert leaked not in rendered, entry["derivation"]

    def test_another_accounts_agreement_is_unreachable_by_search(self, golden, toolset):
        # GS-027. Asked for by name, and the predicate is applied server-side
        # inside the store, so naming it changes nothing.
        entry = golden["GS-027"]
        clauses = toolset("beacon_customer")["search_policy"](query=entry["question"]).data[
            "clauses"
        ]
        rendered = json.dumps(clauses)
        for forbidden in entry["expect"]["must_not_cite"]:
            assert forbidden not in rendered, entry["derivation"]
        assert "ACCT-001" not in rendered
        assert not [c for c in clauses if c["clause_id"].startswith("northstar")]
        # And the general rule is still findable, so the answer is not empty.
        assert entry["expect"]["governing"][0] in {c["clause_id"] for c in clauses}

    def test_the_word_waives_is_not_by_itself_a_leak(self, golden, toolset):
        # GS-027 also lists 'waive' and 'waived' under must_not_leak, and at
        # this layer that cannot be a substring check: SOP v4 §1 is general
        # policy every customer may read, and it says "unless a customer
        # agreement explicitly waives the cancellation fee".
        #
        # What must not happen is Beacon being told that *someone else's*
        # agreement waives the fee - a property of the composed answer, not of
        # the retrieved set. Asserted here as the enforceable half, with the
        # rest deferred to the M7 grounding gate. Second instance of the same
        # lesson as GS-019: these expectations are about asserted claims, and a
        # substring filter over evidence flags correct answers.
        clauses = toolset("beacon_customer")["search_policy"](
            query=golden["GS-027"]["question"]
        ).data["clauses"]
        waiving = [c for c in clauses if "waive" in c["text"].lower()]
        assert waiving, "the general SOP does mention waivers; that is not the leak"
        assert all(
            c["clause_id"].startswith("cancellation_and_service_credit_sop") for c in waiving
        )

    def test_the_prompt_injection_asks_for_a_tool_that_is_not_there(self, golden, toolset):
        # GS-028. The refusal is structural: the toolset was bound before the
        # message was read, so there is no code path from this text to a tool.
        entry = golden["GS-028"]
        available = {t.name for t in toolset("northstar_customer")}
        assert entry["expect"]["tool_absent"] not in available, entry["derivation"]
        assert not available & {"scan_support_health", "explain_finding", "query_tickets"}

    def test_an_agent_cannot_approve_a_credit(self, golden, toolset):
        # GS-029. Not an ACL breach - a role that lacks the scope. The answer
        # should cite the clause that says so.
        entry = golden["GS-029"]
        assert entry["expect"]["tool_absent"] not in {t.name for t in toolset("maya_agent")}
        approval = toolset("maya_agent")["resolve_policy"](
            topic="credit_approval", account_id="ACCT-002"
        ).data
        assert approval["governing_clause"] == entry["expect"]["governing"][0], entry["derivation"]

    def test_the_queue_is_split_by_assignee(self, golden, toolset):
        # GS-030.
        entry = golden["GS-030"]
        data = toolset("maya_agent")["my_queue"]().data
        found = {t["ticket_id"] for t in data["tickets"]}
        assert found == set(entry["expect"]["ticket_ids"]), entry["derivation"]
        assert not found & set(entry["expect"]["must_not_include"])

    def test_an_agent_has_no_ops_scan(self, golden, toolset):
        # GS-032. Same absence as GS-028, reached from the other side of the
        # role boundary - which is why the projection matrix carries a row for
        # a tool this milestone does not build.
        entry = golden["GS-032"]
        assert entry["expect"]["tool_absent"] not in {t.name for t in toolset("maya_agent")}
        assert PROJECTION[entry["expect"]["tool_absent"]] == frozenset({"ops_manager"})


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


class TestComposition:
    """GS-017, GS-024, GS-025: what the block renders and what the gate refuses.

    All three are failures of composition rather than of retrieval or
    arithmetic. The deprecated target is *found* by search and must not reach
    the citable set; the two no-source questions have nothing to find, and the
    failure is answering them anyway.
    """

    def test_the_deprecated_target_never_becomes_a_citable_source(self, golden, toolset):
        # GS-017. Policy v3 says Enterprise P1 = 30 minutes 24x7; the deprecated
        # v2 said 1 hour, and both are excellent lexical matches for the
        # question. Any answer of "1 hour" means a tier-4 clause reached the
        # citable set.
        entry = golden["GS-017"]
        clauses = toolset("axis_customer")["search_policy"](
            query=entry["question"], topic="first_response_target"
        ).data["clauses"]
        citable = {c["clause_id"] for c in clauses if c["citable"]}
        assert POLICY_V2 not in citable, entry["derivation"]

        sources = {c["clause_id"]: c["text"] for c in clauses if c["citable"]}
        block = compose(resolution=None)
        # The deprecated answer is caught by the *unit*, not by the number.
        # Policy v3 §3 itself says "1 business day", so the bare figure 1 is
        # grounded by the citable grid - which is exactly how a gate checking
        # bare numbers would have let "1 hour" through.
        assert check_figures("The target is 1 hour.", block, sources) == ((1.0, "hours"),)
        assert check_figures("The target is 30 minutes, 24x7.", block, sources) == ()

    def test_the_forbidden_answer_is_only_reachable_through_the_excluded_clause(
        self, golden, toolset
    ):
        entry = golden["GS-017"]
        assert entry["expect"]["forbidden_answers"] == ["1 hour", "60 minutes"]
        # Reachable deliberately, by staff, with the flag - and marked as not
        # citable even then.
        deprecated = toolset("priya_manager")["search_policy"](
            query=entry["question"], topic="first_response_target", include_deprecated=True
        ).data["clauses"]
        found = {c["clause_id"]: c for c in deprecated}
        assert found[POLICY_V2]["citable"] is False
        assert "1 hour" in found[POLICY_V2]["text"]

    @pytest.mark.parametrize(
        ("entry_id", "topic"),
        [("GS-024", "account_contact"), ("GS-025", "account_contact")],
    )
    def test_a_question_with_no_source_has_no_basis_to_answer_from(
        self, golden, toolset, entry_id, topic
    ):
        # GS-024 and GS-025. Two differently worded probes, so a system that
        # has merely learned "billing questions escalate" fails the second.
        entry = golden[entry_id]
        tools = toolset(entry["persona"])
        clauses = tools["search_policy"](query=entry["question"]).data["clauses"]
        # Retrieval always returns *something* - the corpus is nineteen clauses
        # and cosine similarity has no notion of "nothing relevant". What makes
        # this answerable-or-not is whether a clause governs the topic.
        resolution = tools["resolve_policy"](topic=topic).data
        assert resolution["has_basis"] is False, entry["derivation"]
        assert clauses  # retrieval was not empty; the gap is authority, not recall

    @pytest.mark.parametrize("entry_id", ["GS-024", "GS-025"])
    def test_it_drafts_a_record_naming_the_gap(self, golden, entry_id):
        entry = golden[entry_id]
        record = escalation_draft(
            principal=to_principal(get_persona(entry["persona"])),
            thread_id=entry_id,
            question=entry["question"],
            reason=DeclineReason.NO_CITABLE_SOURCE,
            subject=entry["question"].rstrip("?"),
        )
        assert entry["expect"]["escalate"] is True
        assert record.question == entry["question"]
        assert "corpus" in record.what_is_unresolved

    def test_the_billing_answer_invents_no_procedure(self, golden):
        # GS-024, the single most valuable question in the set: every SaaS
        # product has a settings page and a model will happily invent the path.
        entry = golden["GS-024"]
        record = escalation_draft(
            principal=to_principal(get_persona(entry["persona"])),
            thread_id="gs24",
            question=entry["question"],
            reason=DeclineReason.NO_CITABLE_SOURCE,
            subject=entry["question"].rstrip("?"),
        )
        said = f"{record.summary} {record.what_is_unresolved}".lower()
        for forbidden in entry["expect"]["must_not_assert"]:
            assert forbidden.lower() not in said

    def test_an_unsupported_answer_is_dropped_rather_than_softened(self, golden):
        # The exit is escalation, never a degraded answer. A system that trims
        # an unsupported claim until it passes has become vaguer, not truer.
        entry = golden["GS-025"]
        answer = assemble(
            "We retain shipment records for seven years.",
            messages=[{"role": "user", "content": entry["question"]}],
            resolution=None,
            principal=to_principal(get_persona(entry["persona"])),
            thread_id="gs25",
            question=entry["question"],
            extractor=_Stub("we retain shipment records for seven years"),
            subject=entry["question"].rstrip("?"),
        )
        assert answer.prose == ""
        assert answer.declined
        assert "seven years" not in answer.escalation.summary


class _Stub:
    def __init__(self, *claims):
        self.claims = list(claims)

    def extract(self, prose):
        return list(self.claims)


class TestCoverage:
    def test_every_entry_is_either_covered_or_explicitly_deferred(self, golden):
        covered = set(CANCELLATION + CREDIT + SLA + CONFLICT + TOOLS + COMPOSITION)
        deferred = set(NOT_YET_COMPUTABLE)
        assert covered | deferred == set(golden)
        assert not covered & deferred, "an entry is both covered and deferred"

    def test_the_deferred_list_shrinks_as_milestones_land(self, golden):
        # A reminder with teeth: when a milestone lands, its entries must move
        # out of NOT_YET_COMPUTABLE or this number stops being true. M4 took
        # three out, M5 ten, M7 three; GS-031 waits on detection.
        assert len(NOT_YET_COMPUTABLE) == 1
        assert all(reason for reason in NOT_YET_COMPUTABLE.values())
