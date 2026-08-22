# ParcelPilot AI Support System — Architecture

**Version:** 2.0 — **ground truth.** Supersedes `docs/00_ARCHITECTURE_v1_SUPERSEDED.md` v1.1 entirely.
**Status:** design agreed, pre-implementation. No code written against this yet.
**Grounded in:** `docs/01_DATA_PACK_FINDINGS.md` (every fact below is verified against the supplied pack).

**Stack:** Python · FastAPI · LangGraph · Streamlit · Chroma Cloud · SQLite · Gemini + OpenRouter
**Scope:** Both user contexts — customer-facing and internal ops — as one agent with role-scoped tool projection.
**Additional client problems:** Problem 1 (proactive detection) as a shared surface; Problem 2 (trust) as the spine of the whole design.

> **Why v1.1 was replaced.** It was written before the data pack was opened. Seven of its
> assumptions turned out to be wrong (corpus size, same-tier conflicts, ticket severity, the
> business-hours consequences of a Sunday snapshot, free-tier availability), and its central
> architectural bet — scripted pipelines ahead of planning — was reversed after review.
> v1.1 is retained for provenance only.

---

## 1. Thesis

This is not a RAG chatbot. It is a **decision system with a conversational surface**, and five
properties decide whether it is correct:

1. **Authority is data, not prose.** Every clause in the corpus is parsed once into a typed registry
   with a tier and an account scope. Precedence is then a SQL `ORDER BY tier`, not a judgment call.
   The precedence rule itself is *quoted from Support Policy v3 §1*, not invented by us.
2. **Containment is structural.** The customer-side model has no vocabulary for a cross-account
   query. Access control lives in the tool signature, the SQL view, and the vector predicate — never
   in a prompt, and never in a field the client can set.
3. **The model orchestrates; Python decides.** The LLM chooses which tools to call and writes the
   explanation. Every number, date, eligibility verdict and clause reference is computed in Python
   and rendered into a fact block the model cannot edit.
4. **The chain is guaranteed by types, not by a script.** Calculators refuse to run on bare IDs.
   They accept only evidence handles minted by upstream tools, so the model is free to plan yet
   structurally unable to skip a step.
5. **Knowing when not to answer is a feature.** Missing sources escalate. Unresolved data conflicts
   block state-changing actions. Deprecated policy is never citable as current.

---

## 2. Decision Register

Decisions carried forward from v1.1 keep their original numbers. Reversals and new decisions are marked.

| # | Decision | Rationale | Rejected |
|---|---|---|---|
| D1 | LangGraph, prebuilt ReAct agent | Durable checkpointer plus `interrupt()` inside the action tool maps 1:1 onto the confirmation requirement and onto resume-after-refresh. | Hand-rolled loop (we would rewrite checkpointing); CrewAI (HITL is its weakest surface) |
| D2 | One agent, role-scoped tool projection | Both personas share ~80% of logic. Two agents means two places the ACL can drift apart. | Separate customer/staff graphs |
| D3 | Chroma with server-side metadata pre-filtering | ACL and tier are query predicates, not post-hoc cleanup. Post-filtering leaks and silently under-returns. | FAISS |
| D4 | Hybrid retrieval (BM25 + dense, RRF) | Queries are clause-shaped and keyword-heavy. Dense-only misses exact clause references. Corpus is ~50 chunks so BM25 is free. | Dense-only |
| D5 | SQLite for structured data | Small and relational. Real SQL for aggregation, plus views as an ACL enforcement point. | Pandas in memory — no query layer, no ACL seam |
| D6 | Frozen `AS_OF` clock | `2026-08-16 11:00 Asia/Kolkata` from the workbook README. Any wall-clock read silently corrupts every window and SLA calculation. A CI grep bans `datetime.now()` in `src/`. | `datetime.now()` |
| D7 | Deterministic calculators as tools | A confidently wrong fee is the exact failure the brief warns about. | LLM arithmetic |
| D8 | Deterministic precedence resolver | Auditable and testable. Rules come from the corpus itself. | LLM-judged precedence |
| ~~D9~~ | ~~OpenRouter primary~~ | **Reversed → D9a** | |
| **D9a** | **Gemini for dev and demo; OpenRouter implemented, switchable, unfunded** | OpenRouter holds $0 credits and its `:free` slugs are actively being delisted (one dead slug found in a single afternoon). It stays fully implemented and unit-tested against a recorded/mocked client, switchable by one env var, but the demo never depends on it. Both providers speak the OpenAI wire format, so the abstraction costs three settings, not a framework. | Funding OpenRouter; single-provider lock-in |
| D10 | Build-time indexing, never at runtime | Reproducible, fast cold start, no rate-limit exposure on the hot path. | Index on startup |
| ~~D11~~ | ~~Classify-then-route with scripted pipelines~~ | **Reversed → D11a** | |
| **D11a** | **Pure tool-calling agent — the model plans every turn** | Scripted pipelines encode our guess at the query distribution and read as hard-coding to a reviewer. Reliability is bought back by D13a (typed handles) and D15a (fact block), which constrain *outcomes* rather than *orchestration*. | Scripted pipelines; deterministic-only routing |
| ~~D12~~ | ~~Hybrid classifier~~ | **Removed.** No classifier exists in v2.0. | |
| **D13a** | **Typed evidence handles between tools** | Guarantees the multi-step chain the brief requires without scripting it. `compute_cancellation_fee` accepts a `resolution_id`, not an `order_id`; only `resolve_policy` can mint one. | Fat tools (hides the chain); thin tools (ordering becomes a hope) |
| **D14a** | **Typed clause registry as the authority spine** | `resolve_policy` needs exact clause identity, not nearest-neighbour text. Precedence over a typed table is provable; precedence over retrieval results is contingent on recall. Vector search still serves open-ended questions and known-issue matching. | Vector-only precedence; whole-corpus-in-context |
| **D15a** | **Facts template-rendered, LLM writes only the surrounding prose** | Makes a confidently wrong number structurally impossible rather than merely unlikely. | Full LLM synthesis; fully templated answers |
| **D16** | **Claim-level grounding gate with bounded repair** | The prose is still model output. Atomic claims must map to a tool result or a citable clause; one targeted re-retrieval, then degrade to the fact block plus escalation. | Structural check only; no gate |
| **D17** | **Server-issued session token; the client never sends its own role** | The brief grades that access control is not enforced by model instructions. A client-settable `role` field would undercut the same principle one layer up. | Client-supplied principal |
| **D18** | **FastAPI backend + Streamlit client over SSE** | The JD explicitly asks for API design. Makes the confirmation interrupt a real HTTP contract and lets a refresh reattach to a live run. | Streamlit monolith; WebSocket |
| **D19** | **`check_data_consistency` as a first-class tool that gates actions** | SOP v4 §3 mandates it verbatim: *"When data conflicts, identify the conflict and request verification before a state-changing action."* The tool codifies the corpus rather than inventing policy. | Warning field on the calculator; prompt-only caveat |
| **D20** | **Chroma Cloud in dev and prod; local Chroma for tests and eval** | Same `VectorStore` interface, two implementations. Tests and the eval harness never require network. | Cloud-only (untestable offline); local-only |
| **D21** | **Three-layer evaluation: pytest invariants, exact-answer golden set, RAGAS** | RAGAS is LLM-judged and cannot express "cross-account query denied" or "fee == INR 0". Each layer catches what the others cannot. | RAGAS alone |
| **D22** | **Business hours = Mon-Fri 09:00-18:00 IST, surfaced in every answer that uses it** | The pack never defines it and the snapshot is a Sunday. Making the assumption visible turns our sharpest finding into a feature. | Calendar hours (contradicts LumenWorks' weekend clause) |
| **D23** | **Severity: deterministic guards for the two P1 triggers named in Policy v3 §2, LLM for the rest** | "Complete production outage preventing all shipment creation" and "suspected credential exposure" are enumerated verbatim in §2, so they must never be downgraded by a sampled token. Everything else is genuine judgment and returns the matched definition span plus a confidence. | Keyword table only (brittle); LLM only (a security P1 could be graded down) |
| **D24** | **Extraction pipeline plus a hand-reviewed committed baseline, with a drift test** | Ingest genuinely parses the PDFs into typed `params`; `clause_params_baseline.yaml` is reviewed once and committed; a unit test fails if extraction ever diverges from it. Real pipeline for the architecture story, verified data for correctness, and CI catches re-ingest drift instead of a user. | Hand-writing params (weaker story, does not survive a 7th PDF); extraction trusted on calculator tests alone (leaves caps, thresholds and dates unverified) |
| **D25** | **Low-confidence severity resolves asymmetrically by surface** | Costs are asymmetric. In ops triage an over-prioritised ticket costs an analyst two minutes, so fail toward the more severe class and label it inferred. In a customer-facing answer, quoting an unsure target is a promise ParcelPilot may not keep, so declare severity undetermined and escalate. | One uniform rule in either direction |
| **D26** | **Three roles, separated by capability rather than by the unreachable approval clause** | `scan_support_health`, `explain_finding` and credit approval above INR 1,000 are `ops_manager` only; `support_agent` additionally gets `my_queue` driven by the previously unused `assigned_to` column. All three projections differ visibly on real rows. The SOP §3 approval gate is implemented and unit-tested but cannot fire on this data — individual credits cap at INR 500 — and the writeup says so. | Two roles (leaves the SOP clause inert); agents scoped strictly to their own queue (visible but wrong as a product decision) |
| **D27** | **Escalation is a drafted record through the same confirmation gate** | The agent composes severity, account, evidence chain and specifically *what it could not determine*, then asks for confirmation. Reuses `prepare_action`/`execute_action`, so escalation, detection and action share one mechanism and one audit trail. | A message with no record (nothing auditable); auto-create without confirmation (contradicts brief requirement 4) |
| **D28** | **The golden set is reviewed and signed off before any test depends on it** | Its expected answers were derived by reading clauses. A misreading shared between the golden set and the implementation is invisible — both agree and both are wrong. Review is cheap because findings §10 already shows the arithmetic. | Codify and revisit on failure |

---

## 3. Topology

Two processes, one HTTP boundary. Locally via `docker-compose`; Railway at the very end.

```
  Streamlit client                          FastAPI backend
  ────────────────                          ───────────────
  persona picker  ──POST /auth/login──────▶  mint signed session token
                                             (Principal held server-side)

  send message    ──POST /threads/{id}/messages──▶  create run, spawn agent task
                  ◀──────────────────────────  { run_id }

  trace panel     ◀──GET /runs/{id}/events──   SSE, replayable from ?from_seq=N
                     run.started
                     tool.started    get_order
                     tool.finished   42 ms
                     policy.resolved governing / overridden / excluded
                     conflict.detected
                     facts.block     (whole, before any prose)
                     token.delta     ...
                     grounding.checked
                     interrupt.await_confirm

  Confirm button  ──POST /runs/{id}/resume──▶  execute_action(token)
                  ◀──────────────────────────  run.completed

  Ops page        ──GET /ops/findings────────▶  same scan_support_health tool
```

**Why the client is thin.** It holds no Principal, no agent state, and no business logic. It holds a
session token and a `thread_id`. Everything else is server state, which is what makes refresh-safe
resume possible and keeps the ACL story honest.

**Event durability.** Every emitted event is appended to `run_events(run_id, seq, type, payload)`
before it is streamed. `GET /runs/{id}/events?from_seq=N` replays from the store then tails live. A
mid-run refresh therefore loses nothing — the async-state-safety requirement, satisfied by design
rather than by a timer.

---

## 4. Identity and Access Control

### 4.1 Principal

```python
@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Literal["customer", "support_agent", "ops_manager"]
    account_id: str | None          # set for customers; None for staff
    scopes: frozenset[str]
```

Created at login from a fixed persona table, immutable for the session, **never a model-supplied
argument and never a client-supplied field**.

Seeded personas:

| Persona | Role | Scope | Why it exists |
|---|---|---|---|
| Northstar Logistics | `customer` | ACCT-001 | Agreement overrides the cancellation SOP outright |
| LumenWorks | `customer` | ACCT-002 | Agreement *declines* to override cancellation but replaces the credit terms |
| Beacon Retail | `customer` | ACCT-003 | No agreement; general policy governs |
| Axis Labs | `customer` | ACCT-004 | Enterprise without premium support; only DELIVERED order; owns the P1 security ticket |
| Maya | `support_agent` | all accounts; `my_queue` = TKT-502, TKT-504, TKT-450 | Reads widely, cannot run detection |
| Rohit | `support_agent` | all accounts; `my_queue` = TKT-501, TKT-503, TKT-505, TKT-451 | Holds both open P1 tickets |
| Priya Mehta | `ops_manager` | all accounts and tickets, ops dashboard, credit approval | The only role that can run detection |

One persona per account, because each account is a *different* policy situation — that is the whole
point of the pack. The Northstar/LumenWorks pair demonstrates the ORD-1001 vs ORD-2001 divergence in
two clicks. Maya and Rohit split the real ticket set via `assigned_to`. Priya is the CSM named in
both the `accounts` sheet and Northstar agreement §4.

### Scope table

| Scope | customer | support_agent | ops_manager |
|---|---|---|---|
| `read:own_account` | yes | — | — |
| `read:any_account` | — | yes | yes |
| `read:ticket_aggregates` | — | yes | yes |
| `read:own_queue` | — | yes | yes |
| `read:sla_status` | — | yes | yes |
| `read:ops_detection` | — | **—** | yes |
| `write:prepare_action` | yes | yes | yes |
| `write:approve_credit` | — | **—** | yes |

`ops_manager` is a strict superset of `support_agent`, and the only scope customers share with staff
is `write:prepare_action` — preparing is always allowed, because executing is gated separately by the
confirmation token.

### 4.2 Session tokens (D17)

`POST /auth/login {persona_id}` returns an opaque signed token. The server maps token → Principal in
a `sessions` table. Every subsequent request carries only the token. There is no `role` or
`account_id` anywhere in a request body. Forging a staff session requires forging a signature.

### 4.3 Tool projection — the containment mechanism

Tools are **curried with the Principal at graph-build time**, so the schema the model sees differs by role:

```python
# Customer session — the account filter is closed over and absent from the schema
def bind_customer_tools(p: Principal):
    @tool
    def get_order(order_id: str) -> OrderSnapshot:
        """Look up an order on your account."""
        return repo.get_order(order_id, account_id=p.account_id)   # not a parameter
    return [get_order, ...]

# Staff session — scope deliberately widened
def bind_staff_tools(p: Principal):
    @tool
    def get_order(order_id: str, account_id: str | None = None) -> OrderSnapshot:
        """Look up any order."""
        return repo.get_order(order_id, account_id=account_id)
```

The customer-side model has **no vocabulary** for a cross-account query. This is why a single agent
is safe: the projection, not the persona, enforces the boundary.

### Projection matrix (D26)

| Tool | customer | support_agent | ops_manager |
|---|---|---|---|
| `get_order`, `get_ticket`, `get_account` — own account | yes | yes | yes |
| `get_order`, `get_ticket`, `get_account` — any account | no | yes | yes |
| `search_policy` | own agreement + general | full | full |
| `query_tickets` | no | yes | yes |
| `my_queue` (driven by `assigned_to`) | no | yes | yes |
| calculators, `check_data_consistency` | yes | yes | yes |
| `sla_first_response_status` | no | yes | yes |
| `scan_support_health`, `explain_finding` | no | **no** | yes |
| `prepare_action` / `execute_action` | yes | yes | yes |
| `approve_credit` (> INR 1,000, SOP v4 §3) | no | **no** | yes |

Three genuinely different schemas, all demonstrable on shipped rows: logged in as Maya the ops page
does not exist; logged in as Priya it returns five ranked findings.

### 4.4 Defence in depth

| Layer | Control | Failure mode it catches |
|---|---|---|
| Session token | Role is server-resolved from a signed token | Client claiming a staff role |
| Tool schema | Unauthorised query is inexpressible | Model attempts a cross-account lookup |
| SQL views | Customer reads hit account-scoped views only | Bug in a tool body |
| Vector predicate | `account_id IN {session_account, NULL}` injected server-side | Retrieval of another account's agreement |
| Clause registry | Same predicate on the authority spine | Precedence resolved against a foreign agreement |
| Response scan | Block and log foreign account identifiers | Leak via summarisation or quoted context |

Every denial emits a `tool.denied` SSE event and is logged with the attempted query. **Denials are a
demo asset.**

### 4.5 Agreements are simultaneously ACL and precedence

Northstar's agreement is Tier 1 authority *and* Northstar-private data. One predicate —
`account_id IN {session_account, NULL}` — satisfies both. Worth stating explicitly in the writeup.

---

## 5. Knowledge Layer

### 5.1 Authority ladder

Quoted from Support Policy v3 §1, not designed by us. Assigned at ingest, carried as clause metadata,
enforced at resolution.

| Tier | Source | Rule |
|---|---|---|
| 0 | Workbook (accounts, orders, tickets) | Ground truth for facts — but status can be stale (§9) |
| 1 | Signed customer agreements | **Overrides** policy, scoped to that account |
| 2 | Support Policy v3 CURRENT, Cancellation & Service Credit SOP v4 | Default policy authority |
| 3 | Product Operations Guide and Known Issues | Current product documentation |
| 4 | Support Policy v2 DEPRECATED | Retrievable only for "what changed". **Never citable as current.** |
| 5 | Historical ticket resolutions | Context and pattern signal only. Never a basis for an answer. |

Tiers 4 and 5 stay in the store. Deleting them makes the conflict problem disappear along with any
ability to demonstrate it was solved. They are excluded from the **citable set** by the resolver, not
from storage — and retaining them enables a real capability: detecting that a past support answer
contradicts current policy (both closed tickets in the pack are wrong, see findings §6).

### 5.2 Ingest produces two artifacts (D14a)

One pass over the six PDFs produces:

**(a) The clause registry** — a typed SQLite table, the authority spine:

| column | example |
|---|---|
| `clause_id` | `northstar_agmt::§2` |
| `doc_id` | `05_northstar_logistics_enterprise_agreement` |
| `clause_ref` | `§2` |
| `title` | `Shipment cancellation` |
| `tier` | `1` |
| `account_id` | `ACCT-001` (NULL = applies to all) |
| `topic_tags` | `["cancellation_fee", "cancellation_window"]` |
| `effective_from` | `2026-01-01` |
| `effective_to` | `2026-12-31` |
| `status` | `CURRENT` / `DEPRECATED` |
| `superseded_by` | `null` |
| `text` | verbatim clause text |
| `params` | typed extraction: `{fee_inr: 0, waiver: true, applies_to_status: ["BOOKED"]}` |

`params` is the bridge from prose to arithmetic. The calculators read `params`, never the prose — so
a calculator can never misread a clause at runtime.

**Extraction is a real pipeline with a verified baseline (D24).** Ingest parses the PDFs into
`params`. `src/knowledge/clause_params_baseline.yaml` holds the hand-reviewed values and is
committed. `test_clause_params.py` asserts `ingest(pdf) == baseline` clause by clause, so a re-ingest
that silently changes a threshold fails CI instead of producing a wrong answer with a correct-looking
citation. The baseline is the highest-value review artifact in the build.

```yaml
northstar_agmt::§2:
  topic: cancellation_fee
  overrides: true
  fee_inr: 0
  waiver: true
  applies_to_status: [BOOKED]
  window_minutes: null          # "regardless of how long ago"

cancel_sop_v4::§1:
  topic: cancellation_fee
  overrides: null               # it IS the default
  free_window_minutes: 30
  fee_after_window_inr: 250
  waivable_by_agreement: true

lumenworks_agmt::§2:
  topic: cancellation_fee
  overrides: false              # explicit non-override

lumenworks_agmt::§3:
  topic: failed_pickup_credit
  overrides: true
  threshold_hours: 4            # replaces SOP's 2
  credit_inr: 300               # replaces lower(500, 10% of fee)
```

**(b) Chroma chunks** — clause-boundary chunks (clause integrity beats uniform length), stamped with
the same metadata, for open-ended search and known-issue matching.

### 5.3 Topic taxonomy

A small curated enum derived once from the corpus. It is the hinge: it lets clauses about the same
subject be grouped across documents, which is what makes conflict detection possible at all.

```
cancellation_fee, cancellation_window, cancellation_status_rules, return_to_origin,
failed_pickup_credit, credit_threshold, credit_amount, credit_cap, credit_approval,
severity_definition, first_response_target, weekend_coverage, escalation_duty,
source_precedence, plan_capability, bulk_upload_limit, shipment_status_semantics,
known_issue, resolved_issue
```

Reused by `resolve_policy`, the repair loop's vocabulary rewriting, and detection.

### 5.4 Vector store (D20)

```python
class VectorStore(Protocol):
    def query(self, text: str, *, principal: Principal, tiers: set[int],
              topic_tags: list[str] | None, k: int) -> list[Chunk]: ...
```

Two implementations: `ChromaCloudStore` (dev and prod) and `ChromaLocalStore` (tests, eval, offline).
The ACL predicate is injected inside the implementation, never passed by a caller.

**Collection naming is namespaced by embedding identity:**
`pp_clauses__{provider}__{model_slug}__{dim}`. Switching embedding provider selects a different
collection rather than silently comparing incompatible vectors. `scripts/provision_chroma.py` creates
the Cloud database and collections (the tenant currently has zero databases).

### 5.5 Retrieval

```
query
  → ACL predicate injected server-side (non-negotiable)
  → BM25 (in-memory, built from the clause table at startup) + dense (Chroma)
  → RRF fusion
  → group candidates by topic_tag
  → precedence resolver
  → { citable_set, context_only, conflicts }
```

BM25 needs no network: at ~50 chunks the corpus is rebuilt in memory at startup from SQLite.

---

## 6. Precedence Resolver

Deterministic. Operates on the clause registry, not on retrieval results.

```sql
SELECT * FROM clauses
WHERE topic_tags @> :topic
  AND (account_id = :session_account OR account_id IS NULL)
  AND tier < 4                      -- deprecated and historical are never citable
  AND :as_of BETWEEN effective_from AND COALESCE(effective_to, '9999-12-31')
ORDER BY tier ASC
```

1. Tier 4 and Tier 5 are moved out of the citable set into `context_only`.
2. Lowest surviving tier wins. A Tier 1 clause scoped to the session account beats Tier 2.
3. **The loser is recorded, not discarded.** The overridden rule becomes part of the answer.
4. **Explicit non-overrides are honoured.** LumenWorks §2 says *"No special cancellation-fee waiver
   applies. Use the current SOP."* Its `params` therefore carry `overrides: false` for
   `cancellation_fee`, and the resolver returns SOP v4 §1 as governing with no override. Northstar §3
   does the same for service credits beyond the cap. **A Tier 1 clause existing is not the same as a
   Tier 1 clause winning** — and the pack contains both cases specifically to catch that.
5. Two same-tier clauses on the same topic with different `params` → `unresolved_conflict` → escalate.
   No such case exists in this pack (findings §8); the branch exists for correctness and is covered
   by a synthetic unit-test fixture, not by a claim that the data exercises it.

Output:

```python
PolicyResolution(
    resolution_id="res_b2",              # the handle downstream tools require
    topic="cancellation_fee",
    governing=ClauseRef("northstar_agmt::§2", tier=1),
    overridden=[ClauseRef("cancel_sop_v4::§1", tier=2)],
    excluded=[ClauseRef("policy_v2::table", tier=4, reason="deprecated")],
    context_only=[ClauseRef("TKT-450::historical_resolution", tier=5)],
    unresolved_conflict=None,
)
```

When an override fires the response is **required** to surface it. Making the override visible is the
Problem 2 deliverable; never let the model silently pick a winner.

---

## 7. Structured Data Layer

Workbook → SQLite at build time. The `.db` is committed so the build is reproducible and the app
parses nothing at startup.

Tables: `accounts`, `orders`, `tickets` (Tier 0); `clauses`, `chunks` (Tier 1-5 registry);
`sessions`, `threads`, `messages`, `runs`, `run_events`, `evidence`, `actions` (app state).

Account-scoped **views** are the only read surface for customer sessions. No raw-SQL tool is exposed
to anyone; staff aggregates go through `query_tickets`, a parameterised builder.

### The clock (D6, D22)

`src/clock.py` is the only time source.

- `AS_OF = 2026-08-16 11:00 Asia/Kolkata` — read from the workbook README at build time. **No
  wall-clock fallback is provided.** Startup fails if it is unset.
- `BUSINESS_WEEK = Mon-Fri`, `BUSINESS_DAY = 09:00-18:00 IST`, holidays ignored.
- `business_hours_between(a, b)` and `add_business_hours(t, n)` are pure functions with unit tests.
- Every temporal calculation takes `as_of` as an argument. A CI grep bans `datetime.now()` in `src/`.

Because AS_OF is a **Sunday**, any answer quoting a business-hours target must also state the
assumption and when the clock actually starts. That text is rendered by Python into the fact block,
not left to the model.

### Evidence store (D13a)

```
evidence(evidence_id, kind, run_id, principal_hash, payload_json, created_at)
```

`kind ∈ {order_snapshot, ticket_snapshot, policy_resolution, calc_result, consistency_report}`.
Handles are minted server-side, scoped to the run, and **validated against the run's Principal on
every consumption** — so a handle cannot be replayed across sessions or accounts even if it leaks
into model context.

---

## 8. Deterministic Calculators

Each reads `params` from the resolved clause, never prose, and returns the governing clause alongside
the number so citation and computation cannot drift apart.

```python
compute_cancellation_fee(snapshot_id, resolution_id)
  → { eligible, fee_inr, window_minutes, minutes_since_booking,
      order_status, governing_clause, overridden_clauses, warnings }

compute_service_credit(snapshot_id, resolution_id)
  → { eligible, credit_inr, threshold_hours, delay_hours, rate_basis,
      requires_manager_approval,           # SOP v4 §3: any credit above INR 1,000
      monthly_cap_inr, governing_clause, overridden_clauses, warnings }

sla_first_response_status(ticket_snapshot_id, resolution_id)   # staff only
  → { severity, severity_basis_clause, severity_confidence,
      target, target_clause, clock_type: "24x7" | "business_hours",
      clock_starts_at, due_at, elapsed, past_target_by,
      measurable: false, measurability_note }
```

`measurable: false` is deliberate and load-bearing. The tickets table has no `first_response_at`
(findings §9), so a real breach cannot be measured — only elapsed-time-versus-target can be computed.
The calculator says so rather than asserting a breach it cannot prove.

### Low-confidence severity (D25)

Severity is inferred, and the target depends on it, so the behaviour below the confidence threshold
is asymmetric by surface — because the cost of being wrong is asymmetric.

| Surface | Behaviour below threshold | Why |
|---|---|---|
| Ops triage (`scan_support_health`) | Assume the **more severe** class, label it `severity_inferred: true`, rank accordingly | An over-prioritised ticket costs an analyst two minutes. A missed P1 costs an outage. |
| Customer-facing answer | **Do not quote a target.** Return `severity: undetermined` and escalate (D27) | Quoting an unsure target is a promise ParcelPilot may not keep |

The two P1 triggers named verbatim in Policy v3 §2 — complete production outage preventing all
shipment creation, and suspected credential exposure — are matched by deterministic guard (D23) and
never reach this path.

Verified expected outputs are enumerated in findings §10 and become the golden set.

---

## 9. Data Consistency Check (D19)

`check_data_consistency(snapshot_id)` cross-references an order or ticket against open tickets and
current known issues, and returns typed conflicts with a confidence and an inference note.

The case it exists for, from the actual pack:

- `ORD-1001` reads `status = BOOKED`, cancellation requested 11:00.
- `TKT-504` (same account, opened 10:50): *"SwiftShip order still shows BOOKED after driver pickup —
  driver collected the parcel around 10 minutes ago."*
- `KI-211`: SwiftShip pickup webhooks arrive up to 20 minutes late.
- ORD-1001 is Northstar's only SwiftShip order, so the ticket almost certainly refers to it — an
  **inference**, reported as such, because the tickets table has no `order_id`.

So the honest answer has two layers: the fee is waived by Northstar §2, **and** the BOOKED status may
be stale, in which case cancellation is not permitted at all and return-to-origin applies.

**`prepare_action` refuses to mint a token while an unresolved conflict of severity `blocking`
stands.** SOP v4 §3 requires exactly this. The tool is a codification of the corpus, not our policy.

Conflict classes implemented:

| Class | Detection | Example in pack |
|---|---|---|
| `stale_status` | Order status contradicted by a ticket plus a current known issue | ORD-1001 / TKT-504 / KI-211 |
| `historical_contradiction` | A Tier 5 resolution contradicts a Tier 1-3 clause | TKT-450, TKT-451 |
| `unresolved_same_tier` | Two same-tier clauses disagree on one topic | none in pack; unit-test fixture |
| `missing_source` | No citable clause exists for the topic asked | TKT-503 billing-contact change |

---

## 10. Tool Catalogue

Sixteen tools, six classes, with typed handles wiring them together.

| Tool | Class | Consumes | Produces | customer | support_agent | ops_manager |
|---|---|---|---|---|---|---|
| `search_policy` | Document | query, topic_tags | chunks + tiers | scoped | full | full |
| `get_order` | Structured | order_id | `snapshot_id` | own account | any | any |
| `get_ticket` | Structured | ticket_id | `snapshot_id` | own account | any | any |
| `get_account` | Structured | — / account_id | `snapshot_id` | own only | any | any |
| `query_tickets` | Structured, aggregate | filters | rows | — | yes | yes |
| `my_queue` | Structured | — | tickets where `assigned_to` = self | — | yes | yes |
| `resolve_policy` | Authority | topic, `snapshot_id` | `resolution_id` | scoped | full | full |
| `compute_cancellation_fee` | Calculation | `snapshot_id`, `resolution_id` | `calc_id` | yes | yes | yes |
| `compute_service_credit` | Calculation | `snapshot_id`, `resolution_id` | `calc_id` | yes | yes | yes |
| `sla_first_response_status` | Calculation | `snapshot_id`, `resolution_id` | `calc_id` | — | yes | yes |
| `check_data_consistency` | Integrity | `snapshot_id` | `report_id` | yes | yes | yes |
| `scan_support_health` | Detection | window | findings | — | **—** | yes |
| `explain_finding` | Detection | finding_id | evidence chain | — | **—** | yes |
| `prepare_action` | State-change (1) | kind, payload, evidence ids | preview + token | yes | yes | yes |
| `execute_action` | State-change (2) | token | receipt | yes | yes | yes |
| `approve_credit` | State-change, authz | `calc_id` | approval record | — | **—** | yes |

**Handle discipline.** `compute_cancellation_fee(order_id="ORD-1001")` is not a valid call — the
signature has no such parameter. Calling it without a `resolution_id` returns a structured
`ToolError` naming the missing prerequisite, which the model reads and corrects. That single design
choice is what turns "the brief requires multi-step" from a hope into an invariant, and it is what
makes the tool trace worth watching.

---

## 11. Agent Runtime

LangGraph `create_react_agent`, one graph, tools bound per Principal at build time.

```
  ┌────────────────────────────────────────────────────────┐
  │  bind tool projection from Principal  (before any LLM) │
  └───────────────────────────┬────────────────────────────┘
                              ▼
                       ┌─────────────┐
              ┌───────▶│    model    │  chooses tools freely
              │        └──────┬──────┘
              │               ▼
              │        ┌─────────────┐
              │        │    tools    │  typed handles enforce order
              │        └──────┬──────┘  interrupt() inside execute_action
              └───────────────┘
                              ▼
                     ┌──────────────────┐
                     │  compose facts   │  Python renders the fact block
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐   unsupported   ┌──────────┐
                     │  grounding gate  │────────────────▶│  repair  │
                     └────────┬─────────┘◀────────────────└──────────┘
                              │ passes             (budget: 2)
                              ▼
                    ┌───────────────────┐
                    │ respond / escalate│
                    └───────────────────┘
```

- **Checkpointer:** `SqliteSaver`, keyed by `thread_id`. Durable, resumable, survives a restart.
- **Interrupt:** `interrupt()` is called *inside* `execute_action`, so only state-changing tools pause.
  No blanket `interrupt_before=["tools"]`.
- **Post-model hook:** composition and the grounding gate.
- **No classifier, no scripted pipelines.** The model plans. Constraints act on outcomes.

### Grounding gate (D16)

1. A cheap model extracts atomic claims from the drafted prose.
2. Each claim must map to a tool result or a citable clause. Numbers are additionally checked against
   the fact block: **any figure in the prose that is not in the fact block is a hard failure.**
3. Unsupported claim → one targeted re-retrieval using the failing claim as the query (a named gap
   makes a far better query than a rewritten question).
4. Still unsupported after the budget → strip the prose, return the fact block plus an escalation.

Budget 2. `attempted_queries` is tracked so a query is never repeated. **Subset check:** if a rewrite
returns only chunks already held, stop immediately — no new information means further rewrites cannot
produce any. The exit is escalation, never a degraded answer.

**Sufficiency is never the model's call.** An LLM assessing its own evidence inside a loop
rationalises: given three attempts it quietly lowers its bar. So sufficiency is structural — *is
there a Tier 1-3 clause for the required topic? Did the calculator return a governing clause?* Those
are booleans. The model may declare evidence **insufficient**; it is never the sole authority
declaring it sufficient.

---

## 12. Answer Composition (D15a)

Python renders a fact block from the calculator and resolver output. The LLM writes only the prose
around it and cannot alter it.

```
[ rendered by Python ]
Verdict      No cancellation fee
Amount       INR 0
Governing    Northstar Enterprise Agreement §2  (tier 1, agreement)
Overridden   Cancellation SOP v4 §1  (tier 2 — INR 250 after 30 minutes)
Excluded     Support Policy v2  (tier 4, deprecated)
Basis        booked 09:00, cancellation requested 11:00 (120 min),
             status BOOKED, AS_OF 2026-08-16 11:00 IST
Caution      TKT-504 and KI-211 indicate BOOKED may be stale — verify carrier status

[ written by the LLM, constrained to the block above ]
Northstar's agreement waives the cancellation fee on any BOOKED shipment before
pickup regardless of elapsed time, so the standard INR 250 charge after 30 minutes
does not apply. One caution before acting: TKT-504 reports a driver collected this
parcel around 10:40, and KI-211 says SwiftShip pickup webhooks can lag 20 minutes,
so BOOKED may be stale. Verify carrier status before cancelling.
```

The block is emitted as a single `facts.block` SSE event **before** the first `token.delta`, so
numbers never appear half-typed.

---

## 13. Confirmation Gate

Two-phase, token-bound:

```
prepare_action(kind, payload, evidence_ids)
   → refuse if check_data_consistency reports a blocking conflict     (D19)
   → { preview, token = HMAC(payload ‖ session_id ‖ nonce) }
   → interrupt()
   → client renders the preview card; human clicks Confirm
   → POST /runs/{id}/resume { token }
execute_action(token)
   → recompute HMAC; refuse on mismatch, reuse, or expiry
```

The pending payload lives in **graph state, not the model's context window**, so the model cannot
mutate it between preview and execution. That is an integrity property, not a UX convention.

Executed actions append to an immutable `actions` table carrying the full evidence chain — every
action is auditable back to the clauses that justified it.

Action kinds (mocked, as permitted): `create_escalation`, `update_ticket_status`,
`create_followup_task`, `request_carrier_verification`, `approve_credit`.

### Escalation is an action, not a sentence (D27)

When the system declines to answer — no citable source, unresolved conflict, undetermined severity,
or an explicitly unsupported exception request — it does not merely say "a human will follow up". It
**drafts an escalation record** and routes it through the same gate:

```
create_escalation {
  account_id, thread_id,
  severity            : derived, or "undetermined" with the reason
  question            : the user's request, verbatim
  what_is_unresolved  : the specific gap — "no clause in the corpus documents
                        how to change a billing contact"
  evidence_chain      : [snapshot_id, resolution_id, report_id, ...]
  sources_consulted   : clause refs and tiers actually read
}
```

`approve_credit` above INR 1,000 is the one action gated by role as well as by confirmation (D26).
It is implemented and unit-tested against a synthetic fixture; on the shipped data individual credits
cap at INR 500, so it cannot fire — which the product note states rather than hides. A test fixture
is not data augmentation; the shipped dataset stays untouched.

TKT-503's billing-contact question is the live demo: nothing in the pack documents the procedure, so
the correct behaviour is a drafted escalation naming exactly that gap.

---

## 14. Problem 1 — Proactive Issue Detection

One implementation, two surfaces (the brief says "view"; the design says "no duplicated reasoning"):

- **Ops page** in the UI renders `GET /ops/findings`, which calls the same `scan_support_health` tool.
- **Chat drill-down**: staff asks *"why is TKT-501 top?"* and the agent calls `explain_finding`.
- Findings feed the same `prepare_action` gate, so detection and action share one mechanism.

### Why not clustering

The pack ships **7 tickets**. Embedding clustering is unstable and spike detection is statistically
meaningless at that n. The primary signal is therefore **matching tickets against the Known Issues
document** — stable at low volume, explainable, and it ties Problem 1 back to the document corpus.
Arguably the better design even with more data: a cluster labelled by a known issue is actionable, an
unlabelled cluster is not.

### Signals

| Signal | Method | Stable at n=7? |
|---|---|---|
| Known-issue recurrence | Semantic match ticket → Known Issues entry, count per issue | yes |
| First-response-target risk | Deterministic elapsed-vs-target against AS_OF, per governing clause | yes |
| Severity concentration | Derived severity (D23) grouped by account and plan | yes |
| Cross-account impact | Same known issue across ≥2 accounts → systemic | yes, but see note |
| Unmatched high severity | P1 with **no** matching known issue → possible new incident | yes |
| Historical contradiction | Tier 5 resolution contradicts current Tier 1-3 | yes |
| Volume spike | Rate versus trailing baseline | **no** — reported with an explicit low-confidence caveat, or suppressed |

Note: no known issue currently spans two accounts, so the cross-account signal will correctly return
nothing. Reporting "no systemic issues detected" is the honest output; manufacturing one would be
data augmentation we declined.

Expected top findings at AS_OF (verified, findings §9):

1. **TKT-505** Axis Labs, suspected credential exposure → P1 by named trigger, 120 min past a 30-min
   24x7 target. Escalate.
2. **TKT-501** Northstar, total shipment-creation outage → P1 by named trigger, 15 min past
   Northstar's 15-min target, **no matching known issue** → possible new incident.
3. **TKT-502** LumenWorks, matches KI-208, second occurrence in 5 days (TKT-451 on 11 Aug), and a
   prior **incorrect** resolution is on file.
4. **TKT-504** Northstar, matches KI-211 → do not tell the customer pickup failed; also invalidates
   the ORD-1001 cancellation request.
5. **TKT-503** Beacon Retail → **no documented procedure exists anywhere in the pack.** Knowledge gap.

---

## 15. Model Layer (D9a)

Both providers speak the OpenAI wire format, so provider-agnosticism is three settings, not a framework.

```python
class ChatProvider(Protocol):
    def complete(self, messages, *, tools=None, response_format=None,
                 model_tier: Literal["cheap","strong"]) -> Completion: ...

class EmbeddingProvider(Protocol):
    identity: str          # "{provider}/{model}/{dim}" — namespaces the collection
    def embed(self, texts: list[str], *, kind: Literal["document","query"]) -> list[Vector]: ...
```

| Stage | Tier | Why |
|---|---|---|
| Tool-calling loop, final prose | strong | The output a human acts on |
| Claim extraction, severity inference, query rewrite | cheap | High volume, enum-constrained, structurally simple |
| Embeddings | pinned, build time for documents; runtime for queries | Changing this invalidates the collection |

Verified working slugs as of 2026-08-22 (see findings §11):

| Provider | cheap | strong | embeddings | status |
|---|---|---|---|---|
| Gemini | `gemini-3.5-flash-lite` | `gemini-3.6-flash` | `gemini-embedding-001` | **primary — dev, tests, demo** |
| OpenRouter | `google/gemini-2.5-flash-lite` | configurable | `openai/text-embedding-3-small` | implemented, switchable, **unfunded** |

**On the unfunded alternate (D9a).** OpenRouter is a first-class implementation of both protocols
with its own unit tests against a recorded client, selected by `LLM_PROVIDER=openrouter`. It is
deliberately not on the demo path: the account holds $0 and its free slugs are being delisted, so
depending on it would trade a real risk for a cosmetic one. The writeup states this plainly rather
than implying failover we have not exercised live.

**Operational guards**

- **Startup preflight** verifies every configured slug with a 1-token call and a 1-token embedding,
  and fails loudly. Two dead slugs were found in one afternoon (`deepseek/...:free` delisted,
  `gemini-2.5-flash` 404 for new users) — slugs belong in config, never in code.
- **Runtime query embedding is cached** in SQLite keyed by `(embedding_identity, sha256(text))`.
  Deterministic per query, and it removes the embeddings quota from the demo's critical path.
- Backoff with jitter; honour `Retry-After`.
- Never embed documents at runtime.
- **Always send `max_tokens`.** OpenRouter reserves the requested budget against the account balance
  *before* running, so an uncapped request 402s on a low balance even when the reply is two words
  (observed: *"you requested up to 65535 tokens, but can only afford 15998"*). Default cap 4096. With
  it, the unfunded OpenRouter account passes the entire live suite.
- **Retries are the SDK's job.** The OpenAI client already backs off with jitter and honours
  `Retry-After`. Layering `tenacity` on top would double both the backoff and the quota burn. What we
  add is refusing to retry a 404 — retrying a delisted slug only delays the moment someone notices.

### Gemini tool calls must echo a `thought_signature` (resolved)

Verified live on 2026-08-22. This closes open item 1.

Gemini 3.x attaches a `thought_signature` to **each tool call**, at
`choices[].message.tool_calls[].extra_content.google.thought_signature`. The next request is rejected
without it:

> `400 — Function call is missing a thought_signature in functionCall parts.`

That breaks every conversation past the first tool call, which is every conversation that matters
here. The fix is small and stays inside the provider layer: `ToolCall` carries an opaque
`provider_meta`, and `ChatProvider.to_assistant_message()` puts it back verbatim when rebuilding the
assistant turn. Nothing outside `src/providers/` knows the field exists, and the native
`google-genai` client is not needed.

Reconstructing an assistant message by hand anywhere else is therefore a bug.

### The typed-handle design was validated before it was built

The same live test walked the real ORD-1001 question with three stub tools whose signatures encode
the prerequisite chain. `gemini-3.6-flash`, `gemini-3.5-flash-lite` and OpenRouter's
`google/gemini-2.5-flash` each independently produced:

```
get_order("ORD-1001")                                     -> snapshot_id
resolve_policy(topic=..., snapshot_id=...)                -> resolution_id
compute_cancellation_fee(snapshot_id=..., resolution_id=...)
```

No scripting, no plan node, and no ordering instruction in the prompt beyond "use tools for every
factual claim". This is the empirical case for D13a over D11's scripted pipelines: the schema shape
alone is enough to make the chain happen, and enough to make skipping a step impossible.

---

## 16. API Contract

```
POST   /auth/login                    {persona_id} → {session_token, principal_public}
GET    /auth/me                       → principal_public
POST   /auth/logout

GET    /threads                       → [{thread_id, title, updated_at, status}]
POST   /threads                       → {thread_id}
DELETE /threads/{thread_id}
GET    /threads/{thread_id}/messages  → replay for reopening a conversation
POST   /threads/{thread_id}/messages  {text} → {run_id}

GET    /runs/active                   → {run_id, thread_id, status} | null   (resume flow)
GET    /runs/{run_id}/events          SSE, supports ?from_seq=N
POST   /runs/{run_id}/resume          {confirm: bool, token}

GET    /ops/findings                  → scan_support_health (staff only)
GET    /ops/findings/{finding_id}     → explain_finding (staff only)

GET    /healthz                       → {status, as_of, providers, index_identity}
```

### SSE event schema

| event | payload |
|---|---|
| `run.started` | `run_id, thread_id, ts` |
| `model.step` | `step_index` |
| `tool.started` | `call_id, name, args_public, ts` |
| `tool.finished` | `call_id, name, ms, summary, evidence_id?` |
| `tool.denied` | `call_id, name, reason` |
| `tool.error` | `call_id, name, error, missing_prerequisite?` |
| `policy.resolved` | `topic, governing, overridden[], excluded[], context_only[]` |
| `conflict.detected` | `class, severity, detail, confidence, inference_note?` |
| `facts.block` | rendered fact block (emitted whole, before any prose) |
| `token.delta` | `text` |
| `grounding.checked` | `claims_total, unsupported[]` |
| `repair.started` | `reason, query, attempt` |
| `interrupt.await_confirm` | `preview, token, blocking_conflicts[]` |
| `run.escalated` | `reason, escalation_preview` |
| `run.completed` | `run_id, confidence, citations[]` |
| `run.failed` | `error` |

Every event is persisted with a monotonic `seq` before streaming, which is what makes `?from_seq=`
reattach work.

---

## 17. Interface

Streamlit, thin client, persona selected in the sidebar.

- **Sidebar** — persona picker (one click from Northstar customer to ops manager), thread list, new
  chat, delete thread.
- **Chat** — message stream; fact block rendered as a distinct card above the prose; prose streams in.
- **Trace panel** — live tool calls with arguments, latency, and returned source tiers.
- **Conflict badge** — loud, whenever an override fired or a consistency conflict was found. This is
  the single most legible demonstration of Problem 2 in a five-minute video.
- **Citations** — document, clause ref, and tier badge on every claim.
- **Confirmation card** — action preview with Confirm / Cancel; the graph is genuinely paused behind
  it, and blocking conflicts are shown on the card.
- **Denial notice** — when the ACL blocks something, say so plainly rather than failing silently.
- **Ops page** (staff only) — ranked findings from `GET /ops/findings`, each with a "Ask about this"
  button that seeds a chat message.
- **Resume** — on mount, `GET /runs/active`; if a run is live, reattach to its SSE stream from
  `from_seq=0`. Session id also mirrored to the URL query param.

---

## 18. Evaluation (D21)

Three layers. Each catches what the others cannot.

### Layer 1 — pytest invariants (must be green, no LLM involved)

| Test | Asserts |
|---|---|
| `test_acl.py` | A LumenWorks session cannot read ORD-1001; the customer `get_order` schema has no `account_id`; vector queries never return a foreign agreement |
| `test_auth.py` | A forged or client-supplied role is rejected; the token is server-resolved |
| `test_precedence.py` | Northstar §2 beats SOP v4 §1; **LumenWorks §2 does not** (explicit non-override); tier 4 is never citable; tier 5 is never a basis |
| `test_calculators.py` | Every row of findings §10 against hand-computed values |
| `test_clock.py` | Business-hours arithmetic across the Sunday boundary; `datetime.now()` absent from `src/` |
| `test_evidence_handles.py` | A calculator refuses a bare order_id; a handle from another run or Principal is rejected |
| `test_confirmation.py` | No execution without a valid token; no token while a blocking conflict stands; token reuse refused |
| `test_consistency.py` | ORD-1001 raises `stale_status`; TKT-450 and TKT-451 raise `historical_contradiction` |
| `test_grounding.py` | A number absent from the fact block fails the gate; the repair loop terminates |
| `test_severity.py` | TKT-501 and TKT-505 are P1 by deterministic guard, not by model sample; low confidence escalates on the customer surface and rounds up on the ops surface (D25) |
| `test_clause_params.py` | `ingest(pdf)` matches the reviewed baseline clause by clause (D24) |
| `test_role_projection.py` | Maya cannot reach `scan_support_health`; `my_queue` returns exactly her three tickets; only `ops_manager` can `approve_credit` |
| `test_escalation.py` | A no-source question drafts an escalation naming the gap, and creates nothing without confirmation (D27) |

### Layer 2 — exact-answer golden set (~30 questions)

`tests/eval/golden_set.yaml`, asserting verdict, amount, governing clause, overridden clause, and
escalation behaviour. Deliberately includes:

- **ORD-1001 vs ORD-2001** — same shape, opposite answers, purely because of the agreement. The
  discriminating pair; a hard-coded system fails one of them.
- The three-hour-late credit question from the brief, asked from a **LumenWorks** session (ineligible,
  4h threshold) and a **Beacon Retail** session (eligible, 2h threshold). Same words, different answers.
- Questions whose deprecated-policy answer differs (catches tier leakage — all nine v2/v3 cells differ).
- ORD-1001 asked such that the staleness conflict must be surfaced.
- TKT-503's billing-contact question — **no source exists**, must escalate, must not improvise.
- Cross-account probes — must be denied.
- Prompt-injection probes from a customer session attempting a staff tool.
- "What changed between policy v2 and v3?" — the one legitimate Tier 4 read.

### Layer 3 — RAGAS

`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` over the same question set,
judged by a Gemini model, embeddings from the configured provider. Reported as a table in the writeup
with an explicit caveat: at n≈30 over a 5k-token corpus these are directional, not statistical.

**Headline metric:** *escalation precision* — of the queries answered directly, what fraction a human
reviewer judges correct **and** adequately sourced. Coverage is trivially inflated by answering
everything; this metric only improves when the system knows what it does not know.

---

## 19. Assumptions Register

Every assumption is surfaced in the UI where it affects an answer, and listed in the writeup.

| # | Assumption | Why | Where surfaced |
|---|---|---|---|
| A1 | Business hours = Mon-Fri 09:00-18:00 IST, holidays ignored | Undefined in the pack; AS_OF is a Sunday | Fact block, whenever a business-hours target is quoted |
| A2 | `AS_OF` is the only "now" | Stated by the workbook README | `/healthz`, fact block |
| A3 | TKT-504 refers to ORD-1001 | Northstar's only SwiftShip order; tickets have no `order_id` | Reported as an inference with confidence, never as fact |
| A4 | Severity is derived, not given | No severity column | Fact block cites the §2 definition span it matched |
| A5 | First-response breach is not measurable | No `first_response_at` column | Phrased as elapsed-vs-target, with `measurable: false` |
| A6 | Actions are mocked | Explicitly permitted by the brief | Confirmation card labels the action as simulated |
| A7 | ACCT-003 and ACCT-004 have no agreement | Absent from the pack; workbook notes confirm | Resolver returns Tier 2 as governing with no override |
| A8 | Enterprise plan ≠ premium support | ACCT-004 is Enterprise with `premium_support = False` | Account fact lookup reports both fields |
| A9 | `assigned_to` defines a support agent's queue | The column exists and is populated; nothing else in the pack uses it | `my_queue` results are labelled as "assigned to you" |
| A10 | SOP §3 manager approval cannot fire on this data | Individual credits cap at INR 500 (fees peak 5,100, so 10% is 510, capped at 500) | Fact block always renders `requires_manager_approval`; product note states the gate is tested but unreachable |
| A11 | Escalation creates a record, not just a message | The brief requires escalation but does not define its artifact | Confirmation card shows the drafted record before anything is created |

---

## 20. Repository Layout

```
parcelpilot/
├── docs/                          # every document lives here
│   ├── 01_DATA_PACK_FINDINGS.md   #   verified ground truth
│   ├── ARCHITECTURE.md            #   this file
│   ├── ARCHITECTURE_NOTE.md       #   submission deliverable (short)
│   ├── PRODUCT_NOTE.md            #   submission deliverable
│   ├── API.md
│   ├── EVALUATION.md
│   ├── ASSUMPTIONS.md
│   └── AI_TOOL_USAGE.md
│
├── src/
│   ├── config.py                  # pydantic-settings
│   ├── clock.py                   # ★ AS_OF + business calendar; the only time source
│   │
│   ├── auth/          principal.py · sessions.py · personas.py
│   ├── providers/     base.py · gemini.py · openrouter.py · registry.py · preflight.py
│   │
│   ├── knowledge/
│   │   ├── ingest.py              # PDF → clause registry + chunks
│   │   ├── clause_parser.py       # clause segmentation + param extraction
│   │   ├── topics.py              # curated topic_tag enum
│   │   ├── clause_params_baseline.yaml   # ★ reviewed, committed, drift-tested
│   │   ├── vectorstore/  base.py · chroma_cloud.py · chroma_local.py
│   │   ├── retriever.py           # BM25 + dense, RRF
│   │   └── resolver.py            # ★ precedence over the clause registry
│   │
│   ├── datastore/     schema.sql · views.sql · etl.py · repo.py · evidence.py
│   │
│   ├── domain/
│   │   ├── severity.py            # ★ guards + LLM, cites the §2 span
│   │   ├── calculators.py         # ★ deterministic, read clause params
│   │   ├── consistency.py         # ★ check_data_consistency
│   │   └── detection.py           # scan_support_health · explain_finding
│   │
│   ├── agent/
│   │   ├── graph.py               # create_react_agent, checkpointer, hooks
│   │   ├── tools/                 # ★ Principal-bound projection, typed handles
│   │   ├── grounding.py           # ★ claim gate + bounded repair
│   │   ├── compose.py             # ★ fact-block renderer
│   │   ├── events.py              # event bus → SSE + run_events table
│   │   └── prompts/
│   │
│   └── api/           main.py · routes/ · schemas.py
│
├── app/               main.py · api_client.py · pages/ · components/
├── scripts/           build_db.py · build_index.py · provision_chroma.py · preflight.py
├── tests/             unit/ · integration/ · eval/{golden_set.yaml,run_golden.py,run_ragas.py}
├── data/              raw/ · parcelpilot.db (committed) · index/ (local fallback)
├── project_docs/      # supplied hiring material (gitignored), provenance only
├── docker-compose.yml
└── requirements.txt
```

★ marks where the assignment is won or lost.

### Dependency deltas from the current `requirements.txt`

Add: `fastapi`, `uvicorn[standard]`, `sse-starlette`, `langgraph-checkpoint-sqlite`,
`langchain-openai` (one client serves both providers via `base_url`), `ragas`, `datasets`,
`itsdangerous` (token signing), `tzdata`.
Reconsider: `pandas` is only needed by the ETL — `openpyxl` alone may suffice.

---

## 21. Build Order

Each milestone ends green and demoable. Nothing depends on a later milestone.

| # | Milestone | Done when |
|---|---|---|
| 0 | Repo hygiene, config, `clock.py`, preflight | **Done.** 174 tests, 95% coverage; `preflight.py` green; live suite 8/8 on both providers |
| 1 | Data layer | `build_db.py` produces `parcelpilot.db`; account-scoped views enforce ACL in tests |
| 2 | Clause registry + ingest | 6 PDFs → typed clauses with reviewed `params`; `provision_chroma.py` creates Cloud collections; tool-calling verified on both providers |
| 2.5 | **Golden-set review gate (D28)** | ~30 expected answers written as a reviewable table and **signed off by you** before any test depends on them |
| 3 | Resolver + calculators | Every row of findings §10 asserted by unit test, including the two explicit non-overrides |
| 4 | Consistency + severity | ORD-1001 staleness and both historical contradictions detected; P1 guards fire |
| 5 | Tools with typed handles | A calculator refuses a bare order_id; ACL denial tested for all six personas; `my_queue` splits Maya's and Rohit's tickets |
| 6 | Agent graph, no UI | CLI harness answers the brief's two example questions from both relevant personas |
| 7 | Composition + grounding gate | A number absent from the fact block fails the gate; repair terminates |
| 8 | FastAPI + SSE + confirmation | `curl` the SSE stream; confirm and cancel both work; `?from_seq=` replays |
| 9 | Streamlit client | Threads, new chat, delete, trace panel, conflict badge, confirm card, mid-run refresh reattaches |
| 10 | Ops page + detection | `GET /ops/findings` returns the five expected findings; drill-down works in chat |
| 11 | Evaluation | Layers 1-3 all runnable by one command each; RAGAS table produced |
| 12 | Docs, demo video, Railway deploy | Hosted link live; architecture and product notes written |

---

## 22. Trade-offs

**Accepted**

- **Pure tool-calling over scripted routing.** More LLM calls per turn and non-deterministic tool
  ordering, in exchange for a system that genuinely reasons over the data and cannot be accused of
  hard-coding. Reliability is recovered at the outcome layer (typed handles, fact block, grounding
  gate) rather than the orchestration layer.
- **Typed clause registry over pure vector RAG.** Requires reading the corpus once and hand-reviewing
  the extracted `params`. Buys provable precedence, exact citations, and calculators that cannot
  misread prose.
- **Two processes over a monolith.** More to run and deploy, in exchange for a real API, durable
  resumable runs, and an honest SSE stream.
- **Facts templated, prose free.** Answers are slightly more structured than a pure chatbot's.
  Worth it: it is the only design where a wrong number is impossible rather than unlikely.
- **Chroma Cloud over a committed local index.** A network dependency on the query path, mitigated by
  a query-embedding cache and a local implementation used for tests and eval.
- **Small-corpus honesty.** We build the seams that would scale and say plainly that the corpus is
  6 pages, 4 accounts, 6 orders and 7 tickets. Unearned scale claims are worse than none.

**Deliberately out of scope**

- Write-back to real carrier or ticketing systems — actions are mocked, as permitted.
- Cross-session memory or personalisation.
- Automated re-ingestion on document change — accommodated by `effective_from` / `superseded_by`
  metadata, not implemented.
- Synthetic data augmentation — declined so evaluation stays honest against the supplied pack. This
  is also why the cross-account-impact detection signal correctly reports nothing.
- Embedding-based ticket clustering and volume-spike detection — statistically meaningless at n=7,
  and saying so is more useful than shipping a number nobody should trust.
- Learned intent classification — would overfit at this data volume.

---

## 23. Open Items

Carried into implementation; none block starting.

1. ~~**Tool-calling reliability on Gemini's OpenAI-compatible endpoint**~~ — **closed in M0.** It
   works, provided each tool call's `thought_signature` is echoed back (§15). The native
   `google-genai` client is not needed.
2. **`params` baseline review** — resolved in principle by D24; the ~25 reviewed values still have to
   be written and checked in Milestone 2. Highest-value review in the build.
3. **Chroma Cloud provisioning** — the tenant has zero databases. Script it so a reviewer can
   reproduce, and confirm the free-tier limits are adequate for two collections of ~50 chunks.
4. ~~**OpenRouter funding**~~ — **closed.** Staying unfunded (D9a). Gemini carries dev, tests and demo.
5. ~~**Severity confidence threshold value**~~ — **closed in M4.** 0.95, calibrated against the
   five open tickets at six samples each (`scripts/calibrate_severity.py`). TKT-502 and TKT-503
   grade at 1.00 and never move; TKT-504 flips between P2 and P3 at 0.80-0.90, because a lagging
   status display is genuinely undecided by §2. The threshold sits in that gap. Caveat recorded in
   `severity.py`: the model reported 0.85 while giving different answers on identical input, so
   self-reported confidence tracks stability only loosely — self-consistency sampling is the
   stronger signal and is deferred to Milestone 11.
6. **Railway topology** — one service running both processes, or two services. Defer to Milestone 12.
