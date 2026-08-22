# ParcelPilot — Task Tracker

Source of truth for scope: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) v2.0.
Verified facts: [`docs/01_DATA_PACK_FINDINGS.md`](../docs/01_DATA_PACK_FINDINGS.md).

## Phase 0 — Brainstorm and design (complete)

- [x] Read the assessment brief and job description
- [x] Inspect the full data pack: 6 PDFs, workbook (4 accounts / 6 orders / 7 tickets)
- [x] Verify OpenRouter key, Gemini key, Chroma Cloud key; find dead model slugs
- [x] Compute ground truth: cancellation windows, credit amounts, SLA clocks, AS_OF weekday
- [x] Close all 7 open items from the v1.1 architecture
- [x] Write `docs/01_DATA_PACK_FINDINGS.md`
- [x] Agree 28 architecture decisions (D1-D28) via structured Q&A
- [x] Write `docs/ARCHITECTURE.md` v2.0 as ground truth
- [x] Mark v1.1 superseded

## Phase 1 — Implementation (not started; no code until instructed)

- [x] M0  Repo hygiene, config, `clock.py`, provider preflight
- [x] M1  Data layer: ETL, schema, account-scoped views
- [x] M2  Clause registry + ingest; `params` baseline; Chroma provisioning; tool-calling check
- [x] M2.5 **Golden-set review gate** — 32 answers signed off
- [x] M3  Precedence resolver + deterministic calculators
- [x] M4  Consistency check + severity inference
- [x] M5  Tools with typed evidence handles + ACL projection
- [ ] M6  Agent graph (LangGraph ReAct), CLI harness
- [ ] M7  Fact-block composition + claim-level grounding gate
- [ ] M8  FastAPI + SSE + confirmation gate
- [ ] M9  Streamlit client: threads, trace panel, conflict badge, confirm card, resume
- [ ] M10 Ops page + proactive detection
- [ ] M11 Evaluation: pytest invariants, golden set, RAGAS
- [ ] M12 Docs, demo video, Railway deploy

## Decisions closed in this session

- [x] D1-D23 agreed (see decision register)
- [x] D9a amended: OpenRouter stays **unfunded**; Gemini carries dev, tests and demo
- [x] D24 clause `params`: extraction pipeline + reviewed committed baseline + drift test
- [x] D25 low-confidence severity: ops triage rounds up, customer surface declines and escalates
- [x] D26 three roles; ops dashboard and credit approval are manager-only; `my_queue` from `assigned_to`
- [x] D27 escalation is a drafted record through the confirmation gate
- [x] D28 golden set reviewed and signed off before any test depends on it

## Still open (none block starting)

- [x] ~~Tool-calling reliability on Gemini~~ — closed in M0 via thought_signature echo
- [x] Write and review the ~25-clause `params` baseline (M2)
- [x] Chroma Cloud database provisioning + free-tier limits (M2) — database created, 19 vectors
- [x] Numeric severity-confidence cut-off — **0.95**, calibrated in M4 against the
      five open tickets; see `severity.py` for the numbers and the caveat
- [ ] Railway topology: one service or two (M12)

---

## M0 — Foundation (complete)

**Goal:** a repo whose skeleton matches ARCHITECTURE.md v2.0, a frozen clock that cannot
be bypassed, a Principal whose scopes match D26, and a provider layer that fails loudly
at startup instead of mid-demo. TDD throughout: test first (RED), implement (GREEN), refactor.

### 0.1 Repo hygiene
- [x] Branch `feat/m0-foundation` off main
- [x] Delete v1.1 dead code: `src/agent/intents.py` (D11a removed the classifier)
- [x] Delete v1.1 dirs that no longer exist in the v2.0 layout: `src/agent/nodes/`,
      `src/agent/pipelines/`, `src/models/`, `src/tools/`
- [x] Create v2.0 dirs: `src/providers/`, `src/domain/`, `src/api/`, `src/agent/tools/`,
      `src/knowledge/vectorstore/`, `scripts/`
- [x] `pyproject.toml` for ruff + pytest config (line length, coverage gate, markers)
- [x] Update `requirements.txt`: fastapi, uvicorn, sse-starlette, langgraph-checkpoint-sqlite,
      langchain-openai, ragas, datasets, itsdangerous, tzdata; drop pandas if openpyxl suffices
- [x] Project `CLAUDE.md` declaring the production branch and what is already in place
- [x] Rewrite `.env.example` for dual providers, Chroma Cloud, three roles

### 0.2 `src/clock.py` — the only time source (D6, D22)
- [x] RED: `tests/unit/test_clock.py` — AS_OF parses as Asia/Kolkata and is a Sunday
- [x] RED: business-hours arithmetic across the Sunday boundary
      (Sun 11:00 + 4 business hours = Mon 13:00 IST)
- [x] RED: `business_hours_between` is zero across a whole weekend
- [x] RED: `add_business_days` skips Sat and Sun
- [x] RED: a clock built with no AS_OF configured raises, with no wall-clock fallback
- [x] GREEN: implement `clock.py`
- [x] Guard test: `datetime.now()` / `date.today()` / `time.time()` absent from `src/`

### 0.3 `src/auth/principal.py` — scopes matching D26
- [x] RED: `support_agent` lacks `read:ops_detection`; only `ops_manager` has it
- [x] RED: only `ops_manager` has `write:approve_credit`
- [x] RED: `support_agent` has `read:own_queue`; customer does not
- [x] RED: a customer without `account_id` raises; staff with one raises
- [x] RED: six seeded personas build with the right scopes and queues
- [x] GREEN: update `principal.py`, add `personas.py`

### 0.4 `src/config.py` — typed settings
- [x] RED: required keys missing fails loudly; provider selection is validated
- [x] RED: `embedding_identity` renders `{provider}/{model}/{dim}` for collection naming
- [x] GREEN: implement; move from `src/utils/config.py`

### 0.5 `src/providers/` — dual provider layer (D9a)
- [x] RED: `ChatProvider` / `EmbeddingProvider` protocol conformance for both impls
- [x] RED: Gemini and OpenRouter both build from config; unknown provider raises
- [x] RED: retry honours `Retry-After` and backs off with jitter
- [x] RED: query-embedding cache is keyed by `(embedding_identity, sha256(text))`
- [x] GREEN: `base.py`, `gemini.py`, `openrouter.py`, `registry.py`
- [x] `scripts/preflight.py`: verify every configured slug with a 1-token call, fail loudly
- [x] Live check: tool-calling reliability on Gemini's OpenAI-compatible endpoint.
      **CLOSED.** Works once each tool call's thought_signature is echoed back;
      the native google-genai client is not needed. Both providers pass 8/8 live.

### 0.6 Close out
- [x] `pytest` green, coverage >= 80% on touched modules
- [x] `ruff check` clean
- [x] Commit in reviewable batches, push, open PR against `main`

---

## M1 — Data layer (complete)

**Goal:** the workbook becomes a queryable, ACL-enforcing SQLite database whose
NULLs survive intact, and whose snapshot time is verified against config rather
than trusted. TDD throughout.

### 1.1 Schema
- [x] `src/datastore/schema.sql`: `accounts`, `orders`, `tickets`, `meta`
- [x] Foreign keys declared and `PRAGMA foreign_keys=ON` on every connection
- [x] Indexes for the real query shapes: `orders.account_id`, `tickets.account_id`,
      `tickets.assigned_to`, `tickets.status`
- [x] Timestamps stored as ISO 8601 **with offset**; a naive value is never written
- [x] Nullable columns stay nullable — no `NOT NULL DEFAULT 0` on money or timestamps

### 1.2 ETL (`src/datastore/etl.py`)
- [x] RED: all four sheets parse; row counts are exactly 4 / 6 / 7
- [x] RED: `AS_OF` is extracted from the README sheet, not hardcoded
- [x] RED: naive workbook datetimes are localised to Asia/Kolkata, never left naive
- [x] RED: absent values stay NULL — `pickup_actual_at` on ORD-1001,
      `cancellation_requested_at` on ORD-2002, `historical_resolution` on open tickets
- [x] RED: xlsx booleans coerce to 0/1, and `premium_support` survives round-trip
- [x] RED: rebuild is idempotent — running twice yields identical rows
- [x] GREEN: implement; `scripts/build_db.py` CLI

### 1.3 Account-scoped views
**Deviation:** no `views.sql`. SQLite cannot bind parameters inside a view
definition, so the account predicate has to be built per Principal. The views
are created in `repo._bind_scoped_views` at connect time, with the account id
validated against `^ACCT-\d{3}$` before interpolation.
- [x] RED: a customer connection's temp views expose only its own rows
- [x] RED: the view set covers orders, tickets and accounts
- [x] GREEN: scoped connection factory binds views from the Principal at connect time

### 1.4 Repository (`src/datastore/repo.py`)
- [x] RED: typed frozen row models; datetimes come back tz-aware
- [x] RED: a customer reading another account's order raises, and is loggable
- [x] RED: a customer cannot widen scope by passing `account_id`
- [x] RED: staff read any account; `my_queue` filters by `assigned_to`
- [x] RED: `query_tickets` is a parameterised builder, never free-text SQL
- [x] RED: batch fetch by ids is one round-trip, not one per id
- [x] GREEN: implement

### 1.5 Invariants that tie config to data
- [x] RED: `meta.as_of` equals `clock.as_of()` — config and workbook cannot drift
- [x] RED: every persona `account_id` exists in `accounts`
- [x] RED: every `tickets.assigned_to` matches a `support_agent` persona `queue_key`
- [x] RED: ground-truth row assertions from findings §10 (statuses, fees, fault flags)

### 1.6 Close out
- [x] `pytest` green, coverage >= 80%
- [x] `ruff check` clean
- [x] Commit in reviewable batches, push, open PR against `main`

---

## M2 — Clause registry, vector store, retrieval (complete)

**Goal:** the six PDFs become a typed authority spine plus a searchable collection.
This is where the assignment is won or lost: a wrong `params` value is a wrong
answer carrying a correct-looking citation.

**Parsing approach** (skill `regex-vs-llm-structured-text`, adapted for D10):
regex-first with per-clause confidence scoring. The skill's LLM-validates-the-
low-confidence-tail step is deliberately **not** adopted — a model call at build
time makes the committed index non-reproducible. The hand-reviewed baseline is
the validator instead, and it validates everything rather than a tail. Confidence
scoring still earns its place: it tells the reviewer where to look.

### 2.1 Text normalisation
- [x] RED: pypdf's doubled spaces and mid-word line breaks are repaired
- [x] RED: bullet glyphs, non-breaking spaces and soft hyphens normalise
- [x] RED: normalisation is idempotent
- [x] GREEN: `src/knowledge/pdftext.py`

### 2.2 Topic taxonomy
- [x] `src/knowledge/topics.py`: the curated enum from ARCHITECTURE §5.3
- [x] RED: every topic is reachable from at least one clause in the corpus
- [x] RED: no clause ends up untagged

### 2.3 Clause segmentation and metadata
- [x] RED: 6 documents parse; each yields the expected clause count
- [x] RED: tier, account scope, status and effective dates come from the header,
      not from a hardcoded table keyed on filename
- [x] RED: Policy v2 is tier 4 and marked DEPRECATED; agreements are tier 1 and
      account-scoped; SOP v4 and Policy v3 are tier 2
- [x] RED: clause text is verbatim, and `clause_id` is stable across runs
- [x] GREEN: `src/knowledge/clause_parser.py`

### 2.4 Typed params + reviewed baseline (D24)
- [x] RED: the discriminating params extract correctly —
      SOP `free_window_minutes=30` / `fee_after_window_inr=250`,
      Northstar `waiver=true` / `fee_inr=0`,
      LumenWorks `overrides=false` on cancellation but
      `threshold_hours=4` / `credit_inr=300` on credits
- [x] RED: the nine v2/v3 response-target cells extract as a typed grid
- [x] RED: confidence scoring flags a clause whose numbers did not parse
- [x] **REVIEW GATE:** print the baseline as a readable table for sign-off
- [x] GREEN: `clause_params_baseline.yaml` committed
- [x] RED: `test_clause_params.py` asserts `ingest(pdf) == baseline`, clause by clause

### 2.5 Registry persistence
**Deviation:** no `chunks` table. The clauses are one paragraph each (longest is 73
words, median 44), so a chunker would only ever split a rule away from the numbers that
qualify it. The clause *is* the chunk; `clause_topics` carries the tagging that a
chunk table would otherwise have held.
- [x] `clauses` and `clause_topics` tables added to `schema.sql`
- [x] RED: registry rebuild is idempotent and account-scoped reads work
- [x] GREEN: extend `etl.py` / add `src/knowledge/ingest.py`

### 2.6 Vector store (D20)
- [x] `VectorStore` protocol; `ChromaLocalStore` and `ChromaCloudStore`
- [x] RED: the ACL predicate is injected inside the store, not passed by callers
- [x] RED: collection name is namespaced by embedding identity
- [x] RED: a customer query never returns another account's agreement
- [x] `scripts/provision_chroma.py`; Cloud database created, free-tier confirmed

**Deviation:** one implementation, two client factories. Local and Cloud differ
by which client object is constructed; duplicating the query logic would give
two copies to keep in step, and the copy under test would be the one that never
ships. `tests/integration/test_vectorstore_live.py` covers the hosted path.

### 2.7 Hybrid retrieval
- [x] RED: BM25 built in memory from the clause table at startup
- [x] RED: RRF fusion prefers a clause both retrievers agree on
- [x] RED: an exact clause reference ("SOP v4 §1") is found by BM25 when dense misses
- [x] RED: the ACL holds on the lexical path too, and through fusion
- [x] RED: dense failure degrades to lexical; a malformed Principal never does
- [x] GREEN: `src/knowledge/retriever.py`, `src/knowledge/registry.py`

### 2.8 End-to-end (per user instruction: test E2E up to what is implemented)
- [x] `tests/integration/test_end_to_end.py` — grows each milestone
- [x] RED: build DB + registry + index from the real files, then as each persona
      run a real retrieval and assert tier ordering and account scoping
- [x] RED: a Northstar session retrieving `cancellation_fee` sees Northstar §2 and
      SOP v4 §1, and never the LumenWorks agreement
- [x] RED: Policy v2 is retrievable but never in the citable set
- [x] RED: retrieval scoping and repository scoping agree for the same principal
- [x] `scripts/demo_m2.py` so the pipeline is runnable by hand
- [x] `scripts/build_index.py`; verified against local Chroma and Chroma Cloud

### 2.9 Close out
- [x] `pytest` green (467), coverage 94%; `ruff check` clean
- [x] `pytest -m live` green (7/7) against Chroma Cloud + Gemini embeddings
- [x] Commit in reviewable batches, push, open PR against `main`

### Bugs found and fixed during M2 retrieval

- **Chroma rejects a single-operand `$and`/`$or`.** Every topic-scoped dense
  query with exactly one topic raised - the most common shape the resolver will
  issue. Missed by the unit tests because none passed a topic to the dense path;
  caught by the end-to-end test. Regression test added.
- **A currency amount matched a section reference.** `INR 1,000` tokenized to
  `1` and `000`, so a query for `SOP v4 §1` ranked §3 first: it held a bare `1`
  and, being the shortest clause in the document, won on BM25 length
  normalisation. The citation looked right and pointed at the wrong rule.
  References are now atomic tokens (`section_1`) and thousands separators are
  stripped before tokenizing.
- **The two retrievers indexed different text.** Dense embedded `text` alone
  while BM25 indexed title + reference + body, so a clause could be findable
  one way and invisible the other. Both now use `Chunk.searchable_text`.

---

## M2.5 — Golden-set review gate (D28)

**Status: signed off.** Nothing depends on the verdicts yet, and
M3 should not start until they have been read. The risk D28 names is a
misreading shared between the golden set and the implementation, which is
invisible because both agree; building the resolver against unreviewed
expectations is exactly how that happens.

- [x] 32 questions across 9 categories, hand-authored from the clauses
- [x] Every entry carries a derivation showing the reasoning
- [x] `scripts/review_golden_set.py` re-derives all arithmetic from the data
- [x] Referential integrity: every clause id and persona resolves
- [x] `tests/eval/test_golden_set_integrity.py` — 21 tests, structure and coverage
- [x] Corruption detection proven: bad arithmetic, wrong deadline, unknown
      clause and unknown persona are each caught and reported
- [x] **Signed off** — M3 builds against these

Coverage against ARCHITECTURE §18's named cases:

| Required case | Entries |
|---|---|
| ORD-1001 vs ORD-2001 discriminating pair | GS-001, GS-002 |
| Three-hour credit from two accounts | GS-008, GS-009 |
| Deprecated-policy answers differ | GS-013, GS-017 (+6 `must_not_cite`) |
| ORD-1001 staleness conflict surfaced | GS-001, GS-006, GS-019 |
| TKT-503 no source, must escalate | GS-024 (+GS-025) |
| Cross-account probes denied | GS-026, GS-027 |
| Prompt injection at a staff tool | GS-028 |
| "What changed between v2 and v3?" | GS-018 |

---

## M3 — Precedence resolver and deterministic calculators (complete)

**Goal:** every number, date and eligibility verdict in an answer is computed in
Python from `params` on a resolved clause, and arrives with the clause that
justifies it. The model gets to plan; it never gets to do arithmetic.

### 3.1 Evidence store and typed handles (D13a)
- [x] `evidence(evidence_id, kind, run_id, principal_hash, payload_json, ...)`
- [x] RED: minted server-side; `mint` has no `evidence_id` parameter
- [x] RED: a handle from another run is refused
- [x] RED: a handle minted for another Principal is refused, staff included
- [x] RED: `kind` is enforced on every read
- [x] RED: a scope denial carries no payload detail
- [x] RED: provenance is recorded at mint and survives two hops
- [x] GREEN: `src/domain/evidence.py`

### 3.2 Precedence resolver
- [x] RED: tier 4 excluded with a reason; never governing for any persona
- [x] RED: lowest surviving tier wins; the loser is recorded (GS-001)
- [x] RED: `overrides: false` honoured and still cited (GS-002)
- [x] RED: `overrides: null` is a baseline, not a decliner
- [x] RED: an account never resolves against another account's clause
- [x] RED: same-tier conflict from a synthetic fixture
- [x] RED: effective-date window respected
- [x] GREEN: `src/domain/resolver.py`

**Deviations from the architecture's sketch, both forced by the data:**
- `deferred` — a Tier 1 clause that declines to override needs its own bucket.
  `context_only` would file a live agreement beside a deprecated policy.
- `supporting` — three Tier 2 clauses carry `first_response_target` and only one
  states a grid. Treating every same-tier clause as a rival authority reports an
  unresolved conflict on every SLA question in the pack.
- `GENERAL_POLICY` sentinel — staff must name an account, but "what changed
  between v2 and v3?" has none. Reachable, never accidental.

### 3.3 Cancellation fee calculator — GS-001 to GS-006 all pass
- [x] Status before money; the waiver does not rescue a PICKED_UP shipment
- [x] Not cancellable reports `fee_inr: None`, never 0
- [x] Attribution is to the clause that decided *this* answer

### 3.4 Service credit calculator — GS-007 to GS-010 all pass
- [x] Threshold replacement in both directions, with the default reported alongside
- [x] `lower_of` is a cap, not a choice
- [x] Unknown shipment fee yields None plus the formula, never a number
- [x] Approval line arrives as its own resolution; absent means unknown, not False

### 3.5 SLA calculator — GS-011 to GS-015 all pass
- [x] Two named P1 triggers by deterministic guard (`src/domain/severity.py`)
- [x] 24x7 targets ignore the Sunday; business-hours targets start Monday 09:00
- [x] `measurable: false` always
- [x] D25 asymmetry: customer declines and escalates, ops rounds up

### 3.6 Golden set wiring
- [x] `tests/eval/test_golden_computable.py` — 15 entries through the real chain
- [x] The other 17 are named individually with the milestone that unblocks each
- [x] `scripts/demo_m3.py`

### 3.7 Close out
- [x] 614 tests green, 93% coverage; `ruff check` and `format` clean

---

## M4 — Consistency check and severity inference (complete)

**Goal:** the two things an answer must not do — repeat a wrong past answer, and
assert a status the data does not support — become detectable in Python before
the model writes a word. Plus the half of severity the guards cannot decide.

### 4.1 Severity inference (D23)
- [x] §2 definition spans read from the clause registry, not retyped
- [x] Tier predicate keeps the deprecated v2 definitions out
- [x] A guard match never reaches the classifier
- [x] A span absent from §2 is not believed — an invented citation is caught
- [x] Severity outside {P1,P2,P3} refused; confidence clamped
- [x] A classifier that fails, returns nothing, or is absent yields
      **undetermined**, never a confident P3
- [x] `LlmSeverityClassifier`, enum-constrained; 7 live tests green

### 4.2 Claim extraction (`src/domain/claims.py`)
- [x] Both recorded resolutions in the pack read correctly, with topics
- [x] Thousands separators survive; prose stating no rule yields nothing

### 4.3 `check_data_consistency` (D19)
- [x] All four classes; ORD-1001 stale, TKT-450 and TKT-451 contradicted
- [x] The TKT-504 → ORD-1001 link reported as an inference (A3), confidence 0.8
- [x] A resolved known issue cannot corroborate (KI-176)
- [x] The contradiction is account-relative — the same sentence is wrong for
      Northstar and right for an account with no agreement
- [x] TKT-451 attaches KI-208, because 3,000 is a defect threshold
- [x] Reports mint as `consistency_report` with the snapshot in provenance

**Deviation as planned:** `check_data_consistency(snapshot_id, *, topics=())`.
The last two classes are properties of a question, not of a row.

### 4.4 Baseline amendment
- [x] KI-176 gains `issue_status: Resolved`, one value re-reviewed. The drift
      test caught the change, which is what it is for.

### 4.5 Golden set
- [x] GS-019, GS-020, GS-021 computable; `NOT_YET_COMPUTABLE` 17 → 14

### 4.6 Close out
- [x] 732 tests green, 93% coverage; `ruff check` and `format` clean
- [x] `scripts/demo_m4.py`, `scripts/calibrate_severity.py`

### Bugs found and fixed during M4

Both in M3's resolver, both on `bulk_upload_limit`, and together they left the
topic with **no governing clause at all** — so "is 5,000 rows my limit?" had no
answer from a corpus that states one in plain words. Neither was visible from
M3's own tests, because no M3 calculator reads that topic.

- **Silence read as disagreement.** `_conflict` compared `params.get(key)`
  across a group, so a key only one clause mentions came back None for the
  other and counted as a differing value. Missing-data-is-not-zero, arriving in
  the precedence layer. Only keys every clause states can now differ.
- **A defect report outranked the rule it deviates from.** With the above
  fixed, the same-tier tie-break sorted by clause id and handed the topic to
  KI-208 — which says in its own text that the supported limit remains 5,000.
  Known issues are now supporting clauses on every topic: reachable, never
  governing.

### Findings worth carrying forward

- **Self-reported confidence is a weak proxy for stability.** TKT-504 flipped
  between P2 and P3 across six runs while reporting 0.85. The threshold catches
  it, but a model that flipped while reporting 1.00 would pass. Self-consistency
  sampling is the stronger signal — deferred to M11, where the eval harness can
  measure whether it helps.
- **A grounding gate must work on claims, not substrings.** KI-211's
  instruction contains "pickup did not occur" as a prohibition of saying it. A
  forbidden-phrase filter would flag the one correct answer. Noted for M7.


---

## M5 — Tools with typed evidence handles and ACL projection (complete)

**Goal:** the containment mechanism. An unauthorised query is not refused at
runtime — it is absent from the schema the model is given, because tools are
curried with the Principal before the first LLM call.

### 5.1 Tool primitives — done
- [x] `Tool` renders its own OpenAI function schema; no framework dependency
- [x] `ToolError` is returned, not raised, and names the tool that mints a
      missing handle (`Param.produced_by`)
- [x] A denial carries a reason code and nothing about what was denied, and is
      indistinguishable from "no such record"
- [x] An internal fault never puts its traceback in model context

### 5.2 Projection matrix — done
- [x] All sixteen tools as data, diffed row by row against ARCHITECTURE §4.3
- [x] `ops_manager` is a strict superset of `support_agent`
- [x] The five unbuilt tools are named with the milestone that adds each
- [x] `build_toolset(context)` takes one argument, so a toolset cannot be built
      for one identity over a repository opened for another
- [x] A startup check fails the import if the matrix and the builders disagree

### 5.3–5.6 Eleven tools — done
- [x] Three genuinely different schemas, and different *parameters*: a
      customer's `get_order` has no `account_id`, `search_policy` no
      `include_deprecated`
- [x] Lookups mint handles; a denied read mints nothing
- [x] `my_queue` splits Maya's three from Rohit's four
- [x] The calculators have no `order_id` parameter at all
- [x] `sla_first_response_status` derives severity rather than accepting it

### 5.7–5.8 Golden set and close out — done
- [x] Ten entries through the tool layer; `NOT_YET_COMPUTABLE` 14 → 4
- [x] End-to-end runs the tool chain over the index and database it built
- [x] `scripts/demo_m5.py` prints the three schemas and walks a chain
- [x] 885 tests green, 93% coverage; `ruff check` and `format` clean

### Bugs found and fixed during M5

- **`ToolError.missing_prerequisite` was unreachable.** The generic
  required-argument check in `Tool.__call__` fired first, so the model got
  "missing required argument(s) ['resolution_id']" with no indication of where
  one comes from — losing the property the whole handle design exists to
  create. Fixed at the schema level with `Param.produced_by`; the dead helper
  was removed rather than left as a second way to say the same thing.
- **`ToolContext.severity_classifier` was referenced and never defined.** No
  unit test exercised the SLA tool's wiring; the end-to-end test found it on
  its first run. Covered from both sides now.
- **`search_policy` reported "unavailable" before validating its topic.** A bad
  topic is the caller's mistake and is actionable; "unavailable" is not
  recoverable, so the order stopped the model fixing an error it had made.

### Findings worth carrying forward

- **Second `must_not_*` substring collision.** GS-027 forbids 'waive' leaking to
  Beacon, and SOP v4 §1 — general policy every customer may read — contains the
  word. As with GS-019's `must_not_assert`, the expectation is about an asserted
  claim, and a substring filter over evidence flags the correct answer. **M7's
  grounding gate must work on claims, not strings.** Two independent instances
  now; this is a design requirement, not a test quirk.


---

## Review

_To be filled in as milestones complete._
