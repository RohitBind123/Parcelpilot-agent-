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
- [ ] M2  Clause registry + ingest; `params` baseline; Chroma provisioning; tool-calling check
- [ ] M2.5 **Golden-set review gate** — you sign off ~30 expected answers before tests depend on them
- [ ] M3  Precedence resolver + deterministic calculators
- [ ] M4  Consistency check + severity inference
- [ ] M5  Tools with typed evidence handles + ACL projection
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
- [ ] Write and review the ~25-clause `params` baseline (M2)
- [ ] Chroma Cloud database provisioning + free-tier limits (M2)
- [ ] Numeric severity-confidence cut-off — behaviour settled, value not (M4)
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

## M2 — Clause registry, vector store, retrieval (in progress)

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
- [ ] RED: pypdf's doubled spaces and mid-word line breaks are repaired
- [ ] RED: bullet glyphs, non-breaking spaces and soft hyphens normalise
- [ ] RED: normalisation is idempotent
- [ ] GREEN: `src/knowledge/pdftext.py`

### 2.2 Topic taxonomy
- [ ] `src/knowledge/topics.py`: the curated enum from ARCHITECTURE §5.3
- [ ] RED: every topic is reachable from at least one clause in the corpus
- [ ] RED: no clause ends up untagged

### 2.3 Clause segmentation and metadata
- [ ] RED: 6 documents parse; each yields the expected clause count
- [ ] RED: tier, account scope, status and effective dates come from the header,
      not from a hardcoded table keyed on filename
- [ ] RED: Policy v2 is tier 4 and marked DEPRECATED; agreements are tier 1 and
      account-scoped; SOP v4 and Policy v3 are tier 2
- [ ] RED: clause text is verbatim, and `clause_id` is stable across runs
- [ ] GREEN: `src/knowledge/clause_parser.py`

### 2.4 Typed params + reviewed baseline (D24)
- [ ] RED: the discriminating params extract correctly —
      SOP `free_window_minutes=30` / `fee_after_window_inr=250`,
      Northstar `waiver=true` / `fee_inr=0`,
      LumenWorks `overrides=false` on cancellation but
      `threshold_hours=4` / `credit_inr=300` on credits
- [ ] RED: the nine v2/v3 response-target cells extract as a typed grid
- [ ] RED: confidence scoring flags a clause whose numbers did not parse
- [ ] **REVIEW GATE:** print the baseline as a readable table for sign-off
- [ ] GREEN: `clause_params_baseline.yaml` committed
- [ ] RED: `test_clause_params.py` asserts `ingest(pdf) == baseline`, clause by clause

### 2.5 Registry persistence
- [ ] `clauses` and `chunks` tables added to `schema.sql`
- [ ] RED: registry rebuild is idempotent and account-scoped reads work
- [ ] GREEN: extend `etl.py` / add `src/knowledge/ingest.py`

### 2.6 Vector store (D20)
- [ ] `VectorStore` protocol; `ChromaLocalStore` and `ChromaCloudStore`
- [ ] RED: the ACL predicate is injected inside the store, not passed by callers
- [ ] RED: collection name is namespaced by embedding identity
- [ ] RED: a customer query never returns another account's agreement
- [ ] `scripts/provision_chroma.py`; confirm free-tier limits

### 2.7 Hybrid retrieval
- [ ] RED: BM25 built in memory from the clause table at startup
- [ ] RED: RRF fusion prefers a clause both retrievers agree on
- [ ] RED: an exact clause reference ("SOP v4 §1") is found by BM25 when dense misses
- [ ] GREEN: `src/knowledge/retriever.py`

### 2.8 End-to-end (per user instruction: test E2E up to what is implemented)
- [ ] `tests/integration/test_end_to_end.py` — grows each milestone
- [ ] RED: build DB + registry + index from the real files, then as each persona
      run a real retrieval and assert tier ordering and account scoping
- [ ] RED: a Northstar session retrieving `cancellation_fee` sees Northstar §2 and
      SOP v4 §1, and never the LumenWorks agreement
- [ ] RED: Policy v2 is retrievable but never in the citable set
- [ ] `scripts/demo_m2.py` so the pipeline is runnable by hand

### 2.9 Close out
- [ ] `pytest` green, coverage >= 80%; `ruff check` clean
- [ ] Commit in reviewable batches, push, open PR against `main`

---

## Review

_To be filled in as milestones complete._
