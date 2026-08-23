# ParcelPilot, explained end to end

This document is written to be read once, straight through, by somebody who has
never seen the code. By the end you should be able to explain what the system
does, why each piece exists, and what happens between a question being typed
and an answer appearing.

No prior knowledge of the codebase is assumed. Where a design decision could
plausibly have gone the other way, the alternative is stated and rejected with
a reason, because "why not the obvious thing?" is the question that actually
gets asked.

---

## 1. The problem

ParcelPilot is a B2B logistics platform. Businesses book shipments through it,
and a twenty-person support team answers hundreds of questions a week: *can I
cancel this order without a fee? has my SLA been breached? why is my parcel
still showing as booked?*

Answering any of those means reading across five different kinds of source:

```text
  POLICIES            AGREEMENTS           PRODUCT DOCS
  Support Policy v3   Northstar contract   Known Issues
  Support Policy v2   Lumenworks contract  Operations Guide
  (deprecated)
        \                   |                    /
         \                  |                   /
          +-----------------+------------------+
                            |
                     THE HARD PART
                            |
          +-----------------+------------------+
         /                                      \
  PAST TICKETS                          LIVE RECORDS
  historical resolutions,               accounts, orders,
  some of which are wrong               open tickets
```

The corpus is **deliberately imperfect**, and that is the whole exercise:

- **Support Policy v2 is deprecated** but still says things, confidently.
- **Customer agreements override general policy** — but only sometimes, and one
  of them explicitly declines to override.
- **Past ticket resolutions contain wrong advice** that a naive system will
  happily repeat.
- **Tickets carry no severity, no priority and no order reference.** All three
  have to be derived, and one of them cannot be derived reliably at all.

A system that treats every document as equally true will give confident wrong
answers. Most of this design is about refusing to do that.

---

## 2. The shape of the whole thing

```text
                            ┌──────────────────────┐
                            │  Streamlit UI        │  ui/
                            │  chat · ops · trace  │
                            └──────────┬───────────┘
                                       │  HTTP + SSE
                                       │  (a session token, nothing else)
                            ┌──────────▼───────────┐
                            │  FastAPI             │  src/api/
                            │  auth · runs · ops   │
                            └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │  Agent graph         │  src/agent/graph.py
                            │  model ⇄ tools       │
                            │  pause for a human   │
                            └──────────┬───────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        │              │               │               │              │
  ┌─────▼────┐  ┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼─────┐  ┌────▼─────┐
  │ RETRIEVE │  │  RESOLVE   │  │ CALCULATE  │  │   CHECK    │  │   ACT    │
  │ hybrid   │  │ precedence │  │ fees, SLA, │  │ conflicts, │  │ propose, │
  │ search   │  │ ladder     │  │ credits    │  │ severity   │  │ confirm  │
  └─────┬────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └────┬─────┘
        │              │               │               │             │
        └──────────────┴───────────────┼───────────────┴─────────────┘
                                       │
                      ┌────────────────▼─────────────────┐
                      │  SQLite  ·  Chroma  ·  clock     │
                      │  records    clauses    frozen    │
                      └──────────────────────────────────┘
```

Read it top to bottom: a person types, HTTP carries it, a graph runs a model
against a set of tools, the tools read a database and a document index, and
everything is anchored to a clock that never moves.

---

## 3. The five ideas that explain everything else

If you remember nothing else, remember these. Every other decision follows.

### Idea 1 — Authority is a property of a source

Not every document is equally true. Each clause is filed on a five-rung ladder:

```text
  TIER 1   Customer agreements          Northstar §2, Lumenworks §1
             ↑ overrides
  TIER 2   Current policy and SOPs      Support Policy v3, Cancellation SOP v4
             ↑ overrides
  TIER 3   Product documentation        Operations Guide, Known Issues
             ↑ overrides
  ─────────────────────────────────────────────────────────────
  TIER 4   Deprecated policy            Support Policy v2      ← never citable
  TIER 5   Historical ticket resolutions                       ← never citable
```

Tiers 4 and 5 are **retrievable but not citable**. The distinction matters: a
staff member asking "what changed between v2 and v3?" needs v2, but no answer
may ever *rest* on it.

The subtlety that makes this real: an agreement overriding a policy is only one
of three things a contract can do.

```text
  overrides = true    Northstar waives the cancellation fee     → DISPLACES the SOP
  overrides = false   Lumenworks says "the standard SOP applies" → DEFERS to the SOP
  overrides = null    a clause on an unrelated topic            → BASELINE, no opinion
```

A two-state model gets Lumenworks wrong. It has an agreement, so a naive
"has-agreement → agreement-wins" rule waives their fee. The contract explicitly
says not to. That is a real INR 250 error on real money.

### Idea 2 — Access control lives in the tool layer, never the prompt

The model is never told "do not read other accounts". Instead, the tools it is
given are **built for one specific user before the first model call**:

```text
  Maya (support agent) signs in
            │
            ▼
  Principal(role=support_agent, account_id=None, scopes={...})
            │
            ▼
  build_toolset(principal)  ←── decides membership from a projection matrix
            │
            ▼
  get_order(order_id, account_id=None)   ← staff version: can name any account
  query_tickets(...)                     ← present
  approve_credit(...)                    ← ABSENT. Not refused. Absent.
```

A customer's `get_order` has **no `account_id` parameter at all**. There is no
sentence for a prompt injection to argue with, because there is no vocabulary
for the query. Maya has no `approve_credit` to be talked into calling.

The measured result:

```text
  customer          10 tools
  support agent     13 tools   (+ ticket search, queue, SLA status)
  ops manager       16 tools   (+ credit approval, ops detection ×2)
```

Defence in depth behind that: SQL views scoped to the account at connect time,
an ACL predicate injected into every vector query *before* ranking, and the
same predicate on the clause registry.

### Idea 3 — The model orchestrates; Python decides

The model never computes a number. It chooses which tool to call; Python works
out the answer and renders it into a **fact block** the model cannot edit:

```text
  ┌─────────────────────────────────────────────────────────┐
  │  Verdict      No cancellation fee                       │  ← computed
  │  Amount       INR 0.00                                  │  ← computed
  │  Basis        120 minutes elapsed since booking         │  ← computed
  │  Governing    northstar_logistics_agreement::§2         │  ← resolved
  │  Overridden   cancellation_and_service_credit_sop_v4::§1│  ← resolved
  │  Caution      ORD-1001 shows BOOKED but TKT-504 reports │  ← detected
  │               the parcel was collected                  │
  └─────────────────────────────────────────────────────────┘
        the model writes the prose around this, never inside it
```

### Idea 4 — Calculators refuse bare IDs

This is the mechanism that makes multi-step reasoning a **property of the
schema** rather than a hope about the model's behaviour.

`compute_cancellation_fee` has no `order_id` parameter. It accepts only
`snapshot_id` and `resolution_id` — handles that have to be minted by earlier
tools. The chain is not scripted; it is the only path that type-checks:

```text
  get_order("ORD-1001")            →  snapshot_id: ev_a1
             │
             ▼
  resolve_policy("cancellation_fee")  →  resolution_id: ev_b2
             │                             (governing, overridden, deferred)
             ▼
  compute_cancellation_fee(ev_a1, ev_b2)  →  calc_id: ev_c3
             │                                verdict, amount, basis
             ▼
  check_data_consistency(ev_a1)    →  report_id: ev_d4
                                        blocking? advisory?
```

Call the calculator with an order id and it does not fail with a validation
error — it replies *"resolution_id comes from resolve_policy(topic=...)"*, which
is an instruction the model can act on rather than a wall to thrash against.

### Idea 5 — An answer is graded before anyone sees it

The prose the model writes is checked against the evidence it was supposed to
be written from. Two mechanisms, deliberately different:

```text
  FIGURES → checked in Python, no judgement involved
            every number in the prose must appear in the fact block
            or in the text of a source that was actually read

  CLAIMS  → extracted by a cheap model, compared in Python
            "the fee is waived" must map to something a source says
```

Figures are checked as **(value, unit) pairs**, not bare numbers. Policy v3 §3
itself contains the number `1` ("1 business day"), so a bare-number check would
happily pass *"the target is 1 hour"* — which is the deprecated v2 answer the
whole tier system exists to prevent.

**On failure the prose is dropped, not shortened.** A system that trims an
unsupported answer until it passes has not become more truthful; it has become
vaguer, and vagueness is harder to catch than a wrong number.

---

## 4. What happens when you ask a question

Follow one real request all the way through.

> **Northstar Logistics asks:** *"Can I cancel ORD-1001 without a fee?"*

```text
 1. UI ──────► POST /threads/t1/messages {"text": "..."}
                    │
 2.                 ├─► server resolves the bearer token to a Principal
                    │   (customer, ACCT-001) — the client never says who it is
                    │
 3.                 └─► run starts in a worker thread, returns a run_id
                        UI opens GET /runs/{id}/events  (SSE)

 4. GRAPH: model ──► get_order("ORD-1001")
                     scoped view returns it; snapshot minted        ev_a1
    ──► event: tool.started / tool.finished
        UI shows:  ◦ Looking up the order ORD-1001

 5. GRAPH: model ──► resolve_policy("cancellation_fee")
                     ladder walked: Northstar §2 (Tier 1, overrides=true)
                     displaces SOP v4 §1 (Tier 2)                   ev_b2
    ──► event: policy.resolved  {governing, overridden}
        UI shows:  ✓ Working out which policy applies

 6. GRAPH: model ──► compute_cancellation_fee(ev_a1, ev_b2)
                     120 minutes elapsed; agreement waives the fee  ev_c3

 7. GRAPH: model ──► check_data_consistency(ev_a1)
                     BLOCKING: order says BOOKED, TKT-504 says collected
    ──► event: conflict.detected
        UI shows a loud badge

 8. GATE:   fact block composed in Python
    ──► event: facts.block          (whole, and before any prose)
            claims extracted from the draft and checked
    ──► event: grounding.checked    {verdict, claims, unsupported[]}

 9. ──► event: token.delta × N      (the graded answer, in chunks)
    ──► event: run.completed        {citations}
```

Every event is **written to SQLite before it is streamed**. That is what makes
`?from_seq=` work: an event the client saw is on disk by construction, so a
browser that reconnects mid-run asks for everything after the last number it
holds and cannot be told about a gap that was never recorded.

---

## 5. Actions, and why confirming one is not a UX detail

Anything that changes state — raising an escalation, updating a ticket,
approving a credit — goes through a two-phase gate.

```text
  model ──► prepare_action(kind, payload, evidence_ids)
                    │
                    ├── blocking conflict in the evidence?  ──► REFUSED, no token
                    │
                    └── mint token = HMAC(payload ‖ kind ‖ evidence
                                          ‖ session ‖ nonce ‖ expiry)
                              │
                    ┌─────────▼──────────┐
                    │  GRAPH PAUSES      │   interrupt() — genuinely stopped,
                    │  state persisted   │   not a spinner
                    └─────────┬──────────┘
                              │  the token goes to the CLIENT
                              │  the payload stays in GRAPH STATE
                              ▼
                    ┌────────────────────┐
                    │  a person reads    │
                    │  the preview and   │
                    │  clicks Confirm    │
                    └─────────┬──────────┘
                              │  POST /runs/{id}/resume {token}
                              ▼
  execute_action(token) ──► recompute the HMAC from state
                            mismatch / reuse / expiry ──► refused
                            otherwise ──► append to an immutable log
```

Three properties are doing real work here:

**The payload lives in graph state, not the conversation.** Between the preview
and the click, the only actor in the loop is a language model. A gate that asks
"confirm?" and then reads a payload the model can still edit has confirmed
nothing.

**The token never enters model context.** It is a digest, not an envelope — no
payload value appears in it, and its length does not change when the payload
grows by 5 KB. A model holding the token could confirm on the human's behalf.

**Single use is enforced by the log itself.** The nonce is written onto the
immutable action row under a UNIQUE constraint, so a replayed token is refused
by the same mechanism that makes the log immutable, in the same statement that
would have written the effect. Separate "seen nonces" bookkeeping can succeed
while the effect fails, or the reverse.

---

## 6. Proactive detection: finding problems nobody asked about

The ops view answers a question no customer typed: *what needs attention right
now?*

**Why this is not clustering.** The pack ships seven tickets. Embedding
clustering is unstable and spike detection is meaningless at that n. The
primary signal is matching tickets against the Known Issues document — stable
at low volume, explainable, and it ties detection back to the same corpus
everything else reasons over. A cluster labelled by a known issue is
actionable; an unlabelled one is a shape on a chart.

```text
  SIGNAL                      ON THIS DATA
  ─────────────────────────────────────────────────────────────────
  first response risk         TKT-505  120 min past a 30-min target
                              TKT-501   15 min past a 15-min target
  unmatched high severity     TKT-501, TKT-505 — P1, no known issue
                              → possible new incidents
  known issue recurrence      TKT-502 → KI-208 (second occurrence)
                              TKT-504 → KI-211
  historical contradiction    TKT-450, TKT-451 — resolutions on file,
                              both wrong, context only
  severity concentration      nothing
  cross account impact        CHECKED — nothing. No known issue spans
                              two accounts on this pack.
  volume spike                SUPPRESSED — 5 open tickets is below the
                              30 a rate would need to mean anything
```

The last two rows are the point. **A signal that finds nothing still reports.**
"We looked and there is nothing" and "we did not look" are different
statements, and a dashboard that cannot tell them apart is one that quietly
stops working. Manufacturing a systemic issue to make the demo livelier would
have been data augmentation, which this design declined.

The matcher earned its own scrutiny. The first version attributed TKT-501 — a
total shipment-creation outage — to KI-208, the bulk-upload issue, because both
mention the words *shipment* and *creation*. Both are ordinary English and
neither distinguishes anything. Issues are now identified by their **curated
title** and extracted params, so KI-208 is `{bulk, upload, csvs, large}` and the
outage matches nothing:

```text
  TKT-502  "Bulk upload fails for 4,200-row CSV"
              ∩ KI-208 {bulk, upload, csvs, large}  =  {bulk, upload}   → MATCH
  TKT-501  "All shipment creation is failing"
              ∩ KI-208 {bulk, upload, csvs, large}  =  {}               → no match
```

And the match carries the words that decided it, because *"why is this ticket
attributed to KI-208?"* is the first question an operator asks and *"the vectors
were close"* is not an answer.

---

## 7. Things that are true about the data, and shaped the design

These are not incidental. Each one forced a decision.

**The clock is frozen at a Sunday.** `AS_OF = 2026-08-16 11:00 Asia/Kolkata`.
A 24×7 target runs immediately; a business-hours target does not start until
Monday 09:00. Two tickets raised ninety minutes apart on that Sunday can be one
past its target and the other not yet started — and that is most of this
dataset, not an edge case.

**Tickets have no `first_response_at` column.** So "has the SLA been breached?"
is *not computable*. Only "time elapsed versus the target" is. Every SLA answer
in the system says `measurable: false` and is phrased as target risk, never as
a measured breach. Reporting a breach you cannot prove is a claim somebody will
eventually ask you to defend.

**Tickets carry no order reference.** TKT-504 names SwiftShip and describes a
pickup; ORD-1001 is the only SwiftShip shipment on that account still awaiting
pickup confirmation. The link is an **inference**, and it is disclosed as one
in its own labelled row rather than buried mid-paragraph.

**One agreement makes things worse for ParcelPilot.** TKT-501 under standard
Enterprise policy (30 min) would be due at exactly 11:00 and not yet late.
Under Northstar's agreement (15 min) it is fifteen minutes past. Precedence has
teeth in both directions, which is a good sign that it is being applied rather
than assumed.

**TKT-503 has no answer anywhere.** Nothing in the entire pack documents how to
change a billing contact. The correct behaviour is to say so and draft an
escalation — not to improvise something plausible.

---

## 8. How the pieces map to files

```text
  A question arrives
        │
        ├── ui/app.py              renders, streams, never computes
        ├── ui/state.py            folds SSE events into one view model
        ├── ui/labels.py           tool names → words a customer can read
        │
        ├── src/api/app.py         routes; resolves the token to a Principal
        ├── src/api/runner.py      drives one run, narrates it as events
        ├── src/api/events.py      persist-then-broadcast; ?from_seq= replay
        │
        ├── src/agent/graph.py     model ⇄ tools loop, and the pause
        ├── src/agent/tools/       16 tools, built per-role
        │     registry.py            the projection matrix
        │     actions.py             prepare / execute / approve
        │     detection.py           the ops scan
        ├── src/agent/facts.py     the fact block
        ├── src/agent/grounding.py the gate
        ├── src/agent/answer.py    assemble, or decline and escalate
        │
        ├── src/domain/resolver.py    the precedence ladder
        ├── src/domain/calculators/   fees, credits, SLA
        ├── src/domain/severity.py    P1 guards + model inference
        ├── src/domain/consistency.py conflict classes
        ├── src/domain/detection.py   the seven signals
        │
        ├── src/knowledge/         PDFs → clauses → hybrid retrieval
        ├── src/datastore/         SQLite, scoped views, runtime state
        └── src/clock.py           the only source of time
```

---

## 9. What I would tell an interviewer

**The single most important decision** was putting access control in the tool
layer. Everything else — one agent for two audiences, safety against prompt
injection, three genuinely different schemas — falls out of it. A prompt-based
boundary would have needed the model's cooperation to hold.

**The decision that took the longest to get right** was the three-state
override. Two states looked obviously sufficient until Lumenworks' agreement
turned out to explicitly *decline* to override, at which point a boolean gives
the wrong answer on real money.

**The bug I am most glad a test caught** was the grounding gate accepting bare
numbers. Policy v3 §3 contains the number `1`, so *"the target is 1 hour"* — the
deprecated v2 answer — was grounded by a citable source. Pairing every figure
with its unit fixed it.

**The bug I am least proud of** shipped and lasted until somebody refreshed the
page. The transcript endpoint read the model's *draft* out of the checkpointer,
so an answer the gate had refused came back verbatim on the next page load.
Every guarantee held live and evaporated at F5. The transcript is now rebuilt
from the events that were actually delivered.

**What I would do next, with more time:** self-consistency sampling for
severity instead of trusting a model's self-reported confidence, which the
calibration showed tracks its own stability only loosely — the classifier
reported 0.85 while giving different answers on identical input.
