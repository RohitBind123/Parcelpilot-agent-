# Data Pack Findings — Ground Truth

**Status:** Verified against the supplied pack on 2026-08-22. Facts only; no design decisions.
**Purpose:** Close the open items in `project_docs/ARCHITECTURE.md` §18 before the real architecture is written.

---

## 1. Corpus size — much smaller than assumed

| Asset | Actual size |
|---|---|
| 6 PDFs | **1 page each. ~4,000 words total.** |
| accounts | **4 rows** |
| orders | **6 rows** |
| tickets | **7 rows** (5 open, 2 closed) |

`ARCHITECTURE.md` assumes "~300 chunks". The real number is closer to **40-60 chunks**, and the
entire document corpus is roughly **5,000 tokens** — it fits in a single context window.

Consequences:
- Retrieval recall is a non-problem. Precision, authority, and conflict handling are the whole game.
- Embedding clustering for Problem 1 is statistically meaningless at n=7 tickets. Confirmed.
- Any "scales to millions of docs" claim would be unearned. The honest framing is: build the
  seams that would scale, and say plainly that the corpus is small.

## 2. The snapshot time is a Sunday

`README!Dataset snapshot = 2026-08-16 11:00 Asia/Kolkata`.

**2026-08-16 is a Sunday.**

This is load-bearing and the existing doc misses it entirely:
- Support Policy v3 expresses Growth and Standard targets in **business hours / business days**.
- LumenWorks' agreement says **"No weekend or after-hours support coverage."**
- Only Enterprise P1 is `24x7`.

So on the snapshot instant, most non-Enterprise SLA clocks have **not started**. The pack never
defines what a business hour or business day is — that must become an explicit, surfaced assumption.

Other relevant dates: 2026-08-11 Tuesday, 2026-08-14 Friday, 2026-08-15 Saturday, 2026-07-12 Sunday.

## 3. The precedence rule is quoted, not invented

Support Policy v3 §1 states it directly:

> "A signed customer agreement may override these defaults. When sources conflict, use the signed
> customer agreement first, then the current support policy, then current product documentation.
> Historical tickets and internal notes are context only and may contain incorrect past guidance."

The authority ladder is therefore **citable to the corpus**, not a design invention. Every
precedence decision can cite Support Policy v3 §1 as its own warrant.

Derived ladder:

| Tier | Source | Status in pack |
|---|---|---|
| 0 | Workbook (accounts, orders, tickets) | Ground truth for facts — but see §7, status can be stale |
| 1 | Signed customer agreements (05 Northstar, 06 LumenWorks) | Overrides policy, account-scoped |
| 2 | Support Policy v3 CURRENT, Cancellation & Service Credit SOP v4 | Default authority |
| 3 | Product Operations Guide & Known Issues | Current product documentation |
| 4 | Support Policy v2 DEPRECATED | "DO NOT USE FOR CURRENT REQUESTS" — never citable as current |
| 5 | Historical ticket resolutions | Context only, "may contain incorrect past guidance" |

## 4. Exactly what each agreement overrides

**Northstar (ACCT-001, Enterprise, term 2026-01-01 to 2026-12-31, ACTIVE)**

| Clause | Override |
|---|---|
| §1 Support terms | P1 **15 min 24x7**, P2 **1 hour**, P3 **8 business hours** — replaces policy v3 defaults (30 min / 2 h / 1 business day) |
| §2 Cancellation | **No cancellation fee on any BOOKED shipment before pickup, regardless of elapsed time** — full waiver of SOP v4 §1 |
| §2 Cancellation | Once PICKED_UP, standard return-to-origin applies — **agrees** with SOP, no conflict |
| §3 Service credits | Monthly aggregate cap **INR 5,000**; otherwise "the current ParcelPilot service-credit SOP applies" — an explicit *non*-override |
| §4 | Dedicated CSM Priya Mehta |

**LumenWorks (ACCT-002, Growth, term 2026-03-01 to 2027-02-28, ACTIVE)**

| Clause | Override |
|---|---|
| §1 Support terms | P1 **2 business hours**, P2 **4 business hours**, P3 **2 business days**; **no weekend or after-hours coverage** |
| §2 Cancellation | "No special cancellation-fee waiver applies. Use the current SOP." — explicit *non*-override |
| §3 Failed-pickup credits | **>4 hours** past window end + carrier fault + no customer fault -> **fixed INR 300**. Text says it "replaces the default failed-pickup credit amount **and timing threshold**" |

Note the two explicit non-overrides. They are as important as the overrides: they prove the
resolver must read the agreement rather than assume "agreement exists therefore agreement wins".

**ACCT-003 Beacon Retail (Standard)** and **ACCT-004 Axis Labs (Enterprise)** have no agreement in
the pack. Standard policy applies. Axis Labs is Enterprise but `premium_support = False`.

## 5. Support Policy v2 vs v3 — the tier-4 trap is live

v2 (DEPRECATED, effective 2025-01-01) vs v3 (CURRENT, effective 2026-05-01), first-response targets:

| Plan | Sev | v2 (deprecated) | v3 (current) |
|---|---|---|---|
| Enterprise | P1 | 1 hour | **30 minutes, 24x7** |
| Enterprise | P2 | 4 hours | **2 hours** |
| Enterprise | P3 | 2 business days | **1 business day** |
| Growth | P1 | 4 business hours | **2 business hours** |
| Growth | P2 | 1 business day | **4 business hours** |
| Growth | P3 | 3 business days | **2 business days** |
| Standard | P1 | 8 business hours | **4 business hours** |
| Standard | P2 | 2 business days | **1 business day** |
| Standard | P3 | 3 business days | **2 business days** |

**Every single cell differs.** Any question about a response target has a materially wrong
deprecated answer available in the index. This is a genuine, well-instrumented tier-leakage test.

v3 also *adds* §1 (source precedence) and §4 (escalation duty) which v2 lacks entirely.

## 6. Known issues and the incorrect-history trap

Product Operations Guide (updated 2026-08-14):
- Bulk Upload available on Growth and Enterprise, **up to 5,000 rows**. Not included on Standard.
- `BOOKED` = created, pickup not yet confirmed. `PICKED_UP` = carrier pickup confirmed.
- **KI-208** (opened 2026-08-10, Investigating): intermittent CSV upload failures above ~3,000 rows
  *even though the supported product limit remains 5,000*. Workaround: split below 3,000.
- **KI-211** (opened 2026-08-12, Monitoring): SwiftShip pickup-confirmation webhooks can arrive **up
  to 20 minutes late**. A parcel may be physically collected while ParcelPilot still shows BOOKED.
  "Before telling a customer that a pickup did not occur, verify the carrier status or wait through
  the known delay window."
- **KI-176** Address validation: Resolved 2026-07-18. "Do not use this resolved issue to explain new
  incidents."

Both closed tickets carry a **wrong** historical resolution, and each is refutable from a
higher-tier source:

| Ticket | Historical resolution | Why it is wrong | Refuted by |
|---|---|---|---|
| TKT-450 (ACCT-001, 2026-07-12) | "INR 250 cancellation fee applied after 30 minutes" | Northstar's agreement waives the fee entirely | Northstar §2 (Tier 1) |
| TKT-451 (ACCT-002, 2026-08-11) | "Growth plan only supports 3,000 rows" | The limit is 5,000; ~3,000 is a bug (KI-208), not a plan limit | Product Ops Guide §1 + KI-208 (Tier 3) |

## 7. The buried data conflict — ORD-1001 vs TKT-504

This is the most interesting thing in the pack and the existing architecture doc does not see it.

- `ORD-1001` (ACCT-001, **SwiftShip**, status **BOOKED**, pickup window 10:30-11:30, no
  `pickup_actual_at`, cancellation requested **11:00**).
- `TKT-504` (ACCT-001, opened **10:50**): *"SwiftShip order still shows BOOKED after driver pickup —
  Driver collected the parcel around 10 minutes ago, but ParcelPilot still shows BOOKED."*
- `KI-211`: SwiftShip pickup webhooks arrive up to 20 minutes late.
- ORD-1001 is Northstar's **only** SwiftShip order, so TKT-504 almost certainly refers to it.
  (An inference, not a stated fact — the tickets table has no `order_id` column.)

So the headline question, "Can Northstar cancel ORD-1001 without a cancellation fee?", has a
two-layer answer:

1. **Precedence layer:** BOOKED + Northstar §2 -> no fee, overriding SOP v4 §1's INR 250 after 30
   minutes (the request came 120 minutes after booking).
2. **Reliability layer:** the BOOKED status may be **stale**. If the parcel was collected at ~10:40,
   the order is really PICKED_UP, cancellation is not permitted at all, and return-to-origin applies.

And SOP v4 §3 mandates exactly this behaviour: *"When data conflicts, identify the conflict and
request verification before a state-changing action."*

This means the "unresolved conflict -> verify before acting" path is **required by the corpus and
naturally triggered by the data**. It is not a synthetic edge case.

## 8. Same-tier document conflicts: none exist

Checked exhaustively. Support Policy v3 and SOP v4 are both Tier 2 but cover disjoint subjects
(severity/response vs cancellation/credits). No two same-tier documents disagree.

Therefore an "unresolved same-tier conflict -> escalate" branch would have **no trigger in the
data**. The real conflict class in this pack is the **Tier 0 vs Tier 3 staleness conflict** of §7,
plus **Tier 5 historical resolutions contradicting Tier 1/Tier 3** (§6).

## 9. Tickets have no severity column — severity must be inferred

Columns: `ticket_id, account_id, created_at, status, subject, description, channel, assigned_to,
last_customer_message_at, historical_resolution`. No severity, no priority, no order_id, no
first_response_at, no resolved_at.

Consequences:
- Severity must be **derived** from Support Policy v3 §2 definitions applied to subject/description.
  That is a genuine reasoning step, and it must be shown and sourced, not silently assumed.
- There is no `first_response_at`, so "has the target been met?" cannot be measured. Only
  "**time elapsed since creation vs the target**" is computable. Any SLA claim must be phrased as
  first-response-target risk, not as a measured breach.

Derived severities and clocks at AS_OF 2026-08-16 11:00 (Sunday):

| Ticket | Account | Plan | Derived severity | Governing target | Elapsed vs target |
|---|---|---|---|---|---|
| TKT-505 | ACCT-004 Axis Labs | Enterprise | **P1** — "suspected credential exposure" is named in v3 §2 | Policy v3: 30 min, 24x7 | created 08:30, due 09:00 -> **120 min past** |
| TKT-501 | ACCT-001 Northstar | Enterprise | **P1** — "complete production outage preventing all shipment creation" | Northstar §1: 15 min, 24x7 | created 10:30, due 10:45 -> **15 min past** |
| TKT-504 | ACCT-001 Northstar | Enterprise | **P3 or P2** — matches KI-211, workaround is to wait | Northstar §1: P2 1 h / P3 8 business hours | created 10:50 -> 50 min left if P2 |
| TKT-502 | ACCT-002 LumenWorks | Growth | **P2** — major feature degraded, workaround exists (KI-208) | LumenWorks §1: 4 business hours, no weekend coverage | created Sunday 09:45 -> **business clock not started** |
| TKT-503 | ACCT-003 Beacon | Standard | **P3** — configuration request | Policy v3: 2 business days | created Sunday 10:05 -> clock not started |

Note TKT-501 is the case where the agreement makes things *worse* for ParcelPilot: under standard
Enterprise policy (30 min) it would be due at exactly 11:00 and not yet breached. Under Northstar's
agreement (15 min) it is 15 minutes past. Precedence has teeth in both directions.

TKT-503 is the clean **no-source** case: nothing anywhere in the pack documents how to change a
billing contact. Correct behaviour is to escalate, not improvise.

## 10. Order-level ground truth

Verified arithmetic against AS_OF 2026-08-16 11:00.

| Order | Account | Status | Elapsed booking -> cancel request | Correct answer | Governing clause | Overridden clause |
|---|---|---|---|---|---|---|
| ORD-1001 | ACCT-001 Northstar | BOOKED | **120 min** | **No fee** — but flag possible stale status (see §7) | Northstar §2 | SOP v4 §1 (INR 250 after 30 min) |
| ORD-1002 | ACCT-001 Northstar | PICKED_UP | 130 min | **Cannot cancel; return-to-origin** | SOP v4 §1 + Northstar §2 (agree) | none |
| ORD-2001 | ACCT-002 LumenWorks | BOOKED | **75 min** | **INR 250 fee applies** | SOP v4 §1 | none — LumenWorks §2 explicitly declines to override |
| ORD-3001 | ACCT-003 Beacon | BOOKED | **15 min** | **No fee** (inside the 30-minute window) | SOP v4 §1 | none |
| ORD-4001 | ACCT-004 Axis Labs | DELIVERED | n/a | **Cannot be cancelled** | SOP v4 §1 | none |
| ORD-2002 | ACCT-002 LumenWorks | BOOKED, carrier_fault | n/a | see below | | |

**ORD-1001 vs ORD-2001 is the discriminating pair.** Same shape, both BOOKED, both past 30 minutes,
opposite answers, purely because of the agreement. Any system that hard-codes ORD-1001 fails
ORD-2001, and vice versa.

**ORD-2002 failed-pickup credit** (LumenWorks, `carrier_fault = True`, `customer_fault = False`,
pickup window ended 06:30, still not picked up at 11:00):

- Delay = **4.50 hours** past window end.
- Default SOP v4 §2: >2 h -> lower of INR 500 or 10% of fee = lower(500, 240) = **INR 240**.
- LumenWorks §3: >4 h -> **fixed INR 300**, replacing both amount and threshold.
- 4.50 h > 4 h, so **eligible, INR 300**. Under INR 1,000, so no manager approval needed (SOP §3).

The threshold replacement matters in both directions: at a 3-hour delay LumenWorks would be
**ineligible** (3 < 4) while a no-agreement account like Beacon Retail would be **eligible** (3 > 2).
That is precisely the brief's second example question — *"A pickup is three hours late because of
carrier fault. Should I get a service credit?"* — and its correct answer **depends on who is asking**.
The question as posed is under-specified; the system must scope it to the session account, or ask.

## 11. Model provider verification (run 2026-08-22)

| Check | Result |
|---|---|
| OpenRouter key valid | Yes |
| OpenRouter `/embeddings` endpoint | **Works.** `openai/text-embedding-3-small` returned vectors |
| OpenRouter chat, paid slug | Works (`google/gemini-2.5-flash-lite`) |
| OpenRouter `:free` slugs | **`deepseek/deepseek-chat-v3.1:free` is delisted.** "This model is unavailable for free" |
| OpenRouter structured output (`json_schema`, strict) | Works, returned valid constrained JSON |
| OpenRouter tool calling | Call succeeded but the model declined to invoke the single offered tool with no system prompt. **Needs a real re-test once prompts exist.** |
| OpenRouter credits | **`total_credits: 0`, `is_free_tier: true`** |
| Gemini key valid | Yes |
| Gemini OpenAI-compatible endpoint | Works at `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Gemini current chat slugs | `gemini-2.5-flash` is **404 for new users**; use `gemini-3.6-flash` / `gemini-3.7-flash` / `gemini-3.5-flash-lite` |
| Gemini embedding slugs | `gemini-embedding-001`, `gemini-embedding-2`, `gemini-embedding-2-preview` |
| Chroma Cloud key | Valid. tenant `ac1846d3-...`, **zero databases provisioned** |
| Local chromadb | 1.5.9 importable in `.venv` |

**Operational risks to decide on:**
1. The OpenRouter account has **zero credits**. Paid slugs answered on grace, and `:free` slugs are
   being delisted. A live demo on this account is fragile. Either fund it or make Gemini the
   demo-time provider.
2. Because both providers speak the OpenAI wire format, provider-agnosticism costs almost nothing:
   one client, three settings (`base_url`, `api_key`, `model`). No abstraction layer needed.
3. Model slugs churn fast (two dead slugs found in one afternoon). Slugs belong in config with a
   startup preflight that fails loudly, never hard-coded.

## 12. Answers to `ARCHITECTURE.md` §18 open items

| # | Question | Answer |
|---|---|---|
| 1 | Workbook schema | `README`, `accounts`(8 cols), `orders`(13), `tickets`(10). Join key `account_id`. **No `order_id` on tickets** — order-to-ticket linkage must be inferred |
| 2 | Exact AS_OF | `2026-08-16 11:00 Asia/Kolkata` — a **Sunday** |
| 3 | Which topics each agreement overrides | Enumerated in §4 above, including two explicit non-overrides |
| 4 | v2 vs v3 difference | **All nine target cells differ** (§5). Tier-4 trap is well instrumented |
| 5 | Do tickets carry severity | **No.** Must be inferred from Policy v3 §2. No `first_response_at` either, so only elapsed-vs-target is computable |
| 6 | Any same-tier conflict | **None.** Real conflicts are Tier 0 vs Tier 3 staleness (§7) and Tier 5 vs Tier 1/3 (§6) |
| 7 | Real question shapes | Entitlement decisions, SLA/severity questions, plan-capability questions, known-issue triage, no-source requests (TKT-503), cross-account probes, and the ORD-1001 conflict case |

## 13. Corrections to `project_docs/ARCHITECTURE.md`

| Claim in v1.1 | Correction |
|---|---|
| "~300 chunks" | ~40-60 chunks; corpus is ~5k tokens total |
| "Two same-tier sources disagree -> escalate" | No such case exists. Replace with the Tier 0 staleness conflict, which the SOP explicitly mandates |
| "Tickets carry severity (open item)" | They do not. Severity inference is a required reasoning step |
| Silent on business hours | AS_OF is a Sunday. Business-hours semantics are undefined in the pack and must be an explicit assumption |
| Silent on the ORD-1001 / TKT-504 conflict | This is the single best demonstration of the trust requirement in the pack |
| "Buy $10 of credits" as advice | Account currently has **$0**; `:free` slugs already delisted. This is now a blocking operational item |
| Success metric: escalation precision | Sound, but with 6 orders and 7 tickets the denominator is tiny. Needs a defined evaluation set to be measurable |
