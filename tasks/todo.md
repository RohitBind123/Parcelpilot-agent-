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
- [x] Agree 23 architecture decisions (D1-D23) via structured Q&A
- [x] Write `docs/ARCHITECTURE.md` v2.0 as ground truth
- [x] Mark v1.1 superseded

## Phase 1 — Implementation (not started; no code until instructed)

- [ ] M0  Repo hygiene, config, `clock.py`, provider preflight
- [ ] M1  Data layer: ETL, schema, account-scoped views
- [ ] M2  Clause registry + ingest; Chroma Cloud provisioning; tool-calling verification
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

## Blocking decisions still open

- [ ] Tool-calling reliability on Gemini's OpenAI-compatible endpoint (verify in M2)
- [ ] Hand-review of `params` extraction on ~25 clauses (highest-value review in the build)
- [ ] Chroma Cloud database provisioning + free-tier limits
- [ ] Whether to fund OpenRouter with $10 so the alternate provider is demonstrably live
- [ ] Severity-confidence threshold for escalate-instead-of-answer
- [ ] Railway topology: one service or two (defer to M12)

## Review

_To be filled in as milestones complete._
