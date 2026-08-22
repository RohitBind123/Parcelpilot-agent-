# ParcelPilot — Project Instructions

Extends the global `~/.claude/CLAUDE.md`. Where they disagree, this file wins.

## What this is

An AI support system for ParcelPilot, a B2B logistics platform. Built for the
CalQuity AI Engineer assessment. Two user contexts (customer and internal ops)
served by one agent with role-scoped tool projection.

**The ground truth is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) v2.0.** Read it
before changing anything structural. `docs/00_ARCHITECTURE_v1_SUPERSEDED.md` is
provenance only — do not implement against it.

**Verified facts about the data live in
[`docs/01_DATA_PACK_FINDINGS.md`](docs/01_DATA_PACK_FINDINGS.md).** Every number in
it was computed from the pack. Do not restate a fact about the data from memory;
check that file, or recompute.

## Branch policy

`main` is the submission branch and is treated as production. Never commit to it
directly. Work happens on `feat/<slug>` or `fix/<slug>`, one branch per milestone,
merged by PR after review. Check `git branch --show-current` before editing.

## Non-negotiables

These are properties of the design, not preferences. Breaking one is a bug even
if tests pass.

1. **There is no "now".** `src/clock.py` is the only time source. Domain time is
   `as_of()`; infrastructure time is `wall_now()`. `datetime.now()`, `date.today()`
   and `time.time()` are banned in `src/` and `app/`, enforced by
   `tests/unit/test_time_discipline.py`. AS_OF is `2026-08-16 11:00 Asia/Kolkata`,
   **a Sunday** — every business-hours target in the pack depends on that.
2. **Access control lives in the tool layer.** Never in a prompt, never in a field
   the client can set. Tools are curried with the Principal at graph-build time, so
   an unauthorised query is absent from the schema rather than refused at runtime.
3. **The model orchestrates; Python decides.** Every number, date, eligibility
   verdict and clause reference is computed in Python and rendered into a fact block
   the model cannot edit. The LLM writes prose around it.
4. **Calculators refuse bare IDs.** They accept only evidence handles minted
   upstream (`snapshot_id`, `resolution_id`). This is what guarantees the multi-step
   chain without scripting it.
5. **Missing data is not zero.** A null price is "unknown", never `INR 0`. A tier-4
   or tier-5 source is never citable as current.
6. **Model slugs live in config, never in code.** Two died in one afternoon.

## Workflow

TDD throughout: write the failing test, implement, refactor. Track work in
`tasks/todo.md` before starting, not after. Commit in reviewable batches by
category — one theme per commit — and run the suite between batches.

```bash
uv pip install -r requirements.txt
uv run python scripts/preflight.py      # verify providers before trusting a run
uv run pytest                            # unit + integration, live tests deselected
uv run pytest -m live                    # hits real providers; needs keys
uv run ruff check . && uv run ruff format --check .
```

## Already in place (global CLAUDE.md Day 1 checklist)

- [x] Frozen clock with business-hours calendar, enforced by a discipline test
- [x] Typed settings; misconfiguration fails at import, not mid-demo
- [x] Provider abstraction with two live implementations and a startup preflight
- [x] Query-embedding cache (SQLite, content-addressed, batch-aware)
- [x] Role-scoped Principal with a canonical scope table
- [x] Coverage gate at 80% (currently 95%)
- [ ] Response envelope and error shape — M8, with the API
- [ ] Background jobs — not needed; runs are in-process and checkpointed
- [ ] Migrations — SQLite is rebuilt from source by `scripts/build_db.py`, not migrated

## Things that will bite you

- **Gemini tool calls carry a `thought_signature`** that must be echoed back on the
  next turn or the API returns 400. Handled by `to_assistant_message`; do not
  reconstruct assistant messages by hand.
- **Always send `max_tokens`.** OpenRouter reserves the requested budget against the
  account balance, so an uncapped request 402s on a low balance.
- **The Chroma collection is namespaced by `{provider}/{model}/{dim}`.** Switching
  embedding provider selects a different collection on purpose.
- **`ruff format` rewrites Python inside Markdown fences.** Markdown is excluded in
  `pyproject.toml`; keep it that way, the docs use illustrative snippets.
- **OpenRouter is unfunded by design (D9a).** It is implemented and switchable, but
  Gemini carries dev, tests and the demo.
