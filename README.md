# ParcelPilot — AI support system

An AI support system for ParcelPilot, a B2B logistics platform, built for the
CalQuity AI Engineer assessment.

Two user contexts served by one agent: a **customer-facing** assistant that
answers about a customer's own account, and an **internal support/operations**
assistant for ParcelPilot staff. Which tools exist is decided by the caller's
role before the model is asked anything, so the boundary between them is not a
sentence in a prompt.

The source corpus is deliberately imperfect: a deprecated policy alongside a
current one, customer agreements that override general policy, and historical
ticket resolutions that are wrong. The system treats authority as a property of
a source rather than assuming every document is equally reliable.

- **Plain-English walkthrough:** [`docs/EXPLAINER.md`](docs/EXPLAINER.md) — read
  this first if you want to understand how it works end to end.
- **Design decisions:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the
  ground truth for what is built and why.
- **Verified facts about the data:** [`docs/01_DATA_PACK_FINDINGS.md`](docs/01_DATA_PACK_FINDINGS.md)

---

## Quick start

Requires **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/). One provider
key is enough to run everything.

```bash
git clone https://github.com/RohitBind123/Parcelpilot-agent-.git
cd parcelpilot

uv venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

cp .env.example .env                          # then add one API key, see below
uv run python scripts/preflight.py            # verifies the providers answer

./scripts/run_local.sh                        # API on :8000, UI on :8501
```

Open **http://127.0.0.1:8501**, pick an identity in the sidebar, and ask
something.

### The one thing you must configure

`.env` needs a key for whichever LLM provider you point at. Gemini is the
default and carries development, tests and the demo:

```bash
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=...
```

OpenRouter is implemented and switchable (`LLM_PROVIDER=openrouter`); it is
left unfunded on purpose, so it is a one-line change rather than a live path.

Retrieval works against a **committed local Chroma index** out of the box, so
there is nothing to build and no vector database to provision. Set
`VECTOR_STORE=chroma_cloud` with `CHROMA_API_KEY` / `CHROMA_TENANT` if you want
the hosted store instead.

### Nothing to build

`data/parcelpilot.db` and `data/index/` are committed on purpose, so a clone
runs immediately and the demo never embeds or parses a PDF at runtime. To
rebuild them from the source pack:

```bash
uv run python scripts/build_db.py      # PDFs + workbook -> SQLite
uv run python scripts/build_index.py   # clauses -> vector index
```

---

## Try it without the UI

```bash
# One question, one persona, from the terminal
uv run python scripts/ask.py --persona maya_agent \
  --question "Can Northstar cancel ORD-1001 without a fee?"

# The scripted demo tour
uv run python scripts/ask.py --demo

# The HTTP surface, including the confirmation gate and SSE replay
uv run uvicorn src.api.main:app --port 8000
./scripts/demo_m8.sh
```

### Questions worth asking in the UI

| Sign in as | Ask | What it shows |
|---|---|---|
| Northstar Logistics | *Can I cancel ORD-1001 without a fee?* | A Tier 1 agreement overriding the Tier 2 SOP, and a blocking data conflict |
| Maya (support agent) | *TKT-503 asks how to change the billing contact* | Nothing in the corpus covers it — the system declines and drafts an escalation |
| Any customer | *What's happening with ORD-1003?* | A refusal: that order belongs to another account |
| Priya (ops manager) | Switch to the **Ops** view | Ranked findings across every account, and which signals ran |
| Maya vs Priya | anything | Priya has `approve_credit` and the ops scan; Maya does not |

---

## Tests

```bash
uv run pytest                    # 1380 offline tests, no network, no keys
uv run pytest -m live            # hits real providers; needs keys
uv run pytest -m ui              # Chromium against the real UI; see below
uv run ruff check . && uv run ruff format --check .
```

`live` and `ui` are deselected by default, so a plain `pytest` needs neither an
API key nor a browser. For the browser suite:

```bash
uv run playwright install chromium
uv run pytest -m ui
```

---

## Layout

```
src/
  clock.py          the only source of time; AS_OF is frozen and enforced
  config.py         typed settings; misconfiguration fails at import
  auth/             Principal, personas, signed session tokens
  datastore/        SQLite ETL, account-scoped views, runtime state
  knowledge/        PDF -> clauses, typed params, hybrid retrieval
  domain/           resolver, calculators, severity, consistency, detection
  providers/        Gemini and OpenRouter behind one interface
  agent/            tools, graph, fact block, grounding gate, escalation
  api/              FastAPI, SSE, the confirmation gate
ui/                 Streamlit client (thin — talks HTTP, computes nothing)
scripts/            build, demo and operational entry points
tests/              unit, integration, e2e (browser), eval
docs/               EXPLAINER.md, ARCHITECTURE.md, data-pack findings
```

---

## Things worth knowing

- **There is no "now".** `src/clock.py` is the only time source, frozen at
  `2026-08-16 11:00 Asia/Kolkata` — **a Sunday**, which is why business-hours
  targets in the pack behave the way they do. `datetime.now()` is banned in
  `src/`, enforced by a test.
- **Access control is in the tool layer, never the prompt.** Tools are built
  for a Principal before the first model call, so an unauthorised query is
  absent from the schema rather than refused at runtime.
- **The model orchestrates; Python decides.** Every figure, date and verdict is
  computed in Python and rendered into a fact block the model cannot edit.
- **Answers are graded before they are shown.** Claims that the sources do not
  support cause the prose to be dropped and an escalation drafted, rather than
  softened until it passes.
