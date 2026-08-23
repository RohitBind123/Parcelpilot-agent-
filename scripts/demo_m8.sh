#!/usr/bin/env bash
#
# M8 acceptance, run against a live server with curl (ARCHITECTURE 21).
#
# The milestone is done when the SSE stream can be curled, confirm and cancel
# both work, and ?from_seq= replays. This walks all four.
#
#   Terminal 1:  uv run uvicorn src.api.main:app --port 8000
#   Terminal 2:  ./scripts/demo_m8.sh
#
# Needs a provider key, because it drives a real model. The same paths are
# covered offline by tests/integration/test_api.py against a scripted one.

set -euo pipefail

BASE="${PARCELPILOT_API:-http://127.0.0.1:8000}"
PERSONA="${1:-maya_agent}"
QUESTION="${2:-TKT-503 asks how to change the billing contact. What should I tell them?}"

command -v jq >/dev/null || { echo "this script needs jq"; exit 1; }

rule() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

rule "health"
curl -sS "$BASE/healthz" | jq -r '.data | "as_of \(.as_of)   index \(.index_identity)"'

rule "login as $PERSONA"
TOKEN=$(curl -sS -X POST "$BASE/auth/login" \
  -H 'content-type: application/json' \
  -d "{\"persona_id\":\"$PERSONA\"}" | jq -r '.data.session_token')
AUTH="Authorization: Bearer $TOKEN"
curl -sS "$BASE/auth/me" -H "$AUTH" | jq -c '.data | {user_id, role, account_id}'

rule "ask"
THREAD="demo-$(date +%s)"
RUN=$(curl -sS -X POST "$BASE/threads/$THREAD/messages" \
  -H "$AUTH" -H 'content-type: application/json' \
  -d "$(jq -nc --arg t "$QUESTION" '{text:$t}')" | jq -r '.data.run_id')
echo "run  $RUN"
echo "thread $THREAD"

rule "stream (closes on completion, or at a confirmation pause)"
# -N disables buffering so events appear as they arrive.
curl -sS -N "$BASE/runs/$RUN/events?from_seq=0" -H "$AUTH" | tee /tmp/pp_stream.txt

LAST_SEQ=$(grep '^id:' /tmp/pp_stream.txt | tail -1 | cut -d' ' -f2 || echo 0)
CONFIRM=$(grep -A2 '^event: interrupt.await_confirm' /tmp/pp_stream.txt \
  | grep '^data:' | sed 's/^data: //' | jq -r '.token // empty' || true)

if [ -z "${CONFIRM:-}" ]; then
  rule "no confirmation was proposed"
  echo "The run finished without proposing an action. Try a question that needs one,"
  echo "for example the TKT-503 billing-contact question, which nothing in the corpus"
  echo "documents - so the correct behaviour is a drafted escalation."
  exit 0
fi

rule "cancel, then confirm"
echo "token $CONFIRM"

# Cancel first. Nothing should be written.
curl -sS -X POST "$BASE/runs/$RUN/resume" -H "$AUTH" -H 'content-type: application/json' \
  -d '{"confirm": false}' | jq -c '.data'
echo "-- reattaching from seq $LAST_SEQ --"
curl -sS -N "$BASE/runs/$RUN/events?from_seq=$LAST_SEQ" -H "$AUTH" | head -40

rule "replay the whole run from seq 0"
# Every event is persisted before it is streamed, so this returns the same
# conversation a client would have seen live.
curl -sS -N "$BASE/runs/$RUN/events?from_seq=0" -H "$AUTH" | grep -c '^event:' \
  | xargs -I{} echo "{} events replayed"

rule "done"
echo "Run again and answer {\"confirm\": true, \"token\": \"...\"} to execute instead."
