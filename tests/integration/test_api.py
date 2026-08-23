"""The HTTP surface end to end (ARCHITECTURE 16).

M8 is done when the SSE stream can be curled, confirm and cancel both work, and
`?from_seq=` replays. Those are the four classes below, driven through a real
ASGI client over a scripted model - so what is under test is the wiring, not
whether a sampled model chooses to call a tool.

The access-control tests matter more than the happy paths. Nothing in a request
body names a role or an account, so if a route can be made to act for someone
else it has to be through the token, and these poke at exactly that.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from typing import NamedTuple

import httpx
import pytest

from src.api.app import create_app
from src.api.events import RunBus
from src.api.service import AgentService
from src.auth.sessions import SessionManager
from src.config import get_settings
from src.datastore.runtime import RuntimeStore
from src.providers.base import Completion, ToolCall

SECRET = "api-test-secret"
MODEL = "scripted/test"

ESCALATION = {
    "kind": "create_escalation",
    "payload": {"question": "how do I change the billing contact?"},
    "evidence_ids": [],
}


def say(text: str) -> Completion:
    return Completion(text=text, model=MODEL, tool_calls=())


def call(name: str, call_id: str = "c1", **arguments) -> Completion:
    return Completion(
        text="", model=MODEL, tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),)
    )


class ScriptedProvider:
    """Returns the completions in `script`, in order, across every run.

    Holds the caller's list rather than copying it, and that matters here. A
    service builds one provider per run, so a copied script would rewind on
    resume - the model would propose the same action a second time and the run
    would interrupt again instead of finishing. A real provider is stateless
    across runs; sharing the list is what makes this one behave the same way.
    """

    name = "scripted"

    def __init__(self, script: list[Completion]):
        self.script = script

    def complete(self, messages, *, tools=None, tier="strong", **kwargs):
        return self.script.pop(0) if self.script else say("(the script ran out)")

    def complete_structured(self, messages, *, schema, schema_name, tier="cheap"):
        raise NotImplementedError

    def to_assistant_message(self, completion):
        message = {"role": "assistant", "content": completion.text}
        if completion.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))},
                }
                for c in completion.tool_calls
            ]
        return message


@pytest.fixture
def script() -> list[Completion]:
    """Completions the app's provider will return, in order.

    A list rather than a provider so a test can set the script after the app is
    built - the app builds one provider per run, and the factory reads this.
    """
    return [say("Hello.")]


@pytest.fixture
def service(tmp_path, script) -> Iterator[AgentService]:
    store = RuntimeStore.open(tmp_path / "runtime.db")
    built = AgentService(
        store=store,
        bus=RunBus(store),
        sessions=SessionManager(store, secret=SECRET),
        provider_factory=lambda: ScriptedProvider(script),
        checkpoint_path=tmp_path / "threads.db",
        db_path=get_settings().db_path,
        action_secret=SECRET,
    )
    yield built
    built.close()


@pytest.fixture
async def client(service) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(service)
    # Bound here rather than through the app's lifespan: `ASGITransport` does
    # not run lifespan events, and without the loop the bus falls back to
    # putting events on queues from a worker thread.
    service.bus.bind_loop(asyncio.get_running_loop())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as opened:
        yield opened


async def login(client, persona_id: str = "maya_agent") -> str:
    response = await client.post("/auth/login", json={"persona_id": persona_id})
    return response.json()["data"]["session_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class Ev(NamedTuple):
    seq: int
    name: str
    data: dict


async def events_for(client, token, run_id, from_seq=0) -> list[Ev]:
    """Read one SSE stream to its end, as a client would.

    Never breaks out early. The server closes the stream itself - on
    completion, on failure, or when the run parks awaiting a confirmation -
    and reading to the end is what proves it does. Breaking early would also
    hang here: `ASGITransport` does not deliver the `http.disconnect` that
    sse-starlette waits for.

    The `id:` field carries the sequence number, which is what a browser sends
    back as `Last-Event-ID` and what these tests pass to `from_seq` when they
    reattach.
    """
    collected: list[Ev] = []
    async with client.stream(
        "GET", f"/runs/{run_id}/events?from_seq={from_seq}", headers=auth(token)
    ) as response:
        seq, name = 0, None
        async for line in response.aiter_lines():
            if line.startswith("id:"):
                seq = int(line.split(":", 1)[1].strip())
            elif line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                collected.append(Ev(seq, name, json.loads(line.split(":", 1)[1].strip())))
                name = None
    return collected


def named(seen, name: str) -> list[dict]:
    return [event.data for event in seen if event.name == name]


async def run_to_pause(client, token, text="escalate this", thread="t1"):
    """Post a message and read until the stream closes.

    Returns the run id and everything seen. For a run that proposes an action
    the stream closes at the confirmation pause, which is the flow a real
    client follows before showing the card.
    """
    started = (
        await client.post(f"/threads/{thread}/messages", json={"text": text}, headers=auth(token))
    ).json()["data"]
    seen = await events_for(client, token, started["run_id"])
    return started["run_id"], seen


class TestAuth:
    async def test_login_returns_a_token_and_the_principal(self, client):
        response = await client.post("/auth/login", json={"persona_id": "maya_agent"})
        body = response.json()
        assert body["ok"] is True
        assert body["data"]["principal"]["role"] == "support_agent"

    async def test_an_unknown_persona_is_refused(self, client):
        response = await client.post("/auth/login", json={"persona_id": "root"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    async def test_an_unauthenticated_request_is_refused(self, client):
        assert (await client.get("/auth/me")).status_code == 401

    async def test_a_forged_token_is_refused(self, client, service):
        forged = SessionManager(service.store, secret="not-the-secret").login("priya_manager")
        response = await client.get("/auth/me", headers=auth(forged))
        assert response.status_code == 401

    async def test_logging_out_revokes_the_token(self, client):
        token = await login(client)
        await client.post("/auth/logout", headers=auth(token))
        assert (await client.get("/auth/me", headers=auth(token))).status_code == 401

    async def test_the_body_cannot_name_a_role(self, client):
        # There is no field for it, so this is really asserting that adding one
        # would be a change somebody has to make deliberately.
        response = await client.post(
            "/auth/login", json={"persona_id": "northstar_customer", "role": "ops_manager"}
        )
        assert response.json()["data"]["principal"]["role"] == "customer"


class TestThreads:
    async def test_a_thread_can_be_created_and_listed(self, client):
        token = await login(client)
        created = (await client.post("/threads", headers=auth(token))).json()["data"]
        listed = (await client.get("/threads", headers=auth(token))).json()["data"]
        assert created["thread_id"] in {t["thread_id"] for t in listed}

    async def test_another_persona_cannot_see_it(self, client):
        mine = await login(client, "maya_agent")
        theirs = await login(client, "priya_manager")
        created = (await client.post("/threads", headers=auth(mine))).json()["data"]
        listed = (await client.get("/threads", headers=auth(theirs))).json()["data"]
        assert created["thread_id"] not in {t["thread_id"] for t in listed}

    async def test_another_persona_cannot_delete_it(self, client):
        mine = await login(client, "maya_agent")
        theirs = await login(client, "priya_manager")
        created = (await client.post("/threads", headers=auth(mine))).json()["data"]
        response = await client.delete(f"/threads/{created['thread_id']}", headers=auth(theirs))
        # Same answer as a thread that does not exist. The difference would
        # make this an existence oracle for other people's conversations.
        assert response.status_code == 404

    async def test_deleting_a_thread_removes_it(self, client):
        token = await login(client)
        created = (await client.post("/threads", headers=auth(token))).json()["data"]
        await client.delete(f"/threads/{created['thread_id']}", headers=auth(token))
        listed = (await client.get("/threads", headers=auth(token))).json()["data"]
        assert created["thread_id"] not in {t["thread_id"] for t in listed}


class TestStreamingARun:
    async def test_a_run_streams_to_completion(self, client):
        token = await login(client)
        started = (
            await client.post("/threads/t1/messages", json={"text": "hello"}, headers=auth(token))
        ).json()["data"]
        names = [e.name for e in await events_for(client, token, started["run_id"])]
        assert names[0] == "run.started"
        assert names[-1] == "run.completed"

    async def test_the_answer_arrives_as_token_deltas(self, client, script):
        script[:] = [say("The cancellation fee is waived.")]
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "fee?"}, headers=auth(token))
        ).json()["data"]["run_id"]
        deltas = [d["text"] for d in named(await events_for(client, token, run), "token.delta")]
        assert "".join(deltas) == "The cancellation fee is waived."

    async def test_a_tool_call_is_narrated(self, client, script):
        script[:] = [call("get_order", order_id="ORD-1001"), say("It is in transit.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "where is ORD-1001?"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        names = [e.name for e in await events_for(client, token, run)]
        assert "tool.started" in names
        assert "tool.finished" in names

    async def test_another_persona_cannot_read_the_stream(self, client):
        mine = await login(client, "maya_agent")
        theirs = await login(client, "priya_manager")
        run = (
            await client.post("/threads/t1/messages", json={"text": "hi"}, headers=auth(mine))
        ).json()["data"]["run_id"]
        response = await client.get(f"/runs/{run}/events", headers=auth(theirs))
        assert response.status_code == 404


class TestReplay:
    async def test_from_seq_replays_only_what_came_after(self, client, script):
        script[:] = [say("A complete answer that will be split into several deltas.")]
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "hi"}, headers=auth(token))
        ).json()["data"]["run_id"]

        whole = await events_for(client, token, run)
        assert len(whole) > 3

        # Reattaching the way a browser would after losing the connection.
        resumed = await events_for(client, token, run, from_seq=2)
        assert [e.name for e in resumed] == [e.name for e in whole[2:]]

    async def test_replaying_a_finished_run_terminates(self, client):
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "hi"}, headers=auth(token))
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        # The run is over. A stream that waited for more would hang forever.
        assert (await events_for(client, token, run, from_seq=0))[-1].name == "run.completed"


class TestTheConfirmationGate:
    async def test_a_proposal_pauses_the_run_and_offers_a_token(self, client, script):
        script[:] = [call("prepare_action", **ESCALATION), say("Raised.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, run)
        (payload,) = named(seen, "interrupt.await_confirm")
        assert payload["preview"]["kind"] == "create_escalation"
        assert payload["token"]

    async def test_confirming_executes_and_the_run_completes(self, client, script, service):
        script[:] = [call("prepare_action", **ESCALATION), say("Raised.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, run)
        confirmation = named(seen, "interrupt.await_confirm")[0]["token"]

        response = await client.post(
            f"/runs/{run}/resume",
            json={"confirm": True, "token": confirmation},
            headers=auth(token),
        )
        assert response.status_code == 200
        # Reattach past the pause, exactly as the client does after the person
        # answers. Replaying from zero would stop at the interrupt again,
        # because that is where the first stream closed.
        rest = await events_for(client, token, run, from_seq=seen[-1].seq)
        assert [e.name for e in rest][-1] == "run.completed"
        assert len(service.store.actions_for_thread("t1")) == 1

    async def test_cancelling_executes_nothing(self, client, script, service):
        script[:] = [call("prepare_action", **ESCALATION), say("Nothing was done.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, run)

        await client.post(f"/runs/{run}/resume", json={"confirm": False}, headers=auth(token))
        rest = await events_for(client, token, run, from_seq=seen[-1].seq)
        assert [e.name for e in rest][-1] == "run.completed"
        assert service.store.actions_for_thread("t1") == ()

    async def test_the_confirmation_token_is_never_in_the_transcript(self, client, script):
        # The client is given the token on the interrupt event. The model's
        # conversation must not contain it, or the gate is decorative.
        script[:] = [call("prepare_action", **ESCALATION), say("Raised.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, run)
        confirmation = named(seen, "interrupt.await_confirm")[0]["token"]

        transcript = (await client.get("/threads/t1/messages", headers=auth(token))).json()
        assert confirmation not in json.dumps(transcript)

    async def test_resuming_a_run_that_is_not_waiting_is_refused(self, client):
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "hi"}, headers=auth(token))
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        response = await client.post(
            f"/runs/{run}/resume", json={"confirm": True, "token": "x"}, headers=auth(token)
        )
        assert response.status_code == 409

    async def test_another_persona_cannot_confirm_it(self, client, script, service):
        script[:] = [call("prepare_action", **ESCALATION), say("Raised.")]
        mine = await login(client, "maya_agent")
        theirs = await login(client, "priya_manager")
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(mine)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(client, mine, run)
        confirmation = named(seen, "interrupt.await_confirm")[0]["token"]

        response = await client.post(
            f"/runs/{run}/resume",
            json={"confirm": True, "token": confirmation},
            headers=auth(theirs),
        )
        assert response.status_code == 404
        assert service.store.actions_for_thread("t1") == ()


class TestTheTranscriptIsWhatWasDelivered:
    """The gate has to survive a page reload.

    `GET /threads/{id}/messages` used to read the checkpointer, which holds the
    model's draft. When the grounding gate declines, the draft is dropped and
    an escalation summary is delivered instead - so the transcript handed back
    the exact prose the gate had refused, one page-load later. The gate held
    live and evaporated on refresh.
    """

    async def test_the_transcript_matches_the_streamed_answer(self, client, script):
        script[:] = [say("The cancellation fee is waived under your agreement.")]
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "fee?"}, headers=auth(token))
        ).json()["data"]["run_id"]
        streamed = "".join(
            d["text"] for d in named(await events_for(client, token, run), "token.delta")
        )

        transcript = (await client.get("/threads/t1/messages", headers=auth(token))).json()["data"]
        assistant = [m for m in transcript if m["role"] == "assistant"]
        assert [m["content"] for m in assistant] == [streamed]

    async def test_the_question_is_the_one_that_was_asked(self, client):
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "where is ORD-1001?"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        transcript = (await client.get("/threads/t1/messages", headers=auth(token))).json()["data"]
        # `run_id` rides along so a client watching a run live can leave that
        # turn out and avoid rendering the answer twice.
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "where is ORD-1001?"
        assert transcript[0]["run_id"] == run

    async def test_turns_come_back_in_order(self, client, script):
        script[:] = [say("First answer."), say("Second answer.")]
        token = await login(client)
        for text in ("one", "two"):
            run = (
                await client.post("/threads/t1/messages", json={"text": text}, headers=auth(token))
            ).json()["data"]["run_id"]
            await events_for(client, token, run)
        transcript = (await client.get("/threads/t1/messages", headers=auth(token))).json()["data"]
        assert [m["content"] for m in transcript] == [
            "one",
            "First answer.",
            "two",
            "Second answer.",
        ]

    async def test_no_tool_traffic_leaks_into_the_transcript(self, client, script):
        script[:] = [call("get_order", order_id="ORD-1001"), say("It is booked.")]
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "status?"}, headers=auth(token))
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        transcript = (await client.get("/threads/t1/messages", headers=auth(token))).json()["data"]
        assert {m["role"] for m in transcript} == {"user", "assistant"}
        assert not any("evidence" in m["content"] for m in transcript)


class TestADeclinedAnswerStaysDeclined:
    """The regression, asserted directly rather than transitively.

    A gate that drops prose live and hands it back on the next page load has
    not dropped it. This drives a run whose draft the gate refuses, and checks
    the refused sentence is nowhere in the transcript.
    """

    DRAFT = "Your cancellation fee is waived and a refund lands in three days."

    @pytest.fixture
    def graded(self, tmp_path, script):
        """A service whose gate rejects everything it is given."""

        class RejectingExtractor:
            def extract(self, prose):
                # One claim, worded so nothing in the corpus could support it.
                return ["a refund lands in three days"]

        store = RuntimeStore.open(tmp_path / "graded.db")
        built = AgentService(
            store=store,
            bus=RunBus(store),
            sessions=SessionManager(store, secret=SECRET),
            provider_factory=lambda: ScriptedProvider(script),
            checkpoint_path=tmp_path / "graded-threads.db",
            db_path=get_settings().db_path,
            action_secret=SECRET,
            extractor=RejectingExtractor(),
        )
        yield built
        built.close()

    @pytest.fixture
    async def graded_client(self, graded):
        app = create_app(graded)
        graded.bus.bind_loop(asyncio.get_running_loop())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as opened:
            yield opened

    async def test_the_gate_declines_the_draft(self, graded_client, script):
        script[:] = [call("get_order", order_id="ORD-1001"), say(self.DRAFT)]
        token = await login(graded_client)
        run = (
            await graded_client.post(
                "/threads/t1/messages", json={"text": "fee?"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        seen = await events_for(graded_client, token, run)
        assert named(seen, "run.escalated")
        delivered = "".join(d["text"] for d in named(seen, "token.delta"))
        assert self.DRAFT not in delivered

    async def test_the_refused_draft_is_absent_from_the_transcript(self, graded_client, script):
        script[:] = [call("get_order", order_id="ORD-1001"), say(self.DRAFT)]
        token = await login(graded_client)
        run = (
            await graded_client.post(
                "/threads/t1/messages", json={"text": "fee?"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        await events_for(graded_client, token, run)

        transcript = (await graded_client.get("/threads/t1/messages", headers=auth(token))).json()[
            "data"
        ]
        # The whole point. Reading this from the checkpointer returned the
        # draft, so the gate's decision lasted exactly until a refresh.
        assert self.DRAFT not in json.dumps(transcript)
        assert any("do not have a source" in m["content"] for m in transcript)


class TestEvidenceSurvivesTheTurn:
    """A conversation is one investigation, not a series of strangers.

    Evidence was scoped per message-run over an in-memory store, so the second
    turn in a thread could not read the first turn's handles at all. The fact
    block silently lost its Governing and Overridden rows, a true claim about
    the override became unsupported, and the answer was dropped and escalated -
    on a question as innocuous as "are you hallucinating".
    """

    async def test_a_later_turn_can_read_an_earlier_turns_handle(self, client, script, service):
        script[:] = [
            call("get_order", order_id="ORD-1001"),
            say("It is booked."),
            say("Still booked."),
        ]
        token = await login(client)
        first = (
            await client.post("/threads/t1/messages", json={"text": "status?"}, headers=auth(token))
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, first)
        handle = next(
            (d.get("evidence_id") for d in named(seen, "tool.finished") if d.get("evidence_id")),
            None,
        )
        assert handle, "the first turn minted no evidence"

        second = (
            await client.post(
                "/threads/t1/messages", json={"text": "and now?"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        await events_for(client, token, second)

        # The store the second run was given must still hold the first's work.
        with service._open((await _principal(client, token)), "sid", "t1", second) as agent:
            assert agent.tool_context.store.kind_of(handle) is not None

    async def test_another_thread_cannot_read_it(self, client, script, service):
        # Scoping by conversation is not the same as no scoping.
        script[:] = [call("get_order", order_id="ORD-1001"), say("Booked.")]
        token = await login(client)
        run = (
            await client.post("/threads/a/messages", json={"text": "status?"}, headers=auth(token))
        ).json()["data"]["run_id"]
        seen = await events_for(client, token, run)
        handle = next(
            (d.get("evidence_id") for d in named(seen, "tool.finished") if d.get("evidence_id")),
            None,
        )
        assert handle

        from src.domain.evidence import EvidenceError

        with (
            service._open((await _principal(client, token)), "sid", "b", "r2") as agent,
            pytest.raises(EvidenceError),
        ):
            agent.tool_context.store.kind_of(handle)


async def _principal(client, token):
    from src.auth.personas import get_persona, to_principal

    body = (await client.get("/auth/me", headers=auth(token))).json()["data"]
    return to_principal(get_persona(body["user_id"]))


class TestResumeFlow:
    async def test_active_run_is_null_when_nothing_is_running(self, client):
        token = await login(client)
        assert (await client.get("/runs/active", headers=auth(token))).json()["data"] is None

    async def test_a_paused_run_is_reported_as_active(self, client, script):
        script[:] = [call("prepare_action", **ESCALATION), say("Raised.")]
        token = await login(client)
        run = (
            await client.post(
                "/threads/t1/messages", json={"text": "escalate this"}, headers=auth(token)
            )
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        active = (await client.get("/runs/active", headers=auth(token))).json()["data"]
        assert active["run_id"] == run
        assert active["status"] == "awaiting_confirmation"

    async def test_a_completed_run_is_not_active(self, client):
        token = await login(client)
        run = (
            await client.post("/threads/t1/messages", json={"text": "hi"}, headers=auth(token))
        ).json()["data"]["run_id"]
        await events_for(client, token, run)
        assert (await client.get("/runs/active", headers=auth(token))).json()["data"] is None


class TestHealth:
    async def test_healthz_reports_the_frozen_clock_and_the_index(self, client):
        body = (await client.get("/healthz")).json()["data"]
        assert body["status"] == "ok"
        assert body["as_of"].startswith("2026-08-16")
        assert body["index_identity"]


class TestTheEnvelope:
    async def test_every_success_has_the_same_shape(self, client):
        token = await login(client)
        for path in ("/auth/me", "/threads", "/runs/active", "/healthz"):
            body = (await client.get(path, headers=auth(token))).json()
            assert set(body) >= {"ok", "data", "error"}
            assert body["ok"] is True and body["error"] is None

    async def test_every_failure_carries_a_stable_code(self, client):
        body = (
            await client.get("/threads/nope/messages", headers=auth(await login(client)))
        ).json()
        assert body["ok"] is False
        assert body["error"]["code"] == "not_found"

    async def test_a_sqlite_row_never_leaks_into_a_response(self, client):
        # Rows are dataclasses by the time they reach a route; this catches a
        # future shortcut that returns one straight from a cursor.
        token = await login(client)
        listed = (await client.get("/threads", headers=auth(token))).json()["data"]
        assert all(not isinstance(item, sqlite3.Row) for item in listed)
