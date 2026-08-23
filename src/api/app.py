"""The HTTP surface (ARCHITECTURE 16).

Two things about this file are worth reading before the routes.

**Nothing in a request body names a role or an account.** Every authenticated
route resolves its Principal from the bearer token through the sessions table
(D17). There is no code path that reads an identity from the client, which is
what makes "forging a staff session requires forging a signature" true rather
than a claim - and it is why `login` takes a persona id but no scopes.

**Runs are in-process and checkpointed.** A run executes in a worker thread and
narrates itself to the bus; the SSE route reads the bus. There is no queue
because there is nothing to lose: the conversation is in the checkpointer and
the events are in `run_events`, so a process that dies mid-run is recovered by
reattaching, not by replaying a job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.api.envelope import ApiError, ErrorCode, failed, ok
from src.api.service import AgentService
from src.auth.principal import Principal
from src.auth.sessions import InvalidToken, UnknownPersona
from src.clock import as_of

logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    persona_id: str = Field(min_length=1, max_length=64)


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ResumeRequest(BaseModel):
    confirm: bool
    token: str = ""


def get_service(request: Request) -> AgentService:
    return request.app.state.service


ServiceDep = Annotated[AgentService, Depends(get_service)]


def principal_for(
    service: ServiceDep,
    authorization: Annotated[str | None, Header()] = None,
) -> tuple[Principal, str]:
    """The caller's Principal, resolved server-side from the bearer token.

    Returns the session id alongside it because confirmation tokens are bound
    to the session, and reading it from anywhere else would be a second source
    of truth for who is asking.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError.unauthenticated()
    token = authorization.split(" ", 1)[1].strip()
    try:
        return service.sessions.resolve(token), service.sessions.peek(token)["sid"]
    except InvalidToken as exc:
        raise ApiError.unauthenticated(str(exc)) from exc


CallerDep = Annotated[tuple[Principal, str], Depends(principal_for)]

# These four live at module scope on purpose. `from __future__ import
# annotations` turns every annotation into a string, and FastAPI resolves those
# against the module namespace - so an `Annotated[...]` alias defined inside the
# factory is invisible to it, and the dependency is silently reinterpreted as a
# query parameter. The symptom is a 422 naming a field nobody declared.


def create_app(service: AgentService | None = None) -> FastAPI:
    """Build the app. A service can be supplied so tests need no real provider."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service.bus.bind_loop(asyncio.get_running_loop())
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="ParcelPilot", version="1.0", lifespan=lifespan)
    app.state.service = service or AgentService.build()

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        # The message never carries the exception. It reaches a browser, and
        # a traceback there is both unhelpful and a disclosure.
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=failed(ErrorCode.INTERNAL, "something went wrong and has been logged"),
        )

    # -- auth ---------------------------------------------------------------

    @app.post("/auth/login")
    def login(body: LoginRequest, service: ServiceDep) -> dict[str, Any]:
        try:
            token = service.sessions.login(body.persona_id)
        except UnknownPersona as exc:
            raise ApiError.invalid(str(exc)) from exc
        principal = service.sessions.resolve(token)
        return ok({"session_token": token, "principal": _public(principal)})

    @app.get("/auth/me")
    def me(caller: CallerDep) -> dict[str, Any]:
        return ok(_public(caller[0]))

    @app.post("/auth/logout")
    def logout(service: ServiceDep, authorization: Annotated[str | None, Header()] = None):
        if authorization and authorization.lower().startswith("bearer "):
            service.sessions.logout(authorization.split(" ", 1)[1].strip())
        return ok({"logged_out": True})

    # -- threads ------------------------------------------------------------

    @app.get("/threads")
    def list_threads(caller: CallerDep, service: ServiceDep) -> dict[str, Any]:
        principal, _ = caller
        found = service.store.threads_for(principal.user_id)
        return ok([thread.to_payload() for thread in found])

    @app.post("/threads")
    def create_thread(caller: CallerDep, service: ServiceDep) -> dict[str, Any]:
        principal, _ = caller
        thread = service.store.upsert_thread(
            thread_id=f"th_{secrets.token_urlsafe(9)}",
            persona_id=principal.user_id,
            title="New conversation",
        )
        return ok(thread.to_payload())

    @app.delete("/threads/{thread_id}")
    def delete_thread(thread_id: str, caller: CallerDep, service: ServiceDep) -> dict[str, Any]:
        _owned_thread(service, caller, thread_id)
        service.store.delete_thread(thread_id)
        return ok({"deleted": thread_id})

    @app.get("/threads/{thread_id}/messages")
    def thread_messages(thread_id: str, caller: CallerDep, service: ServiceDep) -> dict[str, Any]:
        principal, _ = caller
        _owned_thread(service, caller, thread_id)
        return ok(service.transcript(principal, thread_id))

    @app.post("/threads/{thread_id}/messages")
    async def post_message(
        thread_id: str, body: MessageRequest, caller: CallerDep, service: ServiceDep
    ) -> dict[str, Any]:
        principal, session_id = caller
        thread = service.store.get_thread(thread_id)
        if thread is None:
            thread = service.store.upsert_thread(
                thread_id=thread_id, persona_id=principal.user_id, title=body.text[:80]
            )
        elif thread.persona_id != principal.user_id:
            raise ApiError.not_found("no such conversation")
        else:
            service.store.upsert_thread(
                thread_id=thread_id, persona_id=principal.user_id, title=thread.title
            )

        run_id = service.start_run(
            principal=principal,
            session_id=session_id,
            thread_id=thread_id,
            question=body.text,
        )
        return ok({"run_id": run_id, "thread_id": thread_id})

    # -- runs ---------------------------------------------------------------

    @app.get("/runs/active")
    def active_run(caller: CallerDep, service: ServiceDep) -> dict[str, Any]:
        principal, _ = caller
        found = service.store.active_run_for(principal.user_id)
        return ok(found.to_payload() if found else None)

    @app.get("/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        caller: CallerDep,
        service: ServiceDep,
        from_seq: Annotated[int, Query(ge=0)] = 0,
    ) -> EventSourceResponse:
        _owned_run(service, caller, run_id)

        async def stream() -> AsyncIterator[dict[str, Any]]:
            async for record in service.bus.subscribe(run_id, from_seq=from_seq):
                yield {
                    "id": str(record.seq),
                    "event": record.event,
                    "data": _dump(record.payload),
                }

        return EventSourceResponse(stream())

    @app.post("/runs/{run_id}/resume")
    async def resume_run(
        run_id: str, body: ResumeRequest, caller: CallerDep, service: ServiceDep
    ) -> dict[str, Any]:
        principal, session_id = caller
        record = _owned_run(service, caller, run_id)
        if record.status != "awaiting_confirmation":
            raise ApiError.conflict(f"run {run_id} is not waiting for a confirmation")
        service.resume_run(
            principal=principal,
            session_id=session_id,
            record=record,
            answer={"confirm": body.confirm, "token": body.token},
        )
        return ok({"run_id": run_id, "resumed": True})

    # -- health -------------------------------------------------------------

    @app.get("/healthz")
    def healthz(service: ServiceDep) -> dict[str, Any]:
        return ok(
            {
                "status": "ok",
                "as_of": as_of().isoformat(),
                "providers": service.provider_names(),
                "index_identity": service.index_identity(),
            }
        )

    return app


# -- helpers ----------------------------------------------------------------


def _public(principal: Principal) -> dict[str, Any]:
    """What a client may know about itself.

    Scopes are included because the UI hides what it cannot use, and omitting
    them would have the client infer the role from the presence of a feature.
    They are not authorisation: every route re-resolves the Principal.
    """
    return {
        "user_id": principal.user_id,
        "display_name": principal.display_name,
        "role": principal.role,
        "account_id": principal.account_id,
        "scopes": sorted(principal.scopes),
    }


def _owned_thread(service: AgentService, caller: tuple[Principal, str], thread_id: str) -> Any:
    thread = service.store.get_thread(thread_id)
    # One message for "no such thread" and "not yours". The difference would
    # make this an existence oracle for other people's conversations.
    if thread is None or thread.persona_id != caller[0].user_id:
        raise ApiError.not_found("no such conversation")
    return thread


def _owned_run(service: AgentService, caller: tuple[Principal, str], run_id: str) -> Any:
    record = service.store.get_run(run_id)
    if record is None or record.persona_id != caller[0].user_id:
        raise ApiError.not_found("no such run")
    return record


def _dump(payload: Any) -> str:
    return json.dumps(payload, default=str)


__all__ = ["create_app"]
