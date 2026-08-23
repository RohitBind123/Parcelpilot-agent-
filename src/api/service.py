"""What the routes talk to, so the routes stay about HTTP.

Holds the runtime store, the event bus, the session manager and whatever is
needed to build an agent - and nothing about requests. Constructed once and
handed to `create_app`, which is also how a test supplies a scripted provider
instead of a real one without patching a module global.

An agent is opened per run rather than kept alive per session. The reason is
the checkpointer: the conversation lives in SQLite, so a fresh agent over the
same thread id resumes exactly where the last one stopped. Keeping one alive
would buy nothing and would make two runs on one thread share a tool context
whose evidence store is keyed by run.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.context import open_agent
from src.api.events import RunBus
from src.api.runner import RunExecutor
from src.auth.principal import Principal
from src.auth.sessions import SessionManager
from src.config import get_settings
from src.datastore.runtime import RunRecord, RuntimeStore

logger = logging.getLogger(__name__)


@dataclass
class AgentService:
    store: RuntimeStore
    bus: RunBus
    sessions: SessionManager
    provider_factory: Callable[[], Any]
    checkpoint_path: Path
    db_path: Path
    action_secret: str
    retriever: Any | None = None
    extractor: Any | None = None
    severity_classifier: Any | None = None
    #: Live run tasks, kept so nothing is garbage-collected mid-flight.
    #: `asyncio` holds only a weak reference to a task, and a run that
    #: disappears halfway leaves a stream waiting for events nobody will emit.
    _tasks: set[asyncio.Task] = field(default_factory=set, repr=False)

    @classmethod
    def build(cls, **overrides: Any) -> AgentService:
        """The service the server runs with everything wired.

        Retrieval, claim extraction and severity inference are built here and
        not left to default to None. A server missing them still answers, and
        that is the problem: `search_policy` reports itself unavailable, and
        `Agent.ask` returns an ungraded answer because an absent extractor
        means the gate did not run. Both degrade quietly into a system that
        looks like the product and is not it.
        """
        settings = get_settings()
        store = overrides.pop("store", None) or RuntimeStore.open()
        defaults: dict[str, Any] = {
            "store": store,
            "bus": RunBus(store),
            "sessions": SessionManager(store, secret=settings.session_secret),
            "provider_factory": _default_provider,
            "checkpoint_path": Path("data/threads.db"),
            "db_path": settings.db_path,
            "action_secret": settings.session_secret,
            "retriever": _build_retriever(settings.db_path),
            "extractor": _build(lambda p: _claim_extractor(p), "claim extractor"),
            "severity_classifier": _build(lambda p: _severity(p), "severity classifier"),
        }
        defaults.update(overrides)
        return cls(**defaults)

    def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        with suppress(Exception):
            self.store.close()

    # -- runs ---------------------------------------------------------------

    def start_run(
        self, *, principal: Principal, session_id: str, thread_id: str, question: str
    ) -> str:
        run_id = f"run_{secrets.token_urlsafe(9)}"
        self.store.create_run(
            run_id=run_id,
            thread_id=thread_id,
            persona_id=principal.user_id,
            question=question,
        )
        self._schedule(
            lambda: self._execute(
                principal=principal,
                session_id=session_id,
                run_id=run_id,
                thread_id=thread_id,
                question=question,
                answer=None,
            )
        )
        return run_id

    def resume_run(
        self,
        *,
        principal: Principal,
        session_id: str,
        record: RunRecord,
        answer: Mapping[str, Any],
    ) -> None:
        self.store.set_run_status(record.run_id, "running")
        self._schedule(
            lambda: self._execute(
                principal=principal,
                session_id=session_id,
                run_id=record.run_id,
                thread_id=record.thread_id,
                question=record.question,
                answer=dict(answer),
            )
        )

    def _schedule(self, work: Callable[[], None]) -> None:
        """Run blocking work off the event loop, keeping a strong reference."""
        task = asyncio.get_running_loop().create_task(asyncio.to_thread(work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _execute(
        self,
        *,
        principal: Principal,
        session_id: str,
        run_id: str,
        thread_id: str,
        question: str,
        answer: Mapping[str, Any] | None,
    ) -> None:
        try:
            with self._open(principal, session_id, thread_id, run_id) as agent:
                executor = RunExecutor(agent=agent, bus=self.bus, store=self.store)
                if answer is None:
                    executor.start(run_id=run_id, thread_id=thread_id, question=question)
                else:
                    executor.resume(
                        run_id=run_id, thread_id=thread_id, question=question, answer=answer
                    )
        except Exception as exc:
            # `RunExecutor` handles its own failures; this catches the ones
            # before it exists, which would otherwise leave a run marked
            # running forever and a stream waiting on it.
            logger.exception("run %s could not start", run_id)
            self.store.set_run_status(run_id, "failed")
            self.bus.emit(run_id, "run.failed", {"error": f"{type(exc).__name__}: {exc}"})

    def _open(self, principal: Principal, session_id: str, thread_id: str, run_id: str):
        return open_agent(
            principal,
            provider=self.provider_factory(),
            db_path=self.db_path,
            checkpoint_path=self.checkpoint_path,
            retriever=self.retriever,
            severity_classifier=self.severity_classifier,
            extractor=self.extractor,
            run_id=run_id,
            runtime=self.store,
            session_id=session_id,
            thread_id=thread_id,
            action_secret=self.action_secret,
        )

    # -- reads --------------------------------------------------------------

    def transcript(self, principal: Principal, thread_id: str) -> list[dict[str, Any]]:
        """The conversation, as a client should see it.

        Tool traffic and the system prompt are dropped: the first is the trace
        panel's business and reaches the client as events, and the second is
        not something a person said.
        """
        with self._open(principal, "replay", thread_id, "replay") as agent:
            history = agent.history(thread_id)
        return [
            {"role": message["role"], "content": message.get("content") or ""}
            for message in history
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]

    def provider_names(self) -> dict[str, str]:
        settings = get_settings()
        return {"llm": settings.llm_provider, "embeddings": settings.embedding_provider}

    def index_identity(self) -> str:
        return get_settings().embedding_config().identity


def _default_provider() -> Any:
    from src.providers.registry import get_chat_provider

    return get_chat_provider()


def _build(make: Callable[[Any], Any], what: str) -> Any | None:
    """Construct a model-backed helper, or None with a warning.

    None is a real state with real consequences - no gate, or no inferred
    severity - so it is logged at warning rather than swallowed. What it must
    not do is stop the server from starting: a provider that is unreachable at
    boot should degrade the answer, not remove the endpoint.
    """
    try:
        return make(_default_provider())
    except Exception as exc:
        logger.warning("%s unavailable (%s: %s)", what, type(exc).__name__, exc)
        return None


def _claim_extractor(provider: Any) -> Any:
    from src.agent.claims_llm import LlmClaimExtractor

    return LlmClaimExtractor(provider)


def _severity(provider: Any) -> Any:
    from src.domain.severity_llm import LlmSeverityClassifier

    return LlmSeverityClassifier(provider)


def _build_retriever(db_path: Path) -> Any | None:
    """The configured hybrid retriever, or None with a warning.

    The same construction `scripts/ask.py` uses, and for the reason its comment
    gives: a stand-in would have `search_policy` report itself unavailable, and
    the answers the server produced would not be the answers the product gives.
    """
    from src.config import get_settings as _settings
    from src.knowledge.registry import load_chunks
    from src.knowledge.retriever import BM25Index, HybridRetriever
    from src.knowledge.vectorstore.chroma import build_vector_store
    from src.providers.registry import get_embedding_provider

    settings = _settings()
    try:
        chunks = load_chunks(db_path)
        dense = build_vector_store(settings, get_embedding_provider(settings))
        if not dense.count():
            logger.warning("the vector collection is empty; run scripts/build_index.py")
        return HybridRetriever(dense=dense, lexical=BM25Index(chunks))
    except Exception as exc:
        logger.warning("policy search unavailable (%s: %s)", type(exc).__name__, exc)
        return None


__all__ = ["AgentService"]
