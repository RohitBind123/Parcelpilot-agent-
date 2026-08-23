"""A ParcelPilot server whose model is a script. For the browser tests only.

The Playwright suite drives the real UI against the real API. What it must not
also drive is a real model: a sampled answer makes "does the fact block render
above the prose" an intermittent assertion, and one browser test that waits
ninety seconds for Gemini is worse than no browser test at all.

So this module builds the same app with the same service, and swaps one thing -
the provider. Everything the tests exercise (projection, the gate, the token,
the event stream, `?from_seq=`) is real. Only the sentences are canned.

It lives in `tests/` and is never imported by `src/`. The extra route below
would be indefensible in the product; here it is how a test says what the model
should do next.

    uv run uvicorn tests.support.scripted_server:app --port 8765
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Request

from src.api.app import create_app
from src.api.envelope import ok
from src.api.events import RunBus
from src.api.service import AgentService
from src.auth.sessions import SessionManager
from src.config import get_settings
from src.datastore.runtime import RuntimeStore
from src.providers.base import Completion, ToolCall

SECRET = "playwright-secret"

#: The pending script, shared across runs. Shared rather than copied for the
#: same reason the API tests share theirs: a service builds one provider per
#: run, and a copy would rewind on resume - the model would repropose the same
#: action and the run would interrupt again instead of finishing.
SCRIPT: list[Completion] = []


class ScriptedProvider:
    name = "scripted"

    def __init__(self, script: list[Completion]):
        self.script = script

    def complete(self, messages, *, tools=None, tier="strong", **kwargs):
        if self.script:
            return self.script.pop(0)
        return Completion(text="The script ran out.", model="scripted/test", tool_calls=())

    def complete_structured(self, messages, *, schema, schema_name, tier="cheap"):
        raise NotImplementedError("the browser tests do not exercise structured output")

    def to_assistant_message(self, completion: Completion) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": completion.text}
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


def _to_completion(spec: dict[str, Any]) -> Completion:
    """One step of a script, as JSON a test can post."""
    calls = tuple(
        ToolCall(id=c.get("id", f"c{i}"), name=c["name"], arguments=c.get("arguments", {}))
        for i, c in enumerate(spec.get("tool_calls") or [])
    )
    return Completion(text=spec.get("text", ""), model="scripted/test", tool_calls=calls)


def build() -> Any:
    workspace = Path(os.environ.get("PARCELPILOT_TEST_DIR") or tempfile.mkdtemp())
    store = RuntimeStore.open(workspace / "runtime.db")
    service = AgentService(
        store=store,
        bus=RunBus(store),
        sessions=SessionManager(store, secret=SECRET),
        provider_factory=lambda: ScriptedProvider(SCRIPT),
        checkpoint_path=workspace / "threads.db",
        db_path=get_settings().db_path,
        action_secret=SECRET,
        # No retriever and no extractor: `search_policy` and the grounding gate
        # both need a model, and these tests are about the client. The API
        # tests cover the gate; `tests/integration/test_api.py` is where an
        # ungraded answer would be caught.
    )
    application = create_app(service)

    @application.post("/__test__/script")
    async def set_script(request: Request) -> dict[str, Any]:
        """Replace the pending script. Test-only, and only in this module."""
        body = await request.json()
        SCRIPT[:] = [_to_completion(step) for step in body.get("steps", [])]
        return ok({"steps": len(SCRIPT)})

    return application


app = build()

__all__ = ["SCRIPT", "SECRET", "app", "build"]
