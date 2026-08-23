"""A typed client over the M8 API.

The client speaks HTTP and nothing else. It does not import `src.domain`, does
not compute a figure and does not decide a permission - which is the point of
having built M8 first. A Streamlit app is one process with the whole codebase
importable, so reaching into the agent directly would be easy and would quietly
bypass the projection, the grounding gate and the confirmation token. What is
left after that is a demo of something other than the product.

SSE is parsed by hand rather than with a client library. The parsing is fifteen
lines, and the one field that matters is `id:` - the server's sequence number,
which is what a reattach continues from. A library that discards it, or a
client that counts events instead, reintroduces exactly the gap `?from_seq=`
exists to close.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8000"

#: A run can sit on a slow model for a while, and an SSE read has no natural
#: length. Connect fast, read patiently.
_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)


class ApiError(RuntimeError):
    """The server refused, in its own words.

    Carries the envelope's `code` so the caller can branch on something stable
    while showing the message, which is written for a person and may change.
    """

    def __init__(self, code: str, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    name: str
    data: Mapping[str, Any]


class ParcelPilotClient:
    """One session against one server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    # -- plumbing -----------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                timeout=_TIMEOUT,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            # A server that is not running is the commonest failure while
            # developing, and "connection refused" is more useful than a
            # traceback about a transport.
            raise ApiError("unreachable", f"cannot reach {self.base_url}: {exc}") from exc

        body = _body(response)
        if not body.get("ok", False):
            error = body.get("error") or {}
            raise ApiError(
                error.get("code", "unknown"),
                error.get("message", f"HTTP {response.status_code}"),
                response.status_code,
            )
        return body.get("data")

    # -- auth ---------------------------------------------------------------

    def login(self, persona_id: str) -> dict[str, Any]:
        data = self._request("POST", "/auth/login", json={"persona_id": persona_id})
        self.token = data["session_token"]
        return data["principal"]

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def logout(self) -> None:
        try:
            self._request("POST", "/auth/logout")
        except ApiError:
            # Logging out of a session the server has already forgotten has
            # achieved what the caller wanted.
            logger.debug("logout on an already-invalid session")
        self.token = None

    # -- threads ------------------------------------------------------------

    def threads(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/threads") or [])

    def create_thread(self) -> dict[str, Any]:
        return self._request("POST", "/threads")

    def delete_thread(self, thread_id: str) -> None:
        self._request("DELETE", f"/threads/{thread_id}")

    def messages(self, thread_id: str) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/threads/{thread_id}/messages") or [])

    def send(self, thread_id: str, text: str) -> str:
        data = self._request("POST", f"/threads/{thread_id}/messages", json={"text": text})
        return data["run_id"]

    # -- runs ---------------------------------------------------------------

    def active_run(self) -> dict[str, Any] | None:
        return self._request("GET", "/runs/active")

    def resume(self, run_id: str, *, confirm: bool, token: str = "") -> None:
        self._request("POST", f"/runs/{run_id}/resume", json={"confirm": confirm, "token": token})

    def stream(self, run_id: str, from_seq: int = 0) -> Iterator[Event]:
        """Events from `from_seq` until the server closes the stream.

        The server closes on completion, on failure, and at a confirmation
        pause. A caller that wants what comes after a pause reattaches with the
        last `seq` it saw rather than holding this open.
        """
        url = f"{self.base_url}/runs/{run_id}/events"
        with httpx.stream(
            "GET",
            url,
            params={"from_seq": from_seq},
            headers=self._headers,
            timeout=_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                response.read()
                raise ApiError(
                    "stream_failed", f"HTTP {response.status_code}", response.status_code
                )
            yield from _parse_sse(response.iter_lines())

    def ops_findings(self) -> dict[str, Any]:
        return self._request("GET", "/ops/findings") or {}

    def ops_finding(self, finding_id: str) -> dict[str, Any]:
        return self._request("GET", f"/ops/findings/{finding_id}") or {}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")


# -- helpers ----------------------------------------------------------------


def _body(response: httpx.Response) -> dict[str, Any]:
    try:
        parsed = response.json()
    except ValueError:
        return {"ok": False, "error": {"code": "malformed", "message": response.text[:200]}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": {}}


def _parse_sse(lines: Iterator[str]) -> Iterator[Event]:
    """One `Event` per frame.

    `id` is the server's sequence number and is kept, because it is what a
    reattach continues from. Counting frames on the client instead would drift
    the moment a replay overlaps a live stream.
    """
    seq, name = 0, None
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("id:"):
            seq = _as_int(line[3:].strip(), seq)
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and name:
            yield Event(seq=seq, name=name, data=_as_json(line[5:].strip()))
            name = None


def _as_int(text: str, fallback: int) -> int:
    try:
        return int(text)
    except ValueError:
        return fallback


def _as_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["DEFAULT_BASE_URL", "ApiError", "Event", "ParcelPilotClient"]
