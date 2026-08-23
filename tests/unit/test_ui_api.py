"""The client's transport.

Mostly about the SSE parser, because that is the part with a decision in it:
the `id:` field is the server's sequence number and is what a reattach
continues from. A parser that discards it, or a client that counts frames
instead, reintroduces the gap `?from_seq=` exists to close - and it would do so
silently, since counting is right until the first replay overlaps a live stream.

The rest is error shape. A failing call has to surface the envelope's `code`,
not a traceback and not an HTTP number the caller has to interpret.
"""

from __future__ import annotations

import httpx
import pytest

from ui.api import ApiError, ParcelPilotClient, _parse_sse

FRAME = ["id: 7", "event: token.delta", 'data: {"text": "hello"}', ""]


def parse(lines):
    return list(_parse_sse(iter(lines)))


class TestParsingTheStream:
    def test_one_frame_becomes_one_event(self):
        (event,) = parse(FRAME)
        assert (event.seq, event.name, event.data) == (7, "token.delta", {"text": "hello"})

    def test_the_sequence_comes_from_the_id_field(self):
        # Not from a counter. This is the number the client hands back as
        # `from_seq`, and inventing it locally drifts the moment a replay
        # overlaps a live stream.
        events = parse(["id: 12", "event: a", "data: {}", "", "id: 13", "event: b", "data: {}", ""])
        assert [e.seq for e in events] == [12, 13]

    def test_frames_without_an_id_keep_the_previous_one(self):
        # SSE lets a server omit `id`. Resetting to zero would send the next
        # reattach back to the start of the run.
        events = parse(["id: 4", "event: a", "data: {}", "", "event: b", "data: {}", ""])
        assert [e.seq for e in events] == [4, 4]

    def test_data_without_an_event_name_is_skipped(self):
        assert parse(["data: {}"]) == []

    def test_a_comment_or_keepalive_is_ignored(self):
        assert parse([": keep-alive", *FRAME]) == parse(FRAME)

    def test_malformed_json_becomes_an_empty_payload(self):
        # A frame the client cannot read must not take the stream down; the
        # run is still going and the next frame may matter.
        (event,) = parse(["id: 1", "event: token.delta", "data: not json", ""])
        assert event.data == {}

    def test_a_non_object_payload_becomes_an_empty_one(self):
        (event,) = parse(["id: 1", "event: x", "data: [1, 2]", ""])
        assert event.data == {}

    def test_a_non_numeric_id_does_not_break_the_frame(self):
        (event,) = parse(["id: abc", "event: x", "data: {}", ""])
        assert event.seq == 0

    def test_several_frames_come_back_in_order(self):
        lines = []
        for n in range(1, 4):
            lines += [f"id: {n}", "event: token.delta", f'data: {{"text": "{n}"}}', ""]
        assert [e.data["text"] for e in parse(lines)] == ["1", "2", "3"]


class TestErrors:
    def test_an_unreachable_server_is_named_as_such(self):
        api = ParcelPilotClient("http://127.0.0.1:1")
        with pytest.raises(ApiError) as caught:
            api.health()
        assert caught.value.code == "unreachable"

    def test_a_refusal_surfaces_the_envelope_code(self, monkeypatch):
        def fake_request(*_args, **_kwargs):
            return httpx.Response(
                404,
                json={"ok": False, "data": None, "error": {"code": "not_found", "message": "no"}},
            )

        monkeypatch.setattr(httpx, "request", fake_request)
        with pytest.raises(ApiError) as caught:
            ParcelPilotClient().threads()
        assert caught.value.code == "not_found"
        assert caught.value.status == 404

    def test_a_non_json_body_does_not_crash_the_client(self, monkeypatch):
        monkeypatch.setattr(httpx, "request", lambda *a, **k: httpx.Response(502, text="<html>"))
        with pytest.raises(ApiError) as caught:
            ParcelPilotClient().threads()
        assert caught.value.code == "malformed"


class TestAuthHeader:
    def test_no_token_means_no_header(self):
        assert ParcelPilotClient()._headers == {}

    def test_a_token_is_sent_as_a_bearer(self):
        assert ParcelPilotClient(token="abc")._headers == {"Authorization": "Bearer abc"}

    def test_logging_out_forgets_the_token_even_if_the_call_fails(self, monkeypatch):
        # The server may already have dropped the session. Keeping the token
        # afterwards would leave the UI believing it is still signed in.
        monkeypatch.setattr(
            httpx, "request", lambda *a, **k: httpx.Response(401, json={"ok": False, "error": {}})
        )
        api = ParcelPilotClient(token="abc")
        api.logout()
        assert api.token is None

    def test_the_base_url_loses_a_trailing_slash(self):
        assert ParcelPilotClient("http://x/").base_url == "http://x"
