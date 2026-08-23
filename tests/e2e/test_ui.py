"""The Streamlit client in a real browser, against a real API.

Run with `-m ui`, deselected by default in the same way `live` is: these need
Chromium and two servers, and a default `pytest` should not require either.

    uv run playwright install chromium
    uv run pytest -m ui

The model is scripted (`tests/support/scripted_server.py`) and everything else
is real - the projection, the confirmation token, the event stream, the
reattach. A sampled model would turn "does the fact block render above the
prose" into an intermittent assertion, and a browser test that waits ninety
seconds for a completion is worse than none.

What these cover that the reducer tests cannot: that the reducer is actually
wired to the page. `tests/unit/test_ui_state.py` proves a denial survives
folding; only a browser proves somebody can see it.
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.ui

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
sync_playwright = playwright_api.sync_playwright

REPO = Path(__file__).resolve().parents[2]
ORDER_ANSWER = "Your order ORD-1001 is booked and no cancellation fee applies."

ESCALATION_SCRIPT = {
    "steps": [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "prepare_action",
                    "arguments": {
                        "kind": "create_escalation",
                        "payload": {"question": "how do I change the billing contact?"},
                        "evidence_ids": [],
                    },
                }
            ]
        },
        {"text": "I have raised that with a person."},
    ]
}

DENIAL_SCRIPT = {
    "steps": [
        {"tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"order_id": "ORD-1003"}}]},
        {"text": "I could not look that order up on your account."},
    ]
}

#: The same run, slowed down so the in-progress state can be looked at. A
#: scripted model is instant, which makes "did a spinner appear?" untestable
#: without either a real provider's variance or a deliberate pause.
SLOW_SCRIPT = {
    "steps": [
        {
            "tool_calls": [
                {"id": "c1", "name": "get_order", "arguments": {"order_id": "ORD-1001"}}
            ],
            "delay_ms": 1500,
        },
        {"text": ORDER_ANSWER, "delay_ms": 6000},
    ]
}

ANSWER_SCRIPT = {
    "steps": [
        {"tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"order_id": "ORD-1001"}}]},
        {"text": ORDER_ANSWER},
    ]
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for(url: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except httpx.HTTPError:
            time.sleep(0.4)
    raise RuntimeError(f"{url} did not come up within {timeout}s")


@pytest.fixture(scope="module")
def servers(tmp_path_factory) -> Iterator[dict[str, str]]:
    """A scripted API and the Streamlit client, both on free ports."""
    workspace = tmp_path_factory.mktemp("ui")
    api_port, ui_port = free_port(), free_port()
    environment = {
        **dict(__import__("os").environ),
        "PARCELPILOT_TEST_DIR": str(workspace),
        "PYTHONPATH": str(REPO),
    }

    api = subprocess.Popen(
        ["uv", "run", "uvicorn", "tests.support.scripted_server:app", "--port", str(api_port)],
        cwd=REPO,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ui = subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            str(ui_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=REPO,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(f"http://127.0.0.1:{api_port}/healthz")
        wait_for(f"http://127.0.0.1:{ui_port}/")
        yield {
            "api": f"http://127.0.0.1:{api_port}",
            # The client reads its server from `?api=`, which is also how a
            # deployment points the UI at a backend on another host.
            "ui": f"http://127.0.0.1:{ui_port}/?api=http://127.0.0.1:{api_port}",
        }
    finally:
        for process in (ui, api):
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        launched = playwright.chromium.launch()
        yield launched
        launched.close()


@pytest.fixture
def page(browser, servers):
    """A signed-in page with one empty thread, ready for a question."""
    opened = browser.new_page(viewport={"width": 1500, "height": 1200})
    opened.set_default_timeout(30_000)
    opened.goto(servers["ui"], wait_until="networkidle")
    opened.wait_for_timeout(1500)
    yield opened
    opened.close()


def script(servers, body: dict) -> None:
    httpx.post(f"{servers['api']}/__test__/script", json=body, timeout=10.0).raise_for_status()


#: The sidebar is two-level: a context, then an identity within it. The tests
#: name identities by their leading words so a label reworded after the em dash
#: does not break every test in the file.
IDENTITIES = {
    "Northstar Logistics": "Customer",
    "Lumenworks": "Customer",
    "Beacon Retail": "Customer",
    "Axis Freight": "Customer",
    "Maya": "ParcelPilot staff",
    "Rohit": "ParcelPilot staff",
    "Priya": "ParcelPilot staff",
}


def sign_in(page, identity: str = "Northstar Logistics") -> None:
    """Pick the context, then the identity within it, then sign in."""
    context = IDENTITIES[identity]
    page.get_by_test_id("stSidebar").get_by_text(context, exact=True).click()
    page.wait_for_timeout(1200)
    page.get_by_test_id("stSelectbox").click()
    page.wait_for_timeout(500)
    page.get_by_text(identity, exact=False).last.click()
    page.get_by_role("button", name="Sign in", exact=True).click()
    page.wait_for_timeout(2500)


def new_chat(page) -> None:
    page.get_by_role("button", name="New chat", exact=True).click()
    page.wait_for_timeout(2500)


def ask(page, question: str, settle: int = 9000) -> None:
    page.get_by_test_id("stChatInputTextArea").fill(question)
    page.keyboard.press("Enter")
    page.wait_for_timeout(settle)


class TestSigningIn:
    def test_the_page_starts_signed_out(self, page):
        assert "Choose an identity and sign in." in page.inner_text("body")

    def test_signing_in_shows_the_role_the_server_resolved(self, page):
        # The role is not something the client chose. It appears because the
        # server resolved it from the token.
        sign_in(page)
        assert "customer" in page.inner_text("body")
        assert "ACCT-001" in page.inner_text("body")

    def test_a_staff_persona_gets_a_staff_role(self, page):
        sign_in(page, "Maya")
        body = page.inner_text("body")
        # "support agent", not "support_agent": a role is shown to a reader and
        # the underscore is a database detail.
        assert "support agent" in body
        assert "support_agent" not in body
        # Staff carry no account id; the Principal refuses to hold one.
        assert "ACCT-001" not in body


class TestThreads:
    def test_a_new_chat_appears_in_the_sidebar(self, page):
        sign_in(page)
        new_chat(page)
        assert "New conversation" in page.get_by_test_id("stSidebar").inner_text()

    def test_the_chat_input_appears_once_a_thread_is_open(self, page):
        sign_in(page)
        assert page.get_by_test_id("stChatInputTextArea").count() == 0
        new_chat(page)
        assert page.get_by_test_id("stChatInputTextArea").count() == 1


class TestAnsweringAQuestion:
    def test_the_answer_reaches_the_page(self, page, servers):
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")
        assert ORDER_ANSWER in page.inner_text("body")

    def test_the_question_is_echoed_as_the_users_turn(self, page, servers):
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")
        assert "Can I cancel ORD-1001?" in page.inner_text("body")

    def test_the_trace_panel_lists_the_tool_call(self, page, servers):
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")
        page.get_by_text("Trace —", exact=False).click()
        page.wait_for_timeout(700)
        trace = page.inner_text("body")
        assert "get_order" in trace
        assert "ORD-1001" in trace


class TestTheWaitIsNarrated:
    """Nothing is silent between asking and answering.

    A blank assistant bubble for ninety seconds is indistinguishable from a
    broken one, and that is precisely how this looked before: the client had
    every tool event and rendered none of them until the prose arrived.
    """

    def test_a_status_appears_before_any_answer_does(self, page, servers):
        script(servers, SLOW_SCRIPT)
        sign_in(page)
        new_chat(page)
        page.get_by_test_id("stChatInputTextArea").fill("Can I cancel ORD-1001?")
        page.keyboard.press("Enter")
        page.wait_for_timeout(2500)
        body = page.inner_text("body")
        # Something is on screen, and it is not the answer yet.
        assert "…" in body
        assert ORDER_ANSWER not in body

    def test_the_trail_names_the_work_in_plain_words(self, page, servers):
        script(servers, SLOW_SCRIPT)
        sign_in(page)
        new_chat(page)
        page.get_by_test_id("stChatInputTextArea").fill("Can I cancel ORD-1001?")
        page.keyboard.press("Enter")
        page.wait_for_timeout(3500)
        # The reader sees a sentence, not `get_order`. Checked mid-run: the
        # trail is cleared when the run ends, because the Trace expander keeps
        # the detail and a finished answer should not carry its own scaffolding.
        assert "Looking up the order ORD-1001" in page.inner_text("body")

    def test_no_raw_tool_name_is_visible_before_the_trace_is_opened(self, page, servers):
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")
        # `get_order` lives behind the collapsed expander. Anywhere else it is
        # a database identifier shown to somebody asking about a parcel.
        visible = page.inner_text("body")
        assert "get_order" not in visible

    def test_the_status_reports_how_long_it_took(self, page, servers):
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")
        assert "Answered" in page.inner_text("body")


class TestTwoUserContexts:
    """The brief asks for two contexts, and the picker has to show two.

    A flat list of seven identities reads as seven products - which is exactly
    how it was read the first time somebody opened it.
    """

    def test_both_contexts_are_offered(self, page):
        sidebar = page.get_by_test_id("stSidebar").inner_text()
        assert "Customer" in sidebar
        assert "ParcelPilot staff" in sidebar

    def test_the_customer_context_lists_only_customers(self, page):
        page.get_by_test_id("stSelectbox").click()
        page.wait_for_timeout(600)
        options = page.inner_text("body")
        assert "Northstar Logistics" in options
        assert "Maya" not in options

    def test_switching_context_changes_the_identities(self, page):
        page.get_by_text("ParcelPilot staff", exact=True).click()
        page.wait_for_timeout(1200)
        page.get_by_test_id("stSelectbox").click()
        page.wait_for_timeout(600)
        options = page.inner_text("body")
        assert "Maya" in options
        assert "Northstar Logistics" not in options


class TestDenialsAreVisible:
    def test_a_refused_lookup_is_stated_rather_than_swallowed(self, page, servers):
        """ARCHITECTURE 4.4: denials are a demo asset.

        ORD-1003 belongs to another account. The customer's `get_order` has no
        `account_id` parameter, so the refusal comes from the scoped view - and
        the person is told, rather than left with an answer that looks like
        the record does not exist.
        """
        script(servers, DENIAL_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "What is happening with ORD-1003?")
        body = page.inner_text("body")
        # Stated in words. This asserted `get_order` was on the page until the
        # refusal was humanised - the raw name now lives only behind the
        # collapsed trace, which is where a tool identifier belongs.
        assert "Looking up the order ORD-1003 was refused" in body
        assert "not on this account" in body
        assert "get_order" not in body


class TestTheConfirmationCard:
    def test_a_proposed_action_raises_a_card_with_both_buttons(self, page, servers):
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        assert "Confirm:" in page.inner_text("body")
        assert page.get_by_role("button", name="Confirm", exact=True).count() == 1
        assert page.get_by_role("button", name="Cancel", exact=True).count() == 1

    def test_the_preview_names_the_action_and_its_payload(self, page, servers):
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        body = page.inner_text("body")
        assert "Escalation raised" in body.replace(
            "Confirm: escalation raised", "Escalation raised"
        )
        assert "billing contact" in body

    def test_confirming_takes_the_card_down_and_finishes_the_run(self, page, servers):
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        page.get_by_role("button", name="Confirm", exact=True).click()
        page.wait_for_timeout(9000)
        body = page.inner_text("body")
        assert "Confirm:" not in body
        assert "I have raised that with a person." in body

    def test_confirming_leaves_a_receipt(self, page, servers):
        """Confirm and Cancel must not look the same afterwards.

        An action that happens silently is one the person has to go and
        verify, which is the opposite of what a confirmation step is for.
        """
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        page.get_by_role("button", name="Confirm", exact=True).click()
        page.wait_for_timeout(9000)
        body = page.inner_text("body")
        assert "Escalation raised" in body
        assert "Reference" in body

    def test_cancelling_leaves_no_receipt(self, page, servers):
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        page.get_by_role("button", name="Cancel", exact=True).click()
        page.wait_for_timeout(9000)
        assert "Escalation raised" not in page.inner_text("body")

    def test_the_preview_is_labelled_not_dumped_as_json(self, page, servers):
        # The last screen before a change is authorised. `"question":` in a
        # JSON tree is a database key shown at the worst possible moment.
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        body = page.inner_text("body")
        assert "Their question" in body
        assert '"question"' not in body

    def test_cancelling_takes_the_card_down_too(self, page, servers):
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        page.get_by_role("button", name="Cancel", exact=True).click()
        page.wait_for_timeout(9000)
        assert "Confirm:" not in page.inner_text("body")

    def test_the_confirmation_token_is_not_rendered_into_the_page(self, page, servers):
        """The gate's integrity property, checked at the last possible layer.

        The token has to reach the browser - the Confirm button posts it - but
        it must not be printed into the transcript, where a screenshot or a
        copy-paste would carry it somewhere it can be replayed.
        """
        script(servers, ESCALATION_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Escalate the billing contact question")
        body = page.inner_text("body")
        # A token is `nonce.signature`, the signature being 64 hex characters.
        import re

        assert not re.search(r"\b[A-Za-z0-9_-]{20,}\.[0-9a-f]{64}\b", body)


class TestResumeAfterRefresh:
    def test_a_reload_keeps_the_session_and_the_thread(self, page, servers):
        """The session token and thread id live in the URL (§17).

        A refresh mid-conversation is the commonest way to lose state in a
        Streamlit app, and a demo that cannot survive one is a demo nobody
        touches.
        """
        script(servers, ANSWER_SCRIPT)
        sign_in(page)
        new_chat(page)
        ask(page, "Can I cancel ORD-1001?")

        page.reload(wait_until="networkidle")
        page.wait_for_timeout(4000)
        body = page.inner_text("body")
        assert "Choose an identity and sign in." not in body
        # The conversation is read back from the server, not from memory.
        assert "Can I cancel ORD-1001?" in body
