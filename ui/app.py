"""ParcelPilot, as a person uses it (ARCHITECTURE 17).

    uv run uvicorn src.api.main:app --port 8000
    uv run streamlit run ui/app.py

A thin client. Every number, verdict and clause reference on this page arrived
in an event; nothing here computes one. The persona picker is a login, not a
setting - switching persona gets a new session token and the server resolves
the role from it, so the tools available change because the projection changed
and not because this file hid a button.

Two things are deliberately prominent. Denials are shown rather than swallowed,
because "your role cannot see that" is the visible edge of the access-control
design. And the conflict badge is loud whenever an override fired or a
consistency conflict was found, which is the most legible demonstration that
precedence is being resolved rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

if str(Path(__file__).resolve().parents[1]) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.api import ApiError, ParcelPilotClient
from ui.state import AWAITING, FAILED, RunView, apply

PERSONAS: list[tuple[str, str]] = [
    ("northstar_customer", "Northstar Logistics (customer)"),
    ("lumenworks_customer", "Lumenworks (customer)"),
    ("beacon_customer", "Beacon Retail (customer)"),
    ("axis_customer", "Axis Freight (customer)"),
    ("maya_agent", "Maya (support agent)"),
    ("rohit_agent", "Rohit (support agent)"),
    ("priya_manager", "Priya (ops manager)"),
]

TIER_LABELS = {
    1: "Tier 1 - agreement",
    2: "Tier 2 - policy / SOP",
    3: "Tier 3 - product doc",
    4: "Tier 4 - deprecated",
    5: "Tier 5 - historical",
}


# -- session ----------------------------------------------------------------


def client() -> ParcelPilotClient:
    if "client" not in st.session_state:
        base = st.query_params.get("api") or ParcelPilotClient().base_url
        st.session_state.client = ParcelPilotClient(base, token=st.query_params.get("token"))
    return st.session_state.client


def sign_in(persona_id: str) -> None:
    """A new session, not a mutated one.

    The role is server-resolved from the token, so switching persona has to
    mint a new token. Reusing one and asking the UI to pretend would be the
    one thing this client must never do.
    """
    api = client()
    api.logout()
    principal = api.login(persona_id)
    st.session_state.principal = principal
    st.session_state.thread_id = None
    st.session_state.view = None
    # Mirrored to the URL so a refresh is a reattach rather than a restart.
    st.query_params["token"] = api.token or ""
    st.query_params.pop("thread", None)


def open_thread(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.view = None
    if thread_id:
        st.query_params["thread"] = thread_id
    else:
        st.query_params.pop("thread", None)


# -- rendering --------------------------------------------------------------


def render_facts(facts: dict[str, Any]) -> None:
    """The fact block, above the prose and visibly distinct.

    Every row was computed in Python. Rendering it as a table beside the
    model's sentences is the whole claim of D15a made visible: the figures are
    not something the answer chose.
    """
    rows = facts.get("rows") or []
    if not rows:
        return
    with st.container(border=True):
        st.caption("Computed facts - not written by the model")
        for row in rows:
            left, right = st.columns([1, 2])
            left.markdown(f"**{row.get('label', '')}**")
            right.write(row.get("value", ""))


def render_badge(view: RunView) -> None:
    if not view.has_conflict:
        return
    overrides = [r for r in view.resolutions if r.get("overridden")]
    parts = []
    for conflict in view.conflicts:
        severity = str(conflict.get("severity", "")).upper()
        parts.append(f"**{severity}** {conflict.get('conflict_class')}: {conflict.get('detail')}")
    for resolution in overrides:
        # Never render a missing value as itself. A governing clause that is
        # absent is "not established", not the string "None" - the same rule
        # that keeps a null price from being displayed as zero.
        governing = resolution.get("governing") or "no governing clause established"
        parts.append(
            f"**OVERRIDE** {governing} displaces {', '.join(resolution.get('overridden') or [])}"
        )
    st.warning("Conflicts and overrides\n\n" + "\n\n".join(f"- {p}" for p in parts))


def render_denials(view: RunView) -> None:
    """Say plainly what was refused (ARCHITECTURE 4.4).

    A denial that fails silently looks like a system that could not find
    something. This one could, and was not allowed to.
    """
    for step in view.denials:
        st.info(f"`{step.name}` was not available to this role - {step.detail}")


def render_trace(view: RunView) -> None:
    if not view.steps:
        return
    with st.expander(f"Trace - {len(view.steps)} tool call(s)", expanded=False):
        for step in view.steps:
            marker = {"ok": "OK", "denied": "DENIED", "error": "ERROR", "running": "..."}[
                step.outcome
            ]
            arguments = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
            st.markdown(f"`{marker}` **{step.name}**({arguments})")
            if step.evidence_id:
                st.caption(f"evidence: {step.evidence_id}")
            if step.detail:
                st.caption(step.detail)
        if view.grounding:
            unsupported = view.grounding.get("unsupported") or []
            st.markdown(
                f"**Grounding**: {view.grounding.get('verdict')} - "
                f"{view.grounding.get('claims_total')} claim(s), "
                f"{len(unsupported)} unsupported"
            )
            for claim in unsupported:
                st.caption(f"unsupported: {claim}")


def render_citations(view: RunView) -> None:
    if not view.citations:
        return
    with st.expander(f"Citations - {len(view.citations)}", expanded=False):
        for clause in view.citations:
            document, _, ref = str(clause).partition("::")
            st.markdown(f"- **{document.replace('_', ' ')}** {ref}")


def render_card(view: RunView, api: ParcelPilotClient, run_id: str) -> None:
    """The confirmation card. The graph is genuinely paused behind it."""
    if not view.awaiting_confirmation:
        return
    pending = view.pending or {}
    with st.container(border=True):
        st.subheader("Confirm this action")
        st.markdown(f"**{str(pending.get('kind', '')).replace('_', ' ').title()}**")
        st.json(pending.get("payload") or {}, expanded=True)
        for advisory in pending.get("advisories") or []:
            st.warning(advisory)
        confirm, cancel = st.columns(2)
        if confirm.button("Confirm", type="primary", key="confirm"):
            _answer(api, run_id, view, confirm=True)
        if cancel.button("Cancel", key="cancel"):
            _answer(api, run_id, view, confirm=False)


def _answer(api: ParcelPilotClient, run_id: str, view: RunView, *, confirm: bool) -> None:
    api.resume(run_id, confirm=confirm, token=view.confirm_token or "")
    # Reattach from where the stream closed, not from zero. Replaying from the
    # start would stop at the same pause again.
    st.session_state.resume_from = view.last_seq
    st.session_state.active_run = run_id
    st.rerun()


# -- the run loop -----------------------------------------------------------


def consume(api: ParcelPilotClient, run_id: str, from_seq: int, view: RunView) -> RunView:
    """Read one stream to its end, rendering as it goes."""
    facts_slot = st.empty()
    prose_slot = st.empty()
    for event in api.stream(run_id, from_seq=from_seq):
        view = apply(view, event.seq, event.name, event.data)
        if view.facts:
            with facts_slot.container():
                render_facts(dict(view.facts))
        if view.prose:
            prose_slot.markdown(view.prose)
    return view


# -- page -------------------------------------------------------------------


def main() -> None:  # pragma: no cover - exercised by the Playwright suite
    st.set_page_config(page_title="ParcelPilot", page_icon=":package:", layout="wide")
    st.session_state.setdefault("principal", None)
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("view", None)
    st.session_state.setdefault("active_run", None)
    st.session_state.setdefault("resume_from", 0)

    api = client()

    with st.sidebar:
        st.title("ParcelPilot")
        labels = {pid: label for pid, label in PERSONAS}
        current = (st.session_state.principal or {}).get("user_id")
        chosen = st.selectbox(
            "Signed in as",
            [pid for pid, _ in PERSONAS],
            index=[pid for pid, _ in PERSONAS].index(current) if current else 0,
            format_func=lambda pid: labels[pid],
            key="persona",
        )
        # Signing in and restoring a session are different actions, and
        # conflating them loses the thread on every refresh: after a reload
        # `principal` is None, so any "has the persona changed?" test compares
        # against None, decides yes, and calls `sign_in` - which mints a new
        # session and clears the thread from the URL. A click signs in; a page
        # load with a token restores.
        if st.button("Sign in", key="signin"):
            try:
                sign_in(chosen)
            except ApiError as exc:
                st.error(f"{exc.code}: {exc}")
        elif api.token and not st.session_state.principal:
            try:
                st.session_state.principal = api.me()
            except ApiError:
                # The token in the URL is stale. Fall back to signed out
                # rather than showing a half-restored page.
                st.session_state.principal = None
                st.query_params.pop("token", None)

        principal = st.session_state.principal
        if not principal:
            st.info("Choose a persona and sign in.")
            st.stop()

        st.caption(f"{principal['role']}  |  {principal.get('account_id') or 'all accounts'}")
        st.divider()

        if st.button("New chat", key="new"):
            open_thread(api.create_thread()["thread_id"])
            st.rerun()

        for thread in api.threads():
            row, remove = st.columns([4, 1])
            if row.button(thread["title"][:38], key=f"t_{thread['thread_id']}"):
                open_thread(thread["thread_id"])
                st.rerun()
            if remove.button("x", key=f"d_{thread['thread_id']}"):
                api.delete_thread(thread["thread_id"])
                if st.session_state.thread_id == thread["thread_id"]:
                    open_thread(None)
                st.rerun()

    thread_id = st.session_state.thread_id or st.query_params.get("thread")
    if not thread_id:
        st.info("Start a new chat from the sidebar.")
        st.stop()
    st.session_state.thread_id = thread_id

    for message in api.messages(thread_id):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Resume: a run left in flight is picked up rather than lost (§17).
    if st.session_state.active_run is None:
        active = api.active_run()
        if active and active["thread_id"] == thread_id:
            st.session_state.active_run = active["run_id"]
            st.session_state.resume_from = 0

    question = st.chat_input("Ask about an order, a ticket, or a policy")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.active_run = api.send(thread_id, question)
        st.session_state.resume_from = 0
        st.session_state.view = RunView()

    run_id = st.session_state.active_run
    if not run_id:
        return

    view = st.session_state.view or RunView()
    with st.chat_message("assistant"):
        try:
            view = consume(api, run_id, st.session_state.resume_from, view)
        except ApiError as exc:
            st.error(f"{exc.code}: {exc}")
            st.session_state.active_run = None
            return

    st.session_state.view = view
    render_badge(view)
    render_denials(view)
    render_card(view, api, run_id)
    render_trace(view)
    render_citations(view)

    if view.escalation:
        st.warning(f"Escalated: {view.escalation.get('reason')}")
    if view.status == FAILED:
        st.error(view.error or "the run failed")
    if view.status != AWAITING:
        st.session_state.active_run = None
        st.session_state.resume_from = 0


if __name__ == "__main__":  # pragma: no cover
    main()
