"""ParcelPilot, as a person uses it (ARCHITECTURE 17).

    uv run uvicorn src.api.main:app --port 8000
    uv run streamlit run ui/app.py

A thin client. Every number, verdict and clause reference on this page arrived
in an event; nothing here computes one. The identity picker is a login, not a
setting - switching gets a new session token and the server resolves the role
from it, so the tools available change because the projection changed and not
because this file hid a button.

**Two user contexts, several identities.** The brief asks for a system with a
customer-facing context and an internal support/operations context. The picker
is arranged that way because a flat list of seven names reads as seven
products; the contexts are the design, and the identities are how you check
that each context is scoped correctly.

**Nothing is silent while the agent works.** Every event becomes a line of
status in words a reader can use, so the gap between asking and answering is
narrated rather than blank. A blank assistant bubble for ninety seconds is
indistinguishable from a broken one, which is exactly how it looked before.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

if str(Path(__file__).resolve().parents[1]) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.api import ApiError, ParcelPilotClient
from ui.labels import (
    action_kind,
    denial_reason,
    elapsed_label,
    escalation_reason,
    payload_label,
    tool_phrase,
)
from ui.state import AWAITING, FAILED, RunView, apply

#: The two contexts from the brief, and the identities that exercise each.
CONTEXTS: dict[str, dict[str, str]] = {
    "Customer": {
        "northstar_customer": "Northstar Logistics — enterprise agreement",
        "lumenworks_customer": "Lumenworks — service agreement",
        "beacon_customer": "Beacon Retail — standard plan",
        "axis_customer": "Axis Freight — enterprise, no agreement",
    },
    "ParcelPilot staff": {
        "maya_agent": "Maya — support agent",
        "rohit_agent": "Rohit — support agent",
        "priya_manager": "Priya — ops manager",
    },
}

CONTEXT_BLURB = {
    "Customer": "Sees only their own account. Cannot reach another customer's agreement.",
    "ParcelPilot staff": "Sees every account. Only a manager can approve a credit.",
}


# -- session ----------------------------------------------------------------


def client() -> ParcelPilotClient:
    if "client" not in st.session_state:
        base = st.query_params.get("api") or ParcelPilotClient().base_url
        st.session_state.client = ParcelPilotClient(base, token=st.query_params.get("token"))
    return st.session_state.client


def sign_in(persona_id: str) -> None:
    """A new session, not a mutated one.

    The role is server-resolved from the token, so switching identity has to
    mint a new token. Reusing one and asking the UI to pretend would be the one
    thing this client must never do.
    """
    api = client()
    api.logout()
    principal = api.login(persona_id)
    st.session_state.principal = principal
    st.session_state.thread_id = None
    st.session_state.view = None
    st.session_state.active_run = None
    # Mirrored to the URL so a refresh is a reattach rather than a restart.
    st.query_params["token"] = api.token or ""
    st.query_params.pop("thread", None)


def open_thread(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.view = None
    st.session_state.active_run = None
    if thread_id:
        st.query_params["thread"] = thread_id
    else:
        st.query_params.pop("thread", None)


# -- progress ---------------------------------------------------------------


def elapsed_since(started_at: str | None) -> str:
    """How long the run has been going, from the server's start time.

    Anchored to `run.started`, not to a local clock. Streamlit rebuilds this
    whole page on every rerun, and a counter that began when the page did would
    reset to zero each time the script re-ran.
    """
    if not started_at:
        return ""
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return ""
    return elapsed_label((datetime.now(tz=started.tzinfo) - started).total_seconds())


def status_line(view: RunView) -> str:
    activity = view.activity or "Working on it"
    seconds = elapsed_since(view.started_at)
    return f"{activity}… · {seconds}" if seconds else f"{activity}…"


def render_steps(view: RunView) -> None:
    """What has been done so far, one line each, in words not tool names."""
    for step in view.steps:
        phrase = tool_phrase(step.name, step.arguments)
        if phrase is None:
            continue
        if step.outcome == "denied":
            st.markdown(f":grey[✗ {phrase} — {denial_reason(step.detail)}]")
        elif step.outcome == "error":
            st.markdown(f":grey[✗ {phrase}]")
        elif step.outcome == "ok":
            st.markdown(f":grey[✓ {phrase}]")
        else:
            st.markdown(f":grey[◦ {phrase}…]")


# -- rendering --------------------------------------------------------------


def render_facts(facts: dict[str, Any]) -> None:
    """The fact block, above the prose and visibly distinct.

    Every row was computed in Python. Rendering it as a table beside the
    model's sentences is the claim of D15a made visible: these figures are not
    something the answer chose.
    """
    rows = facts.get("rows") or []
    if not rows:
        return
    with st.container(border=True):
        st.caption("Computed from your records — not written by the AI")
        for row in rows:
            left, right = st.columns([1, 3])
            left.markdown(f"**{row.get('label', '')}**")
            right.write(row.get("value", ""))


def render_badge(view: RunView) -> None:
    if not view.has_conflict:
        return
    parts = []
    for conflict in view.conflicts:
        severity = str(conflict.get("severity", "")).upper()
        parts.append(f"**{severity}** {conflict.get('detail')}")
    for resolution in (r for r in view.resolutions if r.get("overridden")):
        # Never render a missing value as itself. An absent governing clause is
        # "not established", not the string "None".
        governing = resolution.get("governing") or "no governing clause established"
        parts.append(
            f"**OVERRIDE** {governing} takes precedence over "
            f"{', '.join(resolution.get('overridden') or [])}"
        )
    st.warning("**Conflicts and overrides**\n\n" + "\n\n".join(f"- {p}" for p in parts))


def render_denials(view: RunView) -> None:
    """Say plainly what was refused (ARCHITECTURE 4.4)."""
    for step in view.denials:
        phrase = tool_phrase(step.name, step.arguments) or "That lookup"
        st.info(f"{phrase} was refused — {denial_reason(step.detail)}.")


def render_trace(view: RunView) -> None:
    """The technical view, for whoever wants it. Raw tool names live here."""
    if not view.steps:
        return
    with st.expander(f"Trace — {len(view.steps)} tool call(s)", expanded=False):
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
                f"**Grounding**: {view.grounding.get('verdict')} — "
                f"{view.grounding.get('claims_total')} claim(s), "
                f"{len(unsupported)} unsupported"
            )
            for claim in unsupported:
                st.caption(f"unsupported: {claim}")


def render_citations(view: RunView) -> None:
    if not view.citations:
        return
    with st.expander(f"Sources — {len(view.citations)}", expanded=False):
        for clause in view.citations:
            document, _, ref = str(clause).partition("::")
            st.markdown(f"- **{document.replace('_', ' ')}** {ref}")


def render_payload(payload: dict[str, Any]) -> None:
    """The action's details, labelled rather than dumped.

    This is the last thing somebody reads before authorising a change, so it
    is not the place for `"unresolved":` and a JSON tree. Keys become headings
    and lists become bullets.
    """
    for key, value in payload.items():
        st.markdown(f"**{payload_label(key)}**")
        if isinstance(value, list | tuple):
            for item in value:
                st.markdown(f"- {item}")
        elif isinstance(value, dict):
            st.json(value, expanded=False)
        else:
            st.markdown(str(value))


def render_executed(view: RunView) -> None:
    """The receipt. Without it, Confirm and Cancel look identical afterwards.

    An action that happened silently is one the person has to go and verify,
    which is the opposite of what a confirmation step is for.
    """
    for record in view.executed:
        reference = record.get("action_id")
        occurred = record.get("occurred_at", "")
        st.success(
            f"**{action_kind(record.get('kind'))}**"
            + (f"  \nReference `{reference}`" if reference else "")
            + (f" · recorded {occurred[:16].replace('T', ' ')}" if occurred else "")
        )


def render_card(view: RunView, api: ParcelPilotClient, run_id: str) -> None:
    """The confirmation card. The graph is genuinely paused behind it."""
    if not view.awaiting_confirmation:
        return
    pending = view.pending or {}
    with st.container(border=True):
        st.subheader(f"Confirm: {action_kind(pending.get('kind')).lower()}")
        st.caption("Nothing happens until you choose. This is the whole record.")
        render_payload(dict(pending.get("payload") or {}))
        for advisory in pending.get("advisories") or []:
            st.warning(advisory)
        confirm, cancel, _ = st.columns([1, 1, 3])
        if confirm.button("Confirm", type="primary", key="confirm", width="stretch"):
            _answer(api, run_id, view, confirm=True)
        if cancel.button("Cancel", key="cancel", width="stretch"):
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
    """Read one stream to its end, narrating it as it goes.

    The status container is created before the first byte arrives, so the wait
    between asking and answering is never a blank bubble. Its label changes
    with each event and its spinner keeps moving between them, which is what
    tells a reader that a thirty-second model call is working rather than hung.
    """
    status = st.status(status_line(view), expanded=False)
    # Beside the status rather than inside it. `status.update()` re-renders the
    # block, which drops anything written into a placeholder nested in it, so
    # the trail lives in its own slot and is cleared when the run ends - the
    # Trace expander keeps the detail for anyone who wants it afterwards.
    steps_slot = st.empty()
    facts_slot = st.empty()
    prose_slot = st.empty()

    for event in api.stream(run_id, from_seq=from_seq):
        view = apply(view, event.seq, event.name, event.data)
        status.update(label=status_line(view))
        with steps_slot.container():
            render_steps(view)
        if view.facts:
            with facts_slot.container():
                render_facts(dict(view.facts))
        if view.prose:
            prose_slot.markdown(view.prose)

    steps_slot.empty()
    took = elapsed_since(view.started_at)
    if view.awaiting_confirmation:
        status.update(label="Waiting for you to confirm", state="complete", expanded=False)
    elif view.status == FAILED:
        status.update(label="Something went wrong", state="error", expanded=True)
    else:
        status.update(
            label=f"Answered in {took}" if took else "Answered",
            state="complete",
            expanded=False,
        )
    return view


# -- page -------------------------------------------------------------------


def sidebar(api: ParcelPilotClient) -> dict[str, Any] | None:  # pragma: no cover
    st.title("ParcelPilot")
    principal = st.session_state.principal

    context = st.radio(
        "User context",
        list(CONTEXTS),
        horizontal=True,
        key="context",
        help="The brief asks for two contexts. The names under each are test identities.",
    )
    st.caption(CONTEXT_BLURB[context])

    identities = CONTEXTS[context]
    chosen = st.selectbox(
        "Sign in as",
        list(identities),
        format_func=lambda pid: identities[pid],
        key="persona",
    )

    signed_in_as = (principal or {}).get("user_id")
    switching = bool(principal) and chosen != signed_in_as
    if st.button(
        f"Switch to {identities[chosen].split(' — ')[0]}" if switching else "Sign in",
        key="signin",
        type="primary" if switching else "secondary",
    ):
        try:
            sign_in(chosen)
            st.rerun()
        except ApiError as exc:
            st.error(f"{exc.code}: {exc}")
    elif api.token and not principal:
        # Restoring a session is not the same action as signing in. Treating
        # them as one lost the thread on every refresh: after a reload
        # `principal` is None, so a "has the identity changed?" test compares
        # against None, decides yes, and mints a new session.
        try:
            st.session_state.principal = principal = api.me()
        except ApiError:
            st.session_state.principal = None
            st.query_params.pop("token", None)

    if not principal:
        st.info("Choose an identity and sign in.")
        return None

    # Who you actually are, which is not always who the picker is showing.
    st.success(
        f"**{principal.get('display_name') or principal['user_id']}**  \n"
        f"{principal['role'].replace('_', ' ')} · "
        f"{principal.get('account_id') or 'all accounts'}"
    )
    st.divider()

    if st.button("New chat", key="new"):
        open_thread(api.create_thread()["thread_id"])
        st.rerun()

    for thread in api.threads():
        row, remove = st.columns([4, 1])
        if row.button(thread["title"][:38], key=f"t_{thread['thread_id']}"):
            open_thread(thread["thread_id"])
            st.rerun()
        if remove.button("✕", key=f"d_{thread['thread_id']}"):
            api.delete_thread(thread["thread_id"])
            if st.session_state.thread_id == thread["thread_id"]:
                open_thread(None)
            st.rerun()
    return principal


SEVERITY_TONE = {"P1": "error", "P2": "warning"}


def render_ops(api: ParcelPilotClient) -> None:  # pragma: no cover - browser-tested
    """The ops page (ARCHITECTURE 14).

    Renders `GET /ops/findings`, which calls the same `scan_support_health` the
    chat calls. Visible only to a role that holds the scope - and the endpoint
    refuses anyway, because a hidden button is a UI preference and not access
    control.
    """
    st.subheader("Support health")
    try:
        report = api.ops_findings()
    except ApiError as exc:
        st.error(f"{exc.code}: {exc}")
        return

    st.caption(report.get("measurability_note", ""))

    findings = report.get("findings") or []
    if not findings:
        st.success("Nothing needs attention.")
    for finding in findings:
        tone = SEVERITY_TONE.get(finding.get("severity"))
        with st.container(border=True):
            headline = finding.get("headline", "")
            if tone == "error":
                st.error(f"**{headline}**")
            elif tone == "warning":
                st.warning(f"**{headline}**")
            else:
                st.info(f"**{headline}**")
            st.write(finding.get("detail", ""))
            if finding.get("suggested_action"):
                st.markdown(f"**What the guide says** — {finding['suggested_action']}")
            if finding.get("evidence"):
                st.caption("Evidence: " + ", ".join(str(e) for e in finding["evidence"] if e))
            if st.button("Ask about this", key=f"ask_{finding['finding_id']}"):
                # Seeds a chat message, so the drill-down is one click and the
                # answer comes from the same tools rather than from this page.
                st.session_state.seeded = f"Explain finding {finding['finding_id']}: {headline}"
                st.session_state.page = "Chat"
                st.rerun()

    with st.expander("What was checked", expanded=False):
        # Signals that found nothing still report. "We looked and there is
        # nothing" and "we did not look" are different statements.
        for entry in report.get("signals") or []:
            mark = "checked" if entry.get("checked") else "not run"
            st.markdown(
                f"- **{entry['signal'].replace('_', ' ')}** — {mark}, "
                f"{entry.get('found', 0)} finding(s)"
                + (f". {entry['note']}" if entry.get("note") else "")
            )


def main() -> None:  # pragma: no cover - exercised by the Playwright suite
    st.set_page_config(page_title="ParcelPilot", page_icon=":package:", layout="wide")
    for key, default in (
        ("principal", None),
        ("thread_id", None),
        ("view", None),
        ("active_run", None),
        ("resume_from", 0),
    ):
        st.session_state.setdefault(key, default)

    st.session_state.setdefault("page", "Chat")
    st.session_state.setdefault("seeded", None)

    api = client()
    with st.sidebar:
        principal = sidebar(api)
        if principal is None:
            st.stop()
        # Only offered to a role that holds the scope. The endpoint refuses
        # regardless - hiding a control is a preference, not access control.
        if "read:ops_detection" in (principal.get("scopes") or []):
            st.divider()
            st.session_state.page = st.radio(
                "View", ["Chat", "Ops"], key="page_choice", horizontal=True
            )

    if st.session_state.page == "Ops":
        render_ops(api)
        return

    thread_id = st.session_state.thread_id or st.query_params.get("thread")
    if not thread_id:
        st.info("Start a new chat from the sidebar.")
        st.stop()
    st.session_state.thread_id = thread_id

    run_id = st.session_state.active_run

    # The run in flight is rendered live below, so its turn is skipped here.
    # Without that the answer appears twice: once replayed from the transcript
    # and once from the stream that is still writing it.
    for message in api.messages(thread_id):
        if run_id and message.get("run_id") == run_id:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Resume: a run left in flight is picked up rather than lost (§17).
    if run_id is None:
        active = api.active_run()
        if active and active["thread_id"] == thread_id:
            st.session_state.active_run = run_id = active["run_id"]
            st.session_state.resume_from = 0
            st.session_state.view = RunView()

    question = st.chat_input("Ask about an order, a ticket, or a policy")
    if st.session_state.seeded:
        question, st.session_state.seeded = st.session_state.seeded, None
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.active_run = run_id = api.send(thread_id, question)
        st.session_state.resume_from = 0
        st.session_state.view = RunView()

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
        render_executed(view)
        render_card(view, api, run_id)
        if view.escalation:
            st.warning(
                f"**Passed to a person** — {escalation_reason(view.escalation.get('reason'))}."
            )
        render_trace(view)
        render_citations(view)

    if view.status == FAILED:
        st.error(view.error or "the run failed")
    if view.status != AWAITING:
        st.session_state.active_run = None
        st.session_state.resume_from = 0


if __name__ == "__main__":  # pragma: no cover
    main()
