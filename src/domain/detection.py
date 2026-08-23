"""Proactive issue detection (ARCHITECTURE 14, Problem 1).

One implementation, two surfaces. The ops page and the chat drill-down both
call this; a second code path for "the same question asked somewhere else" is
how a page and a conversation come to disagree, and the disagreement is always
found by a customer rather than by us.

**Why this is not clustering.** The pack ships seven tickets. Embedding
clustering is unstable and spike detection is meaningless at that n, so the
primary signal is matching tickets against the Known Issues document - stable
at low volume, explainable, and it ties the detection problem back to the
corpus the rest of the system reasons over. A cluster labelled by a known issue
is actionable; an unlabelled one is a shape on a chart.

**Signals that find nothing still report.** No known issue currently spans two
accounts, so the cross-account signal returns empty - and says so, rather than
being omitted. "We looked and there is nothing" and "we did not look" are
different statements, and a dashboard that cannot tell them apart is a
dashboard that quietly stops working.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from src.clock import as_of
from src.datastore.models import Account, Ticket
from src.domain.severity import (
    SeverityDefinitions,
    SeverityVerdict,
    deterministic_severity,
    infer_severity,
    load_severity_definitions,
)

logger = logging.getLogger(__name__)

#: Severity ordering for ranking. P1 first.
_SEVERITY_RANK: Final = {"P1": 0, "P2": 1, "P3": 2, None: 3}

#: Statuses that mean a ticket is still somebody's problem.
OPEN_STATUSES: Final[frozenset[str]] = frozenset({"open", "pending", "in_progress"})

#: Below this many tickets, a rate is not a rate. §14 suppresses the volume
#: signal rather than reporting noise dressed as a trend.
VOLUME_SIGNAL_MINIMUM: Final = 30


class Signal(StrEnum):
    KNOWN_ISSUE_RECURRENCE = "known_issue_recurrence"
    FIRST_RESPONSE_RISK = "first_response_risk"
    SEVERITY_CONCENTRATION = "severity_concentration"
    CROSS_ACCOUNT_IMPACT = "cross_account_impact"
    UNMATCHED_HIGH_SEVERITY = "unmatched_high_severity"
    HISTORICAL_CONTRADICTION = "historical_contradiction"
    VOLUME_SPIKE = "volume_spike"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing worth an operator's attention."""

    finding_id: str
    signal: Signal
    subject_id: str
    subject_kind: str
    account_id: str | None
    headline: str
    detail: str
    severity: str | None = None
    #: Minutes past the governing first-response target, negative if not yet
    #: due. None when no target was resolved.
    past_target_minutes: int | None = None
    #: Never True. There is no `first_response_at` column, so elapsed-versus-
    #: target is computable and a met-or-missed target is not (A4/A5).
    measurable: bool = False
    confidence: float = 1.0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    #: What the corpus says to do, verbatim where it says anything.
    suggested_action: str | None = None

    @property
    def rank(self) -> tuple[int, int, str]:
        """Worst first: severity, then how far past target, then id for ties.

        A stable third key matters more than it looks. Two P1s equally past
        target would otherwise reorder between requests, and an ops list that
        shuffles is one nobody trusts to have been read.
        """
        return (
            _SEVERITY_RANK.get(self.severity, 3),
            -(self.past_target_minutes or 0),
            self.finding_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "signal": self.signal.value,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "account_id": self.account_id,
            "headline": self.headline,
            "detail": self.detail,
            "severity": self.severity,
            "past_target_minutes": self.past_target_minutes,
            "measurable": self.measurable,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "suggested_action": self.suggested_action,
        }


@dataclass(frozen=True, slots=True)
class SignalReport:
    """What one signal looked for, and what it found.

    `checked` is the point. A signal that returns nothing is indistinguishable
    from a signal that never ran unless it says which, and §14 predicts that
    cross-account impact finds nothing on this pack.
    """

    signal: Signal
    checked: bool
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    note: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "signal": self.signal.value,
            "checked": self.checked,
            "found": len(self.findings),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ScanReport:
    findings: tuple[Finding, ...]
    signals: tuple[SignalReport, ...]
    as_of: str

    def find(self, finding_id: str) -> Finding | None:
        return next((f for f in self.findings if f.finding_id == finding_id), None)

    def to_payload(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "findings": [f.to_payload() for f in self.findings],
            "signals": [s.to_payload() for s in self.signals],
            "measurability_note": (
                "Targets are reported as time elapsed against the governing target. "
                "Tickets carry no first-response timestamp, so a met or missed target "
                "cannot be measured from this data."
            ),
        }


# -- known-issue matching ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnownIssue:
    clause_id: str
    reference: str
    title: str
    text: str
    status: str | None
    #: Terms that identify this issue, taken from its **title** and its
    #: extracted params rather than its whole body.
    #:
    #: The body is prose and mentions things in passing. KI-208 is about bulk
    #: upload and says "fails to create shipments" along the way, which was
    #: enough to attribute TKT-501 - a total shipment-creation outage with
    #: nothing to do with CSVs - to it. A title is curated, short, and is the
    #: one place the corpus states what an issue *is*.
    terms: frozenset[str]

    @property
    def is_active(self) -> bool:
        """A resolved issue explains nothing about a ticket raised today.

        KI-176 is resolved and says so in its own text; matching a live ticket
        against it would attribute a current problem to a fixed one.
        """
        return (self.status or "").lower() not in {"resolved", "closed"}


_REFERENCE: Final = re.compile(r"\b(KI-\d+)\b")
_WORD: Final = re.compile(r"[a-z0-9]{4,}")

#: Words that match everything and therefore identify nothing.
_NOISE: Final[frozenset[str]] = frozenset(
    {
        "issue",
        "issues",
        "known",
        "status",
        "opened",
        "customer",
        "customers",
        "parcelpilot",
        "support",
        "workaround",
        "resolved",
        "investigating",
        "monitoring",
        "should",
        "would",
        "before",
        "after",
        "still",
        "shows",
        "their",
        "there",
        "which",
        "while",
        "about",
        "under",
        "these",
        "those",
        "using",
        "please",
        "guide",
        "product",
        "operations",
    }
)

#: How many distinguishing terms a ticket must share with an issue to be called
#: a match. Two is enough at this corpus size and is the difference between
#: "bulk upload CSV" and any ticket that happens to say "upload".
MATCH_TERMS: Final = 2


def build_known_issues(rows: Iterable[Mapping[str, Any]]) -> tuple[KnownIssue, ...]:
    """Known issues from clause rows, with their identifying terms."""
    issues = []
    for row in rows:
        clause_id = str(row["clause_id"])
        text = str(row.get("text") or "")
        reference = _REFERENCE.search(clause_id) or _REFERENCE.search(text)
        params = row.get("params") or {}
        issues.append(
            KnownIssue(
                clause_id=clause_id,
                reference=reference.group(1) if reference else clause_id,
                title=str(row.get("title") or ""),
                text=text,
                status=params.get("issue_status"),
                terms=_terms(row.get("title") or "") | _terms(_param_words(params)),
            )
        )
    return tuple(issues)


def _terms(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _NOISE)


def _param_words(params: Mapping[str, Any]) -> str:
    """String param values, which name carriers and features by hand.

    `{"carrier": "SwiftShip"}` is a reviewed extraction, so the word is as
    load-bearing as anything in the title.
    """
    return " ".join(str(v) for v in params.values() if isinstance(v, str))


#: A term appearing in more than this share of tickets describes the domain,
#: not an issue.
_COMMON_SHARE: Final = 0.5


def common_terms(tickets: Sequence[Ticket], share: float = _COMMON_SHARE) -> frozenset[str]:
    """Words too widespread in this corpus to identify anything.

    IDF, computed over the tickets actually in hand rather than assumed. It
    exists because the first version of the matcher attributed TKT-501 - a
    total shipment-creation outage - to KI-208, the bulk-upload issue, on the
    strength of the words "shipment" and "creation". Both are perfectly good
    English and neither distinguishes one ticket from any other here.
    """
    if not tickets:
        return frozenset()
    counts: dict[str, int] = {}
    for ticket in tickets:
        for word in _terms(f"{ticket.subject} {ticket.description or ''}"):
            counts[word] = counts.get(word, 0) + 1
    ceiling = max(1, int(len(tickets) * share))
    return frozenset(word for word, seen in counts.items() if seen > ceiling)


@dataclass(frozen=True, slots=True)
class Match:
    """Which issue, and the words that decided it.

    The terms are carried because "why is this ticket attributed to KI-208?" is
    the first question an operator asks, and "the vectors were close" is not an
    answer. `explain_finding` hands these back.
    """

    issue: KnownIssue | None
    terms: tuple[str, ...] = ()
    by_reference: bool = False

    @property
    def score(self) -> int:
        return len(self.terms)


def match_issue(
    ticket: Ticket, issues: Sequence[KnownIssue], common: frozenset[str] = frozenset()
) -> Match:
    """The known issue this ticket looks like, on distinguishing words only.

    Term overlap rather than embeddings. At seven tickets a semantic model adds
    variance without adding signal, and a match an operator can read the reason
    for is worth more than one they have to trust.
    """
    words = _terms(f"{ticket.subject} {ticket.description or ''}") - common
    best = Match(issue=None)
    for issue in issues:
        if not issue.is_active:
            continue
        # An explicit reference in the ticket beats any amount of overlap.
        if issue.reference.lower() in (ticket.description or "").lower():
            return Match(issue=issue, terms=(issue.reference,), by_reference=True)
        shared = tuple(sorted(words & (issue.terms - common)))
        if len(shared) > best.score:
            best = Match(issue=issue, terms=shared)
    return best if best.score >= MATCH_TERMS else Match(issue=None, terms=best.terms)


# -- the scan ---------------------------------------------------------------


class HealthScanner:
    """Runs every signal over the open tickets and ranks what comes back."""

    def __init__(
        self,
        *,
        repository: Any,
        resolver: Any,
        severity_classifier: Any | None = None,
        sla: Any | None = None,
    ) -> None:
        self._repo = repository
        self._resolver = resolver
        self._classifier = severity_classifier
        self._sla = sla

    def scan(self, *, limit: int = 50) -> ScanReport:
        every = self._repo.query_tickets(limit=limit)
        tickets = [t for t in every if t.status in OPEN_STATUSES]
        issues = build_known_issues(self._known_issue_rows())
        definitions = self._definitions()

        graded = [(t, self._severity(t, definitions)) for t in tickets]
        # Common terms are computed over every ticket, and prior occurrences
        # over the closed ones too: "second time in five days" is the finding,
        # and it is invisible if history is filtered out before anyone looks.
        widespread = common_terms(every)
        matched = {t.ticket_id: match_issue(t, issues, widespread) for t in every}

        reports = [
            self._recurrence(graded, matched, issues, every),
            self._target_risk(graded),
            self._concentration(graded),
            self._cross_account(graded, matched),
            self._unmatched_high_severity(graded, matched),
            self._historical_contradiction(
                tickets, [t for t in every if t.status not in OPEN_STATUSES]
            ),
            self._volume(tickets),
        ]
        findings = sorted((f for report in reports for f in report.findings), key=lambda f: f.rank)
        return ScanReport(
            findings=tuple(findings), signals=tuple(reports), as_of=as_of().isoformat()
        )

    # -- inputs -------------------------------------------------------------

    def _known_issue_rows(self) -> list[dict[str, Any]]:
        import json

        from src.domain.resolver import KNOWN_ISSUE_TOPIC

        rows = self._repo.connection.execute(
            "SELECT c.clause_id, c.title, c.text, c.params FROM clauses c "
            "JOIN clause_topics t ON t.clause_id = c.clause_id WHERE t.topic = ?",
            (KNOWN_ISSUE_TOPIC,),
        ).fetchall()
        found = []
        for row in rows:
            try:
                params = json.loads(row["params"] or "{}")
            except (TypeError, ValueError):
                params = {}
            found.append(
                {
                    "clause_id": row["clause_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "params": params,
                }
            )
        return found

    def _definitions(self) -> SeverityDefinitions | None:
        try:
            return load_severity_definitions(self._repo.connection)
        except Exception as exc:
            logger.warning("severity definitions unavailable: %s", exc)
            return None

    def _severity(self, ticket: Ticket, definitions: SeverityDefinitions | None) -> SeverityVerdict:
        guard = deterministic_severity(ticket.subject, ticket.description or "")
        if guard or definitions is None:
            return guard or SeverityVerdict(
                severity=None, confidence=0.0, basis_clause=None, basis=""
            )
        return infer_severity(
            ticket.subject,
            ticket.description or "",
            definitions=definitions,
            classifier=self._classifier,
        )

    def _account(self, account_id: str) -> Account | None:
        try:
            return self._repo.get_account(account_id)
        except Exception:
            return None

    # -- signals ------------------------------------------------------------

    def _recurrence(
        self,
        graded: Sequence[tuple[Ticket, SeverityVerdict]],
        matched: Mapping[str, Match],
        issues: Sequence[KnownIssue],
        history: Sequence[Ticket],
    ) -> SignalReport:
        if not issues:
            return SignalReport(
                Signal.KNOWN_ISSUE_RECURRENCE,
                checked=False,
                note="no known-issue clauses are indexed",
            )
        found = []
        for ticket, verdict in graded:
            match = matched.get(ticket.ticket_id) or Match(issue=None)
            issue = match.issue
            if issue is None:
                continue
            earlier = tuple(
                other.ticket_id
                for other in history
                if other.ticket_id != ticket.ticket_id
                and other.created_at < ticket.created_at
                and (matched.get(other.ticket_id) or Match(issue=None)).issue is issue
            )
            occurrence = len(earlier) + 1
            found.append(
                Finding(
                    finding_id=f"kir_{ticket.ticket_id.lower()}",
                    signal=Signal.KNOWN_ISSUE_RECURRENCE,
                    subject_id=ticket.ticket_id,
                    subject_kind="ticket",
                    account_id=ticket.account_id,
                    headline=(
                        f"{ticket.ticket_id} matches {issue.reference} — {issue.title}"
                        + (f" ({_ordinal(occurrence)} occurrence)" if earlier else "")
                    ),
                    detail=(
                        f"{ticket.ticket_id} ({ticket.subject}) matches known issue "
                        f"{issue.reference}, currently {issue.status or 'of unstated status'}. "
                        + (f"Seen before on {', '.join(earlier)}. " if earlier else "")
                        + (
                            "Matched by explicit reference in the ticket."
                            if match.by_reference
                            else f"Matched on: {', '.join(match.terms)}."
                        )
                    ),
                    severity=verdict.severity,
                    confidence=min(1.0, 0.6 + 0.1 * match.score),
                    evidence=(ticket.ticket_id, issue.clause_id, *earlier),
                    suggested_action=_workaround(issue.text),
                )
            )
        return SignalReport(Signal.KNOWN_ISSUE_RECURRENCE, checked=True, findings=tuple(found))

    def _target_risk(self, graded: Sequence[tuple[Ticket, SeverityVerdict]]) -> SignalReport:
        if self._sla is None:
            return SignalReport(
                Signal.FIRST_RESPONSE_RISK,
                checked=False,
                note="no first-response calculator was supplied",
            )
        found = []
        for ticket, verdict in graded:
            outcome = self._sla(ticket, verdict)
            if outcome is None or outcome.get("past_target_by_minutes") is None:
                continue
            past = int(outcome["past_target_by_minutes"])
            if past <= 0:
                continue
            found.append(
                Finding(
                    finding_id=f"frr_{ticket.ticket_id.lower()}",
                    signal=Signal.FIRST_RESPONSE_RISK,
                    subject_id=ticket.ticket_id,
                    subject_kind="ticket",
                    account_id=ticket.account_id,
                    headline=(
                        f"{ticket.ticket_id} is {past} minutes past its "
                        f"{outcome.get('target')} target"
                    ),
                    detail=(
                        f"Created {ticket.created_at.isoformat()}; governing target "
                        f"{outcome.get('target')} from {outcome.get('target_clause')}. "
                        "Tickets carry no first-response timestamp, so this is elapsed "
                        "time against the target, not a measured breach."
                    ),
                    severity=verdict.severity,
                    past_target_minutes=past,
                    measurable=False,
                    evidence=(ticket.ticket_id, str(outcome.get("target_clause") or "")),
                )
            )
        return SignalReport(Signal.FIRST_RESPONSE_RISK, checked=True, findings=tuple(found))

    def _concentration(self, graded: Sequence[tuple[Ticket, SeverityVerdict]]) -> SignalReport:
        by_account: dict[str, list[Ticket]] = {}
        for ticket, verdict in graded:
            if verdict.severity in {"P1", "P2"}:
                by_account.setdefault(ticket.account_id, []).append(ticket)
        found = [
            Finding(
                finding_id=f"sev_{account_id.lower()}",
                signal=Signal.SEVERITY_CONCENTRATION,
                subject_id=account_id,
                subject_kind="account",
                account_id=account_id,
                headline=f"{account_id} has {len(tickets)} open P1/P2 tickets",
                detail=", ".join(f"{t.ticket_id} ({t.subject})" for t in tickets),
                severity="P1",
                confidence=0.8,
                evidence=tuple(t.ticket_id for t in tickets),
            )
            for account_id, tickets in sorted(by_account.items())
            if len(tickets) > 1
        ]
        return SignalReport(Signal.SEVERITY_CONCENTRATION, checked=True, findings=tuple(found))

    def _cross_account(
        self,
        graded: Sequence[tuple[Ticket, SeverityVerdict]],
        matched: Mapping[str, Match],
    ) -> SignalReport:
        spread: dict[str, set[str]] = {}
        for ticket, _ in graded:
            issue = (matched.get(ticket.ticket_id) or Match(issue=None)).issue
            if issue is not None:
                spread.setdefault(issue.reference, set()).add(ticket.account_id)
        found = tuple(
            Finding(
                finding_id=f"xac_{reference.lower()}",
                signal=Signal.CROSS_ACCOUNT_IMPACT,
                subject_id=reference,
                subject_kind="known_issue",
                account_id=None,
                headline=f"{reference} affects {len(accounts)} accounts — possibly systemic",
                detail=f"Seen on {', '.join(sorted(accounts))}.",
                severity="P1",
                evidence=tuple(sorted(accounts)),
            )
            for reference, accounts in sorted(spread.items())
            if len(accounts) > 1
        )
        return SignalReport(
            Signal.CROSS_ACCOUNT_IMPACT,
            checked=True,
            findings=found,
            note=(
                "" if found else "checked; no known issue currently affects more than one account"
            ),
        )

    def _unmatched_high_severity(
        self,
        graded: Sequence[tuple[Ticket, SeverityVerdict]],
        matched: Mapping[str, Match],
    ) -> SignalReport:
        found = []
        for ticket, verdict in graded:
            if verdict.severity != "P1":
                continue
            if (matched.get(ticket.ticket_id) or Match(issue=None)).issue is not None:
                continue
            found.append(
                Finding(
                    finding_id=f"new_{ticket.ticket_id.lower()}",
                    signal=Signal.UNMATCHED_HIGH_SEVERITY,
                    subject_id=ticket.ticket_id,
                    subject_kind="ticket",
                    account_id=ticket.account_id,
                    headline=f"{ticket.ticket_id} is P1 with no matching known issue",
                    detail=(
                        f"{ticket.subject}. Graded P1 by {verdict.basis or 'inference'} and "
                        "nothing in the Known Issues document describes it, so this may be "
                        "a new incident."
                    ),
                    severity="P1",
                    evidence=(ticket.ticket_id, verdict.basis_clause or ""),
                )
            )
        return SignalReport(Signal.UNMATCHED_HIGH_SEVERITY, checked=True, findings=tuple(found))

    def _historical_contradiction(
        self, tickets: Sequence[Ticket], closed: Sequence[Ticket] = ()
    ) -> SignalReport:
        found = [
            Finding(
                finding_id=f"hist_{ticket.ticket_id.lower()}",
                signal=Signal.HISTORICAL_CONTRADICTION,
                subject_id=ticket.ticket_id,
                subject_kind="ticket",
                account_id=ticket.account_id,
                headline=f"{ticket.ticket_id} carries a historical resolution",
                detail=(
                    "Past resolutions are Tier 5 and context only. This one must not be "
                    f"quoted as current guidance: {ticket.historical_resolution}"
                ),
                severity=None,
                confidence=0.9,
                evidence=(ticket.ticket_id,),
            )
            for ticket in tickets
            if ticket.historical_resolution
        ]
        found.extend(
            Finding(
                finding_id=f"hist_{ticket.ticket_id.lower()}",
                signal=Signal.HISTORICAL_CONTRADICTION,
                subject_id=ticket.ticket_id,
                subject_kind="ticket",
                account_id=ticket.account_id,
                headline=(
                    f"{ticket.ticket_id} is closed and its resolution is on file — context only"
                ),
                detail=(
                    "Tier 5. Both historical resolutions in this pack are wrong, so this "
                    "must not be quoted as current guidance: "
                    f"{ticket.historical_resolution}"
                ),
                severity=None,
                confidence=0.9,
                evidence=(ticket.ticket_id,),
            )
            for ticket in closed
            if ticket.historical_resolution
        )
        return SignalReport(Signal.HISTORICAL_CONTRADICTION, checked=True, findings=tuple(found))

    def _volume(self, tickets: Sequence[Ticket]) -> SignalReport:
        # Suppressed rather than omitted. A dashboard that silently drops a
        # signal below some threshold is one whose gaps nobody can see.
        return SignalReport(
            Signal.VOLUME_SPIKE,
            checked=False,
            note=(
                f"suppressed: {len(tickets)} open tickets is below the {VOLUME_SIGNAL_MINIMUM} "
                "needed for a rate to mean anything"
            ),
        )


_ORDINALS: Final = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


def _ordinal(n: int) -> str:
    return _ORDINALS.get(n, f"{n}th")


_WORKAROUND: Final = re.compile(r"(workaround[^.]*\.)", re.IGNORECASE)


def _workaround(text: str) -> str | None:
    """The corpus's own instruction, verbatim where it gives one."""
    found = _WORKAROUND.search(text or "")
    return found.group(1).strip() if found else None


__all__ = [
    "MATCH_TERMS",
    "OPEN_STATUSES",
    "VOLUME_SIGNAL_MINIMUM",
    "Finding",
    "HealthScanner",
    "KnownIssue",
    "Match",
    "ScanReport",
    "Signal",
    "SignalReport",
    "build_known_issues",
    "common_terms",
    "match_issue",
]
