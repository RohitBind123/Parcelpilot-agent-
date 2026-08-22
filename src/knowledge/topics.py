"""The curated topic taxonomy.

This is the hinge of the design (ARCHITECTURE §5.3). Precedence is only
decidable between clauses that are *about the same thing*, so every clause is
tagged from a small closed enum and the resolver groups by tag. Open-ended
tagging would put "cancellation fee" and "termination charge" in different
buckets and the override would never be detected.

The enum is closed on purpose. A tag that does not appear in this file cannot
be produced, so a typo becomes an error rather than a silently unresolvable
group. The trade-off is that extending the corpus means extending this file -
which is the right amount of friction for something the resolver depends on.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final


class Topic(StrEnum):
    """Subjects a clause can be about."""

    # Cancellation
    CANCELLATION_FEE = "cancellation_fee"
    CANCELLATION_WINDOW = "cancellation_window"
    CANCELLATION_STATUS_RULES = "cancellation_status_rules"
    RETURN_TO_ORIGIN = "return_to_origin"

    # Service credits
    FAILED_PICKUP_CREDIT = "failed_pickup_credit"
    CREDIT_CAP = "credit_cap"
    CREDIT_APPROVAL = "credit_approval"

    # Support and SLA
    SEVERITY_DEFINITION = "severity_definition"
    FIRST_RESPONSE_TARGET = "first_response_target"
    WEEKEND_COVERAGE = "weekend_coverage"
    ESCALATION_DUTY = "escalation_duty"

    # Governance
    SOURCE_PRECEDENCE = "source_precedence"
    DATA_CONFLICT = "data_conflict"

    # Product
    PLAN_CAPABILITY = "plan_capability"
    BULK_UPLOAD_LIMIT = "bulk_upload_limit"
    SHIPMENT_STATUS_SEMANTICS = "shipment_status_semantics"
    KNOWN_ISSUE = "known_issue"
    RESOLVED_ISSUE = "resolved_issue"

    # Account
    ACCOUNT_CONTACT = "account_contact"


def _pattern(*fragments: str) -> re.Pattern[str]:
    return re.compile("|".join(fragments), re.IGNORECASE)


#: Ordered so the most specific subject is tested first. A clause may carry
#: several tags; the resolver only needs one of them to match the query.
TOPIC_PATTERNS: Final[tuple[tuple[Topic, re.Pattern[str]], ...]] = (
    (
        Topic.CANCELLATION_FEE,
        _pattern(r"cancellation fee", r"cancellation-fee", r"no fee", r"charge INR"),
    ),
    (
        Topic.CANCELLATION_WINDOW,
        _pattern(
            r"within \d+ minutes of booking",
            r"after \d+ minutes",
            r"before pickup",
            r"regardless of how long ago",
        ),
    ),
    (
        Topic.CANCELLATION_STATUS_RULES,
        _pattern(r"\bDRAFT\b", r"may be cancelled", r"cannot be cancelled", r"do not cancel"),
    ),
    (Topic.RETURN_TO_ORIGIN, _pattern(r"return-to-origin")),
    # `service credit` must not match the SOP's own name: LumenWorks §2 says
    # "use the current ParcelPilot Cancellation & Service Credit SOP" while
    # explicitly declining to override credits, and tagging it here would put
    # it up against its own §3 as a competing tier-1 authority.
    (
        Topic.FAILED_PICKUP_CREDIT,
        _pattern(
            r"service credits?(?!\s+SOP)", r"failed-pickup", r"past the end\b.{0,40}pickup window"
        ),
    ),
    (Topic.CREDIT_CAP, _pattern(r"capped at", r"aggregate service credits", r"\bcap\b")),
    (Topic.CREDIT_APPROVAL, _pattern(r"manager approval", r"requires approval")),
    (
        Topic.SEVERITY_DEFINITION,
        _pattern(
            r"P1 - Critical", r"P2 - High", r"P3 - Normal", r"severity definitions", r"P1 covers"
        ),
    ),
    (
        Topic.FIRST_RESPONSE_TARGET,
        _pattern(r"first-response", r"response targets?", r"P1:\s*\d", r"business hours", r"24x7"),
    ),
    (Topic.WEEKEND_COVERAGE, _pattern(r"weekend", r"after-hours")),
    (
        Topic.ESCALATION_DUTY,
        _pattern(r"escalated immediately", r"recommend escalation", r"escalation"),
    ),
    (
        Topic.SOURCE_PRECEDENCE,
        _pattern(r"sources conflict", r"may override", r"source precedence", r"context only"),
    ),
    (Topic.DATA_CONFLICT, _pattern(r"data conflicts", r"request verification", r"is unknown")),
    (Topic.BULK_UPLOAD_LIMIT, _pattern(r"bulk upload", r"rows per CSV", r"CSV uploads")),
    (
        Topic.PLAN_CAPABILITY,
        _pattern(r"plan capabilities", r"available on \w+ and \w+", r"is not included"),
    ),
    (
        Topic.SHIPMENT_STATUS_SEMANTICS,
        _pattern(r"BOOKED means", r"PICKED_UP means", r"pickup confirmation"),
    ),
    # Not bare "workaround": Policy v3 §2 defines P2 as "a workaround exists",
    # which is a severity definition rather than an operational issue.
    (Topic.KNOWN_ISSUE, _pattern(r"\bKI-\d+", r"known issues?", r"Workaround:")),
    (Topic.RESOLVED_ISSUE, _pattern(r"resolved issue", r"Resolved \d+ \w+ \d{4}")),
    (Topic.ACCOUNT_CONTACT, _pattern(r"dedicated CSM", r"account contact")),
)

ALL_TOPICS: Final[frozenset[str]] = frozenset(t.value for t in Topic)


def tag(text: str) -> tuple[str, ...]:
    """Every topic this text is about, in taxonomy order.

    Returns a tuple so a clause's tags are hashable and immutable, and ordered
    deterministically so the committed registry is byte-stable.
    """
    return tuple(topic.value for topic, pattern in TOPIC_PATTERNS if pattern.search(text))
