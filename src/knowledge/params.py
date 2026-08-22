"""Typed parameters extracted from clause prose.

This module is the bridge from prose to arithmetic. Calculators read `params`
and never the clause text, so a wrong value here produces a wrong answer
carrying a correct-looking citation - the worst failure the system can make,
because a reader who checks the citation sees nothing amiss.

Two decisions follow from that.

**Regex, not a model.** The skill `regex-vs-llm-structured-text` says start
with regex and escalate only the low-confidence tail to an LLM. The escalation
half is deliberately not adopted: the registry is committed (D10), and a model
call at build time makes it non-reproducible. The hand-reviewed baseline in
`clause_params_baseline.yaml` is the validator instead, and it covers every
clause rather than a tail. `score_confidence` is still here, doing the job the
skill gives it - telling a reviewer where to look.

**Absence is explicit.** Northstar's cancellation clause has no time window at
all, and that is the whole point of it. So `window_minutes` is present and
`None` rather than missing, because a missing key is something a caller can
read as zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Literal

from src.knowledge.clause_parser import Clause
from src.knowledge.topics import Topic

Unit = Literal["minutes", "hours", "days"]

_INT: Final = r"(\d[\d,]*)"

_DURATION: Final = re.compile(
    r"\b(\d+)\s+(business\s+)?(minute|hour|day)s?\b(\s*,?\s*24x7)?", re.IGNORECASE
)

_PLAN_ROW: Final = re.compile(
    r"^(Enterprise|Growth|Standard)\s+(.*)$", re.IGNORECASE | re.MULTILINE
)
_SEVERITY_LINE: Final = re.compile(r"P([123])\s*:\s*([^\n]*?)(?=\s*(?:$|\n))", re.MULTILINE)

#: A number in a value-bearing context: an amount of money, a duration, a
#: percentage, a row count. Used to decide whether a clause *should* have
#: produced parameters.
#:
#: Asked of the text rather than inferred from topics, because the two answer
#: different questions. Policy v3 §1 says the policy "defines default support
#: severity and response targets" and §4 says "if a response target is already
#: breached" - both are correctly tagged first_response_target for retrieval,
#: and neither states a target. Deriving "expects numbers" from the tag flags
#: both, and a flag a reviewer learns to ignore is worse than no flag.
_STATES_A_VALUE: Final = re.compile(
    r"INR\s*\d|\d[\d,]*\s*%|\d[\d,]*\s+(?:business\s+)?(?:minute|hour|day|row)s?\b",
    re.IGNORECASE,
)

SEVERITIES: Final = ("P1", "P2", "P3")
PLANS: Final = ("Enterprise", "Growth", "Standard")


@dataclass(frozen=True, slots=True)
class Duration:
    """A response target, keeping the distinction the corpus makes.

    "2 hours" and "2 business hours" are different answers by roughly two days
    when the snapshot falls on a Sunday, so `business` is a field rather than
    something a caller infers from context. `always_on` records the 24x7
    qualifier, which overrides the business-hours calendar entirely.
    """

    amount: int
    unit: Unit
    business: bool = False
    always_on: bool = False

    @classmethod
    def parse(cls, text: str) -> Duration | None:
        """Parse one duration, or return None rather than guess."""
        match = _DURATION.search(text)
        if not match:
            return None
        amount, business, unit, around_clock = match.groups()
        return cls(
            amount=int(amount),
            unit=f"{unit.lower()}s",  # type: ignore[arg-type]
            business=bool(business),
            always_on=bool(around_clock),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "unit": self.unit,
            "business": self.business,
            "always_on": self.always_on,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Duration:
        return cls(
            amount=raw["amount"],
            unit=raw["unit"],
            business=raw.get("business", False),
            always_on=raw.get("always_on", False),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceFlag:
    """Where a reviewer should look, and why."""

    clause_id: str
    score: float
    reasons: tuple[str, ...] = ()


def _int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1).replace(",", "")) if match else None


def extract_params(clause: Clause) -> dict[str, Any]:
    """Every typed parameter this clause carries.

    Extractors are keyed by topic and merged, because a clause can be about
    more than one subject: the SOP's cancellation clause carries both the fee
    schedule and the per-status rules.
    """
    params: dict[str, Any] = {}
    for topic, extractor in _EXTRACTORS:
        if topic in clause.topics:
            params |= extractor(clause)
    return params


# -- per-topic extractors ---------------------------------------------------


def _cancellation(clause: Clause) -> dict[str, Any]:
    text = clause.text
    params: dict[str, Any] = {}

    if window := _int(text, rf"within\s+{_INT}\s+minutes"):
        params["free_window_minutes"] = window
    if fee := _int(text, rf"charge\s+INR\s+{_INT}"):
        params["fee_after_window_inr"] = fee
    if re.search(r"waives the cancellation fee|explicitly waives", text, re.IGNORECASE):
        params["waivable_by_agreement"] = True

    if re.search(r"with no cancellation fee|no cancellation fee", text, re.IGNORECASE):
        params["waiver"] = True
        params["fee_inr"] = 0
    if re.search(r"no special cancellation-fee waiver", text, re.IGNORECASE):
        params["waiver"] = False

    # "regardless of how long ago the shipment was booked": the absence of a
    # window is the substance of the clause, so it is recorded explicitly.
    if re.search(r"regardless of how long ago", text, re.IGNORECASE):
        params["window_minutes"] = None
    if re.search(r"any BOOKED shipment before pickup", text, re.IGNORECASE):
        params["applies_to_status"] = ["BOOKED"]

    params["overrides"] = _override_stance(clause)
    return params


def _cancellation_status_rules(clause: Clause) -> dict[str, Any]:
    text = clause.text
    rules: dict[str, str] = {}
    if re.search(r"DRAFT:.*?no fee", text, re.IGNORECASE | re.DOTALL):
        rules["DRAFT"] = "free"
    if re.search(r"BOOKED, not yet PICKED_UP", text, re.IGNORECASE):
        rules["BOOKED"] = "conditional"
    if re.search(r"PICKED_UP:\s*Do not cancel", text, re.IGNORECASE):
        rules["PICKED_UP"] = "return_to_origin"
    if re.search(r"DELIVERED:\s*Cannot be cancelled", text, re.IGNORECASE):
        rules["DELIVERED"] = "not_cancellable"
    return {"status_rules": rules} if rules else {}


def _credit(clause: Clause) -> dict[str, Any]:
    text = clause.text
    params: dict[str, Any] = {}

    if threshold := _int(text, rf"more than\s+{_INT}\s+hours past"):
        params["threshold_hours"] = threshold
    if flat := _int(text, rf"lower of\s+INR\s+{_INT}"):
        params["credit_flat_inr"] = flat
        params["credit_rule"] = "lower_of"
    if percent := _int(text, rf"{_INT}%\s+of the shipment fee"):
        params["credit_percent"] = percent
    if fixed := _int(text, rf"fixed\s+INR\s+{_INT}"):
        params["credit_inr"] = fixed

    if re.search(r"carrier is at fault", text, re.IGNORECASE):
        params["requires_carrier_fault"] = True
    if re.search(r"no customer-caused issue|customer\s+is not at fault", text, re.IGNORECASE):
        params["requires_no_customer_fault"] = True

    params["overrides"] = _override_stance(clause)
    return params


def _credit_cap(clause: Clause) -> dict[str, Any]:
    if cap := _int(clause.text, rf"capped at\s+INR\s+{_INT}"):
        return {"monthly_cap_inr": cap}
    return {}


def _credit_approval(clause: Clause) -> dict[str, Any]:
    if threshold := _int(clause.text, rf"above\s+INR\s+{_INT}\s+requires manager approval"):
        return {"manager_approval_above_inr": threshold}
    return {}


def _response_targets(clause: Clause) -> dict[str, Any]:
    """Either a plan-by-severity grid, or a flat per-severity list.

    The policies publish a table; the agreements publish three bullets. Both
    end up here because the resolver compares them against each other.
    """
    params: dict[str, Any] = {}

    grid: dict[str, dict[str, Any]] = {}
    for plan, cells in _PLAN_ROW.findall(clause.text):
        parsed = _row_durations(cells)
        if len(parsed) == len(SEVERITIES):
            grid[plan.capitalize()] = dict(zip(SEVERITIES, parsed, strict=True))
    if grid:
        params["targets"] = grid
        return params

    flat = {
        f"P{severity}": duration.to_dict()
        for severity, raw in _SEVERITY_LINE.findall(clause.text)
        if (duration := Duration.parse(raw))
    }
    if flat:
        params["targets"] = flat
    return params


def _row_durations(cells: str) -> list[dict[str, Any]]:
    """Split one table row into its severity cells.

    The row arrives as "30 minutes, 24x7 2 hours 1 business day" with no
    delimiters, so the durations themselves are the delimiter.
    """
    return [
        Duration(
            amount=int(amount),
            unit=f"{unit.lower()}s",  # type: ignore[arg-type]
            business=bool(business),
            always_on=bool(around_clock),
        ).to_dict()
        for amount, business, unit, around_clock in _DURATION.findall(cells)
    ]


def _weekend_coverage(clause: Clause) -> dict[str, Any]:
    if re.search(r"[Nn]o weekend or after-hours", clause.text):
        return {"weekend_coverage": False}
    return {}


def _bulk_upload(clause: Clause) -> dict[str, Any]:
    text = clause.text
    params: dict[str, Any] = {}
    if supported := _int(text, rf"up to\s+{_INT}\s+rows"):
        params["supported_rows"] = supported
        params["plans_included"] = [
            plan for plan in ("Growth", "Enterprise") if re.search(plan, text)
        ]
    # The failure threshold is a bug, not a plan limit. Conflating the two is
    # precisely the wrong answer recorded on TKT-451.
    if failure := _int(text, rf"above approximately\s+{_INT}\s+rows"):
        params["failure_threshold_rows"] = failure
    return params


def _known_issue(clause: Clause) -> dict[str, Any]:
    text = clause.text
    params: dict[str, Any] = {}
    if status := re.search(r"Status:\s*(\w+)", text):
        params["issue_status"] = status.group(1)
    elif re.search(r"\bResolved\b\s+\d", text):
        # KI-176 states its resolution as prose ("Resolved 18 July 2026")
        # instead of a Status line. Recording it means a caller asking "is this
        # issue current?" reads an answer rather than an empty dict, which is
        # also what a broken pattern produces.
        params["issue_status"] = "Resolved"
    if delay := _int(text, rf"up to\s+{_INT}\s+minutes late"):
        params["delay_minutes"] = delay
    if carrier := re.search(r"\b(SwiftShip|BlueDart Pro|RoadRunner)\b", text):
        params["carrier"] = carrier.group(1)
    return params


_EXTRACTORS: Final[tuple[tuple[str, Any], ...]] = (
    (Topic.CANCELLATION_FEE.value, _cancellation),
    (Topic.CANCELLATION_STATUS_RULES.value, _cancellation_status_rules),
    (Topic.FAILED_PICKUP_CREDIT.value, _credit),
    (Topic.CREDIT_CAP.value, _credit_cap),
    (Topic.CREDIT_APPROVAL.value, _credit_approval),
    (Topic.FIRST_RESPONSE_TARGET.value, _response_targets),
    (Topic.WEEKEND_COVERAGE.value, _weekend_coverage),
    (Topic.BULK_UPLOAD_LIMIT.value, _bulk_upload),
    (Topic.KNOWN_ISSUE.value, _known_issue),
)


def _override_stance(clause: Clause) -> bool | None:
    """Whether this clause overrides the default, declines to, or *is* it.

    Three states, not two. `None` means "I am the baseline"; `False` means "an
    agreement exists and deliberately leaves the baseline in place". The
    resolver needs to tell those apart, because a Tier 1 clause existing is not
    the same as a Tier 1 clause winning - and the pack contains both cases
    specifically to catch a resolver that assumes otherwise.
    """
    if clause.tier != 1:
        return None
    if re.search(
        r"no special .*? waiver applies|Unless this agreement states otherwise",
        clause.text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"replaces?\b|with no cancellation fee|receives a fixed", clause.text, re.IGNORECASE
    ):
        return True
    return None


# -- confidence -------------------------------------------------------------


def score_confidence(clause_id: str, params: dict[str, Any]) -> ConfidenceFlag:
    """Where a reviewer should look before signing off the baseline.

    Not a correctness measure - a confident extraction can still be wrong,
    which is what the baseline is for. This only surfaces the clauses whose
    shape suggests something was missed.
    """
    reasons: list[str] = []
    score = 1.0

    expects_numbers = _expects_numbers(clause_id)
    if expects_numbers and not params:
        reasons.append("no_params_extracted")
        score -= 0.6

    grid = params.get("targets")
    if isinstance(grid, dict) and grid:
        if _looks_like_plan_grid(grid):
            missing = [
                f"{plan}.{severity}"
                for plan in PLANS
                for severity in SEVERITIES
                if not grid.get(plan, {}).get(severity)
            ]
            if missing:
                reasons.append("incomplete_grid")
                score -= 0.4
        elif any(severity not in grid for severity in SEVERITIES):
            reasons.append("incomplete_grid")
            score -= 0.4

    if params.get("overrides") is True and len(params) <= 1:
        reasons.append("override_without_values")
        score -= 0.3

    return ConfidenceFlag(clause_id, max(0.0, round(score, 2)), tuple(reasons))


def _looks_like_plan_grid(grid: dict[str, Any]) -> bool:
    return any(plan in grid for plan in PLANS)


@lru_cache(maxsize=1)
def _clause_text_index() -> dict[str, str]:
    from src.knowledge.clause_parser import parse_all

    return {c.clause_id: c.text for doc in parse_all() for c in doc.clauses}


def _expects_numbers(clause_id: str) -> bool:
    """Whether this clause states a value that extraction should have caught."""
    text = _clause_text_index().get(clause_id, "")
    return bool(_STATES_A_VALUE.search(text))
