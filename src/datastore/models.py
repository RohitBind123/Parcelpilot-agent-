"""Typed rows.

Frozen dataclasses rather than `sqlite3.Row`, for three reasons: timestamps
arrive already timezone-aware so no caller has to remember to parse them,
booleans are booleans rather than 0/1, and nothing downstream can mutate a row
it merely read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime


def _dt(value: str | None) -> datetime | None:
    """Parse a stored timestamp, preserving absence.

    None means the fact is not recorded, which is different from a zero time.
    Every stored timestamp carries an offset, so the result is always aware.
    """
    return datetime.fromisoformat(value) if value else None


def _bool(value: int | None) -> bool:
    return bool(value)


def _payload(value: object) -> object:
    """One row field, in a form json.dumps accepts.

    None stays None. An absent pickup time and a pickup at the epoch are not
    the same fact, and every downstream consumer - the calculators, the fact
    block, the trace - depends on being able to tell them apart.
    """
    return value.isoformat() if isinstance(value, datetime) else value


class _Row:
    """Shared serialisation for the typed rows.

    A snapshot payload is what a calculator computes from and what the trace
    renders, so it has to be plain JSON - and it has to preserve None. An
    absent pickup time and a pickup at the epoch are different facts.
    """

    def to_payload(self) -> dict[str, object]:
        return {key: _payload(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class Account(_Row):
    account_id: str
    account_name: str
    plan: str
    status: str
    csm: str | None
    contract_file: str | None
    premium_support: bool
    notes: str | None

    @property
    def has_agreement(self) -> bool:
        """Whether a signed agreement exists for this account in the pack.

        Governing fact, not cosmetic: without one there is no Tier 1 source,
        so general policy is the governing authority with nothing to override.
        """
        return self.contract_file is not None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Account:
        return cls(
            account_id=row["account_id"],
            account_name=row["account_name"],
            plan=row["plan"],
            status=row["status"],
            csm=row["csm"],
            contract_file=row["contract_file"],
            premium_support=_bool(row["premium_support"]),
            notes=row["notes"],
        )


@dataclass(frozen=True, slots=True)
class Order(_Row):
    order_id: str
    account_id: str
    carrier: str
    status: str
    booked_at: datetime
    pickup_window_start: datetime | None
    pickup_window_end: datetime | None
    #: None means ParcelPilot holds no pickup confirmation. Given KI-211, that
    #: is not the same as "the parcel was not collected".
    pickup_actual_at: datetime | None
    shipment_fee_inr: float | None
    carrier_fault: bool
    customer_fault: bool
    cancellation_requested_at: datetime | None
    notes: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Order:
        return cls(
            order_id=row["order_id"],
            account_id=row["account_id"],
            carrier=row["carrier"],
            status=row["status"],
            booked_at=_dt(row["booked_at"]),
            pickup_window_start=_dt(row["pickup_window_start"]),
            pickup_window_end=_dt(row["pickup_window_end"]),
            pickup_actual_at=_dt(row["pickup_actual_at"]),
            shipment_fee_inr=row["shipment_fee_inr"],
            carrier_fault=_bool(row["carrier_fault"]),
            customer_fault=_bool(row["customer_fault"]),
            cancellation_requested_at=_dt(row["cancellation_requested_at"]),
            notes=row["notes"],
        )


@dataclass(frozen=True, slots=True)
class Ticket(_Row):
    ticket_id: str
    account_id: str
    created_at: datetime
    status: str
    subject: str
    description: str | None
    channel: str | None
    assigned_to: str | None
    last_customer_message_at: datetime | None
    #: Tier 5. Context only, and both instances in the pack are wrong.
    historical_resolution: str | None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Ticket:
        return cls(
            ticket_id=row["ticket_id"],
            account_id=row["account_id"],
            created_at=_dt(row["created_at"]),
            status=row["status"],
            subject=row["subject"],
            description=row["description"],
            channel=row["channel"],
            assigned_to=row["assigned_to"],
            last_customer_message_at=_dt(row["last_customer_message_at"]),
            historical_resolution=row["historical_resolution"],
        )
