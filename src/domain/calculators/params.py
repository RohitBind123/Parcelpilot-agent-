"""Reading a parameter off a resolution, with its source recorded.

A calculator needs values that are spread across the clauses a resolution
returned, not just the governing one. Northstar section 2 governs the
cancellation fee and says nothing about what happens to a PICKED_UP shipment;
the SOP it overrode holds the status rules. Both are needed, and the answer
must be able to cite whichever clause each number actually came from.

So lookups walk an explicit priority order and return the source alongside the
value. Never a bare `params.get(...)` on the governing clause: that either
misses the value or, worse, silently falls through to a default nobody can
point at in a document.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.domain.resolver import ClauseRef, PolicyResolution

#: Governing first, then the rules it displaced, then a stronger clause that
#: declined to override, then on-topic prose. Deliberate: an overridden clause
#: still describes reality for anything the override did not speak to.
_PRIORITY = ("governing", "overridden", "deferred", "supporting")


@dataclass(frozen=True, slots=True)
class Sourced:
    """A value and the clause it came from."""

    value: Any
    source: str | None
    found: bool = True

    @property
    def missing(self) -> bool:
        return not self.found


MISSING = Sourced(value=None, source=None, found=False)


def clauses_in_priority(resolution: PolicyResolution) -> Iterator[ClauseRef]:
    for bucket in _PRIORITY:
        found = getattr(resolution, bucket)
        for ref in (found,) if isinstance(found, ClauseRef) else found:
            if ref is not None:
                yield ref


def lookup(resolution: PolicyResolution, key: str) -> Sourced:
    """First clause in priority order that carries `key`.

    Presence is what counts, not truthiness. Northstar's `window_minutes` is
    present and null - having no time window is the substance of that clause -
    and a lookup that treated null as absent would fall through to the SOP's
    30-minute window and reintroduce the rule the agreement removed.
    """
    for ref in clauses_in_priority(resolution):
        if key in ref.params:
            return Sourced(value=ref.params[key], source=ref.clause_id)
    return MISSING
