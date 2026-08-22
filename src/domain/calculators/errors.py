"""Why a calculator refused.

Refusing is a first-class outcome, not an exceptional one. A calculator with no
governing clause, or with two clauses of equal authority disagreeing, must not
produce a number - the whole point of computing in Python is that the number is
defensible, and a guess made under uncertainty is exactly as wrong as a
hallucination while looking considerably more official.

These reach the model as structured tool errors, so the wording is part of the
interface: it has to say what is missing clearly enough that the next action is
obvious, without leaking anything the Principal may not see.
"""

from __future__ import annotations


class CalculationError(RuntimeError):
    """A calculation could not be performed."""


class NoBasis(CalculationError):
    """Nothing governs this topic, or an unresolved conflict stands.

    The correct response is to escalate (D27), not to fall back to a default.
    """


class WrongEvidence(CalculationError):
    """The handles are valid but do not belong together.

    A resolution for the wrong topic, or one describing a different account
    from the snapshot. Both are silent-wrong-answer bugs otherwise: the
    arithmetic succeeds and applies one customer's contract to another's
    shipment.
    """
