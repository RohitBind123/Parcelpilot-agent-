"""Deterministic calculators.

Every number, date and eligibility verdict in an answer is produced here, in
Python, from `params` on a resolved clause - never by the model, and never from
clause prose. Each returns the clause it computed from alongside the number, so
citation and computation cannot drift apart.
"""
