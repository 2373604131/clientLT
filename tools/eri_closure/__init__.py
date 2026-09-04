"""Auditable Evidence--Rewrite Imbalance (ERI) closure experiments.

The package deliberately separates (1) frozen protocol construction, (2)
on-trajectory state dumps, (3) functional attribution, and (4) aggregation
replay.  Nothing in this package changes a training update.
"""

from .protocol import DEFAULT_AUDIT_ROUNDS, DEFAULT_TAIL_CLASSES, parse_eri_rounds

__all__ = ["DEFAULT_AUDIT_ROUNDS", "DEFAULT_TAIL_CLASSES", "parse_eri_rounds"]
