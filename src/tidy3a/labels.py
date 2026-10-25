"""The exact label rule, now in the library: ``tidy3a.labelarray.labels`` (plan milestone 10, step A4).

This name stays so code importing ``tidy3a.labels`` keeps working.
"""

from tidy3a.labelarray.labels import (
    KINDS,
    compare,
    equal,
    exact_cast,
    kind,
    lossy_join_type,
    missing,
    strict_subset,
)

__all__ = ["KINDS", "compare", "equal", "exact_cast", "kind", "lossy_join_type", "missing", "strict_subset"]
