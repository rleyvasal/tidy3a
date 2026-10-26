"""The exact label rule, now in the library: ``labelarray.labels`` (plan milestone 10, steps A4 and A7).

This name stays so code importing ``tidy3a.labels`` keeps working.
"""

from labelarray.labels import (
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
