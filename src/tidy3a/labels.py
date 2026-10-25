"""One exact-label rule for every path that compares labels (plan §5.5, docs/backends.md).

Labels on an axis match only when they are the same kind (booleans, numbers,
times, durations, text, bytes) and exactly equal: a label is never converted
in a way that changes it to make a match. The integer ``2**53 + 1`` does not
match the float ``2**53``, a half-second does not match whole seconds, and
``True`` does not match ``1``. A missing label (NaN or NaT) matches a missing
label of the same kind. Alignment and joins use these functions on both
backends, so they agree whatever the label types or their order; the NumPy
backend's lookups convert labels with ``exact_cast`` too.

This module needs NumPy only.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Hashable
from typing import Any

import numpy as np

from tidy3a.errors import TidyError


class LabelError(TidyError):
    """Labels that cannot be ordered, compared, or looked up."""


# Label kinds that can match each other. Booleans, numbers, times, durations,
# text, and bytes never match across kinds, as in pandas: True is not 1.
KINDS = {"b": "bool", "i": "number", "u": "number", "f": "number", "c": "number",
         "M": "time", "m": "duration", "U": "text", "S": "bytes"}


def kind(values: np.ndarray, op: str = "index", dim: Hashable = None) -> tuple[str, np.ndarray]:
    """The label kind, with object arrays of one Python type converted to it."""
    values = np.asarray(values)
    if values.dtype.kind != "O":
        return KINDS.get(values.dtype.kind, "other"), values
    if values.size == 0:
        return "other", values
    types = {type(v) for v in values.tolist()}
    for name, accepted, convert in (
        ("text", (str,), str), ("bytes", (bytes,), bytes), ("bool", (bool, np.bool_), bool),
    ):
        if all(issubclass(t, accepted) for t in types):
            return name, np.array(values.tolist(), dtype=convert)
    if all(issubclass(t, (int, float, complex, np.number)) and not issubclass(t, (bool, np.bool_)) for t in types):
        return "number", np.array(values.tolist())
    names = ", ".join(sorted(t.__name__ for t in types))
    raise LabelError(f"{op}: labels on {dim!r} mix types ({names}), so they cannot be ordered or looked up; "
                     "give the axis labels of one type")


def missing(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "fc":
        return np.isnan(values)
    if values.dtype.kind in "mM":
        return np.isnat(values)
    return np.zeros(values.shape, dtype=bool)


def exact_cast(wanted: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """*wanted* in *dtype*, and where that conversion is exact.

    A value that changes on the way, such as float64 0.1 as float32, a
    half-second as whole seconds, or -1 as an unsigned integer, is not
    exact; nothing is rounded to make a match. Missing values are not exact.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cast = wanted.astype(dtype)
        exact = cast.astype(wanted.dtype) == wanted
    if dtype.kind == "u" and wanted.dtype.kind in "if":
        exact &= wanted >= 0
    elif dtype.kind == "i" and wanted.dtype.kind == "u":
        exact &= wanted <= np.iinfo(dtype).max
    return cast, np.asarray(exact, dtype=bool)


def equal(a: Any, b: Any) -> bool:
    """The same labels in the same order; a missing label equals a missing label."""
    if a is b:
        return True
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        return False
    if not len(a):
        return True
    try:
        kind_a, a = kind(a)
        kind_b, b = kind(b)
    except LabelError:
        return _keys(a) == _keys(b)
    if kind_a != kind_b:
        return False
    if kind_a == "other":
        return _keys(a) == _keys(b)
    gaps_a, gaps_b = missing(a), missing(b)
    if not np.array_equal(gaps_a, gaps_b):
        return False
    if gaps_a.any():
        a, b = a[~gaps_a], b[~gaps_b]
    if a.dtype == b.dtype:
        return bool(np.array_equal(a, b))
    cast, exact = exact_cast(b, a.dtype)
    return bool(exact.all() and np.array_equal(cast, a))


def compare(a: Any, b: Any, op: str = "align", dim: Hashable = None, *,
            mixed: bool = False) -> tuple[str, int, int]:
    """``("same", 0, 0)``, ``("reordered", 0, 0)``, or ``("different", only in a, only in b)``.

    "Reordered" means both hold the same labels as many times each. Counts
    are of distinct labels. Labels of mixed Python types are an error naming
    *op* and *dim*, unless *mixed* is true; then they are compared one by
    one, each with its own kind.
    """
    if equal(a, b):
        return "same", 0, 0
    a, b = np.asarray(a), np.asarray(b)
    try:
        kind_a, values_a = kind(a, op, dim)
        kind_b, values_b = kind(b, op, dim)
    except LabelError:
        if not mixed:
            raise
        kind_a = kind_b = "other"
    if kind_a == "other" or kind_b == "other":
        keys_a, keys_b = _keys(a), _keys(b)
        if Counter(keys_a) == Counter(keys_b):
            return "reordered", 0, 0
        return "different", len(set(keys_a) - set(keys_b)), len(set(keys_b) - set(keys_a))
    if len(a) == len(b) and kind_a == kind_b and _same_multiset(values_a, values_b):
        return "reordered", 0, 0
    found_a, found_b = _found(kind_a, values_a, kind_b, values_b), _found(kind_b, values_b, kind_a, values_a)
    return "different", _distinct(values_a[~found_a]), _distinct(values_b[~found_b])


def strict_subset(part: Any, whole: Any, op: str = "align", dim: Hashable = None, *, mixed: bool = False) -> bool:
    """Whether *part*'s labels are some, but not all, of *whole*'s."""
    state, only_part, only_whole = compare(part, whole, op, dim, mixed=mixed)
    return state == "different" and only_part == 0 and only_whole > 0


def lossy_join_type(a: Any, b: Any) -> np.dtype | None:
    """The type a join would hold both label arrays in, when it cannot hold them all exactly.

    Joined labels take NumPy's common type, so int64 labels joined with
    float64 ones become float64; ``2**53 + 1`` cannot survive that, and a
    join would then pair it with ``2**53``. None when every label survives.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == b.dtype:
        return None
    try:
        common = np.result_type(a.dtype, b.dtype)
    except TypeError:
        return None
    if common.kind == "O":
        return None
    for values in (a, b):
        if values.dtype != common:
            _, exact = exact_cast(values, common)
            if not (exact | missing(values)).all():
                return common
    return None


def _found(kind_a: str, a: np.ndarray, kind_b: str, b: np.ndarray) -> np.ndarray:
    """Which labels of *a* are among *b*'s, exactly."""
    found = np.zeros(len(a), dtype=bool)
    if kind_a != kind_b or not len(a) or not len(b):
        return found
    gaps_a, gaps_b = missing(a), missing(b)
    found[gaps_a] = gaps_b.any()
    rest = ~gaps_a
    present_a, present_b = a[rest], b[~gaps_b]
    if present_a.dtype == present_b.dtype:
        found[rest] = np.isin(present_a, present_b)
    else:
        cast, exact = exact_cast(present_a, present_b.dtype)
        found[rest] = exact & np.isin(cast, present_b)
    return found


def _same_multiset(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether *a* and *b*, of one kind and length, hold the same labels as many times each."""
    gaps_a, gaps_b = missing(a), missing(b)
    if gaps_a.sum() != gaps_b.sum():
        return False
    a, b = a[~gaps_a], b[~gaps_b]
    if a.dtype != b.dtype:
        b, exact = exact_cast(b, a.dtype)
        if not exact.all():
            return False
    return bool(np.array_equal(np.sort(a), np.sort(b)))


def _distinct(values: np.ndarray) -> int:
    if not len(values):
        return 0
    gaps = missing(values)
    return len(np.unique(values[~gaps])) + int(gaps.any())


def _keys(values: np.ndarray) -> list:
    """Labels as hashable Python values tagged with their kind, a missing label as one key.

    Python compares its numbers exactly (``2**53 + 1 != float(2**53)``);
    the tag keeps ``True`` apart from ``1``.
    """
    return [("missing",) if v != v else (_tag(v), v) for v in np.asarray(values).tolist()]


def _tag(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, float, complex, np.number)):
        return "number"
    return type(value).__name__
