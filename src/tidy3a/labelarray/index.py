"""Label indexes: where labels are on one axis (plan milestone 10, step A4).

An index answers two questions about one axis's labels: the position of
each wanted label (``get_indexer``), and the positions holding a range of
labels (``slice_indexer``). ``index_for`` inspects a label array once and
picks the implementation:

- ``GridIndex`` for exactly evenly spaced numbers, times, or durations, such
  as gridded latitude, longitude, and time, or positions 1, 2, 3: answered
  by arithmetic, with no sort and no arrays held;
- ``SortedIndex`` for other strictly increasing labels, searched directly
  with ``np.searchsorted``;
- ``SortOnceIndex`` for the rest, searched through one sort made when the
  index is built.

There is no hash index: a Python dict of a million labels is slow to build
and cannot be looked up vectorised, so fast hashing needs pandas or a
compiled kernel, and the sort-once index serves unsorted labels until then.

Every index keeps the exact label rule (``labelarray.labels``): a wanted
label matches only a label of the same kind that is exactly equal, so a
lookup is converted to the labels' type with ``exact_cast`` and never
rounded. The integer ``2**53 + 1`` does not match the float ``2**53``,
float64 0.1 does not match float32 0.1, a half-second does not match whole
seconds, and ``True`` does not match 1. Ranges compare exactly too: a range
bound is moved to the nearest label value inside the range, never rounded
to the nearest.

This module needs NumPy only.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tidy3a.labelarray.errors import LabelError
from tidy3a.labelarray.labels import KINDS, exact_cast, kind, missing, register_types


@runtime_checkable
class LabelIndex(Protocol):
    """What every label index answers, under the exact label rule.

    ``kind`` is the labels' kind (``labelarray.labels.KINDS``, or a kind
    registered with ``register_index``); ``size`` the number of labels;
    ``unique`` whether each label names one position (one missing label
    counts as a label; two repeat); ``sorted`` whether the labels other than
    missing ones strictly increase along the axis.
    """

    kind: str
    size: int
    unique: bool
    sorted: bool

    def get_indexer(self, wanted: Any) -> np.ndarray:
        """The position of each wanted label, or -1 where it is absent; an error if labels repeat."""
        ...

    def slice_indexer(self, start: Any = None, stop: Any = None) -> slice | np.ndarray:
        """The positions of labels from *start* to *stop*, both included, in axis order.

        None leaves that end open. A ``slice`` where the positions are one
        run, otherwise increasing positions. A missing label is never in a
        range; a bound of another kind, or missing, is an error.
        """
        ...


# ── Searching ───────────────────────────────────────────────────────────────


class _SearchIndex:
    """Labels searched with ``np.searchsorted``, directly or through one sort.

    A missing label (NaN or NaT) is held apart, since it never sorts or
    compares equal; one such label can still be looked up.
    """

    __slots__ = ("_kept", "_missing", "_sorter", "_values", "dtype", "kind", "size", "sorted", "unique")

    _sort = True

    def __init__(self, labels: Any, op: str = "index", dim: Hashable = None):
        label_kind, values = kind(np.asarray(labels), op, dim)
        self._build(label_kind, values, op, dim)

    def _build(self, label_kind: str, values: np.ndarray, op: str, dim: Hashable) -> None:
        gaps = missing(values)
        kept = np.flatnonzero(~gaps)
        present = values[kept] if gaps.any() else values
        self.kind, self.dtype, self.size = label_kind, present.dtype, len(values)
        self._missing = np.flatnonzero(gaps)
        self._kept = None if len(kept) == len(values) else kept
        try:
            self.sorted = bool(np.all(present[1:] > present[:-1]))
            if self.sorted or not self._sort:
                self._sorter, self._values = None, present
            else:
                # Order among equal labels is irrelevant (repeats cannot be looked up), so the
                # faster unstable sort is enough.
                self._sorter = np.argsort(present)
                self._values = present[self._sorter]
        except TypeError:
            raise LabelError(f"{op}: labels on {dim!r} cannot be ordered, so they cannot be looked up") from None
        repeated = not self.sorted and bool(np.any(self._values[1:] == self._values[:-1]))
        self.unique = not repeated and len(self._missing) <= 1

    def get_indexer(self, wanted: Any) -> np.ndarray:
        result, values = _lookup(self, wanted)
        if values is None:
            return result
        gaps = missing(values)
        if gaps.any() and len(self._missing):
            result[gaps] = self._missing[0]
        rest = np.flatnonzero(~gaps)
        cast, exact = (values[rest], np.ones(len(rest), dtype=bool)) if values.dtype == self.dtype \
            else exact_cast(values[rest], self.dtype)
        if not len(self._values):
            return result
        found = np.searchsorted(self._values, cast)
        clipped = np.minimum(found, len(self._values) - 1)
        hit = exact & (found < len(self._values)) & (self._values[clipped] == cast)
        positions = clipped if self._sorter is None else self._sorter[clipped]
        if self._kept is not None:
            positions = self._kept[positions]
        result[rest[hit]] = positions[hit]
        return result

    def slice_indexer(self, start: Any = None, stop: Any = None) -> slice | np.ndarray:
        first, last = 0, len(self._values)
        if start is not None:
            bound = _bound(self, start, up=True)
            first = last if bound is None else int(np.searchsorted(self._values, bound, "left"))
        if stop is not None:
            bound = _bound(self, stop, up=False)
            last = 0 if bound is None else int(np.searchsorted(self._values, bound, "right"))
        last = max(first, last)
        if self._sorter is None:
            return slice(first, last) if self._kept is None else self._kept[first:last]
        positions = np.sort(self._sorter[first:last])
        return positions if self._kept is None else self._kept[positions]


class SortedIndex(_SearchIndex):
    """Strictly increasing labels (missing ones aside), searched directly: no sort, no copy."""

    __slots__ = ()

    _sort = False

    def _build(self, label_kind: str, values: np.ndarray, op: str, dim: Hashable) -> None:
        super()._build(label_kind, values, op, dim)
        if not self.sorted:
            raise LabelError(f"{op}: labels on {dim!r} are not strictly increasing; they need a sort-once index")


class SortOnceIndex(_SearchIndex):
    """Labels in any order, searched through one sort made when the index is built."""

    __slots__ = ()


# ── A regular grid ──────────────────────────────────────────────────────────


class GridIndex:
    """Exactly evenly spaced numbers, times, or durations, increasing or decreasing.

    Labels are held as a start, a step, and a count of integers, after
    scaling floats by a power of two that makes every label an integer
    (0.25 steps by 4), so lookups and ranges are integer arithmetic: exact,
    with no sort and no arrays held. Labels are a grid only when every step
    is exactly the same; ``np.arange(0, 1, 0.1)`` is not one, since its
    labels are not exactly 0.1 apart in binary.
    """

    __slots__ = ("_high", "_low", "_scale", "_start", "_step", "dtype", "kind", "size", "sorted", "unique")

    def __init__(self, labels: Any, op: str = "index", dim: Hashable = None):
        label_kind, values = kind(np.asarray(labels), op, dim)
        if not self._build(label_kind, values):
            raise LabelError(f"{op}: labels on {dim!r} are not evenly spaced; they need a search index")

    @classmethod
    def _of(cls, label_kind: str, values: np.ndarray) -> GridIndex | None:
        index = object.__new__(cls)
        return index if index._build(label_kind, values) else None

    def _build(self, label_kind: str, values: np.ndarray) -> bool:
        if len(values) < 2 or missing(values).any():
            return False
        domain = _integers(label_kind, values)
        if domain is None:
            return False
        integers, scale = domain
        start, step = int(integers[0]), int(integers[1]) - int(integers[0])
        low, high = int(integers.min()), int(integers.max())
        if step == 0 or high - low >= 2**63:
            return False
        if not np.all(np.diff(integers) == step):  # no overflow: the span fits in int64
            return False
        self.kind, self.dtype, self.size = label_kind, values.dtype, len(values)
        self.unique, self.sorted = True, step > 0
        self._scale, self._start, self._step, self._low, self._high = scale, start, step, low, high
        return True

    def _label(self, integer: int) -> Any:
        """The label at *integer* in the scaled domain, as a scalar of the labels' type."""
        if self.dtype.kind == "f":
            return self.dtype.type(math.ldexp(integer, -self._scale))
        if self.dtype.kind in "mM":
            return np.array(integer, dtype=np.int64).view(self.dtype)[()]
        return self.dtype.type(integer)

    def get_indexer(self, wanted: Any) -> np.ndarray:
        result, values = _lookup(self, wanted)
        if values is None:
            return result
        rest = np.flatnonzero(~missing(values))
        cast, exact = (values[rest], np.ones(len(rest), dtype=bool)) if values.dtype == self.dtype \
            else exact_cast(values[rest], self.dtype)
        with np.errstate(all="ignore"):
            low, high = self._label(self._low), self._label(self._high)
            inside = exact & (cast >= low) & (cast <= high)
            cast = np.where(inside, cast, low)  # outside values could overflow the integer domain
            if self.dtype.kind == "f":
                wide = cast.astype(np.float64)
                scaled = np.ldexp(wide, self._scale)
                # On the grid only if scaling lands on an integer and back on the value: a tiny
                # value scaled down can round to an integer.
                inside &= (scaled == np.floor(scaled)) & (np.ldexp(scaled, -self._scale) == wide)
                integers = np.where(inside, scaled, self._low).astype(np.int64)
            elif self.dtype.kind in "mM":
                integers = cast.view(np.int64)
            else:
                integers = cast.astype(np.int64)
            steps, remainder = np.divmod(integers - self._low, abs(self._step))
        hit = inside & (remainder == 0)
        positions = steps if self._step > 0 else self.size - 1 - steps
        result[rest[hit]] = positions[hit]
        return result

    def slice_indexer(self, start: Any = None, stop: Any = None) -> slice:
        empty = slice(0, 0)
        # Each bound becomes the condition low <= label or label <= high on the scaled integers,
        # as an exact fraction, and then a condition on positions.
        low = high = None
        if start is not None:
            low = self._scaled(_bound(self, start, up=True), up=True)
            if low is _NOTHING:
                return empty
        if stop is not None:
            high = self._scaled(_bound(self, stop, up=False), up=False)
            if high is _NOTHING:
                return empty
        if self._step < 0:
            # Dividing by a negative step turns label >= low into an upper limit on positions.
            low, high = high, low
        first = 0 if low is None else max(0, math.ceil((low - self._start) / self._step))
        last = self.size - 1 if high is None else min(self.size - 1, math.floor((high - self._start) / self._step))
        return slice(first, last + 1) if first <= last else empty

    def _scaled(self, bound: Any, up: bool) -> Any:
        """A bound in the scaled integer domain, None for no condition, or ``_NOTHING`` when no label qualifies."""
        if bound is None:
            return _NOTHING
        if self.dtype.kind == "f":
            if np.isinf(bound):  # the grid holds no infinities
                return None if (bound < 0) == up else _NOTHING
            return Fraction(float(bound)) * Fraction(2) ** self._scale
        if self.dtype.kind in "mM":
            return int(np.array(bound).view(np.int64))
        return int(bound)


# A range bound that no label can meet.
_NOTHING = object()


def _integers(label_kind: str, values: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Labels as int64 integers and the power of two that scaled them there; None if they have none.

    Times and durations are their int64 counts; units of varying length
    (months, years) have no even spacing. Floats are scaled by the power of
    two that makes the smallest bit of any label 1, if every label then fits.
    """
    dtype = values.dtype
    if label_kind in ("time", "duration"):
        unit, _ = np.datetime_data(dtype)
        return None if unit in ("Y", "M", "generic") else (values.view(np.int64), 0)
    if label_kind != "number" or dtype.kind == "c":
        return None
    if dtype.kind in "iu":
        if dtype == np.uint64 and values.max() > np.iinfo(np.int64).max:
            return None
        return values.astype(np.int64, copy=False), 0
    if dtype.itemsize > 8:
        return None
    wide = values.astype(np.float64)
    if not np.isfinite(wide).all():
        return None
    fraction, exponent = np.frexp(wide)  # wide = fraction * 2**exponent, 0.5 <= |fraction| < 1
    significand = np.ldexp(fraction, 53).astype(np.int64)  # exact: float64 has 53 bits
    nonzero = significand != 0
    if not nonzero.any():
        return None
    lowest_bit = significand[nonzero] & -significand[nonzero]
    _, bit_exponent = np.frexp(lowest_bit.astype(np.float64))
    scale = -int((exponent[nonzero] - 53 + bit_exponent - 1).min())
    if int(exponent[nonzero].max()) + scale > 62:  # the labels would not fit in int64 with room for their span
        return None
    return np.ldexp(wide, scale).astype(np.int64), scale


# ── Lookups and bounds, shared ──────────────────────────────────────────────


def _lookup(index: Any, wanted: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """An empty answer, and the wanted labels of the index's kind, or None when none can match."""
    wanted = np.atleast_1d(np.asarray(wanted))
    if wanted.ndim != 1:
        raise LabelError("look up labels as a one-dimensional list")
    result = np.full(len(wanted), -1, dtype=np.intp)
    if not len(wanted) or not index.size:
        return result, None
    if not index.unique:
        raise LabelError("labels repeat, so a label does not name one position")
    return result, _of_kind(index.kind, wanted)


def _of_kind(label_kind: str, wanted: np.ndarray) -> np.ndarray | None:
    """*wanted* as labels of *label_kind*, with text read as times for time labels; None if of another kind."""
    try:
        wanted_kind, values = kind(wanted, "look up", None)
    except LabelError:
        return None  # a label of a different type is simply not present
    if label_kind == "time" and wanted_kind in ("text", "bytes"):
        try:
            values, wanted_kind = np.asarray(values, dtype="datetime64"), "time"
        except ValueError:
            return None
    return values if wanted_kind == label_kind else None


def _bound(index: Any, value: Any, up: bool) -> Any:
    """The labels' type's nearest value inside a range bound, so comparisons with it are exact.

    *up* for a start: the least value at or above *value*; otherwise the
    greatest at or below it. None when no value of the type qualifies.
    """
    values = _of_kind(index.kind, np.atleast_1d(np.asarray(value, dtype=object if isinstance(value, str) else None)))
    if values is None or values.shape != (1,):
        raise LabelError(f"a range of {index.kind} labels cannot start or stop at {value!r}")
    if missing(values)[0]:
        raise LabelError("a range cannot start or stop at a missing label")
    dtype = index.dtype
    if dtype.kind == "c" or values.dtype.kind == "c":
        raise LabelError("complex labels have no order, so they have no range")
    if dtype.kind in "iu":
        return _integer_bound(values[0].item(), dtype, up)
    if dtype.kind == "f":
        return _float_bound(values[0].item(), dtype, up)
    if dtype.kind in "mM":
        return _time_bound(values[0], dtype, up)
    return values[0]  # text, bytes, booleans, and registered kinds compare as they are


def _integer_bound(value: float, dtype: np.dtype, up: bool) -> Any:
    info = np.iinfo(dtype)
    if isinstance(value, float):
        if math.isinf(value):
            if (value > 0) == up:
                return None  # starting above every integer, or stopping below every one
            value = info.min if up else info.max
        else:
            value = math.ceil(value) if up else math.floor(value)  # exact for any float
    if up:
        return None if value > info.max else dtype.type(max(value, info.min))
    return None if value < info.min else dtype.type(min(value, info.max))


def _float_bound(value: float, dtype: np.dtype, up: bool) -> Any:
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            bound = dtype.type(value)  # rounded to nearest, so possibly on the wrong side
        except OverflowError:  # an integer beyond every float
            bound = dtype.type(math.inf if value > 0 else -math.inf)
    toward = dtype.type(math.inf if up else -math.inf)
    # Python compares floats and integers exactly, so this steps over at most one float.
    while (float(bound) < value) if up else (float(bound) > value):
        bound = np.nextafter(bound, toward)
    return bound


def _time_bound(value: Any, dtype: np.dtype, up: bool) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bound = np.array(value).astype(dtype)  # to a coarser unit NumPy rounds down
            # NumPy compares times across units exactly, in the finer unit.
            beyond = bool(bound < value) if up else bool(bound > value)
    except (TypeError, ValueError):  # durations in months or years have no fixed length
        raise LabelError(f"{value!r} cannot be compared exactly with labels of type {dtype}") from None
    count = int(bound.view(np.int64)) + (0 if not beyond else 1 if up else -1)
    return np.array(count, dtype=np.int64).view(dtype)[()]


# ── Choosing an index ───────────────────────────────────────────────────────

_REGISTERED: dict[str, type] = {}


def index_for(labels: Any, op: str = "index", dim: Hashable = None) -> LabelIndex:
    """The index for a label array, chosen by looking at the labels once.

    A class registered for the labels' kind (``register_index``) is used
    first; otherwise a grid index for evenly spaced numbers and times, a
    sorted index for strictly increasing labels, and a sort-once index for
    the rest. *op* and *dim* name the operation and axis in errors.
    """
    label_kind, values = kind(np.asarray(labels), op, dim)
    registered = _REGISTERED.get(label_kind)
    if registered is not None:
        return registered(labels, op, dim)
    grid = GridIndex._of(label_kind, values)
    if grid is not None:
        return grid
    index = object.__new__(SortOnceIndex)
    index._build(label_kind, values, op, dim)
    if index.sorted:
        index.__class__ = SortedIndex  # searched directly, as a sorted index is built
    return index


def register_index(label_kind: str, cls: type | None, *, types: tuple[type, ...] = ()) -> type | None:
    """Use *cls* for labels of *label_kind* from now on; return the class it replaces, or None.

    *label_kind* is a built-in kind (``labelarray.labels.KINDS``), or a new
    one whose labels are objects of *types*, so that labels of those types
    are that kind wherever labels are compared. ``index_for`` calls
    ``cls(labels, op, dim)``; None removes the registration.

    A registered class must follow the exact label rule, which no index may
    loosen: give the index a ``kind``, match a wanted label only when it is
    of that kind and exactly equal, converting lookups to the labels' type
    with ``exact_cast`` and never by rounding or a tolerance, so that
    ``2**53 + 1`` never matches ``2**53``; compare range bounds exactly; and
    raise ``LabelError`` for lookups when labels repeat. Labels of another
    kind are simply absent.
    """
    if cls is not None:
        lacking = [name for name in ("get_indexer", "slice_indexer") if not callable(getattr(cls, name, None))]
        if lacking:
            raise TypeError(f"{cls.__name__} has no {' or '.join(lacking)}, which every label index answers")
    previous = _REGISTERED.pop(label_kind, None)
    if cls is not None:
        _REGISTERED[label_kind] = cls
    if label_kind not in KINDS.values():
        register_types(label_kind, types if cls is not None else ())
    elif types:
        raise ValueError(f"{label_kind} is a built-in label kind; types name the labels of a new kind")
    return previous
