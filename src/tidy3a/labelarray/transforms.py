"""Key transforms: what a key means, applied before positions are grouped (plan milestone 10, step A4).

Grouping itself is on exact equality only: positions share a group when
their keys are the same value, with fixed rules for floats (every NaN is
one missing key, and signed zeros are one value). Domain meaning lives here,
in a transform that turns key values into group codes and labels first:
ranges between edges (``Bins``), windows of time (``Duration``), numbers
rounded to a step (``Round``), and cells of a regular grid (``GridCells``).

The built-in transforms are small records holding their settings. Two
records are equal when they are the same class with equal settings, so a
record can key a cache of groupings; each prints its settings, and each
gives readable group labels in sorted order. A plain function is a
transform too (``as_transform``): it maps key values to keys, which are
then grouped exactly, and since functions cannot be compared it is equal to
nothing but itself.

Every transform puts a value on one side of each bucket edge, so values
either side of an edge fall into different groups however close they are.
That is by design, and the edges are the user's to place: bin edges, or a
step's or cell's ``origin``. Steps and origins are read as the decimals they
print as, so with a step of 0.1 the value 0.3 is in the cell printed
``[0.3, 0.4)``, although ``0.3 / 0.1`` is 2.9999999999999996 in floats.

This module needs NumPy only.
"""

from __future__ import annotations

import datetime
import math
import re
import warnings
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from typing import Any

import numpy as np

from tidy3a.labelarray.errors import GroupError, ShapeError
from tidy3a.labelarray.labels import missing

# Messages about a transform's settings or values are clauses, meant to follow
# the call that made the transform ('bins(t, "2 days"): a duration needs time
# values'); ``outside`` and ``factorize`` give whole sentences.


class Transform:
    """Base class for the built-in transforms.

    ``groups(values)`` takes key values none of which are missing (NaN or
    NaT) and gives one code per value and the group labels in sorted order;
    a value that belongs to no group (outside every bin, or infinite) gets
    the code -1. ``factorize`` adds the missing group and refuses code -1.
    """

    # Text labels ("[0, 15)") label the missing group with text too, so the
    # labels stay of one type; number labels use NaN itself.
    text_labels = True

    def groups(self, values: np.ndarray) -> tuple[np.ndarray, list]:
        raise NotImplementedError

    def outside(self, count: int, name: str | None = None) -> str:
        """The message for *count* values that fall in no group."""
        return f"{count:,} positions{_of(name)} fall in no group of {self!r}"

    def missing_label(self, dtype: np.dtype) -> Any:
        """The label of the group of missing values, for key values of *dtype*."""
        if self.text_labels:
            return "NaT" if dtype.kind in "mM" else "NaN"
        return math.nan

    def _settings(self) -> tuple:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._settings() == other._settings()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._settings()))


class Function:
    """A plain function as a transform: it maps key values to keys, which are grouped exactly.

    Functions cannot be compared for what they compute, so a ``Function``
    equals only itself, and a grouping by one is never taken for another's.
    """

    text_labels = False

    def __init__(self, function: Callable[[np.ndarray], Any]):
        self.function = function

    def __repr__(self) -> str:
        return f"Function({getattr(self.function, '__qualname__', repr(self.function))})"


def as_transform(transform: Any) -> Transform | Function | None:
    """*transform* as a transform: a record as it is, a function wrapped in ``Function``, None as None."""
    if transform is None or isinstance(transform, (Transform, Function)):
        return transform
    if callable(transform):
        return Function(transform)
    raise TypeError(f"a transform is a record such as Round(0.5) or a function of the key values, not {transform!r}")


def factorize(values: Any, transform: Any = None, *, dropna: bool = True) -> tuple[np.ndarray, list]:
    """Group codes for *values* after *transform*, and the group labels in sorted order.

    Without a transform, positions are grouped by exactly equal values. Every
    missing value (NaN of any sign or payload, NaT) is one missing key:
    ``dropna=True`` gives those positions the code -1 and no group;
    ``dropna=False`` makes them the last group, labelled NaN (or NaT; text
    for transforms whose labels are text). A value in no group of the
    transform, such as one outside every bin, is an error: values are never
    dropped silently.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ShapeError(f"factorize needs one key per position, a 1-D array, not {values.ndim} axes")
    transform = as_transform(transform)
    if isinstance(transform, Function):
        keys = np.asarray(transform.function(values))
        if keys.shape != values.shape:
            raise GroupError(f"{transform!r} must give one key per value: {values.shape[0]:,} values gave "
                             f"shape {keys.shape}")
        return factorize(keys, dropna=dropna)
    absent = missing(values)
    present = values[~absent]
    if transform is None:
        found, codes_present = np.unique(present, return_inverse=True)
        labels = _label_list(found)
        missing_label = values.dtype.type("NaT") if values.dtype.kind in "mM" else math.nan
    else:
        codes_present, labels = transform.groups(present)
        outside = codes_present < 0
        if outside.any():
            raise GroupError(transform.outside(int(outside.sum())))
        missing_label = transform.missing_label(values.dtype)
    codes = np.full(values.shape, -1, dtype=np.intp)
    codes[~absent] = codes_present.reshape(-1)
    if not dropna and absent.any():
        codes[absent] = len(labels)
        labels = [*labels, missing_label]
    return codes, labels


# ── Bins ────────────────────────────────────────────────────────────────────


class Bins(Transform):
    """Left-closed, right-open ranges between *edges*: ``[edges[i], edges[i + 1])``.

    Every bin is a group, empty or not, labelled ``"[low, high)"``, so a
    result shows where nothing fell. A value on an edge is in the bin that
    starts there, so values either side of an edge fall into different
    bins, by design; move the edges to move the split. A value outside every
    bin is in no group, which ``factorize`` refuses.
    """

    def __init__(self, edges: Any):
        edges = np.array(edges)  # a copy, so the caller's array stays writeable
        _refuse_zones(edges)
        if edges.ndim != 1 or len(edges) < 2 or not (edges[1:] > edges[:-1]).all():
            raise GroupError("needs at least two increasing edges")
        edges.setflags(write=False)
        self.edges = edges

    def groups(self, values: np.ndarray) -> tuple[np.ndarray, list]:
        _refuse_zones(values)
        codes = np.searchsorted(self.edges, values, side="right") - 1
        codes[codes >= len(self.edges) - 1] = -1
        return codes, [f"[{format_edge(a)}, {format_edge(b)})" for a, b in pairwise(self.edges)]

    def outside(self, count: int, name: str | None = None) -> str:
        low, high = format_edge(self.edges[0]), format_edge(self.edges[-1])
        return (f"{count:,} positions{_of(name)} fall outside every bin [{low}, {high}); they are never "
                "dropped silently.\n  Widen the edges, or filter those positions out first.")

    def _settings(self) -> tuple:
        kind = self.edges.dtype.kind
        if kind == "M":
            return ("time", tuple(_time_key(edge) for edge in self.edges))
        if kind == "m":
            return ("duration", tuple(_time_key(edge) for edge in self.edges))
        # Whole-number float edges group as integer edges do, but may print differently
        # (1e+20 against 100000000000000000000), so they are other settings.
        return ({"i": "int", "u": "int", "f": "float"}.get(kind, kind), tuple(self.edges.tolist()))

    def __repr__(self) -> str:
        if self.edges.dtype.kind in "mM":
            texts = [f'"{format_edge(edge)}"' for edge in self.edges]
        else:
            texts = [_number_text(edge) if isinstance(edge, float) else repr(edge) for edge in self.edges.tolist()]
        return f"Bins(edges=[{', '.join(texts)}])"


# ── Duration ────────────────────────────────────────────────────────────────


class Duration(Transform):
    """Windows of time of a fixed *width*: ``[origin + k * width, origin + (k + 1) * width)``.

    *width* is text such as ``"30 days"`` or ``"1h30min"`` (``parse_duration``)
    or a timedelta; months and years have no fixed length and are not
    widths. *origin* places the windows; by default it is the first time
    present. Every window from the first holding a time to the last is a
    group, empty or not, as with ``Bins``. Times either side of a window
    edge fall into different windows, by design; move the origin to move
    the edges. Times with a time zone are refused, as everywhere in
    labelarray: convert them to times without a zone first.
    """

    def __init__(self, width: Any, origin: Any = None):
        self.width = _width(width)
        self.origin = None if origin is None else _instant(origin)

    def groups(self, values: np.ndarray) -> tuple[np.ndarray, list]:
        _refuse_zones(values)
        if values.dtype.kind != "M":
            raise GroupError("a duration needs time values")
        if self.origin is None and values.size == 0:
            raise GroupError("there are no times to bin")
        if values.size == 0:
            return np.zeros(0, dtype=np.intp), []
        start = values.min() if self.origin is None else self.origin
        # Integer arithmetic on the time counts: a window is exact to the nanosecond.
        steps = (values - start) // self.width
        first, last = int(steps.min()), int(steps.max())
        edges = start + np.arange(first, last + 2) * self.width
        labels = [f"[{format_edge(a)}, {format_edge(b)})" for a, b in pairwise(edges)]
        return (steps - first).astype(np.intp), labels

    def _settings(self) -> tuple:
        return (int(self.width.astype(np.int64)), None if self.origin is None else _time_key(self.origin))

    def __repr__(self) -> str:
        origin = "None" if self.origin is None else f'"{format_edge(self.origin)}"'
        return f'Duration(width="{duration_text(self.width)}", origin={origin})'


# Duration units a duration's text may use, in nanoseconds. Months and years are not
# fixed lengths, and "m" could be either minutes or months, so none of them are units.
_WEEK, _DAY, _HOUR, _MINUTE, _SECOND = 7 * 86_400 * 10**9, 86_400 * 10**9, 3_600 * 10**9, 60 * 10**9, 10**9
_UNITS = {
    **dict.fromkeys(("w", "week", "weeks"), _WEEK),
    **dict.fromkeys(("d", "day", "days"), _DAY),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), _HOUR),
    **dict.fromkeys(("min", "mins", "minute", "minutes"), _MINUTE),
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), _SECOND),
    **dict.fromkeys(("ms", "millisecond", "milliseconds"), 10**6),
    **dict.fromkeys(("us", "microsecond", "microseconds"), 10**3),
    **dict.fromkeys(("ns", "nanosecond", "nanoseconds"), 1),
}
_NUMBER_UNIT = r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)"


def parse_duration(text: str) -> np.timedelta64:
    """A duration such as "30 days", "2D", or "1h30min": numbers with units, added exactly.

    labelarray reads durations itself, so a duration means the same whether
    or not pandas is installed.
    """
    parts = re.findall(_NUMBER_UNIT, text)
    rest = re.sub(_NUMBER_UNIT, "", text).strip()
    units = [_UNITS.get(unit.lower()) for _, unit in parts]
    total = sum((Fraction(number) * unit for (number, _), unit in zip(parts, units) if unit), Fraction(0))
    if not parts or rest or not all(units) or total.denominator != 1 or total <= 0:
        raise GroupError(
            "cannot read the duration; write numbers with units "
            '(weeks, days, hours, minutes, seconds, ms, us, ns), e.g. "30 days" or "1h30min", '
            "a whole number of nanoseconds; months and years have no fixed length"
        )
    return np.timedelta64(int(total), "ns")


_TEXT_UNITS = ((_DAY, "day", "days"), (_HOUR, "hour", "hours"), (_MINUTE, "minute", "minutes"),
               (_SECOND, "second", "seconds"), (10**6, "ms", "ms"), (10**3, "us", "us"), (1, "ns", "ns"))


def duration_text(width: np.timedelta64) -> str:
    """*width* as text ``parse_duration`` reads back: "2 days", "1 hour 30 minutes", "250 ms"."""
    left = int(width.astype("timedelta64[ns]").astype(np.int64))
    parts = []
    for size, one, many in _TEXT_UNITS:
        count, left = divmod(left, size)
        if count:
            parts.append(f"{count} {one if count == 1 else many}")
    return " ".join(parts)


def _width(width: Any) -> np.timedelta64:
    if isinstance(width, str):
        value = parse_duration(width)
    else:
        if hasattr(width, "to_timedelta64"):  # pandas' Timedelta, without importing pandas
            width = width.to_timedelta64()
        if isinstance(width, datetime.timedelta):
            width = np.timedelta64(width)
        if not isinstance(width, np.timedelta64):
            raise GroupError(f'a duration is text such as "30 days" or a timedelta, not {width!r}')
        if np.isnat(width) or np.datetime_data(width.dtype)[0] in ("Y", "M"):
            raise GroupError(f"the duration {width} has no fixed length")
        value = width.astype("timedelta64[ns]")
        if value.astype(width.dtype) != width:
            raise GroupError(f"the duration {width} is not a whole number of nanoseconds")
    if value <= np.timedelta64(0, "ns"):
        raise GroupError(f"a duration must be longer than zero, not {width!r}")
    return value


def _instant(origin: Any) -> np.datetime64:
    """An origin for windows of time, as a NumPy time; a time with a zone is refused."""
    if getattr(origin, "tzinfo", None) is not None:
        raise GroupError(ZONES)
    if hasattr(origin, "to_datetime64"):  # pandas' Timestamp, without importing pandas
        origin = origin.to_datetime64()
    try:
        with warnings.catch_warnings():
            # NumPy reads "2026-01-01T00:00+02:00" as UTC with a warning; a zone is refused instead.
            warnings.simplefilter("error")
            value = origin if isinstance(origin, np.datetime64) else np.datetime64(origin)
    except UserWarning:
        raise GroupError(ZONES) from None
    except (TypeError, ValueError):
        raise GroupError(f"an origin is a time, not {origin!r}") from None
    if np.isnat(value):
        raise GroupError("an origin is a time, not NaT")
    return value


# The clause refusing times with a time zone, also for callers whose arrays carry the zone in their type.
ZONES = "needs times without a time zone; convert them to one zone and drop it (to UTC, say) first"


def _refuse_zones(values: np.ndarray) -> None:
    """Times with a time zone, which reach here only as objects, are refused."""
    if values.dtype.kind == "O" and any(getattr(value, "tzinfo", None) is not None for value in values.flat):
        raise GroupError(ZONES)


# ── Round and GridCells ─────────────────────────────────────────────────────


class _Steps(Transform):
    """Numbers placed on a regular grid ``origin + k * step``, held exactly.

    The step and origin are read as the decimals they print as (``0.1`` is
    one tenth), and each edge is the float nearest the exact edge, so a
    value is in a bucket exactly when it lies between the bucket's edges as
    printed. Integer values with whole-number settings are placed in
    integer arithmetic and labelled with integers. Infinite values are in
    no bucket.
    """

    _step_name = "step"
    _bucket = "step"

    def __init__(self, step: Any, origin: Any = 0):
        self._step = _decimal(step, self._step_name, positive=True)
        self._origin = _decimal(origin, "origin")

    @property
    def origin(self) -> Fraction:
        return self._origin

    def _placed(self, values: np.ndarray, start: Fraction) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Each finite value's bucket k, with ``start + k * step <= value < start + (k + 1) * step``.

        Also which values are finite, and the values as float64 (None when
        they were placed in integer arithmetic).
        """
        values = _numbers(values)
        line = _Line(start, self._step)
        if values.dtype.kind in "iu":
            return line.floor_integers(values), np.ones(values.shape, dtype=bool), None
        floats = values.astype(np.float64)  # exact for float16 and float32
        finite = np.isfinite(floats)
        return line.floor(floats[finite]), finite, floats

    def _whole(self, values: np.ndarray) -> bool:
        return values.dtype.kind in "iu" and self._step.denominator == 1 and self._origin.denominator == 1

    def groups(self, values: np.ndarray) -> tuple[np.ndarray, list]:
        start = self._start()
        steps, finite, _ = self._placed(values, start)
        found, codes = _occupied(steps)
        out = np.full(len(finite), -1, dtype=np.intp)
        out[finite] = codes
        return out, self._labels(found, values)

    def outside(self, count: int, name: str | None = None) -> str:
        return (f"{count:,} positions{_of(name)} are infinite, in no {self._bucket}; they are never dropped "
                "silently.\n  Filter those positions out first.")

    def _settings(self) -> tuple:
        return (self._step, self._origin)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self._step_name}={_fraction_text(self._step)}, "
                f"origin={_fraction_text(self._origin)})")


class Round(_Steps):
    """Numbers rounded to the nearest ``origin + k * step``, each group labelled by that number.

    Every bucket is ``[c - step / 2, c + step / 2)`` around its rounded value
    ``c``, so halves round up, toward positive infinity, not to even as
    ``np.round`` does: one rule for every bucket. Values either side of a
    bucket edge fall into different groups however close they are, by
    design; move the origin to move the edges. Only rounded values that
    occur are groups. Labels are floats, or integers for integer values
    with a whole-number step and origin.
    """

    text_labels = False

    @property
    def step(self) -> Fraction:
        return self._step

    def _start(self) -> Fraction:
        return self._origin - self._step / 2

    def _labels(self, found: np.ndarray, values: np.ndarray) -> list:
        centres = _Line(self._origin, self._step)
        if self._whole(values):
            return [int(value) for value in centres.exact(found)]
        return centres.at(found).tolist()


class GridCells(_Steps):
    """Cells ``[origin + k * size, origin + (k + 1) * size)`` of a regular grid, labelled ``"[low, high)"``.

    One cell per value of a coordinate: used per axis, or combined across
    x, y, and z, the cells bin points into voxels. Only cells holding a
    value are groups, since a grid can be large and sparse. Values either
    side of a cell edge fall into different cells however close they are,
    by design; move the origin to move the edges.
    """

    _step_name = "size"
    _bucket = "cell"

    @property
    def size(self) -> Fraction:
        return self._step

    def _start(self) -> Fraction:
        return self._origin

    def _labels(self, found: np.ndarray, values: np.ndarray) -> list:
        edges = _Line(self._origin, self._step)
        if self._whole(values):
            lows = [str(int(value)) for value in edges.exact(found)]
            highs = [str(int(value)) for value in edges.exact(found + 1)]
        else:
            lows = [_number_text(value) for value in edges.at(found).tolist()]
            highs = [_number_text(value) for value in edges.at(found + 1).tolist()]
        return [f"[{low}, {high})" for low, high in zip(lows, highs)]


class _Line:
    """The points ``start + k * step`` for rational *start* and *step*, as exact numbers or nearest floats."""

    def __init__(self, start: Fraction, step: Fraction):
        self.denominator = math.lcm(start.denominator, step.denominator)
        self.start = int(start * self.denominator)
        self.step = int(step * self.denominator)
        self.float_start, self.float_step = float(start), float(step)

    def exact(self, ks: np.ndarray) -> list[Fraction]:
        return [Fraction(self.start + int(k) * self.step, self.denominator) for k in ks.tolist()]

    def at(self, ks: np.ndarray) -> np.ndarray:
        """The float nearest each point ``start + k * step``."""
        if ks.size == 0:
            return np.zeros(0)
        low, high = int(ks.min()), int(ks.max())
        reach = max(abs(self.start + low * self.step), abs(self.start + high * self.step))
        if reach < 2**53 and self.denominator < 2**53:
            # Both operands are exact floats, and a float division is correctly rounded.
            return (self.start + ks * self.step).astype(np.float64) / float(self.denominator)
        found, inverse = np.unique(ks, return_inverse=True)
        return np.array([float(point) for point in self.exact(found)])[inverse.reshape(-1)]

    def floor(self, values: np.ndarray) -> np.ndarray:
        """Each float value's k, with ``at(k) <= value < at(k + 1)``."""
        with np.errstate(all="ignore"):
            guess = np.floor((values - self.float_start) / self.float_step)
        if guess.size and np.abs(guess).max() >= 2**50:
            raise GroupError(f"values up to {np.abs(values).max():g} are too far from the origin to place in steps "
                             f"of {self.float_step:g}; choose a nearer origin or a larger step")
        ks = guess.astype(np.int64)
        # The guess rounds twice, so it can be one off near an edge; the edges decide.
        while True:
            down = values < self.at(ks)
            up = values >= self.at(ks + 1)
            if not (down.any() or up.any()):
                return ks
            ks = ks - down + up

    def floor_integers(self, values: np.ndarray) -> np.ndarray:
        """Each integer value's k, in integer arithmetic: ``(value * d - start) // step``."""
        reach = (max(abs(int(values.min())), abs(int(values.max()))) * self.denominator + abs(self.start)
                 if values.size else 0)
        if reach < 2**63:
            return (values.astype(np.int64) * self.denominator - self.start) // self.step
        return np.array([(int(v) * self.denominator - self.start) // self.step for v in values.tolist()], dtype=object)


def _occupied(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The buckets that hold values, in order, and each value's code: ``np.unique``'s answer.

    Bucket numbers are integers in a range rarely much wider than the
    values are many, so counting them finds the occupied ones without a sort.
    """
    if steps.dtype == np.int64 and steps.size:
        low, high = int(steps.min()), int(steps.max())
        if high - low < 4 * steps.size + 1_024:
            offsets = steps - low
            occupied = np.bincount(offsets, minlength=high - low + 1) > 0
            ranks = np.cumsum(occupied) - 1
            return np.flatnonzero(occupied) + low, ranks[offsets].astype(np.intp)
    found, codes = np.unique(steps, return_inverse=True)
    return found, codes.reshape(-1)


def _numbers(values: np.ndarray) -> np.ndarray:
    _refuse_zones(values)
    kind = values.dtype.kind
    if kind in "iuf":
        return values
    what = {"b": "booleans", "c": "complex numbers", "M": "times; group times with a Duration",
            "m": "durations", "U": "text", "S": "bytes"}.get(kind, f"{values.dtype} values")
    raise GroupError(f"needs numbers, not {what}")


def _decimal(value: Any, what: str, positive: bool = False) -> Fraction:
    """A setting as the exact number it prints as: the float 0.1 is one tenth."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Fraction, Decimal, np.number)):
        raise GroupError(f"the {what} must be a number, not {value!r}")
    if isinstance(value, (complex, np.complexfloating)):
        raise GroupError(f"the {what} must be a real number, not {value!r}")
    if isinstance(value, (float, np.floating, Decimal)) and not math.isfinite(value):
        raise GroupError(f"the {what} must be a finite number, not {value!r}")
    # str() gives the shortest decimal that reads back as the same float (float32's too).
    number = Fraction(str(value)) if isinstance(value, (float, np.floating)) else Fraction(value)
    if positive and number <= 0:
        raise GroupError(f"the {what} must be a positive number, not {value!r}")
    return number


# ── Labels ──────────────────────────────────────────────────────────────────


def format_edge(value: Any) -> str:
    """An edge as a label shows it: times as dates where whole days, numbers in ``%g``."""
    if isinstance(value, np.datetime64):
        nanoseconds = value.astype("datetime64[ns]")
        if nanoseconds == nanoseconds.astype("datetime64[D]"):
            return str(np.datetime_as_string(nanoseconds, unit="D"))
        for unit in ("s", "us"):  # as pandas prints: whole seconds, microseconds, else nanoseconds
            if nanoseconds == nanoseconds.astype(f"datetime64[{unit}]"):
                return str(np.datetime_as_string(nanoseconds, unit=unit))
        return str(np.datetime_as_string(nanoseconds, unit="ns"))
    if isinstance(value, np.generic) and value.dtype.kind != "m":
        value = value.item()
    return f"{value:g}" if isinstance(value, float) else str(value)


def _number_text(value: float) -> str:
    """The shortest text that reads back as *value*, without a trailing ".0"."""
    return repr(float(value)).removesuffix(".0")


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    as_float = float(value)
    return _number_text(as_float) if Fraction(str(as_float)) == value else f"{value.numerator}/{value.denominator}"


def _label_list(labels: np.ndarray) -> list:
    """Labels as Python values, times and durations as NumPy scalars."""
    if labels.dtype.kind in "mM":
        return list(labels)
    return labels.tolist()


def _of(name: str | None) -> str:
    return "" if name is None else f" of {name}"


# Seconds per NumPy time unit, so equal times in different units are equal settings.
_SECONDS = {"W": 604_800, "D": 86_400, "h": 3_600, "m": 60, "s": 1, "ms": Fraction(1, 10**3),
            "us": Fraction(1, 10**6), "ns": Fraction(1, 10**9), "ps": Fraction(1, 10**12),
            "fs": Fraction(1, 10**15), "as": Fraction(1, 10**18)}


def _time_key(value: np.generic) -> Any:
    unit, count = np.datetime_data(value.dtype)
    ticks = int(value.astype(np.int64)) * count
    return Fraction(ticks) * _SECONDS[unit] if unit in _SECONDS else (unit, ticks)
