"""Prototype: a NumPy-only labelled-array core that could replace xarray.

Not part of the package; ``src/`` does not import it. It tests the design
discussed after the 0.1.0 release (docs/plan.md §13.1): a small immutable
``Variable`` (data, dims, attrs) borrows an array with NumPy's interface,
coordinate labels live once on a ``Dataset`` and are shared by every
variable with that axis, and one dispatcher turns axis names into axis
numbers, checks labels, calls NumPy, and builds the result's metadata.

The rules are tidy3A's (plan §2, §5.3, §5.5): broadcasting is by name;
labels on a shared axis must match exactly, or match in another order and
are reordered to the first operand; a labelled axis never pairs with an
unlabelled one; two unlabelled axes pair by position; cutting an unlabelled
axis first attaches its 1-based positions as labels.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import warnings
import weakref

import numpy as np


class CoreError(ValueError):
    """A name, shape, or label problem, reported before any computation."""


# ── Variable ────────────────────────────────────────────────────────────────


class Variable:
    """Borrowed array storage with read-only dimension and attribute properties.

    Storage ownership is explicit: numpy_pipeline frames take read-only
    snapshots; a standalone Variable does not copy or freeze its caller's data.
    """

    __slots__ = ("_data", "_dims", "_attrs")

    def __init__(self, data: Any, dims: Sequence[Hashable], attrs: Mapping[str, Any] | None = None):
        data = data if hasattr(data, "__array_function__") else np.asarray(data)
        dims = tuple(dims)
        if len(dims) != data.ndim:
            raise CoreError(f"{len(dims)} names {dims} for an array of rank {data.ndim}")
        if len(set(dims)) != len(dims):
            raise CoreError(f"axis names must be unique, got {dims}")
        self._data = data
        self._dims = dims
        self._attrs = MappingProxyType(dict(attrs or {}))

    data = property(lambda self: self._data)
    dims = property(lambda self: self._dims)
    attrs = property(lambda self: self._attrs)
    dtype = property(lambda self: self._data.dtype)

    @property
    def sizes(self) -> dict[Hashable, int]:
        return dict(zip(self._dims, self._data.shape))

    def axis(self, name: Hashable, op: str) -> int:
        """The axis number of *name*, or an error naming the operation."""
        try:
            return self._dims.index(name)
        except ValueError:
            raise CoreError(f"{op}: no axis {name!r}; axes are {self._dims}") from None

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {n}" for d, n in self.sizes.items())
        return f"<Variable ({sizes}) {self.dtype}>"


# ── Labelled: a variable with the labels of its axes ────────────────────────


@dataclass(frozen=True)
class Labelled:
    """A variable plus labels for those of its axes that have them.

    ``Dataset[name]`` returns one; every operation takes and returns them, so
    expressions chain without glue. ``notes`` records what was changed on the
    way, such as a reordering, as tidy3A's inspection output does.
    """

    var: Variable
    coords: Mapping[Hashable, np.ndarray] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self.var.dims

    @property
    def sizes(self) -> dict[Hashable, int]:
        return self.var.sizes

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def dtype(self) -> np.dtype:
        return self.var.dtype

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.var.data)

    def __repr__(self) -> str:
        labelled = ", ".join(map(str, self.coords)) or "none"
        return f"<Labelled {self.var!r}; labelled axes: {labelled}>"


def labelled(data: Any, dims: Sequence[Hashable], coords: Mapping[Hashable, Any] | None = None,
             attrs: Mapping[str, Any] | None = None) -> Labelled:
    """Build a ``Labelled`` from raw values, checking labels against sizes."""
    var = Variable(data, dims, attrs)
    return Labelled(var, _checked_coords(coords or {}, var.sizes))


def _checked_coords(coords: Mapping[Hashable, Any], sizes: Mapping[Hashable, int]) -> dict:
    out = {}
    for dim, labels in coords.items():
        labels = _frozen_labels(labels)
        if dim not in sizes:
            raise CoreError(f"labels for {dim!r}, which is not an axis; axes are {tuple(sizes)}")
        if labels.ndim != 1 or len(labels) != sizes[dim]:
            raise CoreError(f"{dim!r} has {sizes[dim]} positions but {labels.size} labels")
        out[dim] = labels
    return out


def _frozen_labels(labels: Any) -> np.ndarray:
    """Labels that can never change, so their index can be cached by identity.

    A read-only array is kept as it is only when nothing it views is
    writeable; anything else is copied once and the copy is frozen.
    """
    if isinstance(labels, np.ndarray) and _never_changes(labels):
        return labels
    frozen = np.array(labels, copy=True)
    frozen.flags.writeable = False
    return frozen


def _never_changes(array: np.ndarray) -> bool:
    base: Any = array
    while base is not None:
        if not isinstance(base, np.ndarray) or base.flags.writeable:
            return False
        base = base.base
    return True


def _freeze(array: np.ndarray) -> np.ndarray:
    """Freeze a label array this module just created and owns alone."""
    array.flags.writeable = False
    return array


# ── Label index ─────────────────────────────────────────────────────────────

# Label kinds that can match each other. Booleans, numbers, times, durations,
# text, and bytes never match across kinds, as in pandas: True is not 1.
_KINDS = {"b": "bool", "i": "number", "u": "number", "f": "number", "c": "number",
          "M": "time", "m": "duration", "U": "text", "S": "bytes"}


def _kind(values: np.ndarray, op: str, dim: Hashable) -> tuple[str, np.ndarray]:
    """The label kind, with object arrays of one Python type converted to it."""
    if values.dtype.kind != "O":
        return _KINDS.get(values.dtype.kind, "other"), values
    if values.size == 0:
        return "other", values
    types = {type(v) for v in values.tolist()}
    for kind, accepted, convert in (
        ("text", (str,), str), ("bytes", (bytes,), bytes), ("bool", (bool, np.bool_), bool),
    ):
        if all(issubclass(t, accepted) for t in types):
            return kind, np.array(values.tolist(), dtype=convert)
    if all(issubclass(t, (int, float, complex, np.number)) and not issubclass(t, (bool, np.bool_)) for t in types):
        return "number", np.array(values.tolist())
    names = ", ".join(sorted(t.__name__ for t in types))
    raise CoreError(f"{op}: labels on {dim!r} mix types ({names}), so they cannot be ordered or looked up; "
                    "give the axis labels of one type")


def _missing(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "fc":
        return np.isnan(values)
    if values.dtype.kind in "mM":
        return np.isnat(values)
    return np.zeros(values.shape, dtype=bool)


def _exact_cast(wanted: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """*wanted* in the index's type, and where that conversion is exact.

    A value that changes on the way, such as float64 0.1 as float32, a
    half-second as whole seconds, or -1 as an unsigned integer, cannot equal
    any label, so it is not found; nothing is rounded to make a match.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cast = wanted.astype(dtype)
        exact = cast.astype(wanted.dtype) == wanted
    if dtype.kind == "u" and wanted.dtype.kind in "if":
        exact &= wanted >= 0
    elif dtype.kind == "i" and wanted.dtype.kind == "u":
        exact &= wanted <= np.iinfo(dtype).max
    return cast, exact


class LabelIndex:
    """Positions of labels on one axis, built once per label array.

    Labels are searched with ``np.searchsorted``: directly when they are
    already strictly increasing (positions after a filter, most time and
    coordinate axes), otherwise through one stable sort made at build time.
    A missing label (NaN or NaT) is held apart, since it never sorts or
    compares equal; one such label can still be looked up.
    """

    __slots__ = ("kind", "dtype", "size", "unique", "_sorted", "_sorter", "_kept", "_missing", "__weakref__")

    def __init__(self, labels: np.ndarray, op: str = "index", dim: Hashable = None):
        kind, values = _kind(np.asarray(labels), op, dim)
        missing = _missing(values)
        kept = np.flatnonzero(~missing)
        present = values[kept] if missing.any() else values
        self.kind, self.dtype, self.size = kind, present.dtype, len(values)
        self._missing = np.flatnonzero(missing)
        self._kept = None if len(kept) == len(values) else kept
        try:
            increasing = bool(np.all(present[1:] > present[:-1]))
            if increasing:
                self._sorter, self._sorted = None, present
            else:
                # Order among equal labels is irrelevant (repeats cannot be looked up), so the
                # faster unstable sort is enough.
                self._sorter = np.argsort(present)
                self._sorted = present[self._sorter]
        except TypeError:
            raise CoreError(f"{op}: labels on {dim!r} cannot be ordered, so they cannot be looked up") from None
        repeated = not increasing and bool(np.any(self._sorted[1:] == self._sorted[:-1]))
        self.unique = not repeated and len(self._missing) <= 1

    def get_indexer(self, wanted: Any) -> np.ndarray:
        """The position of each wanted label, or -1 where it is absent."""
        wanted = np.atleast_1d(np.asarray(wanted))
        if wanted.ndim != 1:
            raise CoreError("look up labels as a one-dimensional list")
        result = np.full(len(wanted), -1, dtype=np.intp)
        if not len(wanted) or not self.size:
            return result
        if not self.unique:
            raise CoreError("labels repeat, so a label does not name one position")
        try:
            kind, wanted = _kind(wanted, "look up", None)
        except CoreError:
            return result  # a label of a different type is simply not present
        if self.kind == "time" and kind in ("text", "bytes"):
            try:
                wanted, kind = np.asarray(wanted, dtype="datetime64"), "time"
            except ValueError:
                return result
        if kind != self.kind:
            return result
        missing = _missing(wanted)
        if missing.any() and len(self._missing):
            result[missing] = self._missing[0]
        rest = np.flatnonzero(~missing)
        cast, exact = (wanted[rest], np.ones(len(rest), dtype=bool)) if wanted.dtype == self.dtype \
            else _exact_cast(wanted[rest], self.dtype)
        if not len(self._sorted):
            return result
        found = np.searchsorted(self._sorted, cast)
        clipped = np.minimum(found, len(self._sorted) - 1)
        hit = exact & (found < len(self._sorted)) & (self._sorted[clipped] == cast)
        positions = clipped if self._sorter is None else self._sorter[clipped]
        if self._kept is not None:
            positions = self._kept[positions]
        result[rest[hit]] = positions[hit]
        return result


_INDEXES: dict[int, tuple[Any, LabelIndex]] = {}


def label_index(labels: np.ndarray, op: str = "index", dim: Hashable = None) -> LabelIndex:
    """The index for a label array, built on first use and reused while the array lives.

    Only arrays that can never change are cached; others get a fresh index.
    """
    if not _never_changes(labels):
        return LabelIndex(labels, op, dim)
    key = id(labels)
    cached = _INDEXES.get(key)
    if cached is not None and cached[0]() is labels:
        return cached[1]
    index = LabelIndex(labels, op, dim)
    # The callback binds the cache itself: module globals are gone during interpreter shutdown.
    _INDEXES[key] = (weakref.ref(labels, lambda _, key=key, cache=_INDEXES: cache.pop(key, None)), index)
    return index


def _labels_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Same labels in the same order; NaN and NaT equal themselves here."""
    if len(a) != len(b):
        return False
    if a.dtype.kind in "fcmM" and b.dtype.kind in "fcmM":
        return bool(np.array_equal(a, b, equal_nan=True))
    try:
        return bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        return False


# ── Dataset: variables sharing one set of labels ────────────────────────────


class Dataset:
    """Variables with different axes; each axis has one size and one label array."""

    __slots__ = ("_vars", "_coords", "_sizes")

    def __init__(self, variables: Mapping[Hashable, Variable], coords: Mapping[Hashable, Any] | None = None):
        sizes: dict[Hashable, int] = {}
        for name, var in variables.items():
            for dim, size in var.sizes.items():
                if sizes.setdefault(dim, size) != size:
                    raise CoreError(f"{name}: axis {dim!r} has {size} positions, other variables have {sizes[dim]}")
        self._vars = MappingProxyType(dict(variables))
        self._coords = MappingProxyType(_checked_coords(coords or {}, sizes))
        self._sizes = MappingProxyType(sizes)

    variables = property(lambda self: self._vars)
    coords = property(lambda self: self._coords)
    sizes = property(lambda self: self._sizes)
    dims = property(lambda self: tuple(self._sizes))

    def __getitem__(self, name: Hashable) -> Labelled:
        if name not in self._vars:
            raise CoreError(f"no variable {name!r}; variables are {tuple(self._vars)}")
        var = self._vars[name]
        return Labelled(var, {d: self._coords[d] for d in var.dims if d in self._coords})

    def assign(self, **items: Labelled) -> Dataset:
        """Add or replace variables; their labels must equal the dataset's exactly."""
        variables = dict(self._vars)
        coords = dict(self._coords)
        for name, item in items.items():
            for dim in item.dims:
                theirs, ours = item.coords.get(dim), coords.get(dim)
                if dim in self._sizes and item.var.sizes[dim] != self._sizes[dim]:
                    raise CoreError(f"assign {name}: {dim!r} has {item.var.sizes[dim]} positions, the data has {self._sizes[dim]}")
                if (theirs is None) != (ours is None) and dim in self._sizes:
                    raise CoreError(f"assign {name}: {dim!r} is labelled on one side only")
                if theirs is not None and ours is not None and theirs is not ours and not np.array_equal(theirs, ours):
                    raise CoreError(f"assign {name}: labels on {dim!r} differ from the data's")
                if theirs is not None:
                    coords[dim] = theirs
            variables[name] = item.var
        used = {dim for var in variables.values() for dim in var.dims}
        return Dataset(variables, {dim: labels for dim, labels in coords.items() if dim in used})

    def isel(self, op: str = "isel", **indexers: Any) -> Dataset:
        """Positions, slices, or masks on axes, for every variable with them.

        An unlabelled axis gets its 1-based positions as labels first, so
        kept positions keep their identity (plan §5.3).
        """
        variables = dict(self._vars)
        coords = dict(self._coords)
        for dim, index in indexers.items():
            if dim not in self._sizes:
                raise CoreError(f"{op}: no axis {dim!r}; axes are {tuple(self._sizes)}")
            if not isinstance(index, slice):
                index = np.atleast_1d(index)
                if index.size == 0:
                    index = index.astype(np.intp)
                if index.ndim != 1:
                    raise CoreError(f"{op}: positions on {dim!r} must be one-dimensional")
            if getattr(index, "dtype", None) == bool:
                if len(index) != self._sizes[dim]:
                    raise CoreError(f"{op}: mask on {dim!r} has the wrong length")
                # Scan the mask once; every variable and the labels reuse the positions.
                index = np.flatnonzero(index)
            labels = coords.get(dim)
            cut = (np.arange(1, self._sizes[dim] + 1) if labels is None else labels)[index]
            coords[dim] = cut if isinstance(index, slice) and labels is not None else _freeze(cut)
            for name, var in variables.items():
                if dim in var.dims:
                    key = (slice(None),) * var.axis(dim, op) + (index,)
                    variables[name] = Variable(var.data[key], var.dims, var.attrs)
        return Dataset(variables, coords)


# ── Dispatcher ──────────────────────────────────────────────────────────────


def align(operands: Sequence[Any], op: str, expand: bool = True) -> tuple[list[Any], dict[Hashable, np.ndarray], list[Hashable], list[str]]:
    """Check shared axes and line every operand up on the union of axes.

    Returns the raw arrays (scalars pass through), the result's labels, its
    axis order (first appearance), and notes on any reordering. With
    *expand*, each array gets length-1 axes for the axes it lacks, ready for
    NumPy broadcasting; otherwise it keeps only its own axes, in union order.
    """
    dims: list[Hashable] = []
    sizes: dict[Hashable, int] = {}
    coords: dict[Hashable, np.ndarray] = {}
    owners: dict[Hashable, int] = {}
    notes: list[str] = []
    arrays = [item.var.data if isinstance(item, Labelled) else item for item in operands]
    for k, item in enumerate(operands):
        if not isinstance(item, Labelled):
            if np.ndim(item):
                raise CoreError(f"{op}: a bare array has no axis names; wrap it with labelled(..., dims=...)")
            continue
        for dim in item.dims:
            size = item.var.sizes[dim]
            labels = item.coords.get(dim)
            if dim not in sizes:
                dims.append(dim)
                sizes[dim] = size
                owners[dim] = k
                if labels is not None:
                    coords[dim] = labels
                continue
            first = coords.get(dim)
            if (first is None) != (labels is None):
                raise CoreError(f"{op}: {dim!r} is labelled on one operand and unlabelled on another; "
                                "pairing by position could match the wrong points")
            if first is None:
                if size != sizes[dim]:
                    raise CoreError(f"{op}: {dim!r} has {sizes[dim]} positions on one operand and {size} on another")
                continue
            if first is labels or _labels_equal(labels, first):
                continue
            ours, theirs = label_index(first, op, dim), label_index(labels, op, dim)
            if not ours.unique or not theirs.unique:
                raise CoreError(f"{op}: {dim!r} has repeated labels in different sequences; "
                                "matching their occurrences is ambiguous")
            order = theirs.get_indexer(first)
            if len(labels) != len(first) or (order < 0).any():
                only_first = int((order < 0).sum())
                only_this = int((ours.get_indexer(labels) < 0).sum())
                raise CoreError(f"{op}: labels on {dim!r} differ: {only_first} only in the first operand, "
                                f"{only_this} only in operand {k + 1}; join explicitly to combine them")
            arrays[k] = np.take(arrays[k], order, axis=item.var.axis(dim, op))
            notes.append(f"changed: {dim} reordered to match the first operand in {op}")
    lined_up = []
    for item, array in zip(operands, arrays):
        if not isinstance(item, Labelled):
            lined_up.append(array)
            continue
        present = [d for d in dims if d in item.dims]
        array = np.transpose(array, [item.dims.index(d) for d in present]) if list(item.dims) != present else array
        if expand:
            array = np.reshape(array, [sizes[d] if d in item.dims else 1 for d in dims])
        lined_up.append(array)
    return lined_up, coords, dims, notes


def _reorder(first: np.ndarray, other: np.ndarray) -> np.ndarray | None:
    """Positions in *other* that put its labels in *first*'s order, if they are the same set."""
    if len(first) != len(other):
        return None
    order = label_index(other, "align").get_indexer(first)
    return None if (order < 0).any() else order


def apply(ufunc: Callable, *operands: Any, op: str | None = None) -> Labelled:
    """Elementwise *ufunc* with broadcasting by name and label checks."""
    op = op or getattr(ufunc, "__name__", "apply")
    arrays, coords, dims, notes = align(operands, op)
    result = ufunc(*arrays)
    notes = tuple(n for item in operands if isinstance(item, Labelled) for n in item.notes) + tuple(notes)
    return Labelled(Variable(result, dims), coords, notes)


_REDUCERS = {
    "sum": (np.sum, np.nansum), "mean": (np.mean, np.nanmean), "min": (np.min, np.nanmin),
    "max": (np.max, np.nanmax), "median": (np.median, np.nanmedian), "std": (np.std, np.nanstd),
    "var": (np.var, np.nanvar), "any": (np.any, np.any), "all": (np.all, np.all),
}


def _reduce_data(data, op, axes, na_rm=False, min_count=None):
    """NumPy reductions with tidy3A's missing-value and dtype rules."""
    if op not in _REDUCERS:
        raise CoreError(f"no reduction {op!r}; reductions are {tuple(_REDUCERS)}")
    if not axes:
        return data
    if op in ("any", "all") or op == "sum" and min_count is not None:
        missing = np.isnan(data) if data.dtype.kind in "fc" else np.zeros(data.shape, dtype=bool)
    shape = tuple(n for i, n in enumerate(data.shape) if i not in axes)
    if op in ("min", "max") and any(data.shape[axis] == 0 for axis in axes):
        return np.full(shape, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if op in ("any", "all"):
            result = _REDUCERS[op][0](np.where(missing, op == "all", data), axis=axes)
            return result if na_rm or data.dtype.kind in "biu" else np.where(missing.any(axis=axes), np.nan, result)
        result = _REDUCERS[op][na_rm and data.dtype.kind in "fc"](data, axis=axes)
        if op == "sum" and min_count is not None:
            result = np.where((~missing).sum(axis=axes) >= min_count, result, np.nan)
    return result


def reduce(item: Labelled, op: str, over: Hashable | Sequence[Hashable], na_rm: bool = False,
           min_count: int | None = None) -> Labelled:
    """Reduce over named axes; those axes and their labels are dropped."""
    names = [over] if isinstance(over, (str, int)) or not isinstance(over, Sequence) else list(over)
    axes = tuple(item.var.axis(name, op) for name in names)
    result = _reduce_data(item.var.data, op, axes, na_rm, min_count)
    dims = [d for d in item.dims if d not in names]
    return Labelled(Variable(result, dims), {d: c for d, c in item.coords.items() if d in dims}, item.notes)


def sel(item: Labelled, op: str = "sel", **wanted: Any) -> Labelled:
    """Positions picked in requested order; nonempty requests need unique source labels."""
    data, coords = item.var.data, dict(item.coords)
    for dim, labels in wanted.items():
        axis = item.var.axis(dim, op)
        if dim not in coords:
            raise CoreError(f"{op}: {dim!r} has no labels; select positions with isel")
        have = coords[dim]
        labels = np.atleast_1d(np.asarray(labels))
        index = label_index(have, op, dim)
        if labels.size and not index.unique:
            raise CoreError(f"{op}: {dim!r} has repeated labels; select positions with isel")
        found = index.get_indexer(labels)
        if (found < 0).any():
            raise CoreError(f"{op}: {dim!r} has no label {labels[found < 0][0]!r}")
        positions = found.tolist()
        coords[dim] = _freeze(have[positions])
        start = positions[0] if positions else 0
        if positions == list(range(start, start + len(positions))):
            # Consecutive positions are a slice: a view, not a copy.
            data = data[(slice(None),) * axis + (slice(start, start + len(positions)),)]
        else:
            data = np.take(data, positions, axis=axis)
    return Labelled(Variable(data, item.dims, item.var.attrs), coords, item.notes)


def drop_axis(item: Labelled, dim: Hashable, op: str = "drop_axis") -> Labelled:
    """Remove a length-1 axis, as a bare field label drops the field axis (plan §5.2)."""
    axis = item.var.axis(dim, op)
    if item.var.sizes[dim] != 1:
        raise CoreError(f"{op}: {dim!r} has {item.var.sizes[dim]} positions; only a length-1 axis can be dropped")
    dims = tuple(d for d in item.dims if d != dim)
    coords = {d: c for d, c in item.coords.items() if d != dim}
    return Labelled(Variable(np.squeeze(item.var.data, axis=axis), dims), coords, item.notes)


def transpose(item: Labelled, *dims: Hashable) -> Labelled:
    if sorted(map(str, dims)) != sorted(map(str, item.dims)):
        raise CoreError(f"transpose: {dims} must name each of {item.dims} once")
    order = [item.var.axis(d, "transpose") for d in dims]
    return Labelled(Variable(np.transpose(item.var.data, order), dims, item.var.attrs), item.coords, item.notes)


def concat(items: Sequence[Labelled], dim: Hashable, labels: Sequence[Any] | None = None) -> Labelled:
    """Stack items along a new axis *dim* (``pack``); the rest must align."""
    if any(dim in item.dims for item in items):
        raise CoreError(f"concat: {dim!r} is already an axis")
    arrays, coords, dims, notes = align(list(items), "concat")
    data = np.stack([np.broadcast_to(a, [max(x.shape[i] for x in arrays) for i in range(len(dims))]) for a in arrays])
    if labels is not None:
        coords = {dim: np.asarray(labels), **coords}
    return Labelled(Variable(data, (dim, *dims)), _checked_coords(coords, dict(zip((dim, *dims), data.shape))), tuple(notes))


def dot(a: Labelled, b: Labelled, over: Hashable | Sequence[Hashable]) -> Labelled:
    """Multiply and sum over *over*; other axes are kept, by name (plan §6.2)."""
    names = [over] if isinstance(over, str) else list(over)
    for name in names:
        a.var.axis(name, "dot")
        b.var.axis(name, "dot")
    # Check and reorder shared axes as elementwise operations do; each array
    # keeps only its own axes, so nothing is expanded in memory.
    (left, right), coords, dims, notes = align([a, b], "dot", expand=False)
    letters = {d: chr(ord("a") + k) for k, d in enumerate(dims)}
    kept = [d for d in dims if d not in names]
    spec = ",".join("".join(letters[d] for d in dims if d in item.dims) for item in (a, b))
    result = np.einsum(f"{spec}->{''.join(letters[d] for d in kept)}", left, right, optimize=True)
    # einsum may return a transposed view; later steps run several times
    # faster on contiguous memory, so pay for one copy here.
    if np.ndim(result):  # ascontiguousarray would turn a scalar into a length-1 array.
        result = np.ascontiguousarray(result)
    return Labelled(Variable(result, kept), {d: c for d, c in coords.items() if d in kept},
                    a.notes + b.notes + tuple(notes))


def rename_axes(item: Labelled, **names: Hashable) -> Labelled:
    """Rename axes; labels move with them (plan §6.2)."""
    dims = tuple(names.get(d, d) for d in item.dims)
    coords = {names.get(d, d): c for d, c in item.coords.items()}
    return Labelled(Variable(item.var.data, dims, item.var.attrs), coords, item.notes)


# ── Grouping by value ───────────────────────────────────────────────────────


def _factorize(raw: np.ndarray, use_pandas: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique values and each position's code, as ``np.unique`` gives them.

    NumPy finds groups by sorting every value; pandas hashes them, which is
    far faster for many positions. pandas is used when installed, as tidy3A
    does, and gives values such as NaT no code, so those fall back to NumPy.
    """
    if use_pandas:
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            codes, labels = pd.factorize(raw, sort=True)
            if not (codes < 0).any():
                return np.asarray(labels), codes
    labels, codes = np.unique(raw, return_inverse=True)
    return labels, codes.reshape(-1)


def readonly(values, *, copy=False):
    """Freeze owned storage and expose a read-only view; copy unowned input first."""
    values = np.array(values, copy=True) if copy else np.asarray(values)
    values.flags.writeable = False
    view = values.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class GroupIndex:
    """Reusable grouping arrays: no Python object or list per group.

    Equal-sized groups are reduced in batches. Each group's original order
    and contiguous layout are retained, including when its axis is not first.
    """

    axis: Hashable
    name: Hashable
    labels: np.ndarray
    codes: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    order: np.ndarray | None
    by_size: np.ndarray
    sizes: np.ndarray
    size_offsets: np.ndarray
    coords: Mapping

    @classmethod
    def build(cls, key: Labelled, name: Hashable, *, use_pandas=False):
        if key.ndim != 1:
            raise CoreError(f"group by {name!r}: the key varies along {key.dims}, not one axis")
        raw = key.values
        if raw.dtype.kind in "fc" and np.isnan(raw).any():
            raise CoreError(f"group by {name!r}: {int(np.isnan(raw).sum())} positions have no key")
        labels, codes = _factorize(raw, use_pandas=use_pandas)
        counts = np.bincount(codes, minlength=len(labels))
        starts = np.cumsum(counts) - counts
        narrow = np.min_scalar_type(max(len(labels) - 1, 0))
        order = None if np.all(codes[1:] >= codes[:-1]) else np.argsort(codes.astype(narrow), kind="stable")
        sizes, frequencies = np.unique(counts, return_counts=True)
        offsets = np.concatenate(([0], np.cumsum(frequencies)))
        by_size = np.argsort(counts, kind="stable")
        coords = {d: readonly(c, copy=True) if c.flags.writeable else c for d, c in key.coords.items()}
        return cls(key.dims[0], name, readonly(labels), readonly(codes), readonly(counts),
                   readonly(starts), None if order is None else readonly(order), readonly(by_size),
                   readonly(sizes), readonly(offsets), MappingProxyType(coords))

    @property
    def nbytes(self):
        arrays = (self.labels, self.codes, self.counts, self.starts, self.order,
                  self.by_size, self.sizes, self.size_offsets)
        return sum(array.nbytes for array in arrays if array is not None)

    def reduce(self, item, op, na_rm=False, min_count=None, over=()):
        if self.axis not in item.dims:
            raise CoreError(f"group by {self.name!r}: the key's axis {self.axis!r} is not an axis of the data")
        target = labelled(np.broadcast_to(0, (len(self.codes),)), (self.axis,), self.coords)
        (_, raw), coords, dims, notes = align([target, item], "group_by", expand=False)
        raw = np.transpose(raw, [dims.index(d) for d in item.dims])
        axis = item.var.axis(self.axis, "group_by")
        axes = (axis, *(item.var.axis(d, op) for d in over if d != self.axis))
        kept = [d for i, d in enumerate(item.dims) if i not in axes]
        kept_shape = tuple(item.sizes[d] for d in kept)
        # Reduction dtype follows NumPy, including integer promotion and complex variance.
        empty = any(raw.shape[a] == 0 for a in axes)
        sample = _reduce_data(np.zeros((0 if empty else 1,), dtype=item.dtype), op, (0,), na_rm, min_count)
        output = np.empty((len(self.labels), *kept_shape), dtype=np.asarray(sample).dtype)
        ordered = raw if self.order is None else np.take(raw, self.order, axis=axis)
        per_position = max(1, int(np.prod([n for i, n in enumerate(raw.shape) if i != axis])))
        for size, lo, hi in zip(self.sizes, self.size_offsets[:-1], self.size_offsets[1:]):
            groups = self.by_size[lo:hi]
            # Bound temporary gathers to roughly one million elements; one large
            # group still needs its own complete reduction, as ordinary NumPy does.
            batch_size = max(1, 1_048_576 // (int(size) * per_position))
            for offset in range(0, len(groups), batch_size):
                group = groups[offset:offset + batch_size]
                positions = self.starts[group, None] + np.arange(size)
                batch = np.take(ordered, positions, axis=axis)
                batch = np.ascontiguousarray(np.moveaxis(batch, axis, 0))
                output[group] = _reduce_data(batch, op, tuple(a + 1 for a in axes), na_rm, min_count)
        result_coords = {self.name: self.labels, **{d: c for d, c in coords.items() if d in kept}}
        return Labelled(Variable(output, (self.name, *kept)), result_coords, item.notes + tuple(notes))

    def broadcast(self, item):
        """Put one result per group back on the original observation positions."""
        axis = item.var.axis(self.name, "broadcast groups")
        data = np.take(item.var.data, self.codes, axis=axis)
        dims = tuple(self.axis if d == self.name else d for d in item.dims)
        coords = {d: c for d, c in item.coords.items() if d != self.name}
        coords.update(self.coords)
        return Labelled(Variable(data, dims), coords, item.notes)


def group_reduce(item: Labelled, key: Labelled, op: str, name: Hashable, na_rm: bool = False,
                 *, use_pandas: bool = True) -> Labelled:
    """One-shot grouping; reuse GroupIndex directly for several reductions."""
    # A one-shot reduction retains the data's original observation order,
    # including when the grouping key arrives with reordered coordinate labels.
    (_, raw), coords, _, notes = align([item, key], "group_by", expand=False)
    key = Labelled(Variable(raw, key.dims), {d: coords[d] for d in key.dims if d in coords})
    result = GroupIndex.build(key, name, use_pandas=use_pandas).reduce(item, op, na_rm)
    return Labelled(result.var, result.coords, result.notes + tuple(notes))
