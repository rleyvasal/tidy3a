"""The NumPy backend (experimental; plan milestone 9, step 3).

Collections hold NumPy arrays with named axes and one label array per
labelled axis, shared by every variable on that axis. Everything a verb
needs (label lookups, selection, arithmetic under the alignment rules,
reductions, grouping, joins) is implemented here with NumPy; the rules and
messages stay in the shared modules. It holds eager, in-memory data only:
lazy, chunked, and memory-mapped inputs stay on the xarray backend.

xarray and pandas are imported only where needed: xarray to convert to and
from xarray objects, and pandas, when installed, to find groups by hashing
and to list labels for messages exactly as the xarray backend does.

The backend owns its labels: labels from outside are copied once into a
read-only ``_Labels`` array, which nothing else can unlock, so each label
array's index (``LabelIndex``) is built on first use and kept on the array,
released with it. Labels compare under one exact rule (``tidy3a.labels``).
Variable data is not copied or frozen: as with the xarray backend, wrapping
in-memory data shares it.
"""

from __future__ import annotations

import builtins
import operator
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from tidy3a import grouped
from tidy3a import labels as exact
from tidy3a.backend import is_xarray, kept_positions, memory_mapped
from tidy3a.errors import TidyError
from tidy3a.labels import LabelError, exact_cast
from tidy3a.labels import kind as _kind
from tidy3a.labels import missing as _missing

CoreError = LabelError  # the label index below raises it under this name

# Up to this many groups, scanning the axis once per group is cheaper than sorting it.
_FEW_GROUPS = 8

# Python's operators on xarray arrays run without NumPy's floating-point warnings
# (division by zero, overflow); NumPy functions such as sqrt still warn. The same here.
_QUIET = {
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod,
    operator.pow, operator.and_, operator.or_, operator.xor, operator.eq, operator.ne, operator.lt,
    operator.le, operator.gt, operator.ge, operator.neg, operator.pos, operator.invert, operator.abs,
}


# ── Arrays and collections ──────────────────────────────────────────────────


class NumpyArray:
    """A NumPy array with one name per axis, labels for its labelled axes, attributes, and a name."""

    __slots__ = ("data", "dims", "coords", "attrs", "name")

    def __init__(self, data: Any, dims: Sequence[Hashable] = (), coords: Mapping[Hashable, Any] | None = None,
                 attrs: Mapping[Hashable, Any] | None = None, name: Hashable | None = None):
        data = data if isinstance(data, np.ndarray) else np.asarray(data)
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != data.ndim:
            raise TidyError(f"{len(dims)} axis names {dims} for an array of rank {data.ndim}")
        if len(set(dims)) != len(dims):
            raise TidyError(f"axis names must be unique, got {dims}")
        self.data, self.dims, self.name = data, dims, name
        self.coords = _checked_coords(coords or {}, dict(zip(dims, data.shape)))
        self.attrs = dict(attrs or {})

    @classmethod
    def _make(cls, data: np.ndarray, dims: tuple, coords: dict, attrs: dict | None = None,
              name: Hashable | None = None) -> NumpyArray:
        """Build without checks, from parts this module already validated and froze."""
        array = object.__new__(cls)
        array.data, array.dims, array.coords = data, dims, coords
        array.attrs, array.name = attrs if attrs is not None else {}, name
        return array

    @property
    def sizes(self) -> dict[Hashable, int]:
        return dict(zip(self.dims, self.data.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def values(self) -> np.ndarray:
        return self.data

    def axis(self, dim: Hashable) -> int:
        return self.dims.index(dim)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {n}" for d, n in self.sizes.items())
        return f"<NumpyArray {self.name or ''}({sizes}) {self.dtype}>"


class NumpyData:
    """Variables with different axes; each axis has one size and at most one label array."""

    __slots__ = ("variables", "coords", "coord_attrs", "attrs", "sizes")

    def __init__(self, variables: Mapping[Hashable, NumpyArray], coords: Mapping[Hashable, Any] | None = None,
                 attrs: Mapping[Hashable, Any] | None = None, coord_attrs: Mapping[Hashable, Any] | None = None):
        sizes: dict[Hashable, int] = {}
        for name, value in variables.items():
            for dim, size in zip(value.dims, value.data.shape):
                if sizes.setdefault(dim, size) != size:
                    raise TidyError(f"{name}: axis {dim} has {size} positions, other variables have {sizes[dim]}")
        coords = dict(coords or {})
        for dim, labels in coords.items():
            sizes.setdefault(dim, len(labels))
        self.variables = {name: NumpyArray._make(v.data, v.dims, {}, v.attrs, name) for name, v in variables.items()}
        self.coords = _checked_coords(coords, sizes)
        self.coord_attrs = {d: dict(a) for d, a in (coord_attrs or {}).items() if d in self.coords}
        self.attrs = dict(attrs or {})
        self.sizes = sizes

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self.sizes)

    def array(self, name: Hashable) -> NumpyArray:
        value = self.variables[name]
        return NumpyArray._make(value.data, value.dims, {d: self.coords[d] for d in value.dims if d in self.coords},
                                value.attrs, name)

    def evolve(self, variables: Mapping | None = None, coords: Mapping | None = None, **changes: Any) -> NumpyData:
        """A copy with parts replaced; labels for axes no variable uses are dropped only if asked."""
        return NumpyData(
            self.variables if variables is None else variables,
            self.coords if coords is None else coords,
            changes.get("attrs", self.attrs), changes.get("coord_attrs", self.coord_attrs),
        )


def _new_labels(labels: Any) -> np.ndarray:
    """Labels made here: text is stored as objects, as the xarray backend does."""
    values = np.asarray(labels)
    if values.dtype.kind in "UO":
        values = values.astype(object)
    return _freeze(values)


def _checked_coords(coords: Mapping[Hashable, Any], sizes: Mapping[Hashable, int]) -> dict:
    out = {}
    for dim, labels in coords.items():
        labels = _frozen_labels(labels)
        if dim not in sizes:
            raise TidyError(f"labels for {dim}, which is not an axis; axes are {tuple(sizes)}")
        if labels.ndim != 1 or len(labels) != sizes[dim]:
            raise TidyError(f"{dim} has {sizes[dim]} positions but {labels.size} labels")
        out[dim] = labels
    return out


class _Labels(np.ndarray):
    """Labels this backend owns: read-only, never shared writeable, with their index kept on them.

    Views and slices are labels too, each with its own index. Arithmetic on
    labels gives plain arrays, so labels never leak into variable data.
    """

    _index: LabelIndex | None = None

    def __array_finalize__(self, obj: Any) -> None:
        self._index = None

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        inputs = tuple(x.view(np.ndarray) if isinstance(x, _Labels) else x for x in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)


def _owned(labels: Any) -> bool:
    return isinstance(labels, _Labels) and not labels.flags.writeable


def _frozen_labels(labels: Any) -> _Labels:
    """Labels that can never change: this backend's own are kept, others copied once."""
    if _owned(labels):
        return labels
    return _freeze(np.array(labels, copy=True))


def _freeze(array: np.ndarray) -> _Labels:
    """Freeze a label array this module just created and owns alone."""
    if _owned(array):
        return array
    labels = array.view(_Labels)
    labels.flags.writeable = False
    return labels


def _plain(labels: np.ndarray) -> np.ndarray:
    """Labels as a plain NumPy array, for xarray and for callers."""
    return labels.view(np.ndarray)


# ── Label index ─────────────────────────────────────────────────────────────

class LabelIndex:
    """Positions of labels on one axis, built once per label array.

    Labels are searched with ``np.searchsorted``: directly when they are
    already strictly increasing (positions after a filter, most time and
    coordinate axes), otherwise through one stable sort made at build time.
    A missing label (NaN or NaT) is held apart, since it never sorts or
    compares equal; one such label can still be looked up.
    """

    __slots__ = ("kind", "dtype", "size", "unique", "_sorted", "_sorter", "_kept", "_missing")

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
            else exact_cast(wanted[rest], self.dtype)
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


def label_index(labels: np.ndarray, op: str = "index", dim: Hashable = None) -> LabelIndex:
    """The index for a label array: built on first use and kept on labels this backend owns.

    Other arrays, which could change, get a fresh index each time.
    """
    if not _owned(labels):
        return LabelIndex(labels, op, dim)
    if labels._index is None:
        labels._index = LabelIndex(labels, op, dim)
    return labels._index


# ── Helpers ─────────────────────────────────────────────────────────────────


def _broadcast(operands: Sequence[Any]) -> tuple[list[Any], tuple[Hashable, ...], dict]:
    """Raw values lined up on the union of named axes (first appearance), and the union's labels.

    Callers align labels first (``tidy3a.align``); sizes are checked again here.
    """
    dims: list[Hashable] = []
    sizes: dict[Hashable, int] = {}
    coords: dict[Hashable, np.ndarray] = {}
    for item in operands:
        if isinstance(item, NumpyArray):
            for dim, size in zip(item.dims, item.data.shape):
                if dim not in sizes:
                    dims.append(dim)
                    sizes[dim] = size
                elif sizes[dim] != size:
                    raise TidyError(f"{dim} has {sizes[dim]} positions on one operand and {size} on another")
                if dim in item.coords and dim not in coords:
                    coords[dim] = item.coords[dim]
    raw = []
    for item in operands:
        if not isinstance(item, NumpyArray):
            raw.append(item)
            continue
        present = [d for d in dims if d in item.dims]
        data = item.data
        if list(item.dims) != present:
            data = np.transpose(data, [item.dims.index(d) for d in present])
        if len(present) != len(dims):
            data = np.reshape(data, [sizes[d] if d in item.dims else 1 for d in dims])
        raw.append(data)
    return raw, tuple(dims), coords


def _take(value: NumpyArray, dim: Hashable, index: Any) -> NumpyArray:
    """Positions, a slice, or a mask on one axis; an integer removes the axis."""
    axis = value.axis(dim)
    data = value.data[(builtins.slice(None),) * axis + (index,)]
    coords = dict(value.coords)
    if isinstance(index, (int, np.integer)):
        coords.pop(dim, None)
        dims = value.dims[:axis] + value.dims[axis + 1:]
    else:
        dims = value.dims
        if dim in coords:
            cut = coords[dim][index]
            coords[dim] = cut if isinstance(index, builtins.slice) else _freeze(cut)
    return NumpyArray._make(data, dims, coords, value.attrs, value.name)


def _positions_of(labels: np.ndarray, wanted: Any, dim: Hashable) -> Any:
    """An integer for one label, positions for a list, as xarray's ``sel`` reads them."""
    index = label_index(labels, "sel", dim)
    many = not np.isscalar(wanted) and not (isinstance(wanted, np.ndarray) and wanted.ndim == 0)
    found = index.get_indexer(np.atleast_1d(np.asarray(wanted, dtype=object if _is_text(wanted) else None)))
    if (found < 0).any():
        missing = np.atleast_1d(np.asarray(wanted, dtype=object))[found < 0][0]
        raise LabelError(f"{dim} has no label {missing!r}")
    return found if many else int(found[0])


def _is_text(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value))


_REDUCERS = {
    "sum": (np.sum, np.nansum), "mean": (np.mean, np.nanmean), "min": (np.min, np.nanmin),
    "max": (np.max, np.nanmax), "median": (np.median, np.nanmedian), "std": (np.std, np.nanstd),
    "var": (np.var, np.nanvar),
}


def _reduce_data(data: np.ndarray, op: str, axes: int | tuple[int, ...], na_rm: bool, min_count: int | None) -> Any:
    """NumPy reductions with tidy3A's missing-value rules, as the xarray backend applies them."""
    kind = data.dtype.kind
    numbers = (axes,) if isinstance(axes, int) else axes
    # Only some reductions need to know where values are missing; finding out reads the whole array.
    needs_missing = op in ("any", "all", "count") or (op == "sum" and min_count is not None)
    missing = None if not needs_missing else np.isnan(data) if kind in "fc" else np.isnat(data) if kind in "mM" else None
    if op in ("min", "max") and builtins.any(data.shape[axis] == 0 for axis in numbers):
        shape = tuple(n for i, n in enumerate(data.shape) if i not in numbers)
        return np.full(shape, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if op in ("any", "all"):
            filled = data if missing is None else np.where(missing, op == "all", data)
            result = getattr(np, op)(filled, axis=axes)
            if na_rm or kind in "biu" or missing is None:
                return result
            return np.where(~missing.any(axis=axes), result, np.nan)  # missing wherever a value was
        if op == "count":
            present = ~missing if missing is not None else np.ones(data.shape, dtype=bool)
            return present.sum(axis=axes)
        skip = na_rm and kind in "fcmM"
        result = _REDUCERS[op][1 if skip else 0](data, axis=axes)
        if op == "sum" and min_count is not None:
            present = ~missing if missing is not None else np.ones(data.shape, dtype=bool)
            result = np.where(present.sum(axis=axes) >= min_count, result, np.nan)
        return result


def _fill_for(dtype: np.dtype) -> tuple[np.dtype, Any]:
    """The type and missing value a reindexed variable needs, as xarray promotes them."""
    if dtype.kind in "iu":
        # As xarray promotes: integers of up to 16 bits to float32, wider ones to float64.
        return np.dtype(np.float32 if dtype.itemsize <= 2 else np.float64), np.nan
    if dtype.kind == "b":
        return np.dtype(np.float64), np.nan
    if dtype.kind == "f" or dtype.kind == "c":
        return dtype, np.nan
    if dtype.kind in "mM":
        return dtype, np.datetime64("NaT") if dtype.kind == "M" else np.timedelta64("NaT")
    return np.dtype(object), np.nan


def _reindex(value: NumpyArray, dim: Hashable, positions: np.ndarray, labels: np.ndarray) -> NumpyArray:
    """*value* on new labels for *dim*: position -1 becomes missing."""
    axis = value.axis(dim)
    missing = positions < 0
    if not value.data.shape[axis]:  # nothing to take from: every position is missing
        dtype, fill = _fill_for(value.data.dtype) if len(positions) else (value.data.dtype, None)
        shape = value.data.shape[:axis] + (len(positions),) + value.data.shape[axis + 1:]
        taken = np.full(shape, fill, dtype=dtype) if len(positions) else np.empty(shape, dtype=dtype)
        coords = dict(value.coords)
        coords[dim] = labels
        return NumpyArray._make(taken, value.dims, coords, value.attrs, value.name)
    taken = np.take(value.data, np.where(missing, 0, positions), axis=axis)
    if missing.any():
        dtype, fill = _fill_for(value.data.dtype)
        taken = taken.astype(dtype, copy=True) if taken.dtype != dtype else taken.copy()
        index = (builtins.slice(None),) * axis + (missing,)
        taken[index] = fill
    coords = dict(value.coords)
    coords[dim] = labels
    return NumpyArray._make(taken, value.dims, coords, value.attrs, value.name)


def _combine_attrs(*mappings: Mapping) -> dict:
    """xarray's ``drop_conflicts``: keep keys whose values agree wherever they appear."""
    result: dict = {}
    dropped: set = set()
    for mapping in mappings:
        for key, value in mapping.items():
            if key in dropped:
                continue
            if key in result and not _same(result[key], value):
                dropped.add(key)
                del result[key]
            elif key not in result:
                result[key] = value
    return result


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(np.array_equal(a, b)) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else bool(a == b)
    except Exception:  # noqa: BLE001 - values that cannot be compared count as different
        return False


# ── The backend ─────────────────────────────────────────────────────────────


class NumpyBackend:
    """Collections are ``NumpyData`` objects and arrays are ``NumpyArray`` objects."""

    name = "numpy"

    # ── Conversion ──────────────────────────────────────────────────────────

    def unsupported(self, data: Any) -> str | None:
        """Why an xarray object cannot be held by this backend, or None if it can."""
        import xarray as xr

        dataset = data.to_dataset(name=data.name if data.name is not None else "values") \
            if isinstance(data, xr.DataArray) else data
        for name, value in dataset.data_vars.items():
            if not value.variable._in_memory or not isinstance(value.data, np.ndarray) \
                    or memory_mapped(value.data):
                return f"{name} is lazy, chunked, or memory-mapped"
        for name in dataset.coords:
            if name not in dataset.indexes or dataset[name].dims != (name,):
                return f"{name} is a coordinate that is not one axis's labels"
            if type(dataset.indexes[name]).__name__ == "MultiIndex":
                return f"{name} has multi-level labels"
            if getattr(dataset.indexes[name], "tz", None) is not None:
                return f"{name} has time labels with a time zone"
        return None

    def wrap(self, data: Any) -> tuple[NumpyData, str]:
        import xarray as xr

        if isinstance(data, xr.DataArray):
            name = data.name if data.name is not None else "values"
            return self.from_xarray(data.to_dataset(name=name)), "xarray DataArray"
        return self.from_xarray(data), "xarray Dataset"

    def from_xarray(self, dataset: Any) -> NumpyData:
        variables = {name: NumpyArray._make(np.asarray(value.data), tuple(value.dims), {}, dict(value.attrs), name)
                     for name, value in dataset.data_vars.items()}
        coords = {name: dataset[name].values for name in dataset.coords}
        coord_attrs = {name: dict(dataset[name].attrs) for name in dataset.coords}
        return NumpyData(variables, coords, dict(dataset.attrs), coord_attrs)

    def to_xarray(self, data: NumpyData) -> Any:
        try:
            import xarray as xr
        except ImportError:
            raise TidyError(
                "collect() and to_xarray() give an xarray Dataset, and xarray is not installed.\n"
                "  For NumPy arrays: to_dict(), or pull(...) for one variable or axis\n"
                "  For an xarray Dataset: pip install 'tidy3a[xarray]'"
            ) from None

        coords = {d: (d, _plain(labels), dict(data.coord_attrs.get(d, {}))) for d, labels in data.coords.items()}
        return xr.Dataset({n: (v.dims, v.data, dict(v.attrs)) for n, v in data.variables.items()},
                          coords=coords, attrs=dict(data.attrs))

    def collect(self, data: NumpyData) -> Any:
        return self.to_xarray(data)

    def to_dict(self, data: NumpyData) -> dict:
        """As xarray's ``Dataset.to_dict(data="array")`` gives it, without xarray."""
        return {
            "coords": {d: {"dims": (d,), "attrs": dict(data.coord_attrs.get(d, {})), "data": _plain(labels)}
                       for d, labels in data.coords.items()},
            "attrs": dict(data.attrs),
            "dims": dict(data.sizes),
            "data_vars": {n: {"dims": v.dims, "attrs": dict(v.attrs), "data": v.data} for n, v in data.variables.items()},
        }

    # ── Readers: built directly, without an xarray dataset in between ───────

    def from_records(self, variables: Mapping[Hashable, tuple], coords: Mapping[Hashable, np.ndarray]) -> NumpyData:
        arrays = {name: NumpyArray._make(np.asarray(values), (dims,) if isinstance(dims, str) else tuple(dims), {},
                                         {}, name) for name, (dims, values) in variables.items()}
        return NumpyData(arrays, coords)

    def from_array(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   name: Hashable) -> NumpyData:
        return NumpyData({name: NumpyArray._make(values, tuple(dims), {}, {}, name)}, coords)

    def from_split(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   axis: Hashable) -> NumpyData:
        """One variable per label on *axis*, each a view of *values*, as xarray's split gives."""
        number = list(dims).index(axis)
        kept = tuple(d for d in dims if d != axis)
        arrays = {}
        for position, label in enumerate(coords[axis].tolist()):
            name = str(label)
            arrays[name] = NumpyArray._make(values[(builtins.slice(None),) * number + (position,)], kept, {}, {}, name)
        return NumpyData(arrays, {d: c for d, c in coords.items() if d != axis})

    def outside(self, value: Any) -> Any:
        """An array from the calling code (an xarray object) as this backend's array."""
        if not is_xarray(value):
            return value
        if not value.variable._in_memory or not isinstance(value.data, np.ndarray) or memory_mapped(value.data):
            name = f"on {', '.join(map(str, value.dims))}" if value.name is None else repr(value.name)
            raise TidyError(
                f"outside array {name} is lazy, chunked, or memory-mapped, and "
                'backend="numpy" holds eager, in-memory data only.\n'
                "  Load it into memory first (.load()), or use the xarray backend"
            )
        coords = {d: value[d].values for d in value.dims if d in value.indexes}
        return NumpyArray(np.asarray(value.values), value.dims, coords, dict(value.attrs), value.name)

    # ── Collections ─────────────────────────────────────────────────────────

    def variable_names(self, data: NumpyData) -> list[Hashable]:
        return list(data.variables)

    def has_variable(self, data: NumpyData, name: Hashable) -> bool:
        return name in data.variables

    def variable(self, data: NumpyData, name: Hashable) -> NumpyArray:
        return data.array(name)

    def variables(self, data: NumpyData) -> dict[Hashable, NumpyArray]:
        return {name: data.array(name) for name in data.variables}

    def sizes(self, obj: Any) -> Mapping[Hashable, int]:
        return obj.sizes

    def axes(self, obj: Any) -> tuple[Hashable, ...]:
        return tuple(obj.sizes)

    def has_axis(self, obj: Any, dim: Hashable) -> bool:
        try:
            return dim in obj.sizes
        except TypeError:  # an unhashable name is not an axis
            return False

    def has_coordinate(self, data: NumpyData, name: Hashable) -> bool:
        return name in data.coords

    def attrs(self, obj: Any) -> Mapping[Hashable, Any]:
        return obj.attrs

    def coordinate_attrs(self, data: NumpyData, name: Hashable) -> Mapping[Hashable, Any]:
        return data.coord_attrs.get(name, {})

    def coordinate_names(self, data: NumpyData) -> Any:
        return data.coords.keys()

    # ── Labels ──────────────────────────────────────────────────────────────

    def has_labels(self, obj: Any, dim: Hashable) -> bool:
        try:
            return dim in obj.coords
        except TypeError:
            return False

    def labelled_axes(self, obj: Any) -> list[Hashable]:
        return list(obj.coords)

    def label_map(self, obj: Any) -> Mapping[Hashable, Any]:
        return {dim: self.label_list(obj, dim) for dim in obj.coords}

    def label_values(self, obj: Any, dim: Hashable) -> np.ndarray:
        return _plain(obj.coords[dim])

    def label_list(self, obj: Any, dim: Hashable) -> list[Any]:
        labels = obj.coords[dim]
        try:
            import pandas as pd
        except ImportError:  # without pandas, NumPy's own scalars
            return labels.tolist()
        return list(pd.Index(labels))

    def has_label(self, obj: Any, dim: Hashable, label: Any) -> bool:
        labels = obj.coords[dim]
        index = label_index(labels, "look up", dim)
        if not index.unique:
            return bool(np.any(labels == label))
        try:
            return bool(index.get_indexer(np.array([label], dtype=object if isinstance(label, str) else None))[0] >= 0)
        except (TypeError, ValueError):
            return False

    def label_count(self, obj: Any, dim: Hashable) -> int:
        return len(obj.coords[dim])

    def labels_unique(self, obj: Any, dim: Hashable) -> bool:
        return label_index(obj.coords[dim], "check", dim).unique

    def label_position(self, obj: Any, dim: Hashable, label: Any) -> int:
        return _positions_of(obj.coords[dim], label, dim)

    def compare_labels(self, left: Any, right: Any, dim: Hashable) -> tuple[str, int, int]:
        return exact.compare(left.coords[dim], right.coords[dim], "align", dim)

    def labels_strict_subset(self, part: Any, whole: Any, dim: Hashable) -> bool:
        return exact.strict_subset(part.coords[dim], whole.coords[dim], "align", dim)

    def reorder_like(self, value: NumpyArray, dim: Hashable, reference: Any) -> NumpyArray:
        order = label_index(value.coords[dim], "align", dim).get_indexer(reference.coords[dim])
        taken = _take(value, dim, order)
        taken.coords[dim] = reference.coords[dim]
        return taken

    def axis_labels(self, data: NumpyData, axis: Hashable) -> NumpyArray:
        if axis in data.coords:
            return NumpyArray._make(_plain(data.coords[axis]), (axis,), {axis: data.coords[axis]},
                                    dict(data.coord_attrs.get(axis, {})), axis)
        return NumpyArray._make(np.arange(1, data.sizes[axis] + 1), (axis,), {}, {}, axis)

    def axis_coordinate(self, data: NumpyData, axis: Hashable) -> NumpyArray:
        return self.axis_labels(data, axis)

    def keep(self, data: NumpyData, axis: Hashable, index: Any) -> NumpyData:
        """Positions on *axis*, as ``XarrayBackend.keep``: only the kept positions are built."""
        if axis in data.coords:
            return self.isel(data, {axis: index})
        index, kept = kept_positions(index, data.sizes[axis])
        out = self.isel(data, {axis: index})
        np.add(kept, 1, out=kept)  # the indexer has been used; it becomes the labels
        coords = dict(out.coords)
        coords[axis] = _freeze(kept)
        return out.evolve(coords=coords)

    # ── Arrays ──────────────────────────────────────────────────────────────

    def is_array(self, value: Any) -> bool:
        return isinstance(value, NumpyArray)

    def is_lazy(self, value: Any) -> bool:
        return False

    def array(self, values: Any, dims: Sequence[Hashable] = (), coords: Mapping | None = None,
              name: Hashable | None = None) -> NumpyArray:
        if coords:
            coords = {d: _new_labels(c) for d, c in coords.items()}
        return NumpyArray(values, dims, coords, name=name)

    def codes_like(self, codes: np.ndarray, values: NumpyArray) -> NumpyArray:
        return NumpyArray._make(np.asarray(codes), values.dims[:1], dict(values.coords))

    def named(self, value: NumpyArray, name: Hashable) -> NumpyArray:
        return NumpyArray._make(value.data, value.dims, value.coords, value.attrs, name)

    def isel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        if isinstance(obj, NumpyArray):
            for dim, index in indexers.items():
                obj = _take(obj, dim, index)
            return obj
        variables, coords = dict(obj.variables), dict(obj.coords)
        for dim, index in indexers.items():
            if getattr(index, "dtype", None) == bool:
                index = np.flatnonzero(index)  # scan the mask once; every variable reuses the positions
            for name, value in variables.items():
                if dim in value.dims:
                    variables[name] = _take(value, dim, index)
            if dim in coords:
                if isinstance(index, (int, np.integer)):
                    del coords[dim]
                else:
                    cut = coords[dim][index]
                    coords[dim] = cut if isinstance(index, builtins.slice) else _freeze(cut)
        return obj.evolve(variables, coords)

    def sel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        positions = {dim: _positions_of(obj.coords[dim], wanted, dim) for dim, wanted in indexers.items()}
        return self.isel(obj, positions)

    def field(self, data: NumpyData, holder: Hashable, field_axis: Hashable, name: Hashable) -> NumpyArray:
        return self.named(self.sel(data.array(holder), {field_axis: name}), name)

    def transpose(self, value: NumpyArray, *dims: Hashable) -> NumpyArray:
        order = [value.axis(d) for d in dims]
        return NumpyArray._make(np.transpose(value.data, order), tuple(dims), value.coords, value.attrs, value.name)

    def rename_axes(self, value: NumpyArray, names: Mapping[Hashable, Hashable]) -> NumpyArray:
        dims = tuple(names.get(d, d) for d in value.dims)
        coords = {names.get(d, d): c for d, c in value.coords.items()}
        return NumpyArray._make(value.data, dims, coords, value.attrs, value.name)

    def concat(self, values: Sequence[Any], dim: Hashable, labels: Sequence[Any], *, exact: bool = False) -> NumpyArray:
        arrays = [v if isinstance(v, NumpyArray) else NumpyArray(v) for v in values]
        raw, dims, coords = _broadcast(arrays)
        shape = np.broadcast_shapes(*(np.shape(r) for r in raw))
        data = np.stack([np.broadcast_to(r, shape) for r in raw])
        return NumpyArray._make(data, (dim, *dims), {dim: _new_labels(labels), **coords})

    def apply(self, op: Callable, *operands: Any) -> Any:
        if not any(isinstance(item, NumpyArray) for item in operands):
            return op(*operands)
        raw, dims, coords = _broadcast(operands)
        if op in _QUIET:
            with np.errstate(all="ignore"):
                result = op(*raw)
        else:
            result = op(*raw)
        return NumpyArray._make(np.asarray(result), dims, coords)

    def where(self, condition: Any, true: Any, false: Any) -> Any:
        return self.apply(np.where, condition, true, false)

    def isin(self, value: NumpyArray, choices: Sequence[Any]) -> NumpyArray:
        return NumpyArray._make(np.isin(value.data, choices), value.dims, value.coords)

    def dot(self, left: NumpyArray, right: NumpyArray, over: Sequence[Hashable]) -> NumpyArray:
        dims = list(dict.fromkeys([*left.dims, *right.dims]))
        letters = {d: chr(ord("a") + k) for k, d in enumerate(dims)}
        kept = [d for d in dims if d not in over]
        spec = ",".join("".join(letters[d] for d in item.dims) for item in (left, right))
        result = np.einsum(f"{spec}->{''.join(letters[d] for d in kept)}", left.data, right.data)
        coords = {d: c for item in (left, right) for d, c in item.coords.items() if d in kept}
        return NumpyArray._make(np.asarray(result), tuple(kept), coords)

    def fresh(self, value: Any, keep_attrs: bool) -> Any:
        if isinstance(value, NumpyArray):
            return NumpyArray._make(value.data, value.dims, value.coords, dict(value.attrs) if keep_attrs else {},
                                    value.name)
        return value

    # ── Reductions and grouping ─────────────────────────────────────────────

    def reduce(self, array: NumpyArray, op: str, axes: Sequence[Hashable], na_rm: bool = False,
               min_count: int | None = None) -> NumpyArray:
        axes = list(axes)
        if not axes:
            return array
        numbers = tuple(array.axis(a) for a in axes)
        # One axis goes to NumPy as an integer, as xarray passes it: NumPy sums a
        # tuple of axes in another order, which changes float results in the last bit.
        result = _reduce_data(array.data, op, numbers[0] if len(numbers) == 1 else numbers, na_rm, min_count)
        dims = tuple(d for d in array.dims if d not in axes)
        return NumpyArray._make(np.asarray(result), dims, {d: c for d, c in array.coords.items() if d in dims})

    def group_reduce(self, array: NumpyArray, op: str, groups: Any, others: Sequence[Hashable],
                     na_rm: bool, min_count: int | None) -> NumpyArray:
        fast = self._one_pass(array, op, groups, others, na_rm, min_count)
        if fast is not None:
            return fast
        stats = [self.reduce(part, op, [groups.axis, *others], na_rm, min_count)
                 for part in self._group_parts(array, groups)]
        labels = _new_labels(groups.labels)
        if stats:
            data = np.stack([s.data for s in stats])
            return NumpyArray._make(data, (groups.name, *stats[0].dims), {groups.name: labels, **stats[0].coords})
        empty = self.reduce(_take(array, groups.axis, builtins.slice(0, 0)), op, [groups.axis, *others], na_rm, min_count)
        return NumpyArray._make(empty.data[np.newaxis][:0], (groups.name, *empty.dims),
                                {groups.name: labels, **empty.coords})

    def _one_pass(self, array: NumpyArray, op: str, groups: Any, others: Sequence[Hashable],
                  na_rm: bool, min_count: int | None) -> NumpyArray | None:
        """Every group at once where that is exact (``tidy3a.grouped``); the rest one by one."""
        if others or len(groups.labels) <= _FEW_GROUPS:
            return None
        order, starts, ends = groups.blocks()
        fast = grouped.one_pass(array.data, array.axis(groups.axis), op, na_rm, min_count, order, starts, ends)
        if fast is None:
            return None
        values, done = fast
        rest = np.flatnonzero(~done)
        if len(rest):
            stats = np.stack([self.reduce(_take(array, groups.axis, order[int(starts[g]):int(ends[g])]), op,
                                          [groups.axis], na_rm, min_count).data for g in rest])
            if stats.dtype != values.dtype:
                return None
            values[rest] = stats
        dims = tuple(d for d in array.dims if d != groups.axis)
        coords = {d: c for d, c in array.coords.items() if d in dims}
        return NumpyArray._make(values, (groups.name, *dims), {groups.name: _new_labels(groups.labels), **coords})

    def _group_parts(self, array: NumpyArray, groups: Any) -> list[NumpyArray]:
        """Each group's positions, in their original order and in a mask's memory layout.

        Indexing an axis other than the first with positions or a mask lays
        the result out with that axis first in memory; reductions over it then
        add in another order than over a contiguous copy, which changes float
        results in the last bit. Each group is therefore indexed by its
        positions, as a mask would be, so results are bit-identical to the
        xarray backend and to per-group masks. With many groups, the positions
        come from one sort instead of one scan of the axis per group.
        """
        codes = groups.codes.values
        count = len(groups.labels)
        if count <= _FEW_GROUPS:
            return [_take(array, groups.axis, np.flatnonzero(codes == code)) for code in range(count)]
        order, starts, ends = groups.blocks()
        return [_take(array, groups.axis, order[int(start):int(end)]) for start, end in zip(starts, ends)]

    def group_counts(self, groups: Any, per_position: int) -> NumpyArray:
        counts = np.bincount(groups.codes.values, minlength=len(groups.labels)) * per_position
        # Plain NumPy labels, not objects: the xarray backend labels counts this way.
        return NumpyArray._make(counts, (groups.name,), {groups.name: _freeze(np.asarray(groups.labels))})

    def broadcast(self, stats: NumpyArray, groups: Any) -> NumpyArray:
        axis = stats.axis(groups.name)
        data = np.take(stats.data, groups.codes.values, axis=axis)
        dims = stats.dims[:axis] + (groups.axis,) + stats.dims[axis + 1:]
        coords = {d: c for d, c in stats.coords.items() if d != groups.name}
        if groups.axis in groups.codes.coords:
            coords[groups.axis] = groups.codes.coords[groups.axis]
        return NumpyArray._make(data, dims, coords)

    # ── Building collections ────────────────────────────────────────────────

    def select_variables(self, data: NumpyData, names: Sequence[Hashable]) -> NumpyData:
        if not names:
            return data.evolve(variables={})
        variables = {name: data.variables[name] for name in names}
        used = {d for v in variables.values() for d in v.dims}
        return data.evolve(variables, {d: c for d, c in data.coords.items() if d in used})

    def assign(self, data: NumpyData, name: Hashable, value: Any) -> NumpyData:
        if not isinstance(value, NumpyArray):
            value = NumpyArray(value)
        variables = dict(data.variables)
        variables[name] = value
        coords = dict(data.coords)
        for dim, labels in value.coords.items():
            coords.setdefault(dim, labels)
        return data.evolve(variables, coords)

    def replace(self, data: NumpyData, name: Hashable, value: NumpyArray) -> NumpyData:
        return self.assign(data, name, value)

    def with_values(self, target: NumpyArray, values: np.ndarray) -> NumpyArray:
        return NumpyArray._make(values, target.dims, target.coords, {}, target.name)

    def has_metadata(self, value: NumpyArray) -> bool:
        return bool(value.attrs)

    def drop_position_axis(self, data: NumpyData, axis: Hashable) -> NumpyData:
        return self.isel(data, {axis: 0})

    def summary(self, results: Mapping[Hashable, NumpyArray], attrs: Mapping[Hashable, Any],
                coords: Mapping[Hashable, Any]) -> NumpyData:
        labels: dict[Hashable, np.ndarray] = {}
        coord_attrs: dict[Hashable, Any] = {}
        for value in results.values():
            for dim, values in value.coords.items():
                labels.setdefault(dim, values)
        for dim, value in coords.items():
            labels[dim] = value.coords.get(dim, value.data) if isinstance(value, NumpyArray) else np.asarray(value)
            coord_attrs[dim] = dict(value.attrs) if isinstance(value, NumpyArray) else {}
        return NumpyData(dict(results), labels, attrs, coord_attrs)

    def attach(self, data: NumpyData, name: Hashable, values: np.ndarray, along: Hashable) -> NumpyData:
        variables = dict(data.variables)
        variables[name] = NumpyArray._make(np.asarray(values), (along,), {}, {}, name)
        return data.evolve(variables)

    # ── Joins ───────────────────────────────────────────────────────────────

    def rename_variables(self, data: NumpyData, names: Mapping[Hashable, Hashable]) -> NumpyData:
        variables = {names.get(n, n): v for n, v in data.variables.items()}
        return data.evolve(variables)

    def nearest(self, right: NumpyData, dim: Hashable, left: NumpyData, tolerance: Any) -> NumpyData:
        theirs, ours = right.coords[dim], left.coords[dim]
        order = np.argsort(theirs, kind="stable")
        ordered = theirs[order]
        upper = np.clip(np.searchsorted(ordered, ours), 0, len(ordered) - 1)
        lower = np.clip(upper - 1, 0, len(ordered) - 1)
        pick_lower = np.abs(ours - ordered[lower]) <= np.abs(ordered[upper] - ours)
        nearest = np.where(pick_lower, lower, upper)
        positions = np.where(np.abs(ordered[nearest] - ours) <= tolerance, order[nearest], -1)
        return self._reindexed(right, dim, positions, ours)

    def _reindexed(self, data: NumpyData, dim: Hashable, positions: np.ndarray, labels: np.ndarray) -> NumpyData:
        variables = {}
        for name in data.variables:
            value = data.array(name)
            variables[name] = _reindex(value, dim, positions, labels) if dim in value.dims else value
        coords = dict(data.coords)
        coords[dim] = labels
        return data.evolve(variables, coords)

    def merge(self, left: NumpyData, right: NumpyData, how: str) -> NumpyData:
        for dim in [d for d in left.coords if d in right.coords]:
            ours, theirs = left.coords[dim], right.coords[dim]
            if exact.equal(ours, theirs):
                continue
            target = _joined_labels(ours, theirs, how, dim)
            left = self._reindexed(left, dim, label_index(ours, "join", dim).get_indexer(target), target)
            right = self._reindexed(right, dim, label_index(theirs, "join", dim).get_indexer(target), target)
        variables = {**{n: left.array(n) for n in left.variables}, **{n: right.array(n) for n in right.variables}}
        coords = {**right.coords, **left.coords}
        coord_attrs = {d: _combine_attrs(left.coord_attrs.get(d, {}), right.coord_attrs.get(d, {})) for d in coords}
        return NumpyData(variables, coords, _combine_attrs(left.attrs, right.attrs), coord_attrs)


def _joined_labels(ours: np.ndarray, theirs: np.ndarray, how: str, dim: Hashable) -> np.ndarray:
    """The labels a join keeps, in the order xarray's joins give them."""
    if how == "left":
        return ours
    found = label_index(theirs, "join", dim).get_indexer(ours) >= 0
    if how == "inner":
        return _freeze(ours[found])
    extra = theirs[label_index(ours, "join", dim).get_indexer(theirs) < 0]
    union = np.concatenate([ours, extra])
    try:
        return _freeze(np.sort(union, kind="stable"))
    except TypeError:  # labels that cannot be ordered keep their first-seen order
        return _freeze(union)


NUMPY = NumpyBackend()
