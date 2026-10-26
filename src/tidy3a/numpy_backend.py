"""The NumPy backend (plan milestone 9, step 3; the default since milestone 10, decision 16).

An adapter: it implements tidy3a's backend protocol (``backend.Backend``)
with the labelled arrays of the ``labelarray`` package. Collections are
``labelarray.Collection`` objects and arrays ``labelarray.Array`` objects,
with one label array per labelled axis shared by every variable on it,
compared under one exact rule. The rules and messages stay in the shared
modules, and tidy3a's own messages here. It holds eager, in-memory data
only: lazy, chunked, and memory-mapped inputs stay on the xarray backend.

xarray and pandas are imported only where needed: xarray to convert to and
from xarray objects, and pandas, when installed, to list labels for
messages exactly as the xarray backend does.
"""

from __future__ import annotations

import builtins
import operator
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from labelarray import grouped
from labelarray import labels as exact
from labelarray.array import (
    Array,
    broadcast,
    concat,
    label_index,
    labels_from,
    plain,
    positions_of,
)
from labelarray.collection import Collection
from labelarray.errors import LabelError

from tidy3a.backend import is_xarray, kept_positions, memory_mapped
from tidy3a.errors import TidyError

# The names tidy3a used before the library had them; arrays in where() predicates are these.
NumpyArray = Array
NumpyData = Collection

# Python's operators on xarray arrays run without NumPy's floating-point warnings
# (division by zero, overflow); NumPy functions such as sqrt still warn. The same here.
_QUIET = {
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.mod,
    operator.pow, operator.and_, operator.or_, operator.xor, operator.eq, operator.ne, operator.lt,
    operator.le, operator.gt, operator.ge, operator.neg, operator.pos, operator.invert, operator.abs,
}

__all__ = ["NUMPY", "LabelError", "NumpyArray", "NumpyBackend", "NumpyData", "label_index"]


# ── The backend ─────────────────────────────────────────────────────────────


class NumpyBackend:
    """Collections are ``labelarray.Collection`` objects and arrays ``labelarray.Array`` objects."""

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

    def wrap(self, data: Any) -> tuple[Collection, str]:
        import xarray as xr

        if isinstance(data, xr.DataArray):
            name = data.name if data.name is not None else "values"
            return Collection.from_xarray(data.to_dataset(name=name)), "xarray DataArray"
        return Collection.from_xarray(data), "xarray Dataset"

    def to_xarray(self, data: Collection) -> Any:
        try:
            import xarray  # noqa: F401 - only to say so here, in tidy3a's words
        except ImportError:
            raise TidyError(
                "collect() and to_xarray() give an xarray Dataset, and xarray is not installed.\n"
                "  For NumPy arrays: to_dict(), or pull(...) for one variable or axis\n"
                "  For an xarray Dataset: pip install 'tidy3a[xarray]'"
            ) from None
        return data.to_xarray()

    def collect(self, data: Collection) -> Any:
        return self.to_xarray(data)

    def to_dict(self, data: Collection) -> dict:
        return data.to_dict()

    # ── Readers: built directly, without an xarray dataset in between ───────

    def from_records(self, variables: Mapping[Hashable, tuple], coords: Mapping[Hashable, np.ndarray]) -> Collection:
        arrays = {name: Array._make(np.asarray(values), (dims,) if isinstance(dims, str) else tuple(dims), {},
                                         {}, name) for name, (dims, values) in variables.items()}
        return Collection(arrays, coords)

    def from_array(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   name: Hashable) -> Collection:
        return Collection({name: Array._make(values, tuple(dims), {}, {}, name)}, coords)

    def from_split(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   axis: Hashable) -> Collection:
        """One variable per label on *axis*, each a view of *values*, as xarray's split gives."""
        number = list(dims).index(axis)
        kept = tuple(d for d in dims if d != axis)
        arrays = {}
        for position, label in enumerate(coords[axis].tolist()):
            name = str(label)
            arrays[name] = Array._make(values[(builtins.slice(None),) * number + (position,)], kept, {}, {}, name)
        return Collection(arrays, {d: c for d, c in coords.items() if d != axis})

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
        return Array(np.asarray(value.values), value.dims, coords, dict(value.attrs), value.name)

    # ── Collections ─────────────────────────────────────────────────────────

    def variable_names(self, data: Collection) -> list[Hashable]:
        return list(data.variables)

    def has_variable(self, data: Collection, name: Hashable) -> bool:
        return name in data.variables

    def variable(self, data: Collection, name: Hashable) -> Array:
        return data.array(name)

    def variables(self, data: Collection) -> dict[Hashable, Array]:
        return {name: data.array(name) for name in data.variables}

    def layout(self, data: Collection) -> list[tuple[Hashable, tuple[Hashable, ...], tuple[int, ...], Any]]:
        return [(name, value.dims, value.data.shape, value.data.dtype) for name, value in data.variables.items()]

    def sizes(self, obj: Any) -> Mapping[Hashable, int]:
        return obj.sizes

    def axes(self, obj: Any) -> tuple[Hashable, ...]:
        return tuple(obj.sizes)

    def has_axis(self, obj: Any, dim: Hashable) -> bool:
        try:
            return dim in obj.sizes
        except TypeError:  # an unhashable name is not an axis
            return False

    def has_coordinate(self, data: Collection, name: Hashable) -> bool:
        return name in data.coords

    def attrs(self, obj: Any) -> Mapping[Hashable, Any]:
        return obj.attrs

    def coordinate_attrs(self, data: Collection, name: Hashable) -> Mapping[Hashable, Any]:
        return data.coord_attrs.get(name, {})

    def coordinate_names(self, data: Collection) -> Any:
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
        return plain(obj.coords[dim])

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
        return positions_of(obj.coords[dim], label, dim)

    def compare_labels(self, left: Any, right: Any, dim: Hashable) -> tuple[str, int, int]:
        return exact.compare(left.coords[dim], right.coords[dim], "align", dim)

    def labels_strict_subset(self, part: Any, whole: Any, dim: Hashable) -> bool:
        return exact.strict_subset(part.coords[dim], whole.coords[dim], "align", dim)

    def reorder_like(self, value: Array, dim: Hashable, reference: Any) -> Array:
        order = label_index(value.coords[dim], "align", dim).get_indexer(reference.coords[dim])
        taken = value.isel({dim: order})
        taken.coords[dim] = reference.coords[dim]
        return taken

    def axis_labels(self, data: Collection, axis: Hashable) -> Array:
        if axis in data.coords:
            return Array._make(plain(data.coords[axis]), (axis,), {axis: data.coords[axis]},
                                    dict(data.coord_attrs.get(axis, {})), axis)
        return Array._make(np.arange(1, data.sizes[axis] + 1), (axis,), {}, {}, axis)

    def axis_coordinate(self, data: Collection, axis: Hashable) -> Array:
        return self.axis_labels(data, axis)

    def keep(self, data: Collection, axis: Hashable, index: Any) -> Collection:
        """Positions on *axis*, as ``XarrayBackend.keep``: only the kept positions are built."""
        if axis in data.coords:
            return self.isel(data, {axis: index})
        index, kept = kept_positions(index, data.sizes[axis])
        out = self.isel(data, {axis: index})
        np.add(kept, 1, out=kept)  # the indexer has been used; it becomes the labels
        coords = dict(out.coords)
        coords[axis] = labels_from(kept)
        return out.evolve(coords=coords)

    # ── Arrays ──────────────────────────────────────────────────────────────

    def is_array(self, value: Any) -> bool:
        return isinstance(value, Array)

    def array(self, values: Any, dims: Sequence[Hashable] = (), coords: Mapping | None = None,
              name: Hashable | None = None) -> Array:
        if coords:
            coords = {d: labels_from(c) for d, c in coords.items()}
        return Array(values, dims, coords, name=name)

    def codes_like(self, codes: np.ndarray, values: Array) -> Array:
        return Array._make(np.asarray(codes), values.dims[:1], dict(values.coords))

    def _named(self, value: Array, name: Hashable) -> Array:
        return Array._make(value.data, value.dims, value.coords, value.attrs, name)

    def isel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        return obj.isel(indexers)

    def sel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        return obj.sel(indexers)

    def field(self, data: Collection, holder: Hashable, field_axis: Hashable, name: Hashable) -> Array:
        return self._named(self.sel(data.array(holder), {field_axis: name}), name)

    def transpose(self, value: Array, *dims: Hashable) -> Array:
        return value.transpose(*dims)

    def rename_axes(self, value: Array, names: Mapping[Hashable, Hashable]) -> Array:
        return value.rename(names)

    def concat(self, values: Sequence[Any], dim: Hashable, labels: Sequence[Any], *, exact: bool = False) -> Array:
        return concat(values, dim, labels)

    def apply(self, op: Callable, *operands: Any) -> Any:
        if not any(isinstance(item, Array) for item in operands):
            return op(*operands)
        raw, dims, coords = broadcast(operands)
        if op in _QUIET:
            with np.errstate(all="ignore"):
                result = op(*raw)
        else:
            result = op(*raw)
        return Array._make(np.asarray(result), dims, coords)

    def where(self, condition: Any, true: Any, false: Any) -> Any:
        return self.apply(np.where, condition, true, false)

    def isin(self, value: Array, choices: Sequence[Any]) -> Array:
        return Array._make(np.isin(value.data, choices), value.dims, value.coords)

    def dot(self, left: Array, right: Array, over: Sequence[Hashable]) -> Array:
        return left.dot(right, over)

    def fresh(self, value: Any, keep_attrs: bool) -> Any:
        if isinstance(value, Array):
            return Array._make(value.data, value.dims, value.coords, dict(value.attrs) if keep_attrs else {},
                                    value.name)
        return value

    # ── Reductions and grouping ─────────────────────────────────────────────

    def reduce(self, array: Array, op: str, axes: Sequence[Hashable], na_rm: bool = False,
               min_count: int | None = None) -> Array:
        return array.reduce(op, axes, na_rm=na_rm, min_count=min_count)

    def group_reduce(self, array: Array, op: str, groups: Any, others: Sequence[Hashable],
                     na_rm: bool, min_count: int | None) -> Array:
        return grouped.reduce(array, op, groups, groups.axis, groups.name, others, na_rm=na_rm, min_count=min_count)

    def group_counts(self, groups: Any, per_position: int) -> Array:
        return grouped.counts(groups, groups.name, per_position)

    def broadcast(self, stats: Array, groups: Any) -> Array:
        return grouped.broadcast(stats, groups, groups.axis, groups.name, groups.codes_array.coords.get(groups.axis))

    # ── Building collections ────────────────────────────────────────────────

    def select_variables(self, data: Collection, names: Sequence[Hashable]) -> Collection:
        return data.select(names)

    def assign(self, data: Collection, name: Hashable, value: Any) -> Collection:
        return data.assign(name, value)

    def replace(self, data: Collection, name: Hashable, value: Array) -> Collection:
        return self.assign(data, name, value)

    def with_values(self, target: Array, values: np.ndarray) -> Array:
        return Array._make(values, target.dims, target.coords, {}, target.name)

    def has_metadata(self, value: Array) -> bool:
        return bool(value.attrs)

    def drop_position_axis(self, data: Collection, axis: Hashable) -> Collection:
        return self.isel(data, {axis: 0})

    def summary(self, results: Mapping[Hashable, Array], attrs: Mapping[Hashable, Any],
                coords: Mapping[Hashable, Any]) -> Collection:
        labels: dict[Hashable, np.ndarray] = {}
        coord_attrs: dict[Hashable, Any] = {}
        for value in results.values():
            for dim, values in value.coords.items():
                labels.setdefault(dim, values)
        for dim, value in coords.items():
            labels[dim] = value.coords.get(dim, value.data) if isinstance(value, Array) else np.asarray(value)
            coord_attrs[dim] = dict(value.attrs) if isinstance(value, Array) else {}
        return Collection(dict(results), labels, attrs, coord_attrs)

    def attach(self, data: Collection, name: Hashable, values: np.ndarray, along: Hashable) -> Collection:
        variables = dict(data.variables)
        variables[name] = Array._make(np.asarray(values), (along,), {}, {}, name)
        return data.evolve(variables)

    # ── Joins ───────────────────────────────────────────────────────────────

    def rename_variables(self, data: Collection, names: Mapping[Hashable, Hashable]) -> Collection:
        return data.rename(names)

    def nearest(self, right: Collection, dim: Hashable, left: Collection, tolerance: Any) -> Collection:
        return right.nearest(dim, left.coords[dim], tolerance)

    def merge(self, left: Collection, right: Collection, how: str) -> Collection:
        return left.join(right, how)


NUMPY = NumpyBackend()
