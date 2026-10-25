"""Array backends (plan milestone 9): where tidy3A's verbs meet an array library.

The verbs, expressions, formulas, alignment rules, and inspection output hold
every tidy3A rule and message. They reach stored data only through a
backend: a collection of variables sharing axes and labels, and the arrays
in it. Arrays expose a small common surface (``dims``, ``sizes``, ``ndim``,
``dtype``, and ``values``, which computes NumPy values); everything else,
from label lookups to arithmetic under the alignment rules, is a backend
method.

``XarrayBackend`` wraps the xarray code the package used before the
interface existed, unchanged; it is the default. ``NumpyBackend``
(``tidy3a.numpy_backend``) is experimental and holds eager, in-memory data
only. ``set_backend`` chooses the default, and ``tidy()`` and the readers take
``backend=``. A backend raises
tidy3A errors only where a limitation is its own (computing a median of a
whole chunked array); rule errors and their hints come from the callers.

User-supplied arrays, such as a rotation matrix, are xarray objects whatever
the backend; ``is_xarray`` recognises them at that boundary.
"""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tidy3a.errors import TidyError

_EXACT = {"arithmetic_join": "exact"}

# Up to this many groups, scanning the axis once per group is cheaper than sorting it.
_FEW_GROUPS = 8


def is_xarray(value: Any) -> bool:
    """An xarray array supplied from outside, e.g. a rotation matrix."""
    return isinstance(value, xr.DataArray)


def is_xarray_object(value: Any) -> bool:
    return isinstance(value, (xr.DataArray, xr.Dataset))


def label_index(labels: Sequence[Any], name: Hashable) -> pd.Index:
    """A pandas index for new labels; text stays object dtype, which older xarray needs."""
    values = np.asarray(labels)
    if values.dtype.kind in "UO":
        values = values.astype(object)
    return pd.Index(values, name=name, dtype=values.dtype)


class XarrayBackend:
    """Collections are ``xr.Dataset`` objects and arrays are ``xr.DataArray`` objects."""

    name = "xarray"

    # ── Conversion ──────────────────────────────────────────────────────────

    def wrap(self, data: Any) -> tuple[xr.Dataset, str]:
        """A collection from an xarray object, and a description of its source."""
        if isinstance(data, xr.DataArray):
            name = data.name if data.name is not None else "values"
            return data.to_dataset(name=name), "xarray DataArray"
        return data, "xarray Dataset"

    def outside(self, value: Any) -> Any:
        """An array from the calling code, as this backend's array; xarray's already is."""
        return value

    def to_xarray(self, data: xr.Dataset) -> xr.Dataset:
        return data.copy(deep=False)

    def collect(self, data: xr.Dataset) -> xr.Dataset:
        return data.compute()

    # ── Collections ─────────────────────────────────────────────────────────

    def variable_names(self, data: xr.Dataset) -> list[Hashable]:
        return list(data.data_vars)

    def has_variable(self, data: xr.Dataset, name: Hashable) -> bool:
        return name in data.data_vars

    def variable(self, data: xr.Dataset, name: Hashable) -> xr.DataArray:
        return data[name]

    def variables(self, data: xr.Dataset) -> Mapping[Hashable, xr.DataArray]:
        return data.data_vars

    def sizes(self, obj: Any) -> Mapping[Hashable, int]:
        """Axis sizes of a collection or an array, in axis order."""
        return obj.sizes

    def axes(self, obj: Any) -> Sequence[Hashable]:
        return obj.dims

    def has_axis(self, obj: Any, dim: Hashable) -> bool:
        return dim in obj.dims

    def has_coordinate(self, data: xr.Dataset, name: Hashable) -> bool:
        return name in data.coords

    def attrs(self, obj: Any) -> Mapping[Hashable, Any]:
        return obj.attrs

    def coordinate_attrs(self, data: xr.Dataset, name: Hashable) -> Mapping[Hashable, Any]:
        return data[name].attrs

    def coordinate_names(self, data: xr.Dataset) -> Any:
        """Coordinate names as a keys view, so intersections keep xarray's iteration order."""
        return data.coords.keys()

    # ── Labels ──────────────────────────────────────────────────────────────

    def has_labels(self, obj: Any, dim: Hashable) -> bool:
        return dim in obj.indexes

    def labelled_axes(self, obj: Any) -> list[Hashable]:
        return list(obj.indexes)

    def label_map(self, obj: Any) -> Mapping[Hashable, Any]:
        """Labels by axis, for the labelled axes; each supports ``in``, ``len``, and iteration."""
        return obj.indexes

    def label_values(self, obj: Any, dim: Hashable) -> np.ndarray:
        """Labels as a NumPy array, for formatting and arithmetic."""
        return obj.indexes[dim].to_numpy()

    def label_list(self, obj: Any, dim: Hashable) -> list[Any]:
        """Labels as Python values, for messages and order comparisons."""
        return list(obj.indexes[dim])

    def has_label(self, obj: Any, dim: Hashable, label: Any) -> bool:
        return label in obj.indexes[dim]

    def label_count(self, obj: Any, dim: Hashable) -> int:
        return len(obj.indexes[dim])

    def labels_unique(self, obj: Any, dim: Hashable) -> bool:
        return obj.indexes[dim].is_unique

    def label_position(self, obj: Any, dim: Hashable, label: Any) -> Any:
        return obj.indexes[dim].get_loc(label)

    def compare_labels(self, left: Any, right: Any, dim: Hashable) -> tuple[str, int, int]:
        """``("same", 0, 0)``, ``("reordered", 0, 0)``, or ``("different", only in left, only in right)``."""
        left_index, right_index = left.indexes[dim], right.indexes[dim]
        if left_index.equals(right_index):
            return "same", 0, 0
        if len(left_index) == len(right_index) and left_index.sort_values().equals(right_index.sort_values()):
            return "reordered", 0, 0
        return "different", len(left_index.difference(right_index)), len(right_index.difference(left_index))

    def labels_strict_subset(self, part: Any, whole: Any, dim: Hashable) -> bool:
        """Whether *part*'s labels on *dim* are some, but not all, of *whole*'s."""
        theirs, ours = part.indexes[dim], whole.indexes[dim]
        return not ours.equals(theirs) and set(theirs) < set(ours)

    def reorder_like(self, value: xr.DataArray, dim: Hashable, reference: Any) -> xr.DataArray:
        """*value* with its labels on *dim* put in *reference*'s order."""
        return value.sel({dim: reference.indexes[dim]})

    def axis_labels(self, data: xr.Dataset, axis: Hashable) -> xr.DataArray:
        """An axis's labels as data along it; unlabelled axes give 1-based positions."""
        if axis in data.indexes:
            return data[axis].reset_coords(drop=True)
        return xr.DataArray(np.arange(1, data.sizes[axis] + 1), dims=axis, name=axis)

    def axis_coordinate(self, data: xr.Dataset, axis: Hashable) -> xr.DataArray:
        return data[axis]

    def with_positions(self, data: xr.Dataset, axis: Hashable) -> xr.Dataset:
        """Attach 1-based positions as labels, so later steps keep original identities."""
        if axis in data.indexes:
            return data
        return data.assign_coords({axis: np.arange(1, data.sizes[axis] + 1)})

    # ── Arrays ──────────────────────────────────────────────────────────────

    def is_array(self, value: Any) -> bool:
        return isinstance(value, xr.DataArray)

    def is_lazy(self, value: xr.DataArray) -> bool:
        return value.chunks is not None

    def array(self, values: Any, dims: Sequence[Hashable] = (), coords: Mapping | None = None,
              name: Hashable | None = None) -> xr.DataArray:
        if not dims and coords is None and name is None:
            return xr.DataArray(values)
        return xr.DataArray(values, dims=dims, coords=coords, name=name)

    def codes_like(self, codes: np.ndarray, values: xr.DataArray) -> xr.DataArray:
        """Group codes along *values*' one axis, carrying its labels."""
        return xr.DataArray(codes, dims=values.dims[0], coords=values.coords)

    def named(self, value: xr.DataArray, name: Hashable) -> xr.DataArray:
        return value.rename(name)

    def isel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        return obj.isel(indexers, drop=drop)

    def sel(self, obj: Any, indexers: Mapping[Hashable, Any], drop: bool = False) -> Any:
        return obj.sel(indexers, drop=drop)

    def field(self, data: xr.Dataset, holder: Hashable, field_axis: Hashable, name: Hashable) -> xr.DataArray:
        """One field of a holder, with the field axis dropped, named after the field."""
        return data[holder].sel({field_axis: name}, drop=True).rename(name)

    def transpose(self, value: xr.DataArray, *dims: Hashable) -> xr.DataArray:
        return value.transpose(*dims)

    def rename_axes(self, value: xr.DataArray, names: Mapping[Hashable, Hashable]) -> xr.DataArray:
        return value.rename(names)

    def concat(self, values: Sequence[Any], dim: Hashable, labels: Sequence[Any], *, exact: bool = False) -> xr.DataArray:
        """Stack *values* along a new axis *dim* labelled *labels*."""
        if not exact:
            return xr.concat(list(values), dim=label_index(labels, dim))
        with xr.set_options(**_EXACT):
            return xr.concat(list(values), dim=label_index(labels, dim), join="exact")

    def apply(self, op: Callable, *operands: Any) -> Any:
        """*op* on operands whose shared labels already agree (see ``tidy3a.align``)."""
        with xr.set_options(**_EXACT):
            return op(*operands)

    def where(self, condition: Any, true: Any, false: Any) -> Any:
        with xr.set_options(**_EXACT):
            return xr.where(condition, true, false)

    def isin(self, value: xr.DataArray, choices: Sequence[Any]) -> xr.DataArray:
        return value.isin(choices)

    def dot(self, left: xr.DataArray, right: xr.DataArray, over: Sequence[Hashable]) -> xr.DataArray:
        # Always NumPy's einsum. When opt_einsum is installed, xarray uses it instead, which
        # changes results in the last digits (so they would depend on what else is installed)
        # and returns column-major arrays that made later steps twice as slow.
        with xr.set_options(**_EXACT, use_opt_einsum=False):
            return xr.dot(left, right, dim=over)

    def fresh(self, value: Any, keep_attrs: bool) -> Any:
        """A computed result: no storage encoding; attributes only for a direct alias."""
        if isinstance(value, xr.DataArray):
            value = value.copy(deep=False)
            value.encoding = {}
            if not keep_attrs:
                value.attrs = {}
        return value

    # ── Reductions and grouping ─────────────────────────────────────────────

    def reduce(self, array: xr.DataArray, op: str, axes: Sequence[Hashable], na_rm: bool = False,
               min_count: int | None = None) -> xr.DataArray:
        """One reduction over *axes*, with missing values propagated unless *na_rm*."""
        axes = list(axes)
        if not axes:
            return array
        if op == "median" and na_rm and array.chunks is not None and set(axes) == set(array.dims):
            raise TidyError(
                "median(..., na_rm=True) over every axis of a chunked array requires gathering "
                "the whole array into memory. Select the required data and collect() it first, "
                "then wrap the result with tidy() to compute the median."
            )
        if op in ("min", "max") and builtins.any(array.sizes[axis] == 0 for axis in axes):
            # NumPy cannot take min or max of nothing; the result is missing.
            return xr.full_like(array.astype(float).sum(dim=axes), np.nan)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if op in ("any", "all"):
                result = getattr(array.fillna(op == "all"), op)(dim=axes)
                if na_rm or array.dtype.kind in "biu":
                    return result
                return result.where(array.notnull().all(dim=axes))
            if op == "count":
                return array.notnull().sum(dim=axes)
            options = {"skipna": na_rm} if array.dtype.kind in "fcmM" else {}
            result = getattr(array, op)(dim=axes, **options)
            if op == "sum" and min_count is not None:
                present = array.notnull().sum(dim=axes)
                result = result.where(present >= min_count)
            return result

    def group_reduce(self, array: xr.DataArray, op: str, groups: Any, others: Sequence[Hashable],
                     na_rm: bool, min_count: int | None) -> xr.DataArray:
        """*op* per group of *groups* (``tidy3a.expr.Groups``), over its axis and *others*."""
        stats = [
            self.reduce(part, op, [groups.axis, *others], na_rm, min_count)
            for part in self._group_parts(array, groups)
        ]
        index = label_index(groups.labels, groups.name)
        if stats:
            return xr.concat(stats, dim=index)
        empty = self.reduce(array.isel({groups.axis: slice(0, 0)}), op, [groups.axis, *others], na_rm, min_count)
        return empty.expand_dims({groups.name: 0}).assign_coords({groups.name: index})

    def _group_parts(self, array: xr.DataArray, groups: Any) -> list[xr.DataArray]:
        """Each group's positions, in their original order.

        With many groups, in-memory arrays are reordered once so groups sit in
        contiguous blocks, instead of scanning the whole axis once per group.
        Each block holds the same values in the same order and layout as a
        per-group mask would, so reductions give bit-identical results. Chunked
        arrays keep per-group masks, which stay lazy.
        """
        codes = groups.codes.values
        count = len(groups.labels)
        if array.chunks is not None or count <= _FEW_GROUPS:
            return [array.isel({groups.axis: codes == code}) for code in range(count)]
        order, starts, ends = groups.blocks()
        ordered = array.isel({groups.axis: order})
        first = array.get_axis_num(groups.axis) == 0
        parts = []
        for start, end in zip(starts, ends):
            part = ordered.isel({groups.axis: builtins.slice(int(start), int(end))})
            parts.append(part if first else part.copy())  # a contiguous copy, as a mask would give
        return parts

    def group_counts(self, groups: Any, per_position: int) -> xr.DataArray:
        counts = np.bincount(groups.codes.values, minlength=len(groups.labels)) * per_position
        return xr.DataArray(counts, dims=groups.name, coords={groups.name: groups.labels})

    def broadcast(self, stats: xr.DataArray, groups: Any) -> xr.DataArray:
        """A per-group result back on each grouped position."""
        picked = stats.isel({groups.name: groups.codes})
        return picked.drop_vars(groups.name, errors="ignore")

    # ── Building collections ────────────────────────────────────────────────

    def select_variables(self, data: xr.Dataset, names: Sequence[Hashable]) -> xr.Dataset:
        return data[list(names)] if names else data.drop_vars(list(data.data_vars))

    def assign(self, data: xr.Dataset, name: Hashable, value: Any) -> xr.Dataset:
        """Add or replace a variable whose labels already agree with the collection's."""
        if not isinstance(value, xr.DataArray):
            value = xr.DataArray(value)
        with xr.set_options(**_EXACT):
            return data.assign({name: value.drop_vars([c for c in value.coords if c not in value.dims])})

    def replace(self, data: xr.Dataset, name: Hashable, value: xr.DataArray) -> xr.Dataset:
        return data.assign({name: value})

    def with_values(self, target: xr.DataArray, values: np.ndarray) -> xr.DataArray:
        """*target* holding *values* instead, without attributes or storage encoding."""
        updated = target.copy(data=values)
        updated.encoding = {}
        updated.attrs = {}
        return updated

    def has_metadata(self, value: xr.DataArray) -> bool:
        return bool(value.attrs or value.encoding)

    def drop_position_axis(self, data: xr.Dataset, axis: Hashable) -> xr.Dataset:
        """Remove an axis with one position left, with its label."""
        return data.isel({axis: 0}).drop_vars(axis, errors="ignore")

    def summary(self, results: Mapping[Hashable, xr.DataArray], attrs: Mapping[Hashable, Any],
                coords: Mapping[Hashable, Any]) -> xr.Dataset:
        """A collection of summary results, keeping the source's attributes and group labels."""
        return xr.Dataset(dict(results), attrs=dict(attrs)).assign_coords(dict(coords))

    def attach(self, data: xr.Dataset, name: Hashable, values: np.ndarray, along: Hashable) -> xr.Dataset:
        """Attach 1-D values by position along *along*, taking its labels."""
        coords = {along: data.indexes[along]} if along in data.indexes else None
        return data.assign({name: xr.DataArray(values, dims=along, coords=coords)})

    # ── Joins ───────────────────────────────────────────────────────────────

    def rename_variables(self, data: xr.Dataset, names: Mapping[Hashable, Hashable]) -> xr.Dataset:
        return data.rename(dict(names))

    def nearest(self, right: xr.Dataset, dim: Hashable, left: xr.Dataset, tolerance: Any) -> xr.Dataset:
        """*right* on *left*'s labels for *dim*, taking the nearest label within *tolerance*."""
        return right.reindex({dim: left.indexes[dim]}, method="nearest", tolerance=tolerance)

    def merge(self, left: xr.Dataset, right: xr.Dataset, how: str) -> xr.Dataset:
        """Join on shared axes by label: ``left``, ``inner``, or ``outer``; conflicting attributes are dropped."""
        return xr.merge([left, right], join=how, compat="no_conflicts", combine_attrs="drop_conflicts").drop_encoding()


XARRAY = XarrayBackend()

_BACKENDS = ("xarray", "numpy")
_default = "xarray"


def set_backend(name: str) -> None:
    """Choose the backend new data uses when none is named: ``"xarray"`` (the default) or ``"numpy"``.

    The NumPy backend is experimental and holds eager, in-memory data only;
    lazy, chunked, and memory-mapped inputs stay on xarray when it is only
    the default.
    """
    global _default
    _check_name(name)
    _default = name


def get_backend(name: str | None = None) -> Any:
    """The backend called *name*, or the default one."""
    name = _default if name is None else name
    _check_name(name)
    if name == "xarray":
        return XARRAY
    from tidy3a.numpy_backend import NUMPY

    return NUMPY


def backend_for(data: Any, requested: str | None, error: type[TidyError] = TidyError) -> Any:
    """The backend that holds xarray *data*: the one asked for, or the default.

    Data the NumPy backend cannot hold stays on xarray, unless the NumPy
    backend was asked for by name; then it is an error that says why.
    """
    backend = get_backend(requested)
    if backend is XARRAY:
        return XARRAY
    reason = backend.unsupported(data)
    if reason is None:
        return backend
    if requested is not None:
        raise error(
            f'backend="{backend.name}" holds eager, in-memory data only, but {reason}.\n'
            '  Use backend="xarray", or load the values into memory first'
        )
    return XARRAY


def _check_name(name: Any) -> None:
    if name not in _BACKENDS:
        raise TidyError(f'backend must be one of {", ".join(map(repr, _BACKENDS))}, not {name!r}')
