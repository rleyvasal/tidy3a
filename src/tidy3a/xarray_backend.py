"""The xarray backend (plan milestone 9): tidy3a's default storage and computation.

Collections are ``xr.Dataset`` objects and arrays are ``xr.DataArray``
objects; each method is the xarray code the package used before the backend
interface existed. This module is imported on first use, so ``import tidy3a``
does not need xarray or pandas.
"""

from __future__ import annotations

import builtins
import math
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from labelarray import grouped, sums
from labelarray.array import Array
from labelarray import labels as exact
from labelarray.contract import contract
from xarray.indexes import PandasIndex

from tidy3a.backend import kept_positions
from tidy3a.errors import TidyError

_EXACT = {"arithmetic_join": "exact"}


_KEEPS_ATTRS: dict[Any, bool] = {}


def _dask_backed(array: xr.DataArray) -> bool:
    """Whether *array* holds a Dask array (chunked data from another library takes xarray's arithmetic)."""
    return array.chunks is not None and type(array.data).__module__.split(".")[0] == "dask"


def _as_objects(array: xr.DataArray, op: str) -> xr.DataArray:
    """Text as Python objects for min and max, so the result is objects whether grouped or not, as on the NumPy
    backend, and an empty group's NaN has a place."""
    return array.astype(object) if op in ("min", "max") and array.dtype.kind in "US" else array


def _missing_of(dtype: np.dtype) -> tuple[Any, np.dtype]:
    """The missing value of *dtype* and the type that holds it, as labelarray's empty min and max give them: a float
    or complex type's own NaN, a time type's NaT, text and objects as NaN in an object array; else float64's NaN."""
    if dtype.kind in "fc":
        return np.nan, dtype
    if dtype.kind in "mM":
        return np.array("NaT", dtype=dtype)[()], dtype
    if dtype.kind in "OUS":
        return np.nan, np.dtype(object)
    return np.nan, np.dtype(np.float64)


def _reductions_keep_attrs() -> bool:
    """Whether xarray's reductions keep attributes, under the current ``keep_attrs`` setting.

    The default changed across xarray versions, so one tiny reduction answers
    it, once per setting.
    """
    setting = xr.get_options()["keep_attrs"]
    if setting not in _KEEPS_ATTRS:
        probe = xr.DataArray(np.zeros(1, dtype=np.float32), dims="d", attrs={"probe": 1})
        _KEEPS_ATTRS[setting] = bool(probe.sum("d").attrs)
    return _KEEPS_ATTRS[setting]


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

    def unsupported(self, data: Any) -> str | None:
        """Why an xarray object cannot be held here: never, so None."""
        return None

    # ── Readers ─────────────────────────────────────────────────────────────

    def from_records(self, variables: Mapping[Hashable, tuple], coords: Mapping[Hashable, np.ndarray]) -> xr.Dataset:
        """A collection of separately typed fields, each ``(dims, values)`` (``read_binary``)."""
        return xr.Dataset(dict(variables), coords=dict(coords))

    def from_array(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   name: Hashable) -> xr.Dataset:
        """One array as a collection of one variable (``read_array``)."""
        return xr.DataArray(values, dims=dims, coords=dict(coords)).to_dataset(name=name)

    def from_split(self, values: np.ndarray, dims: Sequence[Hashable], coords: Mapping[Hashable, np.ndarray],
                   axis: Hashable) -> xr.Dataset:
        """One variable per label on *axis* (``read_array(..., variables_from=axis)``)."""
        return xr.DataArray(values, dims=dims, coords=dict(coords)).to_dataset(dim=axis)

    def outside(self, value: Any) -> Any:
        """An array from the calling code, as this backend's array; xarray's already is."""
        return value

    def to_xarray(self, data: xr.Dataset) -> xr.Dataset:
        return data.copy(deep=False)

    def collect(self, data: xr.Dataset) -> xr.Dataset:
        return data.compute()

    def to_dict(self, data: xr.Dataset) -> dict:
        return data.to_dict(data="array")

    # ── Collections ─────────────────────────────────────────────────────────

    def variable_names(self, data: xr.Dataset) -> list[Hashable]:
        return list(data.data_vars)

    def has_variable(self, data: xr.Dataset, name: Hashable) -> bool:
        return name in data.data_vars

    def variable(self, data: xr.Dataset, name: Hashable) -> xr.DataArray:
        return data[name]

    def variables(self, data: xr.Dataset) -> Mapping[Hashable, xr.DataArray]:
        return data.data_vars

    def layout(self, data: xr.Dataset) -> list[tuple[Hashable, tuple[Hashable, ...], tuple[int, ...], Any]]:
        stored = data.variables  # xarray Variables: no DataArray is built per variable
        return [(name, stored[name].dims, stored[name].shape, stored[name].dtype) for name in data.data_vars]

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
        """``("same", 0, 0)``, ``("reordered", 0, 0)``, or ``("different", only in left, only in right)``.

        Labels compare exactly (``tidy3a.labels``), not as pandas compares
        them, which pairs the integer ``2**53 + 1`` with the float ``2**53``.
        """
        left_index, right_index = left.indexes[dim], right.indexes[dim]
        if left_index is right_index:
            return "same", 0, 0
        return exact.compare(left_index.to_numpy(), right_index.to_numpy(), "align", dim, mixed=True)

    def labels_strict_subset(self, part: Any, whole: Any, dim: Hashable) -> bool:
        """Whether *part*'s labels on *dim* are some, but not all, of *whole*'s."""
        return exact.strict_subset(part.indexes[dim].to_numpy(), whole.indexes[dim].to_numpy(), "align", dim,
                                   mixed=True)

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

    def keep(self, data: xr.Dataset, axis: Hashable, index: Any) -> xr.Dataset:
        """Positions on *axis* (a mask, a slice, or a list); an unlabelled axis is labelled with
        the 1-based positions kept, so later steps keep original identities.

        Only the kept positions are built, not positions for the whole axis.
        """
        if axis in data.indexes:
            return data.isel({axis: index})
        index, kept = kept_positions(index, data.sizes[axis])
        out = data.isel({axis: index})
        np.add(kept, 1, out=kept)  # the indexer has been used; it becomes the labels
        return out.assign_coords({axis: kept})

    def label_positions(self, data: xr.Dataset, axis: Hashable, start: int) -> xr.Dataset:
        return data.assign_coords({axis: np.arange(start + 1, start + 1 + data.sizes[axis])})

    def unlabel(self, data: xr.Dataset, axis: Hashable) -> xr.Dataset:
        return data.drop_vars(axis, errors="ignore")

    def join_along(self, parts: Sequence[xr.Dataset], axis: Hashable) -> xr.Dataset:
        """``xr.concat`` along an existing axis: only what varies along it is joined, the rest comes from the first
        part, and the first part's attributes and storage encodings are kept."""
        first = parts[0]
        joined = xr.concat(list(parts), dim=axis, data_vars="minimal", coords="minimal", compat="override",
                           join="exact", combine_attrs="override")
        for name, variable in joined.variables.items():
            if name in first.variables:
                variable.encoding = dict(first.variables[name].encoding)
        return joined

    # ── Arrays ──────────────────────────────────────────────────────────────

    def is_array(self, value: Any) -> bool:
        return isinstance(value, xr.DataArray)

    def array(self, values: Any, dims: Sequence[Hashable] = (), coords: Mapping | None = None,
              name: Hashable | None = None) -> xr.DataArray:
        if not dims and coords is None and name is None:
            return xr.DataArray(values)
        return xr.DataArray(values, dims=dims, coords=coords, name=name)

    def codes_like(self, codes: np.ndarray, values: xr.DataArray) -> xr.DataArray:
        """Group codes along *values*' one axis, carrying its labels."""
        return xr.DataArray(codes, dims=values.dims[0], coords=values.coords)

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
        """In memory, the shared contraction (``labelarray.contract``), as the NumPy backend does.

        Lazy arrays use ``xr.dot`` with NumPy's einsum: with ``opt_einsum``
        installed, xarray would use it, which changes results in the last
        digits and returns column-major arrays that made later steps twice as slow.
        """
        if left.chunks is None and right.chunks is None:
            values, kept = contract(left.values, left.dims, right.values, right.dims, over)
            coords = {name: coord for item in (left, right) for name, coord in item.coords.items()
                      if set(coord.dims) <= set(kept)}
            return xr.DataArray(values, dims=kept, coords=coords)
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
               min_count: int | None = None, q: float | None = None) -> xr.DataArray:
        """One reduction over *axes*, with missing values propagated unless *na_rm*."""
        axes = list(axes)
        if not axes:
            return array
        if op == "quantile":  # labelarray's, NumPy's own, so both backends give its type (xarray's gives float64)
            values = Array(np.asarray(array.values), tuple(array.dims)).reduce("quantile", axes, na_rm=na_rm, q=q)
            return self._reduced_like(array, axes, values.data, "median")
        array = _as_objects(array, op)
        if op == "median" and na_rm and array.chunks is not None and set(axes) == set(array.dims):
            raise TidyError(
                "median(..., na_rm=True) over every axis of a chunked array requires gathering "
                "the whole array into memory. Select the required data and collect() it first, "
                "then wrap the result with tidy() to compute the median."
            )
        if op in ("min", "max") and builtins.any(array.sizes[axis] == 0 for axis in axes):
            # NumPy cannot take min or max of nothing; the result is missing, in the values' own type where it has a
            # missing value (NaN of float32, NaT), as labelarray gives it.
            value, dtype = _missing_of(array.dtype)
            return xr.full_like(array.isnull().sum(dim=axes), value, dtype=dtype)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if op in ("any", "all"):
                result = getattr(array.fillna(op == "all"), op)(dim=axes)
                if na_rm or array.dtype.kind in "biu":
                    return result
                return result.where(array.notnull().all(dim=axes))
            if op == "count":
                return array.notnull().sum(dim=axes)
            if op in ("sum", "mean") and array.dtype == np.float32:
                if array.chunks is None:
                    return self._float32(array, op, axes, na_rm, min_count)
                if _dask_backed(array):
                    values = self._float32_chunked(array, op, axes, na_rm, min_count)
                    return self._reduced_like(array, axes, values, op)
            options = {"skipna": na_rm} if array.dtype.kind in "fcmM" else {}
            result = getattr(array, op)(dim=axes, **options)
            if op == "sum" and min_count is not None:
                present = array.notnull().sum(dim=axes)
                result = result.where(present >= min_count)
            return result

    def _float32(self, array: xr.DataArray, op: str, axes: Sequence[Hashable], na_rm: bool,
                 min_count: int | None) -> xr.DataArray:
        """A float32 sum or mean from the shared kernel (``labelarray.sums``).

        The result carries what xarray's reduction would: the kept axes, the
        coordinates on them, the name, and the attributes when reductions keep
        them. It is built directly, which costs a few microseconds where
        reducing a one-position slice to get it cost more than the sum; arrays
        with indexes other than pandas' still take that slice.
        """
        numbers = tuple(array.get_axis_num(a) for a in axes)
        values = sums.reduce(array.values, op, numbers, na_rm=na_rm, min_count=min_count)
        return self._reduced_like(array, axes, values, op)

    def _reduced_like(self, array: xr.DataArray, axes: Sequence[Hashable], values: Any, op: str) -> xr.DataArray:
        """*values* as the result of reducing *array* over *axes* with *op*: its kept axes, coordinates, name, attributes."""
        if any(type(index) is not PandasIndex for index in array.xindexes.values()):
            template = getattr(array.isel({a: slice(0, 1) for a in axes}), op)(dim=list(axes))
            return template.copy(data=values)
        kept = tuple(dim for dim in array.dims if dim not in axes)
        coords = {name: coord.variable for name, coord in array.coords.items() if set(coord.dims) <= set(kept)}
        attrs = dict(array.attrs) if _reductions_keep_attrs() else {}
        return xr.DataArray(values, dims=kept, coords=coords, name=array.name, attrs=attrs)

    def _float32_chunked(self, array: xr.DataArray, op: str, axes: Sequence[Hashable], na_rm: bool,
                         min_count: int | None, groups: Any = None) -> Any:
        """A float32 sum or mean of Dask-backed data: the bits of the in-memory kernel, for any chunking, still lazy.

        One task per chunk estimates its sums (``labelarray.sums.chunk_estimate``);
        the pieces add across chunks in Dask's order, which the error bound
        allows; one last task settles the results (``sums.from_chunks``). The
        few groups the bound cannot settle, if any, get a second pass over the
        chunks for their exact sums (``sums.chunk_exact``), computed from inside
        that task. The result is a one-chunk Dask array, ``(groups, *kept)``
        with *groups*, else the kept axes.
        """
        import dask.array as da

        data = array.data
        numbers = tuple(array.get_axis_num(a) for a in axes)
        letters = tuple(f"x{a}" for a in range(data.ndim))  # Dask's index labels, one per axis
        kept_shape = tuple(data.shape[a] for a in range(data.ndim) if a not in numbers)
        count = len(groups) if groups is not None else 1
        sources: tuple = (data, letters)
        if groups is not None:
            codes = da.from_array(np.asarray(groups.codes), chunks=(data.chunks[numbers[0]],))
            sources += (codes, (letters[numbers[0]],))

        def block_shape(block):  # the kept axes' block sizes, with each reduced axis back as size 1
            return tuple(1 if a in numbers else block.shape[a] for a in range(block.ndim))

        def estimate(block, codes=None):
            pieces = sums.chunk_estimate(block, numbers, na_rm=na_rm, codes=codes, groups=count)
            return pieces.reshape((sums.PIECES, count, *block_shape(block)))

        pieces = da.blockwise(estimate, ("p", "g", *letters), *sources, new_axes={"p": sums.PIECES, "g": count},
                              adjust_chunks={letters[a]: 1 for a in numbers}, dtype=np.float64,
                              meta=np.empty((0,) * (data.ndim + 2), dtype=np.float64))
        blocks = math.prod(data.numblocks[a] for a in numbers)

        def combine(x, axis=None, keepdims=False, **_):  # one consumer of the pieces, so each chunk is read once
            return np.concatenate([x[:sums.TERMS].sum(axis=axis, keepdims=keepdims),
                                   x[sums.TERMS:].max(axis=axis, keepdims=keepdims)])

        combined = da.reduction(pieces, chunk=lambda x, **_: x, combine=combine, aggregate=combine,
                                axis=tuple(2 + a for a in numbers), dtype=np.float64, concatenate=True)

        def exact(chosen):
            def buckets(block, codes=None):
                out = sums.chunk_exact(block, numbers, chosen, codes=codes, groups=count)
                return out.reshape((len(chosen), *block_shape(block), sums.BUCKETS))

            stacked = da.blockwise(buckets, ("c", *letters, "b"), *sources,
                                   new_axes={"c": len(chosen), "b": sums.BUCKETS},
                                   adjust_chunks={letters[a]: 1 for a in numbers}, dtype=np.int64,
                                   meta=np.empty((0,) * (data.ndim + 2), dtype=np.int64))
            return stacked.sum(axis=tuple(1 + a for a in numbers), dtype=np.int64).compute()

        def finish(pieces):
            values = sums.from_chunks(pieces, blocks, op, exact, na_rm=na_rm, min_count=min_count)
            return values if groups is not None else values.reshape(kept_shape)

        # One block holding every group, so the last task settles them all; the result shares the source's
        # tasks with everything else collected, unlike a dask.delayed call, which would wrap them.
        whole = combined.rechunk({axis: -1 for axis in range(combined.ndim)})
        shape = (count, *kept_shape) if groups is not None else kept_shape
        return da.map_blocks(finish, whole, drop_axis=[0] if groups is not None else [0, 1],
                             chunks=tuple((n,) for n in shape), dtype=np.float32,
                             meta=np.empty((0,) * len(shape), dtype=np.float32))

    def group_reduce(self, array: xr.DataArray, op: str, groups: Any, others: Sequence[Hashable],
                     na_rm: bool, min_count: int | None, q: float | None = None) -> xr.DataArray:
        """*op* per group of *groups* (``tidy3a.expr.Groups``), over its axis and *others*."""
        array = _as_objects(array, op)
        if op in ("sum", "mean") and array.dtype == np.float32 and (array.chunks is None or _dask_backed(array)):
            if array.chunks is None:
                axes = (array.get_axis_num(groups.axis), *(array.get_axis_num(o) for o in others))
                values = sums.reduce(array.values, op, axes, na_rm=na_rm, min_count=min_count, codes=groups.codes,
                                     groups=len(groups), sizes=groups.counts())
            else:
                values = self._float32_chunked(array, op, [groups.axis, *others], na_rm, min_count, groups)
            template = self.reduce(array.isel({groups.axis: slice(0, 1)}), op, [groups.axis, *others], na_rm,
                                   min_count)
            coords = dict(template.coords.items())
            coords[groups.name] = label_index(groups.labels, groups.name)
            return xr.DataArray(values, dims=(groups.name, *template.dims), coords=coords, name=template.name,
                                attrs=template.attrs)
        fast = self._one_pass(array, op, groups, others, na_rm, min_count)
        if fast is not None:
            return fast
        stats = [
            self.reduce(part, op, [groups.axis, *others], na_rm, min_count, q)
            for part in self._group_parts(array, groups)
        ]
        index = label_index(groups.labels, groups.name)
        if stats:
            return xr.concat(stats, dim=index)
        empty = self.reduce(array.isel({groups.axis: slice(0, 0)}), op, [groups.axis, *others], na_rm, min_count, q)
        return empty.expand_dims({groups.name: 0}).assign_coords({groups.name: index})

    def _one_pass(self, array: xr.DataArray, op: str, groups: Any, others: Sequence[Hashable],
                  na_rm: bool, min_count: int | None) -> xr.DataArray | None:
        """Every group at once where that is exact (``labelarray.grouped``); the rest one by one.

        One group's result, reduced as before, gives the result's axes,
        coordinates, and name.
        """
        if array.chunks is not None or others or len(groups.labels) <= grouped.FEW_GROUPS:
            return None
        order, starts, ends = groups.blocks()
        axis = groups.axis
        fast = grouped.one_pass(array.values, array.get_axis_num(axis), op, na_rm, min_count, order, starts, ends)
        if fast is None:
            return None
        values, done = fast
        ordered = None
        first = array.get_axis_num(axis) == 0

        def one(group: int) -> xr.DataArray:
            nonlocal ordered
            ordered = array.isel({axis: order}) if ordered is None else ordered
            part = ordered.isel({axis: builtins.slice(int(starts[group]), int(ends[group]))})
            return self.reduce(part if first else part.copy(), op, [axis], na_rm, min_count)

        rest = np.flatnonzero(~done)
        template = one(int(rest[0]) if len(rest) else 0)
        if len(rest):
            stats = np.stack([template.values, *(one(int(g)).values for g in rest[1:])])
            if stats.dtype != values.dtype:
                return None
            values[rest] = stats
        if template.dtype != values.dtype:
            return None
        coords = {name: coord for name, coord in template.coords.items()}
        coords[groups.name] = label_index(groups.labels, groups.name)
        return xr.DataArray(values, dims=(groups.name, *template.dims), coords=coords, name=template.name,
                            attrs=template.attrs)

    def _group_parts(self, array: xr.DataArray, groups: Any) -> list[xr.DataArray]:
        """Each group's positions, in their original order.

        With many groups, in-memory arrays are reordered once so groups sit in
        contiguous blocks, instead of scanning the whole axis once per group.
        Each block holds the same values in the same order and layout as a
        per-group mask would, so reductions give bit-identical results. Chunked
        arrays keep per-group masks, which stay lazy.
        """
        codes = groups.codes
        count = len(groups.labels)
        if array.chunks is not None or count <= grouped.FEW_GROUPS:
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
        counts = groups.counts() * per_position
        return xr.DataArray(counts, dims=groups.name, coords={groups.name: groups.labels})

    def broadcast(self, stats: xr.DataArray, groups: Any) -> xr.DataArray:
        """A per-group result back on each grouped position."""
        picked = stats.isel({groups.name: groups.codes_array})
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
