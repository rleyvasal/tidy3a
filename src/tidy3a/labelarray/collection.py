"""A collection of labelled arrays sharing axes (plan milestone 10, track A, step A5).

A ``Collection`` holds variables with different axes; each axis has one size
and at most one label array, shared by every variable on it. Selection,
assignment, renaming, joins under the exact label rule, and nearest-label
matching are methods. Conversion to and from xarray needs xarray, imported
only when asked for.

This module needs NumPy only.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from tidy3a.labelarray import labels as exact
from tidy3a.labelarray.array import (
    Array,
    _checked_coords,
    _cut,
    _freeze,
    _take,
    label_index,
    plain,
    positions_of,
)
from tidy3a.labelarray.errors import ShapeError


class Collection:
    """Variables with different axes; each axis has one size and at most one label array."""

    __slots__ = ("attrs", "coord_attrs", "coords", "sizes", "variables")

    def __init__(self, variables: Mapping[Hashable, Array], coords: Mapping[Hashable, Any] | None = None,
                 attrs: Mapping[Hashable, Any] | None = None, coord_attrs: Mapping[Hashable, Any] | None = None):
        sizes: dict[Hashable, int] = {}
        for name, value in variables.items():
            for dim, size in zip(value.dims, value.data.shape):
                if sizes.setdefault(dim, size) != size:
                    raise ShapeError(f"{name}: axis {dim} has {size} positions, other variables have {sizes[dim]}")
        coords = dict(coords or {})
        for dim, labels in coords.items():
            sizes.setdefault(dim, len(labels))
        self.variables = {name: Array._make(v.data, v.dims, {}, v.attrs, name) for name, v in variables.items()}
        self.coords = _checked_coords(coords, sizes)
        self.coord_attrs = {d: dict(a) for d, a in (coord_attrs or {}).items() if d in self.coords}
        self.attrs = dict(attrs or {})
        self.sizes = sizes

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self.sizes)

    def array(self, name: Hashable) -> Array:
        """One variable, with the labels of its axes."""
        value = self.variables[name]
        return Array._make(value.data, value.dims, {d: self.coords[d] for d in value.dims if d in self.coords},
                           value.attrs, name)

    def evolve(self, variables: Mapping | None = None, coords: Mapping | None = None, **changes: Any) -> Collection:
        """A copy with parts replaced (``variables``, ``coords``, ``attrs``, ``coord_attrs``)."""
        return Collection(
            self.variables if variables is None else variables,
            self.coords if coords is None else coords,
            changes.get("attrs", self.attrs), changes.get("coord_attrs", self.coord_attrs),
        )

    # ── Selection ───────────────────────────────────────────────────────────

    def isel(self, indexers: Mapping[Hashable, Any] | None = None, **more: Any) -> Collection:
        """Positions, a slice, or a mask per axis, on every variable with that axis; an integer removes it."""
        variables, coords = dict(self.variables), dict(self.coords)
        for dim, index in {**(indexers or {}), **more}.items():
            if getattr(index, "dtype", None) == bool:
                index = np.flatnonzero(index)  # scan the mask once; every variable reuses the positions
            for name, value in variables.items():
                if dim in value.dims:
                    variables[name] = _take(value, dim, index)
            if dim in coords:
                if isinstance(index, (int, np.integer)):
                    del coords[dim]
                else:
                    coords[dim] = _cut(coords[dim], index)
        return self.evolve(variables, coords)

    def sel(self, indexers: Mapping[Hashable, Any] | None = None, **more: Any) -> Collection:
        """Labels per axis, looked up exactly; one label removes the axis, a list keeps it."""
        wanted = {**(indexers or {}), **more}
        return self.isel({dim: positions_of(self.coords[dim], labels, dim) for dim, labels in wanted.items()})

    def select(self, names: Sequence[Hashable]) -> Collection:
        """Only the variables *names*, with the labels of the axes they use."""
        if not names:
            return self.evolve(variables={})
        variables = {name: self.variables[name] for name in names}
        used = {d for v in variables.values() for d in v.dims}
        return self.evolve(variables, {d: c for d, c in self.coords.items() if d in used})

    # ── Building ────────────────────────────────────────────────────────────

    def assign(self, name: Hashable, value: Any) -> Collection:
        """*value* as the variable *name*; its labels are added for axes that have none yet."""
        if not isinstance(value, Array):
            value = Array(value)
        variables = dict(self.variables)
        variables[name] = value
        coords = dict(self.coords)
        for dim, labels in value.coords.items():
            coords.setdefault(dim, labels)
        return self.evolve(variables, coords)

    def rename(self, names: Mapping[Hashable, Hashable]) -> Collection:
        """Variables renamed."""
        return self.evolve({names.get(n, n): v for n, v in self.variables.items()})

    # ── Joins ───────────────────────────────────────────────────────────────

    def join(self, other: Collection, how: str) -> Collection:
        """Both collections' variables, on shared axes' labels joined *how* (left, inner, or outer).

        Labels are paired under the exact rule; positions a side lacks are
        missing, with integers and booleans promoted as xarray promotes them.
        Attributes that conflict are dropped, as xarray's ``drop_conflicts``.
        """
        left, right = self, other
        for dim in [d for d in left.coords if d in right.coords]:
            ours, theirs = left.coords[dim], right.coords[dim]
            if exact.equal(ours, theirs):
                continue
            target = _joined_labels(ours, theirs, how, dim)
            left = left._reindexed(dim, label_index(ours, "join", dim).get_indexer(target), target)
            right = right._reindexed(dim, label_index(theirs, "join", dim).get_indexer(target), target)
        variables = {**{n: left.array(n) for n in left.variables}, **{n: right.array(n) for n in right.variables}}
        coords = {**right.coords, **left.coords}
        coord_attrs = {d: _combine_attrs(left.coord_attrs.get(d, {}), right.coord_attrs.get(d, {})) for d in coords}
        return Collection(variables, coords, _combine_attrs(left.attrs, right.attrs), coord_attrs)

    def nearest(self, dim: Hashable, labels: np.ndarray, tolerance: Any) -> Collection:
        """This collection on the labels *labels* for *dim*, each matched to its nearest label within *tolerance*.

        A label with none near enough gives missing values; a tie goes to the lower label.
        """
        theirs = self.coords[dim]
        order = np.argsort(theirs, kind="stable")
        ordered = theirs[order]
        upper = np.clip(np.searchsorted(ordered, labels), 0, len(ordered) - 1)
        lower = np.clip(upper - 1, 0, len(ordered) - 1)
        pick_lower = np.abs(labels - ordered[lower]) <= np.abs(ordered[upper] - labels)
        nearest = np.where(pick_lower, lower, upper)
        positions = np.where(np.abs(ordered[nearest] - labels) <= tolerance, order[nearest], -1)
        return self._reindexed(dim, positions, labels)

    def _reindexed(self, dim: Hashable, positions: np.ndarray, labels: np.ndarray) -> Collection:
        variables = {}
        for name in self.variables:
            value = self.array(name)
            variables[name] = _reindex(value, dim, positions, labels) if dim in value.dims else value
        coords = dict(self.coords)
        coords[dim] = labels
        return self.evolve(variables, coords)

    # ── Conversion ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """As xarray's ``Dataset.to_dict(data="array")`` gives it, without xarray."""
        return {
            "coords": {d: {"dims": (d,), "attrs": dict(self.coord_attrs.get(d, {})), "data": plain(labels)}
                       for d, labels in self.coords.items()},
            "attrs": dict(self.attrs),
            "dims": dict(self.sizes),
            "data_vars": {n: {"dims": v.dims, "attrs": dict(v.attrs), "data": v.data} for n, v in self.variables.items()},
        }

    def to_xarray(self) -> Any:
        """An xarray ``Dataset`` sharing the values; needs xarray."""
        import xarray as xr

        coords = {d: (d, plain(labels), dict(self.coord_attrs.get(d, {}))) for d, labels in self.coords.items()}
        return xr.Dataset({n: (v.dims, v.data, dict(v.attrs)) for n, v in self.variables.items()},
                          coords=coords, attrs=dict(self.attrs))

    @classmethod
    def from_xarray(cls, dataset: Any) -> Collection:
        """A collection from an in-memory xarray ``Dataset`` whose coordinates are its axes' labels."""
        variables = {name: Array._make(np.asarray(value.data), tuple(value.dims), {}, dict(value.attrs), name)
                     for name, value in dataset.data_vars.items()}
        coords = {name: dataset[name].values for name in dataset.coords}
        coord_attrs = {name: dict(dataset[name].attrs) for name in dataset.coords}
        return cls(variables, coords, dict(dataset.attrs), coord_attrs)


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


def _reindex(value: Array, dim: Hashable, positions: np.ndarray, labels: np.ndarray) -> Array:
    """*value* on new labels for *dim*: position -1 becomes missing."""
    axis = value.axis(dim)
    missing = positions < 0
    if not value.data.shape[axis]:  # nothing to take from: every position is missing
        dtype, fill = _fill_for(value.data.dtype) if len(positions) else (value.data.dtype, None)
        shape = value.data.shape[:axis] + (len(positions),) + value.data.shape[axis + 1:]
        taken = np.full(shape, fill, dtype=dtype) if len(positions) else np.empty(shape, dtype=dtype)
        coords = dict(value.coords)
        coords[dim] = labels
        return Array._make(taken, value.dims, coords, value.attrs, value.name)
    taken = np.take(value.data, np.where(missing, 0, positions), axis=axis)
    if missing.any():
        dtype, fill = _fill_for(value.data.dtype)
        taken = taken.astype(dtype, copy=True) if taken.dtype != dtype else taken.copy()
        index = (slice(None),) * axis + (missing,)
        taken[index] = fill
    coords = dict(value.coords)
    coords[dim] = labels
    return Array._make(taken, value.dims, coords, value.attrs, value.name)


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
