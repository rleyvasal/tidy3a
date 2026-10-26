"""Grouped reductions in one pass over positions sorted by group (review priority 3).

Reducing groups one by one costs a Python step per group, which dominates
when groups are many and small (points binned into voxels). Here every group
is reduced at once wherever the result is bit-identical to reducing each
group on its own, as both backends do otherwise:

- min, max, count, any, and all do not depend on the order values are
  combined in, nor do integer and boolean sums; ``reduceat`` gives them for
  every group.
- Float sums and means depend on the order. NumPy adds fewer than 8 values
  one after another, starting from 0.0, in every memory layout; from 8 on it
  adds in blocks. Groups with fewer than 8 positions are therefore added
  position by position for all groups at once; larger groups are left to the
  caller, which reduces them one by one.

Other reductions (std, var, median), other types, groups with no positions,
and reductions over further axes are left to the caller.
``ONE_PASS = False`` turns it off, for comparison in tests.

``Grouping`` is the object the kernels work from: one integer code per
position and the group labels in code order, with the group sizes and the
positions sorted by group computed once and kept. ``reduce``, ``counts``,
and ``broadcast`` apply a grouping to an ``Array``: every group's result
along a new group axis, and a result per group back on the positions.
This module needs NumPy only.
"""

from __future__ import annotations

import builtins
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from tidy3a.labelarray import sums
from tidy3a.labelarray.array import Array, _freeze, _take, labels_from

ONE_PASS = True

# Up to this many groups, scanning the axis once per group is cheaper than sorting it.
FEW_GROUPS = 8


class Grouping:
    """Positions assigned to groups: one code per position, and each group's label.

    Codes are integers from 0 to ``len(labels) - 1``; ``labels[code]`` is the
    group's label. ``counts()`` and ``blocks()`` are computed on first use and
    kept, so a sort by group serves every reduction of the same grouping.
    """

    def __init__(self, codes: Any, labels: Sequence[Any]):
        self.codes = np.asarray(codes)
        self.labels = list(labels)
        self._counts: np.ndarray | None = None
        self._blocks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> np.ndarray:
        """Positions in each group, in code order."""
        if self._counts is None:
            self._counts = np.bincount(self.codes, minlength=len(self.labels))
        return self._counts

    def blocks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions sorted by group, keeping their order within a group, and each group's start and end."""
        if self._blocks is None:
            # NumPy's stable sort is a radix sort for 8- and 16-bit integers, far faster than for int64.
            narrow = np.min_scalar_type(max(len(self.labels) - 1, 0))
            order = np.argsort(self.codes.astype(narrow, copy=False), kind="stable")
            ends = np.cumsum(self.counts())
            self._blocks = (order, np.concatenate(([0], ends[:-1])), ends)
        return self._blocks

# NumPy adds fewer than this many values one after another; more, in blocks (pairwise summation).
SMALL = 8

_ORDER_FREE = {"min", "max", "count", "any", "all"}


def one_pass(data: np.ndarray, axis: int, op: str, na_rm: bool, min_count: int | None,
             order: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """*op* per group along *axis*, with the group axis first, and which groups it gives.

    Groups are ``order[starts[g]:ends[g]]``. Rows of groups not given are
    left unset for the caller. None when no group can be reduced here.
    """
    if not ONE_PASS or not isinstance(data, np.ndarray):
        return None
    kind = data.dtype.kind
    if kind not in "biuf" or (kind == "f" and data.dtype.itemsize < 4) or op not in (*_ORDER_FREE, "sum", "mean"):
        return None
    sizes = ends - starts
    if not len(sizes) or not sizes.all():
        return None  # groups with no positions (empty bins) are reduced one by one, as before
    any_size = op in _ORDER_FREE or (op == "sum" and kind in "biu")
    done = np.ones(len(sizes), dtype=bool) if any_size else sizes < SMALL
    if not done.any():
        return None
    ordered = np.moveaxis(np.take(data, order, axis=axis), axis, 0)
    missing = np.isnan(ordered) if kind == "f" else None
    with np.errstate(all="ignore"):
        values = _reduce(ordered, missing, kind, op, na_rm, min_count, starts, sizes, done)
    if done.all():
        return values, done
    result = np.empty((len(sizes), *values.shape[1:]), dtype=values.dtype)
    result[done] = values
    return result, done


def _reduce(ordered, missing, kind, op, na_rm, min_count, starts, sizes, done):
    """The reduction for the groups in *done*, matching ``Array.reduce`` group by group."""
    if op in ("min", "max"):
        skip = na_rm and kind == "f"  # nanmin and nanmax ignore NaN; fmin and fmax do too
        ufunc = {("min", False): np.minimum, ("min", True): np.fmin,
                 ("max", False): np.maximum, ("max", True): np.fmax}[op, skip]
        return ufunc.reduceat(ordered, starts, axis=0)
    present = ~missing if missing is not None else None
    if op == "count":
        present = present if present is not None else np.ones(ordered.shape, dtype=bool)
        return np.add.reduceat(present, starts, axis=0, dtype=_sum_type(present))
    if op in ("any", "all"):
        filled = ordered if missing is None else np.where(missing, op == "all", ordered)
        ufunc = np.logical_or if op == "any" else np.logical_and
        result = ufunc.reduceat(filled.astype(bool, copy=False), starts, axis=0)
        if na_rm or kind in "biu" or missing is None:
            return result
        return np.where(~np.logical_or.reduceat(missing, starts, axis=0), result, np.nan)
    skip = na_rm and kind == "f"
    source = np.where(missing, 0.0, ordered) if skip else ordered  # nansum and nanmean count NaN as 0
    if op == "sum":
        if kind in "biu":
            total = np.add.reduceat(source, starts, axis=0, dtype=_sum_type(source))
        else:
            total = _left_to_right(source, starts[done], sizes[done])
        if min_count is None:
            return total
        counts = _counts(present, ordered.shape, starts, sizes)[done]
        return np.where(counts >= min_count, total, np.nan)
    # mean: integers and booleans are added as float64, as np.mean adds them
    total = _left_to_right(source if kind == "f" else source.astype(np.float64), starts[done], sizes[done])
    counts = _counts(present if skip else None, ordered.shape, starts, sizes)[done]
    # np.mean and np.nanmean divide in float64 and store in the sum's type.
    return (total.astype(np.float64) / counts).astype(total.dtype, copy=False)


def _left_to_right(values: np.ndarray, first: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Each group's values added one after another, for groups of fewer than ``SMALL`` positions."""
    total = values[first] + 0.0  # NumPy's sums start from 0.0, so a group of -0.0 sums to 0.0
    for rank in range(1, int(sizes.max())):
        active = np.flatnonzero(sizes > rank)
        total[active] += values[first[active] + rank]
    return total


def _counts(present, shape, starts, sizes) -> np.ndarray:
    """Present values per group (every position when *present* is None), shaped like one group's result."""
    if present is None:
        return sizes.reshape(-1, *([1] * (len(shape) - 1)))
    return np.add.reduceat(present, starts, axis=0, dtype=np.intp)


def _sum_type(values: np.ndarray) -> np.dtype:
    """The type ``np.sum`` gives: small integers and booleans add as the platform integer."""
    return np.sum(values[:0], axis=0).dtype


# ── Groupings applied to arrays ─────────────────────────────────────────────


def reduce(array: Array, op: str, grouping: Grouping, axis: Hashable, name: Hashable,
           others: Sequence[Hashable] = (), *, na_rm: bool = False, min_count: int | None = None) -> Array:
    """*op* per group of positions on *axis*, and over *others*: the groups along a new first axis *name*.

    Each group's result is bit-identical to reducing that group on its own
    (``Array.reduce`` over a mask), whichever path computes it.
    """
    if op in ("sum", "mean") and array.data.dtype == np.float32:
        axes = (array.axis(axis), *(array.axis(o) for o in others))
        values = sums.reduce(array.data, op, axes, na_rm=na_rm, min_count=min_count, codes=grouping.codes,
                             groups=len(grouping), sizes=grouping.counts())
        dims = tuple(d for d in array.dims if d != axis and d not in others)
        coords = {d: c for d, c in array.coords.items() if d in dims}
        return Array._make(values, (name, *dims), {name: labels_from(grouping.labels), **coords})
    fast = _one_pass_array(array, op, grouping, axis, name, others, na_rm, min_count)
    if fast is not None:
        return fast
    stats = [part.reduce(op, [axis, *others], na_rm=na_rm, min_count=min_count)
             for part in _group_parts(array, grouping, axis)]
    labels = labels_from(grouping.labels)
    if stats:
        data = np.stack([s.data for s in stats])
        return Array._make(data, (name, *stats[0].dims), {name: labels, **stats[0].coords})
    empty = _take(array, axis, builtins.slice(0, 0)).reduce(op, [axis, *others], na_rm=na_rm, min_count=min_count)
    return Array._make(empty.data[np.newaxis][:0], (name, *empty.dims), {name: labels, **empty.coords})


def counts(grouping: Grouping, name: Hashable, per_position: int = 1) -> Array:
    """Values per group: positions times *per_position*, along the axis *name*."""
    # Plain NumPy labels, not objects: xarray labels counts this way.
    return Array._make(grouping.counts() * per_position, (name,), {name: _freeze(np.asarray(grouping.labels))})


def broadcast(stats: Array, grouping: Grouping, axis: Hashable, name: Hashable,
              labels: np.ndarray | None = None) -> Array:
    """A result per group (along *name*) back on each position of *axis*, labelled *labels* if given."""
    number = stats.axis(name)
    # Group results come with the group axis first (reduce). Indexing gathers them
    # several times faster than np.take, into the same values and the same C-ordered layout.
    data = stats.data[grouping.codes] if number == 0 else np.take(stats.data, grouping.codes, axis=number)
    dims = stats.dims[:number] + (axis,) + stats.dims[number + 1:]
    coords = {d: c for d, c in stats.coords.items() if d != name}
    if labels is not None:
        coords[axis] = labels
    return Array._make(data, dims, coords)


def _one_pass_array(array: Array, op: str, grouping: Grouping, axis: Hashable, name: Hashable,
                    others: Sequence[Hashable], na_rm: bool, min_count: int | None) -> Array | None:
    """Every group at once where that is exact (``one_pass``); the rest one by one."""
    if others or len(grouping) <= FEW_GROUPS:
        return None
    order, starts, ends = grouping.blocks()
    fast = one_pass(array.data, array.axis(axis), op, na_rm, min_count, order, starts, ends)
    if fast is None:
        return None
    values, done = fast
    rest = np.flatnonzero(~done)
    if len(rest):
        stats = np.stack([_take(array, axis, order[int(starts[g]):int(ends[g])])
                          .reduce(op, [axis], na_rm=na_rm, min_count=min_count).data for g in rest])
        if stats.dtype != values.dtype:
            return None
        values[rest] = stats
    dims = tuple(d for d in array.dims if d != axis)
    coords = {d: c for d, c in array.coords.items() if d in dims}
    return Array._make(values, (name, *dims), {name: labels_from(grouping.labels), **coords})


def _group_parts(array: Array, grouping: Grouping, axis: Hashable) -> list[Array]:
    """Each group's positions, in their original order and in a mask's memory layout.

    Indexing an axis other than the first with positions or a mask lays
    the result out with that axis first in memory; reductions over it then
    add in another order than over a contiguous copy, which changes float
    results in the last bit. Each group is therefore indexed by its
    positions, as a mask would be, so results are bit-identical to xarray
    and to per-group masks. With many groups, the positions come from one
    sort instead of one scan of the axis per group.
    """
    count = len(grouping)
    if count <= FEW_GROUPS:
        return [_take(array, axis, np.flatnonzero(grouping.codes == code)) for code in range(count)]
    order, starts, ends = grouping.blocks()
    return [_take(array, axis, order[int(start):int(end)]) for start, end in zip(starts, ends)]
