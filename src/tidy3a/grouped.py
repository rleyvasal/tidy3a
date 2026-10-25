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
and reductions over further axes are left to the caller. This module needs
NumPy only. ``ONE_PASS = False`` turns it off, for comparison in tests.
"""

from __future__ import annotations

import numpy as np

ONE_PASS = True

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
    """The reduction for the groups in *done*, matching ``numpy_backend._reduce_data`` group by group."""
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
