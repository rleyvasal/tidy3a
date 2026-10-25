"""Float32 sums and means: float64 accumulation, rounded once to float32.

Two ways, chosen by how many values each result adds:

- Every grouped reduction, and every ungrouped one adding ``FEW`` (8) or more
  values per result: the result is correctly rounded, the float32 nearest
  the exact sum or mean of the stored values, ties to even. A float64 sum
  and an error bound settle almost every result; where the bound cannot,
  the values are added exactly (``_exact``). A correctly rounded result is
  unique, so every memory layout, summation order, kernel (NumPy or the
  optional compiled one), thread count, and backend gives the same bits.
- Ungrouped reductions adding fewer than ``FEW`` values per result, such as
  ``sum(p**2, over="i")`` over x, y, and z: the values are added in float64
  in a fixed order, first to last along the reduced axes, and rounded once.
  The order does not depend on memory layout; the compiled kernel, when
  installed, adds in the same order in one pass. Checking each of these results
  would cost several times the sum itself (the float64 sum of a few float32
  values is often exact and exactly halfway between two float32 values, where
  a bound cannot decide), so cancellation among those few values is rounded
  in float64 as it goes: ``[2**53, 1, -2**53]`` sums to 0 here, not 1.

Missing values: without ``na_rm`` a NaN makes its result NaN; with it, NaN
is skipped and a mean divides by the values present. +inf and -inf together
give NaN. A sum of no values is +0.0 and a mean NaN; a zero sum is +0.0, as
NumPy gives it. ``min_count`` makes a sum NaN where fewer values are present.
This module needs NumPy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

FEW = 8
COMPILED = True  # use the optional compiled kernels when Numba is installed

_U = 2.0**-53  # float64 unit roundoff
# Sums are made in chunks of positions, then the chunk totals added, so a result's error bound grows
# with the chunk length plus the number of chunks rather than with every position: about sqrt(n) chunks
# of sqrt(n) positions, with at most this many (chunk, group) bins at once.
_BINS = 1 << 22
_MAX32 = Fraction(float(np.finfo(np.float32).max))


def reduce(values: np.ndarray, op: str, axes: Sequence[int], *, na_rm: bool = False,
           min_count: int | None = None, codes: np.ndarray | None = None, groups: int = 0,
           sizes: np.ndarray | None = None) -> np.ndarray:
    """*op* (``"sum"`` or ``"mean"``) of float32 *values* over the axis numbers *axes*.

    Without *codes*, the result has the kept axes in their order. With
    *codes* (one group code from 0 to ``groups - 1`` per position along
    ``axes[0]``), it has one row per group first: ``(groups, *kept)``;
    *sizes*, positions per group along ``axes[0]``, may be passed when known.
    """
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError(f"float32 values expected, got {values.dtype}")
    axes = tuple(int(a) for a in axes)
    kept = tuple(a for a in range(values.ndim) if a not in axes)
    kept_shape = tuple(values.shape[a] for a in kept)
    positions = math.prod(values.shape[a] for a in axes)
    with np.errstate(all="ignore"):
        if codes is None and positions < FEW:
            compiled = _compiled("in_order") if COMPILED and len(axes) == 1 and values.ndim <= 2 else None
            if compiled is not None:
                # A view: one reduced axis, at most one kept; the width is explicit, since -1 is ambiguous when empty.
                table = np.moveaxis(values, axes[0], 0).reshape(positions, math.prod(kept_shape))
                result, present = compiled(table, op == "mean", na_rm)
                if op == "sum" and min_count is not None:
                    result = np.where(present >= min_count, result, np.float32(np.nan))
                return result.reshape(kept_shape)
            return _in_order(values, axes, kept_shape, op, na_rm, min_count)
        table = np.transpose(values, axes + kept).reshape(positions, math.prod(kept_shape))
        if codes is not None:
            others = math.prod(values.shape[a] for a in axes[1:])
            rows = np.asarray(codes) if others == 1 else np.repeat(np.asarray(codes), others)
            if sizes is not None and others != 1:
                sizes = np.asarray(sizes) * others
            result = _settled(table, rows, groups, sizes, op, na_rm, min_count)
            return result.reshape((groups, *kept_shape))
        return _settled(table, None, 1, None, op, na_rm, min_count).reshape(kept_shape)


def _in_order(values, axes, kept_shape, op, na_rm, min_count) -> np.ndarray:
    """Few values per result: float64, added first to last along *axes*, rounded once.

    Each position along the reduced axes is a view of the input, added in
    place, so no copy is made whatever the layout; missing values are
    counted only when ``na_rm`` or ``min_count`` needs them.
    """
    moved = np.moveaxis(values, axes, tuple(range(len(axes))))
    total = np.zeros(kept_shape)
    counting = na_rm or (op == "sum" and min_count is not None)
    present = np.zeros(kept_shape, dtype=np.intp) if counting else None
    for index in np.ndindex(moved.shape[:len(axes)]):
        row = moved[index]
        if counting:
            missing = np.isnan(row)
            present += ~missing
            if na_rm:
                row = np.where(missing, np.float32(0), row)
        np.add(total, row, out=total)
    terms = math.prod(values.shape[a] for a in axes)
    return _finish(total, present, terms, op, na_rm, min_count)


def _finish(total, present, terms, op, na_rm, min_count) -> np.ndarray:
    if op == "mean":
        np.divide(total, present if na_rm else terms, out=total)
        return total.astype(np.float32)
    result = total.astype(np.float32)
    return np.where(present >= min_count, result, np.float32(np.nan)) if min_count is not None else result


# ── Correctly rounded: an estimate, a bound, and exact sums where needed ────


def _settled(table, rows, groups, sizes, op, na_rm, min_count) -> np.ndarray:
    """Correctly rounded results, one row per group (one row in all when *rows* is None)."""
    total, magnitude, present, values, terms, compensated = _estimate(table, rows, groups, sizes, na_rm)
    counts = present if na_rm else values
    result, sure = _certify(total, magnitude, terms, counts, op, compensated)
    unsure = ~sure
    for column in np.flatnonzero(unsure.any(axis=0)):
        chosen = np.flatnonzero(unsure[:, column])
        result[chosen, column] = _exact(table[:, column], rows, chosen, groups, op, na_rm)
    if op == "sum" and min_count is not None:
        result = np.where(present >= min_count, result, np.float32(np.nan))
    return result


def _estimate(table, rows, groups, sizes, na_rm):
    """Float64 sums, sums of magnitudes, values present, values, and terms for the bound, per group and column.

    The NumPy path adds chunks of positions (one ``np.bincount`` over
    (chunk, group) bins, or ``np.add.reduceat`` without groups) and then the
    chunk totals, so a result's error grows with the chunk length plus the
    number of chunks; "terms" is that effective count, for the bound only.
    """
    positions, width = table.shape
    chunks = max(1, min(math.isqrt(positions), _BINS // max(groups, 1)))
    length = -(-positions // chunks) if positions else 0
    if rows is None:
        missing = np.isnan(table)
        weights = np.where(missing, np.float32(0), table) if na_rm else table
        if positions:
            starts = np.arange(0, positions, length)
            total = np.add.reduceat(weights, starts, axis=0, dtype=np.float64).sum(axis=0)[None]
            magnitude = np.add.reduceat(np.abs(weights), starts, axis=0, dtype=np.float64).sum(axis=0)[None]
        else:
            total = magnitude = np.zeros((1, width))
        present = (positions - missing.sum(axis=0))[None]
        terms = np.full((1, width), min(positions, length + chunks))
        return total, magnitude, present, np.full((1, width), positions), terms, False
    if sizes is None:
        sizes = np.bincount(rows, minlength=groups)
    sizes = np.asarray(sizes)
    kernel = _compiled("grouped_estimate") if COMPILED else None
    if kernel is not None:
        total, magnitude, present = kernel(table, np.ascontiguousarray(rows, dtype=np.intp), groups, na_rm)
        values = np.repeat(sizes[:, None], width, axis=1)
        return total, magnitude, present, values, values, True
    effective = np.minimum(sizes, length) + np.minimum(sizes, chunks)  # additions within a chunk, then of chunk totals
    terms = np.repeat(np.minimum(sizes, effective)[:, None], width, axis=1)
    key = rows if chunks == 1 else (np.arange(positions) // length) * groups + rows
    bins = chunks * groups
    total = np.empty((groups, width))
    magnitude = np.empty((groups, width))
    present = np.empty((groups, width), dtype=np.intp)
    for column in range(width):
        values = table[:, column]
        missing = np.isnan(values)
        anything_missing = bool(missing.any())
        weights = np.where(missing, np.float32(0), values) if na_rm and anything_missing else values
        total[:, column] = np.bincount(key, weights=weights, minlength=bins).reshape(chunks, groups).sum(axis=0)
        magnitude[:, column] = np.bincount(key, weights=np.abs(weights), minlength=bins).reshape(chunks, groups).sum(axis=0)
        present[:, column] = (sizes - np.bincount(rows, weights=missing, minlength=groups).astype(np.intp)
                              if anything_missing else sizes)
    return total, magnitude, present, np.repeat(sizes[:, None], width, axis=1), terms, False


def _certify(total, magnitude, terms, counts, op, compensated):
    """Results where the error bound proves the rounding, and where it does.

    A float64 sum of n values, in any order, is within gamma(n) * sum|x| of
    the exact sum, gamma(n) = n u / (1 - n u); a compensated one within
    (6u + 4 (n u)^2) * sum|x|. Where both ends of the interval round to the
    same float32, the exact value rounds there too.
    """
    n = terms.astype(np.float64)
    gamma = n * _U / (1 - n * _U)
    size = magnitude / (1 - gamma)  # the computed magnitude may itself be low by that factor
    bound = (6 * _U + 4 * (n * _U) ** 2) * size if compensated else gamma * size
    bound = bound * (1 + 8 * _U) + np.abs(total) * (2 * _U) + 1e-300  # the bound's and the ends' own rounding
    if op == "mean":
        centre = total / counts
        bound = bound / counts * (1 + 4 * _U) + np.abs(centre) * (4 * _U)
    else:
        centre = total
    low, high = (centre - bound).astype(np.float32), (centre + bound).astype(np.float32)
    sure = (low == high) & np.isfinite(bound) & np.isfinite(centre)
    if op == "mean":
        sure &= counts > 0
    return high, sure


def _exact(values, rows, chosen, groups, op, na_rm) -> np.ndarray:
    """Correctly rounded results for groups *chosen*, from exact sums of their values."""
    if rows is None:
        part, codes, count = values, np.zeros(len(values), dtype=np.intp), 1
    else:
        remap = np.full(groups, -1, dtype=np.intp)
        remap[chosen] = np.arange(len(chosen))
        codes = remap[rows]
        keep = codes >= 0
        part, codes, count = values[keep], codes[keep], len(chosen)
    totals, present, special = _exact_sums(part, codes, count, na_rm)
    out = np.empty(count, dtype=np.float32)
    for k in range(count):
        if special[k] is not None:
            out[k] = special[k]
        elif op == "sum":
            out[k] = _round32(totals[k])
        else:
            out[k] = _round32(totals[k] / present[k]) if present[k] else np.float32(np.nan)
    if op == "mean":
        out[(present == 0) & np.array([s is None for s in special])] = np.nan
    return out


def _exact_sums(values, codes, count, na_rm):
    """Exact sums as Fractions, values present, and NaN or infinity where they decide the result.

    A float32 is M * 2**(e - 150) with M a signed integer below 2**24 and e
    from 1 to 254. Integers M with the same e add exactly in float64 for
    fewer than 2**29 values, so they are summed per group and exponent with
    ``np.bincount``, and the few totals per group combined in Python integers.
    """
    bits = values.view(np.uint32)
    exponent = ((bits >> 23) & 0xFF).astype(np.intp)
    nan = np.isnan(values)
    infinite = np.isinf(values)
    finite = ~(nan | infinite)
    mantissa = (bits & 0x7FFFFF).astype(np.int64) | np.where(exponent > 0, 1 << 23, 0)
    mantissa = np.where(bits >> 31, -mantissa, mantissa)
    exponent = np.maximum(exponent, 1)
    present = np.bincount(codes, minlength=count) - np.bincount(codes, weights=nan, minlength=count).astype(np.intp)
    has_nan = np.bincount(codes, weights=nan, minlength=count) > 0
    plus = np.bincount(codes, weights=values == np.inf, minlength=count) > 0
    minus = np.bincount(codes, weights=values == -np.inf, minlength=count) > 0
    totals = [Fraction(0)] * count
    if finite.any():
        low, high = int(exponent[finite].min()), int(exponent[finite].max())
        width = high - low + 1
        key = codes[finite] * width + (exponent[finite] - low)
        weights = mantissa[finite].astype(np.float64)
        buckets = np.zeros(count * width, dtype=np.int64)
        chunk = 1 << 28  # each bucket sum stays below 2**52, so float64 holds it exactly
        for start in range(0, len(key), chunk):
            buckets += np.bincount(key[start:start + chunk], weights=weights[start:start + chunk],
                                   minlength=count * width).astype(np.int64)
        buckets = buckets.reshape(count, width)
        scale = Fraction(2) ** (low - 150)
        for k in range(count):
            row = buckets[k]
            whole = sum(int(row[j]) << int(j) for j in np.flatnonzero(row))  # Python integers: shifts beyond 63
            totals[k] = whole * scale
    special = [None] * count
    for k in range(count):
        if (has_nan[k] and not na_rm) or (plus[k] and minus[k]):
            special[k] = np.float32(np.nan)
        elif plus[k] or minus[k]:
            special[k] = np.float32(np.inf if plus[k] else -np.inf)
    return totals, present, special


def _round32(value: Fraction) -> np.float32:
    """*value* correctly rounded to float32, ties to even."""
    if value == 0:
        return np.float32(0.0)
    if abs(value) >= _MAX32 + Fraction(2) ** 103:  # half a step above the largest float32
        return np.float32(math.inf if value > 0 else -math.inf)
    guess = np.float32(float(value))  # float() rounds correctly; the cast can round a second time
    best = guess
    for candidate in (np.nextafter(guess, np.float32(-np.inf)), np.nextafter(guess, np.float32(np.inf))):
        if not np.isfinite(candidate):
            continue
        gap, best_gap = abs(Fraction(float(candidate)) - value), abs(Fraction(float(best)) - value)
        if gap < best_gap or (gap == best_gap and int(candidate.view(np.uint32)) % 2 == 0):
            best = candidate
    return np.float32(best)


# ── The optional compiled kernel ────────────────────────────────────────────

_KERNELS: dict = {}


def _compiled(name: str):
    """A compiled kernel from ``_compiled`` when Numba is installed (``tidy3a[fast]``), else None."""
    if name not in _KERNELS:
        try:
            from tidy3a.labelarray import _compiled as module
        except ImportError:
            _KERNELS[name] = None
        else:
            _KERNELS[name] = getattr(module, name)
    return _KERNELS[name]
