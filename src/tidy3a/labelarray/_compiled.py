"""The compiled grouped estimate for ``sums``, used when Numba is installed (``tidy3a[fast]``).

One parallel pass over the values computes, per group and column, a
compensated float64 sum (Neumaier's improvement of Kahan's method), the sum
of magnitudes, and the values present. Each thread adds its share of the
positions; the threads' compensated sums are then added with compensation
too. The result is within (6u + 4 (n u)^2) * sum|x| of the exact sum, so
``sums._certify`` settles nearly every result even under heavy cancellation,
and the results are the same float32 values the NumPy path gives.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True, nogil=True)
def _estimate(table, rows, groups, na_rm, threads):  # pragma: no cover - compiled
    positions, width = table.shape
    share = (positions + threads - 1) // threads
    sums = np.zeros((threads, groups, width))
    carries = np.zeros((threads, groups, width))
    magnitudes = np.zeros((threads, groups, width))
    present = np.zeros((threads, groups, width), dtype=np.intp)
    for thread in nb.prange(threads):
        for position in range(thread * share, min(positions, (thread + 1) * share)):
            group = rows[position]
            for column in range(width):
                value = np.float64(table[position, column])
                if np.isnan(value):  # skipped with na_rm, otherwise it makes the sum NaN
                    if na_rm:
                        continue
                else:
                    present[thread, group, column] += 1
                total = sums[thread, group, column]
                added = total + value
                if abs(total) >= abs(value):
                    carries[thread, group, column] += (total - added) + value
                else:
                    carries[thread, group, column] += (value - added) + total
                sums[thread, group, column] = added
                magnitudes[thread, group, column] += abs(value)
    total = np.zeros((groups, width))
    carry = np.zeros((groups, width))
    magnitude = np.zeros((groups, width))
    counted = np.zeros((groups, width), dtype=np.intp)
    for thread in range(threads):
        for group in range(groups):
            for column in range(width):
                for value in (sums[thread, group, column], carries[thread, group, column]):
                    running = total[group, column]
                    added = running + value
                    if abs(running) >= abs(value):
                        carry[group, column] += (running - added) + value
                    else:
                        carry[group, column] += (value - added) + running
                    total[group, column] = added
                magnitude[group, column] += magnitudes[thread, group, column]
                counted[group, column] += present[thread, group, column]
    return total + carry, magnitude, counted


def grouped_estimate(table: np.ndarray, rows: np.ndarray, groups: int, na_rm: bool):
    """Compensated float64 sums, sums of magnitudes, and values present, per group and column."""
    # The thread count is passed in: reading it inside the compiled function would stop Numba caching it.
    return _estimate(table, rows, groups, na_rm, nb.get_num_threads())


@nb.njit(parallel=True, cache=True, nogil=True, error_model="numpy")
def _in_order(table, mean, na_rm):  # pragma: no cover - compiled
    positions, width = table.shape
    out = np.empty(width, dtype=np.float32)
    present = np.zeros(width, dtype=np.intp)
    for column in nb.prange(width):
        total = 0.0
        counted = 0
        for position in range(positions):
            value = np.float64(table[position, column])
            if np.isnan(value):
                if na_rm:
                    value = 0.0
            else:
                counted += 1
            total += value
        if mean:
            total = total / (counted if na_rm else positions)
        out[column] = np.float32(total)
        present[column] = counted
    return out, present


def in_order(table: np.ndarray, mean: bool, na_rm: bool):
    """Few values per result, as ``sums._in_order`` adds them, in one parallel pass.

    Each result adds the values of its column first to last in float64,
    starting from 0.0, with NaN as 0.0 under ``na_rm``; a mean divides in
    float64; the result is rounded once to float32. These are the operations
    and the order of the NumPy path, so the bits are the same.
    """
    return _in_order(table, mean, na_rm)
