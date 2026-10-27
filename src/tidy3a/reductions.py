"""Reductions on blocks: each one's partial, merge, and finish, keyed by its name (plan milestone 13).

A frame read in blocks summarises each block into pieces per group and
merges them as the blocks arrive (``tidy3a.chunked``). What a reduction's
pieces are, how two of them merge, and what they give at the end is the
reduction's own business, so each is one entry in ``REGISTRY``:

- ``partial(node, values, codes, groups)``: one block's pieces per group, as
  ``(kind, array)`` with the groups along the array's last axis; the kind
  says which of the entry's forms they take (a float32 sum's estimate, a
  float64 sum, an integer sum);
- ``merge(node, kind, held, piece)``: two groups' pieces as one, in any
  order;
- ``fill(node, kind, dtype)``: a group's pieces before any block has had it;
- ``finish(node, kind, held, counts, blocks)``: each group's answer, and
  the groups it cannot settle from the pieces (or None);
- for a reduction with a second pass, ``exact`` (one block's part of it for
  the groups chosen) and ``exact_finish``, and ``close``, applied once the
  second pass has filled its groups in.

``n()`` is the table's own count of positions per group; its entry has no
pieces. The executor looks reductions up by name and knows none of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from labelarray import sums

from tidy3a.errors import BlockError

WHOLE_AXIS = "read the data with lazy=True, which memory-maps a file, or load it, to run this on every position"
_UNSEEN = object()  # an object accumulator's slot for a group no block has had yet


class Reduction:
    """One reduction's steps on blocks; the built-ins below, each written once."""

    name = ""
    counted = False  # n(): the table counts positions itself, and there are no pieces

    def partial(self, node: Any, value: np.ndarray, codes: np.ndarray, count: int) -> tuple[str, np.ndarray]:
        raise NotImplementedError

    def merge(self, node: Any, kind: str, held: np.ndarray, piece: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fill(self, node: Any, kind: str, dtype: np.dtype) -> Any:
        return 0

    def finish(self, node: Any, kind: str, held: np.ndarray, counts: np.ndarray,
               blocks: int) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def exact(self, node: Any, value: np.ndarray, codes: np.ndarray, chosen: np.ndarray, groups: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no second pass")

    def exact_finish(self, node: Any, buckets: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no second pass")

    def close(self, node: Any, kind: str, held: np.ndarray, column: np.ndarray) -> np.ndarray:
        return column


def _starts(codes: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The order that puts the values group by group, and where each group starts in it."""
    order = np.argsort(codes, kind="stable")
    return order, np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=count))[:-1]])


class Count(Reduction):
    name, counted = "n", True


class Extreme(Reduction):
    """``min`` and ``max``: each block's extreme per group, merged by the same comparison."""

    def __init__(self, name: str):
        self.name = name

    def ufunc(self, node: Any) -> Any:
        if self.name == "min":
            return np.fmin if node.na_rm else np.minimum
        return np.fmax if node.na_rm else np.maximum

    def partial(self, node, value, codes, count):
        order, starts = _starts(codes, count)
        if value.dtype.kind in "US":  # text compares as Python objects, as the backends compare it
            value = value.astype(object)
        return "extreme", self.ufunc(node).reduceat(value[order], starts)

    def merge(self, node, kind, held, piece):
        if held.dtype == object:  # unseen slots take the piece, since no other value compares with them
            unseen = np.frompyfunc(lambda item: item is _UNSEEN, 1, 1)(held).astype(bool)
            held = np.where(unseen, piece, held)
        return self.ufunc(node)(held, piece)

    def fill(self, node, kind, dtype):
        if dtype.kind in "fc":  # NaN where NaN is skipped (fmin, fmax), so a group of NaN stays NaN; else the far end
            return np.nan if node.na_rm else (np.inf if self.name == "min" else -np.inf)
        if dtype.kind in "mM":  # the far ends of the int64 the times are held in; its smallest is NaT
            limits = np.iinfo(np.int64)
            return np.array(limits.max if self.name == "min" else limits.min + 1, dtype=np.int64).view(dtype)[()]
        if dtype.kind == "O":
            return _UNSEEN
        limits = np.iinfo(dtype)
        return limits.max if self.name == "min" else limits.min

    def finish(self, node, kind, held, counts, blocks):
        """Missing where a group had no positions (an empty bin), as in memory: NaN of a float's own type, NaT, NaN
        among objects; integers and booleans have no missing value, so float64 when a group is empty."""
        empty = counts == 0
        if held.dtype.kind == "O":
            unseen = np.frompyfunc(lambda item: item is _UNSEEN, 1, 1)(held).astype(bool)
            return np.where(empty | unseen, np.nan, held), None
        if not empty.any():
            return held, None
        if held.dtype.kind in "fc":
            return np.where(empty, np.nan, held).astype(held.dtype), None
        if held.dtype.kind in "mM":
            return np.where(empty, np.array("NaT", dtype=held.dtype), held), None
        return np.where(empty, np.nan, held.astype(np.float64)), None


class Flag(Reduction):
    """``any`` and ``all``: each block's answer per group, merged by or and and. A missing value is left out as in
    memory (False for any, True for all); for values that can be missing, whether the group had one rides beside
    the answer, merged by the same or and and: for ``all`` the row says the group had none, so one fill (False for
    any, True for all) serves both rows."""

    def __init__(self, name: str):
        self.name = name

    def ufunc(self) -> Any:
        return np.logical_or if self.name == "any" else np.logical_and

    def partial(self, node, value, codes, count):
        order, starts = _starts(codes, count)
        flags = value.astype(bool)
        if value.dtype.kind not in "fcmM":
            return "flag", self.ufunc().reduceat(flags[order], starts)
        missing = np.isnat(value) if value.dtype.kind in "mM" else np.isnan(value)
        flags[missing] = self.name == "all"
        marks = missing if self.name == "any" else ~missing
        return "flagged", np.stack([self.ufunc().reduceat(flags[order], starts),
                                    self.ufunc().reduceat(marks[order], starts)])

    def merge(self, node, kind, held, piece):
        return self.ufunc()(held, piece)

    def fill(self, node, kind, dtype):
        return self.name == "all"

    def finish(self, node, kind, held, counts, blocks):
        if kind == "flag":
            return held, None
        answer, marks = held  # missing where a group had a missing value, as in memory, unless na_rm
        missing = marks if self.name == "any" else ~marks
        return (answer if node.na_rm else np.where(missing, np.nan, answer.astype(np.float64))), None


class Spread(Reduction):
    """``std`` and ``var``: each block's moments per group in float64, merged by Chan, Golub, and LeVeque's update."""

    def __init__(self, name: str):
        self.name = name

    def partial(self, node, value, codes, count):
        """The values present, their mean, their squared deviations' sum, and the NaN among them."""
        floats = value.astype(np.float64)
        missing = np.isnan(floats)
        present = np.bincount(codes, weights=~missing, minlength=count)
        total = np.bincount(codes, weights=np.where(missing, 0.0, floats), minlength=count)
        mean = np.divide(total, present, out=np.zeros(count), where=present > 0)
        deviations = np.where(missing, 0.0, floats - mean[codes]) ** 2
        squares = np.bincount(codes, weights=deviations, minlength=count)
        return "moments", np.stack([present, mean, squares, np.bincount(codes, weights=missing, minlength=count)])

    def merge(self, node, kind, held, piece):
        na, ma, sa, nana = held
        nb, mb, sb, nanb = piece
        n = na + nb
        delta = mb - ma
        share = np.divide(nb, n, out=np.zeros_like(n), where=n > 0)
        return np.stack([n, ma + delta * share, sa + sb + delta * delta * na * share, nana + nanb])

    def finish(self, node, kind, held, counts, blocks):
        present, _, squares, missing = held
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = np.where(present > 0, squares / present, np.nan)  # the population variance, as np.var
        if not node.na_rm:
            variance = np.where(missing > 0, np.nan, variance)
        return (np.sqrt(variance) if self.name == "std" else variance), None


class Total(Reduction):
    """``sum`` and ``mean``. Float32 values give ``labelarray.sums`` estimates, whose groups the error bound cannot
    settle get a second pass of exact sums, so the result is the same bits as in memory; float64 values add in
    float64, as do integers for a mean; integer sums add in int64, wrapping as ``np.sum`` does."""

    def __init__(self, name: str):
        self.name = name

    def partial(self, node, value, codes, count):
        if value.dtype == np.float32:
            return "estimate", sums.chunk_estimate(value, (0,), na_rm=node.na_rm, codes=codes, groups=count)
        counted = np.bincount(codes, minlength=count).astype(np.float64)
        if value.dtype.kind == "f":
            missing = np.isnan(value)
            summed = np.where(missing, 0.0, value) if node.na_rm else value
            return "float", np.stack([np.bincount(codes, weights=summed, minlength=count),
                                      np.bincount(codes, weights=~missing, minlength=count), counted])
        if value.dtype.kind in "biu" and self.name == "mean":  # added in float64, as np.mean adds integers
            return "float", np.stack([np.bincount(codes, weights=value, minlength=count), counted, counted])
        if value.dtype.kind in "biu":
            order, starts = _starts(codes, count)
            return "int", np.stack([np.add.reduceat(value[order].astype(np.int64), starts),
                                    counted.astype(np.int64)])
        raise BlockError(f"{node.text()}: {self.name} of {value.dtype} values is not handled on blocks; {WHOLE_AXIS}")

    def merge(self, node, kind, held, piece):
        if kind == "estimate":
            merged = held.copy()
            merged[:sums.TERMS] += piece[:sums.TERMS]
            merged[sums.TERMS] = np.maximum(held[sums.TERMS], piece[sums.TERMS])
            return merged
        return held + piece

    def finish(self, node, kind, held, counts, blocks):
        if kind == "estimate":
            result, undecided = sums.settle(np.asarray(held, dtype=np.float64), blocks, self.name, node.na_rm)
            return result, (np.flatnonzero(undecided) if undecided.any() else None)
        if kind == "float":
            total, present, counted = held
            with np.errstate(invalid="ignore", divide="ignore"):
                result = total if self.name == "sum" else total / (present if node.na_rm else counted)
            return self._at_least(node, result, present), None
        total, counted = held
        return self._at_least(node, total, counted), None

    def _at_least(self, node: Any, result: np.ndarray, present: np.ndarray) -> np.ndarray:
        """A sum as NaN where fewer than ``min_count`` values are present."""
        if self.name != "sum" or node.min_count is None:
            return result
        return np.where(present >= node.min_count, result, np.nan)

    def exact(self, node, value, codes, chosen, groups):
        return sums.chunk_exact(value, (0,), chosen, codes=codes, groups=groups)

    def exact_finish(self, node, buckets):
        return sums.exact_results(buckets, self.name, node.na_rm)

    def close(self, node, kind, held, column):
        """A float32 sum's ``min_count``, once the second pass has filled its groups in, as ``from_chunks`` does."""
        if self.name == "sum" and node.min_count is not None and kind == "estimate":
            return np.where(held[sums.PRESENT] >= node.min_count, column, np.float32(np.nan))
        return column


REGISTRY: dict[str, Reduction] = {entry.name: entry for entry in (
    Count(), Total("sum"), Total("mean"), Extreme("min"), Extreme("max"), Spread("std"), Spread("var"),
    Flag("any"), Flag("all"),
)}
