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
  the passes it asks for, if the pieces do not settle every group;
- for a reduction that asks for passes (``Ask``): ``again`` (one block's
  answer to an ask), ``join`` (two blocks' answers as one, in any order),
  and ``resolve`` (the answer over every block, into the column, with any
  further asks); and ``close``, once every pass is done.

``n()`` is the table's own count of positions per group; its entry has no
pieces. The executor looks reductions up by name and knows none of them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from labelarray import sums

from tidy3a.errors import BlockError

WHOLE_AXIS = "read the data with lazy=True, which memory-maps a file, or load it, to run this on every position"
_UNSEEN = object()  # an object accumulator's slot for a group no block has had yet


@dataclass(frozen=True)
class Ask:
    """A reduction's request for one more pass over the blocks: what it wants of each block (``state``, its own), and
    how many values one block's answer holds (``size``), which the driver packs passes by."""

    state: Any
    size: int


def pass_budget() -> int:
    """How many values the answers to one pass's asks may hold per block, together: ``labelarray.sums.EXACT_PAIRS``,
    the float32 exact pass's batch, read when a scan runs."""
    return sums.EXACT_PAIRS


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
               blocks: int) -> tuple[np.ndarray, list[Ask]]:
        raise NotImplementedError

    def again(self, node: Any, ask: Ask, value: np.ndarray, codes: np.ndarray, groups: int) -> Any:
        raise NotImplementedError(f"{self.name} asks for no pass")

    def join(self, node: Any, ask: Ask, first: Any, second: Any) -> Any:
        return first + second

    def resolve(self, node: Any, ask: Ask, total: Any, column: np.ndarray) -> tuple[np.ndarray, list[Ask]]:
        raise NotImplementedError(f"{self.name} asks for no pass")

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
            return np.where(empty | unseen, np.nan, held), []
        if not empty.any():
            return held, []
        if held.dtype.kind in "fc":
            return np.where(empty, np.nan, held).astype(held.dtype), []
        if held.dtype.kind in "mM":
            return np.where(empty, np.array("NaT", dtype=held.dtype), held), []
        return np.where(empty, np.nan, held.astype(np.float64)), []


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
            return held, []
        answer, marks = held  # missing where a group had a missing value, as in memory, unless na_rm
        missing = marks if self.name == "any" else ~marks
        return (answer if node.na_rm else np.where(missing, np.nan, answer.astype(np.float64))), []


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
        return (np.sqrt(variance) if self.name == "std" else variance), []


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
        if kind == "estimate":  # the groups the error bound cannot settle ask for their exact sums, a batch a pass
            result, undecided = sums.settle(np.asarray(held, dtype=np.float64), blocks, self.name, node.na_rm)
            rows, batch = np.flatnonzero(undecided), pass_budget()
            return result, [Ask(rows[start:start + batch], len(rows[start:start + batch]))
                            for start in range(0, len(rows), batch)]
        if kind == "float":
            total, present, counted = held
            with np.errstate(invalid="ignore", divide="ignore"):
                result = total if self.name == "sum" else total / (present if node.na_rm else counted)
            return self._at_least(node, result, present), []
        total, counted = held
        return self._at_least(node, total, counted), []

    def _at_least(self, node: Any, result: np.ndarray, present: np.ndarray) -> np.ndarray:
        """A sum as NaN where fewer than ``min_count`` values are present."""
        if self.name != "sum" or node.min_count is None:
            return result
        return np.where(present >= node.min_count, result, np.nan)

    def again(self, node, ask, value, codes, groups):
        """One block's exact sums, as integer buckets, for the groups asked about (``labelarray.sums.chunk_exact``)."""
        return sums.chunk_exact(value, (0,), ask.state, codes=codes, groups=groups)

    def resolve(self, node, ask, total, column):
        """The correctly rounded results from the buckets added over every block, in the groups' places."""
        if total is not None:
            column[ask.state] = sums.exact_results(total, self.name, node.na_rm)
        return column, []

    def close(self, node, kind, held, column):
        """A float32 sum's ``min_count``, once the second pass has filled its groups in, as ``from_chunks`` does."""
        if self.name == "sum" and node.min_count is not None and kind == "estimate":
            return np.where(held[sums.PRESENT] >= node.min_count, column, np.float32(np.nan))
        return column


DIGIT = 16  # the bits of a value's key one pass counts


@dataclass(frozen=True)
class _Q:
    """A stand-in node with a quantile's q, for its result type."""

    q: float


@dataclass(frozen=True)
class _Selection:
    """The values the median still looks for, one or two per group (``Ask.state``): each target's group, its rank
    among the group's values present, the key bits decided so far, and how many passes have run."""

    dtype: np.dtype  # the values' type
    width: int  # the key's bits
    group: np.ndarray  # per target
    rank: np.ndarray  # per target, among the values in its group whose key starts with the prefix
    prefix: np.ndarray  # per target, the key's decided bits (uint64)
    done: int  # passes run
    first: np.ndarray  # per group, its first target, or -1
    second: np.ndarray  # per group, its second target (even counts), or -1
    fraction: np.ndarray | None = None  # per group, a quantile's place between its two values


def _key_type(dtype: np.dtype) -> tuple[np.dtype, int]:
    """The unsigned type a value's order-preserving key is held in, and its width in bits."""
    width = 32 if dtype.kind in "iu" and dtype.itemsize < 4 else dtype.itemsize * 8
    return np.dtype(f"u{max(width // 8, 2)}"), max(width, 16)


def _keys(value: np.ndarray) -> np.ndarray:
    """Unsigned keys in the values' order: floats with the sign bit flipped (negatives inverted), as radix sorts
    order them; signed integers and durations with the sign bit flipped; unsigned integers as they are."""
    unsigned, width = _key_type(value.dtype)
    top = np.array(1, dtype=unsigned) << np.array(width - 1, dtype=unsigned)
    if value.dtype.kind == "f":
        bits = value.view(unsigned)
        return np.where(bits & top, ~bits, bits | top)
    if value.dtype.kind in "im":
        signed = value.view(np.int64) if value.dtype.kind == "m" else value.astype(f"i{width // 8}", copy=False)
        return signed.view(unsigned) ^ top
    return value.astype(unsigned, copy=False)


def _values(keys: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """The values whose keys these are (``_keys`` undone)."""
    unsigned, width = _key_type(dtype)
    keys = keys.astype(unsigned)
    top = np.array(1, dtype=unsigned) << np.array(width - 1, dtype=unsigned)
    if dtype.kind == "f":
        return np.where(keys & top, keys ^ top, ~keys).view(dtype)
    if dtype.kind in "im":
        signed = (keys ^ top).view(f"i{width // 8}")
        return signed.view(dtype) if dtype.kind == "m" else signed.astype(dtype)
    return keys.astype(dtype)


class Median(Reduction):
    """``median`` on blocks, exactly: a group's values are counted by the bits of an order-preserving key, a digit of
    16 bits a pass, and each pass finds the digit holding the rank it looks for, so float32 and 32-bit integers take
    two passes after the first and 64-bit values four; the one or two middle values then give ``np.median``'s answer,
    the same bits as in memory. The first pass counts each group's values present and missing; without ``na_rm`` a
    group with a missing one is missing, as in memory. Where both zeros, -0.0 and 0.0, lie in the middle, the sign of
    a zero result may differ from NumPy's, which does not order them."""

    name = "median"

    def partial(self, node, value, codes, count):
        if value.dtype.kind not in "fium" or value.dtype.kind == "f" and value.dtype.itemsize < 2:
            raise BlockError(f"{node.text()}: the {self.name} of {value.dtype} values is not handled on blocks; "
                             f"{WHOLE_AXIS}")
        missing = np.isnat(value) if value.dtype.kind == "m" else np.isnan(value) if value.dtype.kind == "f" else (
            np.zeros(len(value), dtype=bool))
        present = np.bincount(codes, weights=~missing, minlength=count).astype(np.int64)
        return f"rank:{value.dtype.str}", np.stack([present, np.bincount(codes, weights=missing,
                                                                          minlength=count).astype(np.int64)])

    def merge(self, node, kind, held, piece):
        return held + piece

    def finish(self, node, kind, held, counts, blocks):
        dtype = np.dtype(kind.split(":", 1)[1])
        result = self._result_type(dtype, node)
        present, missing = held
        column = np.full(len(present), np.array("NaT" if result.kind == "m" else np.nan).astype(result))
        wanted = (present > 0) & ((missing == 0) | node.na_rm)
        groups = np.flatnonzero(wanted)
        if not len(groups):
            return column, []
        low, high, fraction = self._ranks(present[groups], node)
        two = high != low
        group = np.concatenate([groups, groups[two]])
        rank = np.concatenate([low, high[two]]).astype(np.int64)
        first = np.full(len(present), -1, dtype=np.int64)
        second = np.full(len(present), -1, dtype=np.int64)
        first[groups] = np.arange(len(groups))
        second[groups[two]] = len(groups) + np.arange(int(two.sum()))
        _, width = _key_type(dtype)
        whole = None if fraction is None else np.zeros(len(present))
        if whole is not None:
            whole[groups] = fraction
        state = _Selection(dtype, width, group, rank, np.zeros(len(group), dtype=np.uint64), 0, first, second, whole)
        return column, [Ask(state, len(group))]

    def _result_type(self, dtype: np.dtype, node: Any) -> np.dtype:
        return np.median(np.zeros(1, dtype)).dtype

    def _ranks(self, present: np.ndarray, node: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The ranks to find per group: the middle one, or the two middle ones of an even count."""
        return (present - 1) // 2, present // 2, None

    def again(self, node, ask, value, codes, groups):
        """One block's count of each target's candidates by their next digit: as ``(target × 2**16 + digit, count)``,
        the candidates being the group's values present whose key starts with the target's prefix."""
        state = ask.state
        keys = _keys(value).astype(np.uint64)
        present = ~(np.isnat(value) if value.dtype.kind == "m" else np.isnan(value)) if value.dtype.kind in "fm" else (
            np.ones(len(value), dtype=bool))
        above = state.width - DIGIT * state.done  # the bits decided lie above this one
        digits = (keys >> np.uint64(above - DIGIT)) & np.uint64(0xFFFF)
        found = []
        for target in (state.first[codes], state.second[codes]):
            match = present & (target >= 0)
            if state.done:
                match &= (keys >> np.uint64(above)) == state.prefix[np.maximum(target, 0)]
            found.append(target[match].astype(np.int64) * 65_536 + digits[match].astype(np.int64))
        pairs, tally = np.unique(np.concatenate(found), return_counts=True)
        return pairs, tally.astype(np.int64)

    def join(self, node, ask, first, second):
        pairs, inverse = np.unique(np.concatenate([first[0], second[0]]), return_inverse=True)
        return pairs, np.bincount(inverse.reshape(-1), weights=np.concatenate([first[1], second[1]])).astype(np.int64)

    def resolve(self, node, ask, total, column):
        """Each target's digit: the one where its group's candidates, counted digit by digit, pass its rank; with every
        digit found, the values, and the median of the one or two per group."""
        state = ask.state
        pairs, tally = total
        target, digit = pairs // 65_536, (pairs % 65_536).astype(np.uint64)
        before = np.cumsum(tally) - tally  # candidates before each (target, digit), over all targets
        start = np.searchsorted(target, np.arange(len(state.group)))
        below = before - before[start[target]]  # within the target
        passed = (below + tally > state.rank[target]) & (below <= state.rank[target])
        chosen = np.flatnonzero(passed)
        rank = state.rank.copy()
        prefix = state.prefix.copy()
        rank[target[chosen]] -= below[chosen]
        prefix[target[chosen]] = (prefix[target[chosen]] << np.uint64(DIGIT)) | digit[chosen]
        done = state.done + 1
        if DIGIT * done < state.width:
            step = _Selection(state.dtype, state.width, state.group, rank, prefix, done, state.first, state.second,
                              state.fraction)
            return column, [Ask(step, ask.size)]
        values = _values(prefix, state.dtype)
        groups = np.flatnonzero(state.first >= 0)
        one = values[state.first[groups]]
        other = np.where(state.second[groups] >= 0, values[np.maximum(state.second[groups], 0)], one)
        column[groups] = self._combine(one, other, state.second[groups] < 0, state, groups)
        return column, []

    def _combine(self, one: np.ndarray, other: np.ndarray, alone: np.ndarray, state: _Selection,
                 groups: np.ndarray) -> np.ndarray:
        """The median of each group's one or two middle values, as ``np.median`` gives it."""
        with np.errstate(all="ignore"):  # an odd group's (v + v) / 2 may overflow; its value is taken whole below
            middle = np.median(np.stack([one, other], axis=1), axis=1)
        return np.where(alone, np.asarray(one).astype(middle.dtype), middle)  # an odd count: the value itself


class Quantile(Median):
    """``quantile(x, q)`` on blocks, as the median: NumPy's linear rule takes the values at ranks ``floor(h)`` and the
    next (the last, at the end), ``h = q × (n − 1)``, so the passes find both, and ``np.quantile`` of the two at
    ``h − floor(h)`` gives each group's answer as NumPy gives it from all of them, the same bits, its edge cases
    included (an infinity next to the rank makes NaN, as in NumPy)."""

    name = "quantile"

    def _result_type(self, dtype, node):
        return np.quantile(np.zeros(1, dtype), node.q).dtype

    def _ranks(self, present, node):
        place = node.q * (present - 1)  # NumPy's virtual index, computed as it computes it
        low = np.floor(place).astype(np.int64)
        return low, np.minimum(low + 1, present - 1), place - low

    def _combine(self, one, other, alone, state, groups):
        fraction = state.fraction[groups]
        result = np.empty(len(groups), dtype=self._result_type(state.dtype, _Q(0.5)))
        pairs = np.stack([one, other], axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for place in np.unique(fraction):  # groups with the same place in one call
                rows = fraction == place
                result[rows] = np.quantile(pairs[rows], place, axis=1)
        return result


REGISTRY: dict[str, Reduction] = {entry.name: entry for entry in (
    Count(), Total("sum"), Total("mean"), Extreme("min"), Extreme("max"), Spread("std"), Spread("var"),
    Flag("any"), Flag("all"), Median(), Quantile(),
)}
