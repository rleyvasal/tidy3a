"""A frame's summary as compiled kernels, a block at a time (plan milestone 16, step 5).

``build(frame)`` compiles two kernels for a summary the planner says compiles. Each block then runs:

1. ``place``: one pass over the positions computing what the filters test and each key's bucket (``fused.keys``),
   keeping the positions that pass, and stopping at a value where agreement with NumPy is not established;
2. each key's groups from its buckets, by the code ``resolve_value_key``'s answer is checked against
   (``fused.keys.groups``), and each kept position's group in dense tables, one slot per combination of the keys'
   groups, as ``np.ravel_multi_index`` numbers them;
3. ``accumulate``: one pass over the kept positions' values adding each into its group's slots
   (``fused.reductions``), twice for the spreads;
4. the occupied groups as the table ``_partial`` gives the block: the keys' groups, the combinations, the counts
   of positions, and each reduction's pieces, which merge into the tree as before.

A block whose values do not agree, or whose dense tables would pass ``DENSE_GROUPS`` groups or ``dense_limit``'s
bytes, runs on NumPy instead (``run`` says why); its table is the same either way.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tidy3a.errors import FilterError, MutateError, SummariseError, TidyError
from tidy3a.expr import Context, evaluate
from tidy3a.fused.expressions import Compiler, Source
from tidy3a.fused.keys import (Key, compile_keys, dense_kernels, groups, kernel_source, key_for, labels_of,
                              missing_label)
from tidy3a.fused.reductions import Slot, compile_summary, counted_values, summary_source

DENSE_GROUPS = 16_000_000  # the owner's limit on a block's dense combinations of the keys' groups
DENSE_BYTES = 2**27  # and on the bytes of its dense tables, at most
DENSE_SHARE = 8  # of a block's share of a budget, the part its dense tables may take


def dense_limit(memory: int | None, threads: int) -> int:
    """The most bytes a block's dense tables may take: ``DENSE_BYTES``, or with a budget an eighth of a running
    block's share of it, if less; ``sizing`` counts it for each block running."""
    if memory is None:
        return DENSE_BYTES
    from tidy3a.sizing import RESERVE

    return min(DENSE_BYTES, max(0, memory - RESERVE) // (DENSE_SHARE * threads))


@dataclass
class Fused:
    """A frame's summary compiled: the compiler's inputs, the filters' tests, the keys, the reductions' slots, and
    the two kernels."""

    frame: Any
    compiler: Compiler
    tests: list[Source]
    keys: list[Key]
    slots: list[Slot]
    place: Any = None
    accumulate: Any = None
    occupied: Any = None  # ``fused.keys.dense_kernels``
    dense: Any = None
    limit: int = DENSE_BYTES  # the bytes its dense tables may take (``dense_limit``)
    group_bytes: int = field(init=False, default=8)

    def __post_init__(self) -> None:
        self.group_bytes = 8 + sum(array.itemsize for slot in self.slots for array in slot.arrays(1).values())

    def run(self, block: Any) -> tuple[Any, str | None]:
        """The block's table, or None and why it runs on NumPy instead."""
        from tidy3a.chunked import _Table

        axis = self.compiler.axis
        context = Context(block, verb="summarise", error=SummariseError, grouped=False)
        inputs = [_positions_first(evaluate(ref, context), axis) for ref, _ in self.compiler.inputs.values()]
        positions = int(block._backend.sizes(block._data)[axis])
        cells = math.prod(size for _, size in (self.compiler.cells or ()))
        count = len(self.keys)
        ks = np.empty((count, positions), np.int64)
        missing = np.zeros((count, positions), np.bool_)
        keep = np.zeros(positions, np.bool_)
        ranges = np.zeros((max(count, 1), 3), np.int64)
        edges = [key.constants["edges"] for key in self.keys if key.how == "bins"]
        placed = [array for k in range(count) for array in (ks[k], missing[k])]
        if self.place(positions, *inputs, *edges, *placed, keep, ranges) != 0:
            return None, "a value where agreement with NumPy is not established"
        groups = self._groups(positions, keep, ks, missing, ranges)
        if groups is None:  # a key's range far wider than its values: its groups by sorting them
            groups = self._sorted(positions, keep, ks, missing, ranges)
        found, sizes, ranked, dense = groups
        combinations = math.prod(sizes) if sizes else 1
        if combinations > DENSE_GROUPS:  # the reasons say no block's own count, so the result counts blocks by them
            return None, f"its keys' groups make more than {DENSE_GROUPS:,} combinations"
        if combinations * self.group_bytes > self.limit:
            return None, f"its keys' groups' dense tables would take more than {_bytes(self.limit)}"
        if dense is None:
            dense = np.empty(positions, np.intp)
            valid = self.dense(positions, keep, ks, missing, *ranked, np.array(sizes, np.int64), dense)
        else:
            valid = int(np.count_nonzero(dense >= 0))
        plan = self.frame._plan
        dropped = int(np.count_nonzero(keep)) - valid
        table = _Table(plan)
        if not valid:
            table.merge(found, None, None, None, dropped)
            return table, None
        arrays = [slot.arrays(combinations) for slot in self.slots]
        counts = np.zeros(combinations, np.int64)
        self.accumulate(positions, cells, dense, *inputs, *(array for slot in arrays for array in slot.values()),
                        counts)
        occupied = np.flatnonzero(counts)
        hidden = [name for name, node in plan.reductions.items() if node.op != "n"]  # the slots' order
        pieces = {name: slot.pieces({key: array[occupied] for key, array in held.items()},
                                    counted_values(counts[occupied], slot, cells))
                  for name, slot, held in zip(hidden, self.slots, arrays)}
        combos = [(labels, missing_label, where) for (labels, missing_label, _), where
                  in zip(found, np.unravel_index(occupied, sizes))] if sizes else []
        table.merge(found, combos, counts[occupied] * plan.per_position, pieces, dropped)
        return table, None

    def _groups(self, positions: int, keep: np.ndarray, ks: np.ndarray, missing: np.ndarray,
                ranges: np.ndarray) -> tuple[list, list[int], tuple, None] | None:
        """Each key's groups by counting its range (``fused.keys.groups``'s rule), and what ``dense`` numbers the
        kept positions' combinations by: each key's lowest bucket, its buckets' ranks among the occupied ones (one
        array, at offsets), and its missing group's code (-1 where dropped); None where a key's range is far wider
        than its values."""
        from tidy3a.chunked import _label_array

        found, sizes, lows, ranks, codes = [], [], [], [], []
        for k, key in enumerate(self.keys):
            low, high, count = (int(value) for value in ranges[k])
            dtype = key.value.dtype
            if key.how == "bins":
                labels = labels_of(key, ks[k, :0], dtype)
                low, rank = 0, np.arange(len(labels), dtype=np.int64)
            elif high < low:
                labels, low, rank = [], 0, np.zeros(0, np.int64)
            elif high - low < 4 * positions + 1_024:
                occupied = np.zeros(high - low + 1, np.bool_)
                self.occupied(positions, keep, ks[k], missing[k], low, occupied)
                labels = labels_of(key, np.flatnonzero(occupied) + low, dtype)
                rank = np.cumsum(occupied, dtype=np.int64) - 1
            else:
                return None
            dropped = count if count and key.part.dropna else 0
            missing_group = missing_label(key, dtype) if count and not dropped else None
            found.append((_label_array(labels), missing_group, dropped))
            sizes.append(len(labels) + (missing_group is not None))
            lows.append(low)
            ranks.append(rank)
            codes.append(len(labels) if missing_group is not None else -1)
        offsets = np.cumsum([0, *(len(rank) for rank in ranks[:-1])], dtype=np.int64) if ranks else np.zeros(0, np.int64)
        ranked = (np.array(lows, np.int64), np.concatenate(ranks) if ranks else np.zeros(0, np.int64), offsets,
                  np.array(codes, np.int64))
        return found, sizes, ranked, None

    def _sorted(self, positions: int, keep: np.ndarray, ks: np.ndarray, missing: np.ndarray,
                ranges: np.ndarray) -> tuple[list, list[int], None, np.ndarray]:
        """The keys' groups as ``fused.keys.groups`` gives them, sorting a range far wider than its values, and each
        position's dense group (-1 for none)."""
        from tidy3a.chunked import _label_array

        found, local, sizes = [], [], []
        valid = keep.copy()
        for k, key in enumerate(self.keys):
            low, high, count = (int(value) for value in ranges[k])
            labels, codes, dropped = groups(key, ks[k][keep], missing[k][keep], low, high, count, key.value.dtype)
            missing_group = None
            if dropped:
                labels = labels[:-1]
                valid[np.flatnonzero(keep)[missing[k][keep]]] = False
            elif count:
                labels, missing_group = labels[:-1], labels[-1]
            found.append((_label_array(labels), missing_group, dropped))
            full = np.zeros(positions, np.intp)
            full[keep] = codes
            local.append(full)
            sizes.append(len(labels) + (missing_group is not None))
        combinations = math.prod(sizes)
        if combinations > DENSE_GROUPS:
            return found, sizes, None, np.empty(0, np.intp)  # refused by the caller before it is read
        dense = np.ravel_multi_index([np.where(valid, codes, 0) for codes in local], sizes)  # a dropped one's is none
        return found, sizes, None, np.where(valid, dense, -1).astype(np.intp)


def _bytes(size: int) -> str:
    from tidy3a.sizing import describe

    return describe(size)


def _positions_first(value: Any, axis: Any) -> np.ndarray:
    """A variable's values with the blocked axis first and the rest flattened, as a kernel indexes them."""
    values = np.asarray(value.values)
    if values.ndim == 1:
        return values  # a field of a record array stays a view: the kernel reads it in place
    dims = list(getattr(value, "dims", ()))
    return np.ascontiguousarray(np.moveaxis(values, dims.index(axis), 0).reshape(values.shape[dims.index(axis)], -1))


def build(frame: Any) -> Fused | None:
    """*frame*'s summary compiled into its kernels, or None when the planner says it runs on NumPy."""
    from tidy3a.chunked import ROW_WISE, _needs
    from tidy3a.fused import plan_for
    from tidy3a.verbs import mutate

    if not plan_for(frame).compiled:
        return None
    blocks = frame._blocks
    rows = blocks.source.empty(_needs(frame))
    if not blocks.labelled:
        rows = rows._evolve(rows._backend.label_positions(rows._data, blocks.axis, 0))
    compiler = Compiler(blocks.axis)
    tests: list[Source] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for verb in frame._steps:
            if verb._operation not in ROW_WISE:
                continue
            if verb._operation == "filter":
                context = Context(rows, verb="filter", error=FilterError)
                tests += [compiler.compile(predicate, context) for predicate in verb._arguments]
            elif verb._operation == "mutate":
                for step in verb._arguments:
                    compiler.define(step.target, step.value, Context(rows, verb="mutate", error=MutateError))
                    rows = rows >> mutate(step)
                continue
            rows = rows >> verb
        plan = frame._plan
        keys = [key_for(part, Context(rows, verb="group_by", error=TidyError, grouped=False), compiler)
                for part in plan.parts]
        context = Context(rows, verb="summarise", error=SummariseError, grouped=False)
        slots = [Slot(k, node, compiler.compile(node.arg, context))
                 for k, node in enumerate(node for node in plan.reductions.values() if node.op != "n")]
    fused = Fused(frame, compiler, tests, keys, slots, limit=dense_limit(blocks.memory, blocks.threads))
    tested = [f"np.bool_({cast_bool(test)})" for test in tests]
    fused.place = compile_keys(kernel_source("place", keys, compiler, tested), "place")
    fused.accumulate = compile_summary(summary_source("accumulate", list(compiler.inputs), slots), "accumulate")
    fused.occupied, fused.dense = dense_kernels()
    return fused


def cast_bool(test: Source) -> str:
    from tidy3a.fused.expressions import cast

    return cast(test, np.dtype(bool))
