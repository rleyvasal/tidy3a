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

A block whose dense tables would pass ``DENSE_GROUPS`` groups or ``dense_limit``'s bytes sorts the combinations its
positions hold instead, and its tables are as large as those, at most its positions. A block whose values do not
agree, or whose keys' groups make more combinations than int64 codes number, runs on NumPy instead (``run`` says
why); its table is the same either way.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tidy3a.errors import FilterError, MutateError, SummariseError, TidyError
from tidy3a.expr import Context, evaluate
from tidy3a.fused.expressions import Compiler, Source, read
from tidy3a.fused.kernels import compile_source
from tidy3a.fused.keys import (Key, compile_keys, dense_kernels, groups, kernel_source, key_for, labels_of,
                              missing_label)
from tidy3a.fused.reductions import Slot, compile_summary, counted_values, summary_source, zeros_ordered

DENSE_GROUPS = 16_000_000  # the owner's limit on a block's dense combinations of the keys' groups
DENSE_BYTES = 2**27  # and on the bytes of its dense tables, at most
CODES = 2**62  # combinations numbered beyond this, which int64 codes could not hold, run on NumPy
BY_POSITION = 256  # values a position holds from which NumPy's reductions take each position first (milestone 17)
KERNEL_THREADS = 8  # threads from which such blocks run the fused kernel instead, where their results allow it:
# measured on the 12-core machine, NumPy's reductions were the faster to four threads, the kernel from eight
ORDER_FREE = frozenset({"estimate", "int", "extreme", "flag", "flagged", "rank"})  # kinds whose results do not follow
# the order the values are added in: float32 sums and means are correctly rounded, the rest exact
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
    values: Any = None  # each kept position's values for NumPy's reductions, where positions hold many
    operands: dict = field(default_factory=dict)  # island's local name: (its operands' kernel, the inputs it reads, it)
    joins: list = field(default_factory=list)  # the frame's joins, which NumPy makes on each block first
    occupied: Any = None  # ``fused.keys.dense_kernels``
    dense: Any = None
    limit: int = DENSE_BYTES  # the bytes its dense tables may take (``dense_limit``)
    threads: int = 1  # the blocks run at once, which choose how positions of many values are reduced
    tested: list = field(default_factory=list)  # the filters' tests as kernel source
    group_bytes: int = field(init=False, default=8)
    row_bytes: int = field(init=False, default=8)
    learned: list = field(init=False, default_factory=list)  # per key, each bucket a block held: its label
    unlearned: bool = field(init=False, default=False)  # a block's groups were sorted, its buckets not learned
    counting: dict = field(init=False, default_factory=dict)  # per slot, its later passes' kernel (or None)
    looked: Any = field(init=False, default=None)  # the table the lookups were made for, and them
    running: dict = field(init=False, default_factory=dict)  # per thread, its running table (``_Running``)

    def __post_init__(self) -> None:
        self.group_bytes = 8 + sum(array.itemsize for slot in self.slots for array in slot.arrays(1).values())
        self.row_bytes = 8 + -(-(self.group_bytes - 8) // 8) * 8  # a running table's row (``interleaved``)
        self.learned = [{} for _ in self.keys]

    def run(self, block: Any) -> tuple[Any, str | None]:
        """The block's table, or None and why it runs on NumPy instead; or, for a summary free of its additions'
        order, the tables its thread's running table gave up (``_Running``), usually none."""
        from tidy3a.chunked import _Table

        if self.order_free:
            place = self._place(block)
            if isinstance(place, str):
                return None, place
            given = self._run_into(*place)
            if given is not None:
                return given, None
            placed = self._finish_placing(*place)
        else:
            placed = self._placed(block)
        if isinstance(placed, str):
            return None, placed
        inputs, positions, cells, found, sizes, dense, valid, dropped = placed
        combinations = math.prod(sizes) if sizes else 1
        plan = self.frame._plan
        table = _Table(plan)
        if not valid:
            table.merge(found, None, None, None, dropped)
            return table, None
        if cells >= BY_POSITION and self.values is not None and self._numpy_first():
            table.merge(*self._by_position(inputs, cells, dense, found, sizes, valid, dropped))
            return table, None
        held = None  # the combinations the block holds, in order, where its tables are only as large as those
        tables = combinations
        if combinations > DENSE_GROUPS or combinations * self.group_bytes > self.limit:
            present = dense >= 0
            held, inverse = np.unique(dense[present], return_inverse=True)
            dense = np.full(positions, -1, np.intp)
            dense[present] = inverse.reshape(-1)
            tables = len(held)
        arrays = [slot.arrays(tables) for slot in self.slots]
        counts = np.zeros(tables, np.int64)
        self.accumulate(positions, cells, dense, *inputs, *(array for slot in arrays for array in slot.values()),
                        counts)
        occupied = np.flatnonzero(counts)
        numbered = occupied if held is None else held[occupied]
        hidden = [name for name, node in plan.reductions.items() if node.op != "n"]  # the slots' order
        pieces = {name: slot.pieces({key: array[occupied] for key, array in held.items()},
                                    counted_values(counts[occupied], slot, cells))
                  for name, slot, held in zip(hidden, self.slots, arrays)}
        combos = [(labels, missing_label, where) for (labels, missing_label, _), where
                  in zip(found, np.unravel_index(numbered, sizes))] if sizes else []
        table.merge(found, combos, counts[occupied] * plan.per_position, pieces, dropped)
        return table, None

    def rows(self, block: Any) -> Any:
        """The block's kept rows as ``chunked._block`` finds them, found by the kernels: each key's groups and its
        rows' codes, and each reduction's values, which a later pass counts (plan milestone 20); None where the block
        runs on NumPy."""
        from tidy3a.chunked import _Rows

        placed = self._placed(block)
        if isinstance(placed, str):
            return None
        inputs, _, cells, found, sizes, dense, valid, dropped = placed
        plan = self.frame._plan
        rows = np.flatnonzero(dense >= 0)
        local = [codes.astype(np.intp) for codes in np.unravel_index(dense[rows], sizes)] if sizes else []
        outputs = [np.empty((valid, cells) if slot.value.cells else valid, slot.dtype) for slot in self.slots]
        if valid and self.slots:
            if self.values is None:  # compiled when a later pass first needs it
                source = values_source("values", self.compiler, [slot.value for slot in self.slots])
                self.values = compile_source(source, "values")
            self.values(rows, cells, *inputs, *outputs)
        hidden = [name for name, node in plan.reductions.items() if node.op != "n"]  # the slots' order
        values = {name: (output.reshape(-1), cells if slot.value.cells else 1)
                  for name, slot, output in zip(hidden, self.slots, outputs)}
        return _Rows(found, local, valid, dropped, plan.per_position, values)

    def count(self, block: Any, table: Any, hidden: str, ask: Any) -> Any:
        """The block's answer to a later pass of the median or a quantile in one compiled loop (``fused.ranks``):
        its filters, its keys' groups in the finished *table*, and its values counted into the thread's running
        table, nothing else made; None where the block is counted otherwise (``rows``), as where it holds a bucket
        or a value the loop cannot place, whose counts it takes back first."""
        from tidy3a.fused import ranks
        from tidy3a.fused.kernels import compile_source
        from tidy3a.fused.keys import HELPERS

        names = [name for name, node in self.frame._plan.reductions.items() if node.op != "n"]
        if self.unlearned or hidden not in names or table.keys is None:
            return None
        k = names.index(hidden)
        if self.slots[k].kind != "rank":  # a float32 sum's exact pass, say: its own steps
            return None
        if k not in self.counting:
            source = ranks.pass_source("counted", self.keys, self.compiler, self.tested, self.slots[k].value)
            self.counting[k] = None if source is None else compile_source(HELPERS + source, "counted", ("_at", "_far"))
        kernel = self.counting[k]
        lookups = self._lookups(table)
        state = ask.state
        if kernel is None or lookups is None:
            return None
        for join in self.joins:
            block = block >> join
        context = Context(block, verb="summarise", error=SummariseError, grouped=False)
        inputs = self._inputs(block, context)
        positions = int(block._backend.sizes(block._data)[self.compiler.axis])
        cells = math.prod(size for _, size in (self.compiler.cells or ()))
        out = ranks.running(state, positions * cells)
        if out is None:
            return None
        edges = [key.constants["edges"] for key in self.keys if key.how == "bins"]
        scratch, bits = ranks.scratch_for(self.slots[k].dtype)
        arguments = (cells, *inputs, *edges, *lookups, scratch, bits, state.first, state.second, state.prefix,
                     state.width - 16 * state.done, state.done, out)
        failed = kernel(0, positions, 1, *arguments)
        if failed < 0:
            kernel(0, -failed - 1, -1, *arguments)  # its counts taken back
            return None
        return ranks.Counted(state)

    def _lookups(self, table: Any) -> tuple | None:
        """Per key, the buckets the blocks held, sorted, and their groups' codes in *table*; each key's missing
        group's code; the combinations' strides; and the table's combinations. None for a table without keys."""
        from tidy3a.chunked import _label_array

        if self.looked is not None and self.looked[0] is table:
            return self.looked[1]
        if len(table.parts) != len(self.keys):
            return None
        arrays: list[np.ndarray] = []
        missing, lows = [], []
        for key, part, learned in zip(self.keys, table.parts, self.learned):
            items = sorted(dict(learned).items())
            buckets = np.array([bucket for bucket, _ in items], np.int64)
            codes = part.codes(_label_array([label for _, label in items]), False) if items else np.zeros(0, np.intp)
            buckets, codes = buckets[codes >= 0], codes[codes >= 0].astype(np.int64)
            low = int(buckets[0]) if len(buckets) else 0
            direct = np.zeros(0, np.int64)  # each bucket's code from the lowest, where the range is narrow
            if len(buckets) and int(buckets[-1]) - low < 4 * len(buckets) + 1_024:
                direct = np.full(int(buckets[-1]) - low + 1, -1, np.int64)
                direct[buckets - low] = codes
            arrays += [buckets, codes, direct]
            lows.append(low)
            missing.append(part.size - 1 if part.missing is not None else -1 if key.part.dropna else -2)
        sizes = list(table.sizes)
        strides = np.array([math.prod(sizes[k + 1:]) for k in range(len(sizes))], np.int64)
        combinations = np.asarray(table.keys, np.int64)
        rows = np.zeros(0, np.int64)
        if math.prod(sizes) <= max(4 * len(combinations), 1 << 20):  # few: each combination's row read directly
            rows = np.full(math.prod(sizes), -1, np.int64)
            rows[combinations] = np.arange(len(combinations))
        looked = (*arrays, np.array(lows, np.int64), np.array(missing, np.int64), strides, combinations, rows)
        self.looked = (table, looked)
        return looked

    @property
    def order_free(self) -> bool:
        """Whether a thread's blocks accumulate into one running table (plan milestone 20): where every reduction's
        result is free of the order of its additions, and none takes each position's values first on NumPy."""
        cells = math.prod(size for _, size in (self.compiler.cells or ()))
        return (all(slot.kind in ORDER_FREE for slot in self.slots)
                and not (cells >= BY_POSITION and self._numpy_first()))

    def _run_into(self, inputs: list, positions: int, cells: int, ks: np.ndarray, missing: np.ndarray,
                  keep: np.ndarray, ranges: np.ndarray) -> list | None:
        """The block added into its thread's running table: the tables that gave up to make room (none, or the
        running table before a block its range could not take), or None where the block's own range is too wide for
        one and it makes a table of its own."""
        import threading

        with _LOCK:
            running = self.running.get(threading.get_ident())
        given = []
        if running is not None and not running.takes(ranges, self):
            given.append(running.table(self))
            running = None
        if running is None:
            running = _Running.for_block(ranges, self)
            if running is None:
                return None
        with _LOCK:
            self.running[threading.get_ident()] = running
        running.add(inputs, positions, cells, ks, missing, keep, ranges, self)
        return given

    def drain(self, pool: Any) -> list:
        """Every thread's running table, once the blocks are done, as tables to come for the merge: one, the threads'
        tables merged densely on the pool, pairwise, where their union fits the dense limits; else one each."""
        with _LOCK:
            running, self.running = list(self.running.values()), {}
        if len(running) < 2:
            return [pool.submit(table.table, self) for table in running]
        lows, spans = list(running[0].lows), list(running[0].spans)
        for other in running[1:]:
            for k in range(len(lows)):
                if other.spans[k] == 0:
                    continue
                if spans[k] == 0:
                    lows[k], spans[k] = other.lows[k], other.spans[k]
                    continue
                top = max(lows[k] + spans[k], other.lows[k] + other.spans[k])
                lows[k] = min(lows[k], other.lows[k])
                spans[k] = top - lows[k]
        if not _Running._fits([span + extra for span, extra in zip(spans, running[0].extra)], self):
            return [pool.submit(table.table, self) for table in running]
        made = [pool.submit(table.dense, self, lows, spans) for table in running]
        while len(made) > 1:  # each merge waits only on work handed to the pool before it
            made = [pool.submit(_merged_dense, made[k], made[k + 1], self) for k in range(0, len(made) - 1, 2)] + (
                made[-1:] if len(made) % 2 else [])
        return [pool.submit(_dense_table, made[0], self)]

    def _placed(self, block: Any) -> Any:
        """The block's kernel inputs, positions, values a position, each key's groups and their sizes, each
        position's combination (-1 where none), the rows kept and those dropped; or why it runs on NumPy."""
        place = self._place(block)
        return place if isinstance(place, str) else self._finish_placing(*place)

    def _finish_placing(self, inputs: list, positions: int, cells: int, ks: np.ndarray, missing: np.ndarray,
                        keep: np.ndarray, ranges: np.ndarray) -> Any:
        """``_placed`` from the placed positions: each key's groups, and each position's combination."""
        groups = self._groups(positions, keep, ks, missing, ranges)
        if groups is None:  # a key's range far wider than its values: its groups by sorting them
            groups = self._sorted(positions, keep, ks, missing, ranges)
        found, sizes, ranked, dense = groups
        if (math.prod(sizes) if sizes else 1) >= CODES:  # the reason names no block's count: the result counts them
            return "its keys' groups make more combinations than 64-bit codes number"
        if dense is None:
            dense = np.empty(positions, np.intp)
            valid = self.dense(positions, keep, ks, missing, *ranked, np.array(sizes, np.int64), dense)
        else:
            valid = int(np.count_nonzero(dense >= 0))
        return inputs, positions, cells, found, sizes, dense, valid, int(np.count_nonzero(keep)) - valid

    def _place(self, block: Any) -> Any:
        """The block's kernel inputs, positions, and values a position, and each position placed: each key's bucket
        and whether it is missing, whether it passed the filters, and each key's range; or why it runs on NumPy."""
        for join in self.joins:  # NumPy joins the block first; the kernels read what it brings (milestone 19)
            block = block >> join

        axis = self.compiler.axis
        context = Context(block, verb="summarise", error=SummariseError, grouped=False)
        inputs = self._inputs(block, context)
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
            return "a value where agreement with NumPy is not established"
        return inputs, positions, cells, ks, missing, keep, ranges

    def _inputs(self, block: Any, context: Context) -> list[np.ndarray]:
        """The kernels' inputs on *block*, positions first: the variables read, and each island, NumPy applying its
        function to the operands its kernel computes from the inputs before it, or evaluating it whole."""
        axis, cells = self.compiler.axis, math.prod(size for _, size in (self.compiler.cells or ()))
        positions = int(block._backend.sizes(block._data)[axis])
        inputs: list[np.ndarray] = []
        named: dict[str, np.ndarray] = {}
        for local, (value, per_value) in self.compiler.inputs.items():
            if local in self.operands:
                kernel, needed, island = self.operands[local]
                outputs = [np.empty((positions, cells) if operand.cells else positions, operand.dtype)
                           for operand in island.operands if _computed(operand)]
                if outputs:
                    kernel(positions, cells, *(named[name] for name in needed), *outputs)
                computed = iter(outputs)
                arguments = []
                for operand in island.operands:
                    if not isinstance(operand, Source):
                        arguments.append(operand)
                        continue
                    # an input read as it is, its array taken as it stands; else the kernel's
                    array = named[operand.text.split("[")[0]] if not _computed(operand) else next(computed)
                    arguments.append(array.reshape(-1, 1) if per_value and not operand.cells else array)
                with np.errstate(all="ignore"):  # NumPy's warnings on a block say nothing the result does not
                    array = np.ascontiguousarray(island.function(*arguments))
            else:
                array = _positions_first(read(value, context), axis)
            named[local] = array
            inputs.append(array)
        return inputs

    def _numpy_first(self) -> bool:
        """Whether NumPy's reductions take each position's values first: below ``KERNEL_THREADS`` threads, where they
        are faster than the kernel's one value at a time; and on any number where a result follows the order of its
        additions (float64 sums and means, integer means, spreads), so a run gives the same bits on any number of
        threads. From ``KERNEL_THREADS`` on, the kernel scales, as NumPy's reductions, held by memory, do not."""
        return self.threads < KERNEL_THREADS or any(slot.kind not in ORDER_FREE for slot in self.slots)

    def _by_position(self, inputs: list[np.ndarray], cells: int, dense: np.ndarray, found: list, sizes: list[int],
                     valid: int, dropped: int) -> tuple:
        """The block's partial, ``chunked._partial``'s, for positions of many values: the kept positions' values
        computed by the ``values`` kernel, then each position reduced first by NumPy's reductions, as the NumPy path
        does, whose vector loops no compensated sum in a kernel matches (plan milestone 17)."""
        from tidy3a.chunked import _partial_rows, _Rows

        rows = np.flatnonzero(dense >= 0)
        local = [codes.astype(np.intp) for codes in np.unravel_index(dense[rows], sizes)] if sizes else []
        outputs = [np.empty((valid, cells) if slot.value.cells else valid, slot.dtype) for slot in self.slots]
        self.values(rows, cells, *inputs, *outputs)
        plan = self.frame._plan
        hidden = [name for name, node in plan.reductions.items() if node.op != "n"]  # the slots' order
        values = {name: (output.reshape(-1), cells if slot.value.cells else 1)
                  for name, slot, output in zip(hidden, self.slots, outputs)}
        partial = _partial_rows(plan, _Rows(found, local, valid, dropped, plan.per_position, values))
        extremes = [(name, slot, output) for name, slot, output in zip(hidden, self.slots, outputs)
                    if slot.kind == "extreme"]
        if extremes:  # the kernel's order of the zeros, which NumPy's on x86 does not keep: the same on either path
            if sizes:
                codes = np.unique(np.ravel_multi_index(local, sizes), return_inverse=True)[1].reshape(-1)
            else:
                codes = np.zeros(valid, np.intp)
            for name, slot, output in extremes:
                kind, held = partial[3][name]
                partial[3][name] = (kind, zeros_ordered(held, output.reshape(valid, -1), codes, slot.node.op == "min"))
        return partial

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
            dtype = key.label_dtype
            if key.how == "bins":
                labels = labels_of(key, ks[k, :0], dtype)
                low, rank = 0, np.arange(len(labels), dtype=np.int64)
                if len(self.learned[k]) != len(labels):
                    self.learned[k].update(zip(range(len(labels)), labels))
            elif high < low:
                labels, low, rank = [], 0, np.zeros(0, np.int64)
            elif high - low < 4 * positions + 1_024:
                occupied = np.zeros(high - low + 1, np.bool_)
                self.occupied(positions, keep, ks[k], missing[k], low, occupied)
                buckets = np.flatnonzero(occupied) + low
                labels = labels_of(key, buckets, dtype)
                self.learned[k].update(zip(buckets.tolist(), labels))  # what a later pass finds groups by
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

        self.unlearned = True  # its buckets not learned: a later pass finds its rows as the first did
        found, local, sizes = [], [], []
        valid = keep.copy()
        for k, key in enumerate(self.keys):
            low, high, count = (int(value) for value in ranges[k])
            labels, codes, dropped = groups(key, ks[k][keep], missing[k][keep], low, high, count, key.label_dtype)
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
        if math.prod(sizes) >= CODES:
            return found, sizes, None, np.empty(0, np.intp)  # refused by the caller before it is read
        dense = np.ravel_multi_index([np.where(valid, codes, 0) for codes in local], sizes)  # a dropped one's is none
        return found, sizes, None, np.where(valid, dense, -1).astype(np.intp)


def _bytes(size: int) -> str:
    from tidy3a.sizing import describe

    return describe(size)


def _positions_first(value: Any, axis: Any) -> np.ndarray:
    """A variable's values with the blocked axis first and the rest flattened, as a kernel indexes them; one without
    the blocked axis flattened alone, a position's values (``Compiler._layout``)."""
    values = np.asarray(value.values)
    dims = list(getattr(value, "dims", ()))
    if values.ndim > 1 and axis not in dims:
        return np.ascontiguousarray(values).reshape(-1)
    if values.ndim == 1:
        return values  # a field of a record array stays a view: the kernel reads it in place
    moved = np.moveaxis(values, dims.index(axis), 0)
    return np.ascontiguousarray(moved.reshape(moved.shape[0], math.prod(moved.shape[1:])))  # no -1: a block may be empty


_LOCK = threading.Lock()


class _Running:
    """A thread's running table for a summary free of its additions' order: dense accumulators over each key's
    buckets from ``lows[k]`` (``spans[k]`` of them, and one more for its missing group where it keeps one), which a
    block's positions add into directly; widened when a block holds buckets beyond them, while the table fits the
    dense limits, else given up as a table of its own. Its order of blocks changes nothing."""

    def __init__(self, lows: list[int], spans: list[int], fused: Fused) -> None:
        self.lows, self.spans = lows, spans
        self.extra = [0 if key.part.dropna else 1 for key in fused.keys]  # the missing group's slot
        self.arrays, self.counts = interleaved(fused.slots, self.size)
        self.dropped = [0 for _ in fused.keys]
        self.left = 0  # rows left out for a missing key
        self.blocks = 0

    @property
    def sizes(self) -> list[int]:
        return [span + extra for span, extra in zip(self.spans, self.extra)]

    @property
    def size(self) -> int:
        return math.prod(self.sizes) if self.sizes else 1

    @staticmethod
    def _fits(sizes: list[int], fused: Fused) -> bool:
        groups = math.prod(sizes) if sizes else 1
        return groups <= DENSE_GROUPS and groups * fused.row_bytes <= fused.limit and groups < CODES

    @classmethod
    def for_block(cls, ranges: np.ndarray, fused: Fused) -> _Running | None:
        lows, spans = [], []
        for k, key in enumerate(fused.keys):
            low, high = cls._range(ranges, k, key)
            lows.append(low)
            spans.append(high - low + 1)
        extra = [0 if key.part.dropna else 1 for key in fused.keys]
        if not cls._fits([span + more for span, more in zip(spans, extra)], fused):
            return None
        return cls(lows, spans, fused)

    @staticmethod
    def _range(ranges: np.ndarray, k: int, key: Key) -> tuple[int, int]:
        """Key *k*'s buckets in the block: every bin for ``bins``, none where it holds no value."""
        if key.how == "bins":
            return 0, len(key.constants["edges"]) - 2
        low, high = int(ranges[k, 0]), int(ranges[k, 1])
        return (0, -1) if high < low else (low, high)

    def takes(self, ranges: np.ndarray, fused: Fused) -> bool:
        """Whether the table holds the block's buckets, widened to if need be and if it fits."""
        lows, spans = list(self.lows), list(self.spans)
        for k, key in enumerate(fused.keys):
            low, high = self._range(ranges, k, key)
            if high < low:
                continue
            if spans[k] == 0:
                lows[k], spans[k] = low, high - low + 1
                continue
            top = max(lows[k] + spans[k] - 1, high)
            lows[k] = min(lows[k], low)
            spans[k] = top - lows[k] + 1
        if lows == self.lows and spans == self.spans:
            return True
        if not self._fits([span + extra for span, extra in zip(spans, self.extra)], fused):
            return False
        self._widen(lows, spans, fused)
        return True

    def _widen(self, lows: list[int], spans: list[int], fused: Fused) -> None:
        """The accumulators widened to *lows* and *spans*, the new groups as each slot starts them (an ``all``'s
        flags true), the old ones moved to their places."""
        old = self.sizes
        where = [np.concatenate([np.arange(span) + (low - new), [new_span] if extra else []]).astype(np.intp)
                 for span, low, new, new_span, extra in zip(self.spans, self.lows, lows, spans, self.extra)]
        self.lows, self.spans = lows, spans
        shape = self.sizes

        def wider(array: np.ndarray, out: np.ndarray) -> None:
            if shape:
                out.reshape(shape)[np.ix_(*where)] = array.reshape(old)
            else:
                out[:] = array

        arrays, counts = interleaved(fused.slots, self.size)
        for before, after in zip(self.arrays, arrays):
            for name, array in before.items():
                wider(array, after[name])
        wider(self.counts, counts)
        self.arrays, self.counts = arrays, counts

    def add(self, inputs: list, positions: int, cells: int, ks: np.ndarray, missing: np.ndarray, keep: np.ndarray,
            ranges: np.ndarray, fused: Fused) -> None:
        sizes = self.sizes
        ranks = [np.arange(span, dtype=np.int64) for span in self.spans]
        offsets = np.cumsum([0, *self.spans[:-1]], dtype=np.int64) if ranks else np.zeros(0, np.int64)
        codes = np.array([span if extra else -1 for span, extra in zip(self.spans, self.extra)], np.int64)
        dense = np.empty(positions, np.intp)
        valid = fused.dense(positions, keep, ks, missing, np.array(self.lows, np.int64),
                            np.concatenate(ranks) if ranks else np.zeros(0, np.int64), offsets, codes,
                            np.array(sizes, np.int64), dense)
        for k, key in enumerate(fused.keys):
            if key.part.dropna:
                self.dropped[k] += int(ranges[k, 2])
        self.left += int(np.count_nonzero(keep)) - valid
        if valid:
            self.blocks += 1
            fused.accumulate(positions, cells, dense, *inputs,
                             *(array for arrays in self.arrays for array in arrays.values()), self.counts)

    def table(self, fused: Fused) -> Any:
        """The running table as the merge takes a block's (``chunked._Table``): its occupied groups, labelled."""
        return self.dense(fused).table(fused)

    def dense(self, fused: Fused, lows: list[int] | None = None, spans: list[int] | None = None) -> _Dense:
        """The table's pieces for every group (``Slot.pieces``), an empty group's as its reduction fills one,
        widened first to *lows* and *spans* where given."""
        from tidy3a.reductions import REGISTRY

        if lows is not None and spans is not None and (lows != self.lows or spans != self.spans):
            self._widen(lows, spans, fused)
        plan = fused.frame._plan
        cells = math.prod(size for _, size in (fused.compiler.cells or ()))
        counts = np.ascontiguousarray(self.counts)
        empty = counts == 0
        hidden = [name for name, node in plan.reductions.items() if node.op != "n"]  # the slots' order
        pieces = {}
        for name, slot, arrays in zip(hidden, fused.slots, self.arrays):
            kind, held = slot.pieces(arrays, counted_values(counts, slot, cells))
            node = plan.reductions[name]
            held = np.asarray(held)  # its own, or the running table's, used no more
            held[..., empty] = REGISTRY[node.op].fill(node, kind, held.dtype)
            pieces[name] = (kind, held)
        return _Dense(list(self.lows), list(self.spans), list(self.extra), counts, pieces, list(self.dropped),
                      self.left, 1)


@dataclass
class _Dense:
    """Running tables' pieces for every group of one layout, which the reductions' own merges add elementwise, so
    the threads' tables merge without being made tables first (plan milestone 20)."""

    lows: list[int]
    spans: list[int]
    extra: list[int]
    counts: np.ndarray
    pieces: dict
    dropped: list[int]
    left: int
    blocks: int  # the running tables in it, which the float32 estimates' bounds count as blocks

    @property
    def sizes(self) -> list[int]:
        return [span + extra for span, extra in zip(self.spans, self.extra)]

    def merged(self, other: _Dense, fused: Fused) -> _Dense:
        from tidy3a.reductions import REGISTRY

        plan = fused.frame._plan
        self.counts += other.counts
        for name, (kind, held) in self.pieces.items():
            node = plan.reductions[name]
            self.pieces[name] = (kind, REGISTRY[node.op].merge(node, kind, held, other.pieces[name][1]))
        self.dropped = [a + b for a, b in zip(self.dropped, other.dropped)]
        self.left += other.left
        self.blocks += other.blocks
        return self

    def table(self, fused: Fused) -> Any:
        """Its occupied groups as a table, labelled, as the merge takes a block's (``chunked._Table``)."""
        from tidy3a.chunked import _label_array, _Table

        plan = fused.frame._plan
        table = _Table(plan)
        occupied = np.flatnonzero(self.counts)
        codes = np.unravel_index(occupied, self.sizes) if self.sizes else []
        found, combos = [], []
        for k, key in enumerate(fused.keys):
            code, span = codes[k], self.spans[k]
            absent = code == span
            dtype = key.label_dtype
            if key.how == "bins":
                labels = labels_of(key, np.zeros(0, np.int64), dtype)
                local = code.astype(np.intp)
            else:
                buckets = np.unique(code[~absent])
                labels = labels_of(key, buckets + self.lows[k], dtype)
                fused.learned[k].update(zip((buckets + self.lows[k]).tolist(), labels))
                local = np.searchsorted(buckets, code).astype(np.intp)
            missing_group = missing_label(key, dtype) if absent.any() else None
            local[absent] = len(labels)
            found.append((_label_array(labels), missing_group, self.dropped[k]))
            combos.append((found[-1][0], missing_group, local))
        if not len(occupied):
            table.merge(found, None, None, None, self.left)
            return table
        pieces = {name: (kind, held[..., occupied]) for name, (kind, held) in self.pieces.items()}
        table.merge(found, combos, self.counts[occupied] * plan.per_position, pieces, self.left)
        table.blocks = self.blocks  # one long block a running table: its estimates' bounds count its values
        return table


def _merged_dense(left: Any, right: Any, fused: Fused) -> _Dense:
    return left.result().merged(right.result(), fused)


def _dense_table(made: Any, fused: Fused) -> Any:
    return made.result().table(fused)


def interleaved(slots: list[Slot], groups: int) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """The slots' accumulators and the counts for *groups* groups, each group's together in one row of a buffer
    (strided views into it), as each slot starts them: so a value's update touches one or two cache lines, not one
    an accumulator, where the groups are many and the threads share the memory."""
    starts = [slot.arrays(1) for slot in slots]
    fields = [(k, name, array.dtype) for k, start in enumerate(starts) for name, array in start.items()]
    fields.sort(key=lambda field: -field[2].itemsize)  # each field aligned within the row
    offsets, row = [], 0
    for _, _, dtype in fields:
        offsets.append(row)
        row += dtype.itemsize
    row = 8 + -(-row // 8) * 8  # the count first, the row a whole number of 8 bytes
    buffer = np.zeros(max(groups, 1) * row, np.uint8)  # a row at least, so the views are valid when no group is

    def view(dtype: np.dtype, offset: int) -> np.ndarray:
        return np.ndarray((groups,), dtype, buffer, offset, (row,))

    arrays: list[dict[str, np.ndarray]] = [{} for _ in slots]
    for (k, name, dtype), offset in zip(fields, offsets):
        array = view(dtype, 8 + offset)
        if starts[k][name][0]:
            array[:] = starts[k][name][0]  # an ``all``'s flags start true
        arrays[k][name] = array
    ordered = [{name: arrays[k][name] for name in starts[k]} for k in range(len(slots))]  # the kernel's order
    return ordered, view(np.dtype(np.int64), 0)


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
                    compiler.define(step.target, step.value, Context(rows, verb="mutate", error=MutateError), step)
                    rows = rows >> mutate(step)
                continue
            rows = rows >> verb
        plan = frame._plan
        keys = [key_for(part, Context(rows, verb="group_by", error=TidyError, grouped=False), compiler)
                for part in plan.parts]
        context = Context(rows, verb="summarise", error=SummariseError, grouped=False)
        slots = [Slot(k, node, compiler.compile(node.arg, context))
                 for k, node in enumerate(node for node in plan.reductions.values() if node.op != "n")]
    tested = [f"np.bool_({cast_bool(test)})" for test in tests]
    fused = Fused(frame, compiler, tests, keys, slots, limit=dense_limit(blocks.memory, blocks.threads),
                  threads=blocks.threads, tested=tested)
    fused.place = compile_keys(kernel_source("place", keys, compiler, tested), "place")
    fused.accumulate = compile_summary(summary_source("accumulate", list(compiler.inputs), slots), "accumulate")
    fused.occupied, fused.dense = dense_kernels()
    fused.operands = operand_kernels(compiler)
    from tidy3a.chunked import JOINS

    fused.joins = [verb for verb in frame._steps if verb._operation in JOINS]
    if slots and compiler.cells:
        fused.values = compile_source(values_source("values", compiler, [slot.value for slot in slots]), "values")
    return fused


def _computed(operand: Any) -> bool:
    """Whether an island's operand is computed by its kernel: not a Python number, nor an input read as it is."""
    import re

    return isinstance(operand, Source) and re.fullmatch(r"a\d+\[i(, j)?\]", operand.text) is None


def operand_kernels(compiler: Compiler) -> dict:
    """For each island whose operands compile, the kernel computing them, which reads only the inputs before it."""
    import re

    from tidy3a.fused.expressions import Island

    kernels = {}
    for local, (value, _) in compiler.inputs.items():
        if not isinstance(value, Island) or value.function is None:
            continue
        sources = [operand for operand in value.operands if _computed(operand)]
        if not sources:  # every operand an input read as it is: NumPy applies the function to them, no kernel
            kernels[local] = (None, [], value)
            continue
        needed = sorted({name for source in sources for name in re.findall(r"\ba\d+\b", source.text)},
                        key=lambda name: int(name[1:]))
        arguments = ", ".join(["positions", "values", *needed, *(f"out{k}" for k in range(len(sources)))])
        lines = [f"def operands({arguments}):", "    for i in range(positions):"]
        lines += [f"        out{k}[i] = {source.text}" for k, source in enumerate(sources) if not source.cells]
        per_value = [(k, source) for k, source in enumerate(sources) if source.cells]
        if per_value:
            lines.append("        for j in range(values):")
            lines += [f"            out{k}[i, j] = {source.text}" for k, source in per_value]
        kernels[local] = (compile_source("\n".join(lines) + "\n", "operands"), needed, value)
    return kernels


def values_source(name: str, compiler: Compiler, outputs: list[Source]) -> str:
    """Source of ``name(rows, values, a0, ..., out0, ...)`` filling row ``r`` of each output with position
    ``rows[r]``'s value, or values, of its source."""
    arguments = ", ".join(["rows", "values", *compiler.inputs, *(f"out{k}" for k in range(len(outputs)))])
    lines = [f"def {name}({arguments}):", "    for r in range(len(rows)):", "        i = rows[r]"]
    lines += [f"        out{k}[r] = {source.text}" for k, source in enumerate(outputs) if not source.cells]
    per_value = [(k, source) for k, source in enumerate(outputs) if source.cells]
    if per_value:
        lines.append("        for j in range(values):")
        lines += [f"            out{k}[r, j] = {source.text}" for k, source in per_value]
    return "\n".join(lines) + "\n"


def cast_bool(test: Source) -> str:
    from tidy3a.fused.expressions import cast

    return cast(test, np.dtype(bool))
