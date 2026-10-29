"""Data read in blocks, and the verbs run over every block (plan milestone 10, decision 10, stage b; milestone 11).

``read_array(path, ..., blocks=rows)``, ``read_array(array, ..., blocks=rows)``,
and ``tidy(dataset, blocks={axis: rows})`` give a ``ChunkedFrame``: a frame
holding a block of no positions with the data's schema (``Source.empty``),
so nothing is read until a result is asked for (plan decision 4). Each verb
runs on that empty block as it is called, so its checks and errors appear
at once, as they do in memory, and ``repr`` and ``explain()`` show each
step's variables, axes, and types, with ``?`` for a size the data decides;
the frame records the verb. ``preview()`` runs the verbs on a sample of
blocks, by default the first, middle, and last. An error found while a
block runs names the block. When a result is asked for (``collect()``, ``pull``,
``to_dict()``, or a verb that needs the whole axis), the recorded verbs run
over every block, ``threads`` blocks at a time: a pipeline ending in
``summarise`` merges the blocks' summaries into one table, so memory is the
blocks in flight and the groups, not the data; any other pipeline
concatenates the blocks' rows, for data that fits once filtered.

``scan(*frames)`` gives several frames' results from one pass over their
shared source: the frames' row-wise verbs form a trie, a verb matching
another by its structure (``tidy3a.identity``), so pipelines written apart
share the steps they have in common; each block is loaded
once and walked depth first with the frame after each shared prefix kept
while its branches run, and each frame takes its summaries or its rows from
where its own verbs branch off. A frame's result is the same bits as its
own run; the float32 sums the error bound cannot settle get one shared
second pass. Asking one frame for its result is a scan of that frame.

The blocks come from a source (``FileSource``, ``ArraySource``,
``DatasetSource``): a file reopened per block, an in-memory array sliced
per block, or an xarray dataset, lazy or not, whose block is computed and
wrapped as ``tidy()`` wraps it; ``"chunks"`` takes a Dask-backed dataset's
own chunk boundaries as the blocks, so each chunk is computed once. The
blocked axis is an observation axis, and a block holds every other axis
whole, so a block of time steps is that many whole grids.

What a block can answer: ``select``, ``filter``, and ``mutate`` of
row-wise expressions; one ``group_by`` of value keys (plain fields,
``cells``, ``bins``, ``rounded``, several at once, with or without
``dropna``); and a ``summarise`` of ``n()``, ``sum``, ``mean``, ``min``,
``max``, ``std``, ``var``, ``any``, and ``all`` of row-wise expressions, or
expressions of those (``mean(x) / n()``). Each key's groups are merged
across blocks in the order the key gives them (a cell's low edge, a bin's
place among the edges, a value's order), the missing group last, and
several keys combine as one code per occurring combination, as
``group_by`` does in memory. Float32 sums and means are the same bits as
in memory whatever the block size (``labelarray.sums``: the blocks'
estimates add, and the groups the error bound cannot settle get a second
pass for their exact sums). Float64 and integer sums and means, and the
spreads, add per block in float64 or int64 and are repeatable, not
bit-identical with the in-memory results. Anything else is refused with a
``BlockError`` when the verb is called, naming what needs the whole axis.
"""

from __future__ import annotations

import copy
import dataclasses
from contextvars import ContextVar
import math
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from labelarray.errors import LabelArrayError
from labelarray.grouped import merge_sorted
from labelarray.transforms import Bins, Duration, GridCells

from tidy3a.errors import BlockError, FilterError, MutateError, ReadError, SummariseError, TidyError, translating
from tidy3a.expr import (
    EVERYTHING,
    AxisKey,
    BinOp,
    Call,
    Context,
    Literal,
    MultiKey,
    Reduce,
    Ref,
    UnaryOp,
    ValueKey,
    _label_array,
    as_expr,
    evaluate,
    recording_reads,
    resolve_value_key,
    walk,
)
from tidy3a.frame import TidyArray
from tidy3a.plan import Record, Snapshot, Step, _variable, snapshot
from tidy3a.reductions import REGISTRY, WHOLE_AXIS, pass_budget

JOINS = ("left_join", "inner_join", "full_join")  # with data in memory, a block at a time (milestone 19)
SLICES = ("slice", "slice_head", "slice_tail")
ROW_WISE = ("select", "filter", "mutate", *JOINS, "slice_axes", "arrange_axes", "arrange_block")
# slice_axes: a slice of other axes (milestone 19); arrange_axes: a sort of another axis, which every block holds whole;
# arrange_block: each block's rows sorted, for a sort of the blocked axis to merge (milestone 21)


# ── Sources ─────────────────────────────────────────────────────────────────


class Source:
    """Where blocks come from: ``size`` positions on ``axis``, a block of them as a frame, and natural chunks.

    ``empty()`` is a block of no positions with the source's schema, every variable, axis, and type, built without
    reading the data (plan decision 4: the verbs are checked on it). ``block(..., names)`` may compute only the
    variables named (plan milestone 12); a source whose rows hold every field together reads them all anyway.
    """

    axis: str
    size: int
    chunks: list[tuple[int, int]] | None = None  # the storage's own block boundaries, if it has any

    def block(self, start: int, stop: int, names: frozenset | None = None) -> TidyArray:
        raise NotImplementedError

    def empty(self, names: frozenset | None = None) -> TidyArray:
        """*names*: the variables a run reads, for a source whose blocks hold only those (``block``'s)."""
        raise NotImplementedError

    def stored(self, names: frozenset | None) -> tuple[list[tuple[int, int]], bool] | None:
        """The storage's chunk boundaries along the axis for the variables *names*, and whether a block within a chunk
        computes the whole chunk (Dask's); None for storage with none (plan milestone 14, step 3)."""
        return None

    def labels(self) -> np.ndarray:
        """The axis's labels, without reading the data: its positions, as the blocks are labelled with them."""
        return np.arange(self.size)

    def block_labels(self, start: int, stop: int) -> np.ndarray:
        """A block's labels on the axis, without reading its data: its 1-based positions, as ``Blocks.load`` labels
        them."""
        return np.arange(start + 1, stop + 1)


class FileSource(Source):
    """A file memory-mapped again for each block; the block is a copy, and the map closes with it."""

    def __init__(self, reopen: Callable[[], np.ndarray], size: int, axis: str,
                 build: Callable[[np.ndarray, int, int], TidyArray], row: tuple[np.dtype, tuple[int, ...]]):
        self.reopen, self.size, self.axis, self.build = reopen, size, axis, build
        self.row = row  # the values' type, in native byte order, and the shape of one position

    def empty(self, names: frozenset | None = None) -> TidyArray:
        dtype, shape = self.row
        return self.build(np.empty((0, *shape), dtype), 0, 0)

    def block(self, start: int, stop: int, names: frozenset | None = None) -> TidyArray:
        mapped = self.reopen()
        values = np.array(mapped[start:stop])
        del mapped
        if not values.dtype.isnative:
            values = values.astype(values.dtype.newbyteorder("="))
        return self.build(values, start, stop)


class ArraySource(Source):
    """An array in memory; each block is a view of it, so what blocks bound is the verbs' intermediates."""

    def __init__(self, values: np.ndarray, axis: str, build: Callable[[np.ndarray, int, int], TidyArray]):
        self.values, self.size, self.axis, self.build = values, int(values.shape[0]), axis, build

    def block(self, start: int, stop: int, names: frozenset | None = None) -> TidyArray:
        return self.build(self.values[start:stop], start, stop)

    def empty(self, names: frozenset | None = None) -> TidyArray:
        return self.build(self.values[0:0], 0, 0)


class DatasetSource(Source):
    """An xarray Dataset or DataArray, lazy or not: a block is its positions on *axis*, computed, wrapped by ``tidy``."""

    def __init__(self, data: Any, axis: str, field_axis: str | None, backend: str | None):
        from tidy3a.backend import is_xarray_object

        if not is_xarray_object(data):
            raise TidyError(f"blocks= reads xarray objects in blocks, not {type(data).__name__}")
        if axis not in data.dims:
            axes = ", ".join(str(dim) for dim in data.dims) or "none"
            raise TidyError(f'blocks= names "{axis}", which is not an axis of this data (axes: {axes})')
        if axis == field_axis:
            raise TidyError(f'blocks= reads runs of "{axis}", which field_axis= turns into columns; '
                            "block an observation axis")
        self.data, self.axis, self.field_axis = data, axis, field_axis
        # The backend is chosen once, for the whole data, so a block holding some of its variables is on the same one.
        self.backend = self._backend() if backend is None else backend
        self.size = int(data.sizes[axis])
        self.chunks = _chunks_along(data, axis)

    def _backend(self) -> str:
        """The default backend for the blocks, chosen on the block of no positions, in memory: every block is
        computed into memory, so the data being lazy does not keep it on xarray, nor do storage encodings, which no
        summary keeps; what else the NumPy backend cannot hold (coordinates beyond an axis's labels, multi-level
        labels, times with a time zone) still does. Measured (milestone 16), NumPy ran a NetCDF4 or Zarr file's
        blocks 10 to 16% faster, with the same bits."""
        from tidy3a.backend import get_backend, xarray_backend

        backend = get_backend(None)
        return (backend if backend.unsupported(self._emptied(None)) is None else xarray_backend()).name

    def block(self, start: int, stop: int, names: frozenset | None = None) -> TidyArray:
        from tidy3a.frame import tidy

        return tidy(self._only(names).isel({self.axis: slice(start, stop)}).compute(), field_axis=self.field_axis,
                    backend=self.backend)

    def _only(self, names: frozenset | None) -> Any:
        """The data with only the variables *names* (and the coordinates that come with them); all of it for None or
        a DataArray. With none named (``n()`` alone), the variable along the axis with fewest values per position,
        so the block still has the axis."""
        data = self.data
        if names is None or not hasattr(data, "data_vars"):
            return data
        kept = [name for name in data.data_vars if name in names]
        if not kept:
            along = [name for name in data.data_vars if self.axis in data[name].dims]
            kept = [min(along, key=lambda name: data[name].size)] if along else []
        return data[kept] if kept else data

    def labels(self) -> np.ndarray:
        """The axis's labels, its index, which xarray holds in memory however lazy the data; else its positions."""
        if self.axis in self.data.coords:
            return np.asarray(self.data[self.axis].values)
        return np.arange(self.size)

    def block_labels(self, start: int, stop: int) -> np.ndarray:
        if self.axis in self.data.coords:
            return np.asarray(self.data[self.axis].values[start:stop])
        return np.arange(start + 1, stop + 1)

    def stored(self, names: frozenset | None) -> tuple[list[tuple[int, int]], bool] | None:
        """The Dask chunks of the variables *names* along the axis; else the chunks a file stores them in (xarray's
        ``preferred_chunks``, for NetCDF4 and Zarr), every variable's whole at once where their sizes differ."""
        data = self._only(names)
        chunks = _chunks_along(data, self.axis)
        if chunks is not None:
            return chunks, True
        variables = data.variables if hasattr(data, "data_vars") else {None: data.variable}
        sizes = {int(variable.encoding.get("preferred_chunks", {}).get(self.axis) or 0)
                 for variable in variables.values() if self.axis in variable.dims} - {0}
        if not sizes:
            return None
        size = math.lcm(*sizes)
        size = size if size <= self.size else max(sizes)
        return [(start, min(start + size, self.size)) for start in range(0, self.size, size)], False

    def empty(self, names: frozenset | None = None) -> TidyArray:
        """No positions on the axis: each lazy value along it an empty NumPy array of its type, since slicing lazy
        data to no positions and computing it still computes a chunk; nothing along the axis is computed. With
        *names*, only those variables, as ``block`` holds them."""
        from tidy3a.frame import tidy

        return tidy(self._emptied(names), field_axis=self.field_axis, backend=self.backend)

    def _emptied(self, names: frozenset | None) -> Any:
        """The xarray object of no positions, its lazy values along the axis empty NumPy arrays."""
        sliced = self._only(names).isel({self.axis: slice(0, 0)})
        dataset = hasattr(sliced, "data_vars")
        held = sliced.variables if dataset else sliced.coords.variables
        empty = {name: variable.copy(deep=False, data=np.empty(variable.shape, variable.dtype))
                 for name, variable in held.items()
                 if self.axis in variable.dims and name not in sliced.indexes
                 and not isinstance(variable.data, np.ndarray)}
        coords = {name: variable for name, variable in empty.items() if name in sliced.coords}
        if dataset:
            sliced = sliced.assign({name: variable for name, variable in empty.items() if name not in coords})
        elif not isinstance(sliced.variable.data, np.ndarray):  # a DataArray's own values
            sliced = sliced.copy(deep=False, data=np.empty(sliced.shape, sliced.dtype))
        return sliced.assign_coords(coords)


def _chunks_along(data: Any, axis: str) -> list[tuple[int, int]] | None:
    """A Dask-backed dataset's chunk boundaries along *axis*, or None when it is not chunked there."""
    try:
        chunks = getattr(data, "chunks", None)
    except ValueError:  # the variables' chunks along the axis disagree: no boundaries to take
        return None
    if not chunks:
        return None
    sizes = chunks.get(axis) if isinstance(chunks, Mapping) else chunks[data.dims.index(axis)]
    if not sizes:
        return None
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


@dataclass(frozen=True)
class Blocks:
    """The blocks of a source: its spans, how many run at once, and whether the blocked axis has labels of its own."""

    source: Source
    rows: int | str  # positions per block, "chunks" for the source's own boundaries, or "auto" before it is chosen
    threads: int
    labelled: bool = False  # else each block is labelled with its positions, so a verb sees them as in memory
    chosen: tuple[int, ...] | None = None  # a sample's blocks (``preview``), counted from 0; else every block
    memory: int | None = None  # the budget in bytes an "auto" size is chosen for (plan milestone 14)
    parts: tuple[tuple[int, int], ...] | None = None  # the storage's chunks an "auto" size's blocks are made of
    tail: bool = False  # a run ends with a block of no positions, a one-pass full join's own (milestone 19)
    positions: bool = False  # rows joined keep the blocks' positions as labels: a slice's own blocks (milestone 19)
    prunes: tuple = ()  # filters reading only the axis's labels, which settle whether a block keeps anything
    skipped: frozenset = frozenset()  # the blocks a run need not read, none of whose positions those filters keep
    narrowed: tuple[tuple[int, int], ...] | None = None  # spans read in place of the blocks: a slice's positions

    @property
    def axis(self) -> str:
        return self.source.axis

    @property
    def size(self) -> int:
        return self.source.size

    @property
    def every(self) -> list[tuple[int, int]]:
        """Every block's span, which a block's number counts in."""
        if self.rows == "auto":
            raise RuntimeError('blocks="auto" has no spans until a size is chosen for the frames that run (sized)')
        if self.rows == "chunks":
            return self.source.chunks or [(0, self.size)]
        rows = int(self.rows)
        if self.parts is not None:
            return _whole(self.parts, rows) or [(0, 0)]
        return [(start, min(start + rows, self.size)) for start in range(0, self.size, rows)] or [(0, 0)]

    @property
    def spans(self) -> list[tuple[int, int]]:
        """The spans a run reads: every block's, or a sample's."""
        if self.narrowed is not None:
            return list(self.narrowed)
        every = self.every
        if self.chosen is not None:
            return [every[number] for number in self.chosen]
        kept = [span for number, span in enumerate(every) if number not in self.skipped] if self.skipped else every
        return [*kept, (self.size, self.size)] if self.tail else kept

    def sized(self, frames: Sequence[ChunkedFrame]) -> Blocks:
        """These blocks for *frames*, which run together: with ``blocks="auto"``, the most positions whose blocks, as
        the frames' verbs run on them, fit the budget by ``tidy3a.sizing``'s estimate (plan milestone 14); a size
        given, as it is. Nothing is read, and the same frames, budget, and threads give the same size."""
        if self.rows != "auto":
            return self
        from tidy3a.sizing import RESERVE, cost, describe

        assert self.memory is not None
        estimate = cost(frames)
        rows = estimate.rows(self.memory, self.threads)
        if rows < 1:
            each = estimate.per_position(self.threads)
            fewer = ", fewer threads" if self.threads > 1 else ""
            raise BlockError(f'blocks="auto" fits no block of {self.axis} in memory={describe(self.memory)}: one '
                             f"position takes about {describe(each)} as these verbs run on {self.threads} "
                             f"thread{'s' if self.threads > 1 else ''}, beyond the "
                             f"{describe(RESERVE + self.threads * estimate.fixed)} set aside; give memory= more"
                             f"{fewer}, or a number of positions with blocks=")
        rows = max(1, min(rows, self.size))
        needs = [_needs(frame) for frame in frames]
        stored = self.source.stored(None if any(need is None for need in needs) else frozenset().union(*needs))
        if stored is None:
            return dataclasses.replace(self, rows=rows)
        parts, whole = stored
        largest = max(stop - start for start, stop in parts)
        if whole and largest > rows:
            raise BlockError(f'blocks="auto" fits {rows:,} positions of {self.axis} in memory={describe(self.memory)}, '
                             f"and the data's Dask chunks hold up to {largest:,}, each computed whole for any of its "
                             f'positions; rechunk it smaller, data.chunk({{"{self.axis}": {rows}}}), or give memory= '
                             "more")
        return dataclasses.replace(self, rows=rows, parts=tuple(parts))

    def load(self, start: int, stop: int, names: frozenset | None = None) -> TidyArray:
        block = self.source.block(start, stop, names)
        if self.labelled:
            return block
        return block._evolve(block._backend.label_positions(block._data, self.axis, start))

    def words(self, counted: str = "") -> str:
        """How many blocks of what: "7 blocks of 3,000 positions on point", or "7 on point, the data's own chunks"."""
        count = f"{len(self.every):,}{counted}"
        if self.rows == "chunks":
            return f"{count} on {self.axis}, the data's own chunks"
        if self.memory is not None:
            from tidy3a.sizing import describe

            if self.parts is not None:
                whole = max(stop - start for start, stop in self.parts) <= int(self.rows)
                return (f"{count} of up to {int(self.rows):,} positions on {self.axis}, "
                        f"{'whole chunks of its storage' if whole else 'its storage chunks split to fit'}, "
                        f"sized for memory={describe(self.memory)}")
            return f"{count} of {int(self.rows):,} positions on {self.axis}, sized for memory={describe(self.memory)}"
        return f"{count} of {int(self.rows):,} positions on {self.axis}"


def _whole(parts: Sequence[tuple[int, int]], rows: int) -> list[tuple[int, int]]:
    """Blocks of whole consecutive *parts*, each as many as fit in *rows* positions; a part larger than that in blocks
    of *rows*, which a file's chunk can be read in."""
    spans: list[tuple[int, int]] = []
    start = stop = None
    for low, high in parts:
        if start is not None and high - start > rows:
            spans.append((start, stop))
            start = None
        if high - low > rows:
            spans += [(at, min(at + rows, high)) for at in range(low, high, rows)]
            continue
        start = low if start is None else start
        stop = high
    if start is not None:
        spans.append((start, stop))
    return spans


def check_rows(rows: Any, threads: Any, source: Source, memory: Any = None) -> int | None:
    """``blocks=`` is a positive number of positions, "chunks" for a source that has them, or "auto" for a size chosen
    for ``memory=`` (the budget in bytes, given back); ``memory=`` goes with "auto" only."""
    if memory is not None and rows != "auto":
        raise ReadError(f'memory= is the budget blocks="auto" sizes blocks for; with blocks={rows!r} the size is '
                        "given, so leave memory= out")
    if rows == "auto":
        pass
    elif rows == "chunks":
        if source.chunks is None:
            raise ReadError('blocks="chunks" takes a Dask-backed dataset\'s chunks along the axis; '
                            f"this data is not chunked along {source.axis}, so give a number of positions")
    elif isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or rows < 1:
        raise ReadError(f'blocks= must be a positive number of positions, "chunks", or "auto", not {rows!r}')
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ReadError(f"threads= must be a positive number, not {threads!r}")
    if rows != "auto":
        return None
    from tidy3a.sizing import budget

    return budget(memory)


class ChunkedFrame(TidyArray):
    """Data read with ``blocks=``: an empty block with its schema, and the verbs to run over every block."""

    __slots__ = ("_blocks", "_steps", "_plan", "_result", "_reads", "_sized")

    @classmethod
    def open(cls, source: Source, rows: int | str, threads: int, memory: Any = None) -> ChunkedFrame:
        """A frame holding the source's empty block: nothing is read until a result is asked for (plan decision 4)."""
        budget = check_rows(rows, threads, source, memory)
        empty = source.empty()
        blocks = Blocks(source, rows, threads, empty._backend.has_labels(empty._data, source.axis), memory=budget)
        frame = cls._wrap(empty, blocks, (), reads=frozenset())
        layout = frame._layout()
        count = len(layout.every)
        frame._record = Record(empty._source or "data", _unknown(snapshot(empty)),
                               preview=f"nothing read yet: {layout.words(' block' if count == 1 else ' blocks')}")
        return frame

    @classmethod
    def _wrap(cls, frame: TidyArray, blocks: Blocks, steps: Sequence[Any], plan: Any = None,
              reads: frozenset | None = None) -> ChunkedFrame:
        """*reads* are the variables the steps read, as the empty block showed them; None reads everything."""
        new = cls.__new__(cls)
        TidyArray.__init__(new, frame._data, backend=frame._backend, field_axis=frame._field_axis,
                           origins=frame._origins, source=frame._source, notes=frame._notes, groups=frame._groups,
                           resolved=frame._resolved, record=frame._record)
        new._blocks, new._steps, new._plan, new._result, new._reads = blocks, tuple(steps), plan, None, reads
        new._sized = None
        return new

    def _layout(self) -> Blocks:
        """The blocks this frame's own run reads: a size given, or one chosen for its verbs (``Blocks.sized``) once and
        kept, which ``repr``, ``preview()``, and the run itself use."""
        if self._sized is None:
            self._sized = self._blocks.sized([self])
        return self._sized

    def _apply_verb(self, verb: Any) -> Any:
        operation = getattr(verb, "_operation", None)
        if operation == "write" and not self._summarised and not self._sorting:  # rows written as the blocks arrive,
            return _write(self, verb)  # never all in memory (a sort's rows are merged first, below)
        blocks = self._blocks
        if self._sorting:  # after a sort of the blocked axis: a slice of it becomes each block's part of it
            last = self._steps[-1]
            if (operation in ("slice_head", "slice_tail") and last._arguments[1] is None
                    and set(verb._arguments) == {blocks.axis}):
                count = verb._arguments[blocks.axis]
                if isinstance(count, (int, np.integer)) and not isinstance(count, bool) and count >= 0:
                    unsorted = self._wrap(self, blocks, self._steps[:-1], self._plan, self._reads)
                    unsorted._record = dataclasses.replace(self._record, steps=self._record.steps[:-1])
                    from tidy3a.verbs import arrange

                    return _arranged(unsorted, arrange(*last._arguments[0]), int(count), operation == "slice_tail",
                                     f"{last.text} >> {verb.text}")
            if operation == "write":  # rows that need not fit: spilled a run a block, then merged into the file
                written = _write_sorted(self, verb)
                if written is not None:
                    return written
            return verb(self._run())  # anything else on the merged rows, in memory
        if operation in SLICES and not self._summarised:
            if blocks.axis in _slice_axes(verb):  # the blocked axis: the blocks holding what it keeps, in memory
                return _sliced(self, verb)
            from tidy3a.verbs import Verb

            verb = Verb(verb.text, verb._apply, operation="slice_axes", arguments=(operation, verb._arguments))
            operation = "slice_axes"  # other axes: a step each block runs, as every block holds them all
        if operation == "arrange" and not self._summarised:
            if _arrange_axis(self, verb) == blocks.axis:  # the blocked axis: each block sorted, then merged, in memory
                return _arranged(self, verb)
            from tidy3a.verbs import Verb

            verb = Verb(verb.text, verb._apply, operation="arrange_axes", arguments=verb._arguments)
            operation = "arrange_axes"  # another axis: a step each block runs, as every block holds it whole
        if operation in JOINS and not self._summarised:
            whole = _whole_join(self, verb)
            paired = None if whole is not None else _by_position(self, verb)
            if paired is not None:
                verb = paired
            elif whole is None and operation == "full_join" and _along(self, verb):
                one = _one_pass(self, verb)
                if isinstance(one, str):
                    whole = one
                else:
                    verb, blocks = one, dataclasses.replace(blocks, tail=True)
            if whole is not None:  # the blocks' rows in memory, joined there, the result saying why
                result = verb(self._run())
                return result._evolve(notes=[f"{verb.text}: joined on the whole axis, as {whole}"])
        if self._summarised or operation not in (*ROW_WISE, "group_by", "summarise"):
            return verb(self._run())  # needs the whole axis: the blocks' result, in memory
        if operation == "select":
            _refuse_value_selectors(verb)  # before the verb runs on the empty block, which would answer for no values
        if operation == "summarise":
            _refuse_unblocked(verb)  # before a registered reduction's own function runs on the empty block
        with warnings.catch_warnings(), recording_reads() as read:  # a mean of nothing warns; here that says nothing
            warnings.simplefilter("ignore", RuntimeWarning)
            result = TidyArray._apply_verb(self, verb)  # on the empty block: the eager checks, and the result's schema
        steps = (*self._steps, verb)
        plan = _check(self, verb, result, steps)
        reads = None if self._reads is None or EVERYTHING in read else self._reads | frozenset(read)
        if operation == "filter" and read and EVERYTHING not in read and set(read) <= {blocks.axis}:
            blocks = dataclasses.replace(blocks, prunes=(*blocks.prunes, verb))  # a block's labels settle it
        frame = self._wrap(result, blocks, steps, plan, reads)
        layout = frame._layout()  # a size chosen now, so a verb that leaves no block within the budget is refused here
        record = frame._record
        last = record.steps[-1]
        frame._record = dataclasses.replace(record, steps=(*record.steps[:-1], Step(last.verb, _unknown(last.after))))
        if self._blocks.rows == "auto":  # the size chosen for these verbs, which explain() shows
            count = len(layout.every)
            frame._record = dataclasses.replace(
                frame._record, preview=f"nothing read yet: {layout.words(' block' if count == 1 else ' blocks')}")
        return frame

    @translating
    def preview(self, blocks: Sequence[int] | None = None) -> TidyArray:
        """The verbs run on a sample of blocks, the result in memory, labelled as a sample; the frame is unchanged.

        By default the first, middle, and last blocks, so the sample is not only
        the start of the data; ``blocks=`` names others, counted from 1 as an
        error names them. It reads those blocks, and only those.
        """
        layout = self._layout()
        every = len(layout.every)
        if blocks is None:
            chosen = sorted({0, every // 2, every - 1})
        else:
            numbers = [blocks] if isinstance(blocks, (int, np.integer)) else list(blocks)
            wrong = [number for number in numbers
                     if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or not 1 <= number <= every]
            if wrong or not numbers:
                raise BlockError(f"preview(blocks=...) takes block numbers from 1 to {every:,}, not {wrong or blocks!r}")
            chosen = sorted({int(number) - 1 for number in numbers})
        sample = dataclasses.replace(layout, chosen=tuple(chosen))
        spans = sample.spans
        shown = ", ".join(str(number + 1) for number in chosen)
        words = (f"a sample: block{'' if len(chosen) == 1 else 's'} {shown} of {every:,} "
                 f"({sum(stop - start for start, stop in spans):,} of {self._blocks.size:,} positions on "
                 f"{self._blocks.axis}); collect() runs every block")
        frame = self._wrap(self, sample, self._steps, self._plan, self._reads)
        frame._record = dataclasses.replace(frame._record, preview=words)
        _scan([frame])
        result = frame._result
        return result._evolve(notes=[*result._notes, f"preview: {words}"])

    @property
    def _summarised(self) -> bool:
        return bool(self._steps) and self._steps[-1]._operation == "summarise"

    @property
    def _sorting(self) -> bool:
        """Whether its last step sorts each block for a sort of the blocked axis, which its run merges."""
        return bool(self._steps) and self._steps[-1]._operation == "arrange_block"

    def _run(self) -> TidyArray:
        """The recorded verbs over every block: the merged summaries, or the rows kept, in memory, computed once."""
        return scan(self)[0]

    @translating
    def collect(self) -> Any:
        """Run the verbs over every block and give the merged result as an xarray Dataset."""
        return self._run().collect()

    @translating
    def to_xarray(self) -> Any:
        return self._run().to_xarray()

    @translating
    def to_dict(self) -> dict:
        return self._run().to_dict()

    def __tidy__(self, **options: Any) -> Any:
        return self._run().__tidy__(**options)

    def explain(self) -> Any:
        """Each step's effect on the schema (``?`` for sizes the data decides), and how a summary's blocks are planned
        to run (``tidy3a.fused``), each step that runs on NumPy and why; or, once a result has been computed, each
        step's effect on every block, and how the blocks ran."""
        if self._result is not None:
            return self._result.explain()
        plan = TidyArray.explain(self)
        if self._summarised:
            from tidy3a.fused import plan_for

            plan.tail = tuple(plan_for(self).lines())
        return plan

    @translating
    def __repr__(self) -> str:
        if self._result is not None:
            return repr(self._result)
        last = self._record.steps[-1].after if self._record is not None and self._record.steps else None
        shown = last if last is not None else self._record.start if self._record is not None else _unknown(
            snapshot(self))
        lines = [f"<TidyArray source: {self._source or 'data'}; nothing read yet>"]
        lines += [_variable(entry) for entry in shown.variables]
        if shown.field_axis is not None:
            lines.append(f"field axis: {shown.field_axis}")
        if shown.groups:
            lines.append(f"groups: {', '.join(shown.groups)}")
        lines += [note for note in shown.notes]
        if self._summarised:
            from tidy3a.fused import plan_for

            lines.append(f"planned run: {plan_for(self).words()}")
        lines.append(f"blocks: {self._layout().words()}; ? is a size the data decides; preview() runs a sample, and "
                     "collect() or pull() every block")
        return "\n".join(lines)


def _emptied(note: str) -> bool:
    """A note only the empty block's emptiness wrote: a filter on no positions removes them all."""
    return "(all removed by " in note


def _unknown(shot: Snapshot) -> Snapshot:
    """A snapshot of a frame on the empty block, with the sizes the data decides unknown (None, shown as ?): those that
    are 0 there, the blocked axis and the groups a key finds; and without the notes its emptiness wrote."""
    sizes = tuple((dim, None if size == 0 else size) for dim, size in shot.sizes)
    variables = tuple((name, dims, tuple(None if size == 0 else size for size in shape), dtype)
                      for name, dims, shape, dtype in shot.variables)
    return dataclasses.replace(shot, variables=variables, sizes=sizes,
                               notes=tuple(note for note in shot.notes if not _emptied(note)))


# ── Checks at verb time ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Plan:
    key: ValueKey | None  # the group_by key: a plain field, a transform, or several keys (MultiKey)
    parts: tuple[ValueKey, ...]  # the key itself, or each key of several
    reductions: dict[str, Reduce]  # hidden name -> the reduction combined across blocks
    summaries: dict[str, Any]  # name -> its expression, with each reduction replaced by its hidden name
    per_position: int  # positions on the other observation axes that n() counts per row, from the whole schema


def _refuse_value_selectors(verb: Any) -> None:
    """A ``where(predicate)`` of the caller's picks columns by their values: on the empty block it would answer for
    no values, and each block could answer differently, so a frame read in blocks refuses it."""
    from tidy3a.selectors import Exclude, Selector

    def found(item: Any) -> Any:
        if isinstance(item, (list, tuple)):
            return next((inner for inner in map(found, item) if inner is not None), None)
        if isinstance(item, Exclude):
            return found(item.item)
        return item if isinstance(item, Selector) and item._settings is None else None

    selector = found(verb._arguments)
    if selector is not None:
        raise BlockError(f"{verb.text}: {selector.text()} picks columns by their values, and a frame read in blocks "
                         "is checked before any value is read, and each block could pick others; select the columns "
                         "by name or type (starts_with(...), where(is_numeric)), or preview() a sample to see them "
                         "and select them by name")


def _refuse_unblocked(verb: Any) -> None:
    """A reduction a block cannot answer, refused before ``summarise`` runs on the empty block (a registered
    reduction without block steps), as ``_hidden`` refuses it after."""
    _, named = verb._arguments
    for name, expr in named.items():
        for node in walk(as_expr(expr)):
            if isinstance(node, Reduce) and node.op not in REGISTRY:
                raise BlockError(f"{name} = {node.text()}: {node.op}() cannot be combined across blocks; "
                                 f"{', '.join(REGISTRY)} can (register_reduction() adds one, with block steps); "
                                 f"{WHOLE_AXIS}")


def _slice_axes(verb: Any) -> dict:
    """The axes a slice names, with what it keeps of each."""
    return verb._arguments[1] if verb._operation == "slice" else verb._arguments


def _sliced(frame: ChunkedFrame, verb: Any) -> TidyArray:
    """A slice of the blocked axis, in memory, from only the blocks holding what it keeps (milestone 19). Where the
    positions are the source's, nothing before the slice removing any, those blocks are known before reading; where a
    filter or an inner join decides them, blocks are read from the start (from the end for ``slice_tail``), one,
    then more at a time, until enough rows are kept, as Spark's ``LIMIT`` scales up the partitions it reads. Known
    positions are read alone, in runs (``_runs``), not the blocks holding them. A slice excluding positions, or
    after a one-pass full join, reads every block."""
    import bisect

    from tidy3a import verbs

    operation, axes = verb._operation, _slice_axes(verb)
    layout = frame._layout()
    axis, every, size = layout.axis, layout.every, layout.size
    spec = axes[axis]
    if operation == "slice":
        if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
            wanted = [int(spec)]
        elif isinstance(spec, (list, tuple, range, np.ndarray)):
            wanted = [int(position) for position in spec]
        else:
            return verb(frame._run())  # the verb's own message
        if not wanted or any(position <= 0 for position in wanted):  # excluded, or refused: every block
            return verb(frame._run())
    elif isinstance(spec, bool) or not isinstance(spec, (int, np.integer)) or spec < 0:
        return verb(frame._run())  # the verb's own message
    if layout.tail:
        return verb(frame._run())
    known = not any(step._operation in ("filter", "inner_join") for step in frame._steps)
    starts = [start for start, _ in every]
    if known:
        if operation == "slice_head":
            index = list(range(min(int(spec), size)))
        elif operation == "slice_tail":
            index = list(range(size - min(int(spec), size), size))
        else:
            index = [position - 1 for position in wanted if position <= size]
        if not index:  # nothing kept: the slice of the block of no positions, nothing read
            return verb._apply(frame)
        spans = _runs(sorted(set(index)), starts)
        rows = _rows_of(frame, dataclasses.replace(layout, narrowed=tuple(spans)), None)
        offsets, at = [], 0
        for start, stop in spans:
            offsets.append(at)
            at += stop - start
        firsts = [start for start, _ in spans]
        relative = [offsets[number] + position - firsts[number]
                    for position in index for number in [bisect.bisect_right(firsts, position) - 1]]
        kept = rows._evolve(rows._backend.keep(rows._data, axis, relative))
        rest = {**axes, axis: list(range(1, len(relative) + 1)) if operation == "slice" else len(relative)}
        again = verbs.slice(drop=verb._arguments[0], **rest) if operation == "slice" else \
            getattr(verbs, operation)(**rest)
        return again(kept)
    need = int(spec) if operation != "slice" else max(wanted)
    order = list(range(len(every)))
    if operation == "slice_tail":
        order.reverse()
    read: list[int] = []
    parts: list[TidyArray] = []
    held = 0
    while order and held < need:  # one block, then more at a time: twice what was read, or a block a thread
        take = 1 if not read else max(layout.threads, len(read))
        batch, order = order[:take], order[take:]
        part = _rows_of(frame, layout, sorted(batch))
        held += part._backend.sizes(part._data).get(axis, 0)
        read += batch
        parts = [part, *parts] if operation == "slice_tail" else [*parts, part]
    backend = frame._backend
    rows = parts[0]._evolve(backend.join_along([part._data for part in parts], axis)) if len(parts) > 1 else parts[0]
    return verb(rows)


def _runs(positions: Sequence[int], starts: Sequence[int]) -> list[tuple[int, int]]:
    """The spans to read for sorted *positions*: each run of them within a block, gaps of up to ``GAP`` positions
    read through, so a slice of ten positions reads ten, not the block holding them, and scattered ones few spans."""
    import bisect

    spans: list[list[int]] = []
    for position in positions:
        block = bisect.bisect_right(starts, position) - 1
        if spans and spans[-1][2] == block and position - spans[-1][1] <= GAP:
            spans[-1][1] = position + 1
        else:
            spans.append([position, position + 1, block])
    return [(start, stop) for start, stop, _ in spans]


GAP = 4_096  # positions a slice's span reads through rather than starting another


def _arrange_axis(frame: ChunkedFrame, verb: Any) -> Any:
    """The axis ``arrange``'s keys vary along, found on the block of no positions (its own checks run on it too);
    None where it has no keys."""
    from tidy3a.verbs import Desc

    TidyArray._apply_verb(frame, verb)  # the refusals, as in memory, when called
    ctx = Context(frame, verb="arrange", error=TidyError)
    for key in verb._arguments:
        value = evaluate(as_expr(key.key if isinstance(key, Desc) else key), ctx)
        return value.dims[0]
    return None


def _arranged(frame: ChunkedFrame, verb: Any, first: int | None = None, last: bool = False,
              shown: str | None = None) -> ChunkedFrame:
    """A sort of the blocked axis (milestone 21), still read in blocks: each block's kept rows sorted on their own, on
    the pool, a row-wise step (``arrange_block``), and, once the blocks are in, merged (``_merged_sort``). With
    *first*, a slice after the sort: each block keeps only its first (or, *last*, its last) *first* rows of the order,
    found by a partial sort, from which the merge takes the overall ones."""
    from tidy3a.verbs import Verb, sorted_frame

    keys, text = verb._arguments, verb.text
    each = Verb(shown or text, lambda block: sorted_frame(block, keys, text, first, last), operation="arrange_block",
                arguments=(keys, first, last))
    return frame._apply_verb(each)


def _write_sorted(frame: ChunkedFrame, verb: Any) -> TidyArray | None:
    """A sort written (milestone 21, step 4): each block's sorted rows spilled to a run of its own, then the runs
    merged into the file (``tidy3a.external``); None where a key is text, which has no merge keys, and the merged rows
    are written from memory."""
    from tidy3a.external import SortedWriter
    from tidy3a.verbs import merge_keys, sort_columns

    keys, first, last = frame._steps[-1]._arguments
    _, columns, _ = sort_columns(frame, keys)  # on the block of no positions: the keys' types
    if merge_keys(columns) is None:
        return None
    path, overwrite = verb._arguments
    writer = SortedWriter(_writer(frame, path, overwrite), frame, keys, first, last)
    _scan([], [(frame, writer)])
    return writer.reopen()


def _merged_sort(frame: ChunkedFrame, rows: TidyArray) -> TidyArray:
    """The sorted blocks' rows, joined in their order, sorted again: NumPy's stable sort finds them in runs and merges
    them, earlier blocks first among ties, so the order is memory's; and sliced as the sort's slice asks."""
    from tidy3a import verbs

    keys, first, last = frame._steps[-1]._arguments
    sort = verbs.arrange(*keys)
    text = sort.text
    rows = sort(rows)
    if first is None:
        return rows._evolve(notes=[*rows._notes, f"{text}: each block sorted, then the blocks merged"])
    axis = frame._blocks.axis
    rows = (verbs.slice_tail if last else verbs.slice_head)(**{axis: first})(rows)
    words = "last" if last else "first"
    return rows._evolve(notes=[*rows._notes, f"{text}: each block's {words} {first:,} found, then the blocks merged"])


def _rows_of(frame: ChunkedFrame, layout: Blocks, chosen: Sequence[int] | None) -> TidyArray:
    """The rows *frame*'s verbs keep from the blocks *chosen* (else the spans the layout narrows the run to), in
    memory, the blocks' positions their labels."""
    chosen = None if chosen is None else tuple(chosen)
    sample = frame._wrap(frame, dataclasses.replace(layout, chosen=chosen, positions=True), frame._steps,
                         frame._plan, frame._reads)
    _scan([sample])
    return sample._result


def _whole_join(frame: ChunkedFrame, verb: Any) -> str | None:
    """Why a join of *frame*'s blocks runs on the whole axis, or None where each block joins on its own (milestone
    19): the other data read in blocks too, or a full join along the blocked axis, which adds labels no block holds."""
    from tidy3a.frame import tidy

    other = verb._arguments[0]
    if isinstance(other, ChunkedFrame):
        return "the other data is read in blocks too"
    return None


def _along(frame: ChunkedFrame, verb: Any) -> bool:
    """Whether a join's other data has the blocked axis."""
    from tidy3a.frame import tidy

    other = verb._arguments[0]
    right = other if isinstance(other, TidyArray) else tidy(other)
    return right._backend.has_axis(right._data, frame._blocks.axis)


def _by_position(frame: ChunkedFrame, verb: Any) -> Any:
    """A join whose other data is unlabelled on the blocked axis, as the data is still (a filter labels it with its
    positions, as in memory), or None: in memory the two pair by position, the same length; each block joins the
    other data's part at its own positions, labelled with them as the block is (``Blocks.load``), so every join
    keeps them all as in memory."""
    from tidy3a import verbs
    from tidy3a.errors import AlignError
    from tidy3a.frame import tidy

    blocks, backend = frame._blocks, frame._backend
    other, suffix, tolerance = verb._arguments
    right = other if isinstance(other, TidyArray) else tidy(other, backend=backend.name)
    axis = blocks.axis
    if backend.has_labels(frame._data, axis) or right._backend is not backend \
            or not backend.has_axis(right._data, axis) or backend.has_labels(right._data, axis):
        return None
    length = backend.sizes(right._data)[axis]
    if length != blocks.size:
        raise AlignError(f"{axis} is unlabelled on both sides with different lengths ({blocks.size:,} and "
                         f"{length:,}); positions cannot be paired")
    numbered = backend.label_positions(right._data, axis, 0)
    join = getattr(verbs, verb._operation)

    def placed(block: TidyArray) -> TidyArray:
        if not backend.has_labels(block._data, axis):  # the block of no positions, checked as in memory
            part = right._evolve(backend.isel(right._data, {axis: slice(0, 0)}))
        else:
            mine = np.asarray(backend.axis_labels(block._data, axis).values, dtype=np.int64)
            part = right._evolve(backend.isel(numbered, {axis: mine - 1}))
        return join(part, suffix=suffix, tolerance=tolerance)._apply(block)

    from tidy3a.verbs import Verb

    return Verb(verb.text, placed, operation=verb._operation, arguments=verb._arguments)


def _one_pass(frame: ChunkedFrame, verb: Any) -> Any:
    """A full join along the blocked axis in one pass (milestone 19, after Spark's broadcast nested-loop join), or why
    not: each block full-joins the part of the other data at its own labels, so the other axes widen as in memory,
    its variables cast to the types the whole join gives them (``_promoted``); and a last block of no positions, the
    run's tail (``Blocks.tail``), joins the other data's labels no block holds. The blocked axis's labels are the
    source's, known before anything is read; a filter or an inner join before it would make them the data's."""
    from tidy3a.frame import tidy
    from tidy3a.verbs import Verb, full_join

    other, suffix, tolerance = verb._arguments
    axis, backend = frame._blocks.axis, frame._backend
    if tolerance is not None:
        return "a full join along the blocked axis matches labels exactly"
    if any(step._operation in ("filter", "inner_join") for step in frame._steps):
        return f"a full join along {axis} after a filter or an inner join keeps labels only the data decides"
    right = other if isinstance(other, TidyArray) else tidy(other, backend=backend.name)
    if right._backend is not backend:
        return "the other data is on another backend"
    ours, theirs = backend.variable_names(frame._data), right._backend.variable_names(right._data)
    if set(ours) & set(theirs):
        return "the two hold variables of the same names"
    labels = frame._blocks.source.labels()
    held = np.asarray(right._backend.axis_labels(right._data, axis).values)
    unheld = ~np.isin(held, labels)  # the other data's labels no block holds: the tail's
    missing = labels[~np.isin(labels, held)]  # the data's labels the other data lacks
    target = _promoted(frame, right, axis, bool(unheld.any()), missing[:1])

    def placed(block: TidyArray) -> TidyArray:
        mine = backend.axis_labels(block._data, axis).values
        chosen = np.flatnonzero(unheld) if _TAIL.get() else np.flatnonzero(np.isin(held, np.asarray(mine)))
        part = right._evolve(right._backend.isel(right._data, {axis: chosen}))
        joined = full_join(part, suffix=suffix)._apply(block)
        data = joined._data
        for name, dtype in target.items():
            if not backend.has_variable(data, name):  # a block holds only the variables a run reads
                continue
            value = backend.variable(data, name)
            if value.dtype != dtype:  # rebuilt in the type the whole join gives it, on its axes and their labels
                dims = tuple(value.dims)
                coords = {dim: backend.axis_labels(data, dim).values for dim in dims if backend.has_labels(data, dim)}
                cast = backend.array(np.asarray(value.values).astype(dtype), dims=dims, coords=coords, name=name)
                data = backend.assign(data, name, cast)
        return joined._evolve(data)

    return Verb(verb.text, placed, operation="full_join", arguments=verb._arguments)


def _promoted(frame: ChunkedFrame, right: TidyArray, axis: Any, unheld: bool, missing: np.ndarray) -> dict:
    """The types the whole full join gives each variable, as the backend fills its gaps: the data's where the other
    data holds labels no block does (the empty block joined with the other data shows them), the other data's where
    the data holds labels it lacks (it joined with one such label shows them)."""
    from tidy3a.verbs import full_join

    backend = frame._backend
    target: dict = {}
    if unheld:
        probe = full_join(right)._apply(frame)._data
        for name in backend.variable_names(frame._data):
            target[name] = backend.variable(probe, name).dtype
    if len(missing):
        label = backend.summary({"_": backend.array(np.zeros(1), dims=(axis,), coords={axis: missing}, name="_")},
                                {}, {})
        probe = backend.merge(label, right._data, "outer")
        for name in right._backend.variable_names(right._data):
            target[name] = backend.variable(probe, name).dtype
    return target


def _check(frame: ChunkedFrame, verb: Any, result: TidyArray, steps: Sequence[Any]) -> _Plan | None:
    operation = verb._operation
    if any(step._operation == "group_by" for step in frame._steps) and operation != "summarise":
        raise BlockError(f"{verb.text}: after group_by, only summarise() runs on blocks; a grouped {operation} "
                         f"needs every position of a group, which blocks split; {WHOLE_AXIS}")
    if operation in ("filter", "mutate"):
        values = [step.value for step in verb._arguments] if operation == "mutate" else verb._arguments
        error = MutateError if operation == "mutate" else FilterError
        axis = frame._blocks.axis
        for value in values:
            node = as_expr(value)
            if hasattr(node, "lower"):  # a formula: what it sums is known once it is read against the data
                node, _ = node.lower(Context(frame, verb=operation, error=error, grouped=False))
            for inner in walk(node):
                if isinstance(inner, Reduce):
                    raise BlockError(f"{verb.text}: {inner.text()} needs the whole axis, which blocks split; "
                                     f"summarise it per group instead, or {WHOLE_AXIS}")
                if isinstance(inner, Call) and inner.name == "dot":
                    over = inner.options.get("over")
                    over = [over] if isinstance(over, str) else list(over or [])
                    if not over or axis in over:
                        raise BlockError(f"{verb.text}: {inner.text()} sums over {axis}, which blocks split; "
                                         f"name other axes with over=, or {WHOLE_AXIS}")
        return None
    if operation == "group_by":
        for key in result._groups:
            if isinstance(key, AxisKey):
                raise BlockError(f"{verb.text}: group_by({key.label}) keeps an axis, which blocks split; "
                                 f"{WHOLE_AXIS}")
        for part in _parts(result._groups):
            if isinstance(part.transform, Duration):
                raise BlockError(f"{verb.text}: {part.call} places its windows from the first time present, "
                                 f"which depends on the whole axis; {WHOLE_AXIS}")
        return None
    if operation == "summarise":
        return _analyse(frame, verb, steps)
    return None


def _parts(keys: Sequence[Any]) -> tuple[ValueKey, ...]:
    """The value keys of a grouping, those of several keys one by one, each with the grouping's ``dropna``."""
    return tuple(part.with_dropna(key.dropna) if isinstance(key, MultiKey) else key
                 for key in keys if isinstance(key, ValueKey)
                 for part in (key.keys if isinstance(key, MultiKey) else (key,)))


def _analyse(frame: ChunkedFrame, verb: Any, steps: Sequence[Any]) -> _Plan:
    _, named = verb._arguments
    reductions: dict[str, Reduce] = {}
    summaries = {name: _hidden(as_expr(expr), reductions, name) for name, expr in named.items()}
    keys = [key for key in frame._groups if isinstance(key, ValueKey)]
    # n() counts the observation axes every variable shares, so it takes them from the schema the verbs ran on, not
    # from a block holding only the variables read (``hour`` along time alone would otherwise not be there).
    return _Plan(keys[0] if keys else None, _parts(frame._groups), reductions, summaries,
                 _per_position(frame, frame._blocks.axis))


def _hidden(node: Any, reductions: dict[str, Reduce], name: str) -> Any:
    """*node* with each reduction replaced by a hidden name, the reductions collected; refuses what cannot combine."""
    if isinstance(node, Reduce):
        if node.op not in REGISTRY:  # the reductions a block can answer, built in or registered with block steps
            raise BlockError(f"{name} = {node.text()}: {node.op}() cannot be combined across blocks; "
                             f"{', '.join(REGISTRY)} can (register_reduction() adds one, with block steps); "
                             f"{WHOLE_AXIS}")
        if node.over is not None:
            raise BlockError(f"{name} = {node.text()}: over= names axes, which blocks split; {WHOLE_AXIS}")
        if node.arg is not None and any(isinstance(inner, Reduce) for inner in walk(node.arg)):
            raise BlockError(f"{name} = {node.text()}: a reduction inside a reduction needs the whole axis; "
                             f"{WHOLE_AXIS}")
        hidden = f"__{len(reductions)}"
        reductions[hidden] = node
        return Ref(hidden)
    if isinstance(node, Literal):
        return node
    if isinstance(node, BinOp):
        return dataclasses.replace(node, left=_hidden(node.left, reductions, name),
                                   right=_hidden(node.right, reductions, name))
    if isinstance(node, UnaryOp):
        return dataclasses.replace(node, operand=_hidden(node.operand, reductions, name))
    if isinstance(node, Call):
        return dataclasses.replace(node, args=tuple(_hidden(arg, reductions, name) for arg in node.args))
    raise BlockError(f"{name} = {node.text()}: a summary on blocks is a reduction, or an expression of reductions "
                     f"such as mean(x) / n(); {WHOLE_AXIS}")


# ── Each key's groups, merged across blocks ─────────────────────────────────


@dataclass
class _Part:
    """One key of the grouping as the blocks have shown it: its labels in the result's order, and its missing group."""

    key: ValueKey
    labels: np.ndarray | None = None  # without the missing group
    missing: Any = None  # the missing group's label once a block had one (it is the last group), else None
    dropped: int = 0  # positions left out for a missing value of this key (``dropna``)

    @property
    def size(self) -> int:
        return (0 if self.labels is None else len(self.labels)) + (self.missing is not None)

    @property
    def subject(self) -> str:
        return self.key.label if self.key.transform is None else self.key.expr.text()

    def full(self) -> np.ndarray:
        """Every group's label, the missing one last."""
        if self.labels is not None and self.labels.dtype.kind in "mM":  # times: .tolist() gives nanoseconds as int
            missing = [] if self.missing is None else [self.missing]
            return np.concatenate([self.labels, np.array(missing, dtype=self.labels.dtype)])
        labels = [] if self.labels is None else self.labels.tolist()
        return _label_array(labels + ([self.missing] if self.missing is not None else []))

    def widen(self, labels: np.ndarray, missing: Any) -> np.ndarray | None:
        """Take a block's labels in; the old groups' new codes when the groups changed, else None."""
        before = self.size
        old = self.labels
        if missing is not None and self.missing is None:
            self.missing = missing
        if old is None:
            self.labels = labels
        elif isinstance(self.key.transform, Bins):  # every bin, in edge order, the same in every block
            if len(labels) != len(old) or (labels != old).any():
                raise BlockError(f"{self.key.call}: the bins differ between blocks")
        elif len(labels) and not (len(labels) <= len(old) and _codes_of(old, labels, self.key).min() >= 0):
            self.labels = _ordered(np.concatenate([old, labels]), self.key)
        if self.size == before:
            return None
        moved = np.arange(self.size - (self.missing is not None)) if old is None else _codes_of(self.labels, old,
                                                                                                  self.key)
        if before > len(moved):  # the old missing slot moves to the new last place
            moved = np.append(moved, self.size - 1)
        return moved

    def codes(self, labels: np.ndarray, has_missing: bool) -> np.ndarray:
        """A block's local codes mapped to this part's codes; the block's missing group, if any, is its last."""
        assert self.labels is not None
        found = _codes_of(self.labels, labels, self.key)
        return np.append(found, self.size - 1) if has_missing else found


def _ordered(labels: np.ndarray, key: ValueKey) -> np.ndarray:
    """The distinct *labels* in the order ``group_by`` gives them: by a cell's low edge, else by value."""
    distinct = np.unique(labels)
    if isinstance(key.transform, GridCells):  # "[low, high)": text order is not numeric order
        return distinct[np.argsort([float(str(label)[1:].split(",")[0]) for label in distinct], kind="stable")]
    return distinct


def _codes_of(ordered: np.ndarray, labels: np.ndarray, key: ValueKey) -> np.ndarray:
    """Where each of *labels* stands in *ordered*, or -1 where absent."""
    if ordered.dtype.kind in "iufmM" and labels.dtype.kind in "iufmM":
        at = np.searchsorted(ordered, labels)
        at[at >= len(ordered)] = 0
        return np.where(ordered[at] == labels, at, -1) if len(ordered) else np.full(len(labels), -1)
    position = {label: index for index, label in enumerate(ordered.tolist())}
    return np.array([position.get(label, -1) for label in labels.tolist()], dtype=np.intp)


def _is_missing_label(label: Any, key: ValueKey) -> bool:
    """Whether a group's label is the missing group's: NaN, NaT, or the text a transform gives it."""
    transform = key.transform
    if transform is not None and getattr(transform, "text_labels", False):
        return label in ("NaN", "NaT")
    if isinstance(label, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(label))
    return isinstance(label, float) and math.isnan(label)


class _Table:
    """The accumulating table: every combination of groups seen so far, sorted, with its counts and pieces."""

    def __init__(self, plan: _Plan):
        self.plan = plan
        self.parts = [_Part(part) for part in plan.parts]
        self.keys: np.ndarray | None = None  # one code per combination: the parts' codes, first key major
        self.counts = np.zeros(0, dtype=np.int64)
        self.pieces: dict[str, tuple[str, np.ndarray]] = {}
        self.blocks = 0  # blocks with rows, for the error bound of the combined float32 estimates
        self.dropped = 0  # rows left out for a missing key

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(part.size for part in self.parts) or (1,)

    def take(self, found: Sequence[tuple[np.ndarray, Any, int]]) -> None:
        """Widen each part by a block's groups, moving the table's codes along when its groups changed."""
        for part, (labels, missing, dropped) in zip(self.parts, found):
            part.dropped += dropped
            old = self.sizes
            moved = part.widen(labels, missing)
            if moved is not None and self.keys is not None:
                codes = list(np.unravel_index(self.keys, old))
                codes[self.parts.index(part)] = moved[codes[self.parts.index(part)]]
                self.keys = np.ravel_multi_index(codes, self.sizes)
        if float(np.prod(self.sizes, dtype=np.float64)) >= 2.0**62:
            raise BlockError("too many combinations of the keys' groups to number on blocks; " + WHOLE_AXIS)

    def add(self, keys: np.ndarray, counts: np.ndarray, pieces: dict[str, tuple[str, np.ndarray]],
            blocks: int = 1) -> None:
        """*blocks* blocks' combinations (sorted, in the table's codes), their counts, and each reduction's pieces,
        which add."""
        if len(keys) == 0:
            return
        self.blocks += blocks
        if self.keys is None:
            self.keys, self.counts, self.pieces = keys, counts, dict(pieces)
            return
        union, where, at = merge_sorted(self.keys, keys)  # one pass over both, both sorted (plan milestone 15)
        if len(union) != len(self.keys):
            self.counts = _spread(self.counts, where, len(union), 0)
            for hidden, (kind, held) in self.pieces.items():
                node = self.plan.reductions[hidden]
                fill = REGISTRY[node.op].fill(node, kind, held.dtype)
                self.pieces[hidden] = (kind, _spread(held, where, len(union), fill))
            self.keys = union
        self.counts[at] += counts
        for hidden, (kind, piece) in pieces.items():
            node, held = self.plan.reductions[hidden], self.pieces[hidden][1]
            held[..., at] = REGISTRY[node.op].merge(node, kind, held[..., at], piece)

    def merge(self, found: Sequence[tuple[np.ndarray, Any, int]], local: Any, counts: Any, pieces: Any,
              dropped: int) -> None:
        """One block's partial (``_partial``) taken into the table."""
        self.take(found)
        self.dropped += dropped
        if local is not None:
            self.add(_combined(local, self), counts, pieces)

    def combine(self, other: _Table) -> None:
        """*other* taken into this table, as if its blocks had been merged here after this table's (plan milestone
        15): each key's groups widened by other's, both tables' codes renumbered to match, and its counts and pieces
        merged by the reductions' own merges, which hold in any order."""
        theirs = list(np.unravel_index(other.keys, other.sizes)) if other.keys is not None and other.parts else None
        for index, (part, their) in enumerate(zip(self.parts, other.parts)):
            part.dropped += their.dropped
            if their.labels is None:
                continue
            old = self.sizes
            moved = part.widen(their.labels, their.missing)
            if moved is not None and self.keys is not None:
                codes = list(np.unravel_index(self.keys, old))
                codes[index] = moved[codes[index]]
                self.keys = np.ravel_multi_index(codes, self.sizes)
            if theirs is not None:
                theirs[index] = part.codes(their.labels, their.missing is not None)[theirs[index]]
        if float(np.prod(self.sizes, dtype=np.float64)) >= 2.0**62:
            raise BlockError("too many combinations of the keys' groups to number on blocks; " + WHOLE_AXIS)
        self.dropped += other.dropped
        if other.keys is not None:
            keys = np.ravel_multi_index(theirs, self.sizes) if theirs is not None else other.keys
            self.add(keys, other.counts.copy(),  # copies: a table holds its arrays and adds into them
                     {hidden: (kind, held.copy()) for hidden, (kind, held) in other.pieces.items()}, other.blocks)

    def complete(self) -> None:
        """A single ``bins()`` key lists every bin, as it does in memory: the bins no block had get an empty group."""
        plan = self.plan
        if (self.keys is None or len(self.parts) != 1 or isinstance(plan.key, MultiKey)
                or not isinstance(self.parts[0].key.transform, Bins)):
            return
        full = np.arange(self.parts[0].size, dtype=np.int64)
        if len(full) == len(self.keys):
            return
        where = np.searchsorted(full, self.keys)
        self.counts = _spread(self.counts, where, len(full), 0)
        for hidden, (kind, held) in self.pieces.items():
            node = plan.reductions[hidden]
            self.pieces[hidden] = (kind, _spread(held, where, len(full), REGISTRY[node.op].fill(node, kind, held.dtype)))
        self.keys = full

    def settle(self) -> tuple[dict[str, np.ndarray], dict[str, list[Any]]]:
        """Each reduction's values per combination from the pieces, and the passes each asks for where the pieces do
        not settle every combination (a float32 sum the error bound does not settle), which ``_passes`` runs."""
        assert self.keys is not None
        columns: dict[str, np.ndarray] = {}
        asks: dict[str, list[Any]] = {}
        for hidden, node in self.plan.reductions.items():
            entry = REGISTRY[node.op]
            if entry.counted:
                columns[hidden] = self.counts
                continue
            kind, held = self.pieces[hidden]
            columns[hidden], asked = entry.finish(node, kind, held, self.counts, self.blocks)
            if asked:
                asks[hidden] = asked
        return columns, asks

    def finish(self, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """*columns*, every pass done, each closed by its reduction (a float32 sum's ``min_count``, as ``from_chunks``
        applies it in memory)."""
        for hidden, node in self.plan.reductions.items():
            if not REGISTRY[node.op].counted:
                kind, held = self.pieces[hidden]
                columns[hidden] = REGISTRY[node.op].close(node, kind, held, columns[hidden])
        return columns

    def group_codes(self) -> list[np.ndarray]:
        """Each part's code for every combination in the table."""
        assert self.keys is not None
        return list(np.unravel_index(self.keys, self.sizes)) if self.parts else []


def _spread(held: np.ndarray, where: np.ndarray, size: int, fill: Any) -> np.ndarray:
    """*held*, whose last axis runs along the groups, placed at *where* in a wider table."""
    wider = np.full((*held.shape[:-1], size), fill, dtype=held.dtype)
    wider[..., where] = held
    return wider


# ── The scan ────────────────────────────────────────────────────────────────


def scan(*frames: Any, writes: Sequence[tuple[Any, Any]] = ()) -> tuple[TidyArray, ...]:
    """Run several chunked frames' verbs over their source in one pass, and give each frame's result, in memory.

    The frames come from one ``read_array(..., blocks=)`` or ``tidy(..., blocks=)``
    and share a prefix of verbs: each block is loaded once, the shared verbs
    run once on it, and each frame takes its summaries or its rows from where
    its own verbs branch off, so ``per_ring, voxels = scan(a, b)`` reads the
    data once where ``a.collect()`` and ``b.collect()`` read it twice. Each
    result is the same bits as the frame's own run. Afterwards each frame
    holds its result, as it does after ``collect()``; a frame already
    computed is given as it is.

    With ``blocks="auto"``, the blocks are sized for every frame of the scan
    together, its written ones included, and every pass reads those blocks:
    they may be smaller than a frame's own, so its float64 sums and means,
    integer means, and spreads, which are repeatable for a given size, may
    differ from its own run's in their last bits; every other result is the
    same bits.

    ``writes=[(frame, path), ...]`` writes those frames' rows block by block
    in the same pass (``tidy3a.write``), and adds each written file, as a
    frame read in blocks, after the results.
    """
    for frame in (*frames, *(pair[0] for pair in writes if isinstance(pair, tuple) and len(pair) == 2)):
        if not isinstance(frame, ChunkedFrame):
            raise BlockError(f"scan() runs frames read with blocks=, not {type(frame).__name__}")
    for pair in writes:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise BlockError(f"scan(writes=...) takes (frame, path) pairs, not {pair!r}")
        if pair[0]._summarised:
            raise BlockError("scan(writes=...) writes rows block by block; a summary fits in memory, so write it "
                             "after its result: summary >> write(path)")
    pending = {id(frame): frame for frame in frames if frame._result is None}  # the same frame twice runs once
    writers = [(frame, _writer(frame, path, False)) for frame, path in writes]
    # one source: frames from one read_array() or tidy() call share it, a one-pass full join's tail aside
    if len({id(frame._blocks.source) for frame in (*pending.values(), *(frame for frame, _ in writers))}) > 1:
        raise BlockError("scan() runs one source in one pass; these frames come from different read_array() or "
                         "tidy() calls, so scan each call's frames together")
    if pending or writers:
        _scan(list(pending.values()), writers)
    return (*(frame._result for frame in frames), *(writer.reopen() for _, writer in writers))


def _writer(frame: ChunkedFrame, path: Any, overwrite: bool) -> Any:
    """The writer for a frame's rows, appended block by block along the blocked axis; checked before any block runs.
    The blocks' positions are dropped where joining them in memory drops them (no filter ran)."""
    import os

    from tidy3a.write import Writer

    blocks = frame._blocks
    unlabel = not blocks.labelled and not any(step._operation == "filter" for step in frame._steps)
    return Writer(os.fspath(path), frame, blocks.axis, overwrite, unlabel, blocks, _source_encodings(frame))


def _source_encodings(frame: ChunkedFrame) -> dict:
    """A dataset's portable encodings for the variables the frame's verbs leave as they are (``write``'s
    ``portable_encodings``): a block on the NumPy backend holds none, and the file written should keep them as it does
    from blocks on xarray. None for a file or an array read in blocks, and none where a ``mutate`` writes into an
    array's fields, which changes a variable in part."""
    from tidy3a.write import portable_encodings

    source = frame._blocks.source
    if not isinstance(source, DatasetSource):
        return {}
    targets = [step.target for verb in frame._steps if verb._operation == "mutate" for step in verb._arguments]
    if any(not isinstance(target, str) for target in targets):
        return {}
    data = source.data
    if hasattr(data, "data_vars"):
        variables = dict(data.variables)
    else:
        variables = {data.name if data.name is not None else "values": data.variable,
                     **{name: data[name].variable for name in data.coords}}
    return portable_encodings(variables, set(targets))


def _write(frame: ChunkedFrame, verb: Any) -> TidyArray:
    """``frame >> write(path)``: the frame's rows written as the blocks arrive, then the file read back in blocks."""
    path, overwrite = verb._arguments
    writer = _writer(frame, path, overwrite)
    _scan([], [(frame, writer)])
    return writer.reopen()


@dataclass
class _Leaf:
    """One frame in a scan, and what the blocks add to its result."""

    frame: ChunkedFrame
    table: _Table | None  # a summarised frame's merging table; else the frame joins its rows
    parts: list[Any]  # a joining frame's rows per block, as the backend holds them
    writer: Any = None  # a writing frame's file (``tidy3a.write.Writer``): its blocks go there, not into parts


class _Node:
    """A prefix of row-wise verbs: the frames whose verbs end there, and the verbs that lengthen it."""

    def __init__(self) -> None:
        self.leaves: list[_Leaf] = []
        self.next: list[tuple[Any, _Node]] = []

    def after(self, verb: Any) -> _Node:
        """The node for this prefix followed by *verb*: verbs with the same structure are the same step, so
        pipelines written apart share their work (``tidy3a.identity``)."""
        for other, node in self.next:
            if other is verb or other.structure() == verb.structure():
                return node
        node = _Node()
        self.next.append((verb, node))
        return node


def _trie(leaves: Sequence[_Leaf]) -> _Node:
    """The leaves' row-wise verbs as a trie; ``group_by`` and ``summarise`` are each leaf's own work on its rows."""
    root = _Node()
    for leaf in leaves:
        node = root
        for verb in leaf.frame._steps:
            if verb._operation in ROW_WISE:
                node = node.after(verb)
        node.leaves.append(leaf)
    return root


TOP = 5  # the merge tree's highest level: 32 blocks a table before it joins the running one


class _Tree:
    """A frame's block summaries merged in a fixed tree by block number, the merges run on the pool (plan milestone
    15): as a binary counter, two tables of 2**k blocks each become one of 2**(k+1) once both are there, and the last
    are merged from the right, so the tree's shape follows the number of blocks alone and its result is the same on
    any number of threads. A merge waits only on work handed to the pool before it, which has begun by the time it
    begins, so the pool cannot deadlock however few its threads.

    Capped at ``TOP`` (milestone 17): a table of 2**TOP blocks merges into the running table of every block before
    it instead of waiting for a partner, so the tree holds at most ``TOP + 1`` tables, only the running one of
    nearly every group, where a binary counter over many small blocks held one such table on each upper level."""

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        self.pool = pool
        self.stack: list[tuple[int, Future]] = []  # (level, the table of 2**level consecutive blocks, to come)

    def add(self, table: _Table | Future) -> None:
        """A block's table, or one to come (a running table being made one)."""
        if isinstance(table, Future):
            merged = table
        else:
            merged = Future()
            merged.set_result(table)
        level = 0
        while self.stack and self.stack[-1][0] == level:
            merged = self.pool.submit(_merged, self.stack.pop()[1], merged)
            level += 1
        if level >= TOP and self.stack:  # the running table, of every block before these: it takes them in
            merged = self.pool.submit(_merged, self.stack.pop()[1], merged)
        self.stack.append((min(level, TOP), merged))

    def result(self, empty: _Table) -> _Table:
        """The merged table: *empty* when no block came."""
        if not self.stack:
            return empty
        merged = self.stack.pop()[1]
        while self.stack:
            merged = self.pool.submit(_merged, self.stack.pop()[1], merged)
        return merged.result()


def _merged(left: Future, right: Future) -> _Table:
    """The table of *left*'s blocks followed by *right*'s."""
    table = left.result()
    table.combine(right.result())
    return table


def _fused(frame: ChunkedFrame) -> tuple[Any, str | None]:
    """*frame*'s summary compiled (``tidy3a.fused.driver``), or None for NumPy, as planned or because Numba could not
    compile the kernel, which is the reason given."""
    from tidy3a.fused import unavailable

    if unavailable() is not None:
        return None, None
    from tidy3a.fused.driver import build

    try:
        return build(frame), None
    except Exception as error:  # Numba's own errors: the blocks run on NumPy, and the result says why
        if not type(error).__module__.startswith("numba"):
            raise
        return None, f"Numba could not compile its kernel ({type(error).__name__}: {str(error).splitlines()[0]})"


def _ran(runs: dict[str | None, int]) -> str:
    """How a compiled summary's blocks ran: "ran: 105 blocks in a compiled kernel", and any on NumPy, and why."""
    compiled = runs.get(None, 0)
    words = [f"{compiled:,} block{'' if compiled == 1 else 's'} in a compiled kernel"] if compiled else []
    words += [f"{count:,} {'block' if count == 1 else 'blocks'} on NumPy, as {why}" if not words
              else f"{count:,} on NumPy, as {why}" for why, count in runs.items() if why is not None]
    return "ran: " + "; ".join(words)


def _walk(node: _Node, frame: TidyArray, visit: Callable[[_Leaf, TidyArray], None]) -> None:
    """Depth first: *visit* each leaf at *node* with the frame so far, then each next verb's node with the frame after
    it, so a prefix's frame lives while its branches run, and each verb runs once per block."""
    for leaf in node.leaves:
        visit(leaf, frame)
    for verb, child in node.next:
        _walk(child, frame >> verb, visit)


@contextmanager
def _naming(blocks: Blocks, span: tuple[int, int]) -> Iterator[None]:
    """An error raised while a block runs, of its own class, with the block named after its message: a value the
    verb was checked on (the empty block has none) can fail any block, and the verb's message alone does not say
    where."""
    token = _TAIL.set(blocks.tail and span == (blocks.size, blocks.size))
    try:
        yield
    except (TidyError, LabelArrayError) as error:
        spans = blocks.every
        if blocks.tail and span == (blocks.size, blocks.size):  # a one-pass full join's own block
            where = f"in the other data's labels no block holds (on {blocks.axis})"
        else:  # a block, or the part of one a slice reads
            number = next(number for number, (start, stop) in enumerate(spans) if start <= span[0] < max(stop, 1))
            where = (f"in block {number + 1:,} of {len(spans):,} "
                     f"(positions {span[0] + 1:,}–{span[1]:,} on {blocks.axis})")
        named = copy.copy(error)
        named.args = (f"{error.args[0]}\n  {where}" if error.args else where, *error.args[1:])
        raise named.with_traceback(error.__traceback__) from None
    finally:
        _TAIL.reset(token)


_TAIL: ContextVar[bool] = ContextVar("tidy3a_tail", default=False)  # the block running is a full join's own


def _in_order(pool: ThreadPoolExecutor, work: Callable[[Any], Any], items: Sequence[Any]):
    """*work* on each item, *threads* + 1 at most started ahead of the one whose result is given next, so the results
    waiting to be taken in order stay bounded however unevenly the blocks take."""
    ahead = pool._max_workers + 1
    pending = [pool.submit(work, item) for item in items[:ahead]]
    for item in items[ahead:]:
        done = pending.pop(0).result()
        pending.append(pool.submit(work, item))
        yield done
    for future in pending:
        yield future.result()


def _needs(frame: ChunkedFrame) -> frozenset | None:
    """The variables a frame's blocks must hold: those its verbs read, and, for a frame that joins its rows, those its
    result keeps; None for all of them."""
    if frame._reads is None:
        return None
    if frame._summarised:
        return frame._reads
    return frame._reads | frozenset(frame._backend.variable_names(frame._data))


def _scan(frames: Sequence[ChunkedFrame], writers: Sequence[tuple[ChunkedFrame, Any]] = ()) -> None:
    """One pass over the blocks for *frames*, then the shared second pass; each frame is left holding its result,
    and each of *writers*' frames has its rows written. A block holds only the variables some frame needs (plan
    milestone 12). A failure leaves no written file behind."""
    try:
        _scan_once(frames, writers)
    except BaseException:
        for _, writer in writers:
            writer.abort()
        raise


def _scan_once(frames: Sequence[ChunkedFrame], writers: Sequence[tuple[ChunkedFrame, Any]]) -> None:
    every = [*frames, *(frame for frame, _ in writers)]
    blocks = every[0]._layout() if len(every) == 1 else every[0]._blocks.sized(every)
    if any(frame._blocks.tail for frame in every) and not blocks.tail:  # a one-pass full join's tail, for every frame
        blocks = dataclasses.replace(blocks, tail=True)
    if blocks.chosen is None and every and all(frame._blocks.prunes for frame in every):
        kept = set().union(*(_kept_blocks(frame, blocks) for frame in every))
        blocks = dataclasses.replace(blocks, skipped=frozenset(range(len(blocks.every))) - kept)
    leaves = [_Leaf(frame, _Table(frame._plan) if frame._summarised else None, []) for frame in frames]
    leaves += [_Leaf(frame, None, [], writer) for frame, writer in writers]
    compiled = {id(leaf): _fused(leaf.frame) for leaf in leaves if leaf.table is not None}
    fused = {key: kernels for key, (kernels, _) in compiled.items() if kernels is not None}
    refused = {key: why for key, (_, why) in compiled.items() if why is not None}  # planned, but Numba refused
    root = _trie([leaf for leaf in leaves if id(leaf) not in fused])  # the rest run as their kernels say
    needs = [_needs(frame) for frame in every]
    names = None if any(need is None for need in needs) else frozenset().union(*needs)

    def block(span: tuple[int, int]) -> tuple[list[tuple[_Leaf, Any]], list[tuple[_Leaf, str | None]]]:
        found: list[tuple[_Leaf, Any]] = []
        ran: list[tuple[_Leaf, str | None]] = []  # how each compiled frame's block ran: None, or why on NumPy

        def visit(leaf: _Leaf, frame: TidyArray) -> None:
            if leaf.table is None:
                found.append((leaf, frame._data))
            else:  # the block's summary, a table of its own, made here in the worker
                table = _Table(leaf.table.plan)
                table.merge(*_partial(leaf.table.plan, frame, blocks.axis))
                found.append((leaf, table))

        with _naming(blocks, span):
            loaded = blocks.load(*span, names)
            for leaf in leaves:
                kernels = fused.get(id(leaf))
                if kernels is None:
                    continue
                table, why = kernels.run(loaded)
                if isinstance(table, list):  # into its thread's running table: what that gave up, if anything
                    found.extend((leaf, given) for given in table)
                    ran.append((leaf, why))
                    continue
                if table is None:  # this block on NumPy: the frame's row-wise verbs, then its summary
                    rows = loaded
                    for verb in leaf.frame._steps:
                        if verb._operation in ROW_WISE:
                            rows = rows >> verb
                    table = _Table(leaf.table.plan)
                    table.merge(*_partial(leaf.table.plan, rows, blocks.axis))
                found.append((leaf, table))
                ran.append((leaf, why))
            _walk(root, loaded, visit)
        return found, ran

    with ThreadPoolExecutor(blocks.threads) as pool:
        trees = {id(leaf): _Tree(pool) for leaf in leaves if leaf.table is not None}
        runs: dict[int, dict[str | None, int]] = {key: {} for key in fused}
        runs.update({key: {why: len(blocks.spans)} for key, why in refused.items()})
        for found, ran in _in_order(pool, block, blocks.spans):  # rows taken as the blocks arrive, in order
            for leaf, why in ran:
                runs[id(leaf)][why] = runs[id(leaf)].get(why, 0) + 1
            for leaf, part in found:
                if leaf.writer is not None:
                    leaf.writer.add(part)
                elif leaf.table is None:
                    leaf.parts.append(part)
                else:
                    trees[id(leaf)].add(part)
        for key, kernels in fused.items():  # the threads' running tables, once every block is in, made tables
            for table in kernels.drain(pool):  # side by side
                trees[key].add(table)
        for leaf in leaves:
            if leaf.table is not None:
                leaf.table = trees[id(leaf)].result(leaf.table)
        settled: dict[int, tuple[dict[str, np.ndarray], dict[str, list[Any]]]] = {}
        for leaf in leaves:
            if leaf.table is not None and leaf.table.keys is not None:
                leaf.table.complete()
                settled[id(leaf)] = leaf.table.settle()
        _passes(leaves, blocks, pool, settled, names, fused)
    for leaf in leaves:
        frame, table = leaf.frame, leaf.table
        if leaf.writer is not None:
            leaf.writer.finish()
        elif table is None:
            frame._result = _joined(frame, leaf.parts)
            if frame._sorting:  # each block sorted: merged now
                frame._result = _merged_sort(frame, frame._result)
        elif table.keys is None:  # no block kept a row, the first included, so its summary is the answer
            frame._result = frame._evolve()
        else:
            frame._result = _result(frame, table.plan, table, table.finish(settled[id(leaf)][0]))
        if blocks.skipped and frame._result is not None:  # what the filters on the axis's labels spared reading
            filters = ", ".join(verb.text for verb in frame._blocks.prunes)
            frame._result = frame._result._evolve(notes=[
                f"skipped {len(blocks.skipped):,} of {len(blocks.every):,} blocks, which {filters} keeps none of"])
        if id(leaf) in runs and blocks.chosen is None:  # how the blocks ran, beside the result (milestone 16)
            frame._result = _recorded(frame._result._evolve(notes=[_ran(runs[id(leaf)])]))


def _kept_blocks(frame: ChunkedFrame, blocks: Blocks) -> set[int]:
    """The blocks *frame*'s filters on the axis's labels keep a position of, tried on each block's labels alone,
    nothing read (milestone 19, as Spark prunes partitions by their statistics)."""
    backend, axis, source = frame._backend, blocks.axis, blocks.source
    kept = set()
    for number, (start, stop) in enumerate(blocks.every):
        labels = source.block_labels(start, stop)
        probe = TidyArray(backend.summary({"_": backend.array(np.zeros(len(labels), dtype=bool), dims=(axis,),
                                                               coords={axis: labels}, name="_")}, {}, {}),
                          backend=backend)
        for verb in frame._blocks.prunes:
            probe = verb._apply(probe)
        if backend.sizes(probe._data).get(axis, 0):
            kept.add(number)
    return kept


def _passes(leaves: Sequence[_Leaf], blocks: Blocks, pool: ThreadPoolExecutor,
            settled: Mapping[int, tuple[dict[str, np.ndarray], dict[str, list[Any]]]],
            names: frozenset | None = None, fused: Mapping[int, Any] | None = None) -> None:
    """The passes reductions ask for after the first (``tidy3a.reductions.Ask``), shared by every frame of the scan:
    the asks are packed, whole and in order, into rounds whose answers hold at most ``pass_budget()`` values per
    block; each round reads the blocks once, every block answers every ask in it (``again``), the answers join across
    the blocks in any order (``join``), and each reduction resolves its ask into its column, perhaps asking again
    (``resolve``), until no ask is left. A compiled frame's rows are found by its kernels (``Fused.rows``), as in
    the first pass, not by NumPy's verbs again (plan milestone 20)."""
    fused = fused or {}
    pending = [(leaf, hidden, ask) for leaf in leaves if id(leaf) in settled
               for hidden, asks in settled[id(leaf)][1].items() for ask in asks]
    budget = pass_budget()
    while pending:
        round_, held = [], 0
        while pending and (not round_ or held + pending[0][2].size <= budget):
            round_.append(pending.pop(0))
            held += round_[-1][2].size
        wanted: dict[int, list[tuple[int, str, Any]]] = {}
        for index, (leaf, hidden, ask) in enumerate(round_):
            wanted.setdefault(id(leaf), []).append((index, hidden, ask))
        asked = list({id(leaf): leaf for leaf, _, _ in round_}.values())
        root = _trie([leaf for leaf in asked if id(leaf) not in fused])

        def block(span: tuple[int, int]) -> list[tuple[int, Any]]:
            answers: list[tuple[int, Any]] = []

            def counted(leaf: _Leaf, rows: _Rows, asks: Sequence[tuple[int, str, Any]] | None = None) -> None:
                assert leaf.table is not None and leaf.table.keys is not None
                if rows.kept == 0:
                    return
                codes = _table_codes(leaf.table, rows)
                for index, hidden, ask in wanted[id(leaf)] if asks is None else asks:
                    value, each = rows.values[hidden]
                    node = leaf.table.plan.reductions[hidden]
                    answers.append((index, _again(node, ask, value, each, codes, len(leaf.table.keys))))

            def visit(leaf: _Leaf, frame: TidyArray) -> None:
                assert leaf.table is not None
                counted(leaf, _block(leaf.table.plan, frame, blocks.axis))

            with _naming(blocks, span):
                loaded = blocks.load(*span, names)
                for leaf in asked:
                    kernels = fused.get(id(leaf))
                    if kernels is None:
                        continue
                    rest = []  # the asks its one-loop pass does not count (``Fused.count``)
                    for index, hidden, ask in wanted[id(leaf)]:
                        found = kernels.count(loaded, leaf.table, hidden, ask)
                        if found is None:
                            rest.append((index, hidden, ask))
                        else:
                            answers.append((index, found))
                    if not rest:
                        continue
                    rows = kernels.rows(loaded)
                    if rows is None:  # this block on NumPy, as its first pass was: the row-wise verbs, then the keys
                        frame = loaded
                        for verb in leaf.frame._steps:
                            if verb._operation in ROW_WISE:
                                frame = frame >> verb
                        rows = _block(leaf.table.plan, frame, blocks.axis)  # type: ignore[union-attr]
                    counted(leaf, rows, rest)
                _walk(root, loaded, visit)
            return answers

        totals: dict[int, Any] = {}
        for answers in _in_order(pool, block, blocks.spans):
            for index, answer in answers:
                leaf, hidden, ask = round_[index]
                node = leaf.table.plan.reductions[hidden]  # type: ignore[union-attr]
                totals[index] = answer if index not in totals else REGISTRY[node.op].join(node, ask, totals[index],
                                                                                          answer)
        for index, (leaf, hidden, ask) in enumerate(round_):  # resolved now, so a round's answers go with it
            columns = settled[id(leaf)][0]
            node = leaf.table.plan.reductions[hidden]  # type: ignore[union-attr]
            columns[hidden], more = REGISTRY[node.op].resolve(node, ask, totals.get(index), columns[hidden])
            pending.extend((leaf, hidden, again) for again in more)
        del totals


SLICE = 1 << 16  # values a later pass works on at once, whatever the block's size (plan milestone 14)


def _again(node: Reduce, ask: Any, value: np.ndarray, each: int, codes: np.ndarray, groups: int) -> Any:
    """One block's answer to *ask*: ``again`` on slices of at most ``SLICE`` values, rows whole, their answers joined
    as blocks' are, so a later pass works in the same memory whatever the block's size."""
    entry = REGISTRY[node.op]
    whole = getattr(entry, "count_whole", None)  # a compiled pass over the whole block, where the reduction has one
    found = whole(node, ask, value, each, codes) if whole is not None else None
    if found is not None:
        return found
    rows = max(1, SLICE // each)
    answer = None
    for start in range(0, max(len(codes), 1), rows):
        part = codes[start:start + rows]
        found = entry.again(node, ask, value[start * each:(start + rows) * each],
                            part if each == 1 else np.repeat(part, each), groups)
        answer = found if answer is None else entry.join(node, ask, answer, found)
    return answer


def _table_codes(table: _Table, rows: _Rows) -> np.ndarray:
    """A block's kept rows' places in the finished table: each key's local codes mapped, combined, and found."""
    assert table.keys is not None
    if not table.parts:
        return np.zeros(rows.kept, dtype=np.intp)
    codes = np.ravel_multi_index([part.codes(labels, missing is not None)[local]
                                  for part, (labels, missing, _), local in zip(table.parts, rows.found, rows.local)],
                                 table.sizes)
    return np.searchsorted(table.keys, codes)


# ── One block's part of a summary ───────────────────────────────────────────


@dataclass
class _Rows:
    """One block through the row-wise verbs: its rows' codes per key, and what each reduction sees of them."""

    found: list[tuple[np.ndarray, Any, int]]  # per key: labels without the missing group, its label, rows dropped
    local: list[np.ndarray]  # per key: each kept row's local code (the missing group last)
    kept: int  # rows kept
    dropped: int  # rows left out for a missing key
    per_position: int  # positions on the other observation axes, which n() counts per row
    values: dict[str, tuple[np.ndarray, int]]  # per reduction: its values, flattened with the row first, and per row


def _block(plan: _Plan, block: TidyArray, axis: str) -> _Rows:
    """One block after the row-wise verbs, grouped: its keys' groups and codes, and the reductions' values."""
    backend = block._backend
    size = backend.sizes(block._data)[axis]
    found: list[tuple[np.ndarray, Any, int]] = []
    local: list[np.ndarray] = []
    kept = np.ones(size, dtype=bool)
    for part in plan.parts:
        groups = resolve_value_key(part, block, [])
        if groups.axis != axis:
            raise BlockError(f"group_by({part.label}) groups {groups.axis}, not the blocked axis {axis}; "
                             f"{WHOLE_AXIS}")
        labels, codes = list(groups.labels), np.asarray(groups.codes)
        missing = None
        if groups.dropped:
            labels, missing = labels[:-1], None  # left out of every group
            kept &= codes < len(labels)
        elif labels and _is_missing_label(labels[-1], part):
            labels, missing = labels[:-1], labels[-1]
        found.append((_label_array(labels), missing, groups.dropped))
        local.append(codes)
    all_kept = bool(kept.all())
    ctx = Context(block, verb="summarise", error=SummariseError, grouped=False)
    values: dict[str, tuple[np.ndarray, int]] = {}
    for hidden, node in plan.reductions.items():
        if node.arg is None:
            continue
        value = evaluate(node.arg, ctx)
        if not backend.is_array(value) or axis not in value.dims:
            dims = ", ".join(map(str, getattr(value, "dims", ()))) or "no axis"
            raise BlockError(f"{node.text()}: {node.arg.text()} varies along {dims}, not the blocked axis {axis}; "
                             f"{WHOLE_AXIS}")
        if block._field_axis in value.dims:
            raise BlockError(f"{node.text()}: {node.arg.text()} varies along the field axis {block._field_axis}; "
                             "name one field")
        array = np.moveaxis(value.values, list(value.dims).index(axis), 0)
        if not all_kept:
            array = array[kept]
        each = int(np.prod(array.shape[1:], dtype=np.int64))  # values per row: the other axes' positions
        values[hidden] = (array.reshape(len(array) * each), each)
    return _Rows(found, [codes if all_kept else codes[kept] for codes in local], size if all_kept else int(kept.sum()),
                 size - (size if all_kept else int(kept.sum())), plan.per_position, values)


def _per_position(block: TidyArray, axis: str) -> int:
    """What ``n()`` counts per row: positions on the other observation axes shared by every non-scalar variable."""
    backend, data = block._backend, block._data
    sizes = backend.sizes(data)
    shared = set(sizes)
    for variable in backend.variables(data).values():
        if variable.ndim:
            shared &= set(variable.dims)
    others = [dim for dim in sizes if dim in shared and dim not in (axis, block._field_axis)]
    return int(np.prod([sizes[dim] for dim in others], dtype=np.int64))


def _combined(local: Sequence[tuple[np.ndarray, Any, np.ndarray]], table: _Table) -> np.ndarray:
    """A block's combinations in the table's codes: each part's local codes mapped, then the parts' codes as one number."""
    if not table.parts:
        return np.zeros(1, dtype=np.int64)
    return np.ravel_multi_index([part.codes(labels, missing is not None)[codes]
                                 for part, (labels, missing, codes) in zip(table.parts, local)], table.sizes)


def _partial(plan: _Plan, block: TidyArray, axis: str):
    """One block's groups; each combination's local codes per key, with the key's labels, to map once the table has
    taken the groups in; the combinations' row counts; each reduction's pieces per combination; and the rows dropped."""
    return _partial_rows(plan, _block(plan, block, axis))


def _partial_rows(plan: _Plan, rows: _Rows):
    """``_partial`` of a block's rows, however they were found (here, or by a compiled block, ``fused.driver``)."""
    if rows.kept == 0:
        return rows.found, None, None, None, rows.dropped
    # Codes per local combination, so the pieces are made once per group, not once per row.
    if plan.parts:
        sizes = [len(labels) + (missing is not None) for labels, missing, _ in rows.found]
        keys, codes = np.unique(np.ravel_multi_index(rows.local, sizes), return_inverse=True)
        codes = codes.reshape(-1)
        combos = [(labels, missing, where)
                  for (labels, missing, _), where in zip(rows.found, np.unravel_index(keys, sizes))]
    else:
        keys, codes, combos = np.zeros(1, dtype=np.int64), np.zeros(rows.kept, dtype=np.intp), []
    count = len(keys)
    counts = np.bincount(codes, minlength=count).astype(np.int64) * rows.per_position
    pieces: dict[str, tuple[str, np.ndarray]] = {}
    for hidden, node in plan.reductions.items():
        if node.op == "n":
            continue
        value, each = rows.values[hidden]
        pieces[hidden] = _pieces(node, value, codes, count, each)
    return rows.found, combos, counts, pieces, rows.dropped


def _pieces(node: Reduce, value: np.ndarray, codes: np.ndarray, count: int, each: int) -> tuple[str, np.ndarray]:
    """One reduction's pieces per group from a block's values, *each* to a position, and the positions' codes
    (``tidy3a.reductions``)."""
    return REGISTRY[node.op].partial_rows(node, value, codes, count, each)


def _result(frame: ChunkedFrame, plan: _Plan, table: _Table, columns: dict[str, np.ndarray]) -> TidyArray:
    """The merged summaries as the frame ``summarise`` gave on the empty block, with every block's values."""
    backend, key = frame._backend, plan.key
    codes = table.group_codes()
    results: dict[Hashable, Any] = {}
    if key is None:
        axis, labels = "group", None
    elif isinstance(key, MultiKey):  # one group per combination, each key's value per group as a variable
        axis, labels = key.name, np.arange(1, len(table.keys) + 1)
        for part, part_codes in zip(table.parts, codes):
            results[part.key.name] = backend.array(part.full()[part_codes], dims=(axis,), coords={axis: labels},
                                                   name=part.key.name)
    else:  # text labels as NumPy text, as each backend stores a key's labels in memory
        found = table.parts[0].full()[codes[0]]
        axis, labels = key.name, (np.asarray(found.tolist()) if found.dtype.kind == "O" else found)
    coords = {axis: labels} if labels is not None else None
    hidden = {name: backend.array(values, dims=(axis,), coords=coords, name=name) for name, values in columns.items()}
    held = TidyArray(backend.summary(hidden, {}, {}), backend=backend)
    ctx = Context(held, verb="summarise", error=SummariseError, grouped=False)
    given = {name: variable.dtype for name, variable in backend.variables(frame._data).items()}
    for name, node in plan.summaries.items():
        value = evaluate(node, ctx)
        values = value.values if backend.is_array(value) else np.asarray(value)
        if name in given and values.dtype != given[name] and not _data_decides(values.dtype, given[name]):
            values = values.astype(given[name])  # the type the summary has in memory
        if key is None:
            values = np.asarray(values).reshape(-1)[0] if np.ndim(values) else values
            results[name] = backend.array(values, name=name)
        else:
            results[name] = backend.array(values, dims=(axis,), coords=coords, name=name)
    data = backend.summary(results, backend.attrs(frame._data), {})
    result = frame._evolve(data)
    result = TidyArray(result._data, backend=backend, field_axis=result._field_axis, origins=result._origins,
                       source=result._source, notes=_notes(frame, plan, table), groups=result._groups,
                       record=result._record)
    return _recorded(result)


def _data_decides(merged: np.dtype, empty: np.dtype) -> bool:
    """Whether a summary's type is one the data decides, so the empty block's type is no guide: an integer min or max
    is float64 when a group is empty (on the empty block, always) and the integers' own type when none is."""
    return {merged.kind, empty.kind} <= set("biuf") and "f" in (merged.kind, empty.kind) and merged.kind != empty.kind


def _recorded(result: TidyArray) -> TidyArray:
    """*result* with its last recorded step's effect taken from it, not from the empty block."""
    record = result._record
    if record is not None and record.steps:
        steps = record.steps
        result._record = Record(record.source, record.start, (*steps[:-1], Step(steps[-1].verb, snapshot(result))),
                                record.preview)
    return result


def _notes(frame: ChunkedFrame, plan: _Plan, table: _Table) -> list[str]:
    """The notes the verbs wrote on the empty block, but those its emptiness wrote, with the count of positions
    dropped for a missing key from every block."""
    key = plan.key
    prefix = f"changed: group_by({key.label}) dropped " if key is not None else None
    notes = [note for note in frame._notes if (prefix is None or not note.startswith(prefix)) and not _emptied(note)]
    dropped = table.dropped
    if key is not None and dropped:
        lacking = [part.subject for part in table.parts if part.dropped]
        notes.append(f"{prefix}{dropped:,} position{'' if dropped == 1 else 's'} with a missing "
                     f"{' or '.join(lacking)}; dropna=False keeps them as "
                     f"{'groups' if isinstance(key, MultiKey) else 'a group'}")
    return notes


# ── Rows kept, concatenated ─────────────────────────────────────────────────


def _joined(frame: ChunkedFrame, parts: Sequence[Any]) -> TidyArray:
    """The blocks' rows after the row-wise verbs, joined along the blocked axis, in memory."""
    blocks, backend, axis = frame._blocks, frame._backend, frame._blocks.axis
    data = backend.join_along(parts, axis)
    if blocks.tail and not blocks.positions:  # a one-pass full join's tail came last: the whole join's order
        order = np.argsort(np.asarray(backend.axis_labels(data, axis).values), kind="stable")
        data = backend.isel(data, {axis: order})
    if not blocks.labelled and not blocks.positions and not any(step._operation == "filter" for step in frame._steps):
        data = backend.unlabel(data, axis)  # the blocks' positions served the verbs; in memory the axis has no labels
    size = backend.sizes(data).get(axis, 0)
    notes = [note for note in frame._notes if size == 0 or "(all removed by" not in note]
    result = TidyArray(data, backend=backend, field_axis=frame._field_axis, origins=frame._origins,
                       source=frame._source, notes=notes, groups=frame._groups, record=frame._record)
    return _recorded(result)
