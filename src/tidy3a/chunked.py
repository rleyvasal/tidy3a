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
import math
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from labelarray import sums
from labelarray.errors import LabelArrayError
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

ROW_WISE = ("select", "filter", "mutate")
COMBINABLE = ("n", "sum", "mean", "min", "max", "std", "var", "any", "all")
WHOLE_AXIS = "read the data with lazy=True, which memory-maps a file, or load it, to run this on every position"


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

    def empty(self) -> TidyArray:
        raise NotImplementedError


class FileSource(Source):
    """A file memory-mapped again for each block; the block is a copy, and the map closes with it."""

    def __init__(self, reopen: Callable[[], np.ndarray], size: int, axis: str,
                 build: Callable[[np.ndarray, int, int], TidyArray], row: tuple[np.dtype, tuple[int, ...]]):
        self.reopen, self.size, self.axis, self.build = reopen, size, axis, build
        self.row = row  # the values' type, in native byte order, and the shape of one position

    def empty(self) -> TidyArray:
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

    def empty(self) -> TidyArray:
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
        from tidy3a.backend import backend_for

        # The backend is chosen once, for the whole data, so a block holding some of its variables is on the same one.
        backend = backend_for(data, backend).name if backend is None else backend
        self.data, self.axis, self.field_axis, self.backend = data, axis, field_axis, backend
        self.size = int(data.sizes[axis])
        self.chunks = _chunks_along(data, axis)

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

    def empty(self) -> TidyArray:
        """No positions on the axis: each lazy value along it an empty NumPy array of its type, since slicing lazy
        data to no positions and computing it still computes a chunk; nothing along the axis is computed."""
        from tidy3a.frame import tidy

        sliced = self.data.isel({self.axis: slice(0, 0)})
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
        return tidy(sliced.assign_coords(coords), field_axis=self.field_axis, backend=self.backend)


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
    rows: int | str  # positions per block, or "chunks" for the source's own boundaries
    threads: int
    labelled: bool = False  # else each block is labelled with its positions, so a verb sees them as in memory
    chosen: tuple[int, ...] | None = None  # a sample's blocks (``preview``), counted from 0; else every block

    @property
    def axis(self) -> str:
        return self.source.axis

    @property
    def size(self) -> int:
        return self.source.size

    @property
    def every(self) -> list[tuple[int, int]]:
        """Every block's span, which a block's number counts in."""
        if self.rows == "chunks":
            return self.source.chunks or [(0, self.size)]
        rows = int(self.rows)
        return [(start, min(start + rows, self.size)) for start in range(0, self.size, rows)] or [(0, 0)]

    @property
    def spans(self) -> list[tuple[int, int]]:
        """The spans a run reads: every block's, or a sample's."""
        every = self.every
        return every if self.chosen is None else [every[number] for number in self.chosen]

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
        return f"{count} of {int(self.rows):,} positions on {self.axis}"


def check_rows(rows: Any, threads: Any, source: Source) -> None:
    """``blocks=`` is a positive number of positions, or "chunks" for a source that has them."""
    if rows == "chunks":
        if source.chunks is None:
            raise ReadError('blocks="chunks" takes a Dask-backed dataset\'s chunks along the axis; '
                            f"this data is not chunked along {source.axis}, so give a number of positions")
    elif isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or rows < 1:
        raise ReadError(f'blocks= must be a positive number of positions, or "chunks", not {rows!r}')
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ReadError(f"threads= must be a positive number, not {threads!r}")


class ChunkedFrame(TidyArray):
    """Data read with ``blocks=``: an empty block with its schema, and the verbs to run over every block."""

    __slots__ = ("_blocks", "_steps", "_plan", "_result", "_reads")

    @classmethod
    def open(cls, source: Source, rows: int | str, threads: int) -> ChunkedFrame:
        """A frame holding the source's empty block: nothing is read until a result is asked for (plan decision 4)."""
        check_rows(rows, threads, source)
        empty = source.empty()
        blocks = Blocks(source, rows, threads, empty._backend.has_labels(empty._data, source.axis))
        frame = cls._wrap(empty, blocks, (), reads=frozenset())
        count = len(blocks.every)
        frame._record = Record(empty._source or "data", _unknown(snapshot(empty)),
                               preview=f"nothing read yet: {blocks.words(' block' if count == 1 else ' blocks')}")
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
        return new

    def _apply_verb(self, verb: Any) -> Any:
        operation = getattr(verb, "_operation", None)
        if operation == "write" and not self._summarised:  # rows written as the blocks arrive, never all in memory
            return _write(self, verb)
        if self._summarised or operation not in (*ROW_WISE, "group_by", "summarise"):
            return verb(self._run())  # needs the whole axis: the blocks' result, in memory
        if operation == "select":
            _refuse_value_selectors(verb)  # before the verb runs on the empty block, which would answer for no values
        with warnings.catch_warnings(), recording_reads() as read:  # a mean of nothing warns; here that says nothing
            warnings.simplefilter("ignore", RuntimeWarning)
            result = TidyArray._apply_verb(self, verb)  # on the empty block: the eager checks, and the result's schema
        steps = (*self._steps, verb)
        plan = _check(self, verb, result, steps)
        reads = None if self._reads is None or EVERYTHING in read else self._reads | frozenset(read)
        frame = self._wrap(result, self._blocks, steps, plan, reads)
        record = frame._record
        last = record.steps[-1]
        frame._record = dataclasses.replace(record, steps=(*record.steps[:-1], Step(last.verb, _unknown(last.after))))
        return frame

    @translating
    def preview(self, blocks: Sequence[int] | None = None) -> TidyArray:
        """The verbs run on a sample of blocks, the result in memory, labelled as a sample; the frame is unchanged.

        By default the first, middle, and last blocks, so the sample is not only
        the start of the data; ``blocks=`` names others, counted from 1 as an
        error names them. It reads those blocks, and only those.
        """
        every = len(self._blocks.every)
        if blocks is None:
            chosen = sorted({0, every // 2, every - 1})
        else:
            numbers = [blocks] if isinstance(blocks, (int, np.integer)) else list(blocks)
            wrong = [number for number in numbers
                     if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or not 1 <= number <= every]
            if wrong or not numbers:
                raise BlockError(f"preview(blocks=...) takes block numbers from 1 to {every:,}, not {wrong or blocks!r}")
            chosen = sorted({int(number) - 1 for number in numbers})
        sample = dataclasses.replace(self._blocks, chosen=tuple(chosen))
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
        """Each step's effect on the schema (``?`` for sizes the data decides), or on every block once a result has
        been computed."""
        return self._result.explain() if self._result is not None else TidyArray.explain(self)

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
        lines.append(f"blocks: {self._blocks.words()}; ? is a size the data decides; preview() runs a sample, and "
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
        if node.op not in COMBINABLE:
            raise BlockError(f"{name} = {node.text()}: {node.op}() cannot be combined across blocks; "
                             f"{', '.join(COMBINABLE)} can; {WHOLE_AXIS}")
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

    def add(self, keys: np.ndarray, counts: np.ndarray, pieces: dict[str, tuple[str, np.ndarray]]) -> None:
        """A block's combinations, their counts, and each reduction's pieces, which add."""
        if len(keys) == 0:
            return
        self.blocks += 1
        if self.keys is None:
            self.keys, self.counts, self.pieces = keys, counts, dict(pieces)
            return
        union = np.union1d(self.keys, keys)
        if len(union) != len(self.keys):
            where = np.searchsorted(union, self.keys)
            self.counts = _spread(self.counts, where, len(union), 0)
            for hidden, (kind, held) in self.pieces.items():
                fill = _fill(kind, held.dtype, self.plan.reductions[hidden])
                self.pieces[hidden] = (kind, _spread(held, where, len(union), fill))
            self.keys = union
        at = np.searchsorted(union, keys)
        self.counts[at] += counts
        for hidden, (kind, piece) in pieces.items():
            held = self.pieces[hidden][1]
            if kind == "estimate":
                held[:sums.TERMS, at] += piece[:sums.TERMS]
                held[sums.TERMS, at] = np.maximum(held[sums.TERMS, at], piece[sums.TERMS])
            elif kind in ("extreme", "flag"):
                held[at] = _merged(self.plan.reductions[hidden], held[at], piece)
            elif kind == "moments":
                held[:, at] = _combine_moments(held[:, at], piece)
            else:
                held[..., at] += piece

    def merge(self, found: Sequence[tuple[np.ndarray, Any, int]], local: Any, counts: Any, pieces: Any,
              dropped: int) -> None:
        """One block's partial (``_partial``) taken into the table."""
        self.take(found)
        self.dropped += dropped
        if local is not None:
            self.add(_combined(local, self), counts, pieces)

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
            fill = _fill(kind, held.dtype, plan.reductions[hidden])
            self.pieces[hidden] = (kind, _spread(held, where, len(full), fill))
        self.keys = full

    def settle(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Each reduction's values per combination from the pieces, and, per float32 sum, the combinations the
        error bound cannot settle (``labelarray.sums.settle``), whose exact sums a second pass finds."""
        assert self.keys is not None
        columns: dict[str, np.ndarray] = {}
        unsure: dict[str, np.ndarray] = {}
        for hidden, node in self.plan.reductions.items():
            if node.op == "n":
                columns[hidden] = self.counts
                continue
            kind, held = self.pieces[hidden]
            if kind == "estimate":
                result, undecided = sums.settle(np.asarray(held, dtype=np.float64), self.blocks, node.op, node.na_rm)
                columns[hidden] = result
                if undecided.any():
                    unsure[hidden] = np.flatnonzero(undecided)
            elif kind == "extreme":
                columns[hidden] = _settled_extreme(held, self.counts)
            elif kind == "flag":
                columns[hidden] = held
            elif kind == "moments":
                present, _, squares, missing = held
                with np.errstate(invalid="ignore", divide="ignore"):
                    variance = np.where(present > 0, squares / present, np.nan)  # the population variance, as np.var
                if not node.na_rm:
                    variance = np.where(missing > 0, np.nan, variance)
                columns[hidden] = np.sqrt(variance) if node.op == "std" else variance
            elif kind == "float":
                total, present, counted = held
                with np.errstate(invalid="ignore", divide="ignore"):
                    result = total if node.op == "sum" else total / (present if node.na_rm else counted)
                columns[hidden] = _at_least(result, present, node)
            else:
                total, counted = held
                columns[hidden] = _at_least(total, counted, node)
        return columns, unsure

    def finish(self, columns: dict[str, np.ndarray],
               exact: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> dict[str, np.ndarray]:
        """*columns* with the exact sums the second pass found in their places, and a float32 sum's ``min_count``
        applied, as ``from_chunks`` does in memory."""
        for hidden, node in self.plan.reductions.items():
            for chosen, found in exact.get(hidden, ()):
                columns[hidden][chosen] = found
            if node.op == "sum" and node.min_count is not None and self.pieces[hidden][0] == "estimate":
                present = self.pieces[hidden][1][sums.PRESENT]
                columns[hidden] = np.where(present >= node.min_count, columns[hidden], np.float32(np.nan))
        return columns

    def group_codes(self) -> list[np.ndarray]:
        """Each part's code for every combination in the table."""
        assert self.keys is not None
        return list(np.unravel_index(self.keys, self.sizes)) if self.parts else []


def _at_least(result: np.ndarray, present: np.ndarray, node: Reduce) -> np.ndarray:
    """A sum as NaN where fewer than ``min_count`` values are present."""
    if node.op != "sum" or node.min_count is None:
        return result
    return np.where(present >= node.min_count, result, np.nan)


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
    if len({id(frame._blocks) for frame in (*pending.values(), *(frame for frame, _ in writers))}) > 1:
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
    return Writer(os.fspath(path), frame, blocks.axis, overwrite, unlabel, blocks)


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
    exact: dict[str, list[tuple[np.ndarray, np.ndarray]]]  # per float32 sum: the combinations a second pass settled
    # (their places in the table, their correctly rounded results)
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
    try:
        yield
    except (TidyError, LabelArrayError) as error:
        spans = blocks.every
        where = (f"in block {spans.index(span) + 1:,} of {len(spans):,} "
                 f"(positions {span[0] + 1:,}–{span[1]:,} on {blocks.axis})")
        named = copy.copy(error)
        named.args = (f"{error.args[0]}\n  {where}" if error.args else where, *error.args[1:])
        raise named.with_traceback(error.__traceback__) from None


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
    blocks = every[0]._blocks
    leaves = [_Leaf(frame, _Table(frame._plan) if frame._summarised else None, [], {}) for frame in frames]
    leaves += [_Leaf(frame, None, [], {}, writer) for frame, writer in writers]
    root = _trie(leaves)
    needs = [_needs(frame) for frame in every]
    names = None if any(need is None for need in needs) else frozenset().union(*needs)

    def block(span: tuple[int, int]) -> list[tuple[_Leaf, Any]]:
        found: list[tuple[_Leaf, Any]] = []

        def visit(leaf: _Leaf, frame: TidyArray) -> None:
            if leaf.table is None:
                found.append((leaf, frame._data))
            else:
                found.append((leaf, _partial(leaf.table.plan, frame, blocks.axis)))

        with _naming(blocks, span):
            _walk(root, blocks.load(*span, names), visit)
        return found

    with ThreadPoolExecutor(blocks.threads) as pool:
        for found in _in_order(pool, block, blocks.spans):  # merged as the blocks arrive, in order
            for leaf, part in found:
                if leaf.writer is not None:
                    leaf.writer.add(part)
                elif leaf.table is None:
                    leaf.parts.append(part)
                else:
                    leaf.table.merge(*part)
        settled: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        for leaf in leaves:
            if leaf.table is not None and leaf.table.keys is not None:
                leaf.table.complete()
                settled[id(leaf)] = leaf.table.settle()
        _exact_pass(leaves, blocks, pool, {key: unsure for key, (_, unsure) in settled.items()}, names)
    for leaf in leaves:
        frame, table = leaf.frame, leaf.table
        if leaf.writer is not None:
            leaf.writer.finish()
        elif table is None:
            frame._result = _joined(frame, leaf.parts)
        elif table.keys is None:  # no block kept a row, the first included, so its summary is the answer
            frame._result = frame._evolve()
        else:
            frame._result = _result(frame, table.plan, table, table.finish(settled[id(leaf)][0], leaf.exact))


def _exact_pass(leaves: Sequence[_Leaf], blocks: Blocks, pool: ThreadPoolExecutor,
                unsure: Mapping[int, dict[str, np.ndarray]], names: frozenset | None = None) -> None:
    """The second pass, shared by the leaves: each block's exact sums for every combination the first pass could not
    settle, over every leaf's float32 sums at once (``labelarray.sums.chunk_exact``), added across the blocks. At most
    ``EXACT_PAIRS`` combinations' buckets are in flight per block, so a larger set takes several passes."""
    asked = [(leaf, hidden, rows[start:start + sums.EXACT_PAIRS])
             for leaf in leaves for hidden, rows in unsure.get(id(leaf), {}).items()
             for start in range(0, len(rows), sums.EXACT_PAIRS)]
    passes: list[list[tuple[_Leaf, str, np.ndarray]]] = []
    held = sums.EXACT_PAIRS
    for request in asked:  # whole requests, in order, until a pass is full
        if held + len(request[2]) > sums.EXACT_PAIRS:
            passes.append([])
            held = 0
        passes[-1].append(request)
        held += len(request[2])
    for requests in passes:
        wanted: dict[int, list[tuple[int, str, np.ndarray]]] = {}
        for index, (leaf, hidden, chosen) in enumerate(requests):
            wanted.setdefault(id(leaf), []).append((index, hidden, chosen))
        root = _trie(list({id(leaf): leaf for leaf, _, _ in requests}.values()))

        def block(span: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
            buckets: list[tuple[int, np.ndarray]] = []

            def visit(leaf: _Leaf, frame: TidyArray) -> None:
                assert leaf.table is not None and leaf.table.keys is not None
                rows = _block(leaf.table.plan, frame, blocks.axis)
                if rows.kept == 0:
                    return
                codes = _table_codes(leaf.table, rows)
                for index, hidden, chosen in wanted[id(leaf)]:
                    value, each = rows.values[hidden]
                    buckets.append((index, sums.chunk_exact(value, (0,), chosen,
                                                            codes=codes if each == 1 else np.repeat(codes, each),
                                                            groups=len(leaf.table.keys))))

            with _naming(blocks, span):
                _walk(root, blocks.load(*span, names), visit)
            return buckets

        totals: dict[int, np.ndarray] = {}
        for buckets in _in_order(pool, block, blocks.spans):  # the buckets add in any order
            for index, part in buckets:
                totals[index] = part if index not in totals else totals[index] + part
        for index, (leaf, hidden, chosen) in enumerate(requests):  # finished now, so the buckets go with the pass
            if index in totals:
                node = leaf.table.plan.reductions[hidden]  # type: ignore[union-attr]
                leaf.exact.setdefault(hidden, []).append((chosen, sums.exact_results(totals[index], node.op,
                                                                                     node.na_rm)))
        del totals


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
    rows = _block(plan, block, axis)
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
        pieces[hidden] = _pieces(node, value, codes if each == 1 else np.repeat(codes, each), count)
    return rows.found, combos, counts, pieces, rows.dropped


def _pieces(node: Reduce, value: np.ndarray, codes: np.ndarray, count: int) -> tuple[str, np.ndarray]:
    """One reduction's pieces per group from a block's values and their codes."""
    if node.op in ("min", "max", "any", "all"):
        order = np.argsort(codes, kind="stable")
        starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=count))[:-1]])
        if node.op in ("any", "all"):
            return "flag", _extreme(node).reduceat(value[order].astype(bool), starts)
        if value.dtype.kind in "US":  # text compares as Python objects, as the backends compare it
            value = value.astype(object)
        return "extreme", _extreme(node).reduceat(value[order], starts)
    if node.op in ("std", "var"):
        return "moments", _moments(value, codes, count)
    if value.dtype == np.float32:
        return "estimate", sums.chunk_estimate(value, (0,), na_rm=node.na_rm, codes=codes, groups=count)
    counted = np.bincount(codes, minlength=count).astype(np.float64)
    if value.dtype.kind == "f":
        missing = np.isnan(value)
        summed = np.where(missing, 0.0, value) if node.na_rm else value
        return "float", np.stack([np.bincount(codes, weights=summed, minlength=count),
                                  np.bincount(codes, weights=~missing, minlength=count), counted])
    if value.dtype.kind in "biu" and node.op == "mean":  # added in float64, as np.mean adds integers
        return "float", np.stack([np.bincount(codes, weights=value, minlength=count), counted, counted])
    if value.dtype.kind in "biu":  # added in int64, wrapping as np.sum does
        order = np.argsort(codes, kind="stable")
        starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=count))[:-1]])
        return "int", np.stack([np.add.reduceat(value[order].astype(np.int64), starts), counted.astype(np.int64)])
    raise BlockError(f"{node.text()}: {node.op} of {value.dtype} values is not handled on blocks; {WHOLE_AXIS}")


def _extreme(node: Reduce) -> Any:
    if node.op == "any":
        return np.logical_or
    if node.op == "all":
        return np.logical_and
    if node.op == "min":
        return np.fmin if node.na_rm else np.minimum
    return np.fmax if node.na_rm else np.maximum


def _moments(value: np.ndarray, codes: np.ndarray, count: int) -> np.ndarray:
    """Per group, in float64: the values present, their mean, their squared deviations' sum, and the NaN among them."""
    floats = value.astype(np.float64)
    missing = np.isnan(floats)
    present = np.bincount(codes, weights=~missing, minlength=count)
    total = np.bincount(codes, weights=np.where(missing, 0.0, floats), minlength=count)
    mean = np.divide(total, present, out=np.zeros(count), where=present > 0)
    deviations = np.where(missing, 0.0, floats - mean[codes]) ** 2
    squares = np.bincount(codes, weights=deviations, minlength=count)
    return np.stack([present, mean, squares, np.bincount(codes, weights=missing, minlength=count)])


def _combine_moments(held: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Two groups' moments as one (Chan, Golub, and LeVeque's parallel update), in float64."""
    na, ma, sa, nana = held
    nb, mb, sb, nanb = piece
    n = na + nb
    delta = mb - ma
    share = np.divide(nb, n, out=np.zeros_like(n), where=n > 0)
    return np.stack([n, ma + delta * share, sa + sb + delta * delta * na * share, nana + nanb])


_UNSEEN = object()  # an object accumulator's slot for a group no block has had yet


def _fill(kind: str, dtype: np.dtype, node: Reduce) -> Any:
    """The value a group not yet seen holds in an accumulator."""
    if kind == "flag":
        return node.op == "all"
    if kind != "extreme":
        return 0
    if dtype.kind in "fc":  # NaN where NaN is skipped (fmin, fmax), so a group of NaN stays NaN; else the far end
        return np.nan if node.na_rm else (np.inf if node.op == "min" else -np.inf)
    if dtype.kind in "mM":  # the far ends of the int64 the times are held in; its smallest is NaT
        limits = np.iinfo(np.int64)
        return np.array(limits.max if node.op == "min" else limits.min + 1, dtype=np.int64).view(dtype)[()]
    if dtype.kind == "O":
        return _UNSEEN
    limits = np.iinfo(dtype)
    return limits.max if node.op == "min" else limits.min


def _merged(node: Reduce, held: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """An accumulator's values for some groups and a block's pieces for them, combined; an object accumulator's
    unseen slots take the piece, since no other value compares with them."""
    if held.dtype == object:
        unseen = np.frompyfunc(lambda item: item is _UNSEEN, 1, 1)(held).astype(bool)
        held = np.where(unseen, piece, held)
    return _extreme(node)(held, piece)


def _settled_extreme(held: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """A min or max per group, missing where a group had no positions (an empty bin), as in memory: NaN of a float's
    own type, NaT, NaN among objects; integers and booleans have no missing value, so float64 when a group is empty."""
    empty = counts == 0
    kind = held.dtype.kind
    if kind == "O":
        unseen = np.frompyfunc(lambda item: item is _UNSEEN, 1, 1)(held).astype(bool)
        return np.where(empty | unseen, np.nan, held)
    if not empty.any():
        return held
    if kind in "fc":
        return np.where(empty, np.nan, held).astype(held.dtype)
    if kind in "mM":
        return np.where(empty, np.array("NaT", dtype=held.dtype), held)
    return np.where(empty, np.nan, held.astype(np.float64))


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
    if not blocks.labelled and not any(step._operation == "filter" for step in frame._steps):
        data = backend.unlabel(data, axis)  # the blocks' positions served the verbs; in memory the axis has no labels
    size = backend.sizes(data).get(axis, 0)
    notes = [note for note in frame._notes if size == 0 or "(all removed by" not in note]
    result = TidyArray(data, backend=backend, field_axis=frame._field_axis, origins=frame._origins,
                       source=frame._source, notes=notes, groups=frame._groups, record=frame._record)
    return _recorded(result)
