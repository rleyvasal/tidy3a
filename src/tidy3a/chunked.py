"""A file read in blocks, and the verbs run over every block (plan milestone 10, decision 10, stage b).

``read_array(path, ..., blocks=rows)`` gives a ``ChunkedFrame``: a frame
holding the file's first block of ``rows`` positions on its first axis.
Each verb runs on that block as it is called, so its checks, notes,
errors, and ``explain()`` appear at once, as they do in memory, and the
frame records the verb. When a result is asked for (``collect()``,
``pull``, ``to_dict()``, or a verb that needs the whole axis), the recorded
verbs run over every block, ``threads`` blocks at a time, and the blocks'
summaries merge into one table: memory is the blocks in flight and the
groups, not the file.

What a block can answer: ``select``, ``filter``, and ``mutate`` of
row-wise expressions; one ``group_by`` on a plain field; and a
``summarise`` of ``n()``, ``sum``, ``mean``, ``min``, and ``max`` of
row-wise expressions, or expressions of those (``mean(x) / n()``). Float32
sums and means are the same bits as in memory whatever the block size
(``labelarray.sums``: the blocks' estimates add, and the groups the error
bound cannot settle get a second pass over the file for their exact sums).
Float64 and integer sums and means add per block in float64 or int64 and
are repeatable, not bit-identical with the in-memory results. Anything else
is refused with a ``BlockError`` when the verb is called, naming what
needs the whole axis.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from labelarray import sums

from tidy3a.errors import BlockError, SummariseError, translating
from tidy3a.expr import (
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
    walk,
)
from tidy3a.frame import TidyArray
from tidy3a.plan import Pipeline, Record, Step, snapshot

ROW_WISE = ("select", "filter", "mutate")
COMBINABLE = ("n", "sum", "mean", "min", "max")
WHOLE_AXIS = "read the file with lazy=True, which memory-maps it, to run this on every position"


@dataclass(frozen=True)
class Blocks:
    """Where the blocks come from: the file reopened per block, and the reader's options applied to each."""

    reopen: Callable[[], np.ndarray]
    size: int  # positions on the blocked axis
    axis: str
    rows: int
    threads: int
    build: Callable[[np.ndarray, int, int], TidyArray]

    @property
    def spans(self) -> list[tuple[int, int]]:
        return [(start, min(start + self.rows, self.size)) for start in range(0, self.size, self.rows)] or [(0, 0)]

    def load(self, start: int, stop: int) -> TidyArray:
        """One block as a frame: a copy out of the map, which closes with it."""
        mapped = self.reopen()
        values = np.array(mapped[start:stop])
        del mapped
        if not values.dtype.isnative:
            values = values.astype(values.dtype.newbyteorder("="))
        return self.build(values, start, stop)


class ChunkedFrame(TidyArray):
    """The first block of a file read with ``blocks=``, with the verbs to run over every block."""

    __slots__ = ("_blocks", "_steps", "_plan", "_result")

    @classmethod
    def open(cls, reopen: Callable[[], np.ndarray], size: int, axis: str, rows: int, threads: int,
             build: Callable[[np.ndarray, int, int], TidyArray]) -> ChunkedFrame:
        blocks = Blocks(reopen, size, axis, rows, threads, build)
        first = blocks.load(*blocks.spans[0])
        frame = cls._wrap(first, blocks, ())
        count = len(blocks.spans)
        frame._record = Record(first._source or "data", snapshot(first),
                               preview=f"the first of {count:,} block{'' if count == 1 else 's'} of {rows:,} "
                                       f"positions on {axis}")
        return frame

    @classmethod
    def _wrap(cls, frame: TidyArray, blocks: Blocks, steps: Sequence[Any], plan: Any = None) -> ChunkedFrame:
        new = cls.__new__(cls)
        TidyArray.__init__(new, frame._data, backend=frame._backend, field_axis=frame._field_axis,
                           origins=frame._origins, source=frame._source, notes=frame._notes, groups=frame._groups,
                           resolved=frame._resolved, record=frame._record)
        new._blocks, new._steps, new._plan, new._result = blocks, tuple(steps), plan, None
        return new

    def _apply_verb(self, verb: Any) -> Any:
        operation = getattr(verb, "_operation", None)
        if self._summarised or operation not in (*ROW_WISE, "group_by", "summarise"):
            return verb(self._run())  # needs the whole axis: the blocks' result, in memory
        result = TidyArray._apply_verb(self, verb)  # on the first block, with its checks and record
        steps = (*self._steps, verb)
        plan = _check(self, verb, result, steps)
        return self._wrap(result, self._blocks, steps, plan)

    @property
    def _summarised(self) -> bool:
        return bool(self._steps) and self._steps[-1]._operation == "summarise"

    def _run(self) -> TidyArray:
        """The recorded verbs over every block, merged: a frame in memory, computed once."""
        if self._result is None:
            if not self._summarised:
                raise BlockError(f"blocks give a result after summarise(); {WHOLE_AXIS}")
            self._result = _execute(self)
        return self._result

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
        """Each step's effect on the first block, or on every block once a result has been computed."""
        return self._result.explain() if self._result is not None else TidyArray.explain(self)

    @translating
    def __repr__(self) -> str:
        if self._result is not None:
            return repr(self._result)
        blocks = self._blocks
        count = len(blocks.spans)
        return (f"{TidyArray.__repr__(self)}\nblocks: {count:,} of {blocks.rows:,} positions on {blocks.axis}; "
                "the first block is shown, and every block runs when a result is asked for")


# ── Checks at verb time ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Plan:
    before: Pipeline  # the row-wise verbs and group_by, run on each block
    key: ValueKey | None
    reductions: dict[str, Reduce]  # hidden name -> the reduction combined across blocks
    summaries: dict[str, Any]  # name -> its expression, with each reduction replaced by its hidden name


def _check(frame: ChunkedFrame, verb: Any, result: TidyArray, steps: Sequence[Any]) -> _Plan | None:
    operation = verb._operation
    if any(step._operation == "group_by" for step in frame._steps) and operation != "summarise":
        raise BlockError(f"{verb.text}: after group_by, only summarise() runs on blocks; a grouped {operation} "
                         f"needs every position of a group, which blocks split; {WHOLE_AXIS}")
    if operation in ("filter", "mutate"):
        values = [step.value for step in verb._arguments] if operation == "mutate" else verb._arguments
        for value in values:
            node = as_expr(value)
            found = next((inner for inner in walk(node) if isinstance(inner, Reduce)), None)
            if found is not None:
                raise BlockError(f"{verb.text}: {found.text()} needs the whole axis, which blocks split; "
                                 f"summarise it per group instead, or {WHOLE_AXIS}")
        return None
    if operation == "group_by":
        keys = result._groups
        key = keys[0] if len(keys) == 1 else None
        if (not isinstance(key, ValueKey) or isinstance(key, MultiKey) or key.transform is not None
                or not key.dropna):
            raise BlockError(f"{verb.text}: on blocks, group_by takes one plain field for now; several keys, "
                             "cells(), bins(), rounded(), dropna=False, and axes are planned (docs/plan.md, "
                             f"decision 10, stage (b), phase 2); {WHOLE_AXIS}")
        return None
    if operation == "summarise":
        return _analyse(frame, verb, steps)
    return None


def _analyse(frame: ChunkedFrame, verb: Any, steps: Sequence[Any]) -> _Plan:
    _, named = verb._arguments
    reductions: dict[str, Reduce] = {}
    summaries = {name: _hidden(as_expr(expr), reductions, name) for name, expr in named.items()}
    keys = [key for key in frame._groups if isinstance(key, ValueKey)]
    axes = [key for key in frame._groups if isinstance(key, AxisKey)]
    if axes:
        raise BlockError(f"{verb.text}: group_by({axes[0].label}) keeps an axis, which blocks split; {WHOLE_AXIS}")
    return _Plan(Pipeline(steps[:-1]), keys[0] if keys else None, reductions, summaries)


def _hidden(node: Any, reductions: dict[str, Reduce], name: str) -> Any:
    """*node* with each reduction replaced by a hidden name, the reductions collected; refuses what cannot combine."""
    if isinstance(node, Reduce):
        if node.op not in COMBINABLE:
            raise BlockError(f"{name} = {node.text()}: {node.op}() cannot be combined across blocks; "
                             f"{', '.join(COMBINABLE)} can; {WHOLE_AXIS}")
        if node.over is not None or node.min_count is not None:
            raise BlockError(f"{name} = {node.text()}: over= and min_count= are not handled on blocks; {WHOLE_AXIS}")
        if node.arg is not None and any(isinstance(inner, Reduce) for inner in walk(node.arg)):
            raise BlockError(f"{name} = {node.text()}: a reduction inside a reduction needs the whole axis; "
                             f"{WHOLE_AXIS}")
        hidden = f"__{len(reductions)}"
        reductions[hidden] = node
        return Ref(hidden)
    if isinstance(node, Literal):
        return node
    if isinstance(node, (BinOp, UnaryOp, Call)):
        node = copy.copy(node)
        if isinstance(node, BinOp):
            node.left, node.right = _hidden(node.left, reductions, name), _hidden(node.right, reductions, name)
        elif isinstance(node, UnaryOp):
            node.operand = _hidden(node.operand, reductions, name)
        else:
            node.args = tuple(_hidden(arg, reductions, name) for arg in node.args)
        return node
    raise BlockError(f"{name} = {node.text()}: a summary on blocks is a reduction, or an expression of reductions "
                     f"such as mean(x) / n(); {WHOLE_AXIS}")


# ── The run ─────────────────────────────────────────────────────────────────


def _execute(frame: ChunkedFrame) -> TidyArray:
    plan, blocks = frame._plan, frame._blocks
    assert plan is not None
    table = _Table(plan)
    with ThreadPoolExecutor(blocks.threads) as pool:
        for partial in pool.map(lambda span: _partial(plan, blocks, span), blocks.spans):
            table.add(partial)
        if table.keys is None:  # no block kept a row, the first included, so its summary is the answer
            return frame._evolve()
        columns = table.finish(lambda chosen, hidden: _exact(plan, blocks, table, pool, chosen, hidden))
    return _result(frame, plan, table, columns)


def _block(plan: _Plan, blocks: Blocks, span: tuple[int, int]):
    """One block through the row-wise verbs: its groups' labels, each kept row's code, and each reduction's values."""
    block = blocks.load(*span) >> plan.before
    backend = block._backend
    if plan.key is None:
        size = backend.sizes(block._data)[blocks.axis]
        labels, codes, dropped = np.zeros(1, dtype=np.int64), np.zeros(size, dtype=np.intp), 0
        kept = None
    else:
        groups = Context(block, verb="summarise", error=SummariseError, mode="summarise").groups()
        labels, codes, dropped = _label_array(groups.shown), np.asarray(groups.codes), groups.dropped
        kept = codes < len(labels) if dropped else None
        if kept is not None:
            codes = codes[kept]
    ctx = Context(block, verb="summarise", error=SummariseError, grouped=False)
    values: dict[str, np.ndarray] = {}
    for hidden, node in plan.reductions.items():
        if node.arg is None:
            continue
        found = evaluate(node.arg, ctx)
        if not backend.is_array(found) or tuple(found.dims) != (blocks.axis,):
            dims = ", ".join(map(str, getattr(found, "dims", ()))) or "no axis"
            raise BlockError(f"{node.text()}: {node.arg.text()} varies along {dims}; on blocks a summary's "
                             f"argument varies along {blocks.axis} alone; {WHOLE_AXIS}")
        values[hidden] = found.values if kept is None else found.values[kept]
    return labels, codes, values, dropped


def _partial(plan: _Plan, blocks: Blocks, span: tuple[int, int]):
    """One block's groups, their row counts, and each reduction's pieces, which add across blocks."""
    labels, codes, values, dropped = _block(plan, blocks, span)
    count = len(labels)
    if len(codes) == 0:
        return labels[:0], np.zeros(0, dtype=np.int64), {}, dropped
    counts = np.bincount(codes, minlength=count).astype(np.int64)
    order = starts = None
    pieces: dict[str, tuple[str, np.ndarray]] = {}
    for hidden, node in plan.reductions.items():
        if node.op == "n":
            continue
        found = values[hidden]
        if node.op in ("min", "max"):
            if order is None:
                order = np.argsort(codes, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            pieces[hidden] = ("extreme", _extreme(node).reduceat(found[order], starts))
        elif found.dtype == np.float32:
            pieces[hidden] = ("estimate", sums.chunk_estimate(found, (0,), na_rm=node.na_rm, codes=codes,
                                                               groups=count))
        elif found.dtype.kind == "f":
            missing = np.isnan(found)
            summed = np.where(missing, 0.0, found) if node.na_rm else found
            pieces[hidden] = ("float", np.stack([np.bincount(codes, weights=summed, minlength=count),
                                                 np.bincount(codes, weights=~missing, minlength=count)]))
        elif found.dtype.kind in "biu":
            if order is None:
                order = np.argsort(codes, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            pieces[hidden] = ("int", np.add.reduceat(found[order].astype(np.int64), starts))
        else:
            raise BlockError(f"{node.text()}: {node.op} of {found.dtype} values is not handled on blocks; "
                             f"{WHOLE_AXIS}")
    return labels, counts, pieces, dropped


def _extreme(node: Reduce) -> Any:
    if node.op == "min":
        return np.fmin if node.na_rm else np.minimum
    return np.fmax if node.na_rm else np.maximum


def _fill(kind: str, dtype: np.dtype, node: Reduce) -> Any:
    """The value a group not yet seen holds in an accumulator."""
    if kind != "extreme":
        return 0
    if dtype.kind == "f":
        return np.inf if node.op == "min" else -np.inf
    limits = np.iinfo(dtype)
    return limits.max if node.op == "min" else limits.min


class _Table:
    """The accumulating table: the groups seen so far, sorted, with their counts and pieces."""

    def __init__(self, plan: _Plan):
        self.plan = plan
        self.keys: np.ndarray | None = None
        self.counts = np.zeros(0, dtype=np.int64)
        self.pieces: dict[str, tuple[str, np.ndarray]] = {}
        self.dropped = 0
        self.blocks = 0  # blocks with rows, for the error bound of the combined float32 estimates

    def add(self, partial: tuple) -> None:
        labels, counts, pieces, dropped = partial
        self.dropped += dropped
        if len(labels) == 0:
            return
        self.blocks += 1
        if self.keys is None:
            self.keys, self.counts, self.pieces = labels, counts, dict(pieces)
            return
        union = np.union1d(self.keys, labels)
        if len(union) != len(self.keys):
            where = np.searchsorted(union, self.keys)
            self.counts = _spread(self.counts, where, len(union), 0)
            for hidden, (kind, held) in self.pieces.items():
                fill = _fill(kind, held.dtype, self.plan.reductions[hidden])
                self.pieces[hidden] = (kind, _spread(held, where, len(union), fill))
            self.keys = union
        at = np.searchsorted(union, labels)
        self.counts[at] += counts
        for hidden, (kind, piece) in pieces.items():
            held = self.pieces[hidden][1]
            if kind == "estimate":
                held[:sums.TERMS, at] += piece[:sums.TERMS]
                held[sums.TERMS, at] = np.maximum(held[sums.TERMS, at], piece[sums.TERMS])
            elif kind == "extreme":
                held[at] = _extreme(self.plan.reductions[hidden])(held[at], piece)
            else:
                held[..., at] += piece

    def finish(self, exact: Callable[[np.ndarray, str], np.ndarray]) -> dict[str, np.ndarray]:
        """Each reduction's values per group; *exact(chosen, hidden)* gives the exact sums a second pass finds."""
        if self.keys is None:
            self.keys = np.zeros(0, dtype=np.int64)
        columns: dict[str, np.ndarray] = {}
        for hidden, node in self.plan.reductions.items():
            if node.op == "n":
                columns[hidden] = self.counts
                continue
            kind, held = self.pieces[hidden]
            if kind == "estimate":
                columns[hidden] = sums.from_chunks(held, self.blocks, node.op, lambda chosen: exact(chosen, hidden),
                                                   na_rm=node.na_rm)
            elif kind == "extreme":
                columns[hidden] = held
            elif kind == "float":
                total, present = held
                columns[hidden] = total if node.op == "sum" else total / (present if node.na_rm else self.counts)
            else:
                columns[hidden] = held if node.op == "sum" else held / self.counts
        return columns


def _spread(held: np.ndarray, where: np.ndarray, size: int, fill: Any) -> np.ndarray:
    """*held*, whose last axis runs along the groups, placed at *where* in a wider table."""
    wider = np.full((*held.shape[:-1], size), fill, dtype=held.dtype)
    wider[..., where] = held
    return wider


def _exact(plan: _Plan, blocks: Blocks, table: _Table, pool: ThreadPoolExecutor, chosen: np.ndarray,
           hidden: str) -> np.ndarray:
    """The second pass: each block's exact sums for the groups *chosen*, added (``labelarray.sums.chunk_exact``)."""
    keys = table.keys
    assert keys is not None
    node = plan.reductions[hidden]

    def block(span: tuple[int, int]) -> Any:
        labels, codes, values, _ = _block(plan, blocks, span)
        if len(codes) == 0:
            return 0
        return sums.chunk_exact(values[hidden], (0,), chosen, codes=np.searchsorted(keys, labels)[codes],
                                groups=len(keys))

    buckets: Any = 0
    for part in pool.map(block, blocks.spans):  # the buckets add in any order
        buckets = buckets + part
    return buckets


def _result(frame: ChunkedFrame, plan: _Plan, table: _Table, columns: dict[str, np.ndarray]) -> TidyArray:
    """The merged summaries as the frame ``summarise`` gave on the first block, with every block's values."""
    backend, key = frame._backend, plan.key
    keys = table.keys
    assert keys is not None
    axis = key.name if key is not None else "group"
    hidden = {name: backend.array(values, dims=(axis,), coords={axis: keys} if key is not None else None, name=name)
              for name, values in columns.items()}
    held = TidyArray(backend.summary(hidden, {}, {}), backend=backend)
    ctx = Context(held, verb="summarise", error=SummariseError, grouped=False)
    results = {}
    for name, node in plan.summaries.items():
        value = evaluate(node, ctx)
        values = value.values if backend.is_array(value) else np.asarray(value)
        if key is None:
            values = np.asarray(values).reshape(-1)[0] if np.ndim(values) else values
            results[name] = backend.array(values, name=name)
        else:
            results[name] = backend.array(values, dims=(axis,), coords={axis: keys}, name=name)
    data = backend.summary(results, backend.attrs(frame._data), {})
    result = frame._evolve(data)
    result = TidyArray(result._data, backend=backend, field_axis=result._field_axis, origins=result._origins,
                       source=result._source, notes=_notes(frame, plan, table), groups=result._groups,
                       record=result._record)
    if result._record is not None and result._record.steps:  # the last step's effect, from every block
        steps = result._record.steps
        result._record = Record(result._record.source, result._record.start,
                                (*steps[:-1], Step(steps[-1].verb, snapshot(result))), result._record.preview)
    return result


def _notes(frame: ChunkedFrame, plan: _Plan, table: _Table) -> list[str]:
    """The first block's notes, with the count of positions dropped for a missing key from every block."""
    key = plan.key
    prefix = f"changed: group_by({key.label}) dropped " if key is not None else None
    notes = [note for note in frame._notes if prefix is None or not note.startswith(prefix)]
    if key is not None and table.dropped:
        count = table.dropped
        notes.append(f"{prefix}{count:,} position{'' if count == 1 else 's'} with a missing {key.label}; "
                     "dropna=False keeps them as a group")
    return notes
