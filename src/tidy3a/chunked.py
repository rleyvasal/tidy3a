"""A file read in blocks, and the verbs run over every block (plan milestone 10, decision 10, stage b).

``read_array(path, ..., blocks=rows)`` gives a ``ChunkedFrame``: a frame
holding the file's first block of ``rows`` positions on its first axis.
Each verb runs on that block as it is called, so its checks, notes,
errors, and ``explain()`` appear at once, as they do in memory, and the
frame records the verb. When a result is asked for (``collect()``,
``pull``, ``to_dict()``, or a verb that needs the whole axis), the recorded
verbs run over every block, ``threads`` blocks at a time: a pipeline ending
in ``summarise`` merges the blocks' summaries into one table, so memory is
the blocks in flight and the groups, not the file; any other pipeline
concatenates the blocks' rows, for data that fits once filtered.

What a block can answer: ``select``, ``filter``, and ``mutate`` of
row-wise expressions; one ``group_by`` of value keys (plain fields,
``cells``, ``bins``, ``rounded``, several at once, with or without
``dropna``); and a ``summarise`` of ``n()``, ``sum``, ``mean``, ``min``,
and ``max`` of row-wise expressions, or expressions of those
(``mean(x) / n()``). Each key's groups are merged across blocks in the
order the key gives them (a cell's low edge, a bin's place among the
edges, a value's order), the missing group last, and several keys combine
as one code per occurring combination, as ``group_by`` does in memory.
Float32 sums and means are the same bits as in memory whatever the block
size (``labelarray.sums``: the blocks' estimates add, and the groups the
error bound cannot settle get a second pass over the file for their exact
sums). Float64 and integer sums and means add per block in float64 or
int64 and are repeatable, not bit-identical with the in-memory results.
Anything else is refused with a ``BlockError`` when the verb is called,
naming what needs the whole axis.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from labelarray import sums
from labelarray.transforms import Bins, Duration, GridCells

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
    resolve_value_key,
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
    labelled: bool = False  # whether the blocked axis has labels of its own (else filter labels it by position)

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
        blocks = Blocks(reopen, size, axis, rows, threads, build, first._backend.has_labels(first._data, axis))
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
        """The recorded verbs over every block: the merged summaries, or the rows kept, in memory, computed once."""
        if self._result is None:
            self._result = _execute(self) if self._summarised else _concatenate(self)
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
    before: Pipeline  # the row-wise verbs, run on each block
    key: ValueKey | None  # the group_by key: a plain field, a transform, or several keys (MultiKey)
    parts: tuple[ValueKey, ...]  # the key itself, or each key of several
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
    before = Pipeline([step for step in steps[:-1] if step._operation != "group_by"])
    return _Plan(before, keys[0] if keys else None, _parts(frame._groups), reductions, summaries)


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
            elif kind == "extreme":
                held[at] = _extreme(self.plan.reductions[hidden])(held[at], piece)
            else:
                held[..., at] += piece

    def finish(self, exact: Callable[[np.ndarray, str], np.ndarray]) -> dict[str, np.ndarray]:
        """Each reduction's values per combination; *exact(chosen, hidden)* gives the exact sums a second pass finds."""
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

    def group_codes(self) -> list[np.ndarray]:
        """Each part's code for every combination in the table."""
        assert self.keys is not None
        return list(np.unravel_index(self.keys, self.sizes)) if self.parts else []


def _spread(held: np.ndarray, where: np.ndarray, size: int, fill: Any) -> np.ndarray:
    """*held*, whose last axis runs along the groups, placed at *where* in a wider table."""
    wider = np.full((*held.shape[:-1], size), fill, dtype=held.dtype)
    wider[..., where] = held
    return wider


# ── The run ─────────────────────────────────────────────────────────────────


def _execute(frame: ChunkedFrame) -> TidyArray:
    plan, blocks = frame._plan, frame._blocks
    assert plan is not None
    table = _Table(plan)
    with ThreadPoolExecutor(blocks.threads) as pool:
        for found, local, counts, pieces, dropped in pool.map(lambda span: _partial(plan, blocks, span),
                                                               blocks.spans):
            table.take(found)
            table.dropped += dropped
            if local is not None:
                table.add(_combined(local, table), counts, pieces)
        if table.keys is None:  # no block kept a row, the first included, so its summary is the answer
            return frame._evolve()
        columns = table.finish(lambda chosen, hidden: _exact(plan, blocks, table, pool, chosen, hidden))
    return _result(frame, plan, table, columns)


def _block(plan: _Plan, blocks: Blocks, span: tuple[int, int]):
    """One block through the row-wise verbs: each key's groups and codes, the rows kept, and each reduction's values.

    Gives, per key, its labels without the missing group, the missing
    group's label if the block had one, how many rows it dropped, and each
    row's local code (the missing group last); then the rows kept (None
    when every row is), and the reductions' values on those rows.
    """
    block = blocks.load(*span) >> plan.before
    backend = block._backend
    size = backend.sizes(block._data)[blocks.axis]
    found: list[tuple[np.ndarray, Any, int]] = []
    local: list[np.ndarray] = []
    kept = np.ones(size, dtype=bool)
    for part in plan.parts:
        groups = resolve_value_key(part, block, [])
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
    values: dict[str, np.ndarray] = {}
    for hidden, node in plan.reductions.items():
        if node.arg is None:
            continue
        value = evaluate(node.arg, ctx)
        if not backend.is_array(value) or tuple(value.dims) != (blocks.axis,):
            dims = ", ".join(map(str, getattr(value, "dims", ()))) or "no axis"
            raise BlockError(f"{node.text()}: {node.arg.text()} varies along {dims}; on blocks a summary's "
                             f"argument varies along {blocks.axis} alone; {WHOLE_AXIS}")
        values[hidden] = value.values if all_kept else value.values[kept]
    kept_count = size if all_kept else int(kept.sum())
    return (found, [codes if all_kept else codes[kept] for codes in local], kept_count, values,
            size - kept_count)


def _combined(local: Sequence[tuple[np.ndarray, Any, np.ndarray]], table: _Table) -> np.ndarray:
    """A block's combinations in the table's codes: each part's local codes mapped, then the parts' codes as one number."""
    if not table.parts:
        return np.zeros(1, dtype=np.int64)
    return np.ravel_multi_index([part.codes(labels, missing is not None)[codes]
                                 for part, (labels, missing, codes) in zip(table.parts, local)], table.sizes)


def _partial(plan: _Plan, blocks: Blocks, span: tuple[int, int]):
    """One block's groups; each combination's local codes per key, with the key's labels, to map once the table has
    taken the groups in; the combinations' row counts; each reduction's pieces per combination; and the rows dropped."""
    found, local, size, values, dropped = _block(plan, blocks, span)
    if size == 0:
        return found, None, None, None, dropped
    # Codes per local combination, so the pieces are made once per group, not once per row.
    if plan.parts:
        sizes = [len(labels) + (missing is not None) for labels, missing, _ in found]
        keys, codes = np.unique(np.ravel_multi_index(local, sizes), return_inverse=True)
        codes = codes.reshape(-1)
        combos = [(labels, missing, where) for (labels, missing, _), where in zip(found, np.unravel_index(keys, sizes))]
    else:
        keys, codes, combos = np.zeros(1, dtype=np.int64), np.zeros(size, dtype=np.intp), []
    count = len(keys)
    counts = np.bincount(codes, minlength=count).astype(np.int64)
    pieces: dict[str, tuple[str, np.ndarray]] = {}
    order = starts = None
    for hidden, node in plan.reductions.items():
        if node.op == "n":
            continue
        value = values[hidden]
        if node.op in ("min", "max"):
            if order is None:
                order = np.argsort(codes, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            pieces[hidden] = ("extreme", _extreme(node).reduceat(value[order], starts))
        elif value.dtype == np.float32:
            pieces[hidden] = ("estimate", sums.chunk_estimate(value, (0,), na_rm=node.na_rm, codes=codes,
                                                               groups=count))
        elif value.dtype.kind == "f":
            missing = np.isnan(value)
            summed = np.where(missing, 0.0, value) if node.na_rm else value
            pieces[hidden] = ("float", np.stack([np.bincount(codes, weights=summed, minlength=count),
                                                 np.bincount(codes, weights=~missing, minlength=count)]))
        elif value.dtype.kind in "biu":
            if order is None:
                order = np.argsort(codes, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            pieces[hidden] = ("int", np.add.reduceat(value[order].astype(np.int64), starts))
        else:
            raise BlockError(f"{node.text()}: {node.op} of {value.dtype} values is not handled on blocks; "
                             f"{WHOLE_AXIS}")
    return found, combos, counts, pieces, dropped


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


def _exact(plan: _Plan, blocks: Blocks, table: _Table, pool: ThreadPoolExecutor, chosen: np.ndarray,
           hidden: str) -> np.ndarray:
    """The second pass: each block's exact sums for the combinations *chosen*, added (``labelarray.sums.chunk_exact``)."""
    keys = table.keys
    assert keys is not None

    def block(span: tuple[int, int]) -> Any:
        found, local, size, values, _ = _block(plan, blocks, span)
        if size == 0:
            return 0
        if table.parts:
            codes = np.ravel_multi_index([part.codes(labels, missing is not None)[rows]
                                          for part, (labels, missing, _), rows in zip(table.parts, found, local)],
                                         table.sizes)
            codes = np.searchsorted(keys, codes)
        else:
            codes = np.zeros(size, dtype=np.intp)
        return sums.chunk_exact(values[hidden], (0,), chosen, codes=codes, groups=len(keys))

    buckets: Any = 0
    for part in pool.map(block, blocks.spans):  # the buckets add in any order
        buckets = buckets + part
    return buckets


def _result(frame: ChunkedFrame, plan: _Plan, table: _Table, columns: dict[str, np.ndarray]) -> TidyArray:
    """The merged summaries as the frame ``summarise`` gave on the first block, with every block's values."""
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
    for name, node in plan.summaries.items():
        value = evaluate(node, ctx)
        values = value.values if backend.is_array(value) else np.asarray(value)
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


def _recorded(result: TidyArray) -> TidyArray:
    """*result* with its last recorded step's effect taken from it, not from the first block."""
    record = result._record
    if record is not None and record.steps:
        steps = record.steps
        result._record = Record(record.source, record.start, (*steps[:-1], Step(steps[-1].verb, snapshot(result))),
                                record.preview)
    return result


def _notes(frame: ChunkedFrame, plan: _Plan, table: _Table) -> list[str]:
    """The first block's notes, with the count of positions dropped for a missing key from every block."""
    key = plan.key
    prefix = f"changed: group_by({key.label}) dropped " if key is not None else None
    notes = [note for note in frame._notes if prefix is None or not note.startswith(prefix)]
    dropped = table.dropped
    if key is not None and dropped:
        lacking = [part.subject for part in table.parts if part.dropped]
        notes.append(f"{prefix}{dropped:,} position{'' if dropped == 1 else 's'} with a missing "
                     f"{' or '.join(lacking)}; dropna=False keeps them as "
                     f"{'groups' if isinstance(key, MultiKey) else 'a group'}")
    return notes


# ── Rows kept, concatenated ─────────────────────────────────────────────────


def _concatenate(frame: ChunkedFrame) -> TidyArray:
    """The row-wise verbs over every block, the blocks' rows joined along the blocked axis, in memory."""
    blocks, backend, axis = frame._blocks, frame._backend, frame._blocks.axis
    before = Pipeline([step for step in frame._steps if step._operation != "group_by"])

    def rows(span: tuple[int, int]) -> tuple[int, dict]:
        return span[0], (blocks.load(*span) >> before).to_dict()

    with ThreadPoolExecutor(blocks.threads) as pool:
        parts = list(pool.map(rows, blocks.spans))
    first = parts[0][1]
    variables: dict[Hashable, tuple] = {}
    for name, entry in first["data_vars"].items():
        dims = tuple(entry["dims"])
        if axis in dims:
            number = dims.index(axis)
            variables[name] = (dims, np.concatenate([part["data_vars"][name]["data"] for _, part in parts], number))
        else:
            variables[name] = (dims, entry["data"])
    coords: dict[Hashable, np.ndarray] = {}
    for name, entry in first["coords"].items():
        if name != axis:
            coords[name] = entry["data"]
        elif blocks.labelled:
            coords[name] = np.concatenate([part["coords"][name]["data"] for _, part in parts])
        else:  # filter labelled the axis with the positions kept, counted within each block
            coords[name] = np.concatenate([part["coords"][name]["data"] + start for start, part in parts])
    data = backend.from_records(variables, coords)
    size = backend.sizes(data).get(axis, 0)
    notes = [note for note in frame._notes if size == 0 or "(all removed by" not in note]
    result = TidyArray(data, backend=backend, field_axis=frame._field_axis, origins=frame._origins,
                       source=frame._source, notes=notes, groups=frame._groups, record=frame._record)
    return _recorded(result)
