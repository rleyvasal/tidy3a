"""Prototype: run a tidy3a pipeline over memory-mapped files, block by block.

Decision 10, stage (b) of the plan, for the one pipeline shape that can be
combined exactly: row-wise verbs (``select``, ``mutate``, ``filter``), then at
most one ``group_by`` on a field, then ``summarise`` of ``n()``, ``sum``,
``mean``, ``min``, and ``max`` of any row-wise expression.

Each block of rows is read from the file (opened with ``mmap_mode="r"`` and
copied, so no mapping outlives the block), wrapped as an ordinary NumPy-backend
frame, and run through tidy3a's own verbs. Its summaries become a small
partial table (the block's groups and, per summary, pieces that add), and the
partial tables are merged by key, so memory is the block and the groups, not
the file.

Float32 sums and means use labelarray's chunk pieces (``chunk_estimate``,
``settle``, ``chunk_exact``), so they equal the in-memory results bit for bit
at any block size, with a second pass over the file only for the groups the
error bound cannot settle. Float64 sums and means are added per block in
float64 and are close to, not bit-identical with, the in-memory results.

Not handled (each raises ``NotImplementedError``): verbs that need the whole
axis (``slice``, joins, a reduction inside ``mutate`` or ``filter``), several
keys or key transforms (``cells``, ``bins``), integer arguments to ``sum`` and
``mean``, ``median``, ``std``, ``var``, ``min_count``, and ``over=``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from labelarray import sums

from tidy3a import infer, mutate, pull, read_array
from tidy3a.expr import Reduce, Ref, as_expr, walk
from tidy3a.plan import Pipeline

ROW_WISE = ("select", "filter", "mutate")
COMBINABLE = ("n", "sum", "mean", "min", "max")


@dataclass(frozen=True)
class Layout:
    """How a block of rows becomes a frame: the observation axis, then the field axis and its labels."""

    observation: str
    fields: Sequence[str]
    field_axis: str = "feature"

    def frame(self, values: np.ndarray) -> Any:
        return read_array(
            values, dims={self.observation: infer, self.field_axis: len(self.fields)},
            labels={self.field_axis: list(self.fields)}, field_axis=self.field_axis, backend="numpy",
        )


@dataclass
class Summary:
    name: str
    op: str
    na_rm: bool
    hidden: str | None  # the per-block variable holding the argument


@dataclass
class Plan:
    before: Pipeline
    key: str | None
    summaries: list[Summary]


@dataclass
class Result:
    """Per-group columns, in the order ``summarise`` would give them, and what the run did."""

    key: str | None
    columns: dict[str, np.ndarray]
    blocks: int = 0
    rows_read: int = 0
    rows_kept: int = 0
    missing_keys: int = 0
    exact_passes: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


# ── Analysis ────────────────────────────────────────────────────────────────


def analyse(pipeline: Pipeline) -> Plan:
    verbs = list(pipeline.verbs)
    ops = [verb._operation for verb in verbs]
    if not ops or ops[-1] != "summarise" or ops.count("summarise") != 1:
        raise NotImplementedError("the pipeline must end in one summarise()")
    body = verbs[:-1]
    key = None
    if body and ops[-2] == "group_by":
        key, body = _key(body[-1]), body[:-1]
    for verb in body:
        _row_wise(verb)
    groups, named = verbs[-1]._arguments
    if not named:
        raise NotImplementedError("summarise() needs at least one summary")
    summaries, hidden = [], {}
    for index, (name, expr) in enumerate(named.items()):
        node = as_expr(expr)
        if not isinstance(node, Reduce) or node.op not in COMBINABLE:
            raise NotImplementedError(f"{name}: only {', '.join(COMBINABLE)} of an expression combine across blocks")
        if node.over is not None or node.min_count is not None:
            raise NotImplementedError(f"{name}: over= and min_count= are not handled")
        if node.arg is not None:
            if any(isinstance(inner, Reduce) for inner in walk(node.arg)):
                raise NotImplementedError(f"{name}: a reduction inside a reduction needs the whole axis")
            hidden[f"__a{index}"] = node.arg
        summaries.append(Summary(name, node.op, node.na_rm, None if node.arg is None else f"__a{index}"))
    before = Pipeline(body + ([mutate(**hidden)] if hidden else []))
    return Plan(before, key, summaries)


def _key(verb: Any) -> str:
    keys, named = verb._arguments
    if named or len(keys) != 1 or not isinstance(keys[0], (str, Ref)):
        raise NotImplementedError("group_by() takes one field in this prototype")
    return keys[0].name if isinstance(keys[0], Ref) else keys[0]


def _row_wise(verb: Any) -> None:
    if verb._operation not in ROW_WISE:
        raise NotImplementedError(f"{verb.text}: {verb._operation} needs the whole axis, not one block")
    if verb._operation == "select":
        return
    values = [step.value for step in verb._arguments] if verb._operation == "mutate" else verb._arguments
    for value in values:
        if any(isinstance(node, Reduce) for node in walk(as_expr(value))):
            raise NotImplementedError(f"{verb.text}: a reduction needs the whole axis; summarise it per group instead")


# ── Blocks ──────────────────────────────────────────────────────────────────


def _spans(sources: Sequence[Path], rows: int) -> tuple[list[tuple[Path, int, int]], int]:
    spans, width = [], None
    for path in sources:
        shape = np.load(path, mmap_mode="r").shape
        if len(shape) != 2 or (width is not None and shape[1] != width):
            raise ValueError(f"{path}: expected 2-D files with the same number of columns")
        width = shape[1]
        spans += [(Path(path), start, min(start + rows, shape[0])) for start in range(0, shape[0], rows)]
    return spans, width


def _load(span: tuple[Path, int, int], layout: Layout, plan: Plan):
    """One block, run through the row-wise verbs: its key per kept row and its summary arguments."""
    path, start, stop = span
    mapped = np.load(path, mmap_mode="r")
    values = np.array(mapped[start:stop])  # a copy: the mapping closes with this block
    del mapped
    frame = layout.frame(values) >> plan.before
    key = None if plan.key is None else frame >> pull(plan.key)
    arguments = {s.hidden: frame >> pull(s.hidden) for s in plan.summaries if s.hidden is not None}
    kept = len(key) if key is not None else len(next(iter(arguments.values()))) if arguments else 0
    return key, arguments, len(values), kept


def _codes(key: np.ndarray | None, size: int):
    """Sorted groups, each row's code, and which rows have a key (rows with a missing key are left out)."""
    if key is None:
        return np.zeros(1, dtype=np.int64), np.zeros(size, dtype=np.intp), None
    present = ~np.isnan(key) if key.dtype.kind == "f" else None
    if present is not None and present.all():
        present = None
    uniq, codes = np.unique(key if present is None else key[present], return_inverse=True)
    return uniq, codes.reshape(-1), present


def _partial(plan: Plan, key, arguments, size):
    """One block's groups, row counts, and per-summary pieces."""
    uniq, codes, present = _codes(key, size)
    missing = 0 if present is None else int((~present).sum())
    if len(codes) == 0:  # nothing kept in this block
        return uniq[:0], np.zeros(0, dtype=np.int64), [], missing
    count = len(uniq)
    counts = np.bincount(codes, minlength=count)
    order = starts = None
    pieces: list[Any] = []
    for s in plan.summaries:
        if s.op == "n":
            pieces.append(None)
            continue
        values = arguments[s.hidden] if present is None else arguments[s.hidden][present]
        if s.op in ("min", "max"):
            if order is None:
                order = np.argsort(codes, kind="stable")
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            take = (np.fmin if s.na_rm else np.minimum) if s.op == "min" else (np.fmax if s.na_rm else np.maximum)
            pieces.append(("extreme", take.reduceat(values[order], starts)))
        elif values.dtype == np.float32:
            pieces.append(("estimate", sums.chunk_estimate(values, (0,), na_rm=s.na_rm, codes=codes, groups=count)))
        elif values.dtype == np.float64:
            nan = np.isnan(values)
            total = np.bincount(codes, weights=np.where(nan, 0.0, values) if s.na_rm else values, minlength=count)
            seen = np.bincount(codes, weights=~nan, minlength=count)
            pieces.append(("float64", np.stack([total, seen])))
        else:
            raise NotImplementedError(f"{s.name}: {s.op} of {values.dtype} is not handled")
    return uniq, counts, pieces, missing


# ── The run ─────────────────────────────────────────────────────────────────


def run(sources: str | Path | Sequence[str | Path], pipeline: Pipeline, layout: Layout, *,
        rows: int = 2_000_000, workers: int = 1) -> Result:
    """Run *pipeline* over the ``.npy`` file(s) in blocks of *rows* rows; *workers* blocks at a time."""
    sources = [Path(sources)] if isinstance(sources, (str, Path)) else [Path(s) for s in sources]
    plan = analyse(pipeline)
    spans, width = _spans(sources, rows)
    if width != len(layout.fields):
        raise ValueError(f"the files have {width} columns, the layout names {len(layout.fields)}")
    clock = time.perf_counter()

    def first_pass(span):
        key, arguments, read, kept = _load(span, layout, plan)
        return (*_partial(plan, key, arguments, kept), read, kept)

    pool = ThreadPoolExecutor(workers)  # the second pass, when a result needs it, uses the same threads
    partials = list(pool.map(first_pass, spans))
    taken = time.perf_counter() - clock

    missing = sum(p[3] for p in partials)
    partials = [p for p in partials if len(p[0])]
    result = Result(plan.key, {}, blocks=len(partials), missing_keys=missing)
    keys = np.unique(np.concatenate([p[0] for p in partials])) if partials else np.zeros(0)
    count = len(keys)
    where = [np.searchsorted(keys, p[0]) for p in partials]

    totals = np.zeros(count, dtype=np.int64)
    for index, p in zip(where, partials):
        totals[index] += p[1]
    columns: dict[str, np.ndarray] = {}
    if plan.key is not None:  # tidy3a labels groups of float keys as float64
        columns[plan.key] = keys.astype(np.float64) if keys.dtype.kind == "f" else keys
    for position, s in enumerate(plan.summaries):
        if s.op == "n":
            columns[s.name] = totals
        elif s.op in ("min", "max"):
            take = (np.fmin if s.na_rm else np.minimum) if s.op == "min" else (np.fmax if s.na_rm else np.maximum)
            fill = np.inf if s.op == "min" else -np.inf
            merged = np.full(count, fill, dtype=partials[0][2][position][1].dtype)
            for index, p in zip(where, partials):
                merged[index] = take(merged[index], p[2][position][1])
            columns[s.name] = merged
        else:
            columns[s.name] = _combine(plan, s, position, partials, where, count, totals, spans, layout, keys, result,
                                       pool)
    result.rows_read = sum(_rows(span) for span in spans)
    result.rows_kept = int(totals.sum())
    result.columns = columns
    result.seconds = {"blocks": taken, "total": time.perf_counter() - clock}
    pool.shutdown()
    return result


def _rows(span) -> int:
    return span[2] - span[1]


def _combine(plan, s, position, partials, where, count, totals, spans, layout, keys, result, pool):
    if partials[0][2][position][0] == "estimate":  # float32: the pieces add, then settle
        acc = np.zeros((sums.PIECES, count))
        for index, p in zip(where, partials):
            piece = p[2][position][1]
            acc[:sums.TERMS, index] += piece[:sums.TERMS]
            acc[sums.TERMS, index] = np.maximum(acc[sums.TERMS, index], piece[sums.TERMS])

        def exact(chosen):
            result.exact_passes += 1

            def block(span):
                key, arguments, _, _ = _load(span, layout, plan)
                values = arguments[s.hidden]
                if key is not None:
                    present = ~np.isnan(key) if key.dtype.kind == "f" else np.ones(len(key), bool)
                    values, key = values[present], key[present]
                if len(values) == 0:
                    return 0
                codes = np.zeros(len(values), dtype=np.intp) if key is None else np.searchsorted(keys, key)
                return sums.chunk_exact(values, (0,), chosen, codes=codes, groups=count)

            buckets = 0
            for part in pool.map(block, spans):  # blocks run in parallel; the buckets add in any order
                buckets = buckets + part
            return buckets

        return sums.from_chunks(acc, len(partials), s.op, exact, na_rm=s.na_rm)
    total = np.zeros(count)
    seen = np.zeros(count)
    for index, p in zip(where, partials):
        total[index] += p[2][position][1][0]
        seen[index] += p[2][position][1][1]
    if s.op == "sum":
        return total
    return total / (seen if s.na_rm else totals)
