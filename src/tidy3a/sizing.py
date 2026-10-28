"""How many positions a block may hold for a memory budget (plan milestone 14).

A run holds, per block running, what the block's variables take, what each
verb adds to them, and what a summary's keys and reductions allocate; and,
per block finished but not yet merged, its result. Each is estimated per
position from the types and shapes the verbs give on the block of no
positions (plan decision 4), so nothing is read, and the same data, verbs,
budget, and threads give the same blocks every time (milestone 14,
decision 1): a size measured on the data would differ between machines and
runs, and with it the float64 sums, which are repeatable for a given size,
and the block numbers that errors and ``preview()`` give.

The estimate is a bound, not a forecast: a filter is taken to keep every
row, and a reduction's later passes to be asked for; ``tests/test_sizing.py``
holds it above the peaks ``tracemalloc`` measures. What the data decides is
not counted: how many groups a block's rows fall into, which each
reduction's pieces and the summary table grow with (a summary with about
as many groups as positions holds about as much as its data), and the rows
a pipeline without ``summarise`` or ``write`` keeps in memory.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tidy3a.chunked import ChunkedFrame, DatasetSource, _Leaf, _needs, _Node, _trie
from tidy3a.errors import GroupError, SummariseError
from tidy3a.expr import Context, Ref, evaluate
from tidy3a.frame import TidyArray
from tidy3a.reductions import REGISTRY

RESERVE = 256 * 2**20  # the interpreter, the libraries, and the summary table, set aside from a budget
OBJECT = 64  # bytes a text or other Python object is taken to hold, beyond the array's pointer to it
KEY = 56  # bytes per row a group key's codes take: its values factorized (a sorted copy, the order, the inverse, the
#   codes) and, in a later pass, its codes in the table's
COMBINED = 40  # bytes per row the keys' combinations take: one number per row, factorized, and the counts


@dataclass(frozen=True)
class Cost:
    """What a run holds: bytes per position of a block running (``work``) and of a block's result waiting to be
    merged, joined, or written (``held``), and bytes per block running whatever its size (``fixed``)."""

    work: int
    held: int
    fixed: int = 0

    def bound(self, rows: int, threads: int) -> int:
        """The most a run of blocks of *rows* positions on *threads* threads holds, by the estimate: *threads* blocks
        running, and at most *threads* + 2 results waiting (``_in_order`` starts *threads* + 1 ahead of the one being
        merged)."""
        return RESERVE + threads * (self.fixed + rows * self.work) + (threads + 2) * rows * self.held

    def rows(self, budget: int, threads: int) -> int:
        """The most positions a block may hold for *budget* bytes on *threads* threads; 0 when not even one fits."""
        free = budget - RESERVE - threads * self.fixed
        per_position = threads * self.work + (threads + 2) * self.held
        return max(0, free // per_position) if per_position else max(0, free)


def cost(frames: Sequence[ChunkedFrame]) -> Cost:
    """The cost of one pass of *frames*, which share their blocks, as ``scan`` runs them: each block loaded with the
    variables some frame needs, walked depth first through the frames' verbs, a prefix's frame kept while its
    branches run, and each frame's summary or rows taken where its verbs end."""
    blocks = frames[0]._blocks
    axis = blocks.axis
    needs = [_needs(frame) for frame in frames]
    names = None if any(need is None for need in needs) else frozenset().union(*needs)
    empty = blocks.source.empty()
    if not blocks.labelled:
        empty = empty._evolve(empty._backend.label_positions(empty._data, axis, 0))
    loaded = _loaded(blocks.source, empty, names, axis)
    with warnings.catch_warnings():  # a reduction of no values warns; here that says nothing
        warnings.simplefilter("ignore", RuntimeWarning)
        work, held, fixed = _walk(_trie([_Leaf(frame, None, []) for frame in frames]), empty, axis)
    return Cost(loaded + work + held, held, fixed)


def _walk(node: _Node, frame: TidyArray, axis: str) -> tuple[int, int, int]:
    """What the walk from *node* adds per position at most beyond *frame*, what its leaves' results hold, and the
    leaves' bytes per block: a verb's new arrays live while its branches run, and its temporaries are taken to be as
    large as what it adds."""
    peak = held = fixed = 0
    for leaf in node.leaves:
        work, result, extra = _leaf(leaf.frame, frame, axis)
        peak, held, fixed = max(peak, work), held + result, max(fixed, extra)
    for verb, child in node.next:
        after = frame >> verb
        new = _new(frame, after, axis)
        work, result, extra = _walk(child, after, axis)
        peak, held, fixed = max(peak, 2 * new, new + work), held + result, max(fixed, extra)
    return peak, held, fixed


def _leaf(frame: ChunkedFrame, rows: TidyArray, axis: str) -> tuple[int, int, int]:
    """One frame's work at the end of its row-wise verbs, on *rows*: its summary, or its rows kept or written."""
    if not frame._summarised:
        return 0, _bytes(rows, axis), 0
    plan = frame._plan
    context = Context(rows, verb="summarise", error=SummariseError, grouped=False)
    keys = [evaluate(part.expr, Context(rows, verb="group_by", error=GroupError, grouped=False))
            for part in plan.parts]
    work = sum(_per_position(key, axis) + KEY for key in keys) + (COMBINED if keys else 0) + 8
    droppable = any(np.dtype(key.dtype).kind in "fcmMO" for key in keys)  # a missing key leaves its rows out
    values = first = later = fixed = 0
    for node in plan.reductions.values():
        if node.arg is None:
            continue
        value = evaluate(node.arg, context)
        dtype = np.dtype(value.dtype)
        each = _each(value, axis)
        if droppable or not isinstance(node.arg, Ref) or value.dims[0] != axis:
            values += each * _itemsize(dtype)  # computed, copied without the rows left out, or reshaped
        spread = 8 * each if each > 1 else 0  # the rows' codes repeated for each value
        working = REGISTRY[node.op].working(dtype)
        first = max(first, spread + each * working.first)
        if working.later:
            later = max(later, spread + each * working.later + 8 * len(keys) + 16)  # and the rows' codes in the table
        fixed = max(fixed, working.fixed)
    return work + values + max(first, later), 0, fixed  # a block's pieces are per group


def _loaded(source: Any, empty: TidyArray, names: frozenset | None, axis: str) -> int:
    """What a block takes as loaded, per position: the variables a dataset's block computes (milestone 12), or every
    one, and the axis's labels."""
    backend, data = empty._backend, empty._data
    total = _labels(empty, axis)
    variables = {name: variable for name, variable in backend.variables(data).items() if axis in variable.dims}
    if names is not None and isinstance(source, DatasetSource):
        kept = [name for name in variables if name in names]
        variables = {name: variables[name] for name in kept} if kept else (
            dict([min(variables.items(), key=lambda item: _per_position(item[1], axis))]) if variables else {})
    return total + sum(_per_position(variable, axis) for variable in variables.values())


def _new(before: TidyArray, after: TidyArray, axis: str) -> int:
    """What a verb adds per position: the arrays along the axis that are not the ones it was given (a mutate makes
    its own; a filter copies every one, and the labels with them)."""
    held = before._backend.variables(before._data)
    along = [(name, variable) for name, variable in after._backend.variables(after._data).items()
             if axis in variable.dims]
    fresh = [variable for name, variable in along if name not in held or variable.data is not held[name].data]
    total = sum(_per_position(variable, axis) for variable in fresh)
    return total + (_labels(after, axis) if fresh and len(fresh) == len(along) else 0)


def _bytes(frame: TidyArray, axis: str) -> int:
    """What a frame's arrays along the axis, and its labels there, take per position."""
    variables = frame._backend.variables(frame._data).values()
    return _labels(frame, axis) + sum(_per_position(variable, axis) for variable in variables if axis in variable.dims)


def _labels(frame: TidyArray, axis: str) -> int:
    backend, data = frame._backend, frame._data
    if not backend.has_labels(data, axis):
        return 0
    return _itemsize(np.asarray(backend.label_values(data, axis)).dtype)


def _per_position(array: Any, axis: str) -> int:
    return _each(array, axis) * _itemsize(np.dtype(array.dtype))


def _each(array: Any, axis: str) -> int:
    """Values per position: the sizes of the array's other axes."""
    return int(np.prod([size for dim, size in zip(array.dims, array.shape) if dim != axis], dtype=np.int64))


def _itemsize(dtype: np.dtype) -> int:
    return dtype.itemsize + OBJECT if dtype.kind == "O" else dtype.itemsize

