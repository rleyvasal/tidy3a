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
row, and a reduction's later passes to be asked for, which work on slices
of a block (``chunked.SLICE``), so they cost the same whatever its size;
``tests/test_sizing.py`` holds it above the peaks ``tracemalloc`` measures.
A budget counts the blocks' arrays at 1.5 times (``ALLOCATOR``), for what
the allocator keeps of what they free. What the data decides is
not counted: how many groups a block's rows fall into, which each
reduction's pieces and the summary table grow with (a summary with about
as many groups as positions holds about as much as its data), and the rows
a pipeline without ``summarise`` or ``write`` keeps in memory.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tidy3a.chunked import SLICE, ChunkedFrame, _Leaf, _needs, _Node, _trie
from tidy3a.errors import GroupError, ReadError, SummariseError
from tidy3a.expr import Context, Ref, evaluate
from tidy3a.frame import TidyArray
from tidy3a.fused.reductions import kind_of
from tidy3a.reductions import REGISTRY

RESERVE = 256 * 2**20  # the interpreter, the libraries, and the summary table, set aside from a budget
ALLOCATOR = (3, 2)  # what the blocks' arrays take in the process, as a ratio to their own size: the allocator keeps
#   some of what a block frees, which over hundreds of blocks of changing sizes measured a quarter more than the
#   estimate on the 29.4 GB sweeps file with one thread (milestone 14, step 6); 1.5 times covers it
OBJECT = 64  # bytes a text or other Python object is taken to hold, beyond the array's pointer to it
KEY = 56  # bytes per row a group key's codes take: its values factorized (a sorted copy, the order, the inverse, the
#   codes) and, in a later pass, its codes in the table's
COMBINED = 40  # bytes per row the keys' combinations take: one number per row, factorized, and the counts


UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
         "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def budget(memory: Any) -> int:
    """``memory=`` as bytes: a number of bytes, or text such as ``"2GB"`` or ``"512 MiB"`` (GB is 10**9 bytes, GiB
    2**30); None for a quarter of the machine's physical memory (milestone 14, decision 3)."""
    if memory is None:
        physical = physical_memory()
        if physical is None:
            raise ReadError('blocks="auto" sizes blocks for a quarter of physical memory, which this system does not '
                            'report; give the budget with memory=, e.g. memory="2GB"')
        return physical // 4
    if isinstance(memory, str):
        found = re.fullmatch(r"\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-zA-Z]*)\s*", memory)
        unit = found and UNITS.get(found.group(2).lower())
        if not unit:
            raise ReadError(f'memory= is a number of bytes, or text such as "2GB" or "512MiB", not {memory!r}')
        memory = float(found.group(1)) * unit
    if isinstance(memory, bool) or not isinstance(memory, (int, float, np.integer, np.floating)) or not memory > 0:
        raise ReadError(f'memory= is a number of bytes, or text such as "2GB" or "512MiB", not {memory!r}')
    return int(memory)


def physical_memory() -> int | None:
    """The machine's physical memory in bytes, where the system reports it."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # not a POSIX system
        pass
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("physical", ctypes.c_ulonglong),
                        *((name, ctypes.c_ulonglong) for name in ("free", "file", "file_free", "virtual",
                                                                   "virtual_free", "extended"))]

        status = Status(ctypes.sizeof(Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.physical)
    except (AttributeError, OSError):
        pass
    return None


def describe(size: int) -> str:
    """Bytes as people read them: "6.4 GB", "512 MB"."""
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("kB", 10**3)):
        if size >= scale:
            return f"{size / scale:.3g} {unit}"
    return f"{size} bytes"


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
        merged), with what the allocator keeps of them (``ALLOCATOR``)."""
        return RESERVE + threads * self.fixed + rows * self.per_position(threads)

    def per_position(self, threads: int) -> int:
        """What one more position in every block costs a run on *threads* threads, the allocator's share included."""
        times, over = ALLOCATOR
        return -(-(threads * self.work + (threads + 2) * self.held) * times // over)

    def rows(self, budget: int, threads: int) -> int:
        """The most positions a block may hold for *budget* bytes on *threads* threads; 0 when not even one fits."""
        free = budget - RESERVE - threads * self.fixed
        per_position = self.per_position(threads)
        return max(0, free // per_position) if per_position else max(0, free)


def cost(frames: Sequence[ChunkedFrame]) -> Cost:
    """The cost of one pass of *frames*, which share their blocks, as ``scan`` runs them: each block loaded with the
    variables some frame needs, walked depth first through the frames' verbs, a prefix's frame kept while its
    branches run, and each frame's summary or rows taken where its verbs end."""
    blocks = frames[0]._blocks
    axis = blocks.axis
    needs = [_needs(frame) for frame in frames]
    names = None if any(need is None for need in needs) else frozenset().union(*needs)
    empty = blocks.source.empty(names)
    if not blocks.labelled:
        empty = empty._evolve(empty._backend.label_positions(empty._data, axis, 0))
    loaded = _bytes(empty, axis)  # a dataset's block holds only the variables read (milestone 12)
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
        new = _new(frame, after, axis) + _held_while(verb, frame, axis)
        work, result, extra = _walk(child, after, axis)
        across = _across(frame, after, axis)
        peak, held, fixed = max(peak, 2 * new, new + work), held + result, max(fixed, across + extra)
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
    kinds, most = [], 1  # how a kernel accumulates each reduction, and the most values a position holds
    for node in plan.reductions.values():
        if node.arg is None:
            continue
        value = evaluate(node.arg, context)
        dtype = np.dtype(value.dtype)
        each = _each(value, axis)
        kinds.append(kind_of(node.op, dtype))
        most = max(most, each)
        if droppable or not isinstance(node.arg, Ref) or value.dims[0] != axis:
            values += each * _itemsize(dtype)  # computed, copied without the rows left out, or reshaped
        entry = REGISTRY[node.op]
        spread = 8 * each if each > 1 and not entry.by_position else 0  # the rows' codes repeated for each value
        working = entry.working(dtype)
        first = max(first, spread + each * working.first)
        if working.later:  # a slice at a time (``chunked.SLICE``), the codes repeated in it, the rows' in the table
            later = max(later, 8 * len(keys) + 16)
            fixed = max(fixed, working.fixed + SLICE * (working.later + 8))
        fixed = max(fixed, working.fixed)
    from tidy3a.fused import plan_for

    fusion = plan_for(frame)
    if fusion.compiled:  # its blocks' dense tables (``fused.driver``), whatever a block's size, and its islands
        from tidy3a.fused.driver import BY_POSITION, KERNEL_THREADS, ORDER_FREE, dense_limit
        from tidy3a.fused.ranks import DENSE_DIGITS

        threads = frame._blocks.threads
        limit = dense_limit(frame._blocks.memory, threads)
        fixed = max(fixed, limit)
        if all(kind in ORDER_FREE for kind in kinds) and not (most >= BY_POSITION and threads < KERNEL_THREADS):
            fixed = max(fixed, 2 * limit)  # a thread's running table, and its copy while it widens or is merged
        if "rank" in kinds:
            fixed = max(fixed, 4 * DENSE_DIGITS)  # a later pass's running table of digits, a thread's (milestone 20)
        work += fusion.island_bytes
    return work + values + max(first, later), 0, fixed  # a block's pieces are per group


def _held_while(verb: Any, frame: TidyArray, axis: str) -> int:
    """What a step holds per position while it runs beyond the arrays it leaves (milestone 21): a join with data read
    in blocks too, the other's part a block reads, as read; a block's sort, its order and its keys."""
    from tidy3a.chunked import JOINS, ChunkedFrame, _needs

    if verb._operation in JOINS and isinstance(verb._arguments[0], ChunkedFrame):
        other = verb._arguments[0]
        part = other._blocks.source.empty(_needs(other))
        return _bytes(part, axis) if part._backend.has_axis(part._data, axis) else 0
    if verb._operation == "arrange_block":
        from tidy3a.verbs import sort_columns

        keys = verb._arguments[0]
        if not keys:
            return 0
        _, columns, _ = sort_columns(frame, keys)
        return 8 + sum(_itemsize(values.dtype) for values, _ in columns)  # the order, and each key's values
    return 0


def _new(before: TidyArray, after: TidyArray, axis: str) -> int:
    """What a verb adds per position: the arrays along the axis that are not the ones it was given (a mutate makes
    its own; a filter copies every one, and the labels with them)."""
    held = before._backend.variables(before._data)
    along = [(name, variable) for name, variable in after._backend.variables(after._data).items()
             if axis in variable.dims]
    fresh = [variable for name, variable in along if name not in held or variable.data is not held[name].data]
    total = sum(_per_position(variable, axis) for variable in fresh)
    return total + (_labels(after, axis) if fresh and len(fresh) == len(along) else 0)


def _across(before: TidyArray, after: TidyArray, axis: str) -> int:
    """What a verb adds that does not lie along the axis, the same for every block: a join's mask or weights over the
    other axes (milestone 19), or a mutate's array of them."""
    held = before._backend.variables(before._data)
    return sum(int(np.prod(variable.shape, dtype=np.int64)) * _itemsize(np.dtype(variable.dtype))
               for name, variable in after._backend.variables(after._data).items()
               if axis not in variable.dims and (name not in held or variable.data is not held[name].data))


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

