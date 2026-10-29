"""Rows sorted and written that need not fit in memory (plan milestone 21, step 4): an external merge sort.

``arrange(...) >> write(path)`` on data read in blocks: each block's kept rows, sorted on their own as the blocks run,
are spilled to a run of their own in a temporary folder beside the file, in pieces of up to ``PIECE`` rows, fewer
where the blocks are many (``WINDOWS`` rows for every run's two pieces together), with each row's
keys as unsigned integers (``verbs.merge_keys``); once every block is in, the runs are merged in one streaming pass,
a compiled heap taking the least keys, earlier runs first among ties, so the order is the stable order memory's sort
gives; and the merged rows are handed to the file's writer in blocks of the blocks' size. A merge round takes at most a
piece of rows from each run, so a run holds at most two pieces in memory, and the merge's memory does not grow with
the number of runs; the runs are spilled on threads of their own as the blocks arrive. The temporary folder is removed
whether the write finishes or fails.
"""

from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PIECE = 65_536  # rows a spilled piece holds at most, and a merge round takes
WINDOWS = 1 << 22  # rows the runs hold in memory together while merging, at most: two pieces each

SOURCE = """
def merge(keys, starts, lengths, more, taken, out_run, out_row):
    runs = len(starts)
    columns = keys.shape[1]
    heap = np.empty(runs, np.int64)  # runs with a row left in their windows, a binary heap by their next rows' keys
    size = 0
    for r in range(runs):
        if taken[r] < lengths[r]:
            heap[size] = r
            size += 1
            i = size - 1
            while i > 0:
                parent = (i - 1) // 2
                if _less(keys, starts, taken, heap[i], heap[parent], columns):
                    heap[i], heap[parent] = heap[parent], heap[i]
                    i = parent
                else:
                    break
    made = 0
    while size > 0:
        r = heap[0]
        out_run[made] = r
        out_row[made] = taken[r]
        made += 1
        taken[r] += 1
        if taken[r] >= lengths[r]:
            if more[r]:  # its window spent, its run not: the next round loads more of it before going on
                return made
            size -= 1
            heap[0] = heap[size]
        i = 0
        while True:  # the root sifted down to its place
            left = 2 * i + 1
            if left >= size:
                break
            child = left
            if left + 1 < size and _less(keys, starts, taken, heap[left + 1], heap[left], columns):
                child = left + 1
            if _less(keys, starts, taken, heap[child], heap[i], columns):
                heap[i], heap[child] = heap[child], heap[i]
                i = child
            else:
                break
    return made


def _less(keys, starts, taken, a, b, columns):
    ra = starts[a] + taken[a]
    rb = starts[b] + taken[b]
    for k in range(columns):
        if keys[ra, k] != keys[rb, k]:
            return keys[ra, k] < keys[rb, k]
    return a < b
"""


@dataclass
class _Run:
    """One block's sorted rows on disk: its pieces' places in its file, and their rows."""

    path: str
    piece_rows: int  # rows a piece holds (its last fewer)
    offsets: list[int] = field(default_factory=list)
    rows: int = 0
    taken: int = 0  # rows the merge has taken
    held: dict = field(default_factory=dict)  # pieces in memory, by number: (keys, data)

    def piece(self, number: int) -> tuple[np.ndarray, Any]:
        if number not in self.held:
            with open(self.path, "rb") as file:
                file.seek(self.offsets[number])
                self.held[number] = pickle.load(file)
        return self.held[number]

    def window(self, count: int) -> np.ndarray:
        """The keys of the next *count* rows (fewer at the run's end), its pieces before them let go."""
        first = self.taken // self.piece_rows
        for number in [number for number in self.held if number < first]:
            del self.held[number]
        stop = min(self.taken + count, self.rows)
        if stop > self.taken and (stop - 1) // self.piece_rows == first:  # within one piece: a view of its keys
            keys, _ = self.piece(first)
            return keys[self.taken - first * self.piece_rows:stop - first * self.piece_rows]
        parts = []
        for number in range(first, (stop - 1) // self.piece_rows + 1 if stop > self.taken else first):
            keys, _ = self.piece(number)
            start = max(self.taken - number * self.piece_rows, 0)
            parts.append(keys[start:stop - number * self.piece_rows])
        return np.concatenate(parts) if parts else np.zeros((0, 0), np.uint64)

    def rows_of(self, start: int, stop: int, backend: Any, axis: Any) -> list:
        """Its rows from *start* to *stop*, as the backend holds them, a part a piece."""
        parts = []
        for number in range(start // self.piece_rows, (stop - 1) // self.piece_rows + 1):
            _, data = self.piece(number)
            low, high = max(start - number * self.piece_rows, 0), min(stop - number * self.piece_rows, self.piece_rows)
            parts.append(backend.isel(data, {axis: slice(low, high)}))
        return parts


class SortedWriter:
    """A sort's rows written block by block (``write``'s ``Writer``), merged from runs spilled to disk: ``add`` each
    sorted block's rows in order, then ``finish``, or ``abort``; *first* and *last* keep only the first (or last)
    rows a slice after the sort asks for."""

    def __init__(self, writer: Any, frame: Any, keys: Any, first: int | None, last: bool) -> None:
        self.writer, self.frame, self.keys = writer, frame, keys
        self.first, self.last = first, last
        self.backend, self.axis = frame._backend, frame._blocks.axis
        rows = frame._blocks.rows
        self.out_rows = rows if isinstance(rows, int) else max((stop - start for start, stop in frame._layout().every),
                                                               default=PIECE)  # the file's blocks
        folder = os.path.dirname(os.path.abspath(os.fspath(writer.path))) or "."
        self.folder = tempfile.mkdtemp(prefix=".tidy3a-sort-", dir=folder)
        self.runs: list[_Run] = []
        self.spilling: list[Any] = []
        blocks = max(len(frame._layout().spans), 1)
        rows = min(PIECE, WINDOWS // (2 * blocks))  # a run a block, two pieces of each held while merging
        memory = frame._blocks.memory
        if memory is not None:  # and within the budget, the merge's rows and their keys
            from tidy3a.sizing import RESERVE, _bytes

            data = _bytes(frame, self.axis)  # a round holds a row in its pieces, its joined, ordered, and pending
            row = 5 * data + 3 * 8 * len(keys) + 16  # parts, and its keys and its place in the order
            rows = min(rows, max(memory - RESERVE, 0) // (blocks * row))
        self.piece_rows = max(min(PIECE, 1024), rows)
        self.threads = max(1, frame._blocks.threads)
        self.pool = ThreadPoolExecutor(self.threads)

    def add(self, data: Any) -> None:
        """One block's rows, sorted: spilled, on the writer's threads, as a run of pieces with their rows' keys."""
        size = int(self.backend.sizes(data).get(self.axis, 0))
        if size == 0:
            return
        waiting = [spill for spill in self.spilling if not spill.done()]
        if len(waiting) >= self.threads:  # a block a thread waiting to be spilled, at most, however fast blocks come
            wait(waiting, return_when=FIRST_COMPLETED)
        run = _Run(os.path.join(self.folder, f"run{len(self.runs)}.pickle"), self.piece_rows, rows=size)
        self.runs.append(run)  # in the blocks' order, whichever is spilled first
        self.spilling.append(self.pool.submit(self._spill, run, data))

    def _spill(self, run: _Run, data: Any) -> None:
        from tidy3a.frame import TidyArray
        from tidy3a.verbs import merge_keys, sort_columns

        backend, axis, size = self.backend, self.axis, run.rows
        rows = TidyArray(data, backend=backend, field_axis=self.frame._field_axis)
        _, columns, _ = sort_columns(rows, self.keys)
        keys = merge_keys(columns)
        with open(run.path, "wb") as file:
            for start in range(0, size, run.piece_rows):
                run.offsets.append(file.tell())
                stop = start + run.piece_rows
                piece = (np.ascontiguousarray(keys[start:stop]), backend.isel(data, {axis: slice(start, stop)}))
                pickle.dump(piece, file, protocol=pickle.HIGHEST_PROTOCOL)
    def finish(self) -> None:
        """Every run in: merged into the file, the slice's rows kept, the temporary folder removed."""
        from tidy3a.fused.kernels import compile_source

        try:
            for spilled in self.spilling:
                spilled.result()
            merge = compile_source(SOURCE, "merge", ("_less",))
            size = self.piece_rows  # a round's rows: at most a piece from any run
            total = sum(run.rows for run in self.runs)
            skip, keep = 0, total
            if self.first is not None:
                keep = min(self.first, total)
                skip = total - keep if self.last else 0
            made, pending, held = 0, [], 0
            backend, axis = self.backend, self.axis
            while made < skip + keep:
                windows = [run.window(size) for run in self.runs]
                lengths = np.array([len(window) for window in windows], np.int64)
                more = np.array([run.taken + len(window) < run.rows for run, window in zip(self.runs, windows)])
                starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
                keys = np.concatenate([window for window in windows if len(window)])
                taken = np.zeros(len(self.runs), np.int64)
                out_run = np.empty(len(keys), np.int64)
                out_row = np.empty(len(keys), np.int64)
                count = merge(keys, starts, lengths, more, taken, out_run, out_row)  # until a window is spent
                count = min(count, skip + keep - made)
                low, high = max(skip - made, 0), count  # the rows of this round the slice keeps
                if high > low:
                    parts, place = [], np.full(len(self.runs), -1, np.int64)
                    at = 0
                    for r, run in enumerate(self.runs):
                        if taken[r]:
                            parts += run.rows_of(run.taken, run.taken + int(taken[r]), backend, axis)
                            place[r] = at
                            at += int(taken[r])
                    joined = backend.join_along(parts, axis) if len(parts) > 1 else parts[0]
                    order = place[out_run[low:high]] + out_row[low:high]
                    pending.append(backend.keep(joined, axis, order))
                    held += high - low
                    if held >= self.out_rows:
                        self.writer.add(backend.join_along(pending, axis) if len(pending) > 1 else pending[0])
                        pending, held = [], 0
                for r, run in enumerate(self.runs):
                    run.taken += int(taken[r])
                made += count
            if pending:
                self.writer.add(backend.join_along(pending, axis) if len(pending) > 1 else pending[0])
            self.writer.finish()
        finally:
            self._clean()

    def abort(self) -> None:
        try:
            self.writer.abort()
        finally:
            self._clean()

    def reopen(self) -> Any:
        return self.writer.reopen()

    def _clean(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.runs, self.spilling = [], []
        shutil.rmtree(self.folder, ignore_errors=True)
