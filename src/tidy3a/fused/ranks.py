"""The median's and quantiles' later passes, compiled (plan milestone 20).

A later pass counts, for each target (a rank a group asks for), its candidates by their next 16 bits: the group's
values present whose order-preserving key starts with the digits found so far. On NumPy each block's counts are
sparse pairs, sorted to count them; here one kernel adds each candidate into a dense table of the targets by their
65,536 digits, where that table is small enough, each thread adding its blocks into one table of its own. The counts
are integers, so they are the same whichever way they are made and added, and so is everything found from them.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np

DENSE_DIGITS = 1 << 22  # a block's dense table of targets × 65,536 digits holds at most this many counts
DIGITS = 65_536

SOURCE = """
def count_digits(keys, present, codes, each, first, second, prefix, above, done, out):
    shift = np.uint64(above - 16)
    for r in range(len(keys)):
        if not present[r]:
            continue
        key = keys[r]
        digit = np.int64((key >> shift) & np.uint64(0xFFFF))
        code = codes[r // each]
        for target in (first[code], second[code]):
            if target >= 0 and (done == 0 or (key >> np.uint64(above)) == prefix[target]):
                out[target * 65536 + digit] += 1
"""


class _Running:
    """One thread's counts for one pass: int32, spilled into int64 before a count could pass 2**31."""

    def __init__(self, size: int) -> None:
        self.counts = np.zeros(size, np.int32)
        self.added = 0
        self.spilled: np.ndarray | None = None

    def room(self, values: int) -> np.ndarray:
        if self.added + values >= 2**31 - 1:
            self.spilled = self.counts.astype(np.int64) if self.spilled is None else self.spilled + self.counts
            self.counts[:] = 0
            self.added = 0
        self.added += values
        return self.counts

    def total(self) -> np.ndarray:
        return self.counts.astype(np.int64) if self.spilled is None else self.spilled + self.counts


_RUNNING: dict[int, dict[int, _Running]] = {}  # per pass (its state's id), each thread's counts
_LOCK = threading.Lock()


@dataclass
class Counted:
    """A block's answer to a pass counted densely: the counts are in its thread's running table for the pass
    (``_RUNNING``), every block of a thread adding into one, so no block's table is made, held, or added on its
    own; the threads' tables add once, when the pass resolves, which their order cannot change."""

    state: Any

    def joined(self, other: Counted) -> Counted:
        return self

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """The threads' counts added, as the sparse pass gives them: ``(target × 65,536 + digit, count)``."""
        with _LOCK:
            running = list(_RUNNING.pop(id(self.state), {}).values())
        total = running[0].total()
        for other in running[1:]:
            total += other.total()
        found = np.flatnonzero(total)
        return found.astype(np.int64), total[found]


def counted(state: Any, keyed: Callable[[], tuple[np.ndarray, np.ndarray]], values: int, each: int,
            codes: np.ndarray) -> Counted | None:
    """A block's counts for the pass *state* asks (``reductions._Selection``), by the kernel, into its thread's
    running table, the values' keys and presence made by *keyed* only then; None where the dense table would be too
    large or nothing compiles, and the sparse pass counts."""
    from tidy3a.fused import unavailable

    targets = len(state.group)
    if not targets or targets * DIGITS > DENSE_DIGITS or unavailable() is not None:
        return None
    from tidy3a.fused.kernels import compile_source

    kernel = compile_source(SOURCE, "count_digits")
    out = running(state, values)
    keys, present = keyed()
    above = state.width - 16 * state.done
    kernel(keys, present, np.ascontiguousarray(codes, np.intp), max(each, 1), state.first, state.second,
           state.prefix, above, state.done, out)
    return Counted(state)


def _forget(key: int) -> None:
    """A collected pass's tables dropped: a finalizer, run wherever the garbage is collected, even in a thread holding
    ``_LOCK``, so it takes no lock (a dict's ``pop`` is safe on its own)."""
    _RUNNING.pop(key, None)


def pass_source(name: str, keys: list, compiler: Any, tests: list[str], value: Any) -> str | None:
    """Source of a later pass in one loop (plan milestone 20): each position through the filters, each key's bucket
    found in the table's (``kd{k}`` from ``lows[k]`` where its range is narrow, else ``kb{k}``, sorted, with ``kc{k}``
    its group's code there) or its missing group's
    (``mc[k]``: -1 where dropped, -2 where the table has none), the combination found in the table's (``rows``
    where the combinations are few, else ``table``, sorted), and each value
    of *value* counted, ``sign`` times, by its target's next digit. Returns 0, or ``-(i + 1)`` at the first position
    whose bucket or combination the table does not hold, or whose value the kernel cannot match NumPy on; its counts
    before that position are then taken back by the same loop with ``sign`` of -1. None for values whose keys it
    does not compute (float16, times, booleans)."""
    dtype = np.dtype(value.dtype)
    if dtype.kind not in "fiu" or dtype.kind == "f" and dtype.itemsize < 4:
        return None
    edges = [f"edges{k}" for k, key in enumerate(keys) if key.how == "bins"]
    lookups = [name for k in range(len(keys)) for name in (f"kb{k}", f"kc{k}", f"kd{k}")]
    arguments = ", ".join(["start", "stop", "sign", "values", *compiler.inputs, *edges, *lookups, "lows", "mc",
                           "strides", "table", "rows", "scratch", "bits", "first", "second", "prefix", "above", "done",
                           "out"])
    lines = [f"def {name}({arguments}):", "    shift = np.uint64(above - 16)", "    for i in range(start, stop):"]
    lines += [f"        if not ({test}):\n            continue" for test in tests]
    lines.append("        combined = 0")
    for k, key in enumerate(keys):
        lines += [f"        m{k} = False", "        for _ in range(1):"]
        lines += [f"            {line}".replace("return -1", "return -(i + 1)") for line in key.body(k, placing=False)]
        lines += [f"        if m{k}:", f"            if mc[{k}] == -1:", "                continue",
                  f"            if mc[{k}] < -1:", "                return -(i + 1)", f"            c{k} = mc[{k}]",
                  f"        elif len(kd{k}):", f"            at = b{k} - lows[{k}]",
                  f"            if at < 0 or at >= len(kd{k}) or kd{k}[at] < 0:", "                return -(i + 1)",
                  f"            c{k} = kd{k}[at]",
                  "        else:", "            lo = 0", f"            hi = len(kb{k})", "            while lo < hi:",
                  "                mid = (lo + hi) // 2", f"                if kb{k}[mid] < b{k}:",
                  "                    lo = mid + 1", "                else:", "                    hi = mid",
                  f"            if lo == len(kb{k}) or kb{k}[lo] != b{k}:", "                return -(i + 1)",
                  f"            c{k} = kc{k}[lo]", f"        combined += c{k} * strides[{k}]"]
    lines += ["        if len(rows):", "            if combined >= len(rows) or rows[combined] < 0:",
              "                return -(i + 1)", "            lo = rows[combined]", "        else:", "            lo = 0",
              "            hi = len(table)", "            while lo < hi:", "                mid = (lo + hi) // 2",
              "                if table[mid] < combined:", "                    lo = mid + 1", "                else:",
              "                    hi = mid", "            if lo == len(table) or table[lo] != combined:",
              "                return -(i + 1)", "        one = first[lo]", "        two = second[lo]",
              "        if one < 0 and two < 0:", "            continue"]
    width = 32 if dtype.kind in "iu" and dtype.itemsize < 4 else dtype.itemsize * 8
    top = f"np.uint{width}({1 << (width - 1)})"
    if dtype.kind == "f":
        ordered = ["        if v != v:", "            continue", "        scratch[0] = v", "        key = bits[0]",
                   f"        key = ~key if key & {top} else key | {top}"]
    elif dtype.kind == "i":
        ordered = ["        scratch[0] = v", "        key = bits[0] ^ " + top]
    else:
        ordered = ["        scratch[0] = v", "        key = bits[0]"]
    count = ordered + ["        key = np.uint64(key)", "        digit = np.int64((key >> shift) & np.uint64(0xFFFF))",
                       "        for target in (one, two):",
                       "            if target >= 0 and (done == 0 or (key >> np.uint64(above)) == prefix[target]):",
                       "                out[target * 65536 + digit] += sign"]
    if value.cells:
        lines += ["        for j in range(values):", f"            v = {value.text}"]
        lines += [f"    {line}" for line in count]
    else:
        lines += [f"        v = {value.text}", *count]
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def scratch_for(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """A one-value array a pass writes each value into, and the same bytes as the unsigned key it reads back."""
    dtype = np.dtype(dtype)
    if dtype.kind in "iu" and dtype.itemsize < 4:
        dtype = np.dtype(f"{dtype.kind}4")
    scratch = np.zeros(1, dtype)
    return scratch, scratch.view(f"u{dtype.itemsize}")


def running(state: Any, values: int) -> np.ndarray | None:
    """This thread's running table for the pass *state* asks, room made for *values* more counts; None where the
    dense table would be too large."""
    targets = len(state.group)
    if not targets or targets * DIGITS > DENSE_DIGITS:
        return None
    # Nothing is allocated under the lock: an allocation may collect garbage, whose finalizers (``_forget``) would
    # otherwise wait on the lock this thread holds.
    with _LOCK:
        tables = _RUNNING.get(id(state))
        table = None if tables is None else tables.get(threading.get_ident())
    if table is not None:
        return table.room(values)
    made = _Running(targets * DIGITS)  # only this thread makes its own
    with _LOCK:
        tables = _RUNNING.get(id(state))
        first = tables is None
        if first:
            tables = _RUNNING[id(state)] = {}
        tables[threading.get_ident()] = made
    if first:
        weakref.finalize(state, _forget, id(state))  # a pass left unresolved (an error) frees its tables too
    return made.room(values)
