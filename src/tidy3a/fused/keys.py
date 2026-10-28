"""Keys as dense codes (plan milestone 16, step 3): each position's bucket, computed in a kernel, then the groups.

A key's value at each position (compiled by ``fused.expressions``) is placed in a bucket numbered by an integer:
the value itself for an integer or boolean key, and for a float key whose values are all whole; the step for
``cells`` and ``rounded``, as labelarray's ``_Line`` places it (a float64 guess, corrected against the exact
edges, or integer arithmetic for integer values); the bin for ``bins``, as ``np.searchsorted`` finds it. The
kernel also finds the block's lowest and highest bucket, flags missing values (NaN), and flags any value where
agreement with NumPy is not established: a float key's value that is not whole, a negative zero, an infinity, a
value outside every bin, a bucket too far out for exact arithmetic. A block with such a value falls back to
NumPy, which gives its result, or its error.

``groups`` turns the buckets into what ``resolve_value_key`` gives a block: the labels in order, each position's
code, and the positions dropped for a missing key; the labels by labelarray's own code, for the occupied buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from tidy3a.expr import Context, ValueKey
from tidy3a.fused.expressions import Compiler, Source, Unsupported, cast

EXACT = 2**53  # integers a float64 holds exactly
GUESS = 2**50  # labelarray's _Line.floor refuses a step further out


@dataclass
class Key:
    """One key's placement: its value's source, how it is placed (``value``, ``steps``, or ``bins``), and the
    constants the kernel needs."""

    part: ValueKey
    value: Source
    how: str
    constants: dict

    def body(self, k: int) -> list[str]:
        """Kernel lines placing position ``i`` of key *k*: ``ks{k}[i]``, ``missing{k}[i]``, and the block's range."""
        v = f"v{k}"
        lines = [f"{v} = {self.value.text}"]
        floating = self.value.dtype.kind == "f"
        if floating:
            lines += [f"if {v} != {v}:", f"    missing{k}[i] = True", f"    count{k} += 1", "    continue"]
        if self.how == "value":
            if floating:
                lines += [f"if not (abs({v}) < {EXACT}.0) or {v} != np.floor({v}) or ({v} == 0 and np.signbit({v})):",
                          "    return -1", f"b{k} = np.int64({v})"]
            else:
                lines += [f"b{k} = np.int64({v})"]
        elif self.how == "steps":
            c = self.constants
            if floating:
                lines += [f"f{k} = np.float64({v})", f"if not np.isfinite(f{k}):", "    return -1",
                          f"g{k} = np.floor((f{k} - {c['float_start']!r}) / {c['float_step']!r})",
                          f"if abs(g{k}) >= {GUESS}.0:", "    return -1", f"b{k} = np.int64(g{k})",
                          f"while f{k} < _at(b{k}, {c['start']}, {c['step']}, {c['denominator']}):",
                          f"    b{k} -= 1",
                          f"while f{k} >= _at(b{k} + 1, {c['start']}, {c['step']}, {c['denominator']}):",
                          f"    b{k} += 1",
                          f"if _far(b{k} + 1, {c['start']}, {c['step']}) or _far(b{k} - 1, {c['start']}, {c['step']}):",
                          "    return -1"]
            else:
                lines += [f"w{k} = np.int64({v})",
                          f"if abs(w{k}) >= {c['reach']}:", "    return -1",
                          f"b{k} = (w{k} * {c['denominator']} - {c['start']}) // {c['step']}"]
        else:  # bins: the first edge above the value, less one, as np.searchsorted(edges, value, side="right") - 1
            compared = self.constants["compared"]
            lines += [f"e{k} = {cast(self.value, compared)}", f"lo{k} = 0", f"hi{k} = len(edges{k})",
                      f"while lo{k} < hi{k}:", f"    mid{k} = (lo{k} + hi{k}) // 2",
                      f"    if edges{k}[mid{k}] > e{k}:", f"        hi{k} = mid{k}", "    else:",
                      f"        lo{k} = mid{k} + 1", f"b{k} = lo{k} - 1",
                      f"if b{k} < 0 or b{k} >= len(edges{k}) - 1:", "    return -1"]
        lines += [f"ks{k}[i] = b{k}", f"if b{k} < low{k}:", f"    low{k} = b{k}", f"if b{k} > high{k}:",
                  f"    high{k} = b{k}"]
        return lines


def key_for(part: ValueKey, context: Context, compiler: Compiler) -> Key:
    """How *part* is placed in a kernel, or ``Unsupported`` and why."""
    from labelarray.transforms import Bins, GridCells, Round

    value = compiler.compile(part.expr, context)
    if value.cells:
        raise Unsupported(f"group_by({part.label}): {part.expr.text()} varies within a position, which runs on NumPy")
    dtype = value.dtype
    if dtype == np.uint64:  # its values may not fit the signed bucket numbers
        raise Unsupported(f"group_by({part.label}): unsigned 64-bit values run on NumPy")
    transform = part.transform
    if transform is None:
        if dtype is None or dtype.kind not in "biuf":
            raise Unsupported(f"group_by({part.label}) runs on NumPy")
        return Key(part, value, "value", {})
    if isinstance(transform, (GridCells, Round)):
        if dtype.kind not in "iuf":
            raise Unsupported(f"group_by({part.label}): its values are not numbers, which runs on NumPy")
        start = transform._start()
        denominator = _lcm(start, transform._step)
        whole_start, whole_step = int(start * denominator), int(transform._step * denominator)
        constants = {"float_start": float(start), "float_step": float(transform._step), "start": whole_start,
                     "step": whole_step, "denominator": denominator,
                     # an integer value's bucket in int64, as labelarray's floor_integers takes it
                     "reach": max(0, (2**63 - 1 - abs(whole_start)) // max(denominator, 1))}
        if denominator >= EXACT:
            raise Unsupported(f"group_by({part.label}): its steps need exact fractions, which runs on NumPy")
        return Key(part, value, "steps", constants)
    if isinstance(transform, Bins):
        edges = transform.edges
        if edges.dtype.kind not in "iuf" or dtype.kind not in "iuf":
            raise Unsupported(f"group_by({part.label}): bins of times run on NumPy")
        return Key(part, value, "bins", {"edges": edges, "compared": np.result_type(edges.dtype, dtype)})
    raise Unsupported(f"group_by({part.label}): {type(transform).__name__} keys run on NumPy")


def _lcm(*fractions: Fraction) -> int:
    from math import lcm

    return lcm(*(fraction.denominator for fraction in fractions))


def kernel_source(name: str, keys: list[Key], compiler: Compiler) -> str:
    """A function ``name(positions, a0, ..., [edges0, ...,] ks0, missing0, ...)`` placing every position of each key,
    returning 0, or -1 at the first value where agreement with NumPy is not established; each key's range and count
    of missing values are written into ``ranges[k] = (low, high, missing)``."""
    edges = [f"edges{k}" for k, key in enumerate(keys) if key.how == "bins"]
    outputs = [name for k in range(len(keys)) for name in (f"ks{k}", f"missing{k}")]
    arguments = ", ".join(["positions", *compiler.inputs, *edges, *outputs, "ranges"])
    lines = [f"def {name}({arguments}):"]
    for k in range(len(keys)):
        lines += [f"    low{k} = np.int64({2**63 - 1})", f"    high{k} = np.int64({-2**63})", f"    count{k} = 0"]
    lines.append("    for i in range(positions):")
    for k, key in enumerate(keys):
        # A missing value skips the key's other lines, not the other keys': each key's placement is a loop body of
        # its own, run once (for ... in range(1)), which `continue` leaves.
        lines.append("        for _ in range(1):")
        lines += [f"            {line}" for line in key.body(k)]
    for k in range(len(keys)):
        lines += [f"    ranges[{k}, 0] = low{k}", f"    ranges[{k}, 1] = high{k}", f"    ranges[{k}, 2] = count{k}"]
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


HELPERS = '''
def _at(k, start, step, denominator):
    return np.float64(start + k * step) / np.float64(denominator)


def _far(k, start, step):
    point = start + k * step
    return point >= 9007199254740992 or point <= -9007199254740992
'''


def compile_keys(source: str, name: str) -> Any:
    """*source*'s function *name* with the helpers it calls, compiled by Numba."""
    import numba

    namespace: dict[str, Any] = {"np": np}
    exec(compile(HELPERS + source, f"<tidy3a keys {name}>", "exec"), namespace)
    for helper in ("_at", "_far"):
        namespace[helper] = numba.njit(nogil=True)(namespace[helper])
    return numba.njit(nogil=True, error_model="numpy")(namespace[name])


def groups(key: Key, ks: np.ndarray, missing: np.ndarray, low: int, high: int, count: int,
           dtype: np.dtype) -> tuple[list, np.ndarray, int]:
    """The labels in order, each position's code, and the positions dropped, as ``resolve_value_key`` gives a
    block, from the kernel's buckets (*ks*, valid where not *missing*) and its range."""
    present = ~missing
    codes = np.empty(len(ks), dtype=np.intp)
    if key.how == "bins":
        labels = key.part.transform.groups(np.zeros(0, dtype=dtype))[1]
        codes[present] = ks[present]
    elif high >= low:
        placed = ks[present]
        if high - low < 4 * len(placed) + 1_024:  # counting the range, as labelarray's _occupied does
            occupied = np.bincount(placed - low, minlength=high - low + 1) > 0
            found = np.flatnonzero(occupied) + low
            codes[present] = (np.cumsum(occupied) - 1)[placed - low]
        else:  # a range much wider than the values: sorting them
            found, inverse = np.unique(placed, return_inverse=True)
            codes[present] = inverse.reshape(-1)
        if key.how == "value":
            labels = found.astype(dtype).tolist()
        else:
            labels = key.part.transform._labels(found, np.zeros(0, dtype=dtype))
    else:
        labels = []
    if not count:
        return labels, codes, 0
    codes[missing] = len(labels)
    if key.how == "value":
        labels = [*labels, *np.array(["nan"], dtype).tolist()]
    else:
        labels = [*labels, key.part.transform.missing_label(dtype)]
    return labels, codes, count if key.part.dropna else 0
