"""Reductions as accumulators in a kernel (plan milestone 16, step 4): each built-in's first pass in one loop.

Each reduction of a summary is a slot: arrays per group, filled as the kernel reads each value of each position
once, and turned into the pieces its registry entry's first pass gives (``Reduction.partial_rows``), so a block's
table is built and merged as before:

- ``n``: the table counts positions itself; the kernel counts each group's positions and values.
- float32 ``sum`` and ``mean``: labelarray's estimate pieces (``chunk_estimate``), the total and the sum of
  magnitudes each added with Neumaier's compensation, which is within ``(6u + 4(nu)^2)`` of the sum of
  magnitudes: within the plain bound of 7 additions while ``n`` is under 2**25.5, so the terms say 7 (or the
  count, beyond it), which labelarray's ``settle`` certifies with, adding the blocks merged; the counts of values
  present, of +inf, and of -inf, exactly.
- float64 sums and means, and means of integers and booleans: the float64 sum in position order (``np.bincount``'s
  order, so a block of one value a position gives its bits), the values present, and the values counted.
- integer sums: int64, wrapping as ``np.sum`` does, and the values counted.
- ``min`` and ``max``: NumPy's ``minimum`` and ``maximum`` (a NaN wins), or ``fmin`` and ``fmax`` with ``na_rm``
  (a NaN is passed over), -0.0 below +0.0 as NumPy 2.5 orders them.
- ``any`` and ``all``: the flags, and for values that can be missing whether the group had one.
- ``std`` and ``var``: the values present, their mean, the squared deviations' sum (a second pass, as NumPy's
  first pass takes them), and the missing values.
- ``median`` and ``quantile``: the values present and missing, which their later passes start from.

Float64 results added in other orders than NumPy's path differ in their last bits (decision 3); the rest are the
same bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from labelarray import sums

from tidy3a.fused.expressions import Source, Unsupported

COMPENSATED = 2**25  # values a group may hold for its compensated sum to be within 7 plain additions' bound
EXACT = getattr(sums, "EXACT", None)  # labelarray's mark of an exact total (0.1.7), where it has one


@dataclass
class Slot:
    """One reduction in a kernel: *k* its number, *value* its argument's source, and its node."""

    k: int
    node: Any
    value: Source

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def kind(self) -> str:
        op, dtype = self.node.op, self.dtype
        if op in ("sum", "mean"):
            if dtype == np.float32:
                return "estimate"
            return "int" if op == "sum" and dtype.kind in "biu" else "float"
        if op in ("min", "max"):
            return "extreme"
        if op in ("any", "all"):
            return "flagged" if dtype.kind == "f" else "flag"
        if op in ("std", "var"):
            return "moments"
        if op in ("median", "quantile"):
            return "rank"
        raise Unsupported(f"{self.node.text()} runs on NumPy")

    def arrays(self, groups: int) -> dict[str, np.ndarray]:
        """The slot's accumulators, allocated for the kernel to fill, by name."""
        k, kind = self.k, self.kind
        if kind == "estimate":
            names = ["total", "carry", "magnitude", "mcarry"]
            arrays = {f"{name}{k}": np.zeros(groups) for name in names}
            arrays |= {f"{name}{k}": np.zeros(groups, np.int64) for name in ("present", "plus", "minus")}
            return arrays | {f"rounded{k}": np.zeros(groups, np.bool_)}  # whether an addition into the total rounded
        if kind == "float":
            return {f"total{k}": np.zeros(groups), f"present{k}": np.zeros(groups, np.int64)}
        if kind == "int":
            return {f"total{k}": np.zeros(groups, np.int64)}
        if kind == "extreme":
            return {f"extreme{k}": np.zeros(groups, self.dtype), f"seen{k}": np.zeros(groups, np.bool_)}
        if kind in ("flag", "flagged"):
            start = self.node.op == "all"
            return {f"flags{k}": np.full(groups, start), f"marks{k}": np.full(groups, start)}
        if kind == "moments":
            return {f"total{k}": np.zeros(groups), f"present{k}": np.zeros(groups, np.int64),
                    f"missing{k}": np.zeros(groups, np.int64), f"squares{k}": np.zeros(groups)}
        return {f"present{k}": np.zeros(groups, np.int64), f"missing{k}": np.zeros(groups, np.int64)}

    def update(self, v: str) -> list[str]:
        """Lines taking value *v* of group ``g`` into the slot."""
        k, kind, na_rm = self.k, self.kind, self.node.na_rm
        floating = self.dtype.kind == "f"
        if kind == "estimate":
            lines = [f"x{k} = np.float64({v})"]
            if na_rm:
                lines += [f"if x{k} != x{k}:", "    pass", "else:"]
                body = _estimate(k)
                lines += [f"    {line}" for line in body]
            else:
                lines += _estimate(k)
            return lines
        if kind == "float":
            lines = [f"x{k} = np.float64({v})"]
            if floating:
                skip = ["if x{k} == x{k}:".format(k=k), f"    present{k}[g] += 1"]
                if na_rm:
                    return lines + skip + [f"    total{k}[g] += x{k}"]
                return lines + skip + [f"total{k}[g] += x{k}"]
            return lines + [f"total{k}[g] += x{k}"]
        if kind == "int":
            return [f"total{k}[g] += np.int64({v})"]
        if kind == "extreme":
            lower = self.node.op == "min"
            better = (f"x{k} < extreme{k}[g] or (x{k} == extreme{k}[g] and np.signbit(x{k}))" if lower else
                      f"x{k} > extreme{k}[g] or (x{k} == extreme{k}[g] and not np.signbit(x{k}))")
            if not floating:
                better = f"x{k} < extreme{k}[g]" if lower else f"x{k} > extreme{k}[g]"
                return [f"x{k} = {v}", f"if not seen{k}[g] or {better}:", f"    extreme{k}[g] = x{k}",
                        f"    seen{k}[g] = True"]
            if na_rm:  # fmin and fmax: a NaN is passed over, unless the group has nothing else
                return [f"x{k} = {v}", f"if x{k} == x{k}:",
                        f"    if not seen{k}[g] or extreme{k}[g] != extreme{k}[g] or {better}:",
                        f"        extreme{k}[g] = x{k}", f"    seen{k}[g] = True",
                        f"elif not seen{k}[g]:", f"    extreme{k}[g] = x{k}", f"    seen{k}[g] = True"]
            return [f"x{k} = {v}", f"if not seen{k}[g] or x{k} != x{k} or "
                                   f"(extreme{k}[g] == extreme{k}[g] and ({better})):",
                    f"    extreme{k}[g] = x{k}", f"    seen{k}[g] = True"]
        if kind in ("flag", "flagged"):
            combine = "or" if self.node.op == "any" else "and"
            if kind == "flag":
                return [f"flags{k}[g] = flags{k}[g] {combine} ({v} != 0)"]
            mark = f"(x{k} != x{k})" if self.node.op == "any" else f"(x{k} == x{k})"
            flag = (f"(x{k} == x{k} and x{k} != 0)" if self.node.op == "any" else f"(x{k} != x{k} or x{k} != 0)")
            return [f"x{k} = {v}", f"flags{k}[g] = flags{k}[g] {combine} {flag}",
                    f"marks{k}[g] = marks{k}[g] {combine} {mark}"]
        if kind == "moments":
            return [f"x{k} = np.float64({v})", f"if x{k} == x{k}:", f"    present{k}[g] += 1",
                    f"    total{k}[g] += x{k}", "else:", f"    missing{k}[g] += 1"]
        missing = f"np.isnan({v})" if floating else "False"
        return [f"if {missing}:", f"    missing{k}[g] += 1", "else:", f"    present{k}[g] += 1"]

    def second(self, v: str) -> list[str]:
        """Lines of a second pass over the values (the spreads' squared deviations), else none."""
        k = self.k
        if self.kind != "moments":
            return []
        return [f"x{k} = np.float64({v})", f"if x{k} == x{k}:",
                f"    d{k} = x{k} - (total{k}[g] / present{k}[g])", f"    squares{k}[g] += d{k} * d{k}"]

    def pieces(self, arrays: dict[str, np.ndarray], counted: np.ndarray) -> tuple[str, np.ndarray]:
        """The registry's pieces for this slot, from its filled arrays and each group's count of values."""
        k, kind = self.k, self.kind
        if kind == "estimate":
            total = arrays[f"total{k}"] + arrays[f"carry{k}"]
            magnitude = arrays[f"magnitude{k}"] + arrays[f"mcarry{k}"]
            terms = np.where(counted <= COMPENSATED, 7, counted)
            if EXACT is not None:  # no addition rounded: the total is the exact sum, which labelarray rounds itself
                terms = np.where(arrays[f"rounded{k}"], terms, EXACT)
            pieces = np.stack([total, magnitude, arrays[f"present{k}"], counted, arrays[f"plus{k}"],
                               arrays[f"minus{k}"], terms]).astype(np.float64)
            return "estimate", pieces
        if kind == "float":
            present = arrays[f"present{k}"] if self.dtype.kind == "f" else counted
            return "float", np.stack([arrays[f"total{k}"], present, counted]).astype(np.float64)
        if kind == "int":
            return "int", np.stack([arrays[f"total{k}"], counted.astype(np.int64)])
        if kind == "extreme":
            return "extreme", arrays[f"extreme{k}"]
        if kind == "flag":
            return "flag", arrays[f"flags{k}"]
        if kind == "flagged":
            return "flagged", np.stack([arrays[f"flags{k}"], arrays[f"marks{k}"]])
        if kind == "moments":
            present = arrays[f"present{k}"].astype(np.float64)
            mean = np.divide(arrays[f"total{k}"], present, out=np.zeros(len(present)), where=present > 0)
            return "moments", np.stack([present, mean, arrays[f"squares{k}"],
                                        arrays[f"missing{k}"].astype(np.float64)])
        return f"rank:{self.dtype.str}", np.stack([arrays[f"present{k}"], arrays[f"missing{k}"]])


def _estimate(k: int) -> list[str]:
    """Neumaier's compensated addition of ``x{k}`` to the group's total and of its magnitude, and the counts."""
    return [f"t{k} = total{k}[g]", f"s{k} = t{k} + x{k}",
            f"if abs(t{k}) >= abs(x{k}):", f"    e{k} = (t{k} - s{k}) + x{k}",
            "else:", f"    e{k} = (x{k} - s{k}) + t{k}",
            f"carry{k}[g] += e{k}", f"if e{k} != 0:", f"    rounded{k}[g] = True",  # a NaN's error is NaN: rounded
            f"total{k}[g] = s{k}",
            f"size{k} = abs(x{k})", f"m{k} = magnitude{k}[g]", f"n{k} = m{k} + size{k}",  # not a{k}: the inputs'
            f"if m{k} >= size{k}:", f"    mcarry{k}[g] += (m{k} - n{k}) + size{k}",
            "else:", f"    mcarry{k}[g] += (size{k} - n{k}) + m{k}", f"magnitude{k}[g] = n{k}",
            f"if x{k} == x{k}:", f"    present{k}[g] += 1",
            f"if x{k} == np.inf:", f"    plus{k}[g] += 1", f"elif x{k} == -np.inf:", f"    minus{k}[g] += 1"]


def summary_source(name: str, inputs: list[str], slots: list[Slot]) -> str:
    """A kernel ``name(positions, cells, codes, a0, ..., <each slot's arrays>, counts)`` reading every value of every
    position once (twice for the spreads), adding it into its group's slots, and counting each group's positions."""
    arrays = [array for slot in slots for array in slot.arrays(1)]
    arguments = ", ".join(["positions", "cells", "codes", *inputs, *arrays, "counts"])
    lines = [f"def {name}({arguments}):", "    for i in range(positions):", "        g = codes[i]",
             "        if g < 0:", "            continue", "        counts[g] += 1"]  # -1: filtered out, or dropped
    lines += _loop(slots, "update")
    if any(slot.second("") for slot in slots):
        lines += ["    for i in range(positions):", "        g = codes[i]", "        if g < 0:", "            continue"]
        lines += _loop([slot for slot in slots if slot.second("")], "second")
    return "\n".join(lines) + "\n"


def _loop(slots: list[Slot], step: str) -> list[str]:
    lines = []
    for slot in slots:
        body = getattr(slot, step)(slot.value.text)
        if slot.value.cells:
            lines.append("        for j in range(cells):")
            lines += [f"            {line}" for line in body]
        else:
            lines += [f"        {line}" for line in body]
    return lines


def counted_values(counts: np.ndarray, slot: Slot, cells: int) -> np.ndarray:
    """Each group's count of the slot's values: its positions, times the values a position holds."""
    return counts * (cells if slot.value.cells else 1)


def compile_summary(source: str, name: str) -> Any:
    """*source*'s function *name*, compiled by Numba (``fused.kernels``)."""
    from tidy3a.fused.kernels import compile_source

    return compile_source(source, name)


assert sums.PIECES == 7  # the estimate's pieces, in labelarray's order: total, magnitude, present, values, +inf, -inf
