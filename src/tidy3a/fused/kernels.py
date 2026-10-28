"""Kernels from expression source (plan milestone 16): a loop over a block's positions, and over each position's
values where a step needs them, compiled by Numba.

Numba is imported here only when a kernel is built, so importing tidy3a loads none. The kernels release the GIL
(``nogil``), so tidy3a's block threads run them side by side (decision 4, as changed), and follow NumPy's rules for
division by zero (``error_model="numpy"``): an infinity or a NaN, not an exception. ``fastmath`` stays off, so no
multiply and add is fused into one rounding.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tidy3a.fused.expressions import Compiler, Source


def loop_source(name: str, compiler: Compiler, outputs: Sequence[Source]) -> str:
    """Source of a function ``name(positions, values, a0, ..., out0, ...)`` filling each output, per position or per
    value of a position as its step is."""
    arguments = ", ".join(["positions", "values", *compiler.inputs, *(f"out{k}" for k in range(len(outputs)))])
    lines = [f"def {name}({arguments}):", "    for i in range(positions):"]
    lines += [f"        out{k}[i] = {source.text}" for k, source in enumerate(outputs) if not source.cells]
    per_value = [(k, source) for k, source in enumerate(outputs) if source.cells]
    if per_value:
        lines.append("        for j in range(values):")
        lines += [f"            out{k}[i, j] = {source.text}" for k, source in per_value]
    return "\n".join(lines) + "\n"


COMPILED: dict[tuple[str, str], Any] = {}  # this process's kernels by their source: each compiled once


def compile_source(source: str, name: str, helpers: tuple[str, ...] = ()) -> Any:
    """*source*'s function *name*, with the *helpers* it defines and calls, compiled by Numba, once a process for
    each source, and kept on disk between runs (``fused.cache``) where it can be."""
    found = COMPILED.get((name, source))
    if found is not None:
        return found
    from tidy3a.fused.cache import load

    found = load(source, name, helpers)
    if found is not None:
        COMPILED[(name, source)] = found
        return found
    import numba

    namespace: dict[str, Any] = {"np": np}
    exec(compile(source, f"<tidy3a kernel {name}>", "exec"), namespace)
    for helper in helpers:
        namespace[helper] = numba.njit(nogil=True)(namespace[helper])
    found = COMPILED[(name, source)] = numba.njit(nogil=True, error_model="numpy")(namespace[name])
    return found
