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

import math

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


def _divmod(bits: str) -> str:
    """NumPy's floored division and remainder of floats (``npy_divmod``, its steps one by one), in *bits*' type."""
    f = f"np.float{bits}"
    return f'''
def _fdiv{bits}(a, b):
    mod = {f}(np.fmod(a, b))
    if not b:
        return {f}(a / b)
    div = {f}({f}(a - mod) / b)
    if mod:
        if (b < 0) != (mod < 0):
            div = {f}(div - {f}(1.0))
    if div:
        floordiv = {f}(np.floor(div))
        if {f}(div - floordiv) > {f}(0.5):
            floordiv = {f}(floordiv + {f}(1.0))
        return floordiv
    return {f}(math.copysign(0.0, {f}(a / b)))


def _fmod{bits}(a, b):
    mod = {f}(np.fmod(a, b))
    if not b:
        return mod
    if mod:
        if (b < 0) != (mod < 0):
            mod = {f}(mod + b)
        return mod
    return {f}(math.copysign(0.0, b))
'''


# Helpers the compiled expressions call (``fused.expressions``), each added to a kernel's source where it calls it.
HELPERS = {
    "_fdiv32": _divmod("32"), "_fmod32": "", "_fdiv64": _divmod("64"), "_fmod64": "",
    "_idiv": '''
def _idiv(a, b):
    if b == 0:
        return a - a
    if b == -1:
        return -a
    return a // b
''',
    "_imod": '''
def _imod(a, b):
    if b == 0 or b == -1:
        return a - a
    return a % b
''',
}  # NumPy's integer rules: 0 for a zero divisor, the smallest value // -1 wrapping, a remainder of the divisor's sign


def _helpers(source: str) -> tuple[str, tuple[str, ...]]:
    """The helpers *source* calls: their source, put before it, and their names."""
    called = [name for name in HELPERS if f"{name}(" in source]
    texts = dict.fromkeys(HELPERS[name] or HELPERS[name.replace("mod", "div")] for name in called)
    return "".join(texts), tuple(called)


COMPILED: dict[tuple[str, str], Any] = {}  # this process's kernels by their source: each compiled once


def compile_source(source: str, name: str, helpers: tuple[str, ...] = ()) -> Any:
    """*source*'s function *name*, with the *helpers* it defines and calls, compiled by Numba, once a process for
    each source, and kept on disk between runs (``fused.cache``) where it can be."""
    found = COMPILED.get((name, source))
    if found is not None:
        return found
    prelude, called = _helpers(source)
    key = (name, source)
    source, helpers = prelude + source, (*helpers, *called)
    from tidy3a.fused.cache import load

    found = load(source, name, helpers)
    if found is not None:
        COMPILED[key] = found
        return found
    import numba

    namespace: dict[str, Any] = {"np": np, "math": math}
    exec(compile(source, f"<tidy3a kernel {name}>", "exec"), namespace)
    for helper in helpers:
        namespace[helper] = numba.njit(nogil=True, error_model="numpy")(namespace[helper])
    found = COMPILED[key] = numba.njit(nogil=True, error_model="numpy")(namespace[name])
    return found
