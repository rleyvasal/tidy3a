"""Expressions compiled to Numba (plan milestone 16, step 2): NumPy's bits, operation by operation.

Each pair of types runs every expression NumPy takes for it, compiled into one kernel, over every pair of
special values (zeros of both signs, infinities, NaN, the smallest subnormal, the largest float, integer limits),
against NumPy's own result: the same type, and the same bits (any NaN for a NaN, whose payload NumPy and the
compiled code need not share).
"""

import re
import warnings

import numpy as np
import pytest

from tidy3a import ceil, conj, exp, fields, floor, formula, if_else, log, nan, sign, sin, sqrt, tidy
from tidy3a.errors import MutateError
from tidy3a.expr import Context, evaluate
from tidy3a.fused.expressions import Compiler, Island, Unsupported, read

pytest.importorskip("numba")
from tidy3a.fused.kernels import compile_source, loop_source  # noqa: E402

x, y, z, h, t = fields("x", "y", "z", "h", "t")
FLOATS = [0.0, -0.0, 1.0, -1.0, 1.5, -2.25, 3.0, 0.1, 7.1, -3.3, 1e-30, np.nan, np.inf, -np.inf]
EXPRESSIONS = {
    "x + y": x + y, "x - y": x - y, "x * y": x * y, "x / y": x / y,
    "x * y + z": x * y + z,  # a fused multiply-add would round once, NumPy rounds twice
    "x ** 2": x ** 2, "x ** 0.5": x ** 0.5, "x ** -1": x ** -1, "x ** 1": x ** 1,
    "-x": -x, "+x": +x, "abs(x)": abs(x), "floor(x)": floor(x), "sqrt(x)": sqrt(x),
    "x < y": x < y, "x <= y": x <= y, "x == y": x == y, "x != y": x != y, "x > y": x > y, "x >= y": x >= y,
    "(x > 0) & (y < 0)": (x > 0) & (y < 0), "(x > 0) | ~(y > 0)": (x > 0) | ~(y > 0), "x & y": x & y,
    "x | y": x | y, "~x": ~x,
    "if_else(x > y, x, y)": if_else(x > y, x, y), "if_else(x > 0, x, 2)": if_else(x > 0, x, 2),
    "if_else(x > 0, x, nan)": if_else(x > 0, x, nan), "x * inf": x * float("inf"), "x - -inf": x - float("-inf"),
    "x - 273.15": x - 273.15, "x * 2": x * 2, "1 / x": 1 / x, "2.5 + x": 2.5 + x, "x + True": x + True,
    "x * y - x / 3 + 1": x * y - x / 3 + 1,
    "formula": formula(r"\sqrt{x^2 + y^2 + z^2}"),
    # NumPy's rules, step by step (milestone 18): floored division and remainder, ceil, sign
    "x // y": x // y, "x % y": x % y, "x // 3": x // 3, "x % -2.5": x % -2.5, "ceil(x)": ceil(x), "sign(x)": sign(x),
    # islands: NumPy's own values, which the kernel reads (milestone 18)
    "x ** 3": x ** 3, "exp(x)": exp(x), "log(x) + y": log(x) + y, "sin(x) * y": sin(x) * y,
    "exp(x + y) - 1": exp(x + y) - 1, "if_else(x > 0, log(x), y)": if_else(x > 0, log(x), y),
}
TYPES = [("float32", "float32"), ("float64", "float64"), ("float32", "float64"), ("float64", "float32"),
         ("int32", "int64"), ("int64", "float32"), ("int32", "float64"), ("uint8", "int8"), ("bool", "bool"),
         ("bool", "float32"), ("int16", "uint16")]


def values_of(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.array(FLOATS + [np.finfo(dtype).smallest_subnormal, np.finfo(dtype).max], dtype)
    if dtype.kind == "b":
        return np.array([False, True])
    info = np.iinfo(dtype)
    return np.array(sorted({0, 1, 2, 7, info.max, info.min, info.max - 1, *([-1, -9] if info.min < 0 else [200])}),
                    dtype)


def same(got, want):
    """The same type and bits, any NaN matching any NaN."""
    if got.dtype != want.dtype or got.shape != want.shape:
        return False
    if got.dtype.kind != "f":
        return got.tobytes() == want.tobytes()
    nan = np.isnan(want)
    return bool((np.isnan(got) == nan).all()) and got[~nan].tobytes() == want[~nan].tobytes()


def frame_of(first, second):
    xr = pytest.importorskip("xarray")
    a, b = values_of(first), values_of(second)
    left, right = np.repeat(a, len(b)), np.tile(b, len(a))  # every pair
    return tidy(xr.Dataset({"x": ("p", left), "y": ("p", right), "z": ("p", left[::-1].copy())}))


@pytest.mark.parametrize("first, second", TYPES)
def test_every_operation_gives_numpys_bits(first, second):
    frame = frame_of(first, second)
    context = Context(frame, verb="mutate", error=MutateError)
    compiler = Compiler("p")
    wanted, sources, names = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, expression in EXPRESSIONS.items():
            try:
                want = np.asarray(evaluate(expression, context).values)
            except Exception:  # NumPy does not take it for these types (~ of a float, a boolean minus a boolean)
                continue
            try:
                sources.append(compiler.compile(expression, context))
            except Unsupported:  # left to NumPy on purpose (an integer to the power 0.5)
                continue
            wanted.append(want)
            names.append(name)
        assert len(names) >= 10, names
        source = loop_source("kernel", compiler, sources)
        kernel = compile_source(source, "kernel")
        arrays = [np.asarray(read(value, context).values) for value, _ in compiler.inputs.values()]
        outputs = [np.empty(len(want), source.dtype) for want, source in zip(wanted, sources)]
        kernel(len(wanted[0]), 1, *arrays, *outputs)
    wrong = [name for name, got, want in zip(names, outputs, wanted) if not same(got, want)]
    assert not wrong, (first, second, wrong, source)


def test_a_grids_values_and_its_positions_mix_as_numpy_mixes_them():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(0)
    data = xr.Dataset({"t": (("time", "cell"), rng.normal(280, 30, (200, 7)).astype(np.float32)),
                       "h": ("time", rng.integers(0, 24, 200))})
    context = Context(tidy(data), verb="mutate", error=MutateError)
    expressions = [t - 273.15, (h >= 6) & (h <= 18), (t - 273.15) * h, if_else(h > 12, t, t * 2), sqrt(t) + h]
    compiler = Compiler("time")
    sources = [compiler.compile(expression, context) for expression in expressions]
    assert [source.cells for source in sources] == [True, False, True, True, True]
    kernel = compile_source(loop_source("kernel", compiler, sources), "kernel")
    arrays = [np.asarray(read(value, context).values) for value, _ in compiler.inputs.values()]
    outputs = [np.empty((200, 7) if source.cells else 200, source.dtype) for source in sources]
    kernel(200, 7, *arrays, *outputs)
    for expression, got in zip(expressions, outputs):
        assert same(got, np.asarray(evaluate(expression, context).values)), expression.text()


@pytest.mark.parametrize("expression, island", [
    (x ** 3, "x ** 3"), (exp(x) + y, "exp(x)"), (exp(x + y) * 2, "exp(x + y)"), (sin(x // y), "sin(x // y)")])
def test_what_does_not_agree_is_computed_by_numpy_as_an_island(expression, island):
    """NumPy's own routines, which a kernel cannot match on every machine: NumPy computes the part, its operands too,
    and the kernel reads it as an input."""
    context = Context(frame_of("float32", "float32"), verb="mutate", error=MutateError)
    compiler = Compiler("p")
    with warnings.catch_warnings():  # exp of float32's largest value overflows, as NumPy says
        warnings.simplefilter("ignore", RuntimeWarning)
        compiler.compile(expression, context)
    islands = [read.text() for read, _ in compiler.inputs.values() if isinstance(read, Island)]
    assert islands == [island], islands


def test_what_no_kernel_reads_stays_on_numpy():
    context = Context(frame_of("float32", "float32"), verb="mutate", error=MutateError)
    with pytest.raises(Unsupported, match="conj"):
        Compiler("p").compile(conj(x), context)


def test_an_integer_to_the_power_one_half_is_an_island():
    """NumPy takes the C library's pow for it, which need not be sqrt on every platform: NumPy computes it."""
    context = Context(frame_of("int32", "int32"), verb="mutate", error=MutateError)
    compiler = Compiler("p")
    assert compiler.compile(x ** 0.5, context).dtype == np.float64
    assert [read.text() for read, _ in compiler.inputs.values() if isinstance(read, Island)] == ["x ** 0.5"]
    assert Compiler("p").compile(x ** 2, context).dtype == np.int32
