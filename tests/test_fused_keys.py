"""Keys as dense codes (plan milestone 16, step 3): the groups a kernel places a block's positions in.

Against ``resolve_value_key`` on the same data: the same labels (their types too), the same code for every
position, and the same positions dropped for a missing key; or the block falls back to NumPy where agreement is
not established (a float key's value that is not whole, a negative zero, an infinity, a value outside every bin,
a step too far from the origin).
"""

import warnings

import numpy as np
import pytest

from tidy3a import bins, cells, fields, rounded, tidy
from tidy3a.errors import GroupError
from tidy3a.expr import Context, ValueKey, evaluate, resolve_value_key
from tidy3a.fused.expressions import Compiler, Unsupported

pytest.importorskip("numba")
from tidy3a.fused.keys import compile_keys, groups, kernel_source, key_for  # noqa: E402

x = fields("x")
xr = pytest.importorskip("xarray")
pytestmark = pytest.mark.backends("numpy", reason="a key's placement sees plain arrays whatever the backend, and "
                                  "the suite holds the backends' grouping alike; one backend halves the compiling")


def place(key, data):
    """The compiled placement's groups for *data* (a variable "x" along "p"), or None when the block falls back."""
    frame = tidy(xr.Dataset({"x": ("p", data)}))
    context = Context(frame, verb="group_by", error=GroupError, grouped=False)
    compiler = Compiler("p")
    placed = key_for(key, context, compiler)
    kernel = compile_keys(kernel_source("place", [placed], compiler), "place")
    arrays = [np.asarray(evaluate(ref, context).values) for ref, _ in compiler.inputs.values()]
    size = len(data)
    ks, missing, ranges = np.empty(size, np.int64), np.zeros(size, bool), np.zeros((1, 3), np.int64)
    edges = [placed.constants["edges"]] if placed.how == "bins" else []
    if kernel(size, *arrays, *edges, ks, missing, ranges) != 0:
        return None, frame
    return groups(placed, ks, missing, *ranges[0], placed.value.dtype), frame


def same_labels(a, b):
    return len(a) == len(b) and all(type(p) is type(q) and (p == q or (p != p and q != q)) for p, q in zip(a, b))


def agrees(key, data):
    found, frame = place(key, data)
    assert found is not None, "fell back"
    labels, codes, dropped = found
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        want = resolve_value_key(key, frame, [])
    assert same_labels(labels, list(want.labels)), (labels[:6], list(want.labels)[:6])
    assert np.array_equal(codes, np.asarray(want.codes)) and dropped == want.dropped


rng = np.random.default_rng(3)


def floats(dtype, size=3_000, scale=30.0):
    values = (rng.normal(size=size) * scale).astype(dtype)
    values[:200] = (np.arange(-100, 100) * 0.1).astype(dtype)  # on and beside edges of tenths
    values[200:260] = (np.arange(-30, 30) * 0.5).astype(dtype)  # halves, which rounded() rounds up
    values[rng.choice(size, 40, replace=False)] = np.nan
    return values


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("make", [lambda: cells(x, 1), lambda: cells(x, 0.1), lambda: cells(x, 2.5, origin=-1),
                                  lambda: cells(x, 0.3, origin=0.05), lambda: rounded(x, 0.5), lambda: rounded(x, 0.1),
                                  lambda: rounded(x, 3, origin=1),
                                  lambda: bins(x, [-200, -10, -0.1, 0, 0.1, 10, 200]),
                                  lambda: cells(x * 2 + 1, 0.5)],
                         ids=["cells 1", "cells 0.1", "cells 2.5 from -1", "cells 0.3 from 0.05", "rounded 0.5",
                              "rounded 0.1", "rounded 3 from 1", "bins", "cells of an expression"])
@pytest.mark.parametrize("dropna", [True, False])
def test_float_keys_place_values_as_labelarray_does(make, dtype, dropna):
    agrees(make().with_dropna(dropna), floats(dtype))


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "uint16"])
@pytest.mark.parametrize("make", [lambda: ValueKey("x", x), lambda: cells(x, 3), lambda: cells(x, 0.5),
                                  lambda: rounded(x, 2), lambda: rounded(x, 2.5, origin=1),
                                  lambda: bins(x, [-100, -1, 0, 1.5, 100])],
                         ids=["value", "cells 3", "cells 0.5", "rounded 2", "rounded 2.5 from 1", "bins"])
def test_integer_keys_place_values_as_labelarray_does(make, dtype):
    info = np.iinfo(dtype)
    values = rng.integers(max(info.min, -100), min(info.max, 100), 2_000).astype(dtype)
    agrees(make(), values)


def test_whole_floats_booleans_and_wide_integers_are_value_keys():
    whole = np.floor(rng.normal(size=2_000) * 40).astype(np.float32)
    whole[::37] = np.nan
    agrees(ValueKey("x", x), whole)
    agrees(ValueKey("x", x).with_dropna(False), whole)
    agrees(ValueKey("x", x), rng.random(500) > 0.5)
    agrees(ValueKey("x", x), rng.integers(-2**62, 2**62, 500))


@pytest.mark.parametrize("key, data", [
    (ValueKey("x", x), np.array([1.0, 2.5, 3.0])),  # a float key's value that is not whole
    (ValueKey("x", x), np.array([1.0, -0.0, 3.0])),  # a negative zero, which NumPy may label as either zero
    (ValueKey("x", x), np.array([1.0, np.inf])),
    (cells(x, 1), np.array([1.0, np.inf])),  # NumPy: in no cell, an error
    (bins(x, [0, 1, 2]), np.array([0.5, 2.5])),  # NumPy: outside every bin, an error
    (cells(x, 1e-12), np.array([1.0, 1e6])),  # NumPy: too far from the origin for its steps, an error
], ids=["not whole", "negative zero", "infinite value", "infinite in cells", "outside the bins", "too far out"])
def test_a_block_falls_back_where_agreement_is_not_established(key, data):
    assert place(key, data)[0] is None


def test_unsigned_64_bit_and_per_value_keys_run_on_numpy():
    frame = tidy(xr.Dataset({"x": ("p", np.arange(5, dtype=np.uint64)), "t": (("p", "c"), np.zeros((5, 2)))}))
    context = Context(frame, verb="group_by", error=GroupError, grouped=False)
    with pytest.raises(Unsupported, match="unsigned 64-bit"):
        key_for(ValueKey("x", x), context, Compiler("p"))
    with pytest.raises(Unsupported, match="varies within a position"):
        key_for(cells(fields("t"), 1), context, Compiler("p"))
