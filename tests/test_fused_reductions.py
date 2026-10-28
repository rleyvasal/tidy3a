"""Reductions as accumulators in a kernel (plan milestone 16, step 4): the pieces each registry entry's first pass
gives, from one pass over a block's values.

Against ``partial_rows`` on the same values and codes, for one value a position and for six (a grid's): the same
bits for counts, integers, extremes, flags, and the median's counts; float64 sums and spreads within rounding (the
same bits for one value a position, added in the same order); and for float32 sums the same counts and, once
labelarray settles them and its exact pass runs where it must, the same float32 results.
"""

import warnings

import numpy as np
import pytest
from labelarray import sums

from tidy3a import fields, tidy
from tidy3a.errors import SummariseError
from tidy3a.expr import Context, Reduce, Ref, evaluate
from tidy3a.fused.expressions import Compiler
from tidy3a.reductions import REGISTRY

pytest.importorskip("numba")
from tidy3a.fused.reductions import Slot, compile_summary, counted_values, summary_source  # noqa: E402

xr = pytest.importorskip("xarray")
pytestmark = pytest.mark.backends("numpy", reason="the kernel sees plain arrays whatever the backend; one backend "
                                  "halves the compiling")
TYPES = {"sum": ["float32", "float64", "int32", "bool"], "mean": ["float32", "float64", "int64", "bool"],
         "min": ["float32", "float64", "int16"], "max": ["float32", "float64", "uint8"],
         "any": ["bool", "float64", "int32"], "all": ["bool", "float32", "int32"],
         "std": ["float32", "float64", "int32"], "var": ["float64"], "median": ["float32", "int64"],
         "quantile": ["float64"]}
EXACT = {"extreme", "flag", "flagged", "int"}


def values_of(dtype, positions, cells, rng):
    shape = (positions, cells) if cells > 1 else positions
    if dtype == "bool":
        return rng.random(shape) > 0.5
    if np.dtype(dtype).kind in "iu":
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -500), min(info.max, 500), shape).astype(dtype)
    values = (rng.normal(size=shape) * 100).astype(dtype)
    flat = values.reshape(-1)
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.finfo(dtype).max, np.finfo(dtype).smallest_subnormal,
                         2.0**24, 1.0, 1.0], dtype)
    flat[rng.choice(len(flat), 60, replace=False)] = rng.choice(specials, 60)
    return values


def run(op, dtype, na_rm, cells, seed):
    rng = np.random.default_rng(seed)
    positions, groups = 1_500, 23
    values = values_of(dtype, positions, cells, rng)
    codes = np.unique(rng.integers(0, groups, positions), return_inverse=True)[1].reshape(-1)
    count = int(codes.max()) + 1
    dims = ("p", "c") if cells > 1 else ("p",)
    frame = tidy(xr.Dataset({"v": (dims, values)}))
    context = Context(frame, verb="summarise", error=SummariseError, grouped=False)
    node = Reduce(op, Ref("v"), na_rm=na_rm, **({"q": 0.3} if op == "quantile" else {}))
    compiler = Compiler("p")
    slot = Slot(0, node, compiler.compile(node.arg, context))
    arrays = slot.arrays(count)
    kernel = compile_summary(summary_source("summary", list(compiler.inputs), [slot]), "summary")
    counts = np.zeros(count, np.int64)
    inputs = [np.asarray(evaluate(ref, context).values) for ref, _ in compiler.inputs.values()]
    kernel(positions, cells, codes.astype(np.intp), *inputs, *arrays.values(), counts)
    got = slot.pieces(arrays, counted_values(counts, slot, cells))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        want = REGISTRY[op].partial_rows(node, values.reshape(-1), codes, count, cells)
    return got, want, values, codes, count


@pytest.mark.parametrize("cells", [1, 6], ids=["a value a position", "six a position"])
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("op", list(TYPES))
def test_each_slot_gives_its_reductions_pieces(op, na_rm, cells):
    for dtype in TYPES[op]:
        (kind, got), (want_kind, want), values, codes, count = run(op, dtype, na_rm, cells, seed=len(op) + cells)
        assert kind == want_kind and got.shape == want.shape and got.dtype == want.dtype, (op, dtype, kind, want_kind)
        base = kind.split(":")[0]
        if base in EXACT or base == "rank":
            assert got.tobytes() == want.tobytes(), (op, dtype)
        elif base == "estimate":
            assert np.array_equal(got[2:6], want[2:6]), (op, dtype)  # present, values, +inf, -inf
            final = [sums.from_chunks(pieces, 1, op, lambda chosen: sums.chunk_exact(
                values if cells > 1 else values, (0, 1) if cells > 1 else (0,), chosen, codes=codes, groups=count),
                na_rm=na_rm) for pieces in (got, want)]
            assert final[0].tobytes() == final[1].tobytes() or np.array_equal(final[0], final[1], equal_nan=True), \
                (op, dtype)
            unsure = [int(sums.settle(pieces, 1, op, na_rm)[1].sum()) for pieces in (got, want)]
            assert unsure[0] <= unsure[1], unsure  # the compensated bound settles as many groups, or more
        elif cells == 1 and base == "float":  # float64 sums in position order, as np.bincount adds them
            assert np.array_equal(got, want, equal_nan=True), (op, dtype)
        elif base == "moments":  # the finished spreads: near float64's largest value the paths' means may overflow
            # at different steps, as their sums go in different orders, and the squares overflow either way
            node = Reduce(op, Ref("v"), na_rm=na_rm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                finished = [REGISTRY[op].finish(node, kind, pieces, None, 1)[0] for pieces in (got, want)]
            assert np.allclose(finished[0], finished[1], rtol=1e-9, equal_nan=True), (op, dtype)
            assert (np.isinf(finished[0]) == np.isinf(finished[1])).all(), (op, dtype)
        else:
            assert np.allclose(got, want, rtol=1e-12, atol=1e-9, equal_nan=True), (op, dtype, kind)


# Whether NumPy orders -0.0 below +0.0 in minimum and maximum, as IEEE 754-2019 does, whatever comes first: it does
# on ARM; on x86 a tie of zeros gives the operand its loops happen to meet second, so the sign follows the order
# (and the vectorized loops) in which values are compared, which a kernel adding values in order cannot follow.
NUMPY_ORDERS_ZEROS = bool(np.signbit(np.minimum(0.0, -0.0)) and np.signbit(np.minimum(-0.0, 0.0))
                          and not np.signbit(np.maximum(-0.0, 0.0)) and not np.signbit(np.maximum(0.0, -0.0)))


def test_extremes_order_the_zeros_and_nan_as_numpy_does():
    """-0.0 below +0.0 whatever comes first; a NaN wins without na_rm and is passed over with it. The zeros' signs
    are NumPy's where NumPy orders them; elsewhere the kernel's own order, the values and NaN NumPy's."""
    data = np.array([0.0, -0.0, 0.0, np.nan, 1.0, -0.0, 0.0, np.nan, np.nan], np.float64)
    codes = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
    frame = tidy(xr.Dataset({"v": ("p", data)}))
    context = Context(frame, verb="summarise", error=SummariseError, grouped=False)
    for op in ("min", "max"):
        for na_rm in (False, True):
            node = Reduce(op, Ref("v"), na_rm=na_rm)
            compiler = Compiler("p")
            slot = Slot(0, node, compiler.compile(node.arg, context))
            arrays = slot.arrays(4)
            kernel = compile_summary(summary_source("summary", list(compiler.inputs), [slot]), "summary")
            kernel(len(data), 1, codes.astype(np.intp), data, *arrays.values(), np.zeros(4, np.int64))
            got = slot.pieces(arrays, np.ones(4, np.int64))[1]
            want = REGISTRY[op].partial_rows(node, data, codes, 4, 1)[1]
            assert np.array_equal(got, want, equal_nan=True), (op, na_rm)
            if NUMPY_ORDERS_ZEROS:
                assert got.tobytes() == want.tobytes() or (np.signbit(got) == np.signbit(want)).all(), (op, na_rm)
            else:  # groups 0 and 2 tie a -0.0 with +0.0: min gives -0.0, max +0.0
                assert np.signbit(got[[0, 2]]).tolist() == [op == "min"] * 2, (op, na_rm)
