"""Reductions of each position first (plan milestone 15, step 3): where a position holds several values with one group,
each reduction reduces the position's values before the group's, without repeating the position's code for each.

Against the same summary in memory, and against the first pass with the codes repeated, which any reduction can fall
back to: the same bits for what adds exactly, close for what adds in float64.
"""

import numpy as np
import pytest
from test_reductions import EXACT, FUNCTIONS

from tidy3a import fields, group_by, n, pull, summarise, tidy
from tidy3a.expr import Reduce, Ref
from tidy3a.reductions import REGISTRY, Reduction

CELLS = 6


def grid(dtype, seed=0):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    count = 400
    if dtype.startswith("float"):
        values = (rng.normal(size=(count, CELLS)) * 100).astype(dtype)
        values[rng.random(values.shape) < 0.03] = np.nan
        values[:3] = np.nan  # positions with nothing present
        values[3:5] = values.dtype.type(2**24)  # ties a float32 estimate cannot settle alone
    else:
        values = rng.integers(-1_000, 1_000, (count, CELLS)).astype(dtype)
    return xr.Dataset({"v": (("point", "cell"), values), "g": ("point", rng.integers(0, 6, count))})


@pytest.mark.parametrize("rows", [1, 150])  # a block a position, and blocks of many
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_each_reduction_of_positions_first_gives_the_in_memory_result(name, dtype, na_rm, rows):
    data = grid(dtype)
    v, g = fields("v", "g")
    pipeline = group_by(g) >> summarise(points=n(), s=FUNCTIONS[name](v, na_rm=na_rm))
    want = tidy(data) >> pipeline
    got = tidy(data, blocks={"point": rows}) >> pipeline
    assert np.array_equal(got >> pull("points"), want >> pull("points"))
    a, b = np.asarray(got >> pull("s")), np.asarray(want >> pull("s"))
    assert a.dtype == b.dtype and a.shape == b.shape
    kind = "f" if dtype == "float32" else "i" if dtype == "int32" else "d"
    if (name, kind) in EXACT:
        assert a.tobytes() == b.tobytes(), (a, b)
    else:
        assert np.allclose(a, b, rtol=1e-6 if dtype == "float32" else 1e-12, equal_nan=True), (a, b)


@pytest.mark.backends("xarray", reason="a reduction's steps work on NumPy arrays, whatever the backend")
@pytest.mark.parametrize("op", sorted(set(REGISTRY) - {"n"}))
def test_positions_first_match_the_codes_repeated(op):
    rng = np.random.default_rng(1)
    rows, count = 2_000, 37
    codes = np.unique(rng.integers(0, count, rows), return_inverse=True)[1].reshape(-1)
    groups = int(codes.max()) + 1
    for dtype in ("float32", "float64", "int64", "bool"):
        if op in ("any", "all") and dtype not in ("bool", "float64") or dtype == "bool" and op not in ("any", "all"):
            continue
        values = (rng.normal(size=rows * CELLS) * 100).astype(dtype) if dtype != "bool" else rng.random(rows * CELLS) > .5
        if dtype.startswith("float"):
            values[rng.random(len(values)) < 0.05] = np.nan
        for na_rm in (False, True):
            node = Reduce(op, Ref("v"), na_rm=na_rm, **({"q": 0.4} if op == "quantile" else {}))
            entry = REGISTRY[op]
            kind, first = entry.partial_rows(node, values, codes, groups, CELLS)
            want_kind, want = Reduction.partial_rows(entry, node, values, codes, groups, CELLS)  # codes repeated
            assert kind == want_kind and first.dtype == want.dtype and first.shape == want.shape, (op, dtype)
            if kind.split(":")[0] in ("extreme", "flag", "flagged", "rank", "int"):
                assert first.tobytes() == want.tobytes(), (op, dtype, na_rm)
            elif kind == "estimate":  # its counts the same, its float64 totals within rounding (terms differ)
                assert np.array_equal(first[2:6], want[2:6]), (op, dtype)
                assert np.allclose(first[:2], want[:2], rtol=1e-12, atol=1e-9, equal_nan=True), (op, dtype)
            else:
                assert np.allclose(first, want, rtol=1e-12, atol=1e-6, equal_nan=True), (op, dtype, na_rm, kind)
