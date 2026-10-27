"""The reduction registry (plan milestone 13): each reduction's partial, merge, and finish, keyed by its name.

Every registered reduction is run over blocks of awkward sizes, for each kind
of value it takes, with and without ``na_rm``, against the same summary in
memory: the same bits where blocks promise them (float32 sums and means,
integer sums, extremes, flags), close where they add in float64 (float64 and
integer means, spreads).
"""

import numpy as np
import pytest

from tidy3a import all, any, fields, group_by, max, mean, min, n, pull, std, sum, summarise, tidy, var
from tidy3a.chunked import COMBINABLE
from tidy3a.reductions import REGISTRY, Reduction

FUNCTIONS = {"sum": sum, "mean": mean, "min": min, "max": max, "std": std, "var": var, "any": any, "all": all}
EXACT = {("sum", "f"), ("mean", "f"), ("sum", "i"), ("min", "f"), ("max", "f"), ("min", "i"), ("max", "i"),
         ("any", "f"), ("all", "f"), ("any", "i"), ("all", "i")}


def test_the_registry_is_what_blocks_can_answer():
    assert COMBINABLE == tuple(REGISTRY)
    assert set(REGISTRY) == {"n", *FUNCTIONS}
    assert all(isinstance(entry, Reduction) and entry.name == name for name, entry in REGISTRY.items())
    assert [name for name, entry in REGISTRY.items() if entry.counted] == ["n"]


def source(dtype, seed=0):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    count = 2_000
    if dtype.startswith("float"):
        values = (rng.normal(size=count) * 100).astype(dtype)
        values[rng.choice(count, 40, replace=False)] = np.nan
        values[:5] = values.dtype.type(2**24)  # ties a float32 estimate cannot settle alone
    else:
        values = rng.integers(-1_000, 1_000, count).astype(dtype)
    group = rng.integers(0, 6, count).astype("int64")
    return xr.Dataset({"v": ("point", values), "g": ("point", group)})


@pytest.mark.parametrize("rows", [1, 7, 333])
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_each_registered_reduction_on_blocks_gives_the_in_memory_result(name, dtype, na_rm, rows):
    data = source(dtype)
    v, g = fields("v", "g")
    pipeline = group_by(g) >> summarise(points=n(), s=FUNCTIONS[name](v, na_rm=na_rm))
    want = tidy(data) >> pipeline
    got = tidy(data, blocks={"point": rows}) >> pipeline
    assert np.array_equal(got >> pull("points"), want >> pull("points"))
    a, b = np.asarray(got >> pull("s")), np.asarray(want >> pull("s"))
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype)
    kind = "f" if dtype == "float32" else "i" if dtype == "int32" else "d"
    if (name, kind) in EXACT:
        assert a.tobytes() == b.tobytes(), (a, b)
    else:
        tolerance = 1e-6 if a.dtype == np.float32 else 1e-12  # a float32 result is one rounding away at most
        assert np.allclose(a, b, rtol=tolerance, atol=0, equal_nan=True), (a, b)
