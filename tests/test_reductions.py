"""The reduction registry (plan milestone 13): each reduction's partial, merge, and finish, keyed by its name.

Every registered reduction is run over blocks of awkward sizes, for each kind
of value it takes, with and without ``na_rm``, against the same summary in
memory: the same bits where blocks promise them (float32 sums and means,
integer sums, extremes, flags), close where they add in float64 (float64 and
integer means, spreads).
"""

import numpy as np
import pytest

from tidy3a import all, any, fields, group_by, max, mean, median, min, n, pull, quantile, std, sum, summarise, tidy, var
from tidy3a.chunked import COMBINABLE
from tidy3a.reductions import REGISTRY, Reduction

FUNCTIONS = {"sum": sum, "mean": mean, "min": min, "max": max, "std": std, "var": var, "any": any, "all": all,
             "median": median, "quantile": lambda x, na_rm=False: quantile(x, 0.3, na_rm=na_rm)}
EXACT = {("sum", "f"), ("mean", "f"), ("sum", "i"), ("min", "f"), ("max", "f"), ("min", "i"), ("max", "i"),
         ("any", "f"), ("all", "f"), ("any", "i"), ("all", "i"), ("median", "f"), ("median", "d"), ("median", "i"),
         ("quantile", "f"), ("quantile", "d"), ("quantile", "i")}


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


# ── Passes after the first (step 2) ─────────────────────────────────────────


def test_passes_are_shared_by_frames_run_in_rounds_and_ask_again(monkeypatch, tmp_path):
    """A max that asks for two more passes: the first counts each group's values equal to its max, and its resolve
    asks for a second that counts them again. Two frames' asks share each round, a round reads every block once,
    a resolve's further ask runs in the next round, and the budget packs rounds; the max is unchanged."""
    from labelarray import sums as labelarray_sums

    import tidy3a.reductions as reductions
    from tidy3a import infer, read_array, scan
    from tidy3a.reductions import Ask, Extreme

    log = []

    class Asking(Extreme):
        def finish(self, node, kind, held, counts, blocks):
            column, asks = super().finish(node, kind, held, counts, blocks)
            return column, [Ask(("count", column.copy()), len(column))]

        def again(self, node, ask, value, codes, groups):
            return np.bincount(codes, weights=value == ask.state[1][codes], minlength=groups)

        def resolve(self, node, ask, total, column):
            log.append((ask.state[0], total.tolist()))
            return column, ([Ask(("again", column.copy()), len(column))] if ask.state[0] == "count" else [])

    rng = np.random.default_rng(1)
    values = rng.normal(size=(1_000, 2)).astype("float32")
    values[:, 1] = rng.integers(0, 4, 1_000)
    path = tmp_path / "v.npy"
    np.save(path, values)
    v, g = fields("v", "g")
    options = dict(dims={"point": infer, "feature": 2}, labels={"feature": ["v", "g"]}, field_axis="feature")
    want = read_array(values, **options) >> group_by(g) >> summarise(top=max(v), low=max(-v))
    monkeypatch.setitem(reductions.REGISTRY, "max", Asking("max"))
    for budget, rounds in ((1_000, 2), (4, 4)):  # both frames' asks in one round, or one ask a round
        monkeypatch.setattr(labelarray_sums, "EXACT_PAIRS", budget)
        log.clear()
        frame = read_array(path, blocks=250, **options)
        source, opened = frame._blocks.source, []
        reopen = source.reopen
        source.reopen = lambda: opened.append(1) or reopen()
        top = frame >> group_by(g) >> summarise(top=max(v))
        low = frame >> group_by(g) >> summarise(low=max(-v))
        got_top, got_low = scan(top, low)
        assert len(opened) == 4 * (1 + rounds)  # the first pass, then each round reads the 4 blocks once
        assert [stage for stage, _ in log] == ["count", "count", "again", "again"]
        assert all(min(total) >= 1 for _, total in log)  # every group's max is among its values
        assert np.array_equal(got_top >> pull("top"), want >> pull("top"))
        assert np.array_equal(got_low >> pull("low"), want >> pull("low"))
