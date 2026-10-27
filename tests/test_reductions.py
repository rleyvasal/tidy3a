"""The reduction registry (plan milestone 13): each reduction's partial, merge, and finish, keyed by its name.

Every registered reduction is run over blocks of awkward sizes, for each kind
of value it takes, with and without ``na_rm``, against the same summary in
memory: the same bits where blocks promise them (float32 sums and means,
integer sums, extremes, flags), close where they add in float64 (float64 and
integer means, spreads).
"""

import numpy as np
import pytest

from tidy3a import (all, any, bins, fields, group_by, max, mean, median, min, n, pull, quantile, scan, std, sum,
                    summarise, tidy, var)
from tidy3a.reductions import REGISTRY, Reduction

FUNCTIONS = {"sum": sum, "mean": mean, "min": min, "max": max, "std": std, "var": var, "any": any, "all": all,
             "median": median, "quantile": lambda x, na_rm=False: quantile(x, 0.3, na_rm=na_rm)}
EXACT = {("sum", "f"), ("mean", "f"), ("sum", "i"), ("min", "f"), ("max", "f"), ("min", "i"), ("max", "i"),
         ("any", "f"), ("all", "f"), ("any", "i"), ("all", "i"), ("median", "f"), ("median", "d"), ("median", "i"),
         ("quantile", "f"), ("quantile", "d"), ("quantile", "i")}


def test_the_registry_is_what_blocks_can_answer():
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


# ── register_reduction (step 4) ─────────────────────────────────────────────


@pytest.fixture
def registered():
    """Names registered in a test, removed after it: registration is global."""
    import tidy3a.reductions as reductions

    names = []
    yield names
    for name in names:
        reductions.REGISTRY.pop(name, None)
        reductions.EAGER.pop(name, None)


def _geometric_mean(registered):
    from tidy3a import register_reduction

    def eager(values, axes, na_rm):
        logs = np.log(values)
        return np.exp(np.nanmean(logs, axis=axes) if na_rm else np.mean(logs, axis=axes))

    def partial(values, codes, groups, na_rm):
        logs = np.log(values.astype(np.float64))
        missing = np.isnan(logs)
        return np.stack([np.bincount(codes, weights=np.where(missing, 0.0, logs) if na_rm else logs, minlength=groups),
                         np.bincount(codes, weights=~missing if na_rm else None, minlength=groups)])

    registered.append("geomean")
    return register_reduction("geomean", eager, partial=partial, merge=lambda held, piece: held + piece,
                              finish=lambda held, counts: np.exp(held[0] / held[1]))


def _value_range(registered):
    from tidy3a import register_reduction

    def eager(values, axes, na_rm):  # given no values too: an empty group, and the empty block blocks check on
        if any(values.shape[axis] == 0 for axis in axes):
            return np.full([size for axis, size in enumerate(values.shape) if axis not in axes], np.nan)
        return np.max(values, axis=axes) - np.min(values, axis=axes)

    def partial(values, codes, groups, na_rm):
        low, high = np.full(groups, np.inf), np.full(groups, -np.inf)
        np.minimum.at(low, codes, values)
        np.maximum.at(high, codes, values)
        return np.stack([low, high])

    registered.append("value_range")
    return register_reduction("value_range", eager, partial=partial,
                              merge=lambda held, piece: np.stack([np.minimum(held[0], piece[0]),
                                                                  np.maximum(held[1], piece[1])]),
                              finish=lambda held, counts: held[1] - held[0],
                              fill=np.array([[np.inf], [-np.inf]]))  # each row's own fill


def positive(seed=5, count=3_000):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.5, 9.0, count)
    values[:4] = np.nan
    return xr.Dataset({"v": ("point", values), "g": ("point", rng.integers(0, 5, count).astype(float))})


@pytest.mark.parametrize("backend", ["numpy", "xarray"])
def test_a_registered_reduction_runs_in_memory_as_its_function_says(registered, backend):
    geomean = _geometric_mean(registered)
    data = positive()
    v, g = fields("v", "g")
    values, group = data["v"].values, data["g"].values
    for na_rm in (False, True):
        got = tidy(data, backend=backend) >> group_by(g) >> summarise(s=geomean(v, na_rm=na_rm)) >> pull("s")
        reduce = np.nanmean if na_rm else np.mean
        want = np.array([np.exp(reduce(np.log(values[group == code]))) for code in range(5)])
        assert got.tobytes() == want.tobytes()
        whole = tidy(data, backend=backend) >> summarise(s=geomean(v, na_rm=na_rm)) >> pull("s")
        assert np.asarray(whole).tobytes() == np.asarray(np.exp(reduce(np.log(values)))).tobytes()
    assert geomean(v, na_rm=True).text() == "geomean(v, na_rm=True)"


@pytest.mark.parametrize("rows", [1, 97, 3_000])
def test_registered_reductions_run_on_blocks(registered, rows):
    """A geometric mean through sums of logs (float64, close), and a range whose two rows each have their own fill
    (the same bits); grouped, by bins with an empty bin, and whole."""
    geomean, value_range = _geometric_mean(registered), _value_range(registered)
    data = positive()
    data["v"].values[:4] = 1.0  # the range has no na_rm rule; give it values
    v, g = fields("v", "g")
    for pipeline in (group_by(g) >> summarise(points=n(), geo=geomean(v), spread=value_range(v)),
                     group_by(bins(g, [0, 2, 2.5, 5])) >> summarise(geo=geomean(v), spread=value_range(v)),
                     summarise(geo=geomean(v), spread=value_range(v))):
        want = tidy(data) >> pipeline
        got = tidy(data, blocks={"point": rows}) >> pipeline
        assert np.allclose(got >> pull("geo"), want >> pull("geo"), rtol=1e-12, equal_nan=True)
        spread_got, spread_want = np.asarray(got >> pull("spread")), np.asarray(want >> pull("spread"))
        empty = np.isnan(spread_want)  # an empty bin: NaN in memory, inf - inf (NaN) from the block steps
        assert np.isnan(spread_got[empty]).all() and spread_got[~empty].tobytes() == spread_want[~empty].tobytes()


def test_a_registered_reduction_without_block_steps_is_refused_on_blocks(registered):
    from tidy3a import BlockError, register_reduction

    registered.append("middle_third")
    middle_third = register_reduction("middle_third", lambda values, axes, na_rm: np.percentile(values, 50, axis=axes))
    data = positive()
    data["v"].values[:4] = 1.0
    v = fields("v")
    assert (tidy(data) >> summarise(s=middle_third(v)) >> pull("s")) == np.percentile(data["v"].values, 50)
    with pytest.raises(BlockError, match=r"middle_third\(\) cannot be combined across blocks.*register_reduction"):
        tidy(data, blocks={"point": 100}) >> summarise(s=middle_third(v))


def test_registration_is_checked(registered):
    from tidy3a import TidyError, register_reduction

    with pytest.raises(TidyError, match="mean is a built-in reduction"):
        register_reduction("mean", lambda values, axes, na_rm: values)
    with pytest.raises(TidyError, match="a Python identifier"):
        register_reduction("not a name", lambda values, axes, na_rm: values)
    with pytest.raises(TidyError, match="partial, merge, and finish, all three, or none"):
        register_reduction("half", lambda values, axes, na_rm: values, partial=lambda *args: 0)
    registered.append("shapeless")
    shapeless = register_reduction("shapeless", lambda values, axes, na_rm: values)  # reduces nothing
    with pytest.raises(Exception, match="gave values of shape .* where the reduction keeps"):
        tidy(positive()) >> summarise(s=shapeless(fields("v")))


def test_frames_with_a_registered_reduction_share_a_scan(registered):
    geomean = _geometric_mean(registered)
    data = positive()
    v, g = fields("v", "g")
    frame = tidy(data, blocks={"point": 600})
    loads = []
    load = frame._blocks.source.block
    frame._blocks.source.block = lambda start, stop, names=None: loads.append(1) or load(start, stop, names)
    first = frame >> group_by(g) >> summarise(s=geomean(v, na_rm=True))
    second = frame >> summarise(s=geomean(v, na_rm=True))
    got_first, got_second = scan(first, second)
    assert len(loads) == 5  # one pass, for both
    assert np.allclose(got_first >> pull("s"), tidy(data) >> group_by(g) >> summarise(s=geomean(v, na_rm=True))
                       >> pull("s"), rtol=1e-12)
