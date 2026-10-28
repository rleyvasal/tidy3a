"""A block's size for a memory budget (plan milestone 14): the estimate of what a run holds.

The estimate is a bound, read off the types and shapes the verbs give on the
block of no positions: these tests hold it above what ``tracemalloc``
measures, for each reduction's own steps and for whole blocks through the
verbs, and check that it reads nothing and gives the same answer each time.
"""

import builtins
import contextlib
import tracemalloc
import warnings
from concurrent.futures import Future

import numpy as np
import pytest

from tidy3a import (any, cells, chunked, fields, filter, formula, group_by, infer, max, mean, median, min, mutate, n,
                    pull, quantile, read_array, scan, select, std, sum, summarise, tidy, var, write)
from tidy3a.reductions import REGISTRY, Ask, Reduction, Working
from tidy3a.sizing import RESERVE, Cost, cost, describe

names = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring, distance = fields(*names, "distance")
t, hour, month, celsius, rain = fields("temperature", "hour", "month", "celsius", "rain")
DISTANCE = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}"))
DAY = filter((hour >= 6) & (hour <= 18)) >> mutate(celsius=t - 273.15) >> group_by(month)


def traced(work) -> int:
    """The most *work* allocates at once, by ``tracemalloc``."""
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        work()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


# ── Each reduction's own steps ──────────────────────────────────────────────

VALUES = 200_000
TYPES = {"sum": ["float32", "float64", "int64", "bool"], "mean": ["float32", "float64", "int64", "bool"],
         "min": ["float32", "float64", "int64"], "max": ["float32", "float64", "int64"],
         "std": ["float32", "float64", "int64"], "var": ["float32", "float64", "int64"],
         "any": ["bool", "float64"], "all": ["bool", "float64"],
         "median": ["float32", "float64", "int64"], "quantile": ["float32", "float64", "int64"]}


def node(op, na_rm=False):
    from tidy3a.expr import Reduce, Ref

    return Reduce(op, Ref("v"), na_rm=na_rm, **({"q": 0.9} if op == "quantile" else {}))


def values_of(dtype, rng):
    if dtype == "bool":
        return rng.normal(size=VALUES) > 0
    values = rng.normal(100, 10, VALUES).astype(dtype)
    if dtype.startswith("float"):
        values[::97] = np.nan
    return values


def codes_for(groups, rng):
    """Codes where every group occurs, as a block's are (``_partial`` factorizes them)."""
    codes = np.unique(rng.integers(0, groups, VALUES), return_inverse=True)[1].reshape(-1)
    return codes, int(codes.max()) + 1


def test_every_reduction_says_what_its_steps_allocate():
    assert set(TYPES) == set(REGISTRY) - {"n"}
    assert REGISTRY["n"].working(np.dtype("int64")) == Working(0)
    assert Reduction().working(np.dtype("float64")).first > 0  # a registered reduction's steps: a guess


@pytest.mark.backends("xarray", reason="a reduction's steps work on NumPy arrays, whatever the backend")
@pytest.mark.parametrize("op", list(TYPES))
def test_a_reductions_first_pass_allocates_no_more_than_it_says(op):
    rng = np.random.default_rng(0)
    entry = REGISTRY[op]
    for dtype in TYPES[op]:
        values = values_of(dtype, rng)
        working = entry.working(values.dtype)
        for groups in (10, 1_000):  # beyond these, what grows with the groups is the data's (``tidy3a.sizing``)
            codes, count = codes_for(groups, rng)
            used = traced(lambda: entry.partial(node(op), values, codes, count))
            assert used <= VALUES * working.first + working.fixed, (dtype, groups, used / VALUES)


@pytest.mark.backends("xarray", reason="a reduction's steps work on NumPy arrays, whatever the backend")
@pytest.mark.parametrize("op", ["median", "quantile", "sum", "mean"])
def test_a_reductions_later_passes_allocate_no_more_than_they_say(op):
    rng = np.random.default_rng(1)
    entry = REGISTRY[op]
    for dtype in ["float32", "float64", "int64"] if op in ("median", "quantile") else ["float32"]:
        values = values_of(dtype, rng)
        working = entry.working(values.dtype)
        for groups in (10, 1_000):
            codes, count = codes_for(groups, rng)
            reduce = node(op, na_rm=True)  # so every group, missing values and all, asks
            if op in ("sum", "mean"):  # every group asked for its exact sum, in batches as finish asks
                asks = [Ask(np.arange(count), count)]
            else:
                kind, piece = entry.partial(reduce, values, codes, count)
                _, asks = entry.finish(reduce, kind, piece, None, 1)
            while asks:  # every digit's pass
                ask = asks[0]
                used = traced(lambda: entry.again(reduce, ask, values, codes, count))
                assert used <= VALUES * working.later + working.fixed, (dtype, used / VALUES)
                if op in ("sum", "mean"):
                    break
                total = entry.again(reduce, ask, values, codes, count)
                _, asks = entry.resolve(reduce, ask, total, np.zeros(count))


# ── Whole blocks through the verbs ──────────────────────────────────────────


class Inline:
    """A pool that runs each block as it is handed in, so a block's work is all that allocates while it runs."""

    _max_workers = 1

    def __init__(self, threads):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, work, *args):
        future = Future()
        future.set_result(work(*args))
        return future


@pytest.fixture
def block_peaks(monkeypatch):
    """The peak of each block's work, every pass included, as the blocks run one at a time."""
    peaks = []
    naming = chunked._naming

    @contextlib.contextmanager
    def measured(blocks, span):
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        with naming(blocks, span):
            yield
        peaks.append(tracemalloc.get_traced_memory()[1] - start)

    monkeypatch.setattr(chunked, "_naming", measured)
    monkeypatch.setattr(chunked, "ThreadPoolExecutor", Inline)

    def run(frame):
        peaks.clear()
        tracemalloc.start()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if frame._summarised:
                    frame >> pull(next(iter(frame._plan.summaries)))
                else:
                    frame.collect()
        finally:
            tracemalloc.stop()
        return builtins.max(peaks)

    return run



def cloud_file(tmp_path):
    rng = np.random.default_rng(2)
    values = rng.normal(0, 30, (60_000, 5)).astype(np.float32)
    values[:, 4] = rng.integers(0, 32, len(values))
    path = tmp_path / "cloud.npy"
    np.save(path, values)
    return lambda rows: read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": names},
                                   field_axis="feature", blocks=rows)


def climate(rows):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    steps, shape = 300, (300, 9, 12)
    data = {name: (("time", "lat", "lon"), rng.normal(280, 10, shape).astype(np.float32))
            for name in ["temperature", "pressure", "rain"]}
    data["hour"] = ("time", np.arange(steps) % 24)
    data["month"] = ("time", np.arange(steps) // 30 % 12 + 1)
    dataset = xr.Dataset(data, coords={"time": np.arange(steps), "lat": np.arange(9.0), "lon": np.arange(12.0)})
    return tidy(dataset, blocks={"time": rows})


CLOUD = {
    "voxels": DISTANCE >> filter(distance <= 50) >> group_by(cells(x, 10), cells(y, 10), cells(z, 10), axis="voxel")
              >> summarise(points=n(), brightness=mean(intensity)),
    "rings": group_by(ring) >> summarise(points=n(), brightness=mean(intensity), spread=std(z), low=min(x)),
    "rows": DISTANCE >> filter(distance <= 50),
}
CLIMATE = {
    "daytime": DAY >> summarise(hours=n(), warmth=mean(celsius), hottest=max(celsius)),
    "median": DAY >> summarise(middle=median(celsius), high=quantile(celsius, 0.9)),
    "sum": group_by(month) >> summarise(total=sum(t), wet=any(rain > 290), spread=var(t)),
    "two keys": group_by(hour, month) >> summarise(k=n(), total=sum(t * 2.0)),
    "rows": filter(hour >= 6) >> mutate(celsius=t - 273.15),
}


def slope(measure, make, pipeline, small, large):
    """What a block's peak grows by per position, and its peak with no positions, from blocks of two sizes."""
    a, b = measure(make(small) >> pipeline), measure(make(large) >> pipeline)
    per_position = (b - a) / (large - small)
    return per_position, a - per_position * small


@pytest.mark.parametrize("name", list(CLOUD))
def test_the_estimate_bounds_a_blocks_peak_read_from_a_file(name, tmp_path, block_peaks):
    make = cloud_file(tmp_path)
    measured, base = slope(block_peaks, make, CLOUD[name], 10_000, 20_000)
    estimate = cost([make(10_000) >> CLOUD[name]])
    assert measured <= estimate.work <= 8 * measured, (measured, estimate)
    assert base <= estimate.fixed + 2**20, (base, estimate)


@pytest.mark.parametrize("name", list(CLIMATE))
def test_the_estimate_bounds_a_blocks_peak_from_a_dataset(name, block_peaks):
    measured, base = slope(block_peaks, climate, CLIMATE[name], 50, 100)
    estimate = cost([climate(50) >> CLIMATE[name]])
    assert measured <= estimate.work <= 8 * measured, (measured, estimate)
    assert base <= estimate.fixed + 2**20, (base, estimate)


def test_the_estimate_reads_nothing_and_is_the_same_each_time(tmp_path, monkeypatch):
    make = cloud_file(tmp_path)
    frame = make(10_000) >> CLOUD["voxels"]
    monkeypatch.setattr(frame._blocks.source, "block", lambda *args, **kwargs: pytest.fail("a block was read"))
    assert cost([frame]) == cost([frame]) == cost([make(5_000) >> CLOUD["voxels"]])


def test_a_dataset_block_holds_only_the_variables_a_run_reads():
    """Milestone 12: a dataset's block computes the variables used, so rows of two fields cost less than of all."""
    two = cost([climate(50) >> select(t, hour) >> filter(hour >= 6)])
    every = cost([climate(50) >> filter(hour >= 6)])
    assert two.work < every.work and two.held < every.held


def test_frames_scanned_together_cost_at_least_each_alone_and_at_most_both():
    data = climate(50)
    frames = [data >> CLIMATE["daytime"], data >> CLIMATE["median"]]
    together, alone = cost(frames), [cost([frame]) for frame in frames]
    assert builtins.max(each.work for each in alone) <= together.work <= builtins.sum(each.work for each in alone)
    assert together.held == builtins.sum(each.held for each in alone)
    assert scan(*frames)  # still runs



def test_a_budget_gives_the_most_positions_that_fit():
    estimate = Cost(work=1_000, held=100, fixed=10 * 2**20)
    for budget in (RESERVE + 50 * 2**20, 2**30, 4 * 2**30):
        for threads in (1, 4, 8):
            rows = estimate.rows(budget, threads)
            assert (rows == 0 or estimate.bound(rows, threads) <= budget) and budget < estimate.bound(rows + 1, threads)
    assert estimate.rows(RESERVE, 1) == 0  # nothing left once the reserve and the fixed part are set aside


# ── blocks="auto" and memory= ───────────────────────────────────────────────


def cloud_array(rows="auto", **options):
    values = np.random.default_rng(4).normal(0, 30, (30_000, 5)).astype(np.float32)
    return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature",
                      blocks=rows, **options)


SUMS = 40 * 2**20  # a float32 sum's fixed part (``Total.working``): a few MB beyond it leave room for some blocks
SMALL = RESERVE + 3 * 10**6  # room for a few thousand of the cloud's positions
SUMMED = SMALL + SUMS


def test_auto_takes_the_most_positions_the_estimate_fits_in_the_budget():
    frame = cloud_array(memory=SUMMED) >> CLOUD["rings"]
    rows = cost([frame]).rows(SUMMED, 1)
    assert 1 <= rows < 30_000
    assert frame._layout().rows == rows
    words = f"of {rows:,} positions on point, sized for memory={describe(SUMMED)}"
    assert words in repr(frame) and words in str(frame.explain())


def test_a_size_chosen_gives_what_the_same_size_given_does():
    pipeline = group_by(ring) >> summarise(m=mean(x * np.float64(1)), s=std(z))  # float64: bits that follow the size
    chosen = cloud_array(memory=SMALL) >> pipeline
    rows = chosen._layout().rows
    assert rows < 30_000
    given = cloud_array(rows) >> pipeline
    for name in ("m", "s"):
        assert np.asarray(chosen >> pull(name)).tobytes() == np.asarray(given >> pull(name)).tobytes()


def test_the_size_follows_the_verbs_that_run_and_the_threads():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(5)
    dataset = xr.Dataset({name: (("time", "cell"), rng.normal(size=(2_000, 500)).astype(np.float32))
                          for name in ("a", "b", "c", "d")} | {"g": ("time", np.arange(2_000) % 7)})
    a, g = fields("a", "g")
    frame = tidy(dataset, blocks={"time": "auto"}, memory=RESERVE + 10**7)
    one = frame >> group_by(g) >> summarise(total=max(a))  # one field read
    every = frame >> filter(g > 2)  # every field kept, in memory
    assert 2_000 > one._layout().rows > every._layout().rows
    assert frame._blocks.sized([one, every]).rows <= every._layout().rows  # a scan's blocks fit both
    threads = [(tidy(dataset, blocks={"time": "auto"}, memory=RESERVE + 2 * 10**7, threads=count) >> group_by(g)
                >> summarise(total=max(a)))._layout().rows for count in (1, 2, 4)]
    assert threads[0] > threads[1] > threads[2]


def test_auto_reads_nothing_to_choose_its_size(monkeypatch):
    frame = cloud_array(memory=SUMMED)
    monkeypatch.setattr(frame._blocks.source, "block", lambda *args, **kwargs: pytest.fail("a block was read"))
    summary = frame >> CLOUD["voxels"]
    assert f"sized for memory={describe(SUMMED)}" in repr(summary) and 0 < summary._layout().rows < 30_000


def test_memory_is_bytes_or_text_and_a_quarter_of_physical_memory_by_default(monkeypatch):
    from tidy3a import sizing

    assert sizing.budget("2GB") == 2 * 10**9 and sizing.budget("512 MiB") == 512 * 2**20
    assert sizing.budget(10**9) == sizing.budget("1gb") == sizing.budget(".001 TB") == 10**9
    monkeypatch.setattr(sizing, "physical_memory", lambda: 16 * 2**30)
    assert sizing.budget(None) == 4 * 2**30
    assert "sized for memory=4.29 GB" in repr(cloud_array())
    for wrong in ("2 GBs", "lots", "", "-1GB", 0, -5, True, [1]):
        with pytest.raises(Exception, match="memory= is a number of bytes"):
            sizing.budget(wrong)
    monkeypatch.setattr(sizing, "physical_memory", lambda: None)
    with pytest.raises(Exception, match="give the budget with memory="):
        cloud_array()


def test_memory_goes_with_auto_only():
    with pytest.raises(Exception, match='memory= is the budget blocks="auto" sizes blocks for'):
        cloud_array(1_000, memory="1GB")
    with pytest.raises(Exception, match='memory= is the budget blocks="auto" sizes blocks for'):
        read_array(np.zeros((10, 2)), dims=["a", "b"], memory="1GB")
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset({"v": ("time", np.arange(10.0))})
    with pytest.raises(Exception, match="memory= is the budget"):
        tidy(dataset, memory="1GB")
    with pytest.raises(Exception, match="memory= is the budget"):
        tidy(dataset, blocks={"time": 5}, memory="1GB")
    assert "1 of 10 positions on time, sized for memory=1 GB" in repr(tidy(dataset, blocks={"time": "auto"},
                                                                            memory="1GB"))
    with pytest.raises(Exception, match='"chunks", or "auto"'):
        tidy(dataset, blocks="auto")


def test_a_budget_no_block_fits_is_refused_with_the_sizes():
    with pytest.raises(Exception, match=r'blocks="auto" fits no block of point in memory=200 MB: one position takes '
                                        r"about .* bytes as these verbs run on 1 thread, beyond the 268 MB set aside"):
        cloud_array(memory="200MB")
    frame = cloud_array(memory=RESERVE + 5 * 10**7, threads=2)  # rows alone fit; two float32 means' working memory
    with pytest.raises(Exception, match="2 threads, .* give memory= more, fewer threads"):
        frame >> group_by(ring) >> summarise(m=mean(intensity))


def test_rows_written_with_auto_are_read_back_with_auto(tmp_path):
    written = cloud_array(memory=SMALL) >> filter(x > 0) >> write(tmp_path / "x.npy")
    assert written._blocks.rows == "auto" and written._blocks.memory == SMALL
    assert f"sized for memory={describe(SMALL)}" in repr(written)
