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

from tidy3a import (any, cells, chunked, exp, fields, filter, formula, full_join, group_by, if_else, infer, inner_join,
                    left_join, log, max, mean, median, min, mutate, n, nan, pull, quantile, read_array, scan, select,
                    std, sum, summarise, tidy, var, write)
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
    """The peak of each block's work, every pass included, as the blocks run one at a time, on NumPy: the estimate's
    per-position costs are NumPy's steps', a bound for a compiled kernel's, which writes no arrays a value."""
    monkeypatch.setenv("TIDY3A_COMPILED", "0")
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


@pytest.fixture
def small_slices(monkeypatch):
    """Later passes in slices of 4,096 values, so blocks of a few thousand values hold several, and the slices' fixed
    part does not hide what grows with the block."""
    from tidy3a import sizing

    monkeypatch.setattr(chunked, "SLICE", 4_096)
    monkeypatch.setattr(sizing, "SLICE", 4_096)


def bounded(measure, make, pipeline, small, large):
    """Each block's peak, at two sizes, within the estimate's bound for its size; and the estimate no more than eight
    times what the peak grows by per position."""
    estimate = cost([make(small) >> pipeline])
    peaks = [measure(make(rows) >> pipeline) for rows in (small, large)]
    for rows, peak in zip((small, large), peaks):
        assert peak <= rows * estimate.work + estimate.fixed + 2**20, (rows, peak, estimate)
    grows = (peaks[1] - peaks[0]) / (large - small)
    assert estimate.work <= 8 * grows, (grows, estimate)


@pytest.mark.parametrize("name", list(CLOUD))
def test_the_estimate_bounds_a_blocks_peak_read_from_a_file(name, tmp_path, block_peaks, small_slices):
    bounded(block_peaks, cloud_file(tmp_path), CLOUD[name], 10_000, 20_000)


@pytest.mark.parametrize("name", list(CLIMATE))
def test_the_estimate_bounds_a_blocks_peak_from_a_dataset(name, block_peaks, small_slices):
    bounded(block_peaks, climate, CLIMATE[name], 50, 100)


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
    assert Cost(work=1_000, held=10).per_position(2) == (2 * 1_000 + 4 * 10) * 3 // 2  # what the allocator keeps


# ── blocks="auto" and memory= ───────────────────────────────────────────────


def cloud_array(rows="auto", **options):
    values = np.random.default_rng(4).normal(0, 30, (30_000, 5)).astype(np.float32)
    return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature",
                      blocks=rows, **options)


FLOAT32 = REGISTRY["mean"].working(np.dtype("float32"))
SUMS = FLOAT32.fixed + chunked.SLICE * (FLOAT32.later + 8)  # a float32 sum's fixed part, with its passes' slices
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


# ── Whole storage chunks ────────────────────────────────────────────────────


def grid():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(6)
    return xr.Dataset({"a": (("time", "cell"), rng.normal(size=(1_000, 400)).astype(np.float32)),
                       "g": ("time", np.arange(1_000) % 7)})


def stored(tmp_path, size, **opening):
    """The grid written to Zarr in chunks of *size* along time, opened lazily (``chunks=None``) or with Dask."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    dataset = grid()
    path = tmp_path / "stored.zarr"
    dataset.chunk({"time": size}).to_zarr(path, consolidated=False)
    return dataset, xr.open_zarr(path, consolidated=False, **opening)


TOP = group_by(fields("g")) >> summarise(top=max(fields("a")))


def edges(spans):
    return {edge for span in spans for edge in span}


@pytest.mark.parametrize("opening", [{"chunks": None}, {}], ids=["lazy", "dask"])
def test_auto_blocks_are_whole_chunks_of_the_storage(tmp_path, opening):
    dataset, opened = stored(tmp_path, 70, **opening)
    frame = tidy(opened, blocks={"time": "auto"}, memory=RESERVE + 3 * 10**6) >> TOP
    layout = frame._layout()
    assert 70 <= layout.rows < 1_000
    assert edges(layout.every) <= set(range(0, 1_000, 70)) | {1_000}
    assert all(stop - start == layout.rows // 70 * 70 for start, stop in layout.every[:-1])
    assert "whole chunks of its storage" in repr(frame)
    assert np.array_equal(frame >> pull("top"), tidy(dataset) >> TOP >> pull("top"))


def test_irregular_dask_chunks_are_grouped_whole():
    pytest.importorskip("dask")
    months = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31) * 2 + (31, 28, 31, 30, 31, 30, 31, 31, 27)
    dataset = grid()
    opened = dataset.chunk({"time": months})
    frame = tidy(opened, blocks={"time": "auto"}, memory=RESERVE + 3 * 10**6) >> TOP
    bounds = set(np.cumsum((0, *months)).tolist())
    layout = frame._layout()
    assert edges(layout.every) <= bounds
    assert all(stop - start <= layout.rows for start, stop in layout.every)
    assert np.array_equal(frame >> pull("top"), tidy(dataset) >> TOP >> pull("top"))


def test_a_dask_chunk_larger_than_a_block_is_refused(tmp_path):
    _, opened = stored(tmp_path, 500)
    with pytest.raises(Exception, match=r"Dask chunks hold up to 500, each computed whole .* rechunk it smaller"):
        tidy(opened, blocks={"time": "auto"}, memory=RESERVE + 3 * 10**6) >> TOP


def test_a_files_chunk_larger_than_a_block_is_read_in_parts(tmp_path):
    dataset, opened = stored(tmp_path, 500, chunks=None)
    frame = tidy(opened, blocks={"time": "auto"}, memory=RESERVE + 3 * 10**6) >> TOP
    layout = frame._layout()
    assert layout.rows < 500 and 500 in edges(layout.every)
    assert "its storage chunks split to fit" in repr(frame)
    assert np.array_equal(frame >> pull("top"), tidy(dataset) >> TOP >> pull("top"))


def test_variables_stored_in_different_chunks_get_blocks_whole_for_each(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    dataset = xr.Dataset({"a": (("time", "cell"), np.ones((1_000, 400), np.float32)), "g": ("time", np.arange(1_000))})
    path = tmp_path / "mixed.zarr"
    dataset.to_zarr(path, consolidated=False, encoding={"a": {"chunks": (30, 400)}, "g": {"chunks": (20,)}})
    frame = tidy(xr.open_zarr(path, consolidated=False, chunks=None), blocks={"time": "auto"},
                 memory=RESERVE + 3 * 10**6) >> TOP
    assert edges(frame._layout().every) <= set(range(0, 1_000, 60)) | {1_000}  # whole chunks of both: 60


def test_a_size_given_or_data_in_memory_is_not_rounded(tmp_path):
    dataset, opened = stored(tmp_path, 70)
    assert (tidy(opened, blocks={"time": 100}) >> TOP)._layout().every[0] == (0, 100)
    in_memory = tidy(dataset, blocks={"time": "auto"}, memory=RESERVE + 3 * 10**6) >> TOP
    assert in_memory._layout().parts is None and "whole chunks" not in repr(in_memory)


# ── A scan's blocks ─────────────────────────────────────────────────────────


def loads_of(frame, monkeypatch):
    """The spans a frame's source is read in, in order."""
    spans = []
    source = frame._blocks.source
    block = source.block
    monkeypatch.setattr(source, "block", lambda start, stop, names=None: spans.append((start, stop))
                        or block(start, stop, names))
    return spans


def test_a_scan_reads_blocks_sized_for_all_its_frames_in_every_pass(monkeypatch):
    data = cloud_array(memory=SUMMED)
    voxels = data >> CLOUD["voxels"]
    middle = data >> group_by(ring) >> summarise(middle=median(z), high=quantile(x, 0.9))  # three more passes
    kept = data >> CLOUD["rows"]
    layout = data._blocks.sized([voxels, middle, kept])
    assert layout.rows <= builtins.min(frame._layout().rows for frame in (voxels, middle, kept))
    spans = loads_of(data, monkeypatch)
    results = scan(voxels, middle, kept)
    assert len(spans) % len(layout.every) == 0 and len(spans) >= 3 * len(layout.every)
    passes = [spans[start:start + len(layout.every)] for start in range(0, len(spans), len(layout.every))]
    assert all(each == layout.every for each in passes)
    alone = [cloud_array(memory=SUMMED) >> pipeline for pipeline in
             (CLOUD["voxels"], group_by(ring) >> summarise(middle=median(z), high=quantile(x, 0.9)), CLOUD["rows"])]
    for got, want in zip(results, alone):  # float32 means, medians, counts, and rows: the same bits as alone
        got, want = got.to_dict(), want.to_dict()
        assert got.keys() == want.keys()
        for name in got["data_vars"]:
            a, b = np.asarray(got["data_vars"][name]["data"]), np.asarray(want["data_vars"][name]["data"])
            assert (np.array_equal(a, b) if a.dtype == object else a.tobytes() == b.tobytes()), name


def test_a_scans_float64_means_are_those_of_its_size_given_by_hand():
    data = cloud_array(memory=SMALL)
    spread = data >> group_by(ring) >> summarise(m=mean(x * np.float64(1)), s=std(z))
    kept = data >> CLOUD["rows"]
    rows = data._blocks.sized([spread, kept]).rows
    got, _ = scan(spread, kept)
    given = cloud_array(rows) >> group_by(ring) >> summarise(m=mean(x * np.float64(1)), s=std(z))
    for name in ("m", "s"):
        assert np.asarray(got >> pull(name)).tobytes() == np.asarray(given >> pull(name)).tobytes()


def test_frames_a_scan_writes_count_toward_its_blocks(tmp_path, monkeypatch):
    data = cloud_array(memory=SUMMED)
    summary, written = data >> CLOUD["rings"], data >> DISTANCE >> filter(distance <= 50) >> select(distance)
    layout = data._blocks.sized([summary, written])
    assert layout.rows < summary._layout().rows
    spans = loads_of(data, monkeypatch)
    _, copy = scan(summary, writes=[(written, tmp_path / "near.npy")])
    assert spans[:len(layout.every)] == layout.every
    assert np.array_equal(copy.collect()["distance"].values, cloud_array(1_000) >> DISTANCE
                          >> filter(distance <= 50) >> pull("distance"))


@pytest.mark.parametrize("each", [1, 6])
def test_later_passes_in_slices_give_the_in_memory_bits(monkeypatch, each):
    """A later pass works on slices of a block (``chunked.SLICE``), their answers joined as blocks' are."""
    xr = pytest.importorskip("xarray")
    monkeypatch.setenv("TIDY3A_COMPILED", "0")  # the float32 tie's exact pass: compiled, its total is rounded exact
    monkeypatch.setattr(chunked, "SLICE", 7)
    sliced = []
    total = type(REGISTRY["sum"])
    again = total.again
    monkeypatch.setattr(total, "again", lambda self, node, ask, value, *rest: sliced.append(len(value))
                        or again(self, node, ask, value, *rest))
    rng = np.random.default_rng(7)
    values = (rng.normal(size=(600, each)) * 100).astype(np.float32)
    values[rng.random(values.shape) < 0.02] = np.nan
    groups = rng.integers(0, 5, 600)
    values[:6] = 0
    values[:6, 0] = np.array([2**24, 1, 1, 2**30, 1, -(2**30)], np.float32)  # a tie the bound cannot settle
    groups[:6] = 5
    data = xr.Dataset({"v": (("point", "cell"), values), "g": ("point", groups)})
    v, g = fields("v", "g")
    pipeline = group_by(g) >> summarise(total=sum(v, na_rm=True), middle=median(v, na_rm=True),
                                        high=quantile(v, 0.7, na_rm=True), mean=mean(v))
    want = tidy(data) >> pipeline
    got = tidy(data, blocks={"point": 50}) >> pipeline
    for name in ("total", "middle", "high", "mean"):
        assert np.asarray(got >> pull(name)).tobytes() == np.asarray(want >> pull(name)).tobytes(), name
    assert sliced and builtins.max(sliced) <= builtins.max(7, each)  # the exact pass ran, a slice at a time


@pytest.fixture
def compiled_peaks(block_peaks, monkeypatch):
    """``block_peaks`` with compiling on (milestone 18): a compiled frame's islands are arrays a block holds. A run
    first compiles the kernels, which the measured runs then find compiled."""
    pytest.importorskip("numba")
    monkeypatch.delenv("TIDY3A_COMPILED")

    def run(frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame._wrap(frame, frame._blocks, frame._steps, frame._plan, frame._reads).collect()  # compiled once
        return block_peaks(frame)

    return run


ISLANDS = {
    "rings with exp and log": DISTANCE >> mutate(w=exp(-distance / 20)) >> filter(distance <= 50) >> group_by(ring)
                              >> summarise(weighted=sum(fields("w") * intensity), glow=mean(log(abs(intensity) + 1))),
    "voxels with exp": DISTANCE >> filter(distance <= 50) >> group_by(cells(x, 10), cells(y, 10), axis="voxel")
                       >> summarise(points=n(), weight=sum(exp(-distance / 30))),
}
GRID_ISLANDS = {
    "vapour": DAY >> summarise(hours=n(), vapour=mean(6.112 * exp(17.67 * celsius / (celsius + 243.5)))),
}


@pytest.mark.parametrize("name", list(ISLANDS))
def test_the_estimate_bounds_a_compiled_blocks_peak_with_islands(name, tmp_path, compiled_peaks, small_slices):
    from tidy3a.fused import plan_for

    make = cloud_file(tmp_path)
    assert plan_for(make(10_000) >> ISLANDS[name]).islands
    bounded(compiled_peaks, make, ISLANDS[name], 10_000, 20_000)


@pytest.mark.parametrize("name", list(GRID_ISLANDS))
def test_the_estimate_bounds_a_compiled_grids_peak_with_islands(name, compiled_peaks, small_slices):
    from tidy3a.fused import plan_for

    assert plan_for(climate(50) >> GRID_ISLANDS[name]).islands
    bounded(compiled_peaks, climate, GRID_ISLANDS[name], 50, 100)


def joined(rows):
    """``climate`` read in blocks, with a mask over lat and lon and a value per time step to join (milestone 19)."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(4)
    frame = climate(rows)
    mask = xr.Dataset({"land": (("lat", "lon"), rng.random((9, 12)) > 0.5)},
                      coords={"lat": np.arange(9.0), "lon": np.arange(12.0)})
    quality = xr.Dataset({"quality": ("time", rng.random(330).astype(np.float32)),
                          "visits": ("time", rng.integers(0, 9, 330))}, coords={"time": np.arange(-30, 300)})
    return frame, mask, quality


JOINED = {
    "a mask, masked and averaged": lambda mask, quality: left_join(mask) >> mutate(lt=if_else(fields("land"), t, nan))
    >> group_by(month) >> summarise(m=mean(fields("lt"), na_rm=True)),
    "a value per time step, filtered on": lambda mask, quality: inner_join(quality)
    >> filter(fields("quality") > 0.3) >> group_by(month) >> summarise(m=mean(t)),
    "a full join along time, filtered on": lambda mask, quality: full_join(quality) >> filter(fields("quality") > 0.3)
    >> select("temperature", "visits"),
    "a full join along time, by month": lambda mask, quality: full_join(quality) >> group_by(month, dropna=False)
    >> summarise(k=n(), v=sum(fields("visits")), m=mean(t, na_rm=True)),
}


@pytest.mark.parametrize("name", list(JOINED))
def test_the_estimate_bounds_a_blocks_peak_with_joins(name, block_peaks, small_slices):
    def make(rows):
        frame, mask, quality = joined(rows)
        return frame

    _, mask, quality = joined(50)
    bounded(block_peaks, make, JOINED[name](mask, quality), 50, 100)


def test_what_a_join_brings_over_the_other_axes_is_a_fixed_cost_per_block():
    """A mask or weights over lat and lon is the same size whatever the block's: counted once a block, not a time
    step (milestone 19)."""
    frame, mask, quality = joined(50)
    weights = {dtype: mask.assign(weight=(("lat", "lon"), np.ones((9, 12), dtype))).drop_vars("land")
               for dtype in ("float32", "float64")}
    small, large = (cost([frame >> left_join(weights[dtype]) >> CLIMATE["sum"]]) for dtype in ("float32", "float64"))
    assert large.fixed - small.fixed == 9 * 12 * 4 and large.work == small.work
