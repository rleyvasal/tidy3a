"""Float32 sums and means on Dask-backed data: the bits of the in-memory kernel, for any chunking.

Each chunk's estimate is one task, the pieces add across chunks in Dask's
order, and one last task settles the results; the groups the error bound
cannot settle, if any, get a second pass over the chunks for their exact
sums. The results stay lazy until ``collect()``. The reference is the
in-memory result of the same verbs, itself checked against exact rational
arithmetic in ``test_float32_sums.py``, and that arithmetic where it is cheap.
"""

import numpy as np
import pytest
import xarray as xr

da = pytest.importorskip("dask.array")
from dask import config, delayed
from dask.callbacks import Callback

from test_float32_sums import F32, corpus, expected, same_bits
from tidy3a import col, group_by, mean, mutate, sum, summarise, tidy


def cuts(n, chunks, seed):
    """*chunks* uneven block lengths adding to *n*, some possibly empty."""
    points = np.sort(np.random.default_rng(seed).integers(0, n + 1, chunks - 1))
    return tuple(np.diff([0, *points.tolist(), n]).tolist())


CHUNKINGS = ["whole", 97, 1000, "uneven"]


def chunked(values, chunking, axis=0):
    chunks = {"whole": values.shape[axis], "uneven": cuts(values.shape[axis], 13, seed=axis)}.get(chunking, chunking)
    full = list(values.shape)
    full[axis] = chunks
    return da.from_array(values, chunks=tuple(full))


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("na_rm", [False, True])
def test_grouped_sums_on_dask_give_the_in_memory_bits(chunking, na_rm):
    values, keys, groups = corpus()
    lazy = xr.Dataset({"v": ("point", chunked(values, chunking)), "key": ("point", keys)})
    loaded = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    with np.errstate(all="ignore"):
        results = [(tidy(data) >> group_by(col("key"))
                    >> summarise(s=sum(col("v"), na_rm=na_rm), m=mean(col("v"), na_rm=na_rm))).collect()
                   for data in (lazy, loaded)]
    for name, op in (("s", "sum"), ("m", "mean")):
        got, want = results[0][name].values, results[1][name].values
        assert got.dtype == F32
        wrong = [k for k in range(len(groups)) if not same_bits(got[k], want[k])]
        assert wrong == [], [(k, groups[k][:4], got[k], want[k]) for k in wrong[:5]]
        assert same_bits(want, np.array([expected(g, op, na_rm) for g in groups], F32))


@pytest.mark.parametrize("na_rm", [False, True])
def test_ungrouped_sums_over_one_axis_keep_the_other(na_rm):
    rng = np.random.default_rng(5)
    grid = (rng.normal(0, 1e4, (500, 7)) * 10.0 ** rng.uniform(-3, 3, (500, 7))).astype(F32)
    grid[3, 2] = np.nan
    grid[:, 4] = np.tile(np.array([2.0**53, 1, -(2.0**53), 0.5], F32), 125)  # cancellation needs the exact pass
    grid[:, 5] = np.tile(np.array([1.0, 2.0**-24], F32), 250)               # a float32 tie
    lazy = da.from_array(grid, chunks=((150, 200, 150), (3, 4)))
    outputs = []
    for data in (lazy, grid):
        frame = tidy(xr.Dataset({"v": (("time", "station"), data)}))
        with np.errstate(all="ignore"):
            out = (frame >> mutate(s=sum(col("v"), over="time", na_rm=na_rm),
                                   m=mean(col("v"), over="time", na_rm=na_rm))).collect()
            whole = (frame >> summarise(m=mean(col("v"), na_rm=na_rm))).collect()
        outputs.append((out, whole))
    for name, op in (("s", "sum"), ("m", "mean")):
        want = np.array([expected(grid[:, k], op, na_rm) for k in range(7)], F32)
        assert same_bits(outputs[0][0][name].values, want), (outputs[0][0][name].values, want)
        assert same_bits(outputs[1][0][name].values, want)
    assert same_bits(outputs[0][1]["m"].values, expected(grid.ravel(), "mean", na_rm))
    assert outputs[0][1]["m"].dims == ()


def reading_blocks(blocks):
    """A Dask array of *blocks*, and the list of the blocks read so far."""
    reads = []

    @delayed
    def read(index):
        reads.append(index)
        return blocks[index]

    parts = [da.from_delayed(read(i), shape=block.shape, dtype=F32) for i, block in enumerate(blocks)]
    return da.concatenate(parts), reads


def test_results_stay_lazy_and_read_each_chunk_once():
    blocks = [np.arange(start, start + 5, dtype=F32) for start in (0, 5, 10)]
    values, reads = reading_blocks(blocks)
    keys = np.arange(15) % 4
    source = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        frame = tidy(source) >> group_by(col("key")) >> summarise(s=sum(col("v")), m=mean(col("v")))
        view = frame.to_xarray()
        assert isinstance(view.s.data, da.Array) and isinstance(view.m.data, da.Array)
        assert view.s.dims == ("key",) and view.s.dtype == F32
        assert tasks == [] and reads == []
        result = frame.collect()
    assert sorted(reads) == [0, 1, 2]  # one pass: the bound settles these sums; both reductions share the reads
    whole = np.concatenate(blocks)
    assert same_bits(result.s.values, [whole[keys == k].sum(dtype=np.float64) for k in range(4)])
    assert same_bits(result.m.values, [whole[keys == k].mean(dtype=np.float64) for k in range(4)])
    assert result.s.attrs == {} and result.s.encoding == {}


def test_ties_take_a_second_pass_over_the_chunks_only():
    halfway = np.array([1.0, 2.0**-24, 3.0, 2.0**-22], F32)  # sums exactly halfway between float32 values
    blocks = [halfway[:2], halfway[2:], np.array([5.0, 6.0], F32)]
    values, reads = reading_blocks(blocks)
    keys = np.array([0, 0, 1, 1, 2, 2])
    source = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    with config.set(scheduler="synchronous"):
        frame = tidy(source) >> group_by(col("key")) >> summarise(s=sum(col("v")))
        assert reads == []
        result = frame.collect()
    assert sorted(reads) == [0, 0, 1, 1, 2, 2]  # the estimate, then the exact sums of the two tied groups
    assert same_bits(result.s.values, [expected(g, "sum", False) for g in (halfway[:2], halfway[2:], blocks[2])])


def test_cancellation_across_a_chunk_boundary_is_exact():
    values = np.array([1e16, 1.0, -1e16, 2.0**24, *[1.0] * 1000, -(2.0**24)], F32)
    keys = np.array([0, 0, 0, *[1] * 1002])
    for chunks in ((1, 1, 1, 1002), (2, 1, 500, 502), (3, 1002), (1005,)):
        lazy = xr.Dataset({"v": ("point", da.from_array(values, chunks=(chunks,))), "key": ("point", keys)})
        out = (tidy(lazy) >> group_by(col("key")) >> summarise(s=sum(col("v")), m=mean(col("v")))).collect()
        assert out.s.values.tolist() == [1.0, 1000.0], chunks
        assert same_bits(out.m.values, [expected(values[:3], "mean", False), expected(values[3:], "mean", False)])
    assert np.array(values[:3]).sum() != 1.0  # NumPy's float32 sum gives 0


def test_min_count_and_missing_keys_across_chunks():
    values = np.array([1.0, np.nan, 2.0, np.nan, np.nan, 4.0, 7.0, 8.0], F32)
    keys = np.array([0, 0, 0, 1, 1, 1, np.nan, np.nan])
    lazy = xr.Dataset({"v": ("point", da.from_array(values, chunks=((3, 3, 2),))), "key": ("point", keys)})
    for min_count, want in ((2, [3.0, np.nan]), (1, [3.0, 4.0])):
        out = (tidy(lazy) >> group_by(col("key"))
               >> summarise(s=sum(col("v"), na_rm=True, min_count=min_count))).collect()
        assert same_bits(out.s.values, want) and out.key.values.tolist() == [0, 1]
    out = (tidy(lazy) >> group_by(col("key"), dropna=False) >> summarise(s=sum(col("v")))).collect()
    assert same_bits(out.s.values, [np.nan, np.nan, 15.0])


def test_grouped_results_broadcast_by_mutate_stay_lazy():
    values, keys, groups = corpus()
    lazy = xr.Dataset({"v": ("point", chunked(values, 1000)), "key": ("point", keys)})
    with np.errstate(all="ignore"), config.set(scheduler="synchronous"):
        frame = tidy(lazy) >> group_by(col("key")) >> mutate(m=mean(col("v"), na_rm=True))
        assert isinstance(frame.to_xarray().m.data, da.Array)
        out = frame.collect()
    want = np.array([expected(g, "mean", True) for g in groups], F32)
    assert same_bits(out["m"].values, want[keys])


def test_chunks_large_enough_for_the_compiled_kernel_give_the_same_bits():
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    values = (rng.normal(0, 1e3, 400_000) * rng.choice([1, 2.0**-20, 2.0**20], 400_000)).astype(F32)
    keys = rng.integers(0, 32, 400_000)
    lazy = xr.Dataset({"v": ("point", da.from_array(values, chunks=((150_000, 150_000, 100_000),))),
                       "key": ("point", keys)})
    loaded = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    results = [(tidy(data) >> group_by(col("key")) >> summarise(s=sum(col("v")), m=mean(col("v")))).collect()
               for data in (lazy, loaded)]
    for name in ("s", "m"):
        assert same_bits(results[0][name].values, results[1][name].values)
