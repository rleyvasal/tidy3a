import numpy as np
import pytest
import xarray as xr

da = pytest.importorskip("dask.array")
from dask import config, delayed
from dask.callbacks import Callback

from tidy3a import all, any, col, max, mean, median, min, mutate, n, select, slice, std, summarise, sum, tidy, var
from tidy3a.errors import TidyError


@pytest.fixture
def chunked_measurements():
    reads = []

    @delayed
    def read_block(name, start):
        reads.append((name, start))
        return np.arange(start, start + 3, dtype="float32")

    variables = {}
    for name in ("temperature", "quality"):
        blocks = [da.from_delayed(read_block(name, start), shape=(3,), dtype="float32")
                  for start in (0, 3)]
        variables[name] = ("sample", da.concatenate(blocks), {"units": "count"})
    return xr.Dataset(variables, coords={"sample": np.arange(101, 107)}), reads


@pytest.mark.parametrize("as_dataarray", [False, True])
def test_wrapping_and_selection_preserve_chunks_without_running_tasks(chunked_measurements, as_dataarray):
    source, reads = chunked_measurements
    original = source.temperature.data
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        frame = tidy(source.temperature if as_dataarray else source)
        assert "temperature" in repr(frame)
        view = (frame >> select("temperature")).to_xarray()
    assert tasks == [] and reads == []
    assert view.temperature.data is original
    assert view.temperature.chunks == ((3, 3),)
    assert view.temperature.attrs == {"units": "count"}


@pytest.mark.parametrize("positions, expected, starts", [
    ([1, 2], [0, 1], {0}),
    ([6, 5, 4], [5, 4, 3], {3}),
    ([2, 4, 6], [1, 3, 5], {0, 3}),
    ([6, 1, 6], [5, 0, 5], {0, 3}),
    ([], [], {0}),  # Dask evaluates the first block even for an empty basic slice.
])
def test_slicing_defers_tasks_and_collection_reads_expected_blocks(chunked_measurements, positions, expected, starts):
    source, reads = chunked_measurements
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        frame = tidy(source) >> select("temperature") >> slice(sample=positions)
        view = frame.to_xarray()
        assert isinstance(view.temperature.data, da.Array)
        assert tasks == [] and reads == []
        result = frame.collect()
    assert isinstance(result.temperature.data, np.ndarray)
    assert result.temperature.dtype == np.dtype("float32")
    assert result.temperature.values.tolist() == expected
    assert result.sample.values.tolist() == [position + 100 for position in positions]
    assert set(reads) == {("temperature", start) for start in starts}
    assert len(reads) == len(starts)
    assert result.temperature.attrs == {"units": "count"}
    assert frame.to_xarray().temperature.data is view.temperature.data
    assert source.temperature.chunks == ((3, 3),)


@pytest.mark.parametrize("center", [False, True])
def test_arithmetic_and_reduction_broadcast_stay_lazy(chunked_measurements, center):
    source, reads = chunked_measurements
    tasks = []
    offset = mean(col("doubled")) if center else 1
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        frame = (tidy(source) >> mutate(doubled=col("temperature") * 2,
                                       result=col("doubled") - offset) >> select("result"))
        view = frame.to_xarray()
        assert isinstance(view.result.data, da.Array)
        assert view.result.chunks == ((3, 3),)
        assert tasks == [] and reads == []
        result = (frame >> slice(sample=[1, 2])).collect()
    np.testing.assert_allclose(result.result.values, [-5, -3] if center else [-1, 1])
    assert result.result.dtype == np.dtype("float32")
    assert result.result.attrs == {} and result.result.encoding == {}
    starts = {0, 3} if center else {0}
    assert set(reads) == {("temperature", start) for start in starts}
    assert len(reads) == len(starts)
    assert frame.to_xarray().result.data is view.result.data


@pytest.mark.parametrize("reducer, expected", [
    (mean, 2.8), (median, 3), (std, np.sqrt(2.96)), (var, 2.96),
    (sum, 14), (min, 0), (max, 5),
])
@pytest.mark.parametrize("na_rm", [False, True])
def test_numeric_reductions_defer_tasks_and_handle_missing_values(chunked_measurements, reducer, expected, na_rm):
    source, reads = chunked_measurements
    source["temperature"] = source.temperature.where(source.sample != 102)
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        if reducer is median and na_rm:
            with pytest.raises(TidyError, match=r"collect\(\) it first"):
                tidy(source) >> summarise(result=reducer(col("temperature"), na_rm=na_rm))
            assert tasks == [] and reads == []
            return
        frame = tidy(source) >> summarise(result=reducer(col("temperature"), na_rm=na_rm))
        view = frame.to_xarray()
        assert isinstance(view.result.data, da.Array) and view.result.dims == ()
        assert tasks == [] and reads == []
        result = frame.collect()
    if na_rm:
        assert result.result.item() == pytest.approx(expected)
    else:
        assert np.isnan(result.result.item())
    assert result.result.attrs == {} and result.result.encoding == {}
    assert set(reads) == {("temperature", 0), ("temperature", 3)} and len(reads) == 2
    assert frame.to_xarray().result.data is view.result.data


def test_partial_axis_median_skips_missing_values_lazily(chunked_measurements):
    source, reads = chunked_measurements
    source = source[["temperature"]].expand_dims(sensor=["a", "b"])
    source["temperature"] = source.temperature.where(source.sample != 102)
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        frame = (tidy(source) >> mutate(midpoint=median(col("temperature"), over="sample", na_rm=True))
                 >> select("midpoint"))
        assert isinstance(frame.to_xarray().midpoint.data, da.Array)
        assert tasks == [] and reads == []
        result = frame.collect()
    assert result.midpoint.dims == ("sensor",)
    assert result.midpoint.values.tolist() == [3, 3]
    assert result.sensor.values.tolist() == ["a", "b"]
    assert set(reads) == {("temperature", 0), ("temperature", 3)} and len(reads) == 2


@pytest.mark.parametrize("min_count, expected", [(5, 14), (6, np.nan)])
def test_sum_checks_minimum_present_count_across_chunks(chunked_measurements, min_count, expected):
    source, reads = chunked_measurements
    source["temperature"] = source.temperature.where(source.sample != 102)
    with config.set(scheduler="synchronous"):
        frame = tidy(source) >> summarise(result=sum(col("temperature"), na_rm=True, min_count=min_count))
        assert isinstance(frame.to_xarray().result.data, da.Array) and reads == []
        result = frame.collect()
    np.testing.assert_allclose(result.result.item(), expected)
    assert set(reads) == {("temperature", 0), ("temperature", 3)} and len(reads) == 2


@pytest.mark.parametrize("reducer, expected", [(any, True), (all, False)])
def test_logical_reductions_stay_lazy(chunked_measurements, reducer, expected):
    source, reads = chunked_measurements
    with config.set(scheduler="synchronous"):
        frame = tidy(source) >> summarise(result=reducer(col("temperature") > 2))
        assert isinstance(frame.to_xarray().result.data, da.Array) and reads == []
        result = frame.collect()
    assert result.result.dtype == bool and result.result.item() == expected
    assert set(reads) == {("temperature", 0), ("temperature", 3)} and len(reads) == 2


def test_count_uses_shape_without_reading_chunks(chunked_measurements):
    source, reads = chunked_measurements
    tasks = []
    with config.set(scheduler="synchronous"), Callback(pretask=lambda key, *_: tasks.append(key)):
        result = (tidy(source) >> summarise(count=n())).collect()
    assert result["count"].item() == 6
    assert tasks == [] and reads == []
