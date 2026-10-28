"""The backend a dataset's blocks run on (plan milestone 16): chosen on the block of no positions, in memory.

Every block is computed into memory, so a lazy or Dask-backed dataset does not keep its blocks on xarray, nor do
its storage encodings, which no summary keeps; what else the NumPy backend cannot hold still does. The default
backend (``set_backend``, which the test configuration sets per run) and ``backend=`` are followed as ever.
"""

import numpy as np
import pytest

from tidy3a import fields, filter, group_by, max, mean, mutate, n, pull, summarise, tidy, write

xr = pytest.importorskip("xarray")
t, hour, month, celsius = fields("temperature", "hour", "month", "celsius")
WARM = (filter((hour >= 6) & (hour <= 18)) >> mutate(celsius=t - 273.15) >> group_by(month)
        >> summarise(hours=n(), warmth=mean(celsius), hottest=max(celsius)))


def grid():
    rng = np.random.default_rng(0)
    steps = 480
    return xr.Dataset({"temperature": (("time", "cell"), rng.normal(280, 10, (steps, 30)).astype(np.float32)),
                       "hour": ("time", np.arange(steps) % 24), "month": ("time", np.arange(steps) // 72 % 12 + 1)})


def stored(tmp_path, **opening):
    pytest.importorskip("zarr")
    data = grid()
    data.to_zarr(tmp_path / "grid.zarr", consolidated=False, encoding={"temperature": {"chunks": (24, 30)}})
    return xr.open_zarr(tmp_path / "grid.zarr", consolidated=False, **opening)


@pytest.mark.parametrize("opening", [{"chunks": None}, {}], ids=["lazy", "dask"])
def test_a_files_blocks_run_on_the_default_backend(tmp_path, opening, tidy3a_backend):
    data = stored(tmp_path, **opening)
    frame = tidy(data, blocks={"time": 48})
    assert frame._blocks.source.backend == tidy3a_backend  # the file's laziness and encodings do not decide
    assert frame._blocks.load(0, 48)._backend.name == tidy3a_backend


def test_dask_chunked_data_in_memory_runs_on_the_default_backend(tidy3a_backend):
    pytest.importorskip("dask")
    frame = tidy(grid().chunk({"time": 24}), blocks={"time": "chunks"})
    assert frame._blocks.source.backend == tidy3a_backend


def test_what_the_numpy_backend_cannot_hold_keeps_the_blocks_on_xarray():
    data = grid().assign_coords(station=("time", np.arange(480)))  # a coordinate that is not an axis's labels
    assert tidy(data, blocks={"time": 48})._blocks.source.backend == "xarray"
    with pytest.raises(Exception, match="station is a coordinate that is not one axis's labels"):
        tidy(data, blocks={"time": 48}, backend="numpy").collect()


def test_a_backend_asked_for_is_the_blocks_backend(tmp_path):
    data = stored(tmp_path, chunks=None)
    assert tidy(data, blocks={"time": 48}, backend="xarray")._blocks.source.backend == "xarray"
    assert tidy(data, blocks={"time": 48}, backend="numpy")._blocks.source.backend == "numpy"


@pytest.mark.parametrize("opening", [{"chunks": None}, {}], ids=["lazy", "dask"])
def test_both_backends_give_a_files_summary_the_same_bits(tmp_path, opening):
    data = stored(tmp_path, **opening)
    results = [tidy(data, blocks={"time": 48}, backend=backend, threads=threads) >> WARM
               for backend in ("numpy", "xarray") for threads in (1, 3)]
    want = tidy(grid()) >> WARM
    for result in results:
        for name in ("month", "hours", "warmth", "hottest"):
            assert np.asarray(result >> pull(name)).tobytes() == np.asarray(want >> pull(name)).tobytes(), name


def test_a_files_rows_write_from_numpy_blocks(tmp_path):
    data = stored(tmp_path, chunks=None)
    written = (tidy(data, blocks={"time": 48}, backend="numpy") >> filter(hour >= 12)
               >> write(tmp_path / "afternoons.zarr"))
    want = tidy(grid()) >> filter(hour >= 12)
    assert np.array_equal(written.collect()["temperature"].values, want.collect()["temperature"].values)


def packed(tmp_path):
    """A Zarr store whose temperature is packed into 16-bit integers, with times in hours since a date."""
    pytest.importorskip("zarr")
    data = grid().assign(when=("time", np.datetime64("2026-01-01T00", "h") + np.arange(480)))
    data.to_zarr(tmp_path / "packed.zarr", consolidated=False, encoding={
        "temperature": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 280.0, "_FillValue": -32768},
        "when": {"units": "hours since 2026-01-01", "calendar": "standard"}})
    return xr.open_zarr(tmp_path / "packed.zarr", consolidated=False, chunks=None)


@pytest.mark.parametrize("verbs, packed_kept", [(filter(hour >= 6), True),
                                                (filter(hour >= 6) >> mutate(temperature=t * 2), False)],
                         ids=["kept", "changed"])
def test_a_file_written_from_numpy_blocks_keeps_the_sources_encodings(tmp_path, verbs, packed_kept):
    """As from blocks on xarray: a variable the verbs leave as it is keeps its packing and time units; one they
    change is written as it is, its packing not applied to values it may not fit."""
    source = packed(tmp_path)
    written = {}
    for backend in ("xarray", "numpy"):
        path = tmp_path / f"out-{backend}.zarr"
        tidy(source, blocks={"time": 50}, backend=backend) >> verbs >> write(path)
        written[backend] = xr.open_zarr(path, consolidated=False)
    for name in ("temperature", "when", "hour"):
        a, b = written["xarray"][name], written["numpy"][name]
        assert a.dtype == b.dtype and np.array_equal(a.values, b.values, equal_nan=True), name
        portable = ("dtype", "scale_factor", "add_offset", "_FillValue", "units", "calendar")
        assert repr({k: v for k, v in sorted(a.encoding.items()) if k in portable}) == \
            repr({k: v for k, v in sorted(b.encoding.items()) if k in portable}), name  # repr: a NaN fill is equal
    encoding = written["numpy"]["temperature"].encoding
    assert (encoding.get("dtype") == np.int16 and encoding.get("scale_factor") == 0.01) == packed_kept
    assert written["numpy"]["when"].encoding.get("units") == "hours since 2026-01-01"


def test_a_dataarrays_encodings_are_carried_too(tmp_path):
    source = packed(tmp_path)["temperature"]
    path = tmp_path / "one.zarr"
    tidy(source, blocks={"time": 50}, backend="numpy") >> write(path)
    assert xr.open_zarr(path, consolidated=False)["temperature"].encoding.get("dtype") == np.int16
