"""The package default (plan milestone 10, decision 16): NumPy for data it holds without loss.

Without ``set_backend()`` or ``backend=``, in-memory data uses the NumPy
backend, except data it cannot hold (lazy, chunked, memory-mapped, other
coordinates, multi-level labels, time zones) or would lose something of
(xarray storage encodings), which stays on xarray. Choosing a backend still
decides: ``set_backend("numpy")`` and ``backend="numpy"`` mean NumPy wherever
it can hold the data, dropping storage encodings as ``docs/backends.md`` says.
"""

import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

import tidy3a.backend as backend_module
from tidy3a import infer, read_array, set_backend, tidy

pytestmark = pytest.mark.backends("xarray", reason="tests the package default itself")


@pytest.fixture
def package_default(monkeypatch):
    monkeypatch.setattr(backend_module, "_default", None)  # no set_backend() yet, as in a fresh session


def _encoded():
    data = xr.Dataset({"a": ("t", np.array([1.0, 2.0], "float32"))}, coords={"t": [1, 2]})
    data.a.encoding = {"dtype": np.dtype("int16"), "scale_factor": 0.5, "_FillValue": np.int16(-9999)}
    return data


def test_in_memory_data_uses_the_numpy_backend_by_default(package_default):
    assert tidy(xr.Dataset({"a": ("t", [1.0, 2.0])}))._backend.name == "numpy"
    assert read_array(np.zeros((3, 2)), dims={"p": infer, "f": 2})._backend.name == "numpy"


def test_a_fresh_session_defaults_to_numpy():
    code = "import tidy3a.backend as b; print(b.get_backend().name)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "numpy"


def test_data_with_storage_encodings_stays_on_xarray_by_default(package_default):
    frame = tidy(_encoded())
    assert frame._backend.name == "xarray"
    assert frame.collect().a.encoding == _encoded().a.encoding
    whole = xr.Dataset({"a": ("t", [1.0, 2.0])})
    whole.encoding = {"unlimited_dims": {"t"}}
    assert tidy(whole)._backend.name == "xarray"
    times = xr.Dataset({"a": ("time", [1.0, 2.0])}, coords={"time": np.array(["2026-01-01", "2026-01-02"],
                                                                             dtype="datetime64[ns]")})
    times.time.encoding = {"units": "days since 2026-01-01", "calendar": "standard"}
    assert tidy(times)._backend.name == "xarray"  # a coordinate's encoding counts too


def test_a_netcdf_round_trip_keeps_its_encodings_by_default(package_default, tmp_path):
    pytest.importorskip("scipy")
    path = tmp_path / "encoded.nc"
    _encoded().to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as opened:
        frame = tidy(opened.load())
        assert frame._backend.name == "xarray"
        frame.collect().to_netcdf(tmp_path / "again.nc", engine="scipy")
    with xr.open_dataset(tmp_path / "again.nc", engine="scipy") as again:
        assert again.a.encoding["scale_factor"] == 0.5 and again.a.encoding["dtype"] == np.dtype("int16")


def test_lazy_data_still_stays_on_xarray_by_default(package_default):
    dask = pytest.importorskip("dask.array")
    assert tidy(xr.Dataset({"a": ("t", dask.ones(4, chunks=2))}))._backend.name == "xarray"


def test_choosing_a_backend_still_decides(package_default):
    assert tidy(_encoded(), backend="numpy")._backend.name == "numpy"
    assert tidy(_encoded(), backend="numpy").collect().a.encoding == {}  # dropped, as documented
    set_backend("numpy")
    assert tidy(_encoded())._backend.name == "numpy"  # NumPy chosen for everything it can hold
    set_backend("xarray")
    assert tidy(xr.Dataset({"a": ("t", [1.0, 2.0])}))._backend.name == "xarray"
