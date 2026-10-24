import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from tidy3a import TidyArray, TidyError, tidy


def labelled():
    return xr.DataArray(
        np.zeros((2, 3)),
        dims=("point", "feature"),
        coords={"feature": ["x", "y", "z"]},
    )


def test_unnamed_dataarray_becomes_values():
    frame = tidy(labelled())
    assert list(frame.collect().data_vars) == ["values"]
    assert "source: xarray DataArray" in repr(frame)


def test_named_dataarray_keeps_its_name():
    assert list(tidy(labelled().rename("cloud")).collect().data_vars) == ["cloud"]


def test_dataset_keeps_variables_and_coordinates():
    ds = xr.Dataset(
        {"intensity": (("time", "sensor"), np.ones((2, 2))), "gain": ("sensor", [1.0, 2.0])},
        coords={"sensor": ["front", "rear"]},
    )
    out = tidy(ds).collect()
    assert list(out.data_vars) == ["intensity", "gain"]
    assert out.sensor.values.tolist() == ["front", "rear"]


def test_tidy_is_idempotent():
    frame = tidy(labelled(), field_axis="feature")
    assert tidy(frame) is frame
    assert tidy(frame, field_axis="feature") is frame


def test_tidy_can_add_a_field_axis_but_not_change_it():
    frame = tidy(labelled())
    assert tidy(frame, field_axis="feature")._field_axis == "feature"
    with pytest.raises(TidyError, match="already has field axis"):
        tidy(tidy(labelled(), field_axis="feature"), field_axis="point")


def test_field_axis_validation():
    with pytest.raises(TidyError, match="has no labels"):
        tidy(labelled(), field_axis="point")
    with pytest.raises(TidyError, match="is not an axis"):
        tidy(labelled(), field_axis="band")
    repeated = labelled().assign_coords(feature=["x", "x", "z"])
    with pytest.raises(TidyError, match="repeated labels"):
        tidy(repeated, field_axis="feature")


def test_tidy_rejects_non_xarray():
    with pytest.raises(TypeError, match="use read_array"):
        tidy(np.zeros(3))


def test_variable_axis_hint_without_field_axis():
    da = xr.Dataset({"temp": ("time", [1.0]), "pressure": ("time", [2.0])}).to_dataarray()
    text = repr(tidy(da))
    assert 'hint: "variable" looks like an axis of variable names' in text
    assert 'tidy(da, field_axis="variable")' in text
    assert "hint" not in repr(tidy(da, field_axis="variable"))


def test_inspection_matches_the_plan_example():
    da = xr.DataArray(
        np.zeros((1200, 2, 3), dtype="float32"),
        dims=("time", "sensor", "feature"),
        coords={"sensor": ["front", "rear"], "feature": ["x", "y", "z"]},
    )
    lines = repr(tidy(da, field_axis="feature")).splitlines()
    assert lines[1] == "values (time: 1200, sensor: 2, feature: 3) float32"
    assert lines[2].split() == ["time:", "unlabeled,", "positions", "1", "…", "1,200"]
    assert lines[3].split()[:3] == ["sensor:", "front", "rear"]
    assert lines[3].endswith("observation axis (labels are values)")
    assert lines[4].endswith("field axis (labels act as columns)")


def test_time_and_float_labels_are_shown_compactly():
    times = np.array(["2013-01-01T00:00", "2013-01-01T06:00"], dtype="datetime64[ns]")
    text = repr(tidy(xr.DataArray([1.0, 2.0], dims="time", coords={"time": times})))
    assert "2013-01-01T00:00:00  2013-01-01T06:00:00    observation axis" in text
    days = np.array(["2026-06-01", "2026-07-01"], dtype="datetime64[ns]")
    assert "2026-06-01  2026-07-01" in repr(tidy(xr.DataArray([1.0, 2.0], dims="t", coords={"t": days})))
    noisy = xr.DataArray([1.0, 2.0], dims="y", coords={"y": [9120746.500028737, 9120718.000028737]})
    assert "y: 9120746.5  9120718    observation axis" in repr(tidy(noisy))


def test_long_label_lists_are_shortened():
    da = xr.DataArray(np.zeros(20), dims="band", coords={"band": [f"b{i}" for i in range(20)]})
    assert "b0  b1  b2  b3  b4  b5  …  b19" in repr(tidy(da))


def test_to_xarray_and_collect_share_values():
    values = np.arange(6.0).reshape(2, 3)
    frame = tidy(xr.DataArray(values, dims=("a", "b")))
    assert np.shares_memory(frame.collect()["values"].data, values)
    view = frame.to_xarray()
    view["extra"] = ("a", [1.0, 2.0])
    assert "extra" not in frame.to_xarray()


def test_pipe_calls_the_verb():
    frame = tidy(labelled())
    assert (frame >> (lambda f: f)) is frame
    with pytest.raises(TypeError, match="must be a tidy3A verb"):
        frame >> 3


def test_import_needs_no_optional_packages():
    code = (
        "import sys, tidy3a; "
        "loaded = [m for m in ('IPython', 'dask', 'flox', 'nuscenes', 'tidy3', 'plot3') if m in sys.modules]; "
        "print(loaded)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_wrapper_type_is_exported():
    assert isinstance(tidy(labelled()), TidyArray)
