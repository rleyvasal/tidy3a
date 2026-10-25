"""Using tidy3a with NumPy alone (review priority 4): input, output, defaults, and durations.

Without xarray, arrays from the calling code are built with ``read_array``:
a one-variable ``TidyArray`` stands for its variable in any expression, on
either backend. ``TidyArray.to_dict()`` gives the data as NumPy arrays, in
the layout of xarray's ``Dataset.to_dict(data="array")``. Durations in
``bins()`` are read by tidy3a itself, the same whether pandas is installed.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    GroupError,
    TidyError,
    bins,
    col,
    dot,
    expr,
    fields,
    group_by,
    infer,
    mean,
    mutate,
    read_array,
    select,
    summarise,
    tidy,
)

x, y, z = fields("x", "y", "z")
XYZ = ["x", "y", "z"]
RECORDS = np.array([[0, -48, 0, 10], [0, -49, 0, 20], [51, 1, 0, 30], [0, 1, 3, 40]], "float32")


def _cloud():
    return read_array(RECORDS, dims={"point": infer, "feature": 4}, labels={"feature": [*XYZ, "intensity"]},
                      field_axis="feature")


TURN = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float32")


def test_a_one_variable_tidy_array_stands_for_its_variable_in_expressions():
    rotation = read_array(TURN, dims={"i": 3, "feature": 3}, labels={"i": XYZ, "feature": XYZ})
    offset = read_array(np.array([1, 0, 0], "float32"), dims={"i": 3}, labels={"i": XYZ})
    from_numpy = (_cloud() >> mutate(p=dot(rotation, [x, y, z], over="feature") + offset)).collect()
    turn = xr.DataArray(TURN, dims=("i", "feature"), coords={"i": XYZ, "feature": XYZ})
    shift = xr.DataArray(np.array([1, 0, 0], "float32"), dims="i", coords={"i": XYZ})
    from_xarray = (_cloud() >> mutate(p=dot(turn, [x, y, z], over="feature") + shift)).collect()
    xr.testing.assert_identical(from_numpy, from_xarray)


def test_an_outside_tidy_array_on_the_other_backend_is_moved_to_this_one():
    for here, there in (("xarray", "numpy"), ("numpy", "xarray")):
        frame = read_array(RECORDS, dims={"point": infer, "feature": 4}, labels={"feature": [*XYZ, "intensity"]},
                           field_axis="feature", backend=here)
        scale = read_array(np.array([2.0, 3.0, 4.0, 5.0]), dims={"point": 4}, backend=there)
        out = (frame >> mutate(s=col("intensity") * scale)).collect()
        assert out["s"].values.tolist() == [20.0, 60.0, 120.0, 200.0]


def test_an_outside_tidy_array_must_hold_one_variable():
    two = tidy(xr.Dataset({"a": ("point", np.ones(4)), "b": ("point", np.ones(4))}))
    with pytest.raises(TidyError, match=r"holds 2 variables \(a, b\).*select\(a\)"):
        _cloud() >> mutate(s=col("intensity") * two)
    out = (_cloud() >> mutate(s=col("intensity") * (two >> select("a")))).collect()
    assert out["s"].values.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_outside_tidy_arrays_are_named_in_messages():
    labelled = read_array(np.ones(4), dims={"point": 4}, labels={"point": [5, 6, 7, 8]})
    with pytest.raises(TidyError, match="<array values>"):
        _cloud() >> mutate(s=col("intensity") * labelled)


def test_to_dict_is_xarrays_layout_and_round_trips():
    frame = _cloud() >> mutate(d=col("intensity") * 2)
    frame = frame >> group_by(col("intensity")) >> summarise(m=mean(col("d")))
    for item in (_cloud(), frame):
        out = item.to_dict()
        expected = item.collect().to_dict(data="array")
        assert out.keys() == expected.keys() and out["dims"] == expected["dims"] and out["attrs"] == expected["attrs"]
        for part in ("coords", "data_vars"):
            assert out[part].keys() == expected[part].keys()
            for name, entry in out[part].items():
                assert isinstance(entry["data"], np.ndarray)
                assert tuple(entry["dims"]) == tuple(expected[part][name]["dims"])
                assert entry["attrs"] == expected[part][name]["attrs"]
                np.testing.assert_array_equal(entry["data"], expected[part][name]["data"])
                assert entry["data"].dtype == np.asarray(expected[part][name]["data"]).dtype
        xr.testing.assert_identical(xr.Dataset.from_dict(out), item.collect())


DURATIONS = {
    "30 days": 30 * 86_400 * 10**9, "2D": 2 * 86_400 * 10**9, "1h30min": 5_400 * 10**9,
    "1 day 2 hours": 93_600 * 10**9, "1.5 h": 5_400 * 10**9, "90s": 90 * 10**9, "2 weeks": 14 * 86_400 * 10**9,
    "250ms": 250 * 10**6, "3 minutes": 180 * 10**9, "1 hr": 3_600 * 10**9, "10 us": 10_000, "7ns": 7,
}


@pytest.mark.parametrize("text, nanoseconds", DURATIONS.items())
def test_durations_read_the_same_with_and_without_pandas(text, nanoseconds, monkeypatch):
    key = bins(col("t"), text)
    assert expr._duration(text, key) == np.timedelta64(nanoseconds, "ns")
    monkeypatch.setattr(expr, "_pandas", lambda: None)
    assert expr._duration(text, key) == np.timedelta64(nanoseconds, "ns")


@pytest.mark.parametrize("text", ["1 month", "2 years", "P1D", "00:30:00", "2", "5 m", "0.5ns", "days", "0 days"])
def test_durations_tidy3a_cannot_read_are_one_error(text):
    with pytest.raises(GroupError, match="cannot read the duration"):
        expr._duration(text, bins(col("t"), text))


NUMPY_ONLY = textwrap.dedent("""
    import importlib.abc, sys

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("xarray", "pandas"):
                raise ImportError(f"{name} is not installed")

    sys.meta_path.insert(0, Blocked())

    import numpy as np
    import tidy3a
    from tidy3a import TidyError, bins, col, dot, fields, group_by, infer, mean, mutate, read_array, summarise

    assert tidy3a.backend.get_backend().name == "numpy"  # the default when xarray is not installed
    x, y, z = fields("x", "y", "z")
    xyz = ["x", "y", "z"]
    cloud = read_array(np.arange(12, dtype="float32").reshape(4, 3), dims={"point": infer, "feature": 3},
                       labels={"feature": xyz}, field_axis="feature")
    turn = read_array(np.eye(3, dtype="float32"), dims={"i": 3, "feature": 3}, labels={"i": xyz, "feature": xyz})
    out = (cloud >> mutate(p=dot(turn, [x, y, z], over="feature"))).to_dict()
    assert out["data_vars"]["p"]["dims"] == ("i", "point")
    assert out["data_vars"]["p"]["data"].tolist() == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    try:
        cloud.collect()
    except TidyError as error:
        assert "to_dict()" in str(error) and "pip install" in str(error), error
    else:
        raise AssertionError("collect() without xarray should say what to use instead")
    times = read_array(np.arange(4.0), dims={"t": 4},
                       labels={"t": np.datetime64("2026-01-01T00") + np.arange(4) * np.timedelta64(45, "m")})
    per = times >> group_by(bins(col("t"), "1h30min")) >> summarise(m=mean(col("values")))
    assert per.to_dict()["data_vars"]["m"]["data"].tolist() == [0.5, 2.5]
    path = sys.argv[1]
    np.arange(8, dtype="float32").tofile(path)
    try:
        read_array(path, value_type="float32", dims={"point": infer}, lazy=True)
    except TidyError as error:
        assert "pip install 'tidy3a[xarray]'" in str(error), error
    else:
        raise AssertionError("memory-mapped data without xarray should say what to install")
""")


@pytest.mark.backends("xarray", reason="runs its own process without xarray")
def test_tidy3a_works_end_to_end_with_numpy_alone(tmp_path):
    script = tmp_path / "numpy_only.py"
    script.write_text(NUMPY_ONLY)
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "values.bin")], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr
