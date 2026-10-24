"""The existing tidy3A syntax executed on the experimental NumPy frame."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import numpy_core as core
import numpy_pipeline as proto
import tidy3a as ta


FIELDS = ["x", "y", "z", "intensity", "ring"]
XYZ = ["x", "y", "z"]
x, y, z, intensity, ring, distance = ta.fields(*FIELDS, "distance")


def records(kind):
    if kind == "boundary":
        return np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2],
                         [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float32")
    if kind == "empty":
        return np.empty((0, 5), dtype="float32")
    if kind == "outside":
        return np.array([[100, 100, 100, 10, 1], [200, 200, 200, 20, 2]], dtype="float32")
    rng = np.random.default_rng(4)
    values = rng.uniform(-80, 80, (1024, 5)).astype("float32")
    values[:, 4] = rng.integers(0, 16, len(values))
    return values


def frame(values, reader):
    return reader(values, dims={"point": ta.infer, "feature": 5},
                  labels={"feature": FIELDS}, field_axis="feature")


def transform(data, R, offset):
    return (data >> ta.mutate(p=ta.dot(R, [x, y, z], over="feature") + offset)
            >> ta.mutate(distance=ta.formula(r"\sqrt{\sum_i p_i^2}"))
            >> ta.filter(distance <= 50))


def summary(data):
    return data >> ta.group_by(ring) >> ta.summarise(points=ta.n(), mean_intensity=ta.mean(intensity))


def assert_same(actual, expected):
    xr.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert list(actual.data_vars) == list(expected.data_vars)
    for name in actual.data_vars:
        assert actual[name].dims == expected[name].dims
        assert actual[name].dtype == expected[name].dtype
    for name in actual.coords:
        np.testing.assert_array_equal(actual[name].values, expected[name].values)
        assert actual[name].dtype == expected[name].dtype


def test_existing_raw_file_workflow_unchanged(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "tests" / "test_workflow.py"
    spec = importlib.util.spec_from_file_location("original_cloud_workflow", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    monkeypatch.setattr(original, "read_array", proto.read_array)
    original.test_cloud_workflow_from_a_raw_file(tmp_path)


@pytest.mark.parametrize("kind", ["boundary", "random", "outside", "empty"])
@pytest.mark.parametrize("reordered", [False, True])
def test_pipeline_matches_current_tidy3a_and_numpy(kind, reordered):
    values = records(kind)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="float32")
    translation = np.array([1, 0, 0], dtype="float32")
    R = xr.DataArray(rotation, dims=("i", "feature"), coords={"i": XYZ, "feature": XYZ})
    offset = xr.DataArray(translation, dims="i", coords={"i": XYZ})
    if reordered:
        R = R.sel(feature=["z", "x", "y"])
        offset = offset.sel(i=["y", "z", "x"])
    original = frame(values, ta.read_array)
    experimental = frame(values, proto.read_array)
    before = experimental.collect()
    expected = transform(original, R, offset)
    actual = transform(experimental, R, offset)
    assert isinstance(actual._data, core.Dataset)
    assert_same(actual.collect(), expected.collect())
    assert_same(summary(actual).collect(), summary(expected).collect())
    xr.testing.assert_identical(experimental.collect(), before)
    p = values[:, :3] @ rotation.T + translation
    d = np.sqrt(np.sum(p * p, axis=1))
    np.testing.assert_array_equal(actual >> ta.pull("point"), np.flatnonzero(d <= 50) + 1)
    np.testing.assert_allclose(actual >> ta.pull(distance), d[d <= 50], rtol=1e-6)
    assert "as array code:" in repr(actual)


def test_pipeline_does_not_use_xarray_or_pandas_computation_between_boundaries(monkeypatch):
    data = frame(records("boundary"), proto.read_array)
    R = xr.DataArray([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),
                     coords={"i": XYZ, "feature": XYZ})
    offset = xr.DataArray([1., 0, 0], dims="i", coords={"i": XYZ})

    def forbidden(*args, **kwargs):
        raise AssertionError("xarray computation inside the NumPy pipeline")

    with monkeypatch.context() as patch:
        patch.setattr(xr.DataArray, "__init__", forbidden)
        patch.setattr(xr.Dataset, "__init__", forbidden)
        patch.setattr(xr, "dot", forbidden)
        patch.setattr(xr, "concat", forbidden)
        patch.setattr(pd, "factorize", forbidden)
        out = summary(transform(data, R, offset))
    np.testing.assert_array_equal(out.collect()["points"], [1, 2])


def test_sequential_assignments_and_external_label_reordering():
    source = xr.Dataset({"v": ("point", [10., 20., 30.])}, coords={"point": [1, 2, 3]})
    external = xr.DataArray([300., 100., 200.], dims="point", coords={"point": [3, 1, 2]})
    verb = ta.mutate(a=external, b=ta.col("a") + ta.col("v"))
    assert_same(verb(proto.tidy(source)).collect(), verb(ta.tidy(source)).collect())


@pytest.mark.parametrize("na_rm", [False, True])
def test_grouped_missing_values_keep_existing_numeric_rules(na_rm):
    source = xr.Dataset({"v": ("point", np.array([1, np.nan, 3, 5], dtype="float32")),
                         "key": ("point", [0, 0, 1, 1])})
    verbs = [ta.group_by("key"), ta.summarise(m=ta.mean(ta.col("v"), na_rm=na_rm),
                                             s=ta.sum(ta.col("v"), na_rm=na_rm), points=ta.n())]
    expected, actual = ta.tidy(source), proto.tidy(source)
    for verb in verbs:
        expected, actual = expected >> verb, actual >> verb
    assert_same(actual.collect(), expected.collect())


def test_axis_grouping_and_formula_binding():
    source = xr.DataArray(np.arange(12, dtype="float32").reshape(3, 4), dims=("time", "sensor"),
                          coords={"sensor": ["a", "b", "c", "d"]}, name="v")
    verbs = [ta.mutate(s=ta.formula(r"\sum_j v_j", bind={"j": "sensor"})),
             ta.group_by("time"), ta.summarise(s=ta.mean(ta.col("s")), count=ta.n())]
    actual, expected = proto.tidy(source), ta.tidy(source)
    for verb in verbs:
        actual, expected = actual >> verb, expected >> verb
    assert_same(actual.collect(), expected.collect())


@pytest.mark.parametrize("verb", [ta.select(ta.everything()), ta.group_by(ta.bins(ring, [0, 2, 4]))])
def test_unsupported_operations_fail_explicitly(verb):
    with pytest.raises(NotImplementedError, match="NumPy prototype"):
        frame(records("boundary"), proto.read_array) >> verb


def test_name_collisions_bad_filters_and_alignment_are_errors():
    data = frame(records("boundary"), proto.read_array)
    with pytest.raises(ta.FilterError, match="true/false"):
        data >> ta.filter(x)
    with pytest.raises(ta.MutateError, match="labels"):
        data >> ta.mutate(p=ta.dot(xr.DataArray([1., 2., 3.], dims="feature",
                                               coords={"feature": ["x", "y", "wrong"]}),
                                   [x, y, z], over="feature"))
    source = xr.Dataset({"values": (("point", "feature"), np.ones((2, 2))),
                         "x": ("point", [1, 2])}, coords={"feature": ["x", "y"]})
    with pytest.raises(ta.RefError, match="both a variable and a label"):
        proto.tidy(source, field_axis="feature") >> ta.pull(x)
