"""Prototype checks for the NumPy-only core in ``numpy_core.py``.

Each result is compared with a plain NumPy calculation. The point-cloud
example is plan §7: rotate and offset the points, keep those within 50 m,
and summarise intensity per ring. The tidy3A spellings are in the comments.
"""

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from numpy_core import (
    CoreError,
    Dataset,
    Labelled,
    Variable,
    apply,
    concat,
    dot,
    drop_axis,
    group_reduce,
    labelled,
    reduce,
    rename_axes,
    sel,
    transpose,
)

FIELDS = ["x", "y", "z", "intensity", "ring"]
XYZ = ["x", "y", "z"]
# A 90° rotation about z, then 1 m along x: the first three points land at
# 49 m, 50 m, and 51 m from the origin, the last at 3 m.
R = labelled([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], ("i", "feature"),
             {"i": XYZ, "feature": XYZ})
OFFSET = labelled([1.0, 0.0, 0.0], ("i",), {"i": XYZ})


def cloud(values=None) -> Dataset:
    # read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, field_axis="feature")
    if values is None:
        values = np.array(
            [
                [0.0, -48.0, 0.0, 10.0, 1.0],
                [0.0, -49.0, 0.0, 20.0, 2.0],
                [51.0, 1.0, 0.0, 30.0, 1.0],
                [0.0, 1.0, 3.0, 40.0, 2.0],
            ],
            dtype="float32",
        )
    return Dataset({"values": Variable(values, ("point", "feature"))}, {"feature": FIELDS})


def field(data: Dataset, name: str) -> Labelled:
    # A bare field label: the field with the field axis dropped (plan §5.2).
    return drop_axis(sel(data["values"], feature=[name]), "feature")


def section_7(data: Dataset):
    # cloud >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
    #       >> mutate(distance = sqrt(sum(p ** 2, over="i")))
    #       >> filter(distance <= 50)
    p = apply(np.add, dot(R, sel(data["values"], feature=XYZ), over="feature"), OFFSET)
    distance = apply(np.sqrt, reduce(apply(np.square, p), "sum", over="i"))
    data = data.assign(p=p, distance=distance)
    keep = apply(np.less_equal, data["distance"], 50.0)
    return data.isel(op="filter", point=keep.values)


def numpy_section_7(values: np.ndarray):
    R_, t = R.values, OFFSET.values
    with np.errstate(all="ignore"):  # NumPy 2.0's matmul on macOS warns spuriously
        p = values[:, :3] @ R_.T + t
    distance = np.sqrt((p ** 2).sum(axis=1))
    keep = distance <= 50
    return keep, p[keep], distance[keep]


# ── Variable and Dataset ────────────────────────────────────────────────────


def test_variable_checks_names_against_rank():
    with pytest.raises(CoreError, match="3 names"):
        Variable(np.zeros((2, 2)), ("a", "b", "c"))
    with pytest.raises(CoreError, match="unique"):
        Variable(np.zeros((2, 2)), ("a", "a"))


def test_variable_is_immutable_and_errors_name_the_operation():
    var = Variable(np.zeros((2, 3)), ("time", "sensor"))
    with pytest.raises(AttributeError):
        var.dims = ("a", "b")
    with pytest.raises(TypeError):
        var.attrs["units"] = "m"
    with pytest.raises(CoreError, match=r"sum: no axis 'height'; axes are \('time', 'sensor'\)"):
        reduce(Labelled(var), "sum", over="height")


def test_dataset_shares_labels_and_sizes_across_variables():
    data = Dataset(
        {"intensity": Variable(np.ones((3, 2)), ("time", "sensor")), "gain": Variable([1.0, 2.0], ("sensor",))},
        {"sensor": ["front", "rear"]},
    )
    assert data["gain"].coords["sensor"].tolist() == ["front", "rear"]
    with pytest.raises(CoreError, match="other variables have 2"):
        Dataset({"a": Variable(np.ones(2), ("sensor",)), "b": Variable(np.ones(3), ("sensor",))})
    with pytest.raises(CoreError, match="3 labels"):
        Dataset({"a": Variable(np.ones(2), ("sensor",))}, {"sensor": ["a", "b", "c"]})


# ── Broadcasting by name, with label checks ─────────────────────────────────


def test_broadcasting_is_by_name_not_position():
    a = labelled(np.arange(6.0).reshape(2, 3), ("time", "sensor"))
    b = labelled(np.arange(3.0), ("sensor",))
    c = labelled(np.arange(6.0).reshape(3, 2), ("sensor", "time"))
    np.testing.assert_array_equal(apply(np.add, a, b).values, a.values + b.values)
    out = apply(np.multiply, a, c)
    assert out.dims == ("time", "sensor")
    np.testing.assert_array_equal(out.values, a.values * c.values.T)
    assert apply(np.add, b, labelled([1.0, 2.0], ("time",))).dims == ("sensor", "time")


def test_broadcasting_checks_sizes_and_labels():
    a = labelled([1.0, 2.0], ("point",), {"point": [1, 2]})
    with pytest.raises(CoreError, match="2 positions on one operand and 3"):
        apply(np.add, labelled([1.0, 2.0], ("point",)), labelled([1.0, 2.0, 3.0], ("point",)))
    with pytest.raises(CoreError, match="labelled on one operand and unlabelled"):
        apply(np.add, a, labelled([1.0, 2.0], ("point",)))
    with pytest.raises(CoreError, match="1 only in the first operand, 1 only in operand 2"):
        apply(np.add, a, labelled([1.0, 2.0], ("point",), {"point": [1, 3]}))
    with pytest.raises(CoreError, match="bare array"):
        apply(np.add, a, np.array([1.0, 2.0]))


def test_same_labels_in_another_order_are_matched_by_label():
    a = labelled([10.0, 20.0, 30.0], ("sensor",), {"sensor": ["a", "b", "c"]})
    b = labelled([3.0, 1.0, 2.0], ("sensor",), {"sensor": ["c", "a", "b"]})
    out = apply(np.add, a, b)
    assert out.values.tolist() == [11.0, 22.0, 33.0]
    assert out.coords["sensor"].tolist() == ["a", "b", "c"]
    assert out.notes == ("changed: sensor reordered to match the first operand in add",)


@pytest.mark.parametrize("labels", [["a", "b", "b"], ["a", "b", "a"]])
def test_alignment_rejects_ambiguous_repeated_labels(labels):
    a = labelled([10, 20, 30], ("point",), {"point": ["a", "a", "b"]})
    b = labelled([1, 2, 3], ("point",), {"point": labels})
    with pytest.raises(CoreError, match="add:.*point.*repeated labels"):
        apply(np.add, a, b)
    with pytest.raises(CoreError, match="add:.*point.*repeated labels"):
        apply(np.add, b, a)


def test_identical_repeated_labels_pair_by_position():
    coords = {"point": ["a", "a", "b"]}
    out = apply(np.add, labelled([10, 20, 30], ("point",), coords),
                labelled([1, 2, 3], ("point",), coords))
    np.testing.assert_array_equal(out.values, [11, 22, 33])
    np.testing.assert_array_equal(out.coords["point"], coords["point"])


# ── Core operations ─────────────────────────────────────────────────────────


def test_reductions_drop_axes_and_follow_missing_value_rules():
    a = labelled([[1.0, np.nan], [3.0, 4.0]], ("time", "sensor"), {"sensor": ["a", "b"]})
    out = reduce(a, "mean", over="time")
    assert out.dims == ("sensor",) and np.isnan(out.values[1]) and out.coords["sensor"].tolist() == ["a", "b"]
    assert reduce(a, "mean", over="time", na_rm=True).values.tolist() == [2.0, 4.0]
    assert reduce(a, "sum", over=["time", "sensor"], na_rm=True).values == 8.0


def test_dot_matches_einsum_with_reordered_axes_and_labels():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(4, 3))
    v = rng.normal(size=(5, 3))
    M = labelled(m, ("out", "feature"), {"feature": ["x", "y", "z"]})
    # Same vectors, stored feature-first with labels in another order.
    V = labelled(v[:, [2, 0, 1]].T, ("feature", "point"), {"feature": ["z", "x", "y"]})
    out = dot(M, V, over="feature")
    assert out.dims == ("out", "point")
    np.testing.assert_allclose(out.values, np.einsum("of,pf->op", m, v))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("shape", [(3,), (2, 3), (0,)])
def test_dot_can_contract_to_a_scalar(shape, dtype):
    values = np.arange(np.prod(shape)).reshape(shape).astype(dtype) + 1
    if dtype == "complex64":
        values *= 1 + 1j
    dims = tuple(f"axis{k}" for k in range(len(shape)))
    item = labelled(values, dims, {d: np.arange(n) for d, n in zip(dims, shape)})
    out = dot(item, item, over=dims)
    assert out.dims == () and out.coords == {} and out.values.shape == ()
    assert out.values.dtype == values.dtype
    np.testing.assert_allclose(out.values, np.sum(values * values))


@pytest.mark.parametrize("size", [0, 3])
def test_empty_label_selection_preserves_shape_coords_and_dtype(size):
    values = np.arange(2 * size, dtype="int16").reshape(2, size)
    item = labelled(values, ("sensor", "point"),
                    {"sensor": ["a", "b"], "point": np.arange(size, dtype="int32")})
    out = sel(item, point=[])
    assert out.dims == item.dims and out.values.shape == (2, 0)
    assert out.values.dtype == values.dtype
    assert out.coords["point"].size == 0 and out.coords["point"].dtype == np.dtype("int32")
    np.testing.assert_array_equal(out.coords["sensor"], item.coords["sensor"])


def test_label_selection_preserves_requested_order_and_repetitions():
    item = labelled([10, 20, 30], ("point",), {"point": ["a", "b", "c"]})
    out = sel(item, point=["c", "a", "c"])
    np.testing.assert_array_equal(out.values, [30, 10, 30])
    np.testing.assert_array_equal(out.coords["point"], ["c", "a", "c"])


def test_label_selection_rejects_repeated_source_labels():
    item = labelled([10, 20, 30], ("point",), {"point": ["a", "a", "b"]})
    with pytest.raises(CoreError, match="sel:.*point.*repeated labels"):
        sel(item, point=["a"])
    assert sel(item, point=[]).values.shape == (0,)


def test_transpose_concat_and_rename_keep_labels():
    a = labelled(np.arange(6.0).reshape(2, 3), ("time", "sensor"), {"sensor": ["a", "b", "c"]})
    assert transpose(a, "sensor", "time").values.tolist() == a.values.T.tolist()
    packed = concat([a, a], "component", labels=["u", "v"])  # pack([u, v], along="component")
    assert packed.dims == ("component", "time", "sensor") and packed.coords["component"].tolist() == ["u", "v"]
    assert rename_axes(a, sensor="channel").coords["channel"].tolist() == ["a", "b", "c"]


def test_filtering_keeps_original_one_based_positions():
    data = cloud()
    kept = data.isel(point=np.array([False, True, False, True]))
    assert kept.coords["point"].tolist() == [2, 4]
    with pytest.raises(CoreError, match="labelled on one operand and unlabelled"):
        # Filtered data no longer pairs by position with unfiltered data.
        apply(np.subtract, kept["values"], data["values"])


# ── Plan §7 and a grouped summary, against NumPy ────────────────────────────


def test_section_7_on_the_boundary_points():
    out = section_7(cloud())
    assert out.coords["point"].tolist() == [1, 2, 4]  # 49 m and 50 m kept, 51 m dropped
    np.testing.assert_allclose(out["distance"].values, [49.0, 50.0, 3.0])
    assert field(out, "intensity").values.tolist() == [10.0, 20.0, 40.0]


def test_section_7_on_a_random_cloud_matches_numpy():
    rng = np.random.default_rng(7)
    values = np.empty((50_000, 5), dtype="float32")
    values[:, :3] = rng.uniform(-80, 80, (50_000, 3))
    values[:, 3] = rng.uniform(0, 255, 50_000)
    values[:, 4] = rng.integers(0, 32, 50_000)
    out = section_7(cloud(values))
    keep, p, distance = numpy_section_7(values)
    np.testing.assert_array_equal(out.coords["point"], np.flatnonzero(keep) + 1)
    np.testing.assert_allclose(transpose(out["p"], "point", "i").values, p, rtol=1e-6)
    np.testing.assert_allclose(out["distance"].values, distance, rtol=1e-6)
    np.testing.assert_array_equal(field(out, "intensity").values, values[keep, 3])

    # >> group_by(ring) >> summarise(mean_intensity = mean(intensity))
    summary = group_reduce(field(out, "intensity"), field(out, "ring"), "mean", "ring")
    rings = values[keep, 4]
    assert summary.coords["ring"].tolist() == np.unique(rings).tolist()
    for k, ring in enumerate(summary.coords["ring"]):
        assert summary.values[k] == values[keep, 3][rings == ring].mean()  # bit-identical


def test_grouping_keeps_other_axes_and_checks_the_key():
    data = labelled(np.arange(12.0).reshape(3, 4), ("sensor", "point"))
    key = labelled([1, 0, 1, 0], ("point",))
    out = group_reduce(data, key, "sum", "parity")
    assert out.dims == ("parity", "sensor")
    assert out.values.tolist() == [[4.0, 12.0, 20.0], [2.0, 10.0, 18.0]]  # key 0: positions 2 and 4
    with pytest.raises(CoreError, match="not an axis of the data"):
        group_reduce(data, labelled([1, 2, 3], ("time",)), "sum", "t")
    with pytest.raises(CoreError, match="2 positions have no key"):
        group_reduce(data, labelled([1.0, np.nan, 1.0, np.nan], ("point",)), "sum", "k")


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("op,dtype,expected_dtype", [
    ("sum", "int16", "intp"), ("mean", "int16", "float64"),
    ("mean", "float32", "float32"), ("mean", "complex64", "complex64"),
    ("min", "float32", "float64"), ("max", "int16", "float64"),
    ("std", "complex64", "float32"), ("var", "float32", "float32"),
    ("median", "float32", "float32"), ("any", "float32", "float64"),
    ("all", "bool", "bool"),
])
def test_empty_grouping_preserves_remaining_axes_and_reduction_dtype(axis, op, dtype, expected_dtype):
    dims, shape = (("point", "sensor"), (0, 2)) if axis == 0 else (("sensor", "point"), (2, 0))
    item = labelled(np.empty(shape, dtype=dtype), dims, {"sensor": ["a", "b"]})
    key = labelled(np.empty(0, dtype="int16"), ("point",))
    out = group_reduce(item, key, op, "group")
    assert out.dims == ("group", "sensor") and out.values.shape == (0, 2)
    assert out.values.dtype == np.dtype(expected_dtype)
    np.testing.assert_array_equal(out.coords["sensor"], ["a", "b"])
    assert out.coords["group"].size == 0 and out.coords["group"].dtype == key.values.dtype


def test_section_7_can_filter_every_point_then_summarise():
    values = np.array([[100, 100, 100, 10, 1], [200, 200, 200, 20, 2]], dtype="float32")
    out = section_7(cloud(values))
    assert out["values"].values.shape == (0, 5) and out.coords["point"].size == 0
    assert out["p"].values.shape == (3, 0) and out["distance"].values.shape == (0,)
    summary = group_reduce(field(out, "intensity"), field(out, "ring"), "mean", "ring")
    assert summary.dims == ("ring",) and summary.values.shape == (0,)
    assert summary.values.dtype == np.dtype("float32")


def test_standalone_core_grouping_never_imports_pandas_or_xarray():
    script = '''
import builtins
import numpy as np
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('pandas', 'xarray', 'tidy3a'):
        raise AssertionError('unexpected dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from numpy_core import GroupIndex, labelled
groups = GroupIndex.build(labelled([0, 0, 1, 1], ('point',)), 'key')
result = groups.reduce(labelled(np.array([1, 3, 10, 14], dtype='float32'), ('point',)), 'mean')
np.testing.assert_array_equal(result.values, [2, 12])
'''
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent, check=True,
                   capture_output=True, text=True)
