"""Milestone 0: check the xarray and NumPy behavior that docs/plan.md relies on.

Each test states one claim from the plan, with the section it comes from.
A failing test means the plan assumed something xarray or NumPy does not
do, and the plan must change before implementation.
"""

import numpy as np
import pytest
import xarray as xr


def cloud():
    """Three points with five fields, as read_array would produce."""
    values = np.array(
        [
            [1.0, 2.0, 0.5, 12.0, 3.0],
            [4.0, 1.0, 0.2, 30.0, 1.0],
            [0.5, 3.0, 1.1, 7.0, 3.0],
        ],
        dtype="float32",
    )
    return xr.DataArray(
        values,
        dims=("point", "feature"),
        coords={"feature": ["x", "y", "z", "intensity", "ring"]},
        name="values",
    )


# ── §5.5 Alignment ──────────────────────────────────────────────────────────


def test_exact_join_rejects_different_label_sets():
    a = xr.DataArray([1.0, 2.0, 3.0], dims="point", coords={"point": [1, 2, 3]})
    b = xr.DataArray([1.0, 2.0], dims="point", coords={"point": [1, 2]})
    with xr.set_options(arithmetic_join="exact"):
        with pytest.raises(ValueError):
            a + b


def test_exact_join_rejects_same_labels_in_different_order():
    # §5.5 says tidy3A must reorder before the exact check, because xarray's
    # exact join treats a different order as unequal.
    a = xr.DataArray([1.0, 2.0], dims="sensor", coords={"sensor": ["front", "rear"]})
    b = xr.DataArray([10.0, 20.0], dims="sensor", coords={"sensor": ["rear", "front"]})
    with xr.set_options(arithmetic_join="exact"):
        with pytest.raises(ValueError):
            a - b


def test_reorder_by_label_then_exact_join_pairs_correctly():
    a = xr.DataArray([1.0, 2.0], dims="sensor", coords={"sensor": ["front", "rear"]})
    b = xr.DataArray([10.0, 20.0], dims="sensor", coords={"sensor": ["rear", "front"]})
    with xr.set_options(arithmetic_join="exact"):
        out = a - b.sel(sensor=a.sensor)
    assert out.sensor.values.tolist() == ["front", "rear"]
    assert out.values.tolist() == [1.0 - 20.0, 2.0 - 10.0]


def test_unlabeled_axis_pairs_with_labeled_axis_by_length():
    # §5.5: xarray matches an axis without labels by length alone, which is
    # why tidy3A raises instead.
    labeled = xr.DataArray([1.0, 2.0], dims="point", coords={"point": [3, 7]})
    unlabeled = xr.DataArray([10.0, 20.0], dims="point")
    with xr.set_options(arithmetic_join="exact"):
        out = labeled + unlabeled
    assert out.values.tolist() == [11.0, 22.0]


def test_dot_respects_exact_join():
    R = xr.DataArray(np.eye(2), dims=("i", "feature"), coords={"feature": ["x", "y"]})
    v = xr.DataArray([1.0, 2.0], dims="feature", coords={"feature": ["x", "z"]})
    with xr.set_options(arithmetic_join="exact"):
        with pytest.raises(ValueError):
            xr.dot(R, v, dim="feature")


def test_merge_raises_on_same_named_variables_with_different_values():
    # §5.5: xr.merge does not rename, so tidy3A renames clashes first.
    a = xr.Dataset({"values": ("point", [1.0, 2.0])})
    b = xr.Dataset({"values": ("point", [5.0, 6.0])})
    with pytest.raises(xr.MergeError):
        xr.merge([a, b], compat="no_conflicts", join="exact")


def test_merge_left_join_fills_gaps_with_missing():
    a = xr.Dataset({"a": ("point", [1.0, 2.0])}, coords={"point": [1, 2]})
    b = xr.Dataset({"b": ("point", [9.0])}, coords={"point": [2]})
    out = xr.merge([a, b], join="left")
    assert out.point.values.tolist() == [1, 2]
    assert np.isnan(out.b.values[0]) and out.b.values[1] == 9.0


def test_reindex_nearest_with_tolerance_matches_rounding_noise():
    # §5.5: opt-in tolerance on joins uses reindex(method="nearest").
    a = xr.DataArray([1.0, 2.0], dims="lon", coords={"lon": [0.1, 0.2]})
    b = xr.DataArray([5.0, 6.0], dims="lon", coords={"lon": [0.1 + 1e-12, 0.2 - 1e-12]})
    out = b.reindex(lon=a.lon, method="nearest", tolerance=1e-9)
    assert out.values.tolist() == [5.0, 6.0]


# ── §4.1 / §3 Field axis and separate variables ─────────────────────────────


def test_to_dataset_turns_field_labels_into_variables():
    ds = cloud().to_dataset(dim="feature")
    assert list(ds.data_vars) == ["x", "y", "z", "intensity", "ring"]
    assert ds["x"].dims == ("point",)


def test_to_dataarray_names_the_packed_axis_variable():
    # §5.1: the `variable` hint relies on this default.
    da = cloud().to_dataset(dim="feature").to_dataarray()
    assert da.dims[0] == "variable"


def test_selecting_one_field_label_drops_the_field_axis():
    # §5.2: bare x is x(point).
    assert cloud().sel(feature="x").dims == ("point",)


def test_selecting_several_field_labels_keeps_the_field_axis():
    # §5.2: [x, y, z] is (point, feature: 3).
    out = cloud().sel(feature=["x", "y", "z"])
    assert out.dims == ("point", "feature") and out.sizes["feature"] == 3


# ── §5.3 Filter and slice ───────────────────────────────────────────────────


def test_isel_with_integer_drops_axis_and_list_keeps_it():
    # §5.3: why slice(time = 1) keeps the axis instead of following xarray.
    da = xr.DataArray(np.zeros((2, 3)), dims=("time", "sensor"))
    assert da.isel(time=0).dims == ("sensor",)
    assert da.isel(time=[0]).dims == ("time", "sensor")


def test_filtering_after_attaching_positions_keeps_original_positions():
    da = cloud()
    da = da.assign_coords(point=np.arange(1, da.sizes["point"] + 1))
    kept = da.isel(point=np.array([False, True, True]))
    assert kept.point.values.tolist() == [2, 3]


# ── §5.4 Mutate value types ─────────────────────────────────────────────────


def test_numpy_scalar_arithmetic_keeps_float32():
    # §5.4: under NumPy 2 promotion, x * 2 on float32 stays float32.
    assert (cloud().sel(feature="x") * 2).dtype == np.float32


def test_numpy_assignment_casts_down_silently():
    # §5.4: why write-back promotes instead of assigning in place.
    arr = np.zeros(2, dtype="float32")
    arr[:] = np.array([1.0000000001, 2.0], dtype="float64")
    assert arr.dtype == np.float32


def test_result_type_gives_the_promotion_target():
    assert np.result_type(np.float32, np.float64) == np.float64
    assert np.result_type(np.float32, np.int16) == np.float32
    assert np.result_type(np.float64, np.complex64) == np.complex128


# ── §5.4 Grouping and missing values ────────────────────────────────────────


def test_groupby_axis_reduces_other_axes():
    da = xr.DataArray(
        [[1.0, 10.0], [3.0, 30.0]],
        dims=("time", "sensor"),
        coords={"sensor": ["front", "rear"]},
    )
    out = da.mean(dim=[d for d in da.dims if d != "sensor"])
    assert out.dims == ("sensor",) and out.values.tolist() == [2.0, 20.0]


def test_groupby_value_field_gives_axis_of_sorted_unique_values():
    da = cloud()
    ring = da.sel(feature="ring", drop=True)
    intensity = da.sel(feature="intensity", drop=True)
    out = intensity.groupby(ring.rename("ring")).mean()
    assert out.dims == ("ring",)
    assert out.ring.values.tolist() == [1.0, 3.0]
    assert out.values.tolist() == [30.0, (12.0 + 7.0) / 2]


def test_groupby_bins_drops_values_outside_every_bin():
    # §5.4: xarray drops them silently; tidy3A raises instead.
    distance = xr.DataArray([5.0, 15.0, 80.0], dims="point", name="distance")
    out = distance.groupby_bins(distance, [0, 10, 20]).count()
    assert int(out.sum()) == 2


def test_float_reductions_skip_missing_values_by_default():
    # §5.4: why tidy3A passes skipna explicitly.
    da = xr.DataArray([1.0, np.nan, 3.0], dims="time")
    assert float(da.mean()) == 2.0
    assert np.isnan(float(da.mean(skipna=False)))


def test_grouped_mutate_broadcasts_back_through_labels():
    da = cloud()
    ring = da.sel(feature="ring", drop=True).rename("ring")
    intensity = da.sel(feature="intensity", drop=True)
    centered = intensity.groupby(ring) - intensity.groupby(ring).mean()
    assert centered.dims == ("point",)
    assert centered.values.tolist() == pytest.approx([2.5, 0.0, -2.5])


# ── §6.2 Dot ────────────────────────────────────────────────────────────────


def test_dot_over_named_axis_keeps_batch_axes():
    R = xr.DataArray(np.eye(3), dims=("i", "feature"), coords={"feature": ["x", "y", "z"]})
    v = xr.DataArray(np.ones((4, 3)), dims=("time", "feature"), coords={"feature": ["x", "y", "z"]})
    out = xr.dot(R, v, dim="feature")
    assert set(out.dims) == {"i", "time"}


def test_dot_does_not_conjugate():
    a = xr.DataArray([1j, 0], dims="s")
    b = xr.DataArray([1j, 0], dims="s")
    assert complex(xr.dot(a, b, dim="s")) == -1
    assert complex(xr.dot(np.conj(a), b, dim="s")) == 1


def test_gate_example_from_plan():
    # §8: the complex-array acceptance example.
    amplitude = xr.DataArray([1 + 0j, 0j], dims="input", coords={"input": [0, 1]})
    gate = xr.DataArray(
        [[0, 1], [1, 0]], dims=("output", "input"),
        coords={"output": [0, 1], "input": [0, 1]},
    )
    out = xr.dot(gate, amplitude, dim="input")
    assert out.dims == ("output",) and out.values.tolist() == [0, 1]


# ── §4.1 Memory-mapping ─────────────────────────────────────────────────────


def test_memmap_wrapped_in_dataarray_is_not_copied(tmp_path):
    path = tmp_path / "raw.bin"
    np.arange(10, dtype="<f4").tofile(path)
    mm = np.memmap(path, dtype="<f4", mode="r").reshape(5, 2)
    da = xr.DataArray(mm, dims=("point", "feature"))
    assert np.shares_memory(da.data, mm)
