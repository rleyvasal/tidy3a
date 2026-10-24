import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    AlignError,
    FilterError,
    GroupError,
    MutateError,
    RefError,
    SelectError,
    SliceError,
    SummariseError,
    all,
    all_of,
    any,
    any_of,
    assign,
    bind_cols,
    bins,
    col,
    contains,
    ends_with,
    env,
    everything,
    fields,
    filter,
    full_join,
    group_by,
    if_else,
    infer,
    inner_join,
    is_numeric,
    label,
    last_col,
    left_join,
    matches,
    max,
    mean,
    min,
    mutate,
    n,
    nan,
    pull,
    read_array,
    select,
    slice,
    slice_head,
    slice_tail,
    starts_with,
    sum,
    summarise,
    tidy,
    ungroup,
    var,
    where,
)

x, y, z, intensity, ring, point, feature = fields("x", "y", "z", "intensity", "ring", "point", "feature")
FIELDS = ["x", "y", "z", "intensity", "ring"]


def cloud(**options):
    values = np.array(
        [
            [0.0, -48.0, 0.0, 10.0, 1.0],
            [0.0, -49.0, 0.0, 20.0, 2.0],
            [51.0, 1.0, 0.0, 30.0, 1.0],
            [0.0, 1.0, 3.0, 40.0, 2.0],
        ],
        dtype="float32",
    )
    return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, **options)


def field_cloud():
    return cloud(field_axis="feature")


def sensors():
    values = np.array(
        [
            [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[3.0, 2.0, 0.0], [30.0, 2.0, 0.0]],
            [[5.0, 4.0, 1.0], [50.0, 4.0, 1.0]],
            [[7.0, 6.0, 1.0], [70.0, 6.0, 1.0]],
        ]
    )
    da = xr.DataArray(
        values,
        dims=("time", "sensor", "feature"),
        coords={"time": [1, 2, 3, 4], "sensor": ["front", "rear"], "feature": ["x", "y", "z"]},
    )
    return tidy(da, field_axis="feature")


def data(frame, name="values"):
    return frame.collect()[name]


# ── select ──────────────────────────────────────────────────────────────────


def test_select_field_labels_keeps_one_array():
    out = data(field_cloud() >> select(x, y, z))
    assert out.dims == ("point", "feature") and out.feature.values.tolist() == ["x", "y", "z"]


def test_select_with_strings_matches_references():
    a = data(field_cloud() >> select("x", "y", "z"))
    b = data(field_cloud() >> select(x, y, z))
    assert a.identical(b)


def test_select_does_not_change_its_source():
    source = field_cloud()
    source >> select(x)
    assert data(source).sizes["feature"] == 5


def test_select_exclusions_and_helpers():
    assert data(field_cloud() >> select(~intensity, ~ring)).feature.values.tolist() == ["x", "y", "z"]
    assert data(field_cloud() >> select(-intensity)).feature.values.tolist() == ["x", "y", "z", "ring"]
    assert data(field_cloud() >> select(starts_with("in"))).feature.values.tolist() == ["intensity"]
    assert data(field_cloud() >> select(ends_with("ng"))).feature.values.tolist() == ["ring"]
    assert data(field_cloud() >> select(contains("t"))).feature.values.tolist() == ["intensity"]
    assert data(field_cloud() >> select(matches("^[xyz]$"))).feature.values.tolist() == ["x", "y", "z"]
    assert data(field_cloud() >> select(all_of(["z", "x"]))).feature.values.tolist() == ["z", "x"]
    assert data(field_cloud() >> select(any_of(["z", "w"]))).feature.values.tolist() == ["z"]
    assert data(field_cloud() >> select(last_col())).feature.values.tolist() == ["ring"]
    assert list((field_cloud() >> select(everything())).collect().data_vars) == ["values"]


def test_select_variables_after_variables_from():
    frame = cloud(variables_from="feature") >> select(x, y, z)
    assert list(frame.collect().data_vars) == ["x", "y", "z"]


def test_select_where_predicate_and_numpy_style_where():
    ds = xr.Dataset({"a": ("t", [1.0, 2.0]), "s": ("t", ["u", "v"])})
    assert list((tidy(ds) >> select(where(is_numeric))).collect().data_vars) == ["a"]
    with pytest.raises(TypeError, match="use if_else"):
        where(x > 1, x, 0)


def test_select_observation_label_points_to_filter():
    with pytest.raises(SelectError, match=r'filter\(sensor == \'front\'\)'):
        sensors() >> select("front")


def test_select_unknown_name_lists_columns():
    with pytest.raises(SelectError, match="labels on feature: x  y  z"):
        field_cloud() >> select("w")


def test_select_needs_a_field_axis_for_labels():
    with pytest.raises(SelectError, match="not a variable, field label, or axis"):
        cloud() >> select(x)


def test_select_label_applies_to_every_variable_with_the_axis():
    ds = xr.Dataset(
        {
            "position": (("time", "feature"), np.zeros((2, 3))),
            "velocity": (("time", "feature"), np.ones((2, 3))),
            "gain": ("time", [1.0, 2.0]),
        },
        coords={"feature": ["x", "y", "z"]},
    )
    out = (tidy(ds, field_axis="feature") >> select(x)).collect()
    assert list(out.data_vars) == ["position", "velocity"]
    assert out.sizes["feature"] == 1


def test_variable_and_label_collision_needs_var_or_label():
    frame = field_cloud() >> mutate(assign("xx", 1.0))
    ds = field_cloud().to_xarray().assign(x=("point", [9.0, 9.0, 9.0, 9.0]))
    frame = tidy(ds, field_axis="feature")
    with pytest.raises(SelectError, match=r'var\("x"\)'):
        frame >> select(x)
    assert list((frame >> select(var("x"))).collect().data_vars) == ["x"]
    assert data(frame >> select(label("x"))).feature.values.tolist() == ["x"]
    out = frame >> mutate(d = var("x") - label("x"))
    assert data(out, "d").values.tolist() == [9.0, 9.0, -42.0, 9.0]


# ── expressions ─────────────────────────────────────────────────────────────


def test_bare_field_label_drops_the_field_axis():
    out = data(field_cloud() >> mutate(r = x * 2), "r")
    assert out.dims == ("point",) and out.values.tolist() == [0.0, 0.0, 102.0, 0.0]


def test_vector_form_keeps_the_field_axis():
    out = data(field_cloud() >> mutate(s = sum([x, y, z], over="feature")), "s")
    assert out.dims == ("point",) and out.values.tolist() == [-48.0, -49.0, 52.0, 4.0]


def test_vector_view_cannot_share_the_field_axis_when_stored():
    with pytest.raises(MutateError, match="has 3 of the 5 labels on feature"):
        field_cloud() >> mutate(v = [x, y, z])


def test_vector_form_rebuilds_variables_from_axis():
    out = data(cloud(variables_from="feature") >> mutate(v = [x, y, z]), "v")
    assert out.dims == ("feature", "point")
    assert out.feature.values.tolist() == ["x", "y", "z"]


def test_vector_of_unrelated_variables_suggests_pack():
    ds = xr.Dataset({"vx": ("t", [1.0]), "vy": ("t", [2.0])})
    vx, vy = fields("vx", "vy")
    with pytest.raises(Exception, match="pack"):
        tidy(ds) >> mutate(v = [vx, vy])


def test_mixed_vector_is_an_error():
    with pytest.raises(Exception, match="mixes data references and plain values"):
        field_cloud() >> mutate(v = [x, 0.0, z])


def test_expressions_have_no_truth_value():
    with pytest.raises(TypeError, match="combine conditions with &"):
        bool(x > 1)


def test_env_reaches_python_values():
    threshold = 15.0  # noqa: F841 - read by env("threshold")
    out = field_cloud() >> filter(intensity > env("threshold"))
    assert data(out).sizes["point"] == 3
    with pytest.raises(RefError, match="no Python value"):
        env("not_defined_anywhere")


def test_numpy_arrays_need_axis_names():
    with pytest.raises(Exception, match="without axis names"):
        field_cloud() >> mutate(r = x + np.array([1.0, 2.0, 3.0, 4.0]))


def test_reductions_fall_back_to_builtins_for_plain_values():
    assert sum([1, 2, 3]) == 6 and min(3, 1) == 1 and max([2, 5]) == 5
    assert any([False, True]) and all([True, True])


# ── filter ──────────────────────────────────────────────────────────────────


def test_filter_keeps_point_associated_values_and_original_positions():
    out = field_cloud() >> filter(intensity >= 20)
    values = data(out)
    assert values.point.values.tolist() == [2, 3, 4]
    assert values.sel(feature="ring").values.tolist() == [2.0, 1.0, 2.0]
    assert (out >> pull(point)).tolist() == [2, 3, 4]


def test_filter_on_observation_labels_and_two_axes():
    out = data(sensors() >> filter(col("sensor") == "front", col("time") > 2))
    assert dict(out.sizes) == {"time": 2, "sensor": 1, "feature": 3}
    same = data(sensors() >> filter((col("sensor") == "front") & (col("time") > 2)))
    assert out.identical(same)


def test_filter_or_across_axes_is_a_multi_axis_error():
    with pytest.raises(FilterError, match="varies along"):
        sensors() >> filter((col("sensor") == "front") | (col("time") > 2))


def test_filter_multi_axis_predicate_hints():
    with pytest.raises(FilterError) as info:
        sensors() >> filter(x > 20)
    message = str(info.value)
    assert 'filter(any(x > 20, over="sensor"))' in message
    assert "if_else(x > 20" in message


def test_filter_any_over_reduces_to_one_axis():
    out = data(sensors() >> filter(any(x > 20, over="sensor")))
    assert out.time.values.tolist() == [2, 3, 4]


def test_filter_empty_result_is_returned_and_flagged():
    out = field_cloud() >> filter(intensity > 999)
    assert data(out).sizes["point"] == 0
    assert "changed: point: 0 (all removed by filter(intensity > 999))" in repr(out)


def test_filter_needs_a_condition():
    with pytest.raises(FilterError, match="not true/false"):
        field_cloud() >> filter(intensity)
    with pytest.raises(FilterError, match="field axis"):
        field_cloud() >> filter(feature == "x")


def test_strict_style_filter_on_field_axis_without_field_axis():
    out = data(cloud() >> filter(feature.is_in(["x", "y", "z"])))
    assert out.feature.values.tolist() == ["x", "y", "z"]


def test_grouped_filter_compares_with_group_summaries():
    out = field_cloud() >> group_by(ring) >> filter(intensity > mean(intensity))
    assert data(out).point.values.tolist() == [3, 4]


# ── slice ───────────────────────────────────────────────────────────────────


def test_slice_is_one_based_and_keeps_the_axis():
    out = data(sensors() >> slice(time = 1))
    assert out.sizes["time"] == 1 and out.time.values.tolist() == [1]
    assert "time" not in data(sensors() >> slice(time = 1, drop=True)).dims


def test_slice_negative_excludes_and_keeps_positions():
    out = field_cloud() >> slice(point = -1)
    assert (out >> pull(point)).tolist() == [2, 3, 4]
    assert (field_cloud() >> slice(point = range(1, 3)) >> pull(point)).tolist() == [1, 2]


def test_slice_errors():
    with pytest.raises(SliceError, match="positions start at 1"):
        field_cloud() >> slice(point = 0)
    with pytest.raises(SliceError, match="all positive"):
        field_cloud() >> slice(point = [1, -2])
    with pytest.raises(SliceError, match="field axis"):
        field_cloud() >> slice(feature = 1)
    with pytest.raises(SliceError, match="one position remains"):
        field_cloud() >> slice(point = [1, 2], drop=True)


def test_slice_head_and_tail():
    assert (field_cloud() >> slice_head(point = 2) >> pull(point)).tolist() == [1, 2]
    assert (field_cloud() >> slice_tail(point = 1) >> pull(point)).tolist() == [4]


# ── mutate ──────────────────────────────────────────────────────────────────


def test_mutate_is_sequential():
    out = (field_cloud() >> mutate(a = intensity * 2, b = col("a") + 1)).collect()
    assert out["b"].values.tolist() == [21.0, 41.0, 61.0, 81.0]


def test_mutate_writes_back_into_field_label():
    out = field_cloud() >> mutate(x = x + 1.0)
    values = data(out)
    assert values.dtype == np.float32
    assert values.sel(feature="x").values.tolist() == [1.0, 1.0, 52.0, 1.0]
    assert values.sel(feature="intensity").values.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_write_back_promotes_and_reports():
    out = field_cloud() >> mutate(x = x + np.float64(0.5))
    assert data(out).dtype == np.float64
    assert "promoted float32 → float64 by mutate(x = …)" in repr(out)


def test_write_back_needs_the_field_shape():
    with pytest.raises(MutateError, match="must keep its shape"):
        sensors() >> mutate(x = mean(x, over="time"))


def test_write_back_on_memory_mapped_data_is_reported(tmp_path):
    path = tmp_path / "cloud.bin"
    field_cloud().collect()["values"].values.tofile(path)
    frame = read_array(path, value_type="float32", dims={"point": infer, "feature": 5},
                       labels={"feature": FIELDS}, field_axis="feature", lazy=True)
    out = frame >> mutate(x = x + 1.0)
    assert "copied into memory to write x" in repr(out)
    assert data(frame).sel(feature="x").values.tolist() == [0.0, 0.0, 51.0, 0.0]


def test_vector_assignment_maps_its_axis_by_label():
    vector = xr.DataArray(np.ones((4, 3)), dims=("point", "i"), coords={"i": ["x", "y", "z"]})
    out = data(field_cloud() >> mutate(assign([x, y, z], col("intensity") * vector)))
    assert out.sel(feature="y").values.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert out.sel(feature="ring").values.tolist() == [1.0, 2.0, 1.0, 2.0]


def test_if_else_masks_and_keeps_shape():
    out = data(field_cloud() >> mutate(m = if_else(intensity > 15, intensity, nan)), "m")
    assert np.isnan(out.values[0]) and out.values[1:].tolist() == [20.0, 30.0, 40.0]


def test_mutate_constant_and_axis_names():
    out = (field_cloud() >> mutate(k = 3)).collect()
    assert out["k"].ndim == 0
    with pytest.raises(MutateError, match="axis name"):
        field_cloud() >> mutate(point = 1)


# ── grouping and summaries ──────────────────────────────────────────────────


def test_time_series_mean_per_sensor():
    out = data(sensors() >> group_by("sensor") >> summarise(mean_x = mean(x)), "mean_x")
    assert out.dims == ("sensor",) and out.values.tolist() == [4.0, 40.0]


def test_summary_keeps_the_field_axis():
    out = data(sensors() >> group_by("sensor") >> summarise(m = mean(col("values"))), "m")
    assert set(out.dims) == {"sensor", "feature"}
    assert out.sel(sensor="front").values.tolist() == [4.0, 3.0, 0.5]


def test_ungrouped_summary_reduces_every_observation_axis():
    out = data(sensors() >> summarise(m = mean(col("values"))), "m")
    assert out.dims == ("feature",)


def test_group_by_value_field():
    out = (field_cloud() >> group_by(ring) >> summarise(m = mean(intensity), k = n())).collect()
    assert out.ring.values.tolist() == [1.0, 2.0]
    assert out["m"].values.tolist() == [20.0, 30.0] and out["k"].values.tolist() == [2, 2]


def test_group_by_bins_and_empty_bins():
    out = (field_cloud() >> group_by(bins(intensity, [0, 15, 35, 100, 200]))
           >> summarise(k = n(), s = sum(intensity), m = mean(intensity))).collect()
    assert out.intensity_bin.values.tolist() == ["[0, 15)", "[15, 35)", "[35, 100)", "[100, 200)"]
    assert out["k"].values.tolist() == [1, 2, 1, 0]
    assert out["s"].values.tolist() == [10.0, 50.0, 40.0, 0.0]
    assert np.isnan(out["m"].values[-1])


def test_bins_never_drop_values_silently():
    with pytest.raises(GroupError, match="fall outside every bin"):
        field_cloud() >> group_by(bins(intensity, [0, 25]))


def test_bins_by_duration():
    da = xr.DataArray(
        np.arange(4.0), dims="time",
        coords={"time": np.array(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"], dtype="datetime64[ns]")},
    )
    out = (tidy(da) >> group_by(bins(col("time"), "2 days")) >> summarise(m = mean(col("values")))).collect()
    assert out.time_bin.values.tolist() == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)"]
    assert out["m"].values.tolist() == [0.5, 2.5]


def test_variable_without_grouping_axis_is_an_error():
    ds = xr.Dataset(
        {"intensity": (("time", "sensor"), np.ones((2, 2))), "temperature": ("time", [1.0, 2.0])},
        coords={"sensor": ["front", "rear"]},
    )
    with pytest.raises(SummariseError, match="temperature has no sensor axis"):
        tidy(ds) >> group_by("sensor") >> summarise(t = mean(col("temperature")))


def test_summary_must_reduce():
    with pytest.raises(SummariseError, match="still varies along time"):
        sensors() >> group_by("sensor") >> summarise(v = x)


def test_over_must_agree_with_grouping():
    with pytest.raises(SummariseError, match="group_by keeps"):
        sensors() >> group_by("sensor") >> summarise(m = mean(x, over="sensor"))


def test_missing_values_propagate_unless_na_rm():
    da = xr.DataArray([[1.0, np.nan], [3.0, 4.0]], dims=("time", "sensor"), coords={"sensor": ["a", "b"]})
    out = (tidy(da) >> group_by("sensor") >> summarise(m = mean(col("values")),
           r = mean(col("values"), na_rm=True))).collect()
    assert out["m"].values[0] == 2.0 and np.isnan(out["m"].values[1])
    assert out["r"].values.tolist() == [2.0, 4.0]


def test_groups_after_summarise():
    grouped = sensors() >> group_by("time", "sensor")
    assert [k.label for k in (grouped >> summarise(m = mean(x)))._groups] == ["time"]
    assert (grouped >> summarise(m = mean(x), groups="drop"))._groups == ()
    assert len((grouped >> summarise(m = mean(x), groups="keep"))._groups) == 2
    assert (grouped >> ungroup())._groups == ()


def test_grouped_mutate_broadcasts_back():
    out = data(field_cloud() >> group_by(ring) >> mutate(c = intensity - mean(intensity)), "c")
    assert out.dims == ("point",) and out.values.tolist() == [-10.0, -10.0, 10.0, 10.0]
    counts = data(field_cloud() >> group_by(ring) >> mutate(k = n()), "k")
    assert counts.values.tolist() == [2, 2, 2, 2]


def test_group_by_rules():
    with pytest.raises(GroupError, match="field axis"):
        field_cloud() >> group_by(feature)
    with pytest.raises(GroupError, match="varies along one axis"):
        sensors() >> group_by(x)


# ── alignment ───────────────────────────────────────────────────────────────


def test_exact_alignment_reports_mismatch_and_suggests_joins():
    gain = xr.DataArray([1.0, 2.0, 3.0], dims="sensor", coords={"sensor": ["front", "rear", "side"]})
    with pytest.raises(AlignError, match="inner_join"):
        sensors() >> mutate(g = x * gain)


def test_reordered_labels_are_matched_and_reported():
    gain = xr.DataArray([2.0, 1.0], dims="sensor", coords={"sensor": ["rear", "front"]})
    out = sensors() >> mutate(g = x * gain)
    assert data(out, "g").sel(sensor="rear").values.tolist() == [20.0, 60.0, 100.0, 140.0]
    assert "sensor reordered to match" in repr(out)


def test_labelled_and_unlabelled_axes_never_pair_by_length():
    scores = xr.DataArray([1.0, 2.0, 3.0], dims="point")
    nearby = field_cloud() >> filter(intensity >= 20)
    with pytest.raises(AlignError, match="bind_cols"):
        nearby >> mutate(s = intensity * scores)


def test_bind_cols_attaches_by_position_with_labels():
    nearby = field_cloud() >> filter(intensity >= 20)
    out = nearby >> bind_cols(scores=np.array([0.1, 0.2, 0.3]), along="point")
    scores = data(out, "scores")
    assert scores.point.values.tolist() == [2, 3, 4]
    with pytest.raises(AlignError, match="equal lengths"):
        nearby >> bind_cols(scores=np.array([0.1]), along="point")
    with pytest.raises(AlignError, match="needs a name"):
        nearby >> bind_cols(np.array([0.1, 0.2, 0.3]), along="point")


def test_joins_and_suffixes():
    left = tidy(xr.DataArray([1.0, 2.0, 3.0], dims="point", coords={"point": [1, 2, 3]}))
    right = tidy(xr.DataArray([20.0, 30.0, 40.0], dims="point", coords={"point": [2, 3, 4]}))
    out = (left >> left_join(right)).collect()
    assert list(out.data_vars) == ["values_x", "values_y"]
    assert out.point.values.tolist() == [1, 2, 3] and np.isnan(out["values_y"].values[0])
    assert (left >> inner_join(right)).collect().point.values.tolist() == [2, 3]
    assert (left >> full_join(right)).collect().point.values.tolist() == [1, 2, 3, 4]
    named = (left >> left_join(right, suffix=("", "_full"))).collect()
    assert list(named.data_vars) == ["values", "values_full"]


def test_left_join_tolerance():
    left = tidy(xr.DataArray([1.0, 2.0], dims="lon", coords={"lon": [0.1, 0.2]}))
    right = tidy(xr.DataArray([5.0, 6.0], dims="lon", coords={"lon": [0.1 + 1e-12, 0.2 - 1e-12]}).rename("b"))
    with pytest.raises(AlignError, match="rounding"):
        left >> mutate(c = col("values") + right.to_xarray()["b"])
    out = (left >> left_join(right, tolerance=1e-9)).collect()
    assert out["b"].values.tolist() == [5.0, 6.0]


def test_pull():
    assert (field_cloud() >> pull(intensity)).tolist() == [10.0, 20.0, 30.0, 40.0]
    assert (field_cloud() >> pull(point)).tolist() == [1, 2, 3, 4]
    assert (sensors() >> pull("sensor")).tolist() == ["front", "rear"]


def test_verbs_need_tidy_data():
    with pytest.raises(TypeError, match="wrap xarray objects with tidy"):
        select(x)(xr.Dataset())
