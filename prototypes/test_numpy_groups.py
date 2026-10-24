"""Cache lifecycle, ownership, numerical parity, and many-small-group execution."""

import numpy as np
import pytest
import xarray as xr

import numpy_core as core
import numpy_pipeline as proto
import tidy3a as ta
from tidy3a.expr import reduce_array
from test_numpy_pipeline import assert_same, frame, records, ring, x, y, z


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "int16", "bool"])
@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "var", "median", "any", "all"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_batched_groups_match_individual_numpy_reductions(axis, dtype, op, na_rm):
    rng = np.random.default_rng(37)
    keys = np.repeat(np.arange(8), [1, 2, 4, 7, 16, 17, 33, 100])
    rng.shuffle(keys)
    raw = rng.choice([-1e8, 1e8, 1, 3, 1e-5], size=(len(keys), 2, 3))
    if dtype in ("int16", "bool"):
        raw = rng.integers(0, 8, size=raw.shape)
    raw = raw.astype(dtype)
    if dtype.startswith("complex"):
        raw += raw * np.complex64(1j)
    if raw.dtype.kind in "fc":
        raw[3, 0, 0] = np.nan
    raw = np.moveaxis(raw, 0, axis)
    dims = list(("sensor", "feature"))
    dims.insert(axis, "point")
    value = core.labelled(raw, dims)
    groups = core.GroupIndex.build(core.labelled(keys, ("point",)), "key")
    actual = groups.reduce(value, op, na_rm)
    expected = []
    for key in np.unique(keys):
        part = xr.DataArray(np.take(raw, np.flatnonzero(keys == key), axis=axis), dims=dims)
        expected.append(reduce_array(part, op, ["point"], na_rm).values)
    np.testing.assert_array_equal(actual.values, np.stack(expected))
    assert actual.dtype == np.stack(expected).dtype


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("op", ["sum", "mean", "std"])
def test_grouped_reduction_of_additional_axes(axis, op):
    keys = np.array([1, 0, 1, 0, 1])
    raw = np.arange(30, dtype="float32").reshape(5, 2, 3)
    raw = np.moveaxis(raw, 0, axis)
    dims = ["sensor", "feature"]
    dims.insert(axis, "point")
    value = core.labelled(raw, dims)
    groups = core.GroupIndex.build(core.labelled(keys, ("point",)), "key")
    result = groups.reduce(value, op, over=("sensor",))
    expected = [reduce_array(xr.DataArray(np.take(raw, np.flatnonzero(keys == k), axis=axis), dims=dims),
                             op, ["point", "sensor"]).values for k in [0, 1]]
    np.testing.assert_array_equal(result.values, expected)
    assert result.dims == ("key", "feature")


def test_mostly_unique_timestamps_use_batched_calls(monkeypatch):
    n = 100_000
    times = np.arange(n).astype("datetime64[ns]")
    times[1::10] = times[::10]
    values = np.arange(n, dtype="float32")
    calls = []
    original = core._reduce_data

    def track(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_reduce_data", track)
    groups = core.GroupIndex.build(core.labelled(times, ("point",)), "time")
    result = groups.reduce(core.labelled(values, ("point",)), "mean")
    labels, codes = np.unique(times, return_inverse=True)
    expected = np.bincount(codes, weights=values) / np.bincount(codes)
    np.testing.assert_array_equal(result.values, expected.astype("float32"))
    np.testing.assert_array_equal(result.coords["time"], labels)
    assert len(calls) <= 3  # dtype probe plus one batch per group size, not 90,000 calls.
    assert groups.nbytes < 64 * n


def source():
    return xr.Dataset({"v": ("point", np.array([1, 3, 10, 14], dtype="float32")),
                       "key": ("point", np.array([0, 0, 1, 1], dtype="int16"))},
                      coords={"point": [11, 12, 13, 14]})


def track_factorization(monkeypatch):
    calls = []
    original = core._factorize

    def track(raw, **kwargs):
        assert kwargs.get("use_pandas") is False
        calls.append(raw.copy())
        return original(raw, **kwargs)

    monkeypatch.setattr(core, "_factorize", track)
    return calls


def test_cache_reused_across_verbs_and_invalidated_by_key_changes(monkeypatch):
    calls = track_factorization(monkeypatch)
    grouped = proto.tidy(source()) >> ta.group_by("key")
    cached = grouped._group_cache
    centered = grouped >> ta.mutate(center=ta.col("v") - ta.mean(ta.col("v")), size=ta.n())
    assert centered._group_cache is cached
    np.testing.assert_array_equal(centered >> ta.pull("center"), [-1, 1, -2, 2])
    np.testing.assert_array_equal(centered >> ta.pull("size"), [2, 2, 2, 2])
    selected = centered >> ta.select("key", "v", "center")
    assert selected._group_cache is cached
    summary = selected >> ta.summarise(m=ta.mean(ta.col("v")), s=ta.sum(ta.col("v")), count=ta.n())
    np.testing.assert_array_equal(summary.collect().m, [2, 12])
    assert len(calls) == 1
    changed = selected >> ta.mutate(key=ta.col("key") * 0, m=ta.mean(ta.col("v")))
    np.testing.assert_array_equal(changed >> ta.pull("m"), [7, 7, 7, 7])
    assert len(calls) == 2 and changed._group_cache is not cached
    # Branching leaves the original grouping reusable.
    grouped >> ta.summarise(m=ta.mean(ta.col("v")))
    assert len(calls) == 2


def test_filter_and_reordering_invalidate_cached_positions(monkeypatch):
    calls = track_factorization(monkeypatch)
    grouped = proto.tidy(source()) >> ta.group_by("key")
    kept = grouped >> ta.filter(ta.col("v") > ta.mean(ta.col("v")))
    assert kept._group_cache is None
    summary = kept >> ta.summarise(m=ta.mean(ta.col("v")), count=ta.n())
    np.testing.assert_array_equal(summary.collect().m, [3, 14])
    np.testing.assert_array_equal(summary.collect()["count"], [1, 1])
    assert len(calls) == 2
    # The collection's common slicing boundary also rejects stale permutations.
    reordered = grouped._evolve(grouped._data.isel(point=[3, 1, 0, 2]))
    assert reordered._group_cache is None
    out = reordered >> ta.mutate(m=ta.mean(ta.col("v")))
    np.testing.assert_array_equal(out >> ta.pull("m"), [12, 2, 2, 12])
    assert len(calls) == 3


def test_expression_grouping_tracks_dependencies_and_snapshots_literals(monkeypatch):
    calls = track_factorization(monkeypatch)
    offset = xr.DataArray([0, 0, 0, 0], dims="point", coords={"point": [11, 12, 13, 14]})
    grouped = proto.tidy(source()) >> ta.group_by(bucket=ta.col("key") + offset)
    offset.values[:] = 100  # captured literals do not invalidate a grouped frame externally.
    unrelated = grouped >> ta.mutate(other=ta.col("v") * 2)
    assert unrelated._group_cache is grouped._group_cache
    changed = unrelated >> ta.mutate(key=ta.col("key") + 2)
    out = changed >> ta.summarise(m=ta.mean(ta.col("v")))
    np.testing.assert_array_equal(out.collect().bucket, [2, 3])
    assert len(calls) == 2


def test_owned_inputs_and_outputs_cannot_silently_change_cached_keys():
    original = source()
    grouped = proto.tidy(original) >> ta.group_by("key")
    original.key.values[:] = 9
    original = original.assign_coords(point=[4, 3, 2, 1])
    assert original.key.values.flags.writeable
    for values in [grouped >> ta.pull("key"), grouped >> ta.pull("point"),
                   grouped.collect().key.values, grouped._group_cache.index.codes]:
        with pytest.raises(ValueError):
            values[0] = 99
    out = grouped >> ta.summarise(m=ta.mean(ta.col("v")))
    np.testing.assert_array_equal(out.collect().key, [0, 1])
    np.testing.assert_array_equal(grouped >> ta.pull("point"), [11, 12, 13, 14])


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "var", "median", "any", "all"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_grouped_mutate_and_filter_match_existing_tidy3a(op, na_rm):
    data = source()
    data.v.values[0] = np.nan
    reduction = getattr(ta, op)(ta.col("v"), na_rm=na_rm)
    verbs = [ta.group_by("key"), ta.mutate(result=reduction), ta.filter(ta.col("v") > 2),
             ta.summarise(result=reduction, size=ta.n())]
    actual, expected = proto.tidy(data), ta.tidy(data)
    for verb in verbs:
        actual, expected = actual >> verb, expected >> verb
        assert_same(actual.collect(), expected.collect())


@pytest.mark.parametrize("count", [1, 2, 3])
def test_grouped_sum_min_count(count):
    data = source()
    data.v.values[0] = np.nan
    verb = ta.summarise(total=ta.sum(ta.col("v"), na_rm=True, min_count=count))
    actual = proto.tidy(data) >> ta.group_by("key") >> verb
    expected = ta.tidy(data) >> ta.group_by("key") >> verb
    assert_same(actual.collect(), expected.collect())


def test_field_key_writeback_rebuilds_groups_and_keeps_original_frame(monkeypatch):
    calls = track_factorization(monkeypatch)
    original = frame(records("boundary"), proto.read_array) >> ta.group_by(ring)
    changed = original >> ta.mutate(ring=ring * 0) >> ta.summarise(count=ta.n())
    np.testing.assert_array_equal(changed.collect()["count"], [4])
    np.testing.assert_array_equal(original >> ta.pull(ring), [1, 2, 1, 2])
    assert len(calls) == 2


@pytest.mark.parametrize("verb", [ta.mutate(x=x / 2), ta.mutate(x=1.5)])
def test_field_writeback_matches_tidy3a(verb):
    actual = frame(records("boundary"), proto.read_array) >> verb
    expected = frame(records("boundary"), ta.read_array) >> verb
    assert_same(actual.collect(), expected.collect())


def test_vector_writeback_and_one_based_slicing():
    original = frame(records("boundary"), proto.read_array)
    expected = frame(records("boundary"), ta.read_array)
    verb = ta.mutate(ta.assign([x, y, z], ta.dot(
        xr.DataArray(np.eye(3, dtype="float32") * 2, dims=("component", "feature"),
                     coords={"component": ["x", "y", "z"], "feature": ["x", "y", "z"]}),
        [x, y, z], over="feature")))
    for step in [verb, ta.slice(point=[4, 1, 4]), ta.slice_tail(point=2)]:
        original, expected = original >> step, expected >> step
        assert_same(original.collect(), expected.collect())


def test_empty_grouped_mutation_and_summary():
    original = proto.tidy(source()) >> ta.group_by("key") >> ta.filter(ta.col("v") > 100)
    actual = original >> ta.mutate(center=ta.col("v") - ta.mean(ta.col("v")), size=ta.n())
    assert actual.collect().sizes["point"] == 0
    out = actual >> ta.summarise(m=ta.mean(ta.col("v")), count=ta.n())
    assert out.collect().sizes["key"] == 0


@pytest.mark.parametrize("op", ["min", "max", "mean", "sum", "any", "all"])
def test_empty_grouped_reduction_matches_existing_dtype(op):
    data = source().isel(point=slice(0, 0))
    verb = ta.summarise(result=getattr(ta, op)(ta.col("v")))
    assert_same((proto.tidy(data) >> ta.group_by("key") >> verb).collect(),
                (ta.tidy(data) >> ta.group_by("key") >> verb).collect())


@pytest.mark.parametrize("op", ["min", "mean", "sum"])
def test_groups_with_empty_additional_axis(op):
    data = xr.Dataset({"v": (("point", "sensor"), np.empty((3, 0), dtype="int16")),
                       "key": ("point", [0, 1, 0])})
    verb = ta.summarise(result=getattr(ta, op)(ta.col("v")), count=ta.n())
    assert_same((proto.tidy(data) >> ta.group_by("key") >> verb).collect(),
                (ta.tidy(data) >> ta.group_by("key") >> verb).collect())


def test_one_shot_grouping_reorders_key_to_data_before_summing():
    raw = np.array([1e8, 1, -1e8, 3], dtype="float32")
    value = core.labelled(raw, ("point",), {"point": [1, 2, 3, 4]})
    key = core.labelled([0, 0, 0, 0], ("point",), {"point": [4, 3, 2, 1]})
    result = core.group_reduce(value, key, "sum", "key", use_pandas=False)
    np.testing.assert_array_equal(result.values, [raw.sum()])


def test_lazy_input_is_rejected_without_computing():
    da = pytest.importorskip("dask.array")
    from dask import delayed

    @delayed
    def forbidden():
        raise AssertionError("lazy data was read")

    source = xr.DataArray(da.from_delayed(forbidden(), shape=(3,), dtype="float32"), dims="point")
    with pytest.raises(NotImplementedError, match="eager NumPy input"):
        proto.tidy(source)
    with pytest.raises(NotImplementedError, match="eager input"):
        proto.read_array(np.zeros(3), dims=["point"], lazy=True)


def test_pack_masking_and_axis_rename_use_existing_expressions():
    data = source()
    verbs = [ta.mutate(masked=ta.if_else(ta.col("key").is_in([0]), ta.col("v"), 0),
                       vector=ta.pack([ta.col("v"), ta.col("key")], along="component")),
             ta.mutate(renamed=ta.rename_axes(ta.col("vector"), component="channel"))]
    actual, expected = proto.tidy(data), ta.tidy(data)
    for verb in verbs:
        actual, expected = actual >> verb, expected >> verb
        assert_same(actual.collect(), expected.collect())


def test_origin_vectors_use_existing_label_dtypes():
    kwargs = {"dims": {"point": ta.infer, "feature": 5},
              "labels": {"feature": ["x", "y", "z", "intensity", "ring"]},
              "variables_from": "feature"}
    actual = proto.read_array(records("boundary"), **kwargs) >> ta.mutate(vector=[x, y, z])
    expected = ta.read_array(records("boundary"), **kwargs) >> ta.mutate(vector=[x, y, z])
    assert_same(actual.collect(), expected.collect())
