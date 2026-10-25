"""Grouped reductions in one pass give the same bits as reducing each group on its own.

``tidy3a.grouped`` reduces every group at once where that is exact: min,
max, count, any, all, and integer sums in any order; float sums and means
of groups with fewer than 8 positions, which NumPy adds one after another
starting from 0.0. Other groups and reductions are reduced one by one, as
before. ``grouped.ONE_PASS = False`` turns the one-pass path off, giving the
reference these tests compare with.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    all,
    any,
    bins,
    col,
    expr,
    group_by,
    grouped,
    max,
    mean,
    median,
    min,
    mutate,
    n,
    std,
    sum,
    summarise,
    tidy,
)

REDUCERS = {"sum": sum, "mean": mean, "min": min, "max": max, "any": any, "all": all, "std": std, "median": median}


def _dataset(rng):
    groups = int(rng.integers(9, 80))
    sizes = rng.integers(1, 8, groups)
    if rng.random() < 0.4:  # some groups of 8 or more, reduced one by one
        sizes[rng.integers(groups, size=3)] = rng.integers(8, 40, 3)
    size = int(sizes.sum())
    key = rng.permutation(np.repeat(rng.permutation(groups) * 3, sizes))
    dtype = str(rng.choice(["float32", "float64", "int16", "int64", "bool"]))
    layout = str(rng.choice(["1d", "first", "last"]))
    shape = (size,) if layout == "1d" else (size, 3)
    values = rng.normal(size=shape) * 10 ** rng.uniform(-3, 8, shape)
    if dtype.startswith("float"):
        values[rng.random(shape) < 0.05] = -0.0
        values[rng.random(shape) < 0.05] = np.nan
        values[rng.random(shape) < 0.01] = np.inf
    elif dtype == "bool":
        values = values > 0
    values = values.astype(dtype)
    coords = {"point": np.arange(size) * 2}
    if layout == "1d":
        return xr.Dataset({"v": ("point", values), "key": ("point", key)}, coords=coords), None
    coords["feature"] = ["f1", "f2", "f3"]
    dims = ("point", "feature")
    if layout == "last":
        values, dims = np.ascontiguousarray(values.T), ("feature", "point")
    return xr.Dataset({"v": (dims, values), "key": ("point", key)}, coords=coords), "feature"


def _run(data, field_axis, op, na_rm, min_count, mode):
    options = {"na_rm": na_rm}
    if op == "sum" and min_count is not None:
        options["min_count"] = min_count
    value = REDUCERS[op](col("v"), **options)
    step = summarise(m=value, k=n()) if mode == "summarise" else mutate(c=value)
    return (tidy(data, field_axis=field_axis) >> group_by(col("key")) >> step).collect()


def _identical(fast, slow):
    xr.testing.assert_identical(fast, slow)
    for name in slow.variables:
        assert fast[name].dtype == slow[name].dtype, name
        if slow[name].dtype.kind != "O":  # object arrays hold pointers; assert_identical compared them
            assert fast[name].values.tobytes() == slow[name].values.tobytes(), name


@pytest.mark.parametrize("seed", range(200))
def test_one_pass_grouped_reductions_match_reducing_each_group(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    data, field_axis = _dataset(rng)
    op = str(rng.choice(list(REDUCERS)))
    na_rm, mode = bool(rng.random() < 0.5), str(rng.choice(["summarise", "mutate"]))
    min_count = int(rng.integers(1, 4)) if rng.random() < 0.3 else None
    fast = _run(data, field_axis, op, na_rm, min_count, mode)
    monkeypatch.setattr(grouped, "ONE_PASS", False)
    _identical(fast, _run(data, field_axis, op, na_rm, min_count, mode))


def test_bins_with_empty_groups_match_too(monkeypatch):
    rng = np.random.default_rng(0)
    data = xr.Dataset({"v": ("point", rng.normal(size=200))})
    edges = np.concatenate([np.linspace(-3, 0, 20), np.linspace(10, 12, 5)])  # the last bins stay empty

    def run():
        return (tidy(data) >> group_by(bins(col("v"), edges)) >> summarise(m=mean(col("v")), lo=min(col("v")),
                                                                         k=n())).collect()

    fast = run()
    monkeypatch.setattr(grouped, "ONE_PASS", False)
    _identical(fast, run())


def test_many_small_groups_are_not_reduced_one_by_one(monkeypatch):
    rng = np.random.default_rng(0)
    key = rng.permutation(np.repeat(np.arange(1000), rng.integers(1, 8, 1000)))
    data = xr.Dataset({"v": ("point", rng.normal(size=len(key))), "key": ("point", key)})
    frame = tidy(data) >> group_by(col("key"))
    backend = type(frame._backend)
    calls = []
    original = backend.reduce
    monkeypatch.setattr(backend, "reduce", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    frame >> summarise(m=mean(col("v")), s=sum(col("v")), hi=max(col("v")))
    assert len(calls) <= 3  # at most one, describing the result, per summary; not one per group


def test_group_by_finds_the_groups_once_for_the_next_verb(monkeypatch):
    data = xr.Dataset({"v": ("point", np.arange(12.0)), "key": ("point", np.arange(12) % 5)})
    found = []
    original = expr._factorize
    monkeypatch.setattr(expr, "_factorize", lambda raw: found.append(1) or original(raw))
    tidy(data) >> group_by(col("key")) >> summarise(m=mean(col("v")), s=sum(col("v")))
    assert len(found) == 1
    found.clear()
    tidy(data) >> group_by(col("key")) >> mutate(a=mean(col("v")), b=max(col("v")), c=col("v") * 2)
    assert len(found) == 1


def test_groups_are_found_again_after_a_step_changes_the_key():
    data = xr.Dataset({"v": ("point", np.arange(6.0)), "key": ("point", [0, 0, 1, 1, 2, 2])})
    out = (tidy(data) >> group_by(col("key")) >> mutate(key=col("key") * 0, m=mean(col("v")))).collect()
    assert out["m"].values.tolist() == [2.5] * 6  # one group once the key is all zeros
    cloud = xr.Dataset({"values": (("point", "feature"), np.array([[1.0, 0], [3, 0], [5, 1], [7, 1]]))},
                       coords={"feature": ["x", "ring"]})
    keep_x = xr.DataArray([1.0, 0.0], dims="feature", coords={"feature": ["x", "ring"]})
    out = (tidy(cloud, field_axis="feature") >> group_by(col("ring"))
           >> mutate(ring=col("ring") * 0, m=mean(col("x")))).collect()
    assert out["m"].values.tolist() == [4.0] * 4
    out = (tidy(cloud, field_axis="feature") >> group_by(col("ring"))
           >> mutate(values=col("values") * keep_x, m=mean(col("x")))).collect()
    assert out["m"].values.tolist() == [4.0] * 4  # the whole holder replaced: ring is all zeros now


# ── Costs that grow with the number of groups or points ─────────────────────


def _many(groups=1000, key_dtype="int64"):
    rng = np.random.default_rng(1)
    key = rng.permutation(np.repeat(np.arange(groups), 3)).astype(key_dtype)
    return xr.Dataset({"v": ("point", rng.normal(size=len(key))), "key": ("point", key)})


def test_group_sizes_are_counted_once_per_step(monkeypatch):
    data = _many()
    counted = []
    original = np.bincount
    monkeypatch.setattr(np, "bincount", lambda *a, **k: counted.append(1) or original(*a, **k))
    tidy(data) >> group_by(col("key")) >> summarise(m=sum(col("v")), k=n())
    assert len(counted) == 1  # once, for both the sort by group and n()


def test_group_labels_are_converted_in_one_call(monkeypatch):
    calls = []
    original = expr._plain
    monkeypatch.setattr(expr, "_plain", lambda value: calls.append(1) or original(value))
    tidy(_many()) >> group_by(col("key")) >> summarise(m=mean(col("v")))
    assert len(calls) < 10, len(calls)


@pytest.mark.parametrize("values", [
    np.array([3, 1, 3, 2]), np.array([0.5, -1.0, 0.5, 2.0], dtype="float32"), np.array([True, False, True, True]),
    np.array(["b", "a", "b", "c"], dtype=object), np.array(["2026-01-02", "2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
    np.array([5, 1, 5], dtype="timedelta64[ns]"), np.array([2, 1, 2], dtype="uint8"),
])
def test_group_labels_keep_their_python_types(values):
    data = xr.Dataset({"v": ("point", np.arange(len(values), dtype=float)), "key": ("point", values)})
    groups = expr.resolve_value_key(expr.ValueKey("key", col("key")), tidy(data), [])
    expected = [expr._plain(label) for label in np.unique(values)]
    assert [type(label) for label in groups.labels] == [type(label) for label in expected]
    assert groups.labels == expected


def test_broadcasting_group_results_back_gathers_once_without_np_take(monkeypatch):
    rng = np.random.default_rng(2)
    data = xr.Dataset({"values": (("point", "feature"), rng.normal(size=(40, 3)).astype("float32")),
                       "key": ("point", rng.integers(0, 12, 40))}, coords={"feature": ["f1", "f2", "f3"]})
    expected = (tidy(data, field_axis="feature") >> group_by(col("key"))
                >> mutate(c=col("values") - mean(col("values")))).collect()
    taken = []
    original = np.take
    monkeypatch.setattr(np, "take", lambda *a, **k: taken.append(1) or original(*a, **k))
    frame = tidy(data, field_axis="feature", backend="numpy") >> group_by(col("key"))
    groups = expr.Context(frame, verb="mutate").groups()
    stats = frame._backend.group_reduce(frame._backend.variable(frame._data, "values"), "mean", groups, [], False, None)
    taken.clear()
    frame._backend.broadcast(stats, groups)
    assert taken == []
    monkeypatch.setattr(np, "take", original)
    out = (frame >> mutate(c=col("values") - mean(col("values")))).collect()
    xr.testing.assert_identical(out, expected)
    assert out["c"].values.tobytes() == expected["c"].values.tobytes()


# ── The grouping object (milestone 10, step A2) ─────────────────────────────


def test_grouping_holds_plain_codes_and_labels_and_caches_its_derived_arrays():
    codes = np.array([2, 0, 2, 1, 0, 2])
    grouping = grouped.Grouping(codes, ["a", "b", "c"])
    assert isinstance(grouping.codes, np.ndarray) and grouping.codes.dtype.kind in "iu"
    assert grouping.labels == ["a", "b", "c"]
    assert len(grouping) == 3
    assert grouping.counts().tolist() == [2, 1, 3]
    order, starts, ends = grouping.blocks()
    assert order.tolist() == [1, 4, 3, 0, 2, 5]  # positions sorted by group, original order within a group
    assert starts.tolist() == [0, 2, 3] and ends.tolist() == [2, 3, 6]
    assert grouping.counts() is grouping.counts() and grouping.blocks() is grouping.blocks()
    empty = grouped.Grouping(np.array([], dtype=np.intp), [])
    assert len(empty) == 0 and empty.counts().tolist() == [] and empty.blocks()[0].tolist() == []


def test_grouping_needs_nothing_from_tidy3a():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(grouped))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {name for name in imported if name and name.startswith("tidy3a")}


def test_groups_is_a_grouping_with_tidy3as_name_axis_and_backend_codes():
    data = xr.Dataset({"v": ("point", np.arange(6.0)), "key": ("point", [3, 1, 3, 2, 1, 3])})
    frame = tidy(data) >> group_by(col("key"))
    groups = expr.Context(frame, verb="summarise").groups()
    assert isinstance(groups, grouped.Grouping)
    assert groups.name == "key" and groups.axis == "point"
    assert groups.codes.tolist() == [2, 0, 2, 1, 0, 2] and groups.labels == [1, 2, 3]
    assert frame._backend.is_array(groups.codes_array) and groups.codes_array.dims == ("point",)
    assert grouped.Grouping(groups.codes, groups.labels).blocks()[0].tolist() == groups.blocks()[0].tolist()
