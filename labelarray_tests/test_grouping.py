"""Groupings and grouped reductions (``labelarray.grouped``, plan milestone 10, steps A2 and A5).

A ``Grouping`` holds one code per position and the group labels, with the
group sizes and the positions sorted by group computed once. ``reduce``
gives every group's result bit-identical to reducing that group on its own,
whether the one-pass kernel (``ONE_PASS``) or the group-by-group path
computes it.
"""

import numpy as np
import pytest

from tidy3a.labelarray import Array, Bins, Grouping, factorize, grouped

OPS = ["sum", "mean", "min", "max", "any", "all", "std", "median", "count"]


def test_grouping_holds_plain_codes_and_labels_and_caches_its_derived_arrays():
    codes = np.array([2, 0, 2, 1, 0, 2])
    grouping = Grouping(codes, ["a", "b", "c"])
    assert isinstance(grouping.codes, np.ndarray) and grouping.codes.dtype.kind in "iu"
    assert grouping.labels == ["a", "b", "c"]
    assert len(grouping) == 3
    assert grouping.counts().tolist() == [2, 1, 3]
    order, starts, ends = grouping.blocks()
    assert order.tolist() == [1, 4, 3, 0, 2, 5]  # positions sorted by group, original order within a group
    assert starts.tolist() == [0, 2, 3] and ends.tolist() == [2, 3, 6]
    assert grouping.counts() is grouping.counts() and grouping.blocks() is grouping.blocks()
    empty = Grouping(np.array([], dtype=np.intp), [])
    assert len(empty) == 0 and empty.counts().tolist() == [] and empty.blocks()[0].tolist() == []


@pytest.mark.parametrize("groups", [3, 40])
@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "count"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_grouped_reductions_match_each_group_alone(groups, op, dtype):
    rng = np.random.default_rng(groups)
    codes = rng.integers(0, groups, 200)
    values = Array((rng.normal(size=(200, 2)) * 100).astype(dtype), ("point", "band"), {"band": ["r", "g"]})
    grouping = Grouping(codes, [f"g{i}" for i in range(groups)])
    got = grouped.reduce(values, op, grouping, "point", "group")
    assert got.dims == ("group", "band") and got.labels("group").tolist() == grouping.labels
    for code in range(groups):
        alone = values.isel(point=np.flatnonzero(codes == code)).reduce(op, ["point"])
        np.testing.assert_array_equal(got.data[code], alone.data)


def test_counts_and_broadcast_back_to_positions():
    grouping = Grouping(np.array([1, 0, 1, 1]), ["a", "b"])
    counts = grouped.counts(grouping, "key", per_position=2)
    assert counts.data.tolist() == [2, 6] and counts.labels("key").tolist() == ["a", "b"]
    stats = Array([10.0, 20.0], ("key",), {"key": ["a", "b"]})
    back = grouped.broadcast(stats, grouping, "point", "key", labels=np.array([5, 6, 7, 8]))
    assert back.dims == ("point",) and back.data.tolist() == [20.0, 10.0, 20.0, 20.0]
    assert back.labels("point").tolist() == [5, 6, 7, 8]


def _random_case(rng):
    groups = int(rng.integers(9, 80))
    sizes = rng.integers(1, 8, groups)
    if rng.random() < 0.4:  # some groups of 8 or more, reduced one by one
        sizes[rng.integers(groups, size=3)] = rng.integers(8, 40, 3)
    key = rng.permutation(np.repeat(rng.permutation(groups) * 3, sizes))
    dtype = str(rng.choice(["float32", "float64", "int16", "int64", "bool"]))
    layout = str(rng.choice(["1d", "first", "last"]))
    shape = (len(key),) if layout == "1d" else (len(key), 3)
    values = rng.normal(size=shape) * 10 ** rng.uniform(-3, 8, shape)
    if dtype.startswith("float"):
        values[rng.random(shape) < 0.05] = -0.0
        values[rng.random(shape) < 0.05] = np.nan
        values[rng.random(shape) < 0.01] = np.inf
    elif dtype == "bool":
        values = values > 0
    values = values.astype(dtype)
    if layout == "1d":
        array = Array(values, ("point",))
    elif layout == "first":
        array = Array(values, ("point", "feature"), {"feature": ["f1", "f2", "f3"]})
    else:
        array = Array(np.ascontiguousarray(values.T), ("feature", "point"), {"feature": ["f1", "f2", "f3"]})
    codes, labels = factorize(key)
    return array, Grouping(codes, labels)


def _same_bits(a: Array, b: Array) -> None:
    assert a.dims == b.dims and a.dtype == b.dtype
    assert a.data.tobytes() == b.data.tobytes()


@pytest.mark.parametrize("seed", range(200))
def test_one_pass_and_group_by_group_reductions_give_the_same_bits(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    array, grouping = _random_case(rng)
    op = str(rng.choice(OPS))
    na_rm = bool(rng.random() < 0.5)
    min_count = int(rng.integers(1, 4)) if op == "sum" and rng.random() < 0.3 else None
    with np.errstate(all="ignore"):
        fast = grouped.reduce(array, op, grouping, "point", "key", na_rm=na_rm, min_count=min_count)
        monkeypatch.setattr(grouped, "ONE_PASS", False)
        slow = grouped.reduce(array, op, grouping, "point", "key", na_rm=na_rm, min_count=min_count)
        _same_bits(fast, slow)
        for code in rng.choice(len(grouping), 5):
            alone = array.isel(point=np.flatnonzero(grouping.codes == code)).reduce(
                op, ["point"], na_rm=na_rm, min_count=min_count)
            assert fast.data[code].tobytes() == alone.data.tobytes()


def test_empty_bins_reduce_as_empty_groups(monkeypatch):
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    edges = np.concatenate([np.linspace(-3, 0, 20), np.linspace(10, 12, 5)])  # the last bins stay empty
    codes, labels = factorize(np.clip(values, -3, 0), Bins(edges))
    grouping = Grouping(codes, labels)
    array = Array(values, ("point",))
    for op in ("mean", "min", "count"):
        with np.errstate(all="ignore"):
            fast = grouped.reduce(array, op, grouping, "point", "bin")
            monkeypatch.setattr(grouped, "ONE_PASS", False)
            _same_bits(fast, grouped.reduce(array, op, grouping, "point", "bin"))
            monkeypatch.setattr(grouped, "ONE_PASS", True)
    assert grouped.counts(grouping, "bin").data[-4:].tolist() == [0, 0, 0, 0]
