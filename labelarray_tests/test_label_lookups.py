"""Label lookups in the library's label indexes (``labelarray.index``), checked against pandas.

pandas' ``Index.get_indexer`` is the reference for which labels match: the
prototype must agree with it wherever labels are unique, including across
numeric types, time units, NaN, and labels of a different kind. Where the
prototype differs on purpose, the test says so: repeated labels raise a
``LabelError`` naming the operation, and integers beyond float64 precision are
compared exactly, where pandas compares them through float64.
"""

import numpy as np
import pandas as pd
import pytest

from tidy3a.labelarray import (
    AlignError,
    Array,
    Collection,
    LabelError,
    align,
    label_index,
)
from tidy3a.labelarray.index import index_for

DAYS = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]")

CASES = {
    "int labels, int lookups": (np.array([10, 30, 20]), np.array([20, 40, 10])),
    "int labels, exact float": (np.array([1, 2]), np.array([2.0, 2.5])),
    "int labels, True is not 1": (np.array([0, 1]), np.array([True])),
    "bool labels, 1 is not True": (np.array([False, True]), np.array([1])),
    "float labels, int lookup": (np.array([1.0, 2.0]), np.array([2])),
    "float32 labels, float64 0.1": (np.array([0.1, 0.2], dtype="float32"), np.array([0.1])),
    "float32 labels, float32 0.1": (np.array([0.1, 0.2], dtype="float32"), np.array([0.1], dtype="float32")),
    "uint64 beyond int64, -1": (np.array([2**63 + 5, 1], dtype="uint64"), np.array([-1, 1])),
    "uint64 beyond int64, same": (np.array([2**63 + 5, 1], dtype="uint64"), np.array([2**63 + 5], dtype="uint64")),
    "-0.0 finds 0.0": (np.array([0.0, 1.0]), np.array([-0.0])),
    "NaN label": (np.array([0.5, np.nan, 1.5]), np.array([np.nan, 1.5])),
    "NaN lookup, no NaN label": (np.array([0.5, 1.5]), np.array([np.nan])),
    "complex labels, real lookup": (np.array([1 + 1j, 2 + 0j]), np.array([2.0])),
    "text labels": (np.array(["front", "rear", "left"]), np.array(["rear", "top"])),
    "text labels as objects": (np.array(["a", "b"], dtype=object), np.array(["b"])),
    "text labels, bytes lookup": (np.array(["a", "b"]), np.array([b"a"])),
    "bytes labels": (np.array([b"a", b"b"]), np.array([b"b", b"c"])),
    "nanoseconds, day lookup": (DAYS, np.array(["2026-01-02"], dtype="datetime64[D]")),
    "nanoseconds, text date": (DAYS, np.array(["2026-01-03", "2026-01-09"])),
    "seconds, half-second lookup": (
        np.array(["2026-01-01T00:00:00", "2026-01-01T00:00:01"], dtype="datetime64[s]"),
        np.array(["2026-01-01T00:00:01.5", "2026-01-01T00:00:01"], dtype="datetime64[ns]"),
    ),
    "NaT label": (np.array(["2026-01-01", "NaT"], dtype="datetime64[ns]"), np.array(["NaT"], dtype="datetime64[ns]")),
    "durations across units": (np.array([1, 2], dtype="timedelta64[s]").astype("timedelta64[ns]"),
                               np.array([2], dtype="timedelta64[s]")),
    "time labels, number lookup": (DAYS, np.array([1])),
    "empty labels": (np.array([], dtype=float), np.array([1.0])),
    "empty lookup": (np.array([1.0, 2.0]), np.array([], dtype=float)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_lookups_match_pandas(name):
    labels, wanted = CASES[name]
    expected = pd.Index(labels).get_indexer(wanted)
    np.testing.assert_array_equal(index_for(labels).get_indexer(wanted), expected)


def _random_labels(dtype: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if dtype in ("int64", "uint64"):
        return rng.choice(10 * n, n, replace=False).astype(dtype) + (2**63 if dtype == "uint64" else 0)
    if dtype in ("float64", "float32"):
        values = rng.choice(10 * n, n, replace=False).astype(dtype) / 7
        return np.unique(values)[:n]
    if dtype == "str":
        return np.array([f"s{k:06d}" for k in rng.choice(10 * n, n, replace=False)])
    if dtype.startswith("datetime64"):
        return np.datetime64("2026-01-01", "s") + rng.choice(10 * n, n, replace=False).astype("timedelta64[s]")
    if dtype == "timedelta64":
        return rng.choice(10 * n, n, replace=False).astype("timedelta64[ms]")
    raise AssertionError(dtype)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "float64", "float32", "str", "datetime64", "timedelta64"])
@pytest.mark.parametrize("seed", range(4))
def test_random_labels_match_pandas_sorted_and_shuffled(dtype, seed):
    rng = np.random.default_rng(seed)
    labels = _random_labels(dtype, 500, rng)
    others = _random_labels(dtype, 500, np.random.default_rng(seed + 100))
    wanted = np.concatenate([rng.choice(labels, 200), rng.choice(others, 200)])
    rng.shuffle(wanted)
    for candidate in (np.sort(labels), labels[rng.permutation(len(labels))]):
        np.testing.assert_array_equal(index_for(candidate).get_indexer(wanted),
                                      pd.Index(candidate).get_indexer(wanted))


def test_large_integers_compare_exactly_unlike_pandas():
    labels, wanted = np.array([2**53 + 1, 3]), np.array([float(2**53)])
    assert pd.Index(labels).get_indexer(wanted).tolist() == [0]  # pandas: through float64, 2**53 == 2**53 + 1
    assert index_for(labels).get_indexer(wanted).tolist() == [-1]  # exact: they differ
    assert index_for(labels).get_indexer([2**53 + 1]).tolist() == [0]


def test_repeated_labels_raise_where_pandas_raises():
    index = index_for(np.array([1.0, 2.0, 1.0]))
    assert not index.unique
    with pytest.raises(LabelError, match="labels repeat"):
        index.get_indexer([1.0])
    assert not index_for(np.array([np.nan, 1.0, np.nan])).unique  # two NaN labels repeat too
    assert index.get_indexer(np.array([], dtype=float)).size == 0


def test_mixed_type_labels_are_an_error_naming_the_operation():
    a = Array([1.0, 2.0], ("k",), {"k": np.array([1, "a"], dtype=object)})
    b = Array([1.0, 2.0], ("k",), {"k": np.array(["a", 1], dtype=object)})
    with pytest.raises(LabelError, match=r"align: labels on 'k' mix types \(int, str\)"):
        align(a, b)
    with pytest.raises(LabelError, match=r"sel: labels on 'k' mix types"):
        a.sel(k=["a"])


# ── The index in selection and alignment ────────────────────────────────────


def test_selection_by_label_across_time_units_and_nan():
    a = Array([1.0, 2.0, 3.0], ("time",), {"time": DAYS})
    assert a.sel(time=np.datetime64("2026-01-02")).values.tolist() == 2.0  # one label: the axis goes
    assert a.sel(time=["2026-01-03"]).values.tolist() == [3.0]  # a list keeps it
    with pytest.raises(LabelError, match="no label"):
        a.sel(time=np.datetime64("2026-01-02T12", "h"))
    b = Array([1.0, 2.0, 3.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    assert b.sel(x=[np.nan]).values.tolist() == [2.0]
    with pytest.raises(LabelError, match="no label"):
        Array([1.0, 2.0], ("k",), {"k": [0, 1]}).sel(k=[True])


def test_alignment_matches_nan_labels_and_refuses_real_differences():
    a = Array([1.0, 2.0, 3.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    b = Array([20.0, 10.0, 30.0], ("x",), {"x": [np.nan, 0.5, 1.5]})
    left, right = align(a, b)
    assert (left + right).values.tolist() == [11.0, 22.0, 33.0]
    assert right.labels("x").tolist()[::2] == [0.5, 1.5]
    same = Array([1.0, 1.0, 1.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    assert align(a, same)[1] is same  # equal labels, NaN included: nothing reordered
    c = Array([1.0, 2.0, 3.0], ("x",), {"x": [0.5, 2.5, 1.5]})
    with pytest.raises(AlignError, match="labels differ"):
        align(a, c)


def test_alignment_across_time_units():
    a = Array([1.0, 2.0], ("time",), {"time": DAYS[:2]})
    b = Array([20.0, 10.0], ("time",), {"time": np.array(["2026-01-02", "2026-01-01"], dtype="datetime64[s]")})
    left, right = align(a, b)
    assert (left + right).values.tolist() == [11.0, 22.0]


# ── Caching and immutability ────────────────────────────────────────────────


def test_index_is_built_once_per_label_array():
    data = Collection({"v": Array(np.zeros(4), ("point",))}, {"point": [4, 3, 2, 1]})
    labels = data.coords["point"]
    assert label_index(labels) is label_index(labels)
    assert data.array("v").sel(point=[3]).coords["point"].tolist() == [3]
    assert label_index(labels) is label_index(data.array("v").coords["point"])


def test_labels_are_frozen_so_a_cached_index_cannot_go_stale():
    source = np.array([10, 20, 30])
    a = Array([1.0, 2.0, 3.0], ("point",), {"point": source})
    source[0] = 99  # the caller's array is not the stored one
    assert a.coords["point"].tolist() == [10, 20, 30]
    assert not a.coords["point"].flags.writeable
    view = source.view()
    view.flags.writeable = False  # read-only, but its base can still change
    b = Array([1.0, 2.0, 3.0], ("point",), {"point": view})
    assert b.coords["point"] is not view
    writeable = np.array([1, 2, 3])
    assert label_index(writeable) is not label_index(writeable)  # never cached


def test_sorted_and_unsorted_labels_give_the_same_positions():
    rng = np.random.default_rng(3)
    labels = np.sort(rng.choice(10**6, 10**4, replace=False))
    shuffled = labels[rng.permutation(len(labels))]
    wanted = rng.choice(labels, 1000)
    np.testing.assert_array_equal(labels[index_for(labels).get_indexer(wanted)], wanted)
    np.testing.assert_array_equal(shuffled[index_for(shuffled).get_indexer(wanted)], wanted)


def test_a_callers_labels_cannot_change_a_frame_later():
    labels = np.array([10, 20, 30])
    labels.flags.writeable = False  # read-only now, but the caller owns it and can unlock it
    a = Array([1.0, 2.0, 3.0], ("p",), {"p": labels})
    label_index(a.coords["p"]).get_indexer([20])  # build and cache the index
    labels.flags.writeable = True
    labels[1] = 99
    assert a.coords["p"].tolist() == [10, 20, 30]
    assert label_index(a.coords["p"]).get_indexer([20]).tolist() == [1]


def test_an_index_is_released_with_its_labels():
    import gc
    import weakref

    data = Collection({"v": Array(np.zeros(5), ("p",))}, {"p": np.arange(5)})
    labels = data.coords["p"]
    label_index(labels).get_indexer([3])
    alive = weakref.ref(labels)
    del data, labels
    gc.collect()
    assert alive() is None


def test_arrays_have_a_shape():
    assert Array(np.zeros((2, 3)), ("a", "b")).shape == (2, 3)
