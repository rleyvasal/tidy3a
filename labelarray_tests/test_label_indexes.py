"""The label index protocol and its implementations (``labelarray.index``, plan milestone 10, step A4).

Every implementation answers the same two questions, the position of each
wanted label and the positions of a label range, under the exact label rule
(``labelarray.labels``): a label matches only one of the same kind
that is exactly equal, so ``2**53 + 1`` never matches ``2**53``, a
half-second never matches whole seconds, and ``True`` never matches 1. The
regular-grid index answers both arithmetically; the tests check it against
the search-based indexes, which are checked against pandas in
``test_label_lookups.py``.
"""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from tidy3a import labelarray
from tidy3a.labelarray import Array, Collection, label_index
from tidy3a.labelarray.errors import LabelError
from tidy3a.labelarray.index import (
    GridIndex,
    LabelIndex,
    SortedIndex,
    SortOnceIndex,
    index_for,
    register_index,
)

IMPLEMENTATIONS = (GridIndex, SortedIndex, SortOnceIndex)
HOURS = np.datetime64("2026-01-01T00", "s") + np.arange(4) * np.timedelta64(3600, "s")


def _positions(result, size):
    """A slice or positions as the positions they select, for comparing implementations."""
    return np.arange(size)[result]


# ── The factory ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("labels", "expected"), [
    (np.arange(10), GridIndex),
    (np.arange(1, 6, dtype="uint8"), GridIndex),
    (np.array([0.0, 0.25, 0.5, 0.75]), GridIndex),
    (np.arange(90, -90.25, -0.25), GridIndex),  # latitude, decreasing
    (np.arange(0, 360, 0.5, dtype="float32"), GridIndex),
    (HOURS, GridIndex),
    (np.arange(5).astype("timedelta64[ms]"), GridIndex),
    (2**53 + 1 + 2 * np.arange(4), GridIndex),
    (np.array([1, 2, 4]), SortedIndex),
    (np.arange(0, 1, 0.1), SortedIndex),  # 0.1 steps are not exactly even in binary
    (np.array([0.0, np.nan, 1.0]), SortedIndex),  # a missing label is not on a grid
    (np.array(["2026-01", "2026-02", "2026-03"], dtype="datetime64[M]"), SortedIndex),  # months differ in length
    (np.array([2**63, 2**63 + 1], dtype="uint64"), SortedIndex),
    (np.array([True, False]), SortOnceIndex),
    (np.array(["a", "b", "c"]), SortedIndex),
    (np.array(["b", "a"], dtype=object), SortOnceIndex),
    (np.array([3, 1, 2]), SortOnceIndex),
    (np.array([1.0, 1.0]), SortOnceIndex),  # repeats are never a grid
    (np.array([5]), SortedIndex),
    (np.array([], dtype=float), SortedIndex),
])
def test_the_factory_inspects_labels_once_and_picks_an_implementation(labels, expected):
    index = index_for(labels)
    assert isinstance(index, expected)
    assert isinstance(index, LabelIndex)  # every implementation follows the protocol


def test_flags_describe_the_labels():
    assert index_for(np.arange(4)).sorted and index_for(np.arange(4)).unique
    assert not index_for(np.arange(4)[::-1]).sorted and index_for(np.arange(4)[::-1]).unique
    assert not index_for(np.array([3, 1, 2])).sorted
    assert not index_for(np.array([1, 2, 1])).unique
    assert not index_for(np.array([np.nan, 1.0, np.nan])).unique  # two missing labels repeat


def test_a_grid_index_holds_no_arrays():
    index = index_for(np.arange(10**6) * 0.5)
    assert isinstance(index, GridIndex)
    slots = [name for cls in type(index).__mro__ for name in getattr(cls, "__slots__", ())]
    held = [getattr(index, name) for name in slots if hasattr(index, name)]
    assert not any(isinstance(value, np.ndarray) for value in held)


def test_backend_label_arrays_keep_the_factorys_index():
    data = Collection({"v": Array(np.zeros(4), ("lat",))}, {"lat": [90.0, 89.75, 89.5, 89.25]})
    assert isinstance(label_index(data.coords["lat"]), GridIndex)
    assert label_index(data.coords["lat"]) is label_index(data.coords["lat"])


def test_each_implementation_refuses_labels_it_cannot_hold():
    with pytest.raises(LabelError, match="evenly spaced"):
        GridIndex(np.array([1, 2, 4]))
    with pytest.raises(LabelError, match="evenly spaced"):
        GridIndex(np.array([0.0, np.nan, 1.0]))
    with pytest.raises(LabelError, match="increasing"):
        SortedIndex(np.array([3, 1, 2]))
    SortOnceIndex(np.array([3, 1, 2]))  # any labels that can be ordered


# ── The exact label rule, for every implementation ──────────────────────────

# Labels evenly spaced, so every implementation can hold them, and lookups with the expected positions.
# A list is looked up one value at a time, each in its own type; an array is looked up whole.
EXACT_CASES = {
    "int labels, exact float and fractions": (np.arange(5), [2.0, 2.5, -0.0, 4, 5], [2, -1, 0, 4, -1]),
    "int labels, True is not 1": (np.arange(3), np.array([True, False]), [-1, -1]),
    "beyond float64 precision": (2**53 + 1 + 2 * np.arange(4), [float(2**53), 2**53 + 1, float(2**53 + 2), 2**53 + 3],
                                 [-1, 0, -1, 1]),
    "int64 near its limit": (np.iinfo("int64").max - 3 + np.arange(4), [np.iinfo("int64").max, 2.0**63, -2.0**63],
                             [3, -1, -1]),
    "uint8 labels, -1 and 300": (np.arange(1, 5, dtype="uint8"), [-1, 300, 4], [-1, -1, 3]),
    "float labels": (np.array([0.0, 0.25, 0.5, 0.75, 1.0]), [0.5, np.float32(0.5), 0.3, 1, np.nan, -0.0, np.inf],
                     [2, 2, -1, 4, -1, 0, -1]),
    "float labels, next float": (np.array([0.0, 0.25, 0.5]), [np.nextafter(0.25, 1), np.nextafter(0.25, 0)], [-1, -1]),
    "float32 labels, float64 0.1": (np.array([0.1, 0.2], dtype="float32"), [0.1, np.float32(0.1), np.float32(0.2)],
                                    [-1, 0, 1]),
    "decreasing latitude": (np.arange(90, -90.25, -0.25), [90, -90.0, 0.0, 0.1, 45.25, 90.25], [0, 720, 360, -1, 179, -1]),
    "tiny steps, a subnormal lookup": (np.array([0.0, 1024.0, 2048.0]), [5e-324, 1024], [-1, 1]),
    "seconds, other units": (HOURS, np.array(["2026-01-01T01", "2026-01-01T00:00:00.5", "NaT"], dtype="datetime64[ms]"),
                             [1, -1, -1]),
    "seconds, a day and text": (HOURS, np.array(["2026-01-01"], dtype="datetime64[D]"), [0]),
    "seconds, a number": (HOURS, [0], [-1]),
    "durations across units": (np.arange(1, 4).astype("timedelta64[s]"), np.array([2000, 2500], dtype="timedelta64[ms]"),
                               [1, -1]),
}


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", list(EXACT_CASES))
def test_every_implementation_keeps_the_exact_label_rule(implementation, name):
    labels, wanted, expected = EXACT_CASES[name]
    order = np.argsort(labels, kind="stable") if implementation is SortedIndex else np.arange(len(labels))
    index = implementation(labels[order])  # the sorted index needs increasing labels
    lookups = [np.array([value]) for value in wanted] if isinstance(wanted, list) else [wanted]
    found = np.concatenate([index.get_indexer(value) for value in lookups])
    assert np.where(found >= 0, order[found], -1).tolist() == expected


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_text_dates_find_time_labels(implementation):
    assert implementation(HOURS).get_indexer(np.array(["2026-01-01T02:00", "2026-01-09"])).tolist() == [2, -1]
    assert implementation(HOURS).get_indexer(np.array(["tomorrow?"])).tolist() == [-1]


@pytest.mark.parametrize("implementation", (SortedIndex, SortOnceIndex))
def test_search_indexes_find_one_missing_label(implementation):
    labels = np.array([0.5, np.nan, 1.5])
    if implementation is SortOnceIndex:
        labels = labels[[2, 1, 0]]
    index = implementation(labels)
    assert index.get_indexer([np.nan, 1.5]).tolist() == pd.Index(labels).get_indexer([np.nan, 1.5]).tolist()


def test_exactness_where_pandas_compares_through_float64():
    labels = 2**53 + 1 + 2 * np.arange(4)
    wanted = np.array([float(2**53)])
    assert pd.Index(labels).get_indexer(wanted).tolist() == [0]  # pandas: 2**53 + 1 == 2**53
    assert index_for(labels).get_indexer(wanted).tolist() == [-1]


# ── The grid index against the search indexes ───────────────────────────────


def _random_grid(rng):
    """Evenly spaced labels of a random type, start, step (either sign), and length."""
    n = int(rng.integers(2, 300))
    sign = 1 if rng.random() < 0.7 else -1
    choice = rng.integers(6)
    if choice == 0:
        start, step = int(rng.integers(-10**6, 10**6)), sign * int(rng.integers(1, 50))
        return (start + step * np.arange(n)).astype(rng.choice(["int64", "int32"]))
    if choice == 1:  # integers beyond float64 precision
        return 2**60 + int(rng.integers(10**6)) + sign * int(rng.integers(1, 5)) * np.arange(n)
    if choice in (2, 3):  # binary fractions: exact in float64, and in float32 while small
        scale = 2.0 ** -int(rng.integers(0, 12))
        values = (int(rng.integers(-1000, 1000)) + sign * int(rng.integers(1, 9)) * np.arange(n)) * scale
        return values.astype("float32" if choice == 3 else "float64")
    unit = rng.choice(["s", "ms", "ns", "D", "h"])
    step = np.timedelta64(int(rng.integers(1, 100)) * sign, unit)
    if choice == 4:
        return np.datetime64("2026-03-01", unit) + np.arange(n) * step
    return np.arange(n) * step


def _wanted(labels, rng):
    """Labels, their neighbours, values between and beyond them, in other types of the same kind."""
    picks = rng.choice(labels, 40)
    if labels.dtype.kind == "f":
        near = np.concatenate([np.nextafter(picks, np.inf), np.nextafter(picks, -np.inf)])
        between = (picks[:-1].astype("float64") + picks[1:].astype("float64")) / 2
        return [picks, near, between, picks.astype("float64"), np.array([labels.min() - 1, labels.max() + 1])]
    if labels.dtype.kind in "iu":
        return [picks, picks + 1, picks - 1, picks.astype("float64"), picks.astype("float64") + 0.5,
                np.array([labels.min() - 1, labels.max() + 1], dtype=object)]
    unit, _ = np.datetime_data(labels.dtype)
    finer = picks.astype(labels.dtype.str[:3] + "[ns]") if unit != "ns" else picks
    return [picks, picks + np.timedelta64(1, unit), finer, finer + np.timedelta64(1, "ns")]


@pytest.mark.parametrize("seed", range(60))
def test_grid_positions_equal_the_search_indexes_on_grid_labels(seed):
    rng = np.random.default_rng(seed)
    labels = _random_grid(rng)
    grid, by_search, by_sort = index_for(labels), SortedIndex(np.sort(labels)), SortOnceIndex(labels)
    assert isinstance(grid, GridIndex)
    for wanted in _wanted(labels, rng):
        expected = by_sort.get_indexer(wanted)
        np.testing.assert_array_equal(grid.get_indexer(wanted), expected)
        within_float64 = labels.dtype.kind in "fiu" and np.abs(labels.astype(float)).max() < 2**53
        if within_float64 and wanted.dtype != object:  # pandas agrees within float64 precision
            np.testing.assert_array_equal(expected, pd.Index(labels).get_indexer(wanted))
        sorted_positions = by_search.get_indexer(wanted)
        order = np.argsort(labels, kind="stable")
        np.testing.assert_array_equal(np.where(sorted_positions >= 0, order[sorted_positions], -1), expected)


def _bounds(labels, rng):
    """Range bounds: labels, values between them, beyond them, and open ends."""
    lo, hi = labels.min().item(), labels.max().item()
    if labels.dtype.kind == "f":
        candidates = [rng.choice(labels), float(rng.choice(labels)) + 1e-9, lo - 1.5, hi + 2, None]
    elif labels.dtype.kind in "iu":
        candidates = [int(rng.choice(labels)), int(rng.choice(labels)) + 0.5, float(rng.choice(labels)),
                      lo - 3, hi + 3, None]
    else:
        pick = rng.choice(labels)
        candidates = [pick, pick.astype(labels.dtype.str[:3] + "[ns]") + np.timedelta64(1, "ns"),
                      labels.min() - np.timedelta64(1, "D"), None]
    return candidates


@pytest.mark.parametrize("seed", range(60))
def test_grid_ranges_equal_the_search_indexes_on_grid_labels(seed):
    rng = np.random.default_rng(1000 + seed)
    labels = _random_grid(rng)
    grid, by_sort = index_for(labels), SortOnceIndex(labels)
    candidates = _bounds(labels, rng)
    for _ in range(30):
        start, stop = candidates[rng.integers(len(candidates))], candidates[rng.integers(len(candidates))]
        got = grid.slice_indexer(start, stop)
        assert isinstance(got, slice)  # a grid's range is contiguous
        expected = _positions(by_sort.slice_indexer(start, stop), len(labels))
        np.testing.assert_array_equal(_positions(got, len(labels)), expected)
        kept = labels[expected]
        if start is not None:
            assert all(_exactly_at_least(v, start) for v in kept)
        if stop is not None:
            assert all(_exactly_at_least(stop, v) for v in kept)


def _exactly_at_least(a, b):
    """``a >= b`` in exact arithmetic: numbers as fractions; NumPy compares times across units exactly."""
    if np.asarray(a).dtype.kind in "fiu":
        return Fraction(np.asarray(a).item()) >= Fraction(np.asarray(b).item())
    return bool(np.asarray(a) >= np.asarray(b))


# ── Ranges ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_ranges_are_exact_and_inclusive(implementation):
    labels = 2**53 - 1 + 2 * np.arange(3)  # 2**53 - 1, 2**53 + 1, 2**53 + 3
    index = implementation(labels)
    assert _positions(index.slice_indexer(None, float(2**53)), 3).tolist() == [0]  # pandas would include 2**53 + 1
    assert _positions(index.slice_indexer(float(2**53), None), 3).tolist() == [1, 2]
    assert _positions(index.slice_indexer(2**53 + 1, 2**53 + 1), 3).tolist() == [1]
    assert _positions(index.slice_indexer(2**53 + 2, 2**53 + 2), 3).tolist() == []
    hours = implementation(HOURS)
    half = np.datetime64("2026-01-01T01:00:00.5", "ms")
    assert _positions(hours.slice_indexer(half, None), 4).tolist() == [2, 3]
    assert _positions(hours.slice_indexer(None, half), 4).tolist() == [0, 1]
    assert _positions(hours.slice_indexer("2026-01-01T01", "2026-01-01T02"), 4).tolist() == [1, 2]
    floats = implementation(np.array([0.1, 0.2], dtype="float32"))
    assert _positions(floats.slice_indexer(None, 0.1), 2).tolist() == []  # float32 0.1 is above float64 0.1
    assert _positions(floats.slice_indexer(None, np.float32(0.1)), 2).tolist() == [0]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_range_bounds_of_another_kind_or_missing_are_errors(implementation):
    index = implementation(np.arange(4))
    with pytest.raises(LabelError, match="range"):
        index.slice_indexer("a", None)
    with pytest.raises(LabelError, match="range"):
        index.slice_indexer(None, np.nan)
    with pytest.raises(LabelError, match="range"):
        index.slice_indexer(True, None)


def test_unsorted_ranges_give_positions_in_axis_order():
    labels = np.array([3.0, np.nan, 1.0, 2.0, 1.0])
    result = index_for(labels).slice_indexer(1.0, 2.5)
    assert result.tolist() == [2, 3, 4]
    assert index_for(np.array(["b", "a", "c"])).slice_indexer("a", "b").tolist() == [0, 1]


def test_sorted_ranges_skip_missing_labels():
    labels = np.array([1.0, np.nan, 2.0, 3.0])
    assert _positions(index_for(labels).slice_indexer(1.0, 2.0), 4).tolist() == [0, 2]


# ── Registering an index ────────────────────────────────────────────────────


class _Counting(SortOnceIndex):
    built = 0

    def __init__(self, labels, op="index", dim=None):
        type(self).built += 1
        super().__init__(labels, op, dim)


def test_a_registered_index_serves_its_kind():
    _Counting.built = 0
    previous = register_index("text", _Counting)
    try:
        index = index_for(np.array(["b", "a"], dtype=object))
        assert isinstance(index, _Counting) and _Counting.built == 1
        assert index.get_indexer(["a", "z"]).tolist() == [1, -1]
        assert index.get_indexer([b"a"]).tolist() == [-1]
        assert not isinstance(index_for(np.arange(3)), _Counting)
    finally:
        register_index("text", previous)
    assert not isinstance(index_for(np.array(["b", "a"])), _Counting)


class _DecimalIndex(SortOnceIndex):
    """Decimal labels: a kind of their own, searched after one sort."""


def test_a_registered_kind_names_its_types_and_never_matches_other_kinds():
    labels = np.array([Decimal("0.1"), Decimal("0.3"), Decimal("0.2")], dtype=object)
    register_index("decimal", _DecimalIndex, types=(Decimal,))
    try:
        index = index_for(labels)
        assert isinstance(index, _DecimalIndex) and index.kind == "decimal"
        assert index.get_indexer(np.array([Decimal("0.2")], dtype=object)).tolist() == [2]
        assert index.get_indexer([0.2]).tolist() == [-1] and index.get_indexer(["0.2"]).tolist() == [-1]
        assert labelarray.labels.kind(labels)[0] == "decimal"
        assert labelarray.labels.compare(labels, np.array([0.1, 0.3, 0.2]))[0] == "different"
    finally:
        register_index("decimal", None)
    with pytest.raises(LabelError, match="mix types"):
        index_for(labels)


def test_a_registered_class_must_answer_the_protocol():
    class Incomplete:
        def get_indexer(self, wanted):
            return np.full(len(wanted), -1)

    with pytest.raises(TypeError, match="slice_indexer"):
        register_index("text", Incomplete)


def test_registered_classes_are_told_the_exact_rule():
    assert "exact_cast" in register_index.__doc__ and "2**53 + 1" in register_index.__doc__
