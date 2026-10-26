"""Key transforms for grouping (plan milestone 10, step A4).

Grouping is on exact equality only; domain meaning lives in a transform
applied to the key before it is factorized. The built-in transforms in
``labelarray.transforms`` are small records holding their settings
(``Bins``, ``Duration``, ``Round``, ``GridCells``): they give group codes and
readable labels in sorted order, compare equal when their settings do, and
print those settings. Plain functions are transforms too, equal only to
themselves. ``factorize`` turns key values into group codes and labels.
"""

import datetime
import math
import re
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from tidy3a.labelarray import errors as library_errors
from tidy3a.labelarray.transforms import (
    Bins,
    Duration,
    GridCells,
    Round,
    as_transform,
    factorize,
)

NAN = float("nan")


def codes_and_labels(transform, values):
    codes, labels = transform.groups(np.asarray(values))
    return codes.tolist(), labels


def times(*texts, unit="ns"):
    return np.array(texts, dtype=f"datetime64[{unit}]")


# ── Bins ────────────────────────────────────────────────────────────────────


def test_bins_give_codes_and_every_bin_as_a_label_in_order():
    codes, labels = codes_and_labels(Bins([0, 15, 35, 100]), [10.0, 20.0, 0.0, 34.9])
    assert codes == [0, 1, 0, 1]
    assert labels == ["[0, 15)", "[15, 35)", "[35, 100)"]  # empty bins are groups too


def test_bins_mark_values_outside_every_bin_and_factorize_refuses_them():
    codes, _ = codes_and_labels(Bins([0, 15]), [-1.0, 5.0, 15.0])
    assert codes == [-1, 0, -1]
    with pytest.raises(library_errors.GroupError, match=r"2 positions fall outside every bin \[0, 15\)"):
        factorize(np.array([-1.0, 5.0, 15.0]), Bins([0, 15]))


def test_bins_need_two_increasing_edges():
    for edges in ([1], [0, 0], [3, 1], [[0, 1], [2, 3]]):
        with pytest.raises(library_errors.GroupError, match="at least two increasing edges"):
            Bins(edges)


def test_bins_of_times_label_their_edges_as_dates():
    edges = times("2026-06-01", "2026-06-03", "2026-06-05")
    codes, labels = codes_and_labels(Bins(edges), times("2026-06-02", "2026-06-04T12"))
    assert codes == [0, 1]
    assert labels == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)"]


def test_bins_compare_equal_when_their_edges_do():
    assert Bins([0, 15, 35]) == Bins((0, 15, 35)) == Bins(np.array([0, 15, 35]))
    assert hash(Bins([0, 15, 35])) == hash(Bins(np.array([0, 15, 35], dtype="int32")))
    assert Bins([0, 15, 35]) != Bins([0, 15, 36])
    # Whole-number float edges group alike but may print differently (1e+20 against 100000000000000000000).
    assert Bins([0, 15]) != Bins([0.0, 15.0])
    assert Bins(times("2026-01-01", "2026-01-02", unit="D")) == Bins(times("2026-01-01", "2026-01-02"))
    assert Bins([0, 1]) != Duration("1 day")
    assert {Bins([0, 1]): "cached"}[Bins([0, 1])] == "cached"


def test_bins_print_their_edges():
    assert repr(Bins([0, 15, 35])) == "Bins(edges=[0, 15, 35])"
    assert repr(Bins([0, 0.1, 0.25])) == "Bins(edges=[0, 0.1, 0.25])"
    assert repr(Bins(times("2026-06-01", "2026-06-03"))) == 'Bins(edges=["2026-06-01", "2026-06-03"])'


# ── Duration ────────────────────────────────────────────────────────────────


def test_duration_windows_start_at_the_first_time_by_default():
    values = times("2026-06-02", "2026-06-01", "2026-06-04", "2026-06-03")
    codes, labels = codes_and_labels(Duration("2 days"), values)
    assert codes == [0, 0, 1, 1]
    assert labels == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)"]


def test_duration_windows_follow_an_origin_before_or_after_the_times():
    values = times("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04")
    codes, labels = codes_and_labels(Duration("2 days", origin="2026-05-31"), values)
    assert codes == [0, 1, 1, 2]
    assert labels == ["[2026-05-31, 2026-06-02)", "[2026-06-02, 2026-06-04)", "[2026-06-04, 2026-06-06)"]
    later = Duration("2 days", origin=np.datetime64("2026-06-10"))
    assert codes_and_labels(later, values) == codes_and_labels(Duration("2 days", origin="2026-05-31"), values)


def test_duration_lists_every_window_between_the_first_and_the_last():
    codes, labels = codes_and_labels(Duration("1 day"), times("2026-06-01", "2026-06-04T06"))
    assert codes == [0, 3]
    assert len(labels) == 4


def test_times_straddling_a_window_edge_fall_in_different_windows_by_design():
    values = np.array([np.datetime64("2026-06-03", "ns") - np.timedelta64(1, "ns"), np.datetime64("2026-06-03", "ns")])
    codes, labels = codes_and_labels(Duration("2 days", origin="2026-06-01"), values)
    assert codes == [0, 1]
    assert labels == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)"]


def test_duration_widths_read_as_text_or_durations():
    two_days = Duration("2 days")
    for width in ("48h", "2D", "1 day 24 hours", np.timedelta64(2, "D"), datetime.timedelta(days=2),
                  pd.Timedelta("2 days")):
        assert Duration(width) == two_days and hash(Duration(width)) == hash(two_days)
    assert Duration("2 days") != Duration("2 days 1 ns")
    assert Duration("2 days", origin="2026-01-01") == Duration("2 days", origin=np.datetime64("2026-01-01T00", "ns"))
    assert Duration("2 days", origin="2026-01-01") == Duration("2 days", origin=datetime.datetime(2026, 1, 1))
    assert Duration("2 days", origin="2026-01-01") != Duration("2 days")


def test_duration_prints_its_width_and_origin():
    assert repr(Duration("2D")) == 'Duration(width="2 days", origin=None)'
    assert repr(Duration("1h30min", origin="2026-01-01")) == 'Duration(width="1 hour 30 minutes", origin="2026-01-01")'
    assert repr(Duration("250ms", origin="2026-01-01T06:30")) == \
        'Duration(width="250 ms", origin="2026-01-01T06:30:00")'
    for width in ("2D", "1h30min", "250ms", "1 week 3 seconds 7ns"):
        printed = re.search(r'width="([^"]+)"', repr(Duration(width))).group(1)
        assert Duration(printed) == Duration(width)


def test_duration_refuses_what_has_no_fixed_length():
    for width in ("1 month", "2 years", "5 m", "0 days", np.timedelta64(1, "M"), np.timedelta64(-1, "D"), 3):
        with pytest.raises(library_errors.GroupError, match="duration"):
            Duration(width)


def test_duration_needs_times():
    with pytest.raises(library_errors.GroupError, match="a duration needs time values"):
        Duration("2 days").groups(np.array([1.0, 2.0]))
    with pytest.raises(library_errors.GroupError, match="there are no times to bin"):
        Duration("2 days").groups(times())
    assert codes_and_labels(Duration("2 days", origin="2026-01-01"), times()) == ([], [])


def test_time_zones_are_refused_in_values_and_origins():
    aware = np.array([datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)], dtype=object)
    with pytest.raises(library_errors.GroupError, match="time zone"):
        Duration("1 day").groups(aware)
    with pytest.raises(library_errors.GroupError, match="time zone"):
        Bins([aware[0], aware[0] + datetime.timedelta(days=1)])
    for origin in (pd.Timestamp("2026-01-01", tz="UTC"), datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
                   "2026-01-01T00:00+02:00"):
        with pytest.raises(library_errors.GroupError, match="time zone"):
            Duration("1 day", origin=origin)
    for transform in (Round(1), GridCells(1)):
        with pytest.raises(library_errors.GroupError, match="time zone"):
            transform.groups(aware)


# ── Round ───────────────────────────────────────────────────────────────────


def test_round_labels_each_group_by_its_rounded_value():
    codes, labels = codes_and_labels(Round(0.5), [0.2, 0.26, 0.74, 0.76, -0.25])
    assert codes == [0, 1, 1, 2, 0]
    assert labels == [0.0, 0.5, 1.0]
    assert all(type(label) is float for label in labels)


def test_round_sends_halves_up_so_every_bucket_is_half_open():
    # np.round sends halves to even (0.5 to 0, 1.5 to 2); here each bucket is [c - step/2, c + step/2).
    codes, labels = codes_and_labels(Round(1), [0.5, 1.5, -0.5, 2.4999])
    assert labels == [0.0, 1.0, 2.0]
    assert codes == [1, 2, 0, 2]


def test_round_reads_steps_as_the_decimals_they_print_as():
    # 0.35 / 0.1 is 3.4999999999999996 in floats; the edge between 0.3 and 0.4 is the float nearest 0.35.
    codes, labels = codes_and_labels(Round(0.1), [0.3, 0.25, 0.35, 0.34999999999999998])
    assert labels == [0.3, 0.4]
    assert codes == [0, 0, 1, 1]  # the last literal is the same float as 0.35


def test_round_follows_an_origin():
    codes, labels = codes_and_labels(Round(1, origin=0.5), [0.9, 1.1, 2.0, -0.1])
    assert labels == [-0.5, 0.5, 1.5, 2.5]
    assert codes == [1, 2, 3, 0]


def test_round_keeps_whole_numbers_whole_and_exact():
    codes, labels = codes_and_labels(Round(10), np.array([4, 5, 14, 15, -5]))
    assert labels == [0, 10, 20] and all(type(label) is int for label in labels)
    assert codes == [0, 1, 1, 2, 0]
    big = np.array([2**53 + 1, 2**53], dtype=np.int64)
    assert codes_and_labels(Round(1), big) == ([1, 0], [2**53, 2**53 + 1])
    assert codes_and_labels(Round(1.0), big) == codes_and_labels(Round(1), big)


def test_round_groups_signed_zeros_as_one_value():
    codes, labels = codes_and_labels(Round(1), [-0.0, 0.0, -0.3, 0.3])
    assert codes == [0, 0, 0, 0] and labels == [0.0]


def test_round_marks_infinite_values_and_factorize_refuses_them():
    codes, _ = codes_and_labels(Round(1), [1.0, math.inf, -math.inf])
    assert codes == [0, -1, -1]
    with pytest.raises(library_errors.GroupError, match="2 positions are infinite"):
        factorize(np.array([1.0, math.inf, -math.inf]), Round(1))


def test_round_float32_values_are_placed_exactly():
    # float32(0.35) and float32(0.45) are just below the decimals, so they are not halves and round down.
    values = np.array([0.25, 0.35, 0.45], dtype=np.float32)
    codes, labels = codes_and_labels(Round(0.1), values)
    assert labels == [0.3, 0.4] and codes == [0, 0, 1]


def test_round_compares_and_prints_its_settings():
    assert Round(0.5) == Round(0.5, origin=0) == Round(Fraction(1, 2)) == Round(np.float64(0.5))
    assert Round(1) == Round(1.0) and hash(Round(1)) == hash(Round(1.0))
    assert Round(0.1) == Round(np.float32(0.1))  # read as the decimal it prints as
    assert Round(0.5) != Round(0.5, origin=0.25)
    assert Round(0.5) != GridCells(0.5)
    assert repr(Round(0.5)) == "Round(step=0.5, origin=0)"
    assert repr(Round(10, origin=2.5)) == "Round(step=10, origin=2.5)"
    assert repr(Round(Fraction(1, 3))) == "Round(step=1/3, origin=0)"


@pytest.mark.parametrize("step", [0, -1, math.nan, math.inf, True, "1", None])
def test_round_needs_a_positive_finite_step(step):
    with pytest.raises(library_errors.GroupError, match="step"):
        Round(step)


def test_round_needs_numbers():
    for values in (times("2026-01-01"), np.array(["a"]), np.array([True]), np.array([1j])):
        with pytest.raises(library_errors.GroupError, match="needs numbers"):
            Round(1).groups(values)


# ── GridCells ───────────────────────────────────────────────────────────────


def test_grid_cells_label_each_occupied_cell_by_its_edges():
    codes, labels = codes_and_labels(GridCells(0.2), [0.05, 0.19, 0.2, 0.41, -0.01])
    assert labels == ["[-0.2, 0)", "[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)"]
    assert codes == [1, 1, 2, 3, 0]


def test_grid_cells_list_only_cells_that_hold_values():
    # A grid can be large and sparse (points in voxels), so empty cells are not groups.
    codes, labels = codes_and_labels(GridCells(1), [0.5, 100.5])
    assert labels == ["[0, 1)", "[100, 101)"] and codes == [0, 1]


def test_grid_cell_edges_are_the_decimals_they_print_as():
    # floor(0.3 / 0.1) is 2 in floats; the cell holding 0.3 is the one printed [0.3, 0.4).
    assert codes_and_labels(GridCells(0.1), [0.3]) == ([0], ["[0.3, 0.4)"])
    assert codes_and_labels(GridCells(0.1), [0.7, 0.6999999999999999]) == ([1, 0], ["[0.6, 0.7)", "[0.7, 0.8)"])


def test_grid_cells_follow_an_origin():
    codes, labels = codes_and_labels(GridCells(1, origin=0.5), [0.4, 0.5, 1.49, 1.5])
    assert labels == ["[-0.5, 0.5)", "[0.5, 1.5)", "[1.5, 2.5)"]
    assert codes == [0, 1, 1, 2]


def test_values_straddling_a_cell_edge_fall_in_different_cells_by_design():
    edge = 0.4
    codes, labels = codes_and_labels(GridCells(0.2), [np.nextafter(edge, 0), edge])
    assert codes == [0, 1] and labels == ["[0.2, 0.4)", "[0.4, 0.6)"]


def test_grid_cells_of_whole_numbers_stay_whole():
    codes, labels = codes_and_labels(GridCells(10), np.array([-1, 0, 9, 10], dtype=np.int16))
    assert labels == ["[-10, 0)", "[0, 10)", "[10, 20)"] and codes == [0, 1, 1, 2]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("seed", range(4))
def test_grid_cells_hold_exactly_the_values_between_their_printed_edges(dtype, seed):
    rng = np.random.default_rng(seed)
    size = [0.1, 0.25, 0.3, 1e-3, 7.0][seed]
    origin = [0, 0.05, -1.1, 0.0005, 3][seed]
    values = (rng.normal(0, 40 * size, 2_000)).astype(dtype)
    values[:50] = (origin + np.arange(50) * size).astype(dtype)  # on the edges, as decimals
    codes, labels = GridCells(size, origin=origin).groups(values)
    edges = [tuple(float(part) for part in label[1:-1].split(", ")) for label in labels]
    assert edges == sorted(edges) and len(set(labels)) == len(labels)
    for value, code in zip(values.astype(float), codes):
        low, high = edges[code]
        assert low <= value < high
    # Each edge is the float nearest the decimal origin + k * size.
    exact = [Fraction(str(origin)) + round((Fraction(str(low)) - Fraction(str(origin))) / Fraction(str(size)))
             * Fraction(str(size)) for low, _ in edges]
    assert [float(e) for e in exact] == [low for low, _ in edges]


def test_grid_cells_compare_and_print_their_settings():
    assert GridCells(0.2) == GridCells(0.2, origin=0.0) and hash(GridCells(0.2)) == hash(GridCells(0.2, 0))
    assert GridCells(0.2) != GridCells(0.2, origin=0.1)
    assert repr(GridCells(0.2, origin=0.1)) == "GridCells(size=0.2, origin=0.1)"


def test_grid_cells_need_a_positive_size():
    with pytest.raises(library_errors.GroupError, match="size"):
        GridCells(0)


# ── Functions and factorize ─────────────────────────────────────────────────


def test_plain_functions_are_transforms_equal_only_to_themselves():
    def floor(values):
        return np.floor(values)

    transform = as_transform(floor)
    same = transform
    assert transform == same
    assert transform != as_transform(floor)
    assert as_transform(lambda v: v) != as_transform(lambda v: v)
    assert transform != Round(1)
    assert as_transform(Round(1)) is not None and as_transform(Round(1)) == Round(1)
    codes, labels = factorize(np.array([1.5, 0.2, 1.9]), floor)
    assert codes.tolist() == [1, 0, 1] and labels == [0.0, 1.0]
    assert "floor" in repr(transform)


def test_a_function_must_give_one_key_per_value():
    with pytest.raises(library_errors.GroupError, match="one key per value"):
        factorize(np.array([1.0, 2.0]), lambda v: v[:1])


def test_anything_else_is_not_a_transform():
    with pytest.raises(TypeError, match="transform"):
        as_transform(0.5)


def test_factorize_without_a_transform_groups_exactly_equal_values():
    codes, labels = factorize(np.array([2**53 + 1, 2**53, 2**53 + 1]))
    assert codes.tolist() == [1, 0, 1] and labels == [2**53, 2**53 + 1]
    codes, labels = factorize(np.array([0.0, -0.0]))
    assert codes.tolist() == [0, 0] and len(labels) == 1


@pytest.mark.parametrize("transform, missing_label", [
    (None, "nan value"), (Round(1), "nan value"), (GridCells(1), "NaN"), (Bins([0, 10]), "NaN"),
])
def test_factorize_drops_missing_values_or_keeps_them_as_the_last_group(transform, missing_label):
    values = np.array([1.0, NAN, 2.0, -NAN])
    codes, labels = factorize(values, transform)
    assert codes[[1, 3]].tolist() == [-1, -1] and len(labels) == (1 if transform == Bins([0, 10]) else 2)
    kept_codes, kept = factorize(values, transform, dropna=False)
    assert kept_codes[[1, 3]].tolist() == [len(kept) - 1] * 2
    assert kept[:-1] == labels
    if missing_label == "NaN":
        assert kept[-1] == "NaN"
    else:
        assert isinstance(kept[-1], float) and math.isnan(kept[-1])


def test_factorize_keeps_missing_times_as_nat():
    values = times("2026-06-01", "NaT", "2026-06-02")
    assert factorize(values, Duration("1 day"), dropna=False)[1][-1] == "NaT"
    assert np.isnat(factorize(values, dropna=False)[1][-1])
