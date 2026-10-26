"""Key transforms for grouping (plan milestone 10, step A4).

Grouping is on exact equality only; domain meaning lives in a transform
applied to the key before it is factorized. The built-in transforms in
``tidy3a.labelarray.transforms`` are small records holding their settings
(``Bins``, ``Duration``, ``Round``, ``GridCells``): they give group codes and
readable labels in sorted order, compare equal when their settings do, and
print those settings. Plain functions are transforms too, equal only to
themselves. In tidy3a, ``bins()`` is built on ``Bins`` and ``Duration``, and
``rounded()`` and ``cells()`` expose the other two to ``group_by``; the
transforms themselves are tested with the library (``labelarray_tests``).
"""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidy3a import (
    GroupError,
    TidyError,
    bins,
    cells,
    col,
    group_by,
    mean,
    mutate,
    n,
    rounded,
    summarise,
    tidy,
)
from tidy3a import errors as tidy_errors
from tidy3a.labelarray import errors as library_errors
from tidy3a.labelarray.transforms import (
    Bins,
    Duration,
    GridCells,
    Round,
)

NAN = float("nan")


def times(*texts, unit="ns"):
    return np.array(texts, dtype=f"datetime64[{unit}]")


def test_the_library_group_error_is_tidy3as_too():
    assert issubclass(library_errors.GroupError, library_errors.LabelArrayError)
    assert issubclass(tidy_errors.GroupError, TidyError) and issubclass(tidy_errors.GroupError, library_errors.GroupError)


# ── bins() on the records ───────────────────────────────────────────────────


def test_bins_keys_hold_the_records():
    assert bins(col("x"), [0, 15, 35]).transform == Bins([0, 15, 35])
    assert bins(col("t"), "2 days").transform == Duration("2 days")
    assert bins(col("t"), "2 days", origin="2026-06-01").transform == Duration("2 days", origin="2026-06-01")


def stamps():
    values = times("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04")
    return tidy(xr.Dataset({"v": ("point", [1.0, 2.0, 3.0, 4.0]), "t": ("point", values)}))


def test_bins_by_duration_take_an_origin():
    out = (stamps() >> group_by(bins(col("t"), "2 days", origin="2026-05-31")) >> summarise(m=mean(col("v")))).collect()
    assert out.t_bin.values.tolist() == ["[2026-05-31, 2026-06-02)", "[2026-06-02, 2026-06-04)",
                                         "[2026-06-04, 2026-06-06)"]
    assert out["m"].values.tolist() == [1.0, 2.5, 4.0]


def test_bins_with_edges_take_no_origin():
    with pytest.raises(GroupError, match=r"bins\(x, …\): an origin places the windows of a duration"):
        bins(col("x"), [0, 1], origin=0)


def test_bins_say_which_call_could_not_read_its_duration():
    with pytest.raises(GroupError, match=r'bins\(t, "1 month"\): cannot read the duration'):
        bins(col("t"), "1 month")


def test_bins_refuse_times_with_a_time_zone():
    ds = xr.Dataset({"v": ("t", [1.0, 2.0, 3.0])}, coords={"t": pd.date_range("2026-01-01", periods=3, tz="UTC")})
    # The NumPy backend does not hold zoned times at all; on xarray, grouping refuses them.
    with pytest.raises(TidyError, match="time zone"):
        tidy(ds) >> group_by(bins(col("t"), "2 days"))


# ── rounded() and cells() through group_by ──────────────────────────────────


def ranges(distance, v=None):
    v = np.arange(len(distance), dtype=float) if v is None else v
    return tidy(xr.Dataset({"v": ("point", np.asarray(v, dtype=float)),
                            "distance": ("point", np.asarray(distance, dtype=float))}))


def test_group_by_rounded_values():
    out = ranges([0.2, 0.26, 0.74, 0.76, 1.1]) >> group_by(rounded(col("distance"), 0.5))
    assert "groups: rounded(distance)" in repr(out)
    data = (out >> summarise(k=n(), m=mean(col("v")))).collect()
    assert data.distance_rounded.values.tolist() == [0.0, 0.5, 1.0]
    assert data["k"].values.tolist() == [1, 2, 2]
    assert data["m"].values.tolist() == [0.0, 1.5, 3.5]


def test_group_by_rounded_values_with_an_origin():
    data = (ranges([0.9, 1.1, 2.0]) >> group_by(rounded(col("distance"), 1, origin=0.5)) >> summarise(k=n())).collect()
    assert data.distance_rounded.values.tolist() == [0.5, 1.5, 2.5]
    assert data["k"].values.tolist() == [1, 1, 1]


def test_group_by_rounded_drops_missing_values_with_a_note_or_keeps_them_last():
    frame = ranges([0.2, NAN, 0.8])
    out = frame >> group_by(rounded(col("distance"), 0.5)) >> summarise(k=n())
    assert out.collect().distance_rounded.values.tolist() == [0.0, 1.0]
    assert ("changed: group_by(rounded(distance)) dropped 1 position with a missing distance; "
            "dropna=False keeps them as a group") in out._notes
    kept = (frame >> group_by(rounded(col("distance"), 0.5), dropna=False) >> summarise(k=n())).collect()
    labels = kept.distance_rounded.values.tolist()
    assert labels[:2] == [0.0, 1.0] and math.isnan(labels[2])
    assert kept["k"].values.tolist() == [1, 1, 1]


def test_group_by_cells():
    out = ranges([0.05, 0.19, 0.2, 0.41, -0.01]) >> group_by(cells(col("distance"), 0.2))
    assert "groups: cells(distance)" in repr(out)
    data = (out >> summarise(k=n())).collect()
    assert data.distance_cell.values.tolist() == ["[-0.2, 0)", "[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)"]
    assert data["k"].values.tolist() == [1, 2, 1, 1]


def test_group_by_cells_keeps_missing_values_as_a_text_label():
    data = (ranges([0.5, NAN, 1.5]) >> group_by(cells(col("distance"), 1, origin=0.5), dropna=False)
            >> summarise(k=n())).collect()
    assert data.distance_cell.values.tolist() == ["[0.5, 1.5)", "[1.5, 2.5)", "NaN"]
    assert data["k"].values.tolist() == [1, 1, 1]


def test_grouped_mutate_with_cells():
    out = (ranges([0.1, 0.2, 1.1, 1.3], v=[1.0, 3.0, 10.0, 20.0]) >> group_by(cells(col("distance"), 1))
           >> mutate(c=col("v") - mean(col("v")))).collect()
    assert out["c"].values.tolist() == [-1.0, 1.0, -5.0, 5.0]


def test_the_verb_prints_the_helper_as_bins_prints():
    assert repr(group_by(rounded(col("distance"), 0.5))) == "<verb group_by(rounded(distance))>"
    assert repr(group_by(cells(col("x"), 0.2), dropna=False)) == "<verb group_by(cells(x), dropna = False)>"


def test_helper_keys_hold_the_records():
    assert rounded(col("x"), 0.5).transform == Round(0.5)
    assert cells(col("x"), 0.2, origin=0.1).transform == GridCells(0.2, origin=0.1)


def test_helpers_refuse_bad_settings_naming_the_call():
    with pytest.raises(GroupError, match=r"rounded\(x, 0\): .*step"):
        rounded(col("x"), 0)
    with pytest.raises(GroupError, match=r"cells\(x, -1\): .*size"):
        cells(col("x"), -1)


def test_helpers_refuse_values_they_cannot_place():
    with pytest.raises(GroupError, match=r"rounded\(t, 1\): .*needs numbers"):
        stamps() >> group_by(rounded(col("t"), 1))
    with pytest.raises(GroupError, match="1 positions of distance are infinite"):
        ranges([1.0, math.inf]) >> group_by(cells(col("distance"), 1))


def test_several_transformed_keys_are_still_one_too_many():
    frame = tidy(xr.Dataset({"x": ("point", [0.1, 0.2]), "y": ("point", [0.3, 0.4])}))
    with pytest.raises(GroupError, match=r"more than one value field is not supported yet; got cells\(x\), cells\(y\)"):
        frame >> group_by(cells(col("x"), 1), cells(col("y"), 1))
