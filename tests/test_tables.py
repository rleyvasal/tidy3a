"""Tables to and from tidy3a: Polars and tidy3 (plan milestone 8, part 2).

``as_table()`` gives a Polars DataFrame with one row per position on the
observation axis: the axis's labels (original positions after a filter, as
``pull`` gives them) in a column named after it, the field-axis labels as
columns, and the other variables along the axis as columns. ``long=True``
flattens a grid into one row per cell. ``tidy(table, axis=...)`` reads a
Polars DataFrame or LazyFrame, or a tidy3 frame, back: each column a
variable along the axis, a column named after the axis its labels, and
``fields=`` packing columns into one array with a field axis. A
``TidyArray`` defines ``__tidy__``, so tidy3's ``tidy()`` accepts it.
"""

import sys
import types

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    ReadError,
    TidyError,
    as_table,
    col,
    dot,
    fields,
    filter,
    group_by,
    infer,
    mean,
    mutate,
    pull,
    read_array,
    summarise,
    tidy,
)

pl = pytest.importorskip("polars")

x, y, z, intensity, ring = fields("x", "y", "z", "intensity", "ring")
RECORDS = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float32")
FIELDS = ["x", "y", "z", "intensity", "ring"]


def cloud():
    return read_array(RECORDS, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, field_axis="feature")


def stations():
    times = np.datetime64("2026-01-01T00", "s") + np.arange(3) * np.timedelta64(3600, "s")
    return tidy(xr.Dataset(
        {"temp": ("time", np.array([1.5, 2.5, 3.5], "float32")),
         "count": ("time", np.array([3, 4, 5], "int16")),
         "site": ("time", np.array(["a", "b", "c"], dtype=object))},
        coords={"time": times.astype("datetime64[ns]")},
    ))


# ── as_table ────────────────────────────────────────────────────────────────


def test_one_row_per_point_with_fields_as_columns():
    table = cloud() >> as_table()
    assert isinstance(table, pl.DataFrame)
    assert table.columns == ["point", *FIELDS]
    assert table["point"].to_list() == [1, 2, 3, 4]  # unlabelled axis: 1-based positions, as pull gives
    assert table["x"].dtype == pl.Float32
    np.testing.assert_array_equal(table["intensity"].to_numpy(), RECORDS[:, 3])


def test_filtered_points_keep_their_original_positions():
    near = cloud() >> filter(col("x") < 10)
    table = near >> as_table()
    assert table["point"].to_list() == (near >> pull("point")).tolist() == [1, 2, 4]


def test_other_variables_along_the_axis_become_columns_with_their_types():
    table = stations() >> as_table()
    assert table.columns == ["time", "temp", "count", "site"]
    assert table.schema["time"] == pl.Datetime("ns") and table.schema["count"] == pl.Int16
    assert table.schema["site"] == pl.String and table["site"].to_list() == ["a", "b", "c"]


def test_times_and_durations_in_units_polars_lacks_are_converted_exactly():
    seconds = np.datetime64("2026-01-01T00:00:01", "s") + np.arange(2) * np.timedelta64(1, "s")
    frame = read_array(np.arange(2.0), dims=["t"], labels={"t": seconds})
    frame = frame >> mutate(gap=col("t") - col("t"))
    table = frame >> as_table()
    assert table.schema["t"] == pl.Datetime("ms")
    np.testing.assert_array_equal(table["t"].to_numpy(), seconds)
    assert table.schema["gap"] == pl.Duration("ms")


def test_grouping_does_not_change_the_table():
    assert (cloud() >> group_by(ring) >> as_table()).equals(cloud() >> as_table())


def test_more_than_one_observation_axis_needs_long():
    grid = read_array(np.zeros((3, 4)), dims=["time", "lat"])
    with pytest.raises(TidyError, match=r"(?s)one observation axis.*time \(3\), lat \(4\).*as_table\(long=True\).*12 rows"):
        grid >> as_table()


def test_a_variable_without_the_observation_axis_is_an_error_naming_its_axes():
    data = tidy(xr.Dataset({"v": ("point", [1.0, 2.0]), "scale": ((), 3.0)}))
    with pytest.raises(TidyError, match=r'"scale" has no point axis \(its axes: none\)'):
        data >> as_table()


def test_complex_values_and_repeated_column_names_are_refused():
    complex_frame = read_array(np.array([1 + 2j, 3 - 1j]), dims=["k"])
    with pytest.raises(TidyError, match=r"complex.*real\(\).*imag\(\)"):
        complex_frame >> as_table()
    clash = read_array(np.zeros((2, 2), "float32"), dims=["x", "feature"], labels={"feature": ["x", "y"]},
                       field_axis="feature")
    with pytest.raises(TidyError, match='column "x" twice'):
        clash >> as_table()


def test_long_tables_have_one_row_per_cell_and_broadcast_what_lacks_an_axis():
    data = tidy(xr.Dataset(
        {"air": (("time", "lat"), np.arange(6.0).reshape(3, 2)), "weight": ("lat", [0.25, 0.75])},
        coords={"time": [10, 20, 30], "lat": [-5.0, 5.0]},
    ))
    table = data >> as_table(long=True)
    assert table.columns == ["time", "lat", "air", "weight"]
    assert table.height == 6
    assert table["time"].to_list() == [10, 10, 20, 20, 30, 30]
    assert table["lat"].to_list() == [-5.0, 5.0] * 3
    assert table["air"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert table["weight"].to_list() == [0.25, 0.75] * 3


def test_the_verb_prints_as_called():
    assert repr(as_table(long=True)) == "<verb as_table(long = True)>"


# ── tidy(table, axis=...) ───────────────────────────────────────────────────


def test_each_column_becomes_a_variable_and_the_axis_column_its_labels():
    table = pl.DataFrame({"point": [10, 20, 30], "x": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
    frame = tidy(table, axis="point")
    assert (frame >> pull("point")).tolist() == [10, 20, 30]
    assert (frame >> pull("x")).tolist() == [1.0, 2.0, 3.0]
    assert (frame >> pull("name")).tolist() == ["a", "b", "c"]
    assert "source: Polars DataFrame" in repr(frame)


def test_without_an_axis_column_the_axis_has_no_labels():
    frame = tidy(pl.DataFrame({"x": [1.0, 2.0]}), axis="point")
    assert (frame >> pull("point")).tolist() == [1, 2]  # positions, as for any unlabelled axis
    assert (frame >> mutate(d=col("x") * 2) >> pull("d")).tolist() == [2.0, 4.0]


def test_fields_pack_columns_into_one_array_with_a_field_axis():
    table = pl.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0], "z": [0.0, 2.0], "id": [7, 8]},
                         schema_overrides={"x": pl.Float32, "y": pl.Float32, "z": pl.Float32})
    frame = tidy(table, axis="point", field_axis="feature", fields=["x", "y", "z"])
    assert (frame >> pull("feature")).tolist() == ["x", "y", "z"]
    assert (frame >> pull(x)).dtype == np.float32
    r = xr.DataArray(np.eye(3, dtype="float32"), dims=("i", "feature"),
                     coords={"i": ["x", "y", "z"], "feature": ["x", "y", "z"]})
    rotated = frame >> mutate(p=dot(r, [x, y, z], over="feature")) >> pull("p")
    np.testing.assert_array_equal(rotated, np.array([[3, 0], [4, 0], [0, 2]], "float32"))  # (i, point)
    assert (frame >> pull("id")).tolist() == [7, 8]


def test_fields_need_a_field_axis_and_one_type():
    table = pl.DataFrame({"x": [1.0], "y": [2]})
    with pytest.raises(ReadError, match="fields= needs field_axis="):
        tidy(table, axis="point", fields=["x", "y"])
    with pytest.raises(ReadError, match=r"field_axis= needs fields="):
        tidy(table, axis="point", field_axis="feature")
    with pytest.raises(ReadError, match=r"fields x, y have different types \(Float64, Int64\)"):
        tidy(table, axis="point", field_axis="feature", fields=["x", "y"])
    with pytest.raises(ReadError, match='no column "w"'):
        tidy(table, axis="point", field_axis="feature", fields=["x", "w"])


def test_lazy_frames_and_tidy3_frames_are_collected_with_a_note():
    lazy = pl.DataFrame({"x": [1.0, 2.0]}).lazy()
    frame = tidy(lazy, axis="point")
    assert "materialized: collected the Polars LazyFrame" in repr(frame)

    class Tidy3Like:
        def to_polars(self):
            return pl.DataFrame({"x": [5.0]})

    assert "materialized: collected the tidy3 frame" in repr(tidy(Tidy3Like(), axis="point"))


def test_types_map_to_numpy_and_nulls_follow_the_rules():
    table = pl.DataFrame({
        "f": pl.Series([1.0, None], dtype=pl.Float32),
        "i": pl.Series([1, None], dtype=pl.Int64),
        "small": pl.Series([1, None], dtype=pl.Int16),
        "b": pl.Series([True, None]),
        "kind": pl.Series(["a", "b"], dtype=pl.Categorical),
        "day": pl.Series(np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")),
        "stamp": pl.Series(np.array(["2026-01-01T00:00", "NaT"], dtype="datetime64[us]")),
        "gap": pl.Series(np.array([1, 2], dtype="timedelta64[ms]")),
    })
    frame = tidy(table, axis="row")
    values = {name: frame >> pull(name) for name in table.columns}
    assert values["f"].dtype == np.float32 and np.isnan(values["f"][1])
    assert values["i"].dtype == np.float64 and np.isnan(values["i"][1])
    assert values["small"].dtype == np.float32  # as xarray promotes integers of up to 16 bits
    assert values["b"].dtype == np.float64 and values["b"][0] == 1.0 and np.isnan(values["b"][1])
    assert values["kind"].tolist() == ["a", "b"]
    # Dates stay dates; xarray stores them at its own resolution (seconds or nanoseconds), NumPy as days.
    np.testing.assert_array_equal(values["day"], np.array(["2026-01-01", "2026-01-02"], "datetime64[D]"))
    assert values["stamp"][0] == np.datetime64("2026-01-01T00:00") and np.isnat(values["stamp"][1])
    np.testing.assert_array_equal(values["gap"], np.array([1, 2], dtype="timedelta64[ms]"))
    text = repr(frame)
    assert "changed: i had 1 missing value; stored as float64 with NaN" in text
    assert "changed: b had 1 missing value; stored as float64 with NaN" in text


@pytest.mark.parametrize(("column", "words"), [
    (pl.Series("v", [[1, 2], [3]]), "a list column"),
    (pl.Series("v", [{"a": 1}, {"a": 2}]), "a struct column"),
    (pl.Series("v", [b"a", b"b"]), "a binary column"),
    (pl.Series("v", [[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)), "a fixed-size array column"),
    (pl.Series("v", np.array(["2026-01-01"], "datetime64[ns]")).dt.replace_time_zone("UTC"), "time zone"),
])
def test_columns_tidy3a_cannot_hold_are_errors_naming_them(column, words):
    with pytest.raises(ReadError, match=rf'column "v" is {words}' if "column" in words else words):
        tidy(pl.DataFrame([column]), axis="row")


def test_missing_labels_and_unknown_inputs_are_errors():
    with pytest.raises(ReadError, match='the "point" column has 1 missing value'):
        tidy(pl.DataFrame({"point": [1, None], "x": [1.0, 2.0]}), axis="point")
    with pytest.raises(ReadError, match=r"axis= names the axis the rows run along"):
        tidy(pl.DataFrame({"x": [1.0]}))
    with pytest.raises(TidyError, match="axis= and fields= are for tables"):
        tidy(xr.Dataset({"v": ("t", [1.0])}), axis="t")


# ── Round trips ─────────────────────────────────────────────────────────────


def test_a_table_round_trip_gives_back_the_same_data():
    original = stations()
    back = tidy(original >> as_table(), axis="time")
    xr.testing.assert_identical(back.collect(), original.collect())


def test_a_point_cloud_round_trips_with_its_fields():
    original = cloud() >> filter(col("x") < 10)
    back = tidy(original >> as_table(), axis="point", field_axis="feature", fields=FIELDS)
    np.testing.assert_array_equal(back >> pull("values"), original >> pull("values"))
    assert (back >> pull("point")).tolist() == [1, 2, 4]
    left = (original >> group_by(ring) >> summarise(m=mean(intensity))).collect()
    right = (back >> group_by(ring) >> summarise(m=mean(intensity))).collect()
    xr.testing.assert_identical(left, right)


# ── tidy3 ───────────────────────────────────────────────────────────────────


def test_dunder_tidy_hands_tidy3_the_table(monkeypatch):
    seen = {}
    fake = types.ModuleType("tidy3")

    def fake_tidy(table):
        seen["table"] = table
        return "tidy3 frame"

    fake.tidy = fake_tidy
    monkeypatch.setitem(sys.modules, "tidy3", fake)
    assert cloud().__tidy__() == "tidy3 frame"
    assert seen["table"].equals(cloud() >> as_table())


def test_dunder_tidy_without_tidy3_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "tidy3", None)
    with pytest.raises(TidyError, match="pip install tidy3"):
        cloud().__tidy__()


def test_tidy3s_tidy_accepts_a_tidy3a_frame():
    tidy3 = pytest.importorskip("tidy3")
    frame = tidy3.tidy(cloud())
    assert frame.collect().height == 4
    back = tidy(frame, axis="point", field_axis="feature", fields=FIELDS)
    np.testing.assert_array_equal(back >> pull("values"), cloud() >> pull("values"))
