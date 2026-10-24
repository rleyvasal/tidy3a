import numpy as np
import pytest
import xarray as xr

from tidy3a import col, full_join, inner_join, left_join, mean, mutate, select, slice, summarise, tidy, var


@pytest.fixture
def measurements():
    data = xr.Dataset(
        {"temperature": ("sample", np.array([273, 274, 275, 276], dtype="float32"), {"units": "K"}),
         "quality": ("sample", np.array([0, 1, 0, 1], dtype="int16"), {"long_name": "quality flag"})},
        coords={"sample": ("sample", np.array([101, 102, 103, 104], dtype="int32"),
                           {"long_name": "sample identifier"})},
        attrs={"title": "Sensor measurements", "Conventions": "CF-1.8"},
    )
    data.temperature.encoding = {"dtype": np.dtype("int16"), "scale_factor": 0.5,
                                 "add_offset": 273.0, "_FillValue": np.int16(-32768)}
    data.sample.encoding = {"dtype": np.dtype("int32")}
    data.encoding = {"unlimited_dims": {"sample"}}
    return data


@pytest.mark.parametrize("method", ["to_xarray", "collect"])
def test_dataset_metadata_and_types_survive_wrapping(measurements, method):
    result = getattr(tidy(measurements), method)()
    xr.testing.assert_identical(result, measurements)
    assert result.encoding == measurements.encoding
    for name in measurements.variables:
        assert result[name].dtype == measurements[name].dtype
        assert result[name].encoding == measurements[name].encoding
    assert result.temperature.dtype == np.dtype("float32")  # Serialization dtype is not a cast.


def test_dataarray_metadata_belongs_to_its_variable(measurements):
    source = measurements.temperature
    result = tidy(source).to_xarray()
    xr.testing.assert_identical(result.temperature, source)
    assert result.temperature.encoding == source.encoding
    assert result.sample.encoding == source.sample.encoding


def test_shallow_view_has_independent_metadata_dictionaries(measurements):
    frame = tidy(measurements)
    view = frame.to_xarray()
    view.attrs["title"] = "Changed title"
    view.temperature.attrs["units"] = "degC"
    view.temperature.encoding["dtype"] = np.dtype("int8")
    assert measurements.attrs["title"] == "Sensor measurements"
    assert measurements.temperature.attrs["units"] == "K"
    assert frame.to_xarray().temperature.encoding["dtype"] == np.dtype("int16")
    assert np.shares_memory(view.temperature.data, measurements.temperature.data)


@pytest.mark.parametrize("positions, values, labels", [
    ([2, 4], [274, 276], [102, 104]),
    ([4, 2], [276, 274], [104, 102]),
    ([4, 1], [276, 273], [104, 101]),
    ([1], [273], [101]),
])
def test_netcdf_values_stay_lazy_until_collection(tmp_path, monkeypatch, measurements, positions, values, labels):
    pytest.importorskip("scipy")
    from xarray.backends.scipy_ import ScipyArrayWrapper

    path = tmp_path / "measurements.nc"
    measurements.to_netcdf(path, engine="scipy")
    reads = []
    original = ScipyArrayWrapper.__getitem__

    def track_reads(array, key):
        reads.append(array.variable_name)
        return original(array, key)

    with xr.open_dataset(path, engine="scipy", cache=False) as opened:
        monkeypatch.setattr(ScipyArrayWrapper, "__getitem__", track_reads)
        frame = tidy(opened)
        assert "temperature" in repr(frame)
        selected = frame >> select("temperature") >> slice(sample=positions)
        view = selected.to_xarray()
        assert view.attrs == opened.attrs and view.temperature.attrs == opened.temperature.attrs
        assert view.temperature.encoding == opened.temperature.encoding
        assert view.temperature.dtype == opened.temperature.dtype
        assert reads == []
        result = selected.collect()
        assert set(reads) == {"temperature"}
        assert result.temperature.values.tolist() == values
        assert result.sample.values.tolist() == labels
        assert result.temperature.encoding == opened.temperature.encoding


@pytest.mark.parametrize("keep_attrs", [False, True])
@pytest.mark.parametrize("target", ["temperature", "celsius"])
def test_computed_values_clear_variable_metadata(measurements, keep_attrs, target):
    with xr.set_options(keep_attrs=keep_attrs):
        result = (tidy(measurements) >> mutate(**{target: col("temperature") - 273})).collect()
    assert result[target].values.tolist() == [0, 1, 2, 3]
    assert result[target].attrs == {} and result[target].encoding == {}
    assert result.attrs == measurements.attrs and result.sample.attrs == measurements.sample.attrs
    assert result.sample.encoding == measurements.sample.encoding
    assert result.quality.attrs == measurements.quality.attrs
    assert measurements.temperature.attrs == {"units": "K"}
    assert measurements.temperature.encoding["dtype"] == np.dtype("int16")


@pytest.mark.parametrize("explicit_array", [False, True])
def test_aliases_and_explicit_arrays_keep_attributes_but_clear_encoding(measurements, explicit_array):
    value = measurements.temperature if explicit_array else col("temperature")
    result = (tidy(measurements) >> mutate(copied=value)).collect()
    assert result.copied.attrs == measurements.temperature.attrs
    assert result.copied.encoding == {} and measurements.temperature.encoding
    assert result.sample.encoding == measurements.sample.encoding
    assert np.shares_memory(result.copied.data, measurements.temperature.data)


@pytest.mark.parametrize("keep_attrs", [False, True])
def test_summaries_keep_dataset_context_and_clear_computed_metadata(measurements, keep_attrs):
    with xr.set_options(keep_attrs=keep_attrs):
        result = (tidy(measurements) >> summarise(average=mean(col("temperature")),
                                                spread=var(col("temperature")))).collect()
    assert result.attrs == measurements.attrs and result.encoding == {}
    assert result.average.item() == 274.5 and result.spread.item() == 1.25
    for variable in result.data_vars.values():
        assert variable.attrs == {} and variable.encoding == {}


@pytest.fixture
def encoded_positions():
    array = xr.DataArray(np.array([[1, 2], [3, 4]], dtype="int16"), dims=("sample", "feature"),
                         coords={"feature": ["x", "y"]}, attrs={"units": "m"})
    array.encoding = {"dtype": np.dtype("int16"), "_FillValue": np.int16(-32768)}
    array.feature.attrs = {"long_name": "component"}
    return tidy(array, field_axis="feature")


def test_field_write_back_clears_holder_metadata(encoded_positions):
    changed = encoded_positions >> mutate(x=col("x") / 2)
    result = changed.collect()["values"]
    assert result.sel(feature="x").values.tolist() == [0.5, 1.5]
    assert result.sel(feature="y").values.tolist() == [2, 4]
    assert result.attrs == {} and result.encoding == {}
    assert result.feature.attrs == {"long_name": "component"}
    assert encoded_positions.collect()["values"].attrs == {"units": "m"}
    assert "cleared attributes and encoding for values after field write-back" in repr(changed)


def test_field_write_back_serializes_fractional_values_without_old_integer_encoding(tmp_path, encoded_positions):
    pytest.importorskip("scipy")
    result = (encoded_positions >> mutate(x=col("x") / 2)).collect()
    path = tmp_path / "fractional.nc"
    result.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as restored:
        xr.testing.assert_allclose(restored["values"], result["values"])


@pytest.mark.parametrize("join", [left_join, inner_join, full_join])
def test_joins_report_dropped_metadata_conflicts(join):
    left = xr.Dataset({"distance": ("point", [1.0, 2.0], {"units": "m"})},
                      coords={"point": [1, 2]}, attrs={"title": "left", "project": "survey"})
    right = xr.Dataset({"elapsed": ("point", [3.0, 4.0], {"units": "s"})},
                       coords={"point": [1, 2]}, attrs={"title": "right", "project": "survey"})
    left.point.attrs = {"long_name": "left index", "units": "count"}
    right.point.attrs = {"long_name": "right index", "units": "count"}
    left.distance.encoding = {"dtype": np.dtype("int16")}
    left.encoding = {"unlimited_dims": {"point"}}
    frame = tidy(left) >> join(right)
    result = frame.collect()
    assert result.attrs == {"project": "survey"}
    assert result.point.attrs == {"units": "count"}
    assert result.distance.attrs == {"units": "m"} and result.elapsed.attrs == {"units": "s"}
    assert result.encoding == {} and all(not variable.encoding for variable in result.variables.values())
    assert "dropped conflicting attributes for dataset: title" in repr(frame)
    assert "dropped conflicting attributes for coordinate point: long_name" in repr(frame)
    assert left.attrs["title"] == "left" and left.distance.encoding["dtype"] == np.dtype("int16")
