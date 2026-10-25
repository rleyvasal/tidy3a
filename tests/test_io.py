import numpy as np
import pytest
import xarray as xr

from tidy3a import ReadError, infer, read_array


def write(tmp_path, values, name="data.bin", dtype="<f4", header=b""):
    path = tmp_path / name
    path.write_bytes(header + np.asarray(values, dtype=dtype).tobytes())
    return path


def values_of(frame, name="values"):
    return frame.collect()[name]


# ── raw files: shapes and layouts ───────────────────────────────────────────


@pytest.mark.parametrize("shape", [(6,), (2, 3), (2, 3, 4), (2, 1, 3, 2), (1, 2, 3, 2, 2)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_raw_round_trip_any_rank(tmp_path, shape, order):
    expected = np.arange(np.prod(shape), dtype="float32").reshape(shape, order=order)
    path = tmp_path / "data.bin"
    expected.ravel(order=order).tofile(path)
    names = [f"a{i}" for i in range(len(shape))]
    frame = read_array(
        path, value_type="float32", dims=dict(zip(names, shape)), order=order,
    )
    out = values_of(frame)
    assert out.dims == tuple(names)
    np.testing.assert_array_equal(out.values, expected)


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_infer_works_on_any_axis(tmp_path, rank):
    shape = (2, 3, 1, 2, 2)[:rank]
    path = write(tmp_path, np.arange(np.prod(shape)))
    for position in range(rank):
        dims = {f"a{i}": (infer if i == position else size) for i, size in enumerate(shape)}
        frame = read_array(path, value_type="float32", dims=dims)
        assert values_of(frame).shape == shape


def test_point_cloud_records(tmp_path):
    records = np.arange(20, dtype="float32").reshape(4, 5)
    path = write(tmp_path, records)
    frame = read_array(
        path,
        value_type="float32",
        dims={"point": infer, "feature": 5},
        labels={"feature": ["x", "y", "z", "intensity", "ring"]},
        field_axis="feature",
    )
    out = values_of(frame)
    assert out.sizes == {"point": 4, "feature": 5}
    assert out.sel(feature="ring").values.tolist() == [4.0, 9.0, 14.0, 19.0]
    assert "inferred: point = 80 bytes / 4 bytes / 5 = 4" in repr(frame)


def test_big_endian(tmp_path):
    path = write(tmp_path, [1.5, -2.0, 3.25], dtype=">f8")
    frame = read_array(path, value_type="float64", dims={"t": infer}, byte_order="big")
    out = values_of(frame)
    assert out.values.tolist() == [1.5, -2.0, 3.25]
    assert out.dtype == np.float64 and out.dtype.isnative
    assert "big-endian" in repr(frame)


def test_offset_skips_header(tmp_path):
    path = write(tmp_path, [1, 2, 3, 4], header=b"HEADER!!")
    frame = read_array(path, value_type="float32", dims={"t": infer}, offset=8)
    assert values_of(frame).values.tolist() == [1, 2, 3, 4]
    assert "8-byte offset" in repr(frame)


def test_complex_values_are_interleaved_pairs(tmp_path):
    path = write(tmp_path, [1.0, 2.0, 3.0, -4.0])  # two complex64 values
    frame = read_array(path, value_type="complex64", dims={"sample": infer})
    assert values_of(frame).values.tolist() == [1 + 2j, 3 - 4j]


def test_integer_and_single_byte_types(tmp_path):
    path = write(tmp_path, [0, 128, 255], dtype="u1")
    frame = read_array(path, value_type="uint8", dims={"pixel": 3})
    assert values_of(frame).values.tolist() == [0, 128, 255]
    assert "uint8, row-major" in repr(frame)


# ── raw files: malformed input ──────────────────────────────────────────────


def test_partial_record_is_an_error(tmp_path):
    path = write(tmp_path, np.arange(11))
    with pytest.raises(ReadError, match="not a whole number of records") as info:
        read_array(path, value_type="float32", dims={"point": infer, "feature": 5})
    assert "5 feature × 4 bytes (float32) = 20 bytes" in str(info.value)


def test_partial_value_is_an_error(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x00" * 7)
    with pytest.raises(ReadError, match="not a whole number of float32 values"):
        read_array(path, value_type="float32", dims={"t": infer})


def test_wrong_total_without_infer_is_an_error(tmp_path):
    path = write(tmp_path, np.arange(12))
    with pytest.raises(ReadError, match="never truncated or padded"):
        read_array(path, value_type="float32", dims={"a": 2, "b": 5})


def test_two_infer_markers(tmp_path):
    path = write(tmp_path, np.arange(6))
    with pytest.raises(ReadError, match=r'two axes marked infer \("a", "b"\)'):
        read_array(path, value_type="float32", dims={"a": infer, "b": infer})


def test_raw_needs_value_type_and_sizes(tmp_path):
    path = write(tmp_path, np.arange(6))
    with pytest.raises(ReadError, match="value_type= is required"):
        read_array(path, dims={"a": 6})
    with pytest.raises(ReadError, match="needs sizes"):
        read_array(path, value_type="float32", dims=["a"])
    with pytest.raises(ReadError, match="dims= is required"):
        read_array(path, value_type="float32")


def test_non_numeric_value_type(tmp_path):
    path = write(tmp_path, np.arange(6))
    with pytest.raises(ReadError, match="not numeric"):
        read_array(path, value_type="U4", dims={"a": infer})


def test_conflicting_byte_order(tmp_path):
    path = write(tmp_path, np.arange(6))
    with pytest.raises(ReadError, match="state the byte order once"):
        read_array(path, value_type=">f4", dims={"a": infer})


def test_offset_past_end(tmp_path):
    path = write(tmp_path, np.arange(2))
    with pytest.raises(ReadError, match="past the end"):
        read_array(path, value_type="float32", dims={"a": infer}, offset=100)


@pytest.mark.parametrize("bad", [-1, 2.5, True, "3"])
def test_bad_sizes(tmp_path, bad):
    path = write(tmp_path, np.arange(6))
    with pytest.raises(ReadError, match="non-negative integer or infer"):
        read_array(path, value_type="float32", dims={"a": bad})


# ── empty observations ──────────────────────────────────────────────────────


def test_empty_file_gives_zero_observations(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    frame = read_array(
        path,
        value_type="float32",
        dims={"point": infer, "feature": 5},
        labels={"feature": ["x", "y", "z", "intensity", "ring"]},
    )
    assert values_of(frame).sizes == {"point": 0, "feature": 5}
    assert "unlabeled, empty (0 positions)" in repr(frame)


def test_empty_file_lazy(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    frame = read_array(path, value_type="float32", dims={"point": infer, "feature": 5}, lazy=True)
    assert values_of(frame).sizes == {"point": 0, "feature": 5}


def test_zero_record_width_cannot_infer(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(ReadError, match="cannot infer"):
        read_array(path, value_type="float32", dims={"point": infer, "feature": 0})


# ── labels ──────────────────────────────────────────────────────────────────


def test_label_count_must_match(tmp_path):
    path = write(tmp_path, np.arange(10))
    with pytest.raises(ReadError, match='label count for "feature" is 4 but its size is 5'):
        read_array(
            path, value_type="float32", dims={"point": infer, "feature": 5},
            labels={"feature": ["x", "y", "z", "i"]},
        )


def test_labels_must_be_unique_lists_on_known_axes(tmp_path):
    path = write(tmp_path, np.arange(6))
    dims = {"point": infer, "feature": 3}
    with pytest.raises(ReadError, match="repeat 'x'"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": ["x", "y", "x"]})
    with pytest.raises(ReadError, match="not one string"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": "x y z"})
    with pytest.raises(ReadError, match="not in dims"):
        read_array(path, value_type="float32", dims=dims, labels={"band": ["a", "b", "c"]})


def test_numeric_labels(tmp_path):
    path = write(tmp_path, np.arange(6))
    frame = read_array(
        path, value_type="float32", dims={"lon": 3, "lat": 2},
        labels={"lon": np.array([-1.5, 0.0, 1.5])},
    )
    assert values_of(frame).lon.values.tolist() == [-1.5, 0.0, 1.5]


def test_duplicate_axis_names_in_list(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((2, 2)))
    with pytest.raises(ReadError, match='duplicate axis name "x"'):
        read_array(path, dims=["x", "x"])


# ── .npy files ──────────────────────────────────────────────────────────────


def test_npy_reads_stored_type_and_shape(tmp_path):
    path = tmp_path / "sensors.npy"
    np.save(path, np.arange(6, dtype="int16").reshape(3, 2))
    frame = read_array(path, dims=["time", "sensor"], labels={"sensor": ["front", "rear"]})
    out = values_of(frame)
    assert out.dtype == np.int16 and out.sizes == {"time": 3, "sensor": 2}
    assert '.npy file "sensors.npy"' in repr(frame)


def test_npy_detected_by_header_not_suffix(tmp_path):
    path = tmp_path / "looks_raw.bin"
    with path.open("wb") as handle:
        np.save(handle, np.arange(4.0))
    frame = read_array(path, dims={"t": infer})
    assert values_of(frame).values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert "inferred: t = 4 from the stored shape" in repr(frame)


def test_npy_value_type_conflict(tmp_path):
    path = tmp_path / "sensors.npy"
    np.save(path, np.zeros(3, dtype="float32"))
    with pytest.raises(ReadError, match='value_type="float64" conflicts with the stored type float32'):
        read_array(path, value_type="float64", dims=["t"])
    assert values_of(read_array(path, value_type="float32", dims=["t"])).dtype == np.float32


def test_npy_shape_mismatches(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((3, 2)))
    with pytest.raises(ReadError, match="dims has 1 axes but"):
        read_array(path, dims=["t"])
    with pytest.raises(ReadError, match='"sensor" has size 3'):
        read_array(path, dims={"time": infer, "sensor": 3})


def test_npy_object_arrays_are_not_unpickled(tmp_path):
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ReadError, match="does not unpickle"):
        read_array(path, dims=["t"])


def test_raw_options_rejected_for_npy(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.zeros(3))
    with pytest.raises(ReadError, match="byte_order= applies only to raw files"):
        read_array(path, dims=["t"], byte_order="big")


def test_npy_lazy_memory_maps(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.arange(5.0))
    frame = read_array(path, dims=["t"], lazy=True)
    data = frame.to_xarray()["values"].data
    # Older xarray wraps the memmap as a plain read-only view of the same buffer.
    backing = data if isinstance(data, np.memmap) else data.base
    assert isinstance(backing, np.memmap) and not data.flags.writeable


# ── in-memory arrays ────────────────────────────────────────────────────────


def test_in_memory_array(tmp_path):
    frame = read_array(np.array([1 + 0j, 0j]), dims={"input": 2}, labels={"input": [0, 1]}, name="amplitude")
    out = values_of(frame, "amplitude")
    assert out.dtype == np.complex128 and out.input.values.tolist() == [0, 1]
    assert "source: NumPy array" in repr(frame)


def test_in_memory_array_rejects_lazy():
    with pytest.raises(ReadError, match="already loaded"):
        read_array(np.zeros(3), dims=["t"], lazy=True)


def test_unsupported_source():
    with pytest.raises(TypeError, match="use tidy\\(\\) for xarray objects"):
        read_array(xr.DataArray([1.0]), dims=["t"])


# ── lazy raw files ──────────────────────────────────────────────────────────


def test_lazy_raw_is_memory_mapped_read_only_and_uncopied(tmp_path):
    path = write(tmp_path, np.arange(10))
    frame = read_array(path, value_type="float32", dims={"point": infer, "feature": 2}, lazy=True)
    data = frame.to_xarray()["values"].data
    assert isinstance(data, np.ndarray) and not data.flags.writeable
    assert np.shares_memory(frame.collect()["values"].data, data)
    assert "memory-mapped" in repr(frame)


# ── field axis and variables_from ───────────────────────────────────────────


FIELDS = ["x", "y", "z", "intensity", "ring"]


def cloud_path(tmp_path):
    return write(tmp_path, np.arange(20))


def test_field_axis_marked_in_inspection(tmp_path):
    frame = read_array(
        cloud_path(tmp_path), value_type="float32", dims={"point": infer, "feature": 5},
        labels={"feature": FIELDS}, field_axis="feature",
    )
    assert "field axis (labels act as columns)" in repr(frame)
    assert "positions 1 … 4" in repr(frame)


def test_field_axis_needs_labels_and_existing_axis(tmp_path):
    path = cloud_path(tmp_path)
    dims = {"point": infer, "feature": 5}
    with pytest.raises(ReadError, match="has no labels"):
        read_array(path, value_type="float32", dims=dims, field_axis="feature")
    with pytest.raises(ReadError, match="is not an axis"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": FIELDS}, field_axis="band")


def test_variables_from_splits_and_records_origin(tmp_path):
    frame = read_array(
        cloud_path(tmp_path), value_type="float32", dims={"point": infer, "feature": 5},
        labels={"feature": FIELDS}, variables_from="feature",
    )
    out = frame.collect()
    assert list(out.data_vars) == FIELDS
    assert out["ring"].dims == ("point",) and out["ring"].values.tolist() == [4, 9, 14, 19]
    assert frame._origins["y"] == ("feature", tuple(FIELDS))
    assert "from axis feature (variables_from)" in repr(frame)


def test_variables_from_rules(tmp_path):
    path = cloud_path(tmp_path)
    dims = {"point": infer, "feature": 5}
    with pytest.raises(ReadError, match="needs labels"):
        read_array(path, value_type="float32", dims=dims, variables_from="feature")
    with pytest.raises(ReadError, match="use one of them"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": FIELDS},
                   variables_from="feature", name="cloud")
    with pytest.raises(ReadError, match="turns into variables"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": FIELDS},
                   variables_from="feature", field_axis="feature")
    with pytest.raises(ReadError, match="needs text labels"):
        read_array(path, value_type="float32", dims=dims, labels={"feature": [1, 2, 3, 4, 5]},
                   variables_from="feature")


def test_variables_from_with_a_different_field_axis(tmp_path):
    path = write(tmp_path, np.arange(12))
    frame = read_array(
        path, value_type="float32", dims={"time": infer, "sensor": 2, "feature": 3},
        labels={"sensor": ["front", "rear"], "feature": ["x", "y", "z"]},
        variables_from="feature", field_axis="sensor",
    )
    assert frame.collect()["x"].dims == ("time", "sensor")
    assert frame._field_axis == "sensor"


def test_name_names_the_single_array(tmp_path):
    frame = read_array(cloud_path(tmp_path), value_type="float32", dims={"point": infer, "f": 5}, name="cloud")
    assert list(frame.collect().data_vars) == ["cloud"]


# ── Labels of one kind (the exact label rule) ───────────────────────────────


@pytest.mark.parametrize("labels, message", [
    ([1, "a"], r"mix numbers and text \(int, str\)"),             # NumPy would make both text: "1", "a"
    (np.array([1, "a"], dtype=object), r"mix numbers and text \(int, str\)"),
    ([1.5, "x"], r"mix numbers and text \(float, str\)"),
    ([True, 2], r"mix booleans and numbers \(bool, int\)"),      # NumPy would make True the number 1
    (["a", b"b"], r"mix bytes and text \(bytes, str\)"),
    ([1, None], r"mix None and numbers \(NoneType, int\)"),
    ([np.datetime64("2026-01-01"), 3], r"mix numbers and times \(datetime64, int\)"),
    ([np.timedelta64(1, "s"), 3], r"mix durations and numbers \(int, timedelta64\)"),
])
def test_labels_of_mixed_kinds_are_one_error_not_converted(labels, message):
    with pytest.raises(ReadError, match=r'labels for "k" ' + message):
        read_array(np.zeros((2, 3)), dims={"k": 2, "p": infer}, labels={"k": labels})


def test_integers_and_floats_mix_only_when_float64_holds_every_label():
    frame = read_array(np.zeros((2, 3)), dims={"k": 2, "p": infer}, labels={"k": [1, 2.5]})
    assert frame.to_dict()["coords"]["k"]["data"].tolist() == [1.0, 2.5]
    with pytest.raises(ReadError, match=r'labels for "k" mix integers and floats that float64 cannot hold exactly'):
        read_array(np.zeros((2, 3)), dims={"k": 2, "p": infer}, labels={"k": [2**53 + 1, 0.5]})


@pytest.mark.parametrize("labels, expected", [
    (["x", "y"], ["x", "y"]),
    ([3, 1], [3, 1]),
    ([np.int64(1), 2], [1, 2]),                                    # NumPy and Python integers are one kind
    ([0.5, np.float32(1.5)], [0.5, 1.5]),
    ([True, False], [True, False]),
    ([b"a", b"b"], [b"a", b"b"]),
    ([np.datetime64("2026-01-01", "ns"), np.datetime64("2026-01-02", "ns")],
     [np.datetime64("2026-01-01", "ns"), np.datetime64("2026-01-02", "ns")]),
])
def test_labels_of_one_kind_are_kept_as_before(labels, expected):
    frame = read_array(np.zeros((2, 3)), dims={"k": 2, "p": infer}, labels={"k": labels})
    np.testing.assert_array_equal(frame.to_dict()["coords"]["k"]["data"], np.array(expected))
