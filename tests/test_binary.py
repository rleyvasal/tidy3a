import numpy as np
import pytest

from tidy3a import ReadError, col, filter, fixed_text, mean, mutate, n, read_binary, summarise


@pytest.mark.parametrize("byte_order, raw", [
    ("little", "0000c03f0700000020c00c00"),
    ("big", "3fc000000007c0200000000c"),
])
def test_packed_fields_preserve_values_and_types(tmp_path, byte_order, raw):
    path = tmp_path / "records.bin"
    path.write_bytes(bytes.fromhex(raw))
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                        dim="point", byte_order=byte_order)
    data = frame.collect()
    assert list(data.data_vars) == ["x", "ring"]
    assert data.x.dims == data.ring.dims == ("point",)
    assert data.x.dtype == np.dtype("float32") and data.x.values.tolist() == [1.5, -2.5]
    assert data.ring.dtype == np.dtype("uint16") and data.ring.values.tolist() == [7, 12]
    assert "record: 6 bytes; x: float32 at byte 0, ring: uint16 at byte 4" in repr(frame)
    assert "inferred: point = 12 bytes / 6 bytes = 2" in repr(frame)


@pytest.mark.parametrize("size", [5, 7])
@pytest.mark.parametrize("lazy", [False, True])
def test_incomplete_record_is_rejected(tmp_path, size, lazy):
    path = tmp_path / "partial.bin"
    path.write_bytes(bytes(size))
    with pytest.raises(ReadError, match="not a whole number of records of 6 bytes"):
        read_binary(path, value_types={"x": "float32", "ring": "uint16"}, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_records_keep_declared_types(tmp_path, lazy):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    data = read_binary(path, value_types={"x": "float32", "ring": "uint16"}, lazy=lazy).collect()
    assert data.sizes == {"record": 0}
    assert data.x.dtype == np.dtype("float32") and data.ring.dtype == np.dtype("uint16")


def test_numeric_records_work_with_the_existing_verbs(tmp_path):
    path = tmp_path / "records.bin"
    path.write_bytes(bytes.fromhex("0000c03f0700000020c00c00"))
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16"})
    result = (frame >> filter(col("ring") == 7) >> mutate(twice=col("x") * 2)
              >> summarise(count=n(), average=mean(col("twice")))).collect()
    assert result["count"].item() == 1 and result["average"].item() == 3.0


@pytest.mark.parametrize("schema", [{}, [], {"": "float32"}, {1: "float32"}, {"record": "float32"}])
def test_invalid_field_mapping_is_rejected(tmp_path, schema):
    with pytest.raises(ReadError):
        read_binary(tmp_path / "unused.bin", value_types=schema)


@pytest.mark.parametrize("value_type", ["object", "S4", ("f4", (2,)), [("nested", "f4")]])
def test_non_scalar_numeric_fields_are_rejected(tmp_path, value_type):
    with pytest.raises(ReadError, match="not numeric"):
        read_binary(tmp_path / "unused.bin", value_types={"x": value_type})


@pytest.mark.parametrize("value_type", [">f4", np.dtype(">f4")])
def test_conflicting_explicit_endian_marker_is_rejected(tmp_path, value_type):
    with pytest.raises(ReadError, match="state the byte order once"):
        read_binary(tmp_path / "unused.bin", value_types={"x": value_type})


@pytest.mark.parametrize("byte_order, raw", [
    ("little", "aabb07000000c03fccddeeff0c00000020c01122"),
    ("big", "aabb00073fc00000ccddeeff000cc02000001122"),
])
def test_offsets_preserve_mapping_order_and_skip_padding(tmp_path, byte_order, raw):
    path = tmp_path / "padded.bin"
    path.write_bytes(bytes.fromhex(raw))
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                        field_offsets={"ring": 2, "x": 4}, record_size=10, byte_order=byte_order)
    data = frame.collect()
    assert list(data.data_vars) == ["x", "ring"]
    assert data.x.dtype == np.dtype("float32") and data.x.values.tolist() == [1.5, -2.5]
    assert data.ring.dtype == np.dtype("uint16") and data.ring.values.tolist() == [7, 12]
    assert "record: 10 bytes; x: float32 at byte 4, ring: uint16 at byte 2" in repr(frame)


def test_record_size_is_inferred_from_the_last_occupied_byte(tmp_path):
    path = tmp_path / "gap.bin"
    path.write_bytes(bytes.fromhex("0000c03fdeadbeef0700"))
    data = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                       field_offsets={"x": 0, "ring": 8}).collect()
    assert data.x.item() == 1.5 and data.ring.item() == 7


def test_record_size_alone_adds_trailing_padding(tmp_path):
    path = tmp_path / "trailing.bin"
    path.write_bytes(bytes.fromhex("0000c03f0700aabbccdd000020c00c00eeff1122"))
    data = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                       record_size=np.int64(10)).collect()
    assert data.x.values.tolist() == [1.5, -2.5] and data.ring.values.tolist() == [7, 12]


@pytest.mark.parametrize("offsets, message", [
    ({"x": 0}, "every field"),
    ({"x": 0, "ring": 4, "extra": 6}, "extra fields"),
    ([], "every field"),
    ({"x": -1, "ring": 4}, "non-negative integer"),
    ({"x": True, "ring": 4}, "non-negative integer"),
    ({"x": 0.5, "ring": 4}, "non-negative integer"),
    ({"x": 0, "ring": 2}, "overlap"),
])
def test_invalid_field_offsets_are_rejected(tmp_path, offsets, message):
    with pytest.raises(ReadError, match=message):
        read_binary(tmp_path / "unused.bin", value_types={"x": "float32", "ring": "uint16"},
                    field_offsets=offsets)


@pytest.mark.parametrize("size", [0, -1, True, 3.5, 5])
def test_invalid_record_size_is_rejected(tmp_path, size):
    with pytest.raises(ReadError, match="record_size must be a positive integer covering all fields"):
        read_binary(tmp_path / "unused.bin", value_types={"x": "float32", "ring": "uint16"},
                    record_size=size)


def test_complete_fields_without_required_trailing_padding_are_rejected(tmp_path):
    path = tmp_path / "incomplete-padding.bin"
    path.write_bytes(bytes.fromhex("0000c03f0700"))
    with pytest.raises(ReadError, match="not a whole number of records of 8 bytes"):
        read_binary(path, value_types={"x": "float32", "ring": "uint16"}, record_size=8)


def test_empty_padded_records_keep_their_types(tmp_path):
    path = tmp_path / "empty-padded.bin"
    path.write_bytes(b"")
    data = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                       field_offsets={"x": np.int64(2), "ring": 8}, record_size=12).collect()
    assert data.sizes == {"record": 0}
    assert data.x.dtype == np.dtype("float32") and data.ring.dtype == np.dtype("uint16")


@pytest.mark.parametrize("byte_order, first, second", [
    ("little", "0000c03f0700", "000020c00c00"),
    ("big", "3fc000000007", "c0200000000c"),
])
def test_mixed_numeric_and_multibyte_text_records(tmp_path, byte_order, first, second):
    path = tmp_path / "text.bin"
    path.write_bytes(bytes.fromhex(first) + b"caf\xc3\xa9"
                     + bytes.fromhex(second) + b"\xe7\x8c\xab\0\0")
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16",
                                          "label": fixed_text(5, encoding="UTF_8")},
                        byte_order=byte_order)
    data = frame.collect()
    assert list(data.data_vars) == ["x", "ring", "label"]
    assert data.label.values.tolist() == ["café", "猫"]
    assert data.x.dtype == np.dtype("float32") and data.x.values.tolist() == [1.5, -2.5]
    assert data.ring.dtype == np.dtype("uint16") and data.ring.values.tolist() == [7, 12]
    assert "record: 11 bytes" in repr(frame)
    assert "materialized: label decoded as utf-8 text" in repr(frame)


@pytest.mark.parametrize("raw, padding, expected", [
    (b" a \0", b"\0", " a "),
    (b"a\0b\0", b"\0", "a\0b"),
    (b" a  ", b" ", " a"),
    (b"a\0  ", b" ", "a\0"),
    (b"a\0\0\0", None, "a\0\0\0"),
    (b"\0\0\0\0", b"\0", ""),
])
def test_text_padding_preserves_other_bytes(tmp_path, raw, padding, expected):
    path = tmp_path / "padding.bin"
    path.write_bytes(raw)
    data = read_binary(path, value_types={"label": fixed_text(4, encoding="US-ASCII", padding=padding)}).collect()
    assert data.label.item() == expected


def test_text_uses_explicit_record_layout(tmp_path):
    path = tmp_path / "text-layout.bin"
    path.write_bytes(b"??ab\0\0XX\0\x07!!")
    data = read_binary(path, value_types={"ring": "uint16", "label": fixed_text(4, encoding="ascii")},
                       byte_order="big", field_offsets={"label": 2, "ring": 8}, record_size=12).collect()
    assert data.ring.item() == 7 and data.label.item() == "ab"


@pytest.mark.parametrize("encoding, invalid", [
    ("utf-8", b"\xc3\0"),
    ("utf-8", b"\xff\0"),
    ("ascii", b"\xc3\xa9"),
])
@pytest.mark.parametrize("lazy", [False, True])
def test_invalid_text_reports_field_and_record(tmp_path, encoding, invalid, lazy):
    path = tmp_path / "invalid-text.bin"
    path.write_bytes(b"OK" + invalid)
    with pytest.raises(ReadError, match="field 'label', record 2: invalid"):
        read_binary(path, value_types={"label": fixed_text(2, encoding=encoding)}, lazy=lazy)


def test_text_records_work_with_filters(tmp_path):
    path = tmp_path / "categories.bin"
    path.write_bytes(b"red\0blue")
    frame = read_binary(path, value_types={"category": fixed_text(4, encoding="ascii")})
    result = (frame >> filter(col("category") == "blue") >> summarise(count=n())).collect()
    assert result["count"].item() == 1


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_text_records_keep_string_values(tmp_path, lazy):
    path = tmp_path / "empty-text.bin"
    path.write_bytes(b"")
    data = read_binary(path, value_types={"label": fixed_text(4, encoding="utf-8")}, lazy=lazy).collect()
    assert data.sizes == {"record": 0} and data.label.dtype.kind in ("O", "U", "T")


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_fixed_text_rejects_invalid_widths(width):
    with pytest.raises(ReadError, match="positive integer number of bytes"):
        fixed_text(width, encoding="utf-8")


@pytest.mark.parametrize("encoding", ["utf-16-le", "latin-1", "not-an-encoding"])
def test_fixed_text_rejects_unsupported_encodings(encoding):
    with pytest.raises(ReadError, match="encoding"):
        fixed_text(4, encoding=encoding)


@pytest.mark.parametrize("padding", [b"", b"XX", b"\t", " ", bytearray(b" ")])
def test_fixed_text_rejects_invalid_padding(padding):
    with pytest.raises(ReadError, match="padding"):
        fixed_text(4, encoding="utf-8", padding=padding)


def assert_readonly_map(values):
    backing = values
    while isinstance(backing, np.ndarray) and not isinstance(backing, np.memmap):
        backing = backing.base
    assert isinstance(backing, np.memmap) and backing.mode == "r"
    assert np.shares_memory(values, backing) and not values.flags.writeable
    return backing


@pytest.mark.parametrize("byte_order, raw, offsets, size", [
    ("little", "0000c03f0700000020c00c00", None, None),
    ("big", "3fc000000007c0200000000c", None, None),
    ("little", "aabb07000000c03fccddeeff0c00000020c01122", {"x": 4, "ring": 2}, 10),
    ("big", "aabb00073fc00000ccddeeff000cc02000001122", {"x": 4, "ring": 2}, 10),
])
def test_numeric_records_remain_mapped_in_stored_byte_order(tmp_path, byte_order, raw, offsets, size):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes.fromhex(raw))
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                        byte_order=byte_order, field_offsets=offsets, record_size=size, lazy=True)
    data = frame.to_xarray()
    endian = "<" if byte_order == "little" else ">"
    for name, kind in [("x", "f4"), ("ring", "u2")]:
        backing = assert_readonly_map(data[name].data)
        assert str(backing.filename) == str(path)
        assert data[name].dtype == np.dtype(endian + kind)
        assert data[name].data.strides == (size or 6,)
        assert np.shares_memory(frame.collect()[name].data, data[name].data)
    assert data.x.values.tolist() == [1.5, -2.5] and data.ring.values.tolist() == [7, 12]
    with pytest.raises(ValueError, match="read-only"):
        data.x.data[0] = 99
    assert path.read_bytes() == bytes.fromhex(raw)
    assert "memory-mapped" in repr(frame)


@pytest.mark.parametrize("byte_order, first, second", [
    ("little", "0000c03f", "000020c0"),
    ("big", "3fc00000", "c0200000"),
])
def test_mapping_numeric_fields_still_decodes_text_eagerly(tmp_path, byte_order, first, second):
    path = tmp_path / "mapped-text.bin"
    path.write_bytes(bytes.fromhex(first) + b"caf\xc3\xa9"
                     + bytes.fromhex(second) + b"\xe7\x8c\xab\0\0")
    frame = read_binary(path, value_types={"x": "float32", "label": fixed_text(5, encoding="utf-8")},
                        byte_order=byte_order, lazy=True)
    data = frame.to_xarray()
    assert_readonly_map(data.x.data)
    assert data.x.values.tolist() == [1.5, -2.5] and data.label.values.tolist() == ["café", "猫"]
    assert "memory-mapped" in repr(frame)
    assert "materialized: label decoded as utf-8 text" in repr(frame)


def test_mutating_mapped_records_preserves_the_source_file(tmp_path):
    path = tmp_path / "mapped-mutate.bin"
    raw = bytes.fromhex("0000c03f0700000020c00c00")
    path.write_bytes(raw)
    frame = read_binary(path, value_types={"x": "float32", "ring": "uint16"}, lazy=True)
    changed = frame >> mutate(x=col("x") + 1)
    assert changed.collect().x.values.tolist() == [2.5, -1.5]
    assert frame.collect().x.values.tolist() == [1.5, -2.5]
    assert_readonly_map(changed.to_xarray().ring.data)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("lazy", [False, True])
def test_header_offset_and_labels_with_a_padded_mixed_layout(tmp_path, lazy):
    path = tmp_path / "header.bin"
    raw = (b"HDR" + bytes.fromhex("07000000c03f") + b"red\0!!"
           + bytes.fromhex("0c00000020c0") + b"blue??")
    path.write_bytes(raw)
    frame = read_binary(
        path, value_types={"x": "float32", "ring": "uint16", "category": fixed_text(4, encoding="ascii")},
        dim="point", labels={"point": ["front", "rear"]}, offset=np.int64(3),
        field_offsets={"x": 2, "ring": 0, "category": 6}, record_size=12, lazy=lazy,
    )
    data = frame.to_xarray()
    assert data.point.values.tolist() == ["front", "rear"]
    assert data.x.values.tolist() == [1.5, -2.5] and data.ring.values.tolist() == [7, 12]
    assert data.category.values.tolist() == ["red", "blue"]
    if lazy:
        assert_readonly_map(data.x.data)
        assert_readonly_map(data.ring.data)
    kept = (frame >> filter(col("point") == "rear")).collect()
    assert kept.point.values.tolist() == ["rear"] and kept.ring.item() == 12
    assert "3-byte header skipped" in repr(frame)
    assert "inferred: point = 24 bytes / 12 bytes = 2" in repr(frame)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("lazy", [False, True])
def test_header_only_file_returns_empty_labeled_variables(tmp_path, lazy):
    path = tmp_path / "header-only.bin"
    path.write_bytes(b"HDR")
    frame = read_binary(path, value_types={"x": "float32", "label": fixed_text(4, encoding="utf-8")},
                        dim="point", labels={"point": []}, offset=3, lazy=lazy)
    data = frame.collect()
    assert data.sizes == {"point": 0} and data.point.values.tolist() == []
    assert data.x.dtype == np.dtype("float32") and data.label.dtype.kind in ("O", "U", "T")
    if lazy:
        assert "empty (no mapping)" in repr(frame)


@pytest.mark.parametrize("offset", [-1, True, 1.5])
def test_invalid_header_offset_is_rejected(tmp_path, offset):
    with pytest.raises(ReadError, match="offset must be a non-negative number of bytes"):
        read_binary(tmp_path / "unused.bin", value_types={"x": "float32"}, offset=offset)


def test_header_offset_past_end_is_rejected(tmp_path):
    path = tmp_path / "short-header.bin"
    path.write_bytes(b"HDR")
    with pytest.raises(ReadError, match="offset=4 is past the end of the file"):
        read_binary(path, value_types={"x": "float32"}, offset=4)


@pytest.mark.parametrize("lazy", [False, True])
def test_record_validation_uses_only_bytes_after_the_header(tmp_path, lazy):
    path = tmp_path / "partial-payload.bin"
    path.write_bytes(b"HDR" + bytes(5))
    with pytest.raises(ReadError, match="5 bytes is not a whole number of records of 6 bytes"):
        read_binary(path, value_types={"x": "float32", "ring": "uint16"}, offset=3, lazy=lazy)


@pytest.mark.parametrize("labels, message", [
    ({"point": ["front"]}, 'label count for "point" is 1 but its size is 2'),
    ({"point": ["front", "front"]}, "labels must be unique"),
    ({"other": ["front", "rear"]}, 'labels for "other", which is not in dims'),
    ({"point": [1, "rear"]}, r'labels for "point" mix numbers and text \(int, str\)'),
])
def test_record_axis_labels_are_validated(tmp_path, labels, message):
    path = tmp_path / "labels.bin"
    path.write_bytes(bytes.fromhex("0000c03f000020c0"))
    with pytest.raises(ReadError, match=message):
        read_binary(path, value_types={"x": "float32"}, dim="point", labels=labels)
