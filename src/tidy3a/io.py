"""Read homogeneous arrays and mixed numeric/text records with named axes.

Raw files store only bytes, so the caller describes everything; ``.npy``
files store their type and shape, so the caller only names axes. Nothing
is guessed from the values, and values are never truncated or padded.
"""

from __future__ import annotations

import codecs
import datetime
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tidy3a.backend import reader_backend
from tidy3a.errors import ReadError, translating
from tidy3a.frame import TidyArray, check_field_axis

_NPY_MAGIC = b"\x93NUMPY"
_NUMERIC_KINDS = set("biufc")


class _Infer:
    """The marker for the one axis size ``read_array`` works out."""

    _instance: _Infer | None = None

    def __new__(cls) -> _Infer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "infer"

    def __reduce__(self) -> str:
        return "infer"


infer = _Infer()


@dataclass(frozen=True)
class _FixedText:
    width: int
    encoding: str
    padding: bytes | None

    def __repr__(self) -> str:
        return f"fixed_text({self.width}, encoding={self.encoding!r}, padding={self.padding!r})"


def fixed_text(width: int, *, encoding: str, padding: bytes | None = b"\0") -> _FixedText:
    """Describe a fixed byte width of strict UTF-8/ASCII text for ``read_binary``."""
    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width <= 0:
        raise ReadError("fixed_text width must be a positive integer number of bytes")
    try:
        encoding = codecs.lookup(encoding).name
    except LookupError as exc:
        raise ReadError(f"unknown text encoding {encoding!r}") from exc
    if encoding not in ("utf-8", "ascii"):
        raise ReadError("fixed_text supports UTF-8 and ASCII encodings")
    if padding is not None and (not isinstance(padding, bytes) or padding not in (b"\0", b" ")):
        raise ReadError('fixed_text padding must be b"\\0", b" ", or None')
    return _FixedText(int(width), encoding, padding)


def _decode_fixed_text(values: np.ndarray, spec: _FixedText, name: str) -> np.ndarray:
    decoded = []
    for position, value in enumerate(values, start=1):
        raw = value.tobytes()
        if spec.padding is not None:
            raw = raw.rstrip(spec.padding)
        try:
            decoded.append(raw.decode(spec.encoding))
        except UnicodeDecodeError as exc:
            raise ReadError(
                f"field {name!r}, record {position}: invalid {spec.encoding} text ({exc.reason})"
            ) from exc
    return np.asarray(decoded, dtype=object)


@translating
def read_binary(
    source: str | os.PathLike[str], *, value_types: Mapping[str, Any],
    dim: str = "record", labels: Mapping[str, Sequence[Any]] | None = None,
    byte_order: str = "little", offset: int = 0,
    field_offsets: Mapping[str, int] | None = None, record_size: int | None = None,
    lazy: bool = False, backend: str | None = None,
) -> TidyArray:
    """Read fixed-width numeric and text fields into separate typed variables.

    Fields are packed in mapping order unless ``field_offsets`` specifies
    byte positions. ``offset`` skips a file header; ``labels`` labels the
    observation axis. ``record_size`` can add trailing padding. ``lazy=True``
    memory-maps numeric fields read-only, retaining their stored byte order.
    Text uses ``fixed_text`` and is decoded eagerly even when memory-mapped.
    """
    if not isinstance(dim, str) or not dim:
        raise ReadError("dim must be a non-empty axis name")
    if not isinstance(value_types, Mapping) or not value_types:
        raise ReadError("value_types must be a non-empty mapping of field names to types")
    if byte_order not in ("little", "big"):
        raise ReadError(f'byte_order must be "little" or "big", not {byte_order!r}')
    layout = []
    for name, value_type in value_types.items():
        if not isinstance(name, str) or not name or name == dim:
            raise ReadError(f"field name {name!r} must be non-empty text different from dim={dim!r}")
        kind = (
            np.dtype(f"V{value_type.width}") if isinstance(value_type, _FixedText)
            else _raw_dtype(value_type, byte_order)
        )
        layout.append((name, kind))
    dtype = _record_dtype(layout, field_offsets, record_size)
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ReadError(f"offset must be a non-negative number of bytes, not {offset!r}")
    offset = int(offset)
    path = Path(source)
    file_bytes = path.stat().st_size
    if offset > file_bytes:
        raise ReadError(f"offset={offset:,} is past the end of the file ({file_bytes:,} bytes)")
    nbytes = file_bytes - offset
    if nbytes % dtype.itemsize:
        raise ReadError(
            f"{nbytes:,} bytes is not a whole number of records of {dtype.itemsize:,} bytes; "
            "check value_types, offset, field_offsets, and record_size. Partial records are never truncated."
        )
    count = nbytes // dtype.itemsize
    coords = _parse_labels(labels, [dim], (count,))
    if lazy and nbytes:
        records = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    else:
        records = np.fromfile(path, dtype=dtype, offset=offset, count=count)
    if not lazy and not records.dtype.isnative:
        records = records.astype(records.dtype.newbyteorder("="))
    variables = {}
    text_notes = []
    for name, value_type in value_types.items():
        if isinstance(value_type, _FixedText):
            values = _decode_fixed_text(records[name], value_type, name)
            text_notes.append(
                f"materialized: {name} decoded as {value_type.encoding} text "
                f"({value_type.width} bytes per record)"
            )
        else:
            values = records[name]
        variables[name] = (dim, values)
    holder = reader_backend({name: values for name, (_, values) in variables.items()}, backend, ReadError)
    data = holder.from_records(variables, coords)
    fields = ", ".join(
        f"{name}: {value_types[name] if isinstance(value_types[name], _FixedText) else kind.name} at byte {position}"
        for name, (kind, position) in dtype.fields.items()
    )
    storage = "packed" if field_offsets is None and record_size is None else "explicit layout"
    if lazy:
        storage += ", memory-mapped" if nbytes else ", empty (no mapping)"
    if offset:
        storage += f", {offset:,}-byte header skipped"
    return TidyArray(
        data, backend=holder,
        source=f'raw record file "{path.name}", {byte_order}-endian, {storage}',
        notes=(f"record: {dtype.itemsize} bytes; {fields}",
               f"inferred: {dim} = {nbytes:,} bytes / {dtype.itemsize} bytes = {len(records):,}",
               *text_notes),
    )


def _record_dtype(layout, field_offsets, record_size) -> np.dtype:
    packed = np.dtype(layout, align=False)
    if field_offsets is None:
        field_offsets = {name: spec[1] for name, spec in packed.fields.items()}
    elif not isinstance(field_offsets, Mapping) or set(field_offsets) != set(packed.names):
        raise ReadError("field_offsets must name every field exactly once, without extra fields")
    ranges = []
    for name, kind in layout:
        start = field_offsets[name]
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0:
            raise ReadError(f"field_offsets[{name!r}] must be a non-negative integer")
        ranges.append((int(start), int(start) + kind.itemsize, name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ReadError(f"fields {previous[2]!r} and {current[2]!r} overlap")
    minimum = max(end for _, end, _ in ranges)
    size = minimum if record_size is None else record_size
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < minimum:
        raise ReadError(f"record_size must be a positive integer covering all fields (at least {minimum} bytes)")
    return np.dtype({
        "names": list(packed.names), "formats": [kind for _, kind in layout],
        "offsets": [int(field_offsets[name]) for name in packed.names], "itemsize": int(size),
    })


@translating
def read_array(
    source: str | os.PathLike[str] | np.ndarray,
    value_type: Any = None,
    dims: Mapping[str, Any] | Sequence[str] | None = None,
    *,
    labels: Mapping[str, Sequence[Any]] | None = None,
    field_axis: str | None = None,
    variables_from: str | None = None,
    name: str | None = None,
    byte_order: str | None = None,
    order: str | None = None,
    offset: int | None = None,
    lazy: bool = False,
    blocks: int | None = None,
    threads: int = 1,
    backend: str | None = None,
) -> TidyArray:
    """Read a homogeneous numeric array and name its axes.

    ``source`` is a raw file, a ``.npy`` file (recognised by its header,
    not its suffix), or an in-memory NumPy array.

    ``value_type`` is the stored type. Raw files need it; for ``.npy`` files
    and arrays it is optional and must match the stored type.

    ``dims`` maps each axis name, in storage order, to its size or ``infer``
    (at most one). ``.npy`` files and arrays may give just the names.

    ``labels`` maps axes to label lists, which must match the axis sizes.

    ``field_axis`` names the axis whose labels act as column names.
    ``variables_from`` instead turns each label on that axis into its own
    variable. ``name`` (default ``"values"``) names the single array.

    Raw files only: ``byte_order`` ("little", the default, or "big"),
    ``order`` ("C", row-major records, the default, or "F"), and ``offset``
    (bytes to skip, default 0). ``lazy=True`` memory-maps the file instead
    of loading it.

    ``blocks=rows`` reads the data in blocks of that many positions on its
    first axis, a file larger than memory or an array whose intermediates
    should stay small: the result holds the first block, so each verb runs
    on it as it is called, and the verbs run over every block, ``threads``
    blocks at a time, when a result is asked for (``docs/plan.md``,
    decision 10, stage (b), and milestone 11). A pipeline ending in
    ``summarise()`` merges the blocks' summaries; any other joins the
    blocks' rows in memory. What a block cannot answer is refused when the
    verb is called.

    ``backend=`` is ``"xarray"`` or ``"numpy"``; without it, the default
    from ``set_backend()`` is used, or with none chosen, NumPy.
    Memory-mapped data stays on xarray.
    """
    kind, path = _classify(source)
    description = _describe_source(kind, path)
    _check_raw_options(kind, description, byte_order, order, offset, lazy)

    names, sizes = _parse_dims(dims, kind)
    notes: list[str] = []
    if blocks is not None:
        _check_blocks(kind, description, blocks, lazy, order, names, field_axis, variables_from)
        lazy = kind != "array"  # a file is memory-mapped, and each block copied out of the map

    if kind == "raw":
        values, inferred = _read_raw(
            path, value_type, names, sizes, byte_order or "little", order or "C",
            offset or 0, lazy,
        )
        stored = values.dtype
        endian = "" if stored.itemsize == 1 else (
            " little-endian" if (byte_order or "little") == "little" else " big-endian"
        )
        layout = "row-major" if (order or "C") == "C" else "column-major"
        source_text = f"{description}, {stored.name}{endian}, {layout}"
        if offset:
            source_text += f", {offset:,}-byte offset"
    else:
        values = _load_stored(source, kind, path, lazy)
        _check_value_type(value_type, values.dtype, description)
        inferred = _match_stored_shape(values.shape, names, sizes, description)
        source_text = description
    if lazy and blocks is None:
        source_text += ", memory-mapped"
    if inferred is not None:
        notes.append(inferred)

    shape = values.shape
    coords = _parse_labels(labels, names, shape)

    if name is not None and variables_from is not None:
        raise ReadError(
            "name= names the single array, and variables_from= makes one variable per label; "
            "use one of them"
        )
    if name is not None and (not isinstance(name, str) or not name):
        raise ReadError("name= must be a non-empty string")

    if blocks is not None:
        from tidy3a.chunked import ArraySource, ChunkedFrame, FileSource

        axis = names[0]

        def build(block: np.ndarray, start: int, stop: int) -> TidyArray:
            block_coords = {dim: (held[start:stop] if dim == axis else held) for dim, held in coords.items()}
            return _frame(block, names, block_coords, field_axis, variables_from, name, backend, source_text, notes)

        if kind == "array":
            return ChunkedFrame.open(ArraySource(values, axis, build), blocks, threads)
        if kind == "raw":
            def reopen() -> np.ndarray:
                return _read_raw(path, value_type, names, sizes, byte_order or "little", "C", offset or 0, True)[0]
        else:
            def reopen() -> np.ndarray:
                return _load_stored(source, kind, path, True)

        size = int(shape[0])
        stored = values.dtype if values.dtype.isnative else values.dtype.newbyteorder("=")  # as FileSource.block gives
        del values  # the map closes; each block reopens the file
        return ChunkedFrame.open(FileSource(reopen, size, axis, build, (stored, tuple(shape[1:]))), blocks, threads)
    return _frame(values, names, coords, field_axis, variables_from, name, backend, source_text, notes)


def _frame(values: np.ndarray, names: list[str], coords: Mapping[str, Any], field_axis: str | None,
           variables_from: str | None, name: str | None, backend: str | None, source_text: str,
           notes: Sequence[str]) -> TidyArray:
    """The frame of an array read with its options checked: the single array, or one variable per label."""
    holder = reader_backend({name or "values": values}, backend, ReadError)
    origins: dict[Hashable, tuple[str, tuple[Any, ...]]] = {}
    if variables_from is not None:
        _check_split(names, variables_from, coords)
        dataset = holder.from_split(values, names, coords, variables_from)
        axis_labels = tuple(coords[variables_from])
        origins = {label: (variables_from, axis_labels) for label in axis_labels}
        if field_axis == variables_from:
            raise ReadError(
                f'field_axis="{field_axis}" names an axis that '
                f'variables_from="{variables_from}" turns into variables.\n'
                f'  Use one: variables_from="{variables_from}" already makes '
                f"{_label_list(axis_labels)} selectable as columns."
            )
    else:
        dataset = holder.from_array(values, names, coords, name or "values")

    if field_axis is not None:
        check_field_axis(dataset, field_axis, ReadError, holder)

    return TidyArray(
        dataset, backend=holder, field_axis=field_axis, origins=origins,
        source=source_text, notes=notes,
    )


# ── Source kinds ────────────────────────────────────────────────────────────


def _classify(source: Any) -> tuple[str, Path | None]:
    if isinstance(source, np.ndarray):
        return "array", None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        with path.open("rb") as handle:
            head = handle.read(len(_NPY_MAGIC))
        return ("npy" if head == _NPY_MAGIC else "raw"), path
    raise TypeError(
        f"read_array() reads a file path or a NumPy array, not {type(source).__name__}; "
        "use tidy() for xarray objects"
    )


def _describe_source(kind: str, path: Path | None) -> str:
    if kind == "array":
        return "NumPy array"
    assert path is not None
    return f'{".npy file" if kind == "npy" else "raw file"} "{path.name}"'


def _check_blocks(kind: str, description: str, blocks: Any, lazy: bool, order: Any,
                  names: list[str], field_axis: str | None, variables_from: str | None) -> None:
    if blocks == "chunks":
        raise ReadError('blocks="chunks" takes a Dask-backed dataset\'s chunks; '
                        f"{description} has none, so give a number of positions")
    if lazy:
        raise ReadError("blocks= and lazy=True both keep the file on disk; use one of them")
    if order == "F":
        raise ReadError(f"blocks= reads runs of the first axis, which {description} stores scattered "
                        '(order="F"); store the file row-major to read it in blocks')
    if not names:
        raise ReadError("blocks= needs an axis to read along: name the axes with dims=")
    if names[0] in (field_axis, variables_from):
        role = "field_axis" if names[0] == field_axis else "variables_from"
        raise ReadError(f'blocks= reads runs of the first axis, "{names[0]}", which {role}= turns into columns; '
                        "put the observation axis first")


def _check_raw_options(
    kind: str, description: str, byte_order: Any, order: Any, offset: Any, lazy: bool,
) -> None:
    if kind != "raw":
        for option, value in (("byte_order", byte_order), ("order", order), ("offset", offset)):
            if value is not None:
                raise ReadError(
                    f"{option}= applies only to raw files; {description} stores its own layout"
                )
    if kind == "array" and lazy:
        raise ReadError("lazy=True memory-maps files; an in-memory array is already loaded")


# ── dims ────────────────────────────────────────────────────────────────────


def _parse_dims(dims: Any, kind: str) -> tuple[list[str], list[Any]]:
    if dims is None:
        raise ReadError(
            'dims= is required: name every axis, e.g. dims={"point": infer, "feature": 5}'
        )
    if isinstance(dims, Mapping):
        names = list(dims)
        sizes = list(dims.values())
    elif isinstance(dims, (list, tuple)):
        if kind == "raw":
            raise ReadError(
                "raw files store no shape, so dims= needs sizes: "
                'dims={"point": infer, "feature": 5}'
            )
        names = list(dims)
        sizes = [None] * len(names)
    else:
        raise ReadError("dims= must map axis names to sizes, or list axis names")

    seen: set[str] = set()
    for axis in names:
        if not isinstance(axis, str) or not axis:
            raise ReadError(f"axis names must be non-empty strings, not {axis!r}")
        if axis in seen:
            raise ReadError(f'duplicate axis name "{axis}" — axis names must be unique')
        seen.add(axis)
    for axis, size in zip(names, sizes):
        if size is None or size is infer:
            continue
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
            raise ReadError(f'size of "{axis}" must be a non-negative integer or infer, not {size!r}')

    marked = [axis for axis, size in zip(names, sizes) if size is infer]
    if len(marked) > 1:
        listed = ", ".join(f'"{axis}"' for axis in marked)
        raise ReadError(
            f"{_count_word(len(marked))} axes marked infer ({listed}); "
            "at most one size can be inferred"
        )
    return names, sizes


def _count_word(count: int) -> str:
    return {2: "two", 3: "three", 4: "four"}.get(count, str(count))


# ── Raw files ───────────────────────────────────────────────────────────────


def _raw_dtype(value_type: Any, byte_order: str) -> np.dtype:
    if value_type is None:
        raise ReadError(
            'raw files store no type, so value_type= is required, e.g. value_type="float32"'
        )
    if byte_order not in ("little", "big"):
        raise ReadError(f'byte_order must be "little" or "big", not {byte_order!r}')
    try:
        dtype = np.dtype(value_type)
    except TypeError as exc:
        raise ReadError(f"value_type={value_type!r} is not a NumPy type") from exc
    _check_numeric(dtype, value_type)
    code = "<" if byte_order == "little" else ">"
    marker = value_type[:1] if isinstance(value_type, str) else dtype.byteorder
    if dtype.itemsize > 1 and marker in ("<", ">"):
        if marker != code:
            raise ReadError(
                f'value_type={value_type!r} says {"little" if marker == "<" else "big"}-endian, '
                f'but byte_order="{byte_order}"; state the byte order once'
            )
    return dtype.newbyteorder(code)


def _check_numeric(dtype: np.dtype, value_type: Any) -> None:
    if dtype.kind not in _NUMERIC_KINDS:
        raise ReadError(
            f"{value_type!r} is not numeric. read_array expects homogeneous numeric values; "
            "read_binary accepts scalar numeric fields or fixed_text descriptors."
        )


def _read_raw(
    path: Path | None, value_type: Any, names: list[str], sizes: list[Any],
    byte_order: str, order: str, offset: int, lazy: bool,
) -> tuple[np.ndarray, str | None]:
    assert path is not None
    dtype = _raw_dtype(value_type, byte_order)
    if order not in ("C", "F"):
        raise ReadError(f'order must be "C" (row-major) or "F" (column-major), not {order!r}')
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ReadError(f"offset must be a non-negative number of bytes, not {offset!r}")
    file_bytes = path.stat().st_size
    if offset > file_bytes:
        raise ReadError(f"offset={offset:,} is past the end of the file ({file_bytes:,} bytes)")
    nbytes = file_bytes - offset
    itemsize = dtype.itemsize
    type_name = np.dtype(value_type).name

    known = [int(size) for size in sizes if size is not infer]
    record = math.prod(known)
    inferred_axis = next((axis for axis, size in zip(names, sizes) if size is infer), None)

    if nbytes % itemsize:
        raise ReadError(
            f"{nbytes:,} bytes is not a whole number of {type_name} values ({itemsize} bytes each).\n"
            "  Check value_type or offset."
        )
    if inferred_axis is not None and known and record > 0 and nbytes % (record * itemsize):
        record_axes = [(axis, size) for axis, size in zip(names, sizes) if size is not infer]
        parts = " × ".join(f"{size} {axis}" for axis, size in record_axes)
        record_bytes = record * itemsize
        raise ReadError(
            f"{nbytes:,} bytes is not a whole number of records.\n"
            f"  record = {parts} × {itemsize} bytes ({type_name}) = {record_bytes:,} bytes; "
            f"{nbytes:,} / {record_bytes:,} = {nbytes / record_bytes:,.2f}\n"
            "  Check value_type, the axis sizes, or offset."
        )
    count = nbytes // itemsize

    note = None
    if inferred_axis is not None:
        if record == 0:
            raise ReadError(
                f'cannot infer "{inferred_axis}": the other axes have a zero size, '
                "so any size fits; give it explicitly"
            )
        inferred = count // record
        sizes = [inferred if size is infer else size for size in sizes]
        divisor = " × ".join(str(size) for size in known)
        if len(known) > 1:
            divisor = f"({divisor})"
        steps = f"{nbytes:,} bytes / {itemsize} bytes" + (f" / {divisor}" if known else "")
        note = f"inferred: {inferred_axis} = {steps} = {inferred:,}"
    elif count != record:
        shape = " × ".join(str(size) for size in sizes) or "a single value"
        raise ReadError(
            f"{nbytes:,} bytes hold {count:,} {type_name} values, but dims need "
            f"{shape} = {record:,}.\n"
            "  Values are never truncated or padded; check the sizes, value_type, or offset."
        )

    shape = tuple(int(size) for size in sizes)
    if count == 0:
        values = np.empty(shape, dtype=dtype.newbyteorder("="))
    elif lazy:
        flat = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
        values = flat.reshape(shape, order=order)
    else:
        flat = np.fromfile(path, dtype=dtype, count=count, offset=offset)
        if not flat.dtype.isnative:
            flat = flat.astype(flat.dtype.newbyteorder("="))
        values = flat.reshape(shape, order=order)
    return values, note


# ── .npy files and arrays ───────────────────────────────────────────────────


def _load_stored(source: Any, kind: str, path: Path | None, lazy: bool) -> np.ndarray:
    if kind == "array":
        values = source
    else:
        try:
            values = np.load(path, allow_pickle=False, mmap_mode="r" if lazy else None)
        except ValueError as exc:
            raise ReadError(
                f'.npy file "{path.name if path else source}" holds Python objects, which '
                "would need pickle to load; read_array does not unpickle files"
            ) from exc
    _check_numeric(values.dtype, str(values.dtype))
    return values


def _check_value_type(value_type: Any, stored: np.dtype, description: str) -> None:
    if value_type is None:
        return
    try:
        wanted = np.dtype(value_type)
    except TypeError as exc:
        raise ReadError(f"value_type={value_type!r} is not a NumPy type") from exc
    if wanted.newbyteorder("=") != stored.newbyteorder("="):
        raise ReadError(
            f'value_type="{wanted.name}" conflicts with the stored type {stored.name} '
            f"in {description}.\n"
            "  Omit value_type to use the stored type, or convert after import with mutate()."
        )


def _match_stored_shape(
    shape: tuple[int, ...], names: list[str], sizes: list[Any], description: str,
) -> str | None:
    if len(names) != len(shape):
        raise ReadError(
            f"dims has {len(names)} axes but {description} has {len(shape)}: shape {shape}"
        )
    note = None
    for axis, size, actual in zip(names, sizes, shape):
        if size is infer:
            note = f"inferred: {axis} = {actual:,} from the stored shape"
        elif size is not None and int(size) != actual:
            raise ReadError(f'dims says "{axis}" has size {size} but {description} has {actual}')
    return note


# ── labels and variables ────────────────────────────────────────────────────


def _parse_labels(
    labels: Any, names: list[str], shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if labels is None:
        return {}
    if not isinstance(labels, Mapping):
        raise ReadError('labels= must map axis names to label lists, e.g. {"feature": ["x", "y"]}')
    coords: dict[str, np.ndarray] = {}
    for axis, values in labels.items():
        if axis not in names:
            raise ReadError(f'labels for "{axis}", which is not in dims')
        if isinstance(values, (str, bytes)):
            raise ReadError(
                f'labels for "{axis}" must be a list, e.g. ["x", "y", "z"], not one string'
            )
        array = _label_array(axis, values)
        if array.ndim != 1:
            raise ReadError(f'labels for "{axis}" must be a flat list')
        size = shape[names.index(axis)]
        if len(array) != size:
            raise ReadError(f'label count for "{axis}" is {len(array)} but its size is {size}')
        unique, counts = np.unique(array, return_counts=True)
        if (counts > 1).any():
            repeated = unique[counts > 1][0]
            raise ReadError(
                f'labels for "{axis}" repeat {_plain(repeated)!r}; labels must be unique'
            )
        coords[axis] = array
    return coords


def _label_array(axis: str, values: Any) -> np.ndarray:
    """Labels as one array, refusing a list that mixes kinds instead of letting NumPy convert it.

    NumPy turns ``[1, "a"]`` into the text ``"1"`` and ``"a"``, and ``[True, 2]``
    into the numbers 1 and 2; the exact label rule keeps kinds apart (1 is not
    "1", True is not 1), so a mix is an error naming the types. Integers and
    floats are one kind, joined only when float64 holds every label exactly.
    """
    if isinstance(values, np.ndarray) and values.dtype != object:
        return values
    items = values.tolist() if isinstance(values, np.ndarray) else list(values)
    types = {type(item) for item in items}
    kinds = {_label_kind(kind) for kind in types}
    if len(kinds) > 1:
        names = ", ".join(sorted(kind.__name__ for kind in types))
        raise ReadError(
            f'labels for "{axis}" mix {" and ".join(sorted(kinds))} ({names}); labels on one axis are one '
            'kind, so give them one type'
        )
    array = np.asarray(values)
    if kinds == {"numbers"} and len(types) > 1 and not all(a == b for a, b in zip(items, array.tolist())):
        raise ReadError(
            f'labels for "{axis}" mix integers and floats that float64 cannot hold exactly; '
            "give them one type"
        )
    return array


def _label_kind(kind: type) -> str:
    """The kind a label's type belongs to; labels of different kinds never match."""
    if issubclass(kind, (bool, np.bool_)):
        return "booleans"
    if issubclass(kind, (np.datetime64, datetime.date)):
        return "times"
    if issubclass(kind, (np.timedelta64, datetime.timedelta)):  # before numbers: NumPy counts it as one
        return "durations"
    if issubclass(kind, (int, float, complex, np.number)):
        return "numbers"
    if issubclass(kind, str):
        return "text"
    if issubclass(kind, bytes):
        return "bytes"
    return "None" if kind is type(None) else kind.__name__


def _check_split(names: list[str], axis: Any, coords: Mapping[str, np.ndarray]) -> None:
    if axis not in names:
        raise ReadError(f'variables_from="{axis}" is not in dims')
    if axis not in coords:
        raise ReadError(
            f'variables_from="{axis}" needs labels on "{axis}"; they become the variable names'
        )
    if not all(isinstance(_plain(label), str) for label in coords[axis]):
        raise ReadError(
            f'variables_from="{axis}" needs text labels, since they become variable names'
        )


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _label_list(labels: Sequence[Any]) -> str:
    shown = [str(label) for label in labels]
    if len(shown) > 3:
        shown = [*shown[:3], "…"]
    return ", ".join(shown)
