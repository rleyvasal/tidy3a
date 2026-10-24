# Mixed-record reader contract

This defines the mixed-record work in milestone 6. `read_binary` is now
implemented and exported for numeric and fixed-width text records, with
`value_types`, `dim`, `labels`, `byte_order`, `offset`, `field_offsets`,
`record_size`, and `lazy`.
`fixed_text` supports strict UTF-8/ASCII decoding and explicit padding.
The reader interface described below is implemented.

The current numeric interface is:

```python
records = read_binary(path, value_types={"x": "float32", "ring": "uint16"}, dim="point")
```

To skip a three-byte header and label two observations:

```python
records = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                      offset=3, dim="point", labels={"point": ["front", "rear"]}, lazy=True)
```

The header offset is relative to the start of the file. Field offsets
remain relative to each record, and the inferred count uses payload bytes
after the header. Labels identify observations and work with `filter`.

For a four-byte gap between fields and two bytes of trailing padding:

```python
records = read_binary(path, value_types={"x": "float32", "ring": "uint16"},
                      field_offsets={"x": 0, "ring": 8}, record_size=12)
```

Text fields use the same reader:

```python
records = read_binary(
    path,
    value_types={"x": "float32", "ring": "uint16",
                 "category": fixed_text(20, encoding="utf-8")},
    dim="point",
)
```

The result is a `TidyArray` containing separate variables `x(point)`,
`ring(point)`, and `category(point)`. Numeric widths and kinds are preserved:
the ring stays uint16 instead of being promoted to float32. There is no
field axis, and no vector origin axis is invented; use `pack` when needed.

## Reader arguments

| Argument | Meaning and default |
|---|---|
| `source` | Path to a raw fixed-width record file. |
| `value_types` | Required ordered mapping of field names to scalar numeric types or `fixed_text` descriptors. Mapping order is output variable order. |
| `dim` | Name of the single observation axis; defaults to `"record"`. |
| `labels` | Optional axis-label mapping, as in `read_array`; only `dim` can be labeled and the label count must match the record count. |
| `byte_order` | Numeric byte order, `"little"` by default or `"big"`; conflicting explicit endian markers are errors. |
| `offset` | Bytes to skip before the first record; defaults to zero. |
| `field_offsets` | Optional mapping of every field name to its zero-based byte offset within a record. Without it, fields are packed in mapping order. |
| `record_size` | Optional total bytes per record, including trailing padding. Defaults to the end of the last occupied byte. |
| `lazy` | Defaults to `False`; `True` memory-maps the numeric fields. Text decoding still materializes text, as described below. |

`value_types` is nonempty. Field and axis names are nonempty strings, and
a field cannot have the same name as the observation axis. Numeric types
are scalar bool, integer, floating-point, or complex types, matching
`read_array`. Object, nested-record, and subarray types are outside this
initial implementation; text needs an explicit `fixed_text` descriptor.
The reader does not infer layout from file suffixes or byte contents.

## Record layout and padding

The example's packed record occupies 26 bytes: four for `x`, two for
`ring`, and twenty for `category`. Native platform alignment is never
applied automatically.

Explicit `field_offsets` must name every field exactly once. Offsets must
be nonnegative integers and occupied field ranges cannot overlap.
`record_size` must be a positive integer covering every field; it can add
trailing padding. Gaps are ignored without checking their contents.
For example, offsets 0, 4, and 8 with a record size of 32 leave two bytes
between `ring` and `category`, and four bytes after `category`.

`offset` must be a nonnegative integer within the file. After skipping the
header, the remaining byte count must be an exact multiple of the record
size; partial records raise `ReadError` instead of being truncated.
An empty payload is valid and returns zero-length variables with their
declared types. No timestamp, label, or physical unit is invented.

## Fixed-width text

`fixed_text(width, *, encoding, padding=b"\0")` declares a positive byte
width. Width counts stored bytes, not Unicode characters. Encoding is
required; the first implementation supports UTF-8 and ASCII, including
their standard aliases. Other encodings are deferred until an example
establishes their padding requirements.

Padding is one of `b"\0"` (default), `b" "`, or `None`. Remove only trailing
occurrences of the chosen padding byte before decoding. Preserve leading
padding bytes, interior bytes, and other trailing whitespace. With `None`,
decode the full width. This keeps an actual trailing space unless the
caller explicitly declares space padding.

Decode strictly: malformed bytes or an incomplete UTF-8 character are
errors, with the field name and 1-based record position. Do not replace
characters, drop undecodable bytes, or silently truncate fields. Return
decoded text variables rather than numeric byte arrays.

## Memory and inspection

Eager reads load the records into memory. Numeric fields remain separate
typed variables; conversion to native byte order may copy numeric data but
must preserve its width and kind. Text decoding allocates decoded strings.

With `lazy=True`, numeric variables are views of a read-only memory map,
retaining stored byte order. Empty files return zero-length arrays without
creating a mapping; inspection reports this explicitly. Decoding text reads
every text field and allocates its decoded values even in this mode. Inspection reports the
record size, field offsets/types, inferred record count, and any text
materialization. This option does not promise a lazy verb pipeline or
Dask execution.

## Implementation order

Packed and explicitly padded numeric/text records, record-count validation,
and strict UTF-8/ASCII decoding are implemented and tested against
independent byte fixtures. Text tests include multibyte characters, padding
preservation, and field/record context on errors. Memory-mapping is also
implemented: tests verify read-only file-backed numeric views, stored byte
order, sharing through `collect()`, eager text decoding, and source-file
preservation under mutation. Axis labels and header offsets are implemented,
including validation and empty payloads at the end of a header-only file.
Metadata-rich readers and Dask behavior remain separate parts of milestone 6.

Verification: all 88 record-reader tests pass on the local dependencies
and on NumPy 2.0.x/xarray 2024.6.0 with Python 3.11. The full local suite
reports 353 passed and 2 skipped.
