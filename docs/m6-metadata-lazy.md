# Metadata and lazy input

Use `tidy()` with an xarray DataArray or Dataset from an existing reader.
For NetCDF, that means `tidy(xr.open_dataset(...))`; Zarr and other formats
can use their corresponding xarray readers and optional dependencies.
The NetCDF checks currently use xarray's SciPy backend.

## Metadata and casting at import

Wrapping preserves dimension names, coordinates, numeric types, attributes,
and encodings. Dataset attributes stay on the Dataset; DataArray attributes
stay on its variable when it becomes a one-variable Dataset. `to_xarray()`
and `collect()` preserve this metadata as well.

An encoding such as `{"dtype": "int16", "scale_factor": 0.5}` describes
serialization. It does not cast a float32 variable when `tidy()` wraps it.
When xarray opens a format with packed values or encoded dates, its decoder
determines the in-memory values and type. tidy3a preserves that decoded
representation and its encoding metadata.

For `.npy` input, `read_array(..., value_type=...)` checks the stored numeric
kind and width and rejects a conflict. Raw array and record readers use the
declared physical layout because the bytes provide no schema. They do not
guess units or cast fields into a common numeric type.

`to_xarray()` makes a shallow view: the numeric values share storage, while
the returned Dataset and variable metadata dictionaries can have entries
changed independently. Nested mutable attribute values retain xarray's
shallow-copy behavior.

## Metadata during transformations

Untouched variables retain their attributes and serialization encodings.
New variables assigned by `mutate` start without serialization encoding.
A direct variable alias or an explicitly supplied DataArray retains its
attributes; computed expressions clear inherited variable attributes,
independently of xarray's `keep_attrs` setting. Assignment preserves the
metadata of unchanged coordinates.

Updating a field inside a shared array clears that array's attributes and
encoding, since they describe the whole array. Inspection reports this
clearing when metadata was present. This prevents stale integer storage
encoding from rounding newly computed fractional values during export.

`summarise` preserves Dataset attributes and clears Dataset serialization
encoding. Its result variables follow the same alias/computed-value rules.
Joins retain equal or nonconflicting attributes and drop conflicting
Dataset and shared-coordinate attributes, reporting the dropped keys in
inspection notes. Joins clear serialization encoding after alignment.

These rules do not infer, validate, or convert units. Supply appropriate
attributes and output encoding explicitly when exporting derived values.

## Lazy file-backed input

The NetCDF regression tests verify that wrapping, inspection, `select`,
regular positional `slice`, and `to_xarray()` do not request measurement values from
the backend. `collect()` loads the selected variable's values and preserves
its attributes and encoding. The unselected variable is not requested.
Coordinate indexes may already have been loaded by xarray when opening the
source; this check concerns measurement-array materialization.

Regular position ranges use basic slicing, including reversed and strided
ranges. This also avoids an [upstream SciPy/CF indexing issue](https://github.com/pydata/xarray/issues/8909)
reproduced with xarray 2024.6. Irregular gathers on that older backend can
still fail at collection. Load the required variables explicitly before an
irregular selection, or use a newer xarray/backend. The wrapper does not
silently load a whole source to bypass that limitation. Arbitrary position
order and repetitions are preserved for in-memory arrays.

Keep the source open while its values are needed:

```python
with xr.open_dataset(path) as source:
    result = (tidy(source) >> select("temperature")).collect()
```

This check establishes behavior for the tested file backend.

## Dask-backed selection and collection

Dask remains optional. Install `dask[array]` to use an xarray Dataset or
DataArray backed by Dask, then wrap it with `tidy()`. This follows
[xarray's Dask integration](https://docs.xarray.dev/en/stable/user-guide/dask.html).

Wrapping, inspection, variable selection, positional slicing, and
`to_xarray()` do not execute Dask tasks. Wrapping and variable selection
preserve the original array and its chunks. Slicing builds a lazy graph;
the resulting chunk sizes follow xarray/Dask indexing behavior.

`collect()` computes the selected variables into an in-memory Dataset.
The wrapped frame and its source remain lazy. Tests cover contiguous,
reversed, strided, repeated/irregular, and empty position lists, preserving
values, coordinate order, numeric type, and variable attributes.
Unused variables are not read, and nonempty selections read only the
required source chunks in these fixtures. With the tested Dask 2026.8.0,
collecting an empty basic slice still evaluates the selected variable's
first source chunk. An empty result therefore does not guarantee zero I/O.

## Dask-backed arithmetic and reductions

Arithmetic assigned to separate variables by `mutate` stays lazy, including
sequential assignments and broadcasting a reduction back to observations.
Computed results follow the metadata rules above. Selecting two values
from a centered array still needs every source chunk used by its mean;
selection after a reduction cannot remove that dependency.

Numeric summaries (`mean`, `median`, `std`, `var`, `sum`, `min`, and `max`)
build lazy results, subject to the median limitation below. Tests verify
missing-value propagation and `na_rm=True`, including `sum(min_count=...)`
across chunk boundaries. Boolean `any`/`all` summaries stay lazy, and
ungrouped `n()` uses array sizes without reading measurement chunks.

Median reduction can be expensive: Dask gathers the reduced axes into
single chunks, as described in its
[median documentation](https://docs.dask.org/en/stable/generated/dask.array.nanmedian.html).
Reducing only some axes with `na_rm=True` is supported and stays lazy.
For `median(..., na_rm=True)` over every axis of a chunked array, tidy3a
preserves xarray's rejection of the full-array gather and raises a clear
error before any tasks run. Select the required data and call `collect()`
explicitly, then wrap that in-memory result with `tidy()` to calculate
the median. Check that the selected data fits in memory before collection.

Grouping, field write-back, and other operations that need values still
require separate Dask checks. These tests do not establish lazy behavior
for every verb or a general query optimizer.

## Verification

`tests/test_metadata.py` checks Dataset and DataArray metadata, serialization
encodings versus numeric types, shallow views, and a real NetCDF round trip.
It also checks computed values, direct aliases, summary attributes, join
conflicts, and fractional field updates exported to NetCDF.
The backend test records array requests so that an eager read before
`collect()` fails the test. It skips when SciPy is unavailable; the other
checks need only the core dependencies.

All 21 metadata checks pass on the local dependencies and on NumPy
2.0.x/xarray 2024.6.0 with Python 3.11 and SciPy. All 29 Dask checks pass
with Dask 2026.8.0 on local xarray 2026.9.0 and the minimum NumPy/xarray
versions. Dask is included in the development dependencies and both CI
dependency configurations; these tests skip when Dask is absent.
The full local suite reports 411 passed and 2 skipped.
