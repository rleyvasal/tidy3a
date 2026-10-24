# NumPy execution prototype

Run the end-to-end example from the repository root, using the project's
installed development environment:

```bash
.venv/bin/python prototypes/demo_numpy_pipeline.py
```

It uses the existing tidy3A expressions, formula parser, and verb constructors
to rotate points, keep those within 50 metres, centre intensity within each
ring, and summarise intensity per ring while reusing the grouping cache.
Expected kept positions are `[1, 2, 4]`, distances are `[49, 50, 3]`, and mean
intensities per ring are `[10, 30]`.

Only the entry point changes: import `read_array` or `tidy` from
`numpy_pipeline` when running a script in this directory. These return an
experimental `NumpyTidyArray`. Existing readers convert inputs once; intermediate
results contain `numpy_core.Dataset` and `Variable` objects. `collect()` and
`to_xarray()` convert the result back to xarray for compatibility.

Inputs are copied once into owned, read-only array buffers. Outside edits to
the original input cannot invalidate cached groups. `pull()` and conversion
results share read-only measurement storage; use `.copy()` to edit an exported
array. The standalone core's `Variable` borrows storage; snapshot ownership is
the experimental frame's responsibility.

## Current scope

- Eager NumPy arrays, dimension coordinates, dataset and variable attributes.
- Explicit-name `select`, variable/field/vector assignments in `mutate`,
  `filter`, `pull`, `slice`, `slice_head`, `slice_tail`, `group_by`, `summarise`,
  and `ungroup`. Slices use the existing 1-based rules, including repetitions,
  exclusions, and `drop=True`; ungroup before slicing grouped data.
- Existing references, arithmetic, field vectors, named `dot`, and the formula
  parser and axis-binding rules, numeric functions, `if_else`, `is_in`,
  `rename_axes`, and `pack`.
- Axis grouping or one value key, with reductions over additional observation
  axes where required. `sum`, `mean`, `min`, `max`, `std`, `var`, `median`,
  `any`, `all`, and `n` follow existing numeric missing-value rules, including
  `na_rm` and sum's `min_count`.
- Grouped mutation and filtering broadcast summaries back to original positions.
  Grouping state persists across unrelated assignments and selections. Changes
  to key dependencies, filtering, or reordering invalidate the cached positions.
  External arrays in grouping expressions are captured at `group_by` time.
- NumPy factorization for this pipeline. The original direct-core benchmark
  still uses optional pandas factorization by default, as before.

This does not yet replace the package backend. The existing tidy3A imports and
reader/output boundaries still require xarray and pandas. No import-time
dependency reduction is claimed for this adapter.

Unsupported operations raise `NotImplementedError`: selectors, joins, bins, and
`bind_cols`. Chunked/lazy input and auxiliary coordinates are rejected.
Serialization encodings and coordinate attributes are not carried by this
prototype. This is an eager execution prototype, not a complete backend replacement.

## Grouping and memory

`GroupIndex` stores unique keys, integer codes, optional ordering, counts, and
block boundaries in NumPy arrays. It does not allocate a Python list per group.
Groups of equal size are reduced in batches with the same value order and
contiguous block layout as independent NumPy reductions. Temporary gathers
target roughly one million elements; a single larger group still requires its
complete block. The group arrays use O(N + G) storage for N positions and G groups.

Warm summaries reuse factorization and ordering. Computing new summaries still
reads measurement values and writes G results. Cache validation compares the
identities of owned variables and coordinates, without rescanning key values.
Changing a field in a shared holder conservatively invalidates grouping on
other fields in that holder too.

Measure input snapshots, preparation, cold/warm statistics, output conversion,
and grouping-array memory separately:

```bash
.venv/bin/python prototypes/bench_numpy_pipeline.py --points 1000000
```

The two workloads have 32 groups and mostly unique nanosecond timestamps.
[Recorded local results](BENCHMARKS.md) include the environment and measurement scope.

## Label lookups

`LabelIndex` finds labels for `sel` and for alignment. It is built once per
label array and cached; labels are frozen when stored (copied only when the
caller could still change them), so a cached index cannot go stale. Strictly
increasing labels are searched directly with `np.searchsorted`; others go
through one sort made when the index is built. Booleans, numbers, times,
durations, text, and bytes never match across kinds; numbers and times match
only when a lookup converts to the label's type exactly, across time units
too; one NaN or NaT label can be looked up; mixed-type labels raise
`CoreError` naming the operation. `test_numpy_labels.py` compares lookups with
pandas' `get_indexer`; the two deliberate differences are tidy3A's error for
repeated labels and exact comparison of integers beyond float64 precision.

## Verification

```bash
.venv/bin/python -m pytest -q prototypes/test_numpy_core.py prototypes/test_numpy_pipeline.py prototypes/test_numpy_groups.py prototypes/test_numpy_labels.py
```

The integration tests run the original raw-file workflow with only its reader
replaced, compare numerical results against NumPy and current tidy3A, and check
dimensions, labels, and dtypes. Cases include reordered labels, empty input,
all points removed, missing measurements, sequential assignments, and grouped
mutation. Tests check cache reuse and invalidation, input ownership, and
high-cardinality grouping. Adversarial float32 and complex reductions are
compared exactly with per-group results across different axis layouts. A guard
test prevents xarray object construction and pandas factorization inside the
pipeline, between its conversion boundaries.
