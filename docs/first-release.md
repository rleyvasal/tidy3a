# First-release scope and limitations

Version `0.1.0` is prepared for the first release and has not been published.
The release notes are in [the changelog](../CHANGELOG.md). Progress and verification
are recorded in [milestone 7](plan.md#milestone-7--independent-first-release).

## Supported scope

- Python 3.11 or newer, NumPy 2.0 or newer, pandas 2.2.2 or newer, and
  xarray 2024.6 or newer.
- Homogeneous raw numeric files, `.npy` files, and in-memory NumPy arrays
  through `read_array`; xarray objects through `tidy()`.
- Mixed fixed-width numeric/text records through `read_binary`, with
  explicit padding layouts and strict UTF-8/ASCII `fixed_text` decoding.
  The supported reader options are listed in [the contract](m6-read-binary.md).
- Tidy verbs, exact label alignment, explicit joins, named-axis mathematics,
  complex values, and the implemented plain-text and LaTeX formula dialect.
- Script expressions with `fields()` or `col()`, and optional IPython
  shorthand through `%load_ext tidy3a.jupyter`.

## Memory and execution

NumPy-backed operations run eagerly. `lazy=True` on `read_array` creates a
read-only memory map; it does not defer the whole pipeline. Arithmetic can
allocate arrays, filtering evaluates its predicate mask, and field-label
updates copy the whole containing array, even when changing one field.
`pull()` returns NumPy values and materializes the requested data.

`tidy()` does not itself compute a lazy xarray input. Selection, regular
positional slicing, arithmetic, and reductions on Dask-backed or file-backed
xarray data stay lazy and are checked in the test suite
([metadata and lazy input](m6-metadata-lazy.md)). Grouping by field values
or bins reads the grouping values, filtering reads the mask, and field-label
write-back reads and copies the containing array. `median(..., na_rm=True)`
over every axis of a chunked array is an error that asks you to collect
first. `collect()` computes the resulting Dataset. `to_xarray()` makes a
shallow copy without computing it. Dask-backed grouping and performance
guarantees are not part of this release; memory-mapping alone does not
promise bounded memory use.

## Known limitations

- Grouping supports multiple axis keys but only one value-field or bin key.
  Grouped `slice` and its head/tail helpers require `ungroup()` first.
- Reductions with `over=` cannot be combined with grouping by a value field.
  A summary cannot reference an earlier result in the same `summarise`;
  derive from summary results in a following `mutate`.
- `tolerance=` is implemented only for `left_join`. Other joins use exact
  labels. `case_when` is not implemented; use `if_else` for binary choices.
- xarray on the left of an arithmetic operator does not defer to a tidy3A
  expression. Write `x * R` rather than `R * x`, or use `dot()`.
- Integer arithmetic follows NumPy. For unsigned data, promote before a
  subtraction that may be negative: `nir * 1.0 - red`.
- Formulas support the dialect documented in [plan §6.3](plan.md#63-formula), with
  errors for unsupported syntax. Product reductions, integrals, sums over
  more than one index letter at once, and implicit multiplication in plain
  text are not supported. Write `2*x` rather than `2x` in plain text.
- Notebook pipes spanning lines need parentheses; tidy3-style backtick
  names are not supported. Use `col("field name")` for such names.
- Coexistence with tidy3's notebook extension requires tidy3's ownership
  check (`_owned_by_tidy3`); the prerequisite change is recorded in the
  implementation plan. The coexistence tests skip when it is unavailable.

Mixed-record reading, metadata handling, and the lazy-input checks were
brought forward from milestone 6 before publication. Dedicated NetCDF/Zarr
readers, Dask-backed grouping, text encodings beyond UTF-8/ASCII, and
variable-length records remain planned. Existing
xarray readers can already be used with `tidy()`; their optional dependencies
are installed separately.
