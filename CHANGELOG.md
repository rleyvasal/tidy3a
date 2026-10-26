# Changelog

## Unreleased

- Plans: each verb records what it did as it runs. `explain()` lists the
  source and each step with its effect (axes resized, added, or removed;
  variables added, removed, or retyped; groups; and the step's notes),
  computing nothing. Verbs compose before any data into a `Pipeline`
  (`mutate(...) >> filter(...)`), run with `data >> steps` and previewed
  on the first positions with `steps.explain(data, head=1000)`; a result's
  `plan()` gives its verbs back as a pipeline to run on other data.
  Recording reads names, sizes, and types only, a few microseconds per
  verb; on 34,688-point pipelines `main` is within 3% of 0.3.0. The
  inspection output is unchanged.
- Tables: `as_table()` gives data with one observation axis as a Polars
  DataFrame, one row per position (`long=True` for one row per cell of a
  grid), and `tidy(table, axis=...)` reads a Polars DataFrame or LazyFrame,
  or a tidy3 frame, back, with `fields=` packing columns along a field
  axis. `tidy3.tidy()` accepts tidy3a data through the `__tidy__` hook.
  Polars is optional (`tidy3a[polars]`), and neither it nor tidy3 is
  imported until a table is made or read.
- tidy3a now depends on [labelarray](https://pypi.org/project/labelarray/)
  0.1.0 or later, its array layer released as a package of its own, and no
  longer carries a copy: `pip install tidy3a` installs NumPy and
  labelarray. The `fast` extra installs `labelarray[fast]`. Results are
  unchanged. `tidy3a.labelarray` remains as another name for the package
  and its modules (`tidy3a.labelarray.sums` is `labelarray.sums`), so code
  written against 0.3.0 keeps working.
- The NumPy backend is a thin adapter over `labelarray.Array` and
  `labelarray.Collection`. `where(...)` predicates on the NumPy backend now
  receive a `labelarray.Array`; `tidy3a.numpy_backend.NumpyArray` and
  `NumpyData` remain as names for the two classes, so code that imports or
  checks them keeps working, but `type(array).__name__` is now `"Array"`.
- The library's own tests, README, and changelog are in its repository,
  [rleyvasal/labelarray](https://github.com/rleyvasal/labelarray); `tests/`
  remains tidy3a's suite, run on both backends.

## 0.3.0 — 2026-10-25

The NumPy backend becomes the default, float32 sums and means become
correctly rounded, and grouping gains missing keys, rounding, and grid
cells. Changes that can alter results or code written for 0.2.0:

- Float32 sums and means change in their last bits (they add in float64
  and round once), so a filter on a value close to a threshold can keep or
  drop a point differently. `dot` can differ in the last bits for general
  matrices. Float64 data is unchanged.
- In-memory data uses the NumPy backend unless a backend is chosen; results
  are the same, but `where(...)` predicates receive a `NumpyArray` instead
  of an xarray `DataArray`. `set_backend("xarray")` keeps 0.2.0's behaviour.
- Missing group keys are dropped by default (`dropna=True`); NaT keys formed
  a group in 0.2.0.
- `NumpyArray` is no longer hashable, since `==` compares values.

### Numbers

- Every grouped float32 `sum` and `mean`, and every ungrouped one adding 8 or
  more values per result, is correctly rounded: the float32 nearest the exact
  result, the same on both backends, for any memory layout, with or without
  compiled kernels. Before, NumPy added in float32 and could be off by tens
  of float32 steps on data that cancels, or give `inf` for a mean whose sum
  overflows float32. Ungrouped reductions of fewer than 8 values per result,
  such as `sum(p**2, over="i")` over x, y, z, add in float64 in a fixed order.
- `dot` multiplies with NumPy's `matmul`, arranging the operands the same way
  for every layout: a 3 x 3 rotation of 10M points takes 6.5 ms instead of
  13.3 (68.7 on row-major input). An overflow in `dot` shows as infinity
  without a warning, since BLAS libraries do not report it reliably.

### Backends

- The NumPy backend is the default where it holds the data without loss.
  Data it cannot hold or would lose something of stays on xarray: lazy,
  chunked, and memory-mapped values, other coordinates, multi-level labels,
  time zones, and data with xarray storage encodings, so a netCDF file read
  and written back keeps them.
- `NumpyArray` supports NumPy's ufuncs, `np.where`, and Python's operators,
  broadcasting by axis name under the exact alignment rule.

### Grouping

- `group_by` on a value with missing values no longer raises: every NaN is
  one group, and every NaT. `dropna=True`, the default as in pandas, leaves
  those positions out of every group, with a note saying how many;
  `dropna=False` keeps them as a group labelled NaN (NaT for times), last.
  `bins()` follows the same rule.
- `rounded(field, step, origin=0)` and `cells(field, size, origin=0)` group
  by a value rounded to a step or by the grid cell it falls in, as
  `group_by(rounded(distance, 0.5))` or `group_by(cells(x, 0.2))`. Edges are
  where the printed settings put them: with size 0.1, 0.3 is in
  [0.3, 0.4). `bins()` takes `origin=` for durations; its results are
  unchanged.

### Speed

- Grouped float32 sums and means take one pass instead of one step per
  group: about 1.1 times faster on 10M points in 32 groups with NumPy alone.
- The `fast` extra, `pip install 'tidy3a[fast]'`, adds Numba: grouped float32
  sums and means, and sums over a few values such as x, y, z, in one
  compiled, parallel pass, with the same results. Numba is imported on the
  first such reduction, not with tidy3a. Measured against 0.2.0 on 10M
  points: grouped means about 1.8 times faster, the rotate, distance, and
  filter pipeline about 1.2 times on the NumPy backend. Compiled kernels
  run only from the sizes where they pay (100,000 values for grouped sums,
  300,000 for few-value sums).
- Without `fast`, the xarray backend's 10M-point rotate, distance, and
  filter pipelines are about 15 to 20% slower than in 0.2.0, the cost of
  adding x, y, and z in float64.
- Label lookups on evenly spaced labels (grid latitude, longitude, time)
  are arithmetic, with no sort and no index held.

### Fixes

- `read_array` and `read_binary` converted labels that mix kinds:
  `[1, "a"]` became the text labels "1" and "a", and `[True, 2]` the
  numbers 1 and 2; others raised a raw `TypeError`. Labels mixing booleans,
  numbers, times, durations, text, bytes, or `None` are now a `ReadError`
  naming them, and integers and floats are joined only when float64 holds
  every label exactly.
- On ARM, a float equal to `2**63` was treated as an exact int64 label,
  since the cast saturates.

## 0.2.0 — 2026-10-25

NumPy is now the only requirement, and labels match under one exact rule.
Results are numerically unchanged from 0.1.0; every change below is a
behaviour fix, a new way in or out, or a speed-up.

### Installing

- `pip install tidy3a` installs NumPy and tidy3a. `pip install 'tidy3a[xarray]'`
  adds xarray and pandas, which xarray inputs and outputs and lazy, chunked,
  or memory-mapped data need; existing installs that use xarray should add
  the extra. Without xarray the default backend is NumPy; where xarray is
  needed and missing, the error says so and names the extra.
- `import tidy3a` no longer imports xarray or pandas: 41 ms instead of 218.

### Backends

- Verbs reach stored data through a backend interface; the xarray backend
  is the default when xarray is installed and behaves as before.
- Experimental NumPy backend for eager, in-memory data:
  `set_backend("numpy")`, or `backend="numpy"` on `tidy()`, `read_array()`,
  and `read_binary()`. Lazy, chunked, and memory-mapped data stay on xarray.
  It keeps attributes but not xarray storage encodings; `docs/backends.md`
  lists the deliberate differences. It stays experimental: it meets the
  correctness, dependency, import-time, and size criteria but not yet the
  speed one (plan milestone 9).
- Without xarray, arrays from the calling code can be one-variable
  `read_array` results, and `TidyArray.to_dict()` returns NumPy arrays in
  xarray's `to_dict` layout.

### Labels and alignment

- One exact rule for labels in alignment and joins, on both backends:
  labels match only when they are the same kind and exactly equal. On the
  xarray backend, pandas compared integer and float labels through
  float64, so the label `2**53 + 1` was paired with `2**53`; that is now an
  `AlignError`. A join whose labels cannot all be held exactly in the type
  it keeps them in, or that would match repeated labels that differ
  between the two sides, is an `AlignError` too.
- Repeated labels on an axis that must be matched in another order are an
  `AlignError` naming the axis, instead of a pandas `InvalidIndexError`.
- Durations in `bins()` are read by tidy3a itself, the same whether or not
  pandas is installed; `"1h30min"` and `"2 weeks"` work everywhere, and
  months, years, and clock or ISO forms are an error.

### Fixes

- `dot` always uses NumPy's `einsum`. With `opt_einsum` installed, xarray
  used it instead, which changed results in the last digits and made later
  steps of a pipeline about twice as slow.
- NumPy backend: labels from the calling code are copied once, so changing
  the caller's array later cannot change a frame; label indexes are
  released with their labels; join gaps promote types as xarray does;
  joining with data that has no labels on an axis works; a lazy array from
  the calling code is an error before anything is computed; time labels
  with a time zone stay on xarray; arrays have `.shape`.
- Writing a field of a read-only in-memory array no longer notes that it
  was memory-mapped.
- The package is written `tidy3a` everywhere, in messages and documentation,
  as it is installed and imported.

### Speed

- Grouped reductions over many small groups run in one pass instead of one
  step per group, with bit-identical results, and grouping with many groups
  or many points is faster on both backends: 90,000 groups of 100,000
  points take about 12 ms instead of 4.8 s on the xarray backend.
- `group_by` keeps the groups it found, so the next verb does not find them
  again.
- `filter` and the `slice` verbs on an unlabelled axis build only the kept
  positions, using less than half the memory and time on large axes.
- A vector assignment into fields of one variable copies it once, not once
  per component.
- Printing a frame formats only the labels it shows, so frames with many
  labels print at once.

## 0.1.0 — 2026-10-24

First release.

- `read_array` reads raw numeric files, `.npy` files, and in-memory NumPy
  arrays, with `infer`, labels, `field_axis=`, `variables_from=`, byte
  order, storage order, header offsets, and memory-mapping.
- `read_binary` reads packed or explicitly padded mixed records into
  separate typed variables. `fixed_text` declares strict UTF-8/ASCII text
  decoding and padding; numeric fields retain their widths and kinds.
- `tidy()` wraps xarray objects; `TidyArray` reports its axes, labels, axis
  roles, and anything the reader inferred.
- Tidy-layer verbs: `select` with tidy3's selection helpers, `filter`,
  `slice`/`slice_head`/`slice_tail`, `mutate` with write-back and vector
  assignment, `group_by` by axis, value, or `bins()`, `summarise`,
  `ungroup`, `pull`, `bind_cols`, and `left_join`/`inner_join`/`full_join`.
- Expressions: `col`, `variable`, `label`, `env`, `fields`, `[x, y, z]`
  vectors, `if_else`, and reductions `mean`, `sum`, `min`, `max`, `median`,
  `std`, `var` (variance, as in tidy3 and dplyr), `any`, and `all` with
  `over=` and `na_rm=`, plus `n()` for observation counts.
- Exact label alignment, with reordering by label and explicit joins.
- Metadata: wrapping keeps attributes and encodings; new and computed
  results clear stale serialization encoding and inherited attributes,
  direct aliases keep theirs, and joins report dropped conflicting
  attributes.
- Lazy input: NetCDF and Dask-backed data stay lazy through inspection,
  selection, regular positional slicing, arithmetic, and reductions until
  `collect()`; grouping, filtering, and field write-back read the values
  they need, as documented.
- Array layer: `dot` with `over=`, elementwise numeric functions, `conj`,
  `real`, `imag`, `angle`, `arctan2`, `rename_axes`, and `pack`.
- `formula()`: plain mathematical text and a strict LaTeX dialect with
  plot3's conventions, subscripts as axis indices, labels, and positions,
  `\sum` with optional checked bounds, `bind=` for index letters and
  values, and show-back of how each formula was read.
- Notebook layer (`%load_ext tidy3a.jupyter`): bare names in tidy3a verbs,
  settled when the verb runs, with an error when a name is both data and a
  notebook variable; `[x, y, z] = …` inside `mutate`; only tidy3a's own
  calls are rewritten, so tidy3 and tidy3a share a notebook.
- `n()` counts observations on the axes every non-scalar variable shares.
- Inspection output shows time labels to the second (or as dates) and
  floats to 10 significant digits, with a gap before each axis role.
- Examples on real data: a nuScenes Mini LiDAR sweep, and three public
  sample datasets (air temperature, ERA-Interim wind, a Landsat scene),
  each checked against plain NumPy or xarray.
- Source distributions include documentation, examples, and prototype
  tests. Packaging checks build a wheel from the source archive and
  validate both distributions' metadata. The build requires setuptools
  77.0.3 or newer for the license metadata.
- Requires Python 3.11+, NumPy 2.0+, pandas 2.2.2+ (now declared, as
  tidy3a imports it directly), and xarray 2024.6+.
- Publishing a GitHub Release runs the version check and reusable CI gate,
  then uploads the validated distributions to PyPI through trusted publishing.
