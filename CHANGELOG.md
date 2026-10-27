# Changelog

## Unreleased

- `write(path)` writes a frame's data to a `.npy` file (one array) or a
  Zarr store (a dataset; the `zarr` extra), and gives the file back as a
  frame. A frame read with `blocks=` that keeps rows writes them block by
  block, in order, as the blocks arrive, so a result need not fit in
  memory, and the frame given back reads the file in blocks of the same
  size. Files are written under a temporary name and renamed when every
  block is in; an existing path is refused unless `overwrite=True`.
  `scan(..., writes=[(frame, path)])` writes beside other results in one
  pass. What is written reads back the same bits as `collect()` (plan
  milestone 12, step 2).
- A block reads only the variables used: a frame read with `blocks=`
  records, while each verb is checked on the empty block, which variables
  it reads (by name, field label, coordinate, or formula), and a dataset's
  blocks compute only those, with the variables a frame that keeps rows
  keeps. A selector's predicate of the caller's, `where(...)`, which may
  look at any values, reads everything. `n()` counts the axes every
  variable shares as before, from the whole schema. A file's rows hold
  every field together, so files gain nothing. The backend is chosen once
  for the whole data, so every block is on the same one (plan milestone
  12, step 1).

## 0.6.0 — 2026-10-27

- Nothing is read until a result is asked for: a frame read with `blocks=`
  checks each verb as it is called on a block of no positions with the
  data's schema, instead of reading and running the first block, and
  `repr` and `explain()` show each step's variables, axes, and types, with
  `?` for a size the data decides (plan decision 4). `preview()` runs the
  verbs on a sample of blocks, by default the first, middle, and last
  (`preview(blocks=[2, 5])` names others), and gives that result in
  memory, labelled as a sample. A value-level error (a value outside every
  bin) is now found when the blocks run, and names its block; a
  Dask-backed source has each chunk computed once, the first included.
- `min()` and `max()` of no positions, or of an empty bin, keep the
  values' type where it has a missing value: NaN of float32 stays float32
  and times give NaT (a float NaN among times failed on the NumPy
  backend); integers have none, so they still become float64. Text takes
  `min()` and `max()` on both backends, as objects; `any()` and `all()`
  of text are refused, as in R and dplyr (the backends gave `True` and
  `1.0`), and so are `std()` and `var()` of durations. The mean of times is
  the xarray backend's, and the NumPy backend says so. On blocks, `min()`
  and `max()` of times, durations, and text merge as in memory. Needs
  labelarray 0.1.4 (`labelarray>=0.1.4`).
- Several results from one pass: `scan(*frames)` runs several frames read
  with `blocks=` over their source in one pass and gives their results as
  a tuple (`per_ring, voxels = scan(a, b)`). The frames' row-wise verbs
  form a trie, so each block is loaded once and a shared `mutate` or
  `filter` runs once on it, and each frame takes its summaries or its
  rows from where its own verbs branch off; the float32 sums the error
  bound cannot settle get one shared second pass. Each result is the same
  bits as the frame's own run (plan milestone 11, step 2). Verbs match by
  their structure (`tidy3a.identity`): every option they were given, with
  values typed (`1`, `1.0`, `True`, and `np.float64(1)` are four values)
  and arrays by identity, so pipelines written apart share their common
  steps, and a verb holding a function of the caller's is like no other.
- `any_of()` given a generator selected nothing (the generator was read
  twice); a `select()` with `~selector` showed the exclusion as an object
  address in its text.
- Reductions of text (`sum`, `mean`, `std`, `var`, `median`) and of times
  (`sum`, `std`, `var`, `median`, `any`, `all`) raise a `SummariseError`
  naming the reduction and the variable, instead of NumPy's `TypeError`.
- Blocks over any source: `read_array(array, ..., blocks=rows)` runs an
  in-memory array block by block, so the verbs' intermediates stay
  block-sized, and `tidy(dataset, blocks={"time": 100})` runs an xarray
  dataset, lazy, file-backed, or Dask-backed, a block of positions at a
  time, each computed and wrapped on its own; `blocks="chunks"` takes a
  Dask-backed dataset's own chunks as the blocks, so each is computed
  once. A block holds every other axis whole, so gridded data summarises
  block by block too (plan milestone 11, step 1).
- Files larger than memory: `read_array(path, ..., blocks=rows)` reads a
  file in blocks of that many positions on its first axis. The result holds
  the first block, so each verb runs on it as it is called, with its checks,
  notes, and `explain()`; when a result is asked for (`collect()`, `pull`,
  `to_dict()`), the verbs run over every block, `threads` blocks at a time,
  and the blocks' summaries merge, so memory is the blocks in flight and
  the groups, not the file. A summarised pipeline is `select`, `filter`,
  and `mutate` of row-wise expressions, `group_by` of value keys (fields,
  `cells`, `bins`, `rounded`, several at once, with or without `dropna`),
  then `summarise` of `n`, `sum`, `mean`, `min`, `max`, `std`, `var`,
  `any`, `all`, or expressions of them; each key's groups merge across
  blocks in the order `group_by` gives them. Float32 sums and means are
  the same bits as in memory whatever the block size; float64 and integer
  sums and means, and `std` and `var`, add per block in float64 and are
  close, not identical. A pipeline without `summarise` joins the blocks'
  rows in memory. What a block cannot answer (`median`, `over=`, a
  reduction inside `mutate` or `filter`) is refused with a `BlockError`
  when the verb is called (plan decision 10, stage (b), phases 1 to 3). On nuScenes Mini's 122.6M lidar points, a distance
  filter and 1 m voxels took 8.0 s at 1.65 GB in blocks against 36.5 s at
  7.14 GB in memory; the same data as a 29.4 GB file on a 24 GB machine
  ran at the drive's read speed with a peak of 1.85 GB and the same bits.

## 0.5.1 — 2026-10-26

- Requires labelarray 0.1.3: summaries by several keys pass their groups'
  labels as an array, which labelarray now keeps, so a voxel summary of
  200,000 points takes 20 ms instead of 22; and float32 sums of a few
  values per result (`sum(p**2, over="i")`) take about 10 µs less, with
  the same bits.

## 0.5.0 — 2026-10-26

Float32 sums and means are now correctly rounded on lazy, chunked (Dask)
data too, a removed field axis ends, and summaries by several keys are
faster. Changes that can alter results or code written for 0.4:

- Float32 sums and means on chunked data change in their last bits: they
  now equal the in-memory, correctly rounded results whatever the chunking,
  where they used Dask's own arithmetic before. Requires labelarray 0.1.2.
- After a verb removes the field axis, the result names no field axis;
  `tidy(result, field_axis="feature")` naming the removed axis is an error
  instead of returning the data unchanged.

- Float32 sums and means on lazy, chunked (Dask) data are correctly
  rounded, the same bits as in memory whatever the chunking (decision 10,
  stage (b), the measured gap). Each chunk's estimate is one Dask task
  (`labelarray.sums.chunk_estimate`), the pieces add across chunks in any
  order, and one last task settles the results; the few the error bound
  cannot settle, if any, get a second pass over the chunks for their exact
  sums. The result stays lazy, in one chunk, and each source chunk is read
  once, shared with everything else collected. Needs labelarray 0.1.2.
- A verb that removes the field axis (a summary over every observation
  axis, for instance) ends it: the result names no field axis,
  `explain()` says `field axis: feature → none`, and `tidy(...,
  field_axis=)` can declare a new one. Results are unchanged; expressions
  already ignored a field axis that no longer existed.
- Grouping by several keys builds each group's key values with NumPy
  indexing instead of a Python loop: a voxel count and float32 mean of
  200,000 points in 96,846 voxels takes 22 ms on the NumPy backend instead
  of 33.

## 0.4.2 — 2026-10-26

- The §7 pipeline (rotate, measure, filter) is as fast as in 0.2.0 again,
  and faster on the NumPy backend: 0.377 ms against 0.414 in 0.2.0 and
  0.455 in 0.4.1 on 34,688 points; on the xarray backend 1.212 ms against
  1.195 and 1.317. The NumPy backend now checks whether a name is one of a
  few text labels (field axes) by comparing text directly instead of
  through the label index, 0.35 µs instead of 9, a check that runs several
  times per verb; the xarray backend builds float32 sums' and means'
  results directly instead of reducing a slice to get their axes,
  coordinates, and attributes. Results are unchanged. Grouped summaries
  are 5–8% faster than in 0.4.1 too.

## 0.4.1 — 2026-10-26

- Times with a time zone stored as values, not only as labels, stay on the
  xarray backend. With xarray 2024.6 and pandas 2.2.2, such a variable
  reached the NumPy backend as integer nanoseconds, so its zone was lost
  without a word and `bins()` on it said "a duration needs time values";
  with recent xarray, `backend="numpy"` called it lazy, chunked, or
  memory-mapped. Now the NumPy backend refuses it ("has times with a time
  zone"), so by default it stays on xarray, and `bins()`, `rounded()`, and
  `cells()` give the time-zone message on both backends.

## 0.4.0 — 2026-10-26

Grouping by several keys (voxels), plans with `explain()` and reusable
pipelines, and tables to and from Polars and tidy3; the array layer is now
the package labelarray. Changes that can alter code written for 0.3.0:

- On the NumPy backend, `where(...)` predicates receive a
  `labelarray.Array`, whose type name is `Array`; `NumpyArray` and
  `NumpyData` remain importable from `tidy3a.numpy_backend` as names for
  the same classes, and `tidy3a.labelarray` still imports the library.
- `group_by` accepts several value keys instead of refusing them, and
  `axis` is a reserved keyword there, as `dropna` is.

- Grouping by several value keys: `group_by(cells(x, 1), cells(y, 1),
  cells(z, 1))` makes one group per combination that occurs, sorted by
  the first key and then the next, along one new axis (`group`, or the
  name `axis=` gives). `summarise` gives each key's value per group as a
  variable named after the key; with `dropna`, a position missing any key
  is in no group, with one note. Counting 200,000 points into their
  96,846 occupied voxels takes 5 ms.
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
  0.1.1 or later (0.1.1 makes float32 sums and means over many small
  groups, such as voxels, about 18 times faster, with the same bits), its array layer released as a package of its own, and no
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
