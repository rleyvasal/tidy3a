# Changelog

## Unreleased

- Verbs reach stored data through a backend interface; the xarray backend
  is the default and behaves as before.
- Experimental NumPy backend for eager, in-memory data:
  `set_backend("numpy")`, or `backend="numpy"` on `tidy()`, `read_array()`,
  and `read_binary()`. Lazy, chunked, and memory-mapped data stay on xarray.
  The NumPy backend keeps attributes but not xarray storage encodings;
  `docs/backends.md` lists the deliberate differences. It stays experimental:
  it meets the correctness, import-time, and size criteria but not yet the
  speed and dependency ones (plan milestone 9).
- `dot` always uses NumPy's `einsum`. With `opt_einsum` installed, xarray
  used it instead, which changed results in the last digits and made later
  steps of a pipeline about twice as slow.
- Repeated labels on an axis that must be matched in another order are an
  `AlignError` naming the axis, instead of a pandas `InvalidIndexError`.
- One exact rule for labels in alignment and joins, on both backends: labels match only when they are the same kind and exactly
  equal. On the xarray backend, pandas compared integer and float labels
  through float64, so the label `2**53 + 1` was paired with `2**53`; that is
  now an `AlignError`. A join whose labels cannot all be held exactly in the
  type it keeps them in, or that would match repeated labels that differ
  between the two sides, is an `AlignError` too.
- NumPy backend fixes: labels from the calling code are copied once, so
  changing the caller's array later cannot change a frame; label indexes
  are released with their labels; join gaps promote types as xarray does;
  joining with data that has no labels on an axis works; a lazy array from
  the calling code is an error before anything is computed; time labels
  with a time zone stay on xarray; arrays have `.shape`.
- Writing a field of a read-only in-memory array no longer notes that it
  was memory-mapped.
- Printing a frame formats only the labels it shows, so frames with many
  labels print at once.
- A vector assignment into fields of one variable copies it once, not once
  per component.
- `filter` and the `slice` verbs on an unlabelled axis build only the kept
  positions, using less than half the memory and time on large axes.
- Grouped reductions over many small groups run in one pass instead of one
  step per group, with bit-identical results: 90,000 groups of 100,000
  points take about 30 ms instead of 4.8 s on the xarray backend.
- `group_by` keeps the groups it found, so the next verb does not find them
  again.
- NumPy is the only requirement: `pip install 'tidy3a[xarray]'` adds xarray
  and pandas. Without them, the default backend is NumPy, arrays from the
  calling code can be one-variable `read_array` results, and
  `TidyArray.to_dict()` returns NumPy arrays in xarray's `to_dict` layout.
- Grouping with many groups or many points is faster on both backends:
  group labels are converted in one call and group sizes counted once, and
  the NumPy backend puts group results back on each point by indexing. 90,000
  groups take about 12 ms instead of 31 ms; results are unchanged.
- Durations in `bins()` are read by tidy3A itself, the same whether or not
  pandas is installed; `"1h30min"` and `"2 weeks"` work everywhere, and
  months, years, and clock or ISO forms are an error.

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
- Notebook layer (`%load_ext tidy3a.jupyter`): bare names in tidy3A verbs,
  settled when the verb runs, with an error when a name is both data and a
  notebook variable; `[x, y, z] = …` inside `mutate`; only tidy3A's own
  calls are rewritten, so tidy3 and tidy3A share a notebook.
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
  tidy3A imports it directly), and xarray 2024.6+.
- Publishing a GitHub Release runs the version check and reusable CI gate,
  then uploads the validated distributions to PyPI through trusted publishing.
