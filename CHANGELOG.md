# Changelog

## Unreleased

- `read_array` reads raw numeric files, `.npy` files, and in-memory NumPy
  arrays, with `infer`, labels, `field_axis=`, `variables_from=`, byte
  order, storage order, header offsets, and memory-mapping.
- `tidy()` wraps xarray objects; `TidyArray` reports its axes, labels, axis
  roles, and anything the reader inferred.
- Tidy-layer verbs: `select` with tidy3's selection helpers, `filter`,
  `slice`/`slice_head`/`slice_tail`, `mutate` with write-back and vector
  assignment, `group_by` by axis, value, or `bins()`, `summarise`,
  `ungroup`, `pull`, `bind_cols`, and `left_join`/`inner_join`/`full_join`.
- Expressions: `col`, `variable`, `label`, `env`, `fields`, `[x, y, z]`
  vectors, `if_else`, and reductions `mean`, `sum`, `min`, `max`, `median`,
  `std`, `var` (variance, as in tidy3 and dplyr), `any`, `all`, `n` with
  `over=` and `na_rm=`.
- Exact label alignment, with reordering by label and explicit joins.
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
- `n()` counts observations on the axes every variable shares.
