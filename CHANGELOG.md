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
- Expressions: `col`, `var`, `label`, `env`, `fields`, `[x, y, z]` vectors,
  `if_else`, and reductions `mean`, `sum`, `min`, `max`, `median`, `std`,
  `any`, `all`, `n` with `over=` and `na_rm=`.
- Exact label alignment, with reordering by label and explicit joins.
- Array layer: `dot` with `over=`, elementwise numeric functions, `conj`,
  `real`, `imag`, `angle`, `arctan2`, `rename_axes`, and `pack`.
