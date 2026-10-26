# Backends

tidy3a's verbs, rules, and messages are the same on every backend; a backend
only stores the data and computes with it (plan milestones 9 and 10).

Without a choice, new data uses the NumPy backend wherever it holds the data
without loss, and the xarray backend for the rest: lazy, chunked, and
memory-mapped values, coordinates other than an axis's own labels,
multi-level labels, times with a time zone, and data with xarray
storage encodings (as `xr.open_dataset` gives them), so a file read and
written back keeps its encodings. Choosing a backend decides:

```python
import tidy3a
tidy3a.set_backend("xarray")           # the xarray backend for all new data, as before 0.3.0
tidy3a.set_backend("numpy")            # NumPy wherever it can hold the data, encodings dropped
frame = tidy3a.tidy(dataset, backend="numpy")
cloud = tidy3a.read_array(..., backend="xarray")
```

The whole test suite runs on both backends, and property tests compare them
on random data and pipelines: printouts, values, types, labels, attributes,
error messages, and warnings must agree. The differences below are
deliberate; each is tested in `tests/test_backend_contract.py`.

| Difference | xarray backend | NumPy backend |
|---|---|---|
| Storage encodings (`dtype`, `scale_factor`, `_FillValue`, `unlimited_dims`, …) | Kept from the source as the milestone 6 rules describe | Not kept: they describe how xarray writes files, and the NumPy backend does not write files. Attributes are kept by the same rules on both |
| Lazy, chunked, and memory-mapped values | Held lazily | Not held. When NumPy is only the default, such data stays on xarray; asking for `backend="numpy"` is an error saying why |
| Lazy, chunked, or memory-mapped arrays from the calling code in an expression (a Dask `DataArray`) | Combined lazily | An error naming the array, raised before anything is computed |
| Coordinates other than an axis's own labels (a scalar `station`, an auxiliary `lat` along `point`), multi-level labels, times with a time zone (as labels or values) | Held | Not held: the data stays on xarray, or `backend="numpy"` is an error |
| Arrays passed to a `where(predicate)` selection helper | xarray `DataArray` | A `labelarray.Array` (also importable as `tidy3a.numpy_backend.NumpyArray`), with `dims`, `sizes`, `shape`, `ndim`, `dtype`, and `values` |

Behaviour that is the same on both, once easy to get wrong:

- Labels match under one exact rule (`tidy3a.labels`) in alignment and
  joins: the same kind (booleans, numbers, times, durations,
  text, bytes) and exactly equal. The integer label `2**53 + 1` never matches
  the float `2**53`, though pandas, and so xarray, would pair them; equal
  values of different number types (`1` and `1.0`) still match.
- A join whose labels cannot all be held exactly in the type it keeps them
  in (`int64` labels joined with `float64` ones beyond `2**53`) is an
  `AlignError`; so is a join on repeated labels, unless both sides have the
  very same labels. Gaps promote types as xarray does: integers of up to 16
  bits to `float32`, wider integers and booleans to `float64`.
- Results are bit-identical on both backends. Float32 sums
  and means go through one shared kernel (`labelarray.sums`): every
  grouped one, and every ungrouped one adding 8 or more values per result, is
  correctly rounded (the float32 nearest the exact result), so it is the same
  for any memory layout, with or without the compiled kernels, on any number
  of threads; ungrouped reductions of fewer than 8 values per result add in
  float64 in a fixed order and round once. `dot` goes through one shared
  contraction (`labelarray.contract`, NumPy's `matmul`) whose result
  does not depend on layout. On lazy, chunked (Dask) float32 values the
  sums and means get the kernel's bits for any chunking: one task estimates
  each chunk, the pieces combine in any order, one last task settles the
  results, and the few the error bound cannot settle get a second pass for
  their exact sums; the result stays lazy, in one chunk. Other reductions
  keep NumPy's per-group results, indexed as a mask would index each group,
  and on lazy values xarray's and Dask's own arithmetic, which can round
  differently.
- Python's operators run without NumPy's floating-point warnings (division by
  zero); NumPy functions such as `sqrt` and `log` warn, as they do on xarray
  arrays.
- Repeated labels on an axis that has to be matched in another order are an
  alignment error naming the axis on both backends.
- Values are shared with in-memory input on both: wrapping does not copy.
