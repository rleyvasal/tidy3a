# Backends

tidy3A's verbs, rules, and messages are the same on every backend; a backend
only stores the data and computes with it (plan milestone 9). The xarray
backend is the default. The NumPy backend is experimental:

```python
import tidy3a
tidy3a.set_backend("numpy")            # the default for new data
cloud = tidy3a.read_array(..., backend="numpy")
frame = tidy3a.tidy(dataset, backend="numpy")
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
| Coordinates other than an axis's own labels (a scalar `station`, an auxiliary `lat` along `point`), multi-level labels, time labels with a time zone | Held | Not held: the data stays on xarray, or `backend="numpy"` is an error |
| Arrays passed to a `where(predicate)` selection helper | xarray `DataArray` | The backend's `NumpyArray`, with `dims`, `sizes`, `shape`, `ndim`, `dtype`, and `values` |

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
- Results are bit-identical, including float means over an axis that is not
  first: each group is indexed as a mask would index it, so NumPy adds the
  values in the same order. Both backends use the same one-pass grouped
  reductions (`tidy3a.grouped`) where those stay bit-identical.
- Python's operators run without NumPy's floating-point warnings (division by
  zero); NumPy functions such as `sqrt` and `log` warn, as they do on xarray
  arrays.
- Repeated labels on an axis that has to be matched in another order are an
  alignment error naming the axis on both backends.
- Values are shared with in-memory input on both: wrapping does not copy.
