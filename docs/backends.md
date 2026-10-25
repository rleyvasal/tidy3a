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
| Coordinates other than an axis's own labels (a scalar `station`, an auxiliary `lat` along `point`), multi-level labels | Held | Not held: the data stays on xarray, or `backend="numpy"` is an error |
| Labels compared across number types (`int64` labels against `float64` ones) | Through float64, as pandas does: the label `2**53 + 1` matches `2**53`, so values can be paired with the wrong positions | Exactly: those labels differ, and the expression is an alignment error. The NumPy backend follows the exact-alignment rule (plan §5.5); the xarray backend's behaviour comes from pandas |
| Arrays passed to a `where(predicate)` selection helper | xarray `DataArray` | The backend's `NumpyArray`, with `dims`, `sizes`, `ndim`, `dtype`, and `values` |

Behaviour that is the same on both, once easy to get wrong:

- Results are bit-identical, including float means over an axis that is not
  first: each group is indexed as a mask would index it, so NumPy adds the
  values in the same order.
- Python's operators run without NumPy's floating-point warnings (division by
  zero); NumPy functions such as `sqrt` and `log` warn, as they do on xarray
  arrays.
- Repeated labels on an axis that has to be matched in another order are an
  alignment error naming the axis on both backends.
- Values are shared with in-memory input on both: wrapping does not copy.
