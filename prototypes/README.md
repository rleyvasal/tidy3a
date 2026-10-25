# Prototypes

`test_examples.py` and `test_xarray_assumptions.py` are milestone 0's
checks: the acceptance examples in plain xarray, and the xarray behaviour the
plan relies on.

The NumPy-only core prototyped after the 0.1.0 release now lives in the
package as the experimental NumPy backend (`src/tidy3a/numpy_backend.py`,
plan milestone 9). Its label index and tests moved with it
(`tests/test_numpy_labels.py`); the prototype's own pipeline, which
implemented every verb a second time, was replaced by tidy3a's verbs running
on the backend. The prototype as it was is in git history (commit `c878eac`).

`demo_numpy_pipeline.py` runs the point-cloud example on the NumPy backend:

```bash
.venv/bin/python prototypes/demo_numpy_pipeline.py
```
