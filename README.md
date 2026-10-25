# tidy3a

Tidy-style verbs for named multidimensional arrays: point clouds, images,
sensor measurements, scientific and complex-valued arrays. Built on xarray
and NumPy, in the spirit of [tidy3](https://github.com/rleyvasal/tidy3).

**Status:** version `0.2.0`; `0.1.0` is on [PyPI](https://pypi.org/project/tidy3a/).
The design decisions are in [docs/plan.md](docs/plan.md); the rules fit on one page in
[docs/decisions.md](docs/decisions.md).

Requires Python 3.11 or newer and NumPy. From a checkout, install the package
with xarray, for xarray inputs and outputs and for lazy and memory-mapped data:

```bash
python -m pip install -e ".[xarray]"
```

Without the extra, tidy3a needs NumPy alone: in-memory pipelines run on the
experimental NumPy backend, arrays come in through `read_array` and
`read_binary`, and results go out through `pull()` and `to_dict()`. The `fast`
extra (`".[xarray,fast]"`) adds Numba for compiled grouped sums and means;
results are the same with or without it.

## Quick start

```python
import numpy as np
from tidy3a import fields, filter, formula, infer, mutate, pull, read_array

points = np.array([[3, 4, 0], [0, 0, 50], [60, 0, 0]], dtype="float32")
cloud = read_array(
    points,
    dims={"point": infer, "feature": 3},
    labels={"feature": ["x", "y", "z"]},
    field_axis="feature",
)
x, y, z, distance = fields("x", "y", "z", "distance")

nearby = (
    cloud
    >> mutate(distance = formula(r"\sqrt{x^2 + y^2 + z^2}"))
    >> filter(distance <= 50)
)
print(nearby >> pull(distance))  # [ 5. 50.]
```

The implemented API includes `read_array`, `read_binary`, `fixed_text`,
`infer`, `tidy()`, tidy-layer
verbs (`select`, `filter`, `slice`, `mutate`, `group_by`, `summarise`, joins,
`pull`, `bind_cols`), array helpers (`dot`, numeric functions, `conj`,
`rename_axes`, `pack`), and `formula()` for plain mathematical text and a
strict LaTeX dialect. Scripts declare names with
`fields()` or write `col("x")`. `collect()` returns a computed xarray
Dataset; `to_xarray()` returns a shallow copy sharing the underlying values;
`to_dict()` returns NumPy arrays in xarray's `to_dict` layout. An array from
the calling code, such as a rotation matrix, is an xarray `DataArray` or a
one-variable `read_array(...)` result.

## Notebooks

Install IPython separately to use the notebook extension. In a notebook cell:

```ipython
%load_ext tidy3a.jupyter
from tidy3a import *
```

The extension enables bare field names inside verbs, so the quick-start
example can omit the `fields()` declaration. Multi-line pipes need
parentheses. Sharing a notebook with tidy3 requires its ownership-aware
transformer; the prerequisite is recorded in [the plan](docs/plan.md#56-notebook-layer).

## Execution and first-release scope

The first release runs eagerly on NumPy. `read_array` accepts homogeneous
raw numeric files, `.npy` files, and NumPy arrays. `read_binary` reads mixed
fixed-width records into separate variables, preserving numeric types and
strictly decoding UTF-8/ASCII text declared with `fixed_text`. Records can
be packed or have explicit field offsets and padding; see the
[reader contract](docs/m6-read-binary.md) for supported options. `tidy()` wraps xarray
DataArrays and Datasets; optional file readers stay in xarray's ecosystem.

`read_array(..., lazy=True)` memory-maps raw and `.npy` files. It does not
make the verb pipeline lazy: arithmetic can allocate results, filtering evaluates its
mask, and updating a field label copies its containing array into memory.
`tidy()` preserves a lazy xarray input initially; `collect()` computes it.
Dask execution guarantees remain planned.

Integer arithmetic follows NumPy, including unsigned subtraction wrapping
around. Convert before subtracting when negative results are needed, for
example `nir * 1.0 - red` for uint8 image bands. See
[first-release limitations](docs/first-release.md) for the supported subset.

## Examples and development

The point-cloud example runs on real nuScenes Mini data and checks itself
against plain NumPy:

```bash
python examples/nuscenes_sweep.py /path/to/nuscenes-mini
```

Three public sample datasets (gridded air temperature, ERA-Interim wind, and
stars' Landsat scene) run against plain xarray; see
[examples/sample_data.py](examples/sample_data.py) for the files, optional
reader dependencies, and checks.

To run the local tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Before pushing, run the CI checks in fresh environments with only declared
dependencies (requires [uv](https://docs.astral.sh/uv/)):

```bash
scripts/check.sh
```

The [release procedure](docs/releasing.md) describes the GitHub Release
workflow and the one-time GitHub/PyPI setup.

## License

MIT
