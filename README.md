# tidy3a

Tidy-style verbs for labelled multidimensional arrays: point clouds,
gridded climate data, images, sensor measurements, and complex-valued
arrays. Pipelines of `select`, `filter`, `mutate`, `group_by`, and
`summarise` work on named axes and labels, in the spirit of
[tidy3](https://github.com/rleyvasal/tidy3) and dplyr. NumPy is the only
requirement; xarray is optional.

**Status:** version `0.3.0`, on [PyPI](https://pypi.org/project/tidy3a/).
What changed in each release is in [CHANGELOG.md](CHANGELOG.md); the design
is in [docs/plan.md](docs/plan.md), and the rules fit on one page in
[docs/decisions.md](docs/decisions.md).

## Installing

Requires Python 3.11 or newer.

```bash
pip install tidy3a                  # NumPy only
pip install "tidy3a[xarray]"        # with xarray and pandas: xarray inputs and outputs, lazy and file-backed data
pip install "tidy3a[xarray,fast]"   # also Numba, for compiled grouped sums and means
```

Results are the same with or without `fast`; it only makes large grouped
sums and means faster.

## Quick start

Read an array, name its axes, compute a value per point, and keep the
points within 50 m:

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

`field_axis="feature"` makes the labels on that axis act as columns, so
`x`, `y`, and `z` name fields of the array. Verbs chain with `>>`; `pull`
takes one result out as a NumPy array.

## Grouping

Group by a value, by an axis, or by a transform of a value, then summarise
each group or compute within it:

```python
import numpy as np
from tidy3a import cells, fields, group_by, infer, mean, mutate, n, pull, read_array, rounded, summarise

points = np.array([
    # x,    y,    z, intensity, ring
    [ 1.0,  2.0, 0.5,  10, 0],
    [ 1.5,  2.5, 0.4,  20, 0],
    [-3.0,  0.5, 0.2,  30, 1],
    [-3.2,  0.1, 0.3,  40, 1],
    [ 0.2,  4.1, 0.9,  50, 2],
], dtype="float32")
cloud = read_array(points, dims={"point": infer, "feature": 5},
                   labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
x, y, z, intensity, ring = fields("x", "y", "z", "intensity", "ring")

per_ring = cloud >> group_by(ring) >> summarise(points=n(), brightness=mean(intensity))
print(per_ring >> pull("brightness"))  # [15. 35. 50.]

centred = cloud >> group_by(ring) >> mutate(centred=intensity - mean(intensity))
print(centred >> pull("centred"))  # [-5.  5. -5.  5.  0.]

per_cell = cloud >> group_by(cells(x, 2.0)) >> summarise(points=n())
print(per_cell >> pull("x_cell"))  # ['[-4, -2)' '[0, 2)']

heights = cloud >> group_by(rounded(z, 0.5)) >> summarise(points=n())
print(heights >> pull("points"))  # [1 3 1]
```

`bins(field, edges)` groups by explicit edges, or by a duration for times
(`bins(col("time"), "1 day")`). Cell and bin edges fall where the printed
settings put them: with a size of 0.1, the value 0.3 is in `[0.3, 0.4)`.

A missing key (NaN, or NaT for times) is one group, left out by default
with a note saying how many positions it held; `group_by(..., dropna=False)`
keeps it as the last group:

```python
import numpy as np
from tidy3a import fields, group_by, infer, n, pull, read_array, summarise

values = np.array([[10, 0], [20, 0], [30, 1], [40, np.nan]], dtype="float32")
data = read_array(values, dims={"point": infer, "feature": 2},
                  labels={"feature": ["intensity", "ring"]}, field_axis="feature")
intensity, ring = fields("intensity", "ring")

print(data >> group_by(ring) >> summarise(points=n()) >> pull("points"))  # [2 1]
print(data >> group_by(ring, dropna=False) >> summarise(points=n()) >> pull("ring"))  # [ 0.  1. nan]
```

## Working with xarray

`tidy()` wraps an xarray `Dataset` or `DataArray`, and `collect()` gives the
result back as a `Dataset`:

```python
import numpy as np
import xarray as xr
from tidy3a import bins, col, group_by, mean, pull, summarise, tidy

times = np.datetime64("2026-01-01T00", "ns") + np.arange(8) * np.timedelta64(6, "h")
air = xr.Dataset(
    {"air": (("time", "lat"), np.arange(16, dtype="float64").reshape(8, 2) + 270)},
    coords={"time": times, "lat": [10.0, 20.0]},
)

per_lat = tidy(air) >> group_by("lat") >> summarise(mean_air=mean(col("air")))
print(per_lat >> pull("mean_air"))  # [277. 278.]

daily = tidy(air) >> group_by(bins(col("time"), "1 day"), "lat") >> summarise(mean_air=mean(col("air")))
result = daily.collect()
print(dict(result.sizes))  # {'time_bin': 2, 'lat': 2}
```

Labels on a shared axis must match exactly: the same labels in another order
are matched by label (and the change is noted), and different labels need a
join (`left_join`, `inner_join`, `full_join`). The integer label `2**53 + 1`
never matches the float `2**53`.

## Backends

New data uses the NumPy backend wherever it holds the data without loss, and
the xarray backend for the rest: lazy, chunked, and memory-mapped values,
coordinates other than an axis's labels, multi-level labels, time zones, and
data with xarray storage encodings, so a netCDF file read and written back
keeps them. Results are the same on both. `set_backend("xarray")` chooses
xarray for all new data, as before 0.3.0; [docs/backends.md](docs/backends.md)
lists the few differences.

## Numbers

Float32 sums and means add in float64 and round once. Every grouped one, and
every ungrouped one over 8 or more values, is correctly rounded: the float32
nearest the exact result, the same for any memory layout, on either backend,
with or without `fast`. Float64 data follows NumPy. Integer arithmetic
follows NumPy too, including unsigned subtraction wrapping around: convert
first (`nir * 1.0 - red`) when a result can be negative.

## The API at a glance

| | |
|---|---|
| Reading | `read_array` (raw files, `.npy`, NumPy arrays, `lazy=True` to memory-map), `read_binary` (fixed-width mixed records), `fixed_text`, `infer`, `tidy()` |
| Verbs | `select`, `filter`, `slice`, `slice_head`, `slice_tail`, `mutate`, `group_by`, `ungroup`, `summarise`, `left_join`, `inner_join`, `full_join`, `bind_cols`, `pull` |
| Grouping keys | a field or an axis name, `bins`, `rounded`, `cells` |
| Reductions | `mean`, `sum`, `min`, `max`, `median`, `std`, `var`, `any`, `all`, `n`, with `over=` and `na_rm=` |
| Expressions | `fields`, `col`, `variable`, `label`, `env`, `if_else`, `[x, y, z]` vectors, `assign`, `formula` (plain maths or a strict LaTeX dialect) |
| Arrays | `dot`, `pack`, `rename_axes`, `conj`, `real`, `imag`, `angle`, `sqrt`, `exp`, `log`, trigonometric functions, … |
| Selecting fields | `where`, `starts_with`, `ends_with`, `contains`, `matches`, `everything`, `is_numeric`, … |
| Results | `pull`, `collect()` (an xarray `Dataset`), `to_xarray()`, `to_dict()` (NumPy arrays, no xarray needed) |
| Settings | `set_backend` |

An array from the calling code, such as a rotation matrix, can be an xarray
`DataArray` or a one-variable `read_array(...)` result.

Laziness: `read_array(..., lazy=True)` memory-maps a file and `tidy()` keeps
a lazy xarray input lazy until `collect()`, but the verbs themselves run
eagerly: arithmetic allocates results and `filter` evaluates its mask.
`read_binary`'s options are in [the reader contract](docs/m6-read-binary.md).

## Notebooks

Install IPython separately to use the notebook extension. In a notebook cell:

```ipython
%load_ext tidy3a.jupyter
from tidy3a import *
```

The extension enables bare field names inside verbs, so the examples above
can omit `fields(...)`. Multi-line pipes need parentheses. Sharing a notebook
with tidy3 requires its ownership-aware transformer; the prerequisite is
recorded in [the plan](docs/plan.md#56-notebook-layer).

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

From a checkout, install the development dependencies and run the tests:
tidy3a's suite in `tests/` runs every test on both backends, and the
library's in `labelarray_tests/` imports nothing of tidy3a but
`tidy3a.labelarray`. `python -m pytest -q` runs both.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Before pushing, run the CI checks in fresh environments with only declared
dependencies (requires [uv](https://docs.astral.sh/uv/)):

```bash
scripts/check.sh
```

`scripts/bench.sh` times both backends on the reference workloads in a
pinned environment (`--fast` adds the compiled kernels). The
[release procedure](docs/releasing.md) describes the GitHub Release
workflow. The NumPy layer is becoming a library of its own, `labelarray`,
for now the subpackage `tidy3a.labelarray`, with
[its own README](src/tidy3a/labelarray/README.md).

## License

MIT
