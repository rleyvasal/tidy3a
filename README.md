# tidy3a

Tidy-style verbs for labelled multidimensional arrays: point clouds,
gridded climate data, images, sensor measurements, and complex-valued
arrays. Pipelines of `select`, `filter`, `mutate`, `group_by`, and
`summarise` work on named axes and labels, in the spirit of
[tidy3](https://github.com/rleyvasal/tidy3) and dplyr. It needs only NumPy
(and [labelarray](https://github.com/rleyvasal/labelarray), its array layer,
which needs only NumPy too); xarray is optional.

**Status:** version `0.5.1`, on [PyPI](https://pypi.org/project/tidy3a/).
What changed in each release is in [CHANGELOG.md](CHANGELOG.md); the design
is in [docs/plan.md](docs/plan.md), and the rules fit on one page in
[docs/decisions.md](docs/decisions.md).

## Installing

Requires Python 3.11 or newer.

```bash
pip install tidy3a                  # NumPy and labelarray only
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

voxels = cloud >> group_by(cells(x, 2.0), rounded(z, 0.5), axis="voxel") >> summarise(points=n())
print(voxels >> pull("x_cell"))  # ['[-4, -2)' '[-4, -2)' '[0, 2)' '[0, 2)']
print(voxels >> pull("z_rounded"))  # [0.  0.5 0.5 1. ]
print(voxels >> pull("points"))  # [1 1 2 1]
```

Several value keys make one group per combination that occurs, as dplyr,
pandas, and Polars do, sorted by the first key and then the next. The
groups lie along one new axis, `group` unless `axis=` names it, and
`summarise` gives each key's value per group as a variable named after the
key. Empty combinations are never made, so voxels cost what is occupied,
not the bounding box.

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

## Explaining and reusing pipelines

Each verb records what it did as it runs. `explain()` lists the source and
every step with its effect, from those records, computing nothing; verbs
also compose before any data into a pipeline to run, preview, or reuse:

```python
import numpy as np
from tidy3a import fields, filter, group_by, infer, mean, mutate, n, pull, read_array, summarise

points = np.array([[3, 4, 0, 10, 1], [0, 0, 50, 20, 2], [60, 0, 0, 30, 1], [1, 1, 1, 40, 2]], dtype="float32")
cloud = read_array(points, dims={"point": infer, "feature": 5},
                   labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
x, y, z, intensity, ring, d = fields("x", "y", "z", "intensity", "ring", "d")

steps = mutate(d=x + y + z) >> filter(d <= 50) >> group_by(ring) >> summarise(points=n(), m=mean(intensity))
per_ring = cloud >> steps
print(per_ring >> pull("m"))  # [10. 30.]
print(len(per_ring.plan()))  # 4
```

`print(per_ring.explain())` shows:

```text
<plan source: NumPy array>
  values (point: 4, feature: 5) float32
1. mutate(d = (x + y) + z)
     + d (point: 4) float32
2. filter(d <= 50)
     point: 4 → 3
3. group_by(ring)
     groups: ring (2)
4. summarise(points = n(), m = mean(intensity))
     - axis point
     - axis feature
     + axis ring: 2
     - values
     - d
     + points (ring: 2) int64
     + m (ring: 2) float32
     field axis: feature → none
     groups: none
```

`steps.explain(cloud, head=1000)` runs a pipeline on the first 1,000
positions of each observation axis and explains that, to check a long
pipeline on a large array quickly. `per_ring.plan()` gives the verbs that
made a result as a pipeline, so `other_cloud >> per_ring.plan()` repeats
them on other data.

## Files larger than memory

`read_array(path, ..., blocks=rows)` reads a file in blocks of that many
positions on its first axis. The result holds the first block, so each verb
runs on it as it is called, with its checks, notes, and `explain()`; when a
result is asked for (`pull`, `collect()`, `to_dict()`), the verbs run over
every block, `threads` blocks at a time, and the blocks' summaries merge,
so memory is the blocks in flight and the groups, not the file. The same
`blocks=` on an in-memory array keeps the verbs' intermediates block-sized,
and `tidy(dataset, blocks={"time": 100})` runs an xarray dataset, lazy or
Dask-backed, a block of time steps at a time (`blocks="chunks"` takes its
own chunks), each block computed and wrapped on its own:

```python
import os
import tempfile

import numpy as np
from tidy3a import fields, filter, formula, group_by, infer, mean, mutate, n, pull, read_array, summarise

rng = np.random.default_rng(0)
points = (rng.normal(size=(100_000, 5)) * 30).astype("float32")
points[:, 4] = rng.integers(0, 4, 100_000)
path = os.path.join(tempfile.mkdtemp(), "cloud.npy")
np.save(path, points)

x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
cloud = read_array(path, dims={"point": infer, "feature": 5},
                   labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature",
                   blocks=20_000, threads=2)
per_ring = (
    cloud
    >> mutate(distance = formula(r"\sqrt{x^2 + y^2 + z^2}"))
    >> filter(distance <= 60)
    >> group_by(ring)
    >> summarise(points = n(), brightness = mean(intensity))
)
print(per_ring >> pull("ring"))  # [0. 1. 2. 3.]
```

A summarised pipeline is `select`, `filter`, and `mutate` of row-wise
expressions, `group_by` of value keys (fields, `cells`, `bins`, `rounded`,
several at once, with or without `dropna`), then `summarise` of `n`,
`sum`, `mean`, `min`, `max`, `std`, `var`, `any`, `all`, or expressions
of them (`mean(x) / n()`). Each key's groups merge across blocks in the
order `group_by` gives them, and float32 sums and means are the same bits
as in memory whatever the block size; float64 sums and means, integer
means, and `std` and `var` add per block in float64 and are close, not
identical; integer sums add in int64, wrapping as `np.sum` does. A
pipeline without `summarise` joins the blocks' rows in memory,
for data that fits once filtered; a verb that needs the whole axis
(`slice`, joins) then runs on the joined rows. What a block cannot answer
(a reduction inside `mutate` or `filter`, a grouped `mutate`, an axis key,
`median`, `over=`) is refused when the verb is called, naming `lazy=True`,
which memory-maps the file, as the whole-axis alternative. Measured on all
122.6M points of nuScenes Mini's lidar sweeps (2.45 GB) through a distance
filter and 1 m voxels (`cells(x, 1), cells(y, 1), cells(z, 1)`, 84,475 of
them), blocks of 2M points on four threads took 8.0 s at a peak of
1.65 GB, against 36.5 s and 7.14 GB in memory, with identical results. The
same file written twelve times (29.4 GB, on a 24 GB machine, from a USB
drive reading at 36 MB/s) ran at the drive's speed with a peak of 1.85 GB
and the same voxels and mean bits; the plan has the measurements
(decision 10, stage (b)).

## Tables: Polars and tidy3

With [Polars](https://pola.rs) installed (`pip install "tidy3a[polars]"`),
`as_table()` gives data with one observation axis as a Polars DataFrame,
one row per position, and `tidy(table, axis=...)` reads a table back:

```python
import numpy as np
from tidy3a import as_table, col, filter, infer, pull, read_array, tidy

points = np.array([[3, 4, 0], [0, 0, 50], [60, 0, 0]], dtype="float32")
cloud = read_array(points, dims={"point": infer, "feature": 3},
                   labels={"feature": ["x", "y", "z"]}, field_axis="feature")
table = cloud >> filter(col("x") < 10) >> as_table()
print(table.columns)  # ['point', 'x', 'y', 'z']
print(table["point"].to_list())  # [1, 2]

back = tidy(table, axis="point", field_axis="feature", fields=["x", "y", "z"])
print(back >> pull("z"))  # [ 0. 50.]
```

The axis's labels become a column named after it (original positions after
a filter, as `pull` gives them), and field-axis labels and the other
variables along the axis become columns. Data with more than one
observation axis needs `as_table(long=True)`, one row per cell. Reading a
table, each column becomes a variable along `axis=`, a column of that name
becomes its labels, and `fields=` packs columns into one array along
`field_axis=`. A LazyFrame is collected, with a note; Polars nulls become
NaN, promoting integer and boolean columns to floats, also with a note.

With [tidy3](https://github.com/rleyvasal/tidy3) installed, `tidy3.tidy(cloud)`
gives a tidy3 frame (tidy3a defines the `__tidy__` hook tidy3 calls), and
`tidy(frame, axis="point")` reads one back.

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
with or without `fast`, and on lazy, chunked (Dask) data whatever the
chunking: each chunk is summed in its own task, the chunk sums combine in
any order, and the few results an error bound cannot settle get a second
pass for their exact sums. Float64 data follows NumPy. Integer arithmetic
follows NumPy too, including unsigned subtraction wrapping around: convert
first (`nir * 1.0 - red`) when a result can be negative.

## The API at a glance

| | |
|---|---|
| Reading | `read_array` (raw files, `.npy`, NumPy arrays, `lazy=True` to memory-map, `blocks=` to run the verbs block by block), `read_binary` (fixed-width mixed records), `fixed_text`, `infer`, `tidy()` (xarray objects, or Polars and tidy3 tables with `axis=`) |
| Verbs | `select`, `filter`, `slice`, `slice_head`, `slice_tail`, `mutate`, `group_by`, `ungroup`, `summarise`, `left_join`, `inner_join`, `full_join`, `bind_cols`, `pull` |
| Grouping keys | a field or an axis name, `bins`, `rounded`, `cells` |
| Reductions | `mean`, `sum`, `min`, `max`, `median`, `std`, `var`, `any`, `all`, `n`, with `over=` and `na_rm=` |
| Expressions | `fields`, `col`, `variable`, `label`, `env`, `if_else`, `[x, y, z]` vectors, `assign`, `formula` (plain maths or a strict LaTeX dialect) |
| Arrays | `dot`, `pack`, `rename_axes`, `conj`, `real`, `imag`, `angle`, `sqrt`, `exp`, `log`, trigonometric functions, … |
| Selecting fields | `where`, `starts_with`, `ends_with`, `contains`, `matches`, `everything`, `is_numeric`, … |
| Plans | `explain()` (each step and its effect), `plan()` (the verbs as a pipeline), `verb >> verb` (a `Pipeline`), `Pipeline.explain(data, head=)` |
| Results | `pull`, `collect()` (an xarray `Dataset`), `to_xarray()`, `to_dict()` (NumPy arrays, no xarray needed), `as_table()` (a Polars DataFrame) |
| Settings | `set_backend` |

An array from the calling code, such as a rotation matrix, can be an xarray
`DataArray` or a one-variable `read_array(...)` result.

Laziness: `read_array(..., lazy=True)` memory-maps a file and `tidy()` keeps
a lazy xarray input lazy until `collect()`, but the verbs themselves run
eagerly: arithmetic allocates results and `filter` evaluates its mask.
`read_array(..., blocks=rows)` runs them block by block instead, for
pipelines that end in `summarise` (above).
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

From a checkout, install the development dependencies and run the tests
(the suite runs every test on both backends):

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
workflow. tidy3a's arrays, exact labels, grouped reductions, and float32
sums come from [labelarray](https://github.com/rleyvasal/labelarray), a
library of their own that needs only NumPy and has its own tests; the name
`tidy3a.labelarray` still imports it.

## License

MIT
