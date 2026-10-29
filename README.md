# tidy3a

Tidy-style verbs for labelled multidimensional arrays: point clouds,
gridded climate data, images, sensor measurements, and complex-valued
arrays. Pipelines of `select`, `filter`, `mutate`, `group_by`, and
`summarise` work on named axes and labels, in the spirit of
[tidy3](https://github.com/rleyvasal/tidy3) and dplyr. It needs only NumPy
(and [labelarray](https://github.com/rleyvasal/labelarray), its array layer,
which needs only NumPy too); xarray is optional.

**Status:** version `0.11.0`, on [PyPI](https://pypi.org/project/tidy3a/).
What changed in each release is in [CHANGELOG.md](CHANGELOG.md); the design
is in [docs/plan.md](docs/plan.md), and the rules fit on one page in
[docs/decisions.md](docs/decisions.md).

## Installing

Requires Python 3.11 or newer.

```bash
pip install tidy3a                  # NumPy and labelarray only
pip install "tidy3a[xarray]"        # with xarray and pandas: xarray inputs and outputs, lazy and file-backed data
pip install "tidy3a[xarray,fast]"   # also Numba: compiled grouped sums and means, and summaries of blocks
```

Results are the same with or without `fast`; it only makes large grouped
sums and means, and summaries of data read in blocks, faster. tidy3a is
tested on Python 3.11 to 3.14 and on free-threaded 3.14 with the GIL off.

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
positions on its first axis. Nothing is read until a result is asked for:
each verb is checked as it is called, on a block of no positions with the
file's schema, and `repr` and `explain()` show what each step makes, with
`?` for a size the data decides. `preview()` runs the verbs on a sample of
blocks, by default the first, middle, and last, and gives that result in
memory. When a result is asked for (`pull`, `collect()`, `to_dict()`), the
verbs run over every block, `threads` blocks at a time, and the blocks'
summaries merge, so memory is the blocks in flight and the groups, not the
file; an error found then names its block ("in block 3 of 4 (positions
2,001–3,000 on point)"). The same
`blocks=` on an in-memory array keeps the verbs' intermediates block-sized,
and `tidy(dataset, blocks={"time": 100})` runs an xarray dataset, lazy or
Dask-backed, a block of time steps at a time (`blocks="chunks"` takes its
own chunks), each block computed and wrapped on its own, with only the
variables the pipeline reads, so a summary of one field of a wide
NetCDF or Zarr file reads that field and its keys, not the rest:

```python
import os
import tempfile

import numpy as np
from tidy3a import (cells, fields, filter, formula, group_by, infer, mean, median, mutate, n, pull, read_array, scan,
                    select, summarise, write)

rng = np.random.default_rng(0)
points = (rng.normal(size=(100_000, 5)) * 30).astype("float32")
points[:, 4] = rng.integers(0, 4, 100_000)
path = os.path.join(tempfile.mkdtemp(), "cloud.npy")
np.save(path, points)

x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
cloud = read_array(path, dims={"point": infer, "feature": 5},
                   labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature",
                   blocks=20_000, threads=2)
near = cloud >> mutate(distance = formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60)
per_ring = near >> group_by(ring) >> summarise(points = n(), brightness = mean(intensity),
                                               middle = median(distance))  # exact: two more passes
voxels = near >> group_by(cells(x, 30), cells(y, 30), axis="voxel") >> summarise(points = n())
sample = per_ring.preview()  # blocks 1, 3, and 5 of 5, in memory; nothing else is read
per_ring, voxels = scan(per_ring, voxels)  # one pass over the file for both
print(sample >> pull("ring"))  # [0. 1. 2. 3.]
print(per_ring >> pull("ring"))  # [0. 1. 2. 3.]
written = near >> select("values") >> write(os.path.join(tempfile.mkdtemp(), "near.npy"))  # block by block
print((written >> pull("values")).shape)  # (73686, 5)
```

`scan(*frames)` gives several frames' results from one pass over their
source: the frames' row-wise verbs form a trie, each block is loaded once
and the shared verbs (`mutate` and `filter` above) run once on it, and
each frame takes its summaries or its rows from where its own verbs branch
off; the float32 sums no error bound can settle get one shared second
pass. Verbs are matched by what they do, not by being the same object, so
two pipelines written apart share the steps they have in common. Each
result is the same bits as the frame's own run, and afterwards each frame
holds its result, as after `collect()`. A `select(where(...))` with a
function of yours, which picks columns by their values, is refused on data
read in blocks, since nothing has been read when the verb is checked and
each block could pick others: select by name or type, or `preview()` a
sample to see the columns and select them by name.

`write(path)` writes rows that need not fit in memory: a pipeline read in
blocks that keeps rows writes them as the blocks arrive, in order, to a
`.npy` file (one array; NumPy only) or a Zarr store (a dataset, appended
along the blocked axis; `pip install "tidy3a[zarr]"`), and gives the file
back as a frame read in blocks of the same size, so a pipeline goes on from
it. The file is written under a temporary name and renamed when every
block is in, so a failed run leaves nothing under the name given; an
existing path is refused unless `overwrite=True`. It reads back the same
bits as `collect()`; a `.npy` file holds values only, so the blocked axis
comes back without the positions a filter kept, which a Zarr store keeps.
`scan(summary, writes=[(rows, "near.zarr")])` writes beside other results
in the same pass.

`blocks="auto"` chooses the number of positions for you: the most whose
blocks fit `memory=` (bytes, or text such as `"2GB"`; a quarter of the
machine's memory by default) as the verbs run on `threads` of them at a
time, so `read_array(path, ..., blocks="auto", memory="2GB", threads=4)`
or `tidy(dataset, blocks={"time": "auto"})` need no guess. The size comes
from an estimate read off the data's types and what each verb makes of
them on the block of no positions, so nothing is read and the same data,
verbs, budget, and threads give the same size every time. It is chosen for
the verbs that run, when a result is asked for, and `repr` and `explain()`
show it before that ("8 blocks of 39,432 positions on point, sized for
memory=1 GB"): a summary of one field of a wide dataset gets larger blocks
than rows of every field. A dataset's blocks are whole chunks of its
storage (its Dask chunks, or the chunks a NetCDF4 or Zarr file stores), and
a scan's blocks are sized for all its frames together, so its float64
sums and means, integer means, and spreads may differ in their last bits
from a frame's own run, as between any two sizes. The estimate is a bound:
a filter is taken to keep every row, and the passes of a median or a
float32 sum to run. It leaves out what the data decides: the groups a
summary finds (one with about as many groups as positions holds about as
much as its data) and the rows a pipeline without `summarise` or `write`
keeps in memory. A budget that fits no position is refused when the data
is opened or the verb called, with the sizes.

The budget counts what the verbs allocate, not what the allocator keeps
after NumPy frees it. On Linux, glibc's `malloc` keeps a pool for each
thread, and memory freed there may stay with the process: the
540,000-voxel summary of 40M points with `memory="1GB"` on twelve threads
peaked at 1.10 GB, though its own allocations stayed well under the
budget, and at 0.70 GB, as fast, with the pools limited to two. To hold
runs on many threads to a tight budget there, start Python with
`MALLOC_ARENA_MAX=2` in its environment (`MALLOC_ARENA_MAX=2 python
run.py`); tidy3a leaves the allocator alone, since a limit set by a
running process stays for the rest of it and does nothing once the pools
exist.

With `threads=` above one, blocks run side by side, and their summaries
merge in a fixed tree on the worker threads, so every number of threads
gives the same bits. How much more threads give depends on the machine:
NumPy's steps each stream whole arrays through memory, so on a desktop
whose memory one thread nearly fills, two to four threads give most of
it. Where each position holds a grid of values with one group, as a time
step grouped by month does, the reductions take each position's values
together, which makes such summaries several times faster on any number
of threads.

With `fast` installed, a summary whose every step compiles runs each block
in a few compiled passes instead of NumPy's steps, each of which makes
whole arrays: one tests the filters and places each key's value, one adds
every value into its group's reductions. They release the GIL, so blocks
on many threads keep scaling. What compiles:
`+ - * /`, `**` with the exponents 2, 0.5, -1, and 1 (so
`\sqrt{x^2 + y^2 + z^2}` does), `//` and `%` (NumPy's floored rules, step
by step), comparisons, `& | ~`, `sqrt`, `abs`, `floor`, `ceil`, `sign`,
and `if_else`, on booleans, integers, float32, and float64; keys of
integer, boolean, and whole float fields and their `cells`, `bins`, and
`rounded`, and of times and durations, each a group or in `bins` of
times; and the built-in reductions. `exp`, `log`, the trigonometric and
hyperbolic functions, and other powers are NumPy's own routines, which no
kernel matches on every machine, so NumPy computes them for each block,
as islands the kernel reads, a kernel computing their operands: the
results stay NumPy's. The rest (half-precision and complex values, text
keys, a time in an expression, `conj`, `dot`, a reduction of your own)
runs on NumPy as before. Before a run, `repr` says which ("planned run: a
compiled kernel a block, with NumPy for exp(-distance / 20)", or "NumPy,
as mutate(e = conj(x)): conj() runs on NumPy") and `explain()` lists every
step that does not compile and why; afterwards the result notes how its
blocks ran ("ran: 40 blocks in a compiled kernel").
A block runs on NumPy instead, with the same result, where one of its
values is one the kernel cannot match NumPy on (a float key that is not
whole, a negative zero, an infinity). Where its keys' groups make more
combinations than dense tables hold (16 million, or what a block's share
of `memory=` allows), the block sorts the combinations its positions hold
and keeps tables only for those, still compiled. The
results are NumPy's bits, but for one case: on x86, NumPy's `min` and `max`
of a -0.0 and a +0.0 give whichever they meet second, where the kernels,
as NumPy on ARM, give -0.0 and +0.0. Kernels are compiled the first time
a summary of their shape runs and kept in `~/.cache/tidy3a/kernels`
(`TIDY3A_KERNELS` moves them), so later runs, in any process, read them;
the folder is held to 1 GB (`TIDY3A_KERNELS_MAX`), the least recently used
kernels going first, and `tidy3a.fused.cache.clear()` empties it.
`TIDY3A_COMPILED=0` turns compiling off. On a 12-core machine, the rings
summary of 40M points in memory (the distance, a filter, and the count and
mean brightness per ring) took 0.10 s on twelve threads against 0.42 s on
NumPy, the daytime summary of 2,000 grids 0.038 s against 0.069 s, and
the 540,000-voxel summary 1.39 s against 3.16 s; with islands, the rings
weighted by `exp(-distance / 20)` with the mean of
`log(abs(intensity) + 1)` 0.16 s against 0.51 s, and the daytime summary
with the vapour pressure, `6.112 * exp(17.67 * c / (c + 243.5))`, 0.074 s
against 0.185 s; 10M timestamped values in weekly `bins` of times 0.038 s
against 0.199 s. A compiled block knows
which float32 totals no addition rounded, and those settle from the total
itself, so the voxels' means, many on a tie between two float32 values,
need no second pass for their exact sums.

A summarised pipeline is `select`, `filter`, and `mutate` of row-wise
expressions, `group_by` of value keys (fields, `cells`, `bins`, `rounded`,
several at once, with or without `dropna`), then `summarise` of `n`,
`sum`, `mean`, `min`, `max`, `median`, `quantile`, `std`, `var`, `any`,
`all`, or expressions of them (`mean(x) / n()`). Each key's groups merge across blocks in the
order `group_by` gives them, and float32 sums and means are the same bits
as in memory whatever the block size; float64 sums and means, integer
means, and `std` and `var` add per block in float64 and are close, not
identical; integer sums add in int64, wrapping as `np.sum` does. The
median and quantiles are exact, the same bits as in memory: each group's
values are counted by the bits of an order-preserving key, 16 bits a pass,
until the rank is found, so float32 takes two passes after the first and
float64 four, which a `scan` shares and which read only the variables
used. A pipeline without `summarise` joins the blocks' rows in memory,
for data that fits once filtered; a verb that needs the whole axis
(`slice`, joins) then runs on the joined rows. What a block cannot answer
(a reduction inside `mutate` or `filter`, a grouped `mutate`, an axis key,
`over=`) is refused when the verb is called, naming `lazy=True`,
which memory-maps the file, as the whole-axis alternative. Measured on all
122.6M points of nuScenes Mini's lidar sweeps (2.45 GB) through a distance
filter and 1 m voxels (`cells(x, 1), cells(y, 1), cells(z, 1)`, 84,475 of
them), blocks of 2M points on four threads took 8.0 s at a peak of
1.65 GB, against 36.5 s and 7.14 GB in memory, with identical results. The
same file written twelve times (29.4 GB, on a 24 GB machine, from a USB
drive reading at 36 MB/s) ran at the drive's speed with a peak of 1.85 GB
and the same voxels and mean bits; the plan has the measurements
(decision 10, stage (b)).

A reduction of your own (experimental): `register_reduction` takes its
NumPy function for data in memory and, for blocks, how one block's pieces
are made, merged, and finished. The function is also given no values, for
an empty group and for the block of no positions a frame read in blocks is
checked on, so it must answer then:

```python
import numpy as np
from tidy3a import fields, infer, pull, read_array, register_reduction, summarise


def eager(values, axes, na_rm):  # max - min over the axes; NaN for no values
    if any(values.shape[axis] == 0 for axis in axes):
        return np.full([size for axis, size in enumerate(values.shape) if axis not in axes], np.nan)
    return values.max(axis=axes) - values.min(axis=axes)


def partial(values, codes, groups, na_rm):  # one block: each group's lowest and highest value
    low, high = np.full(groups, np.inf), np.full(groups, -np.inf)
    np.minimum.at(low, codes, values)
    np.maximum.at(high, codes, values)
    return np.stack([low, high])


spread = register_reduction(
    "spread", eager, partial=partial,
    merge=lambda held, piece: np.stack([np.minimum(held[0], piece[0]), np.maximum(held[1], piece[1])]),
    finish=lambda held, counts: held[1] - held[0], fill=np.array([[np.inf], [-np.inf]]))
v = fields("v")
values = np.array([[3.0], [9.0], [4.0], [1.0]])
options = dict(dims={"point": infer, "feature": 1}, labels={"feature": ["v"]}, field_axis="feature")
in_memory = read_array(values, **options) >> summarise(s=spread(v))
in_blocks = read_array(values, blocks=2, **options) >> summarise(s=spread(v))
print(in_memory >> pull("s"))  # 8.0
print(in_blocks >> pull("s"))  # 8.0
```

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
| Reading | `read_array` (raw files, `.npy`, NumPy arrays, `lazy=True` to memory-map, `blocks=` to run the verbs block by block, `blocks="auto"` with `memory=` to size the blocks for a budget), `read_binary` (fixed-width mixed records), `fixed_text`, `infer`, `tidy()` (xarray objects, or Polars and tidy3 tables with `axis=`) |
| Verbs | `select`, `filter`, `slice`, `slice_head`, `slice_tail`, `mutate`, `group_by`, `ungroup`, `summarise`, `left_join`, `inner_join`, `full_join`, `bind_cols`, `pull` |
| Grouping keys | a field or an axis name, `bins`, `rounded`, `cells` |
| Reductions | `mean`, `sum`, `min`, `max`, `median`, `quantile`, `std`, `var`, `any`, `all`, `n`, with `over=` and `na_rm=`; `register_reduction` adds one of your own (experimental) |
| Expressions | `fields`, `col`, `variable`, `label`, `env`, `if_else`, `[x, y, z]` vectors, `assign`, `formula` (plain maths or a strict LaTeX dialect) |
| Arrays | `dot`, `pack`, `rename_axes`, `conj`, `real`, `imag`, `angle`, `sqrt`, `exp`, `log`, trigonometric functions, … |
| Selecting fields | `where`, `starts_with`, `ends_with`, `contains`, `matches`, `everything`, `is_numeric`, … |
| Plans | `explain()` (each step and its effect), `plan()` (the verbs as a pipeline), `verb >> verb` (a `Pipeline`), `Pipeline.explain(data, head=)` |
| Results | `pull`, `collect()` (an xarray `Dataset`), `to_xarray()`, `to_dict()` (NumPy arrays, no xarray needed), `as_table()` (a Polars DataFrame) |
| Settings | `set_backend`; the environment's `TIDY3A_COMPILED=0` (compile nothing) `TIDY3A_KERNELS` (where kernels are kept), and `TIDY3A_KERNELS_MAX` (how much of them) |

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
can omit `fields(...)`. A variable named like a builtin or a tidy3a helper
(`id`, `type`, `max`, `n`) is the data's wherever only data fits: `id > 1`,
`max * 2`, `mean(type)`. Multi-line pipes need parentheses. Sharing a notebook
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

`scripts/check.sh --parallel` runs the jobs side by side, for a machine with
cores to spare (about 5 minutes on 24 cores). The jobs are those CI runs:
Python 3.11 to 3.14, free-threaded 3.14 with the GIL off, the minimum NumPy
and xarray, and the distributions.

`scripts/bench.sh` times both backends on the reference workloads in a
pinned environment (`--fast` adds the compiled kernels). The
[release procedure](docs/releasing.md) describes the GitHub Release
workflow. tidy3a's arrays, exact labels, grouped reductions, and float32
sums come from [labelarray](https://github.com/rleyvasal/labelarray), a
library of their own that needs only NumPy and has its own tests; the name
`tidy3a.labelarray` still imports it.

## License

MIT
