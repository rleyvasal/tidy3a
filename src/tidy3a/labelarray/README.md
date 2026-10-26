# labelarray

Labelled N-D arrays in pure NumPy, with exact label semantics and
reproducible reductions. It is the lower layer of tidy3a, for now the
subpackage `tidy3a.labelarray`, and will become a package of its own once
its API has settled. It needs NumPy only; xarray is used only to convert to
and from xarray objects.

It is deliberately small. What it offers that other array libraries do not:

1. **Exact labels.** Labels match only when they are the same kind
   (booleans, numbers, times, durations, text, bytes) and exactly equal, in
   lookups, alignment, and joins. The integer `2**53 + 1` never matches the
   float `2**53`; pandas, and so xarray, pair labels through float64.
2. **Reproducible reductions.** Float32 sums and means are correctly
   rounded: the float32 nearest the exact result, the same in any memory
   layout. Grouped reductions are bit-identical to reducing each group on
   its own.
3. **Collections.** Arrays sharing axes, with one label array per axis,
   joined and aligned under those rules.

## Arrays

An `Array` is a NumPy array with one name per axis, labels for any of its
axes, attributes, and a name:

```python
import numpy as np
from tidy3a.labelarray import Array

air = Array(
    np.array([[271.0, 272.0, 273.0], [281.0, 282.0, 283.0]], dtype="float32"),
    dims=("lat", "time"),
    coords={"lat": [10.0, 20.0], "time": np.arange("2026-01-01", "2026-01-04", dtype="datetime64[D]")},
    attrs={"units": "K"},
    name="air",
)
print(air.sel(lat=20.0).data)  # [281. 282. 283.]
print(air.isel(time=[0, 2]).labels("time"))  # ['2026-01-01' '2026-01-03']
print(air.reduce("mean", ["time"]).data)  # [272. 282.]
```

`isel` takes positions, slices, or masks; `sel` takes labels, looked up
exactly; one label or position removes the axis, a list keeps it.
`transpose`, `rename` (axes), `reduce` (sum, mean, min, max, median, std,
var, any, all, count, with `na_rm` and `min_count`), and `dot` (the sum of
products over named axes) complete the methods.

NumPy's ufuncs, `np.where`, and Python's operators broadcast by axis name.
Labels on a shared axis must match: the same labels in another order are
matched by label, and anything else is an `AlignError`. There is no outer
join by default; ask for one with `Collection.join`.

```python
import numpy as np
from tidy3a.labelarray import Array

a = Array([1.0, 2.0, 3.0], ("x",), {"x": [1, 2, 3]})
b = Array([30.0, 10.0, 20.0], ("x",), {"x": [3, 1, 2]})
print((a + b).data)  # [11. 22. 33.]
print(np.sqrt(a * 4).data)  # [2.         2.82842712 3.46410162]
```

Labels are copied once into a read-only array the library owns, and each
label array's index is built on first use and kept with it: a sorted index,
a sort-once index for unsorted labels, or a grid index for evenly spaced
numbers and times, which looks labels up arithmetically. `register_index`
adds index classes for other label types, under the same exactness
obligations.

## Collections

A `Collection` holds variables with different axes, each axis with one size
and at most one label array:

```python
from tidy3a.labelarray import Array, Collection

left = Collection({"a": Array([1.0, 2.0, 3.0], ("k",))}, {"k": [1, 3, 5]})
right = Collection({"b": Array([10.0, 20.0], ("k",))}, {"k": [5, 1]})
both = left.join(right, "inner")
print(both.coords["k"])  # [1 5]
print(both.array("b").data)  # [20. 10.]
print(left.join(right, "left").array("b").data)  # [20. nan 10.]
```

`isel`, `sel`, `select` (variables), `assign`, `rename` (variables), `join`
(left, inner, or outer), `nearest` (each label matched to its nearest within
a tolerance), `to_dict`, `to_xarray`, and `from_xarray` are its methods.
Missing positions after a join promote integers and booleans as xarray
does, and attributes that conflict are dropped.

## Grouping

A `Grouping` assigns each position a group code; `factorize` makes one from
key values, optionally after a key transform (`Bins`, `Duration`, `Round`,
`GridCells`), with every missing key (NaN, NaT) one group, dropped unless
asked for. `grouped.reduce` reduces every group at once:

```python
import numpy as np
from tidy3a.labelarray import Array, Grouping, factorize, grouped

intensity = Array(np.array([10, 20, 30, 40, 50], dtype="float32"), ("point",))
codes, labels = factorize(np.array([0, 0, 1, 1, 2]))
rings = Grouping(codes, labels)
print(grouped.reduce(intensity, "mean", rings, "point", "ring").data)  # [15. 35. 50.]
print(grouped.counts(rings, "ring").data)  # [2 2 1]
```

`grouped.broadcast` puts a result per group back on each position.

## Errors

Every error is a `LabelArrayError` (a `ValueError`): `LabelError` for labels
that cannot be found, ordered, or compared; `AlignError` for axes that
cannot be paired; `GroupError` for keys that cannot be grouped; and
`ShapeError` for axis names, sizes, and ranks that do not fit.
