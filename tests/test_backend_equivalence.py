"""Property tests: random data and pipelines give the same result on both backends.

Each seed builds a random dataset (axes of random sizes, some empty; labels
that are missing, integers, floats with NaN, text, or times; variables over
random subsets of the axes in random order, with missing values; sometimes a
field axis) and a random pipeline of verbs over it, including joins and
arithmetic with arrays whose labels are the same, reordered, partly
different, repeated, or of another number type. Both backends must give
the same printout and the same collected data, types included, or the same
error, word for word, and the same warnings.
"""

import pickle
import warnings

import numpy as np
import pytest
import xarray as xr

from tidy3a import (col, filter, full_join, group_by, inner_join, left_join, max, mean, min, mutate, n, select, slice,
                    std, sum, summarise, tidy)

pytestmark = pytest.mark.backends("xarray", reason="compares the backends itself")

AXES = ["point", "time", "sensor"]
FIELDS = ["x", "y", "z"]
SEEDS = range(300)
JOINS = [left_join, inner_join, full_join]
BIG = 2**53 + 1  # an integer that float64 cannot represent


def _labels(kind, size, rng):
    if kind == "int":
        labels = rng.permutation(np.arange(10, 10 + 3 * size, 3))[:size]
        if size and rng.random() < 0.5:
            labels[-1] = BIG
        return labels
    if kind == "float":
        values = rng.permutation(np.arange(size) * 0.25 + 0.5)
        if size > 1 and rng.random() < 0.5:
            values[rng.integers(size)] = np.nan
        return values
    if kind == "text":
        return np.array([f"s{k}" for k in rng.permutation(size)], dtype=object)
    if kind == "time":
        return np.datetime64("2026-01-01", "ns") + np.sort(rng.choice(30, size, replace=False)).astype("timedelta64[D]")
    raise AssertionError(kind)


def _other_labels(labels, rng):
    """Labels for other data on the same axis: the same, reordered, partly new, repeated, or another type."""
    choice = rng.integers(5)
    if choice == 1:
        return rng.permutation(labels)
    if choice == 2 and len(labels):
        return labels[rng.permutation(len(labels))[: rng.integers(len(labels) + 1)]]
    if choice == 3 and len(labels):
        return np.concatenate([labels[:1], labels[:-1]])  # the first label twice
    if choice == 4 and labels.dtype.kind == "i":
        return labels.astype("float64")  # the same values as floats, except BIG, which rounds
    return labels


def _other(axis, labels, rng):
    """One variable on *axis*, as an xarray array, with labels from ``_other_labels``."""
    other = _other_labels(labels, rng)
    return xr.DataArray(rng.normal(size=len(other)), dims=axis, coords={axis: other}, name="j")


def _values(dtype, shape, rng):
    if dtype == "bool":
        return rng.random(shape) < 0.5
    if dtype.startswith("int"):
        return rng.integers(-5, 6, shape).astype(dtype)
    values = rng.normal(size=shape).astype(dtype)
    if values.size and rng.random() < 0.5:
        values.flat[rng.integers(values.size)] = np.nan
    return values


def random_dataset(rng):
    axes = list(rng.permutation(AXES)[: rng.integers(1, 4)])
    sizes = {axis: int(rng.integers(0 if rng.random() < 0.1 else 1, 6)) for axis in axes}
    coords = {}
    for axis in axes:
        kind = rng.choice(["none", "int", "float", "text", "time"])
        if kind != "none":
            coords[axis] = _labels(kind, sizes[axis], rng)
    variables = {}
    for k in range(int(rng.integers(1, 4))):
        dims = list(rng.permutation(axes)[: rng.integers(1, len(axes) + 1)])
        dtype = str(rng.choice(["float32", "float64", "int16", "int64", "bool"]))
        variables[f"v{k}"] = (dims, _values(dtype, [sizes[d] for d in dims], rng), {"units": "m"} if k == 0 else {})
    first = axes[0]
    variables["key"] = ([first], rng.integers(0, 3, sizes[first]).astype("int64"))
    field_axis = None
    if rng.random() < 0.3:
        dims = [first, "feature"]
        variables["values"] = (dims, _values("float32", [sizes[first], 3], rng))
        coords["feature"] = FIELDS
        field_axis = "feature"
    return xr.Dataset(variables, coords=coords, attrs={"title": "random"}), axes, sizes, field_axis


def random_pipeline(rng, data, axes, sizes, field_axis):
    names = [v for v in data.data_vars if v.startswith("v")]
    numeric = [v for v in names if data[v].dtype != bool] or names
    steps = []
    for _ in range(int(rng.integers(3, 6))):
        choice = rng.integers(12)
        a, b = rng.choice(numeric), rng.choice(numeric)
        axis = str(rng.choice(axes))
        if choice == 0:
            op = rng.choice(["+", "-", "*", "/", ">"])
            expr = {"+": col(a) + col(b), "-": col(a) - col(b), "*": col(a) * col(b),
                    "/": col(a) / col(b), ">": col(a) > col(b)}[op]
            steps.append(mutate(w=expr))
        elif choice == 1:
            reducer = rng.choice([mean, sum, max, min, std])
            over = [d for d in data[a].dims if d != field_axis]
            steps.append(mutate(r=reducer(col(a), over=str(rng.choice(over)), na_rm=bool(rng.random() < 0.5))))
        elif choice == 2:
            steps.append(filter(col("key") > int(rng.integers(0, 3))))
        elif choice == 3:
            if sizes[axis]:
                wanted = sorted(set(int(p) for p in rng.integers(1, sizes[axis] + 1, 2)))
                steps.append(slice(**{axis: wanted if rng.random() < 0.7 else [-wanted[0]]}))
        elif choice == 4:
            steps.append(select(*sorted(set([a, "key"]))))
        elif choice == 5:
            steps.append(group_by(axis) if axis != axes[0] else group_by(col("key")))
            steps.append(summarise(m=mean(col(a), na_rm=bool(rng.random() < 0.5)), k=n()))
            break
        elif choice == 6:
            steps.append(group_by(col("key")))
            steps.append(mutate(c=col(a) - mean(col(a))))
        elif choice == 7 and field_axis:
            steps.append(mutate(x=col("x") * 2))
        elif choice == 8 and field_axis:
            steps.append(select(*rng.choice(FIELDS, 2, replace=False)))
        elif choice == 9:
            steps.append(summarise(s=sum(col(a), na_rm=True, min_count=2), k=n()))
            break
        elif choice in (10, 11) and axis in data.coords:
            other = _other(axis, data[axis].values, rng)
            if choice == 10:
                steps.append(JOINS[int(rng.integers(3))](other.to_dataset()))
            else:
                steps.append(mutate(o=col(a) + other))
    return steps


def outcome(data, field_axis, steps, backend):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            frame = tidy(data, field_axis=field_axis, backend=backend)
            for step in steps:
                frame = frame >> step
            result = frame.collect()
        except Exception as exc:  # noqa: BLE001 - errors are part of the behaviour compared
            return ("error", type(exc).__name__, str(exc), _warned(caught))
    return ("frame", repr(frame), result, _warned(caught))


def _warned(caught):
    return sorted({(w.category.__name__, str(w.message)) for w in caught})


@pytest.mark.parametrize("seed", SEEDS)
def test_random_pipelines_agree_across_backends(seed):
    rng = np.random.default_rng(seed)
    data, axes, sizes, field_axis = random_dataset(rng)
    steps = random_pipeline(rng, data, axes, sizes, field_axis)
    on_xarray = outcome(data, field_axis, steps, "xarray")
    on_numpy = outcome(data, field_axis, steps, "numpy")
    assert on_xarray[0] == on_numpy[0], (on_xarray[:3], on_numpy[:3])
    assert on_xarray[-1] == on_numpy[-1]  # the same warnings
    if on_xarray[0] == "error":
        assert on_xarray[1:3] == on_numpy[1:3]
        return
    assert on_xarray[1] == on_numpy[1]
    left, right = on_xarray[2], on_numpy[2]
    xr.testing.assert_identical(left, right)
    assert {v: left[v].dtype for v in left.variables} == {v: right[v].dtype for v in right.variables}
    assert pickle.dumps(left.attrs) == pickle.dumps(right.attrs)


def test_most_random_pipelines_run_without_error():
    # The comparison above means little if nearly every pipeline stops at an error.
    finished = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        data, axes, sizes, field_axis = random_dataset(rng)
        steps = random_pipeline(rng, data, axes, sizes, field_axis)
        finished += outcome(data, field_axis, steps, "xarray")[0] == "frame"
    assert finished >= len(SEEDS) // 2, finished
