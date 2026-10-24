"""Time the NumPy-only core prototype against tidy3A and plain NumPy.

Run by hand (pytest does not collect it): ``python prototypes/bench_numpy_core.py``.
Every workload uses the same float32 data and float32 rotation in all three,
and each result is checked against NumPy before it is timed.
"""

import sys
import time

import numpy as np
import xarray as xr

import tidy3a as ta
from numpy_core import Dataset, Variable, apply, dot, drop_axis, group_reduce, labelled, reduce, sel
from tidy3a import col, filter, group_by, infer, mean, mutate, read_array, sqrt, summarise
from tidy3a import sum as tsum

FIELDS = ["x", "y", "z", "intensity", "ring"]
XYZ = ["x", "y", "z"]
ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
OFFSET = np.array([1.0, 2.0, 0.5], dtype="float32")


def best(run, repeats=5):
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    values = np.empty((n, 5), dtype="float32")
    values[:, :3] = rng.uniform(-80, 80, (n, 3))
    values[:, 3] = rng.uniform(0, 255, n)
    values[:, 4] = rng.integers(0, 32, n)
    return values


def run_cloud(n):
    values = cloud(n)

    # Prototype core
    R = labelled(ROTATION, ("i", "feature"), {"i": XYZ, "feature": XYZ})
    t = labelled(OFFSET, ("i",), {"i": XYZ})
    data = Dataset({"values": Variable(values, ("point", "feature"))}, {"feature": FIELDS})

    def core_filter():
        p = apply(np.add, dot(R, sel(data["values"], feature=XYZ), over="feature"), t)
        distance = apply(np.sqrt, reduce(apply(np.square, p), "sum", over="i"))
        kept = data.assign(p=p, distance=distance)
        return kept.isel(op="filter", point=apply(np.less_equal, kept["distance"], 50.0).values)

    def field(d, name):
        return drop_axis(sel(d["values"], feature=[name]), "feature")

    def core_group():
        return group_reduce(field(data, "intensity"), field(data, "ring"), "mean", "ring")

    # tidy3A
    frame = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, field_axis="feature")
    Rx = xr.DataArray(ROTATION, dims=("i", "feature"), coords={"i": XYZ, "feature": XYZ})
    tx = xr.DataArray(OFFSET, dims="i", coords={"i": XYZ})
    x, y, z = ta.fields("x", "y", "z")

    def tidy3a_filter():
        return (frame >> mutate(p=ta.dot(Rx, [x, y, z], over="feature") + tx)
                >> mutate(distance=sqrt(tsum(col("p") ** 2, over="i"))) >> filter(col("distance") <= 50))

    def tidy3a_group():
        return frame >> group_by(col("ring")) >> summarise(m=mean(col("intensity")))

    # Plain NumPy
    def numpy_filter():
        p = values[:, :3] @ ROTATION.T + OFFSET
        distance = np.sqrt((p ** 2).sum(axis=1))
        keep = distance <= 50
        return values[keep], p[keep], distance[keep]

    def numpy_group():
        rings, codes = np.unique(values[:, 4], return_inverse=True)
        return np.bincount(codes, values[:, 3]) / np.bincount(codes)

    # Results agree before anything is timed.
    kept_values, _, kept_distance = numpy_filter()
    np.testing.assert_allclose(core_filter()["distance"].values, kept_distance, rtol=1e-6)
    np.testing.assert_allclose(tidy3a_filter().collect()["distance"].values, kept_distance, rtol=1e-6)
    assert core_filter()["values"].values.shape == kept_values.shape
    np.testing.assert_array_equal(core_group().values, tidy3a_group().collect()["m"].values)

    rows = [
        ("§7 rotate + 50 m filter", best(core_filter), best(tidy3a_filter), best(numpy_filter)),
        ("mean intensity per ring", best(core_group), best(tidy3a_group), best(numpy_group)),
    ]
    return rows


def run_grid():
    rng = np.random.default_rng(1)
    values = rng.normal(280, 10, (2920, 50, 100)).astype("float32")
    da = xr.DataArray(values, dims=("time", "lat", "lon"), name="air")
    frame = ta.tidy(da)
    air = labelled(values, ("time", "lat", "lon"))

    def core_mean():
        return reduce(air, "mean", over="time")

    def core_anomaly():
        return apply(np.subtract, air, reduce(air, "mean", over="time"))

    def tidy3a_mean():
        return frame >> group_by("lat", "lon") >> summarise(m=mean(col("air")))

    def tidy3a_anomaly():
        return frame >> group_by("lat", "lon") >> mutate(anomaly=col("air") - mean(col("air")))

    np.testing.assert_array_equal(core_mean().values, values.mean(axis=0))
    np.testing.assert_array_equal(tidy3a_mean().collect()["m"].values, values.mean(axis=0))
    return [
        ("grid mean over time", best(core_mean, 3), best(tidy3a_mean, 3), best(lambda: values.mean(axis=0), 3)),
        ("grid anomaly", best(core_anomaly, 3), best(tidy3a_anomaly, 3), best(lambda: values - values.mean(axis=0), 3)),
    ]


def main():
    print(f"NumPy {np.__version__}, xarray {xr.__version__}, tidy3A {ta.__version__}, Python {sys.version.split()[0]}")
    print(f"{'workload':40s} {'prototype':>10s} {'tidy3A':>10s} {'NumPy':>10s}   (ms, best of runs)")
    for n in (34_688, 10_000_000):
        for name, core, tidy, plain in run_cloud(n):
            print(f"{name + f' ({n:,} points)':40s} {core:10.2f} {tidy:10.2f} {plain:10.2f}")
    for name, core, tidy, plain in run_grid():
        print(f"{name + ' (14.6M cells)':40s} {core:10.2f} {tidy:10.2f} {plain:10.2f}")


if __name__ == "__main__":
    main()
