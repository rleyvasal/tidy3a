"""Measure grouping preparation separately from reusable grouped reductions.

Run: python prototypes/bench_numpy_pipeline.py --points 1000000
Input snapshots and output conversion are timed separately. NumPy-only
factorization is used throughout the experimental pipeline.
"""

import argparse
import platform
import time

import numpy as np
import xarray as xr

import numpy_pipeline as proto
import tidy3a as ta


def best(run, repeats):
    run()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        timings.append((time.perf_counter() - start) * 1000)
    return min(timings)


def workload(n, high_cardinality):
    rng = np.random.default_rng(8)
    values = rng.normal(size=n).astype("float32")
    if high_cardinality:
        keys = np.arange(n, dtype="int64")
        # Ten percent of timestamps repeat the preceding point's timestamp.
        duplicate = np.arange(1, n, 10)
        keys[duplicate] -= 1
        keys = keys.astype("datetime64[ns]")
    else:
        keys = rng.integers(0, 32, n, dtype="int32")
    source = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    return source, keys, values


def measure(n, high_cardinality, repeats):
    source, keys, values = workload(n, high_cardinality)
    data = proto.tidy(source)
    grouping = ta.group_by("key")
    stats = ta.summarise(total=ta.sum(ta.col("v")), mean=ta.mean(ta.col("v")), count=ta.n())
    grouped = data >> grouping
    output = grouped >> stats
    cache = grouped._group_cache
    actual = output.collect()
    labels, codes = np.unique(keys, return_inverse=True)
    np.testing.assert_array_equal(actual["count"], np.bincount(codes))
    np.testing.assert_array_equal(actual.key, labels)
    # Sample independent per-group NumPy calculations without a Python loop
    # over every timestamp in the high-cardinality workload.
    for group in np.unique([0, len(labels) // 2, len(labels) - 1]):
        part = values[codes == group]
        np.testing.assert_array_equal(actual.total.values[group], part.sum())
        np.testing.assert_array_equal(actual["mean"].values[group], part.mean())

    def warm():
        result = grouped >> stats
        assert grouped._group_cache is cache
        return result

    timings = {
        "snapshot": best(lambda: proto.tidy(source), repeats),
        "prepare": best(lambda: data >> grouping, repeats),
        "cold_stats": best(lambda: data >> grouping >> stats, repeats),
        "warm_stats": best(warm, repeats),
        "to_xarray": best(output.collect, repeats),
    }
    print(f"{'mostly unique times' if high_cardinality else '32 groups':21s} {len(labels):10,d} "
          + " ".join(f"{timings[name]:10.2f}" for name in timings)
          + f" {cache.nbytes / 2**20:10.2f}")
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.points < 2 or args.repeats < 1:
        parser.error("--points must be at least 2 and --repeats at least 1")
    print(f"{platform.machine()}, Python {platform.python_version()}, NumPy {np.__version__}, "
          f"xarray {xr.__version__}; {args.points:,} points; best of {args.repeats}")
    print(f"{'workload':21s} {'groups':>10s} {'snapshot':>10s} {'prepare':>10s} {'cold stats':>10s} "
          f"{'warm stats':>10s} {'to_xarray':>10s} {'cache MiB':>10s}")
    print("Timings in milliseconds; warm stats still read values and produce one result per group.")
    for high_cardinality in (False, True):
        measure(args.points, high_cardinality, args.repeats)


if __name__ == "__main__":
    main()
