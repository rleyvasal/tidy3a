# NumPy pipeline benchmark

Measured locally on 2026-10-24: arm64, Python 3.12.12, NumPy 2.5.4,
xarray 2026.9.0. One million observations; milliseconds, best of three warm
timing attempts. Results vary with machine state.

```bash
.venv/bin/python prototypes/bench_numpy_pipeline.py --points 1000000
```

| Workload | Groups | Input snapshot | Prepare groups | Cold statistics | Warm statistics | Convert output to xarray | Group arrays, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Integer key | 32 | 0.19 | 27.39 | 39.17 | 10.56 | 0.07 | 15.26 |
| Mostly unique nanosecond timestamps | 900,000 | 0.32 | 21.42 | 30.44 | 9.84 | 0.51 | 35.10 |

Statistics compute a sum, mean, and count. Cold statistics build grouping state
and compute those results; warm statistics reuse the state. Input snapshot and
output conversion costs are separate. The memory column counts the cached
NumPy grouping arrays, including compatibility labels; it excludes input,
output, temporary arrays, and small Python metadata objects.

The high-cardinality input repeats every tenth timestamp once, leaving group
sizes of one or two. NumPy factorization is used; no pandas grouping accelerator
is enabled. There are no Python lists per group or Python reductions per group:
groups of equal size are processed in batches. Warm computation remains O(N + G).

Before timing, counts and coordinate labels are checked against NumPy for every
group. Sums and means for sampled groups are checked against independent NumPy
calculations. The tests additionally compare exact per-group results over
different dtypes, axis layouts, missing values, and group sizes.

These figures measure the experimental pipeline and are not the earlier
direct-core benchmark. The older `bench_numpy_core.py` uses optional pandas
factorization for its direct-core grouping comparison and includes different
workloads and timing boundaries.
