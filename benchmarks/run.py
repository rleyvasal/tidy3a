"""Reference workloads for comparing tidy3A's backends (plan milestone 9, step 5).

Run through ``scripts/bench.sh``, which builds the pinned environment. Each
workload runs on both backends; their results are checked to be identical
before anything is timed, and the backends alternate while timing so that a
drift in machine speed affects both. The environment (package versions,
Python, CPU, git commit) is recorded with the results.

Workloads: the plan's §7 pipeline and grouped summaries on synthetic clouds
of 34,688 and 10M points; the §7 pipeline on the first nuScenes Mini LIDAR
sweep (``NUSCENES_DATAROOT``); and the sample-data pipelines of
``examples/sample_data.py`` (``TIDY3A_SAMPLES``). Data that is opened lazily
is loaded into memory first, so both backends hold it.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

import tidy3a as ta
from tidy3a import (bins, col, dot, fields, filter, formula, group_by, infer, mean, mutate, n, pack, read_array,
                    sqrt, sum, summarise, tidy)

REPO = Path(__file__).resolve().parents[1]
BACKENDS = ("xarray", "numpy")
x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
XYZ = ["x", "y", "z"]
FIELDS = ["x", "y", "z", "intensity", "ring"]


# ── Workloads ───────────────────────────────────────────────────────────────


def cloud_workloads(values: np.ndarray, rotation: xr.DataArray, offset: xr.DataArray, label: str):
    """The §7 pipeline and grouped summaries on one cloud; each returns a function of the backend."""

    def cloud(backend):
        return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS},
                          field_axis="feature", backend=backend)

    def section_7(frame):
        return (frame >> mutate(p=dot(rotation, [x, y, z], over="feature") + offset)
                >> mutate(distance=sqrt(sum(col("p") ** 2, over="i"))) >> filter(distance <= 50))

    def prepared(run):
        def make(backend):
            frame = cloud(backend)
            return lambda: run(frame)
        return make

    return {
        f"{label}: §7 rotate, distance, 50 m filter": prepared(section_7),
        f"{label}: §7 with the formula": prepared(lambda f: f >> mutate(p=dot(rotation, [x, y, z], over="feature")
                                                                       + offset)
                                                  >> mutate(distance=formula(r"\sqrt{\sum_i p_i^2}"))),
        f"{label}: mean intensity per ring": prepared(lambda f: f >> group_by(ring)
                                                      >> summarise(points=n(), m=mean(intensity))),
        f"{label}: intensity centred per ring": prepared(lambda f: f >> group_by(ring)
                                                         >> mutate(c=intensity - mean(intensity))),
    }


def synthetic(size: int) -> dict:
    rng = np.random.default_rng(0)
    values = np.empty((size, 5), "float32")
    values[:, :3] = rng.uniform(-80, 80, (size, 3))
    values[:, 3] = rng.uniform(0, 255, size)
    values[:, 4] = rng.integers(0, 32, size)
    rotation = xr.DataArray(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float32"), dims=("i", "feature"),
                            coords={"i": XYZ, "feature": XYZ})
    offset = xr.DataArray(np.array([1, 2, 0.5], "float32"), dims="i", coords={"i": XYZ})
    return cloud_workloads(values, rotation, offset, f"synthetic {size:,} points")


def nuscenes(root: Path) -> dict:
    sys.path.insert(0, str(REPO / "examples"))
    from nuscenes_sweep import first_lidar_sweep, rotation_from_wxyz

    path, calibration, _ = first_lidar_sweep(root)
    rotation = xr.DataArray(rotation_from_wxyz(calibration["rotation"]), dims=("i", "feature"),
                            coords={"i": XYZ, "feature": XYZ})
    offset = xr.DataArray(calibration["translation"], dims="i", coords={"i": XYZ})
    values = np.fromfile(path, dtype="<f4").reshape(-1, 5)
    work = cloud_workloads(values, rotation, offset, "nuScenes sweep")
    work["nuScenes sweep: read the .bin file"] = lambda backend: lambda: read_array(
        path, value_type="float32", dims={"point": infer, "feature": 5}, labels={"feature": FIELDS},
        field_axis="feature", backend=backend)
    return work


def samples(root: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # eraint_uvz.nc has non-standard _FillValue attributes
        air = xr.open_dataset(root / "air_temperature.nc").load()
        wind = xr.open_dataset(root / "eraint_uvz.nc").load()
    import rioxarray

    with rioxarray.open_rasterio(root / "L7_ETMs.tif") as opened:
        raster = opened.load()
    raster = raster.drop_vars([c for c in raster.coords if c not in raster.dims])  # spatial_ref
    raster = raster.assign_coords(band=["blue", "green", "red", "nir", "swir1", "swir2"])

    a, lat, lon, time_ = fields("air", "lat", "lon", "time")
    u, v, level, latitude, speed = fields("u", "v", "level", "latitude", "speed")
    red, nir, px = fields("red", "nir", "x")
    turn = xr.DataArray([[0.0, -1.0], [1.0, 0.0]], dims=("rotated", "component"),
                        coords={"rotated": ["u", "v"], "component": ["u", "v"]})

    def on(data, run, **options):
        return lambda backend: (lambda frame: lambda: run(frame))(tidy(data, backend=backend, **options))

    return {
        "air temperature: filter a lat/lon window": on(air, lambda f: f >> filter(lat >= 30, lat <= 60,
                                                                                    lon >= 235, lon <= 290)),
        "air temperature: mean per latitude": on(air, lambda f: f >> group_by(lat) >> summarise(m=mean(a))),
        "air temperature: anomaly per grid point": on(air, lambda f: f >> group_by(lat, lon)
                                                      >> mutate(anomaly=a - mean(a))),
        "air temperature: 30-day windows per grid point": on(air, lambda f: f >> group_by(bins(time_, "30 days"),
                                                                                          lat, lon)
                                                             >> summarise(m=mean(a))),
        "air temperature: formula air - 273.15": on(air, lambda f: f >> mutate(c=formula("air - 273.15"))),
        "ERA-Interim: wind speed formula": on(wind, lambda f: f >> mutate(speed=formula(r"\sqrt{u^2 + v^2}"))),
        "ERA-Interim: rotate packed wind": on(wind, lambda f: f >> mutate(r=dot(turn, pack([u, v], along="component"),
                                                                                over="component"))),
        "ERA-Interim: northern mean speed per level": on(wind, lambda f: f >> mutate(speed=formula(r"\sqrt{u^2 + v^2}"))
                                                         >> filter(latitude >= 0) >> group_by(level)
                                                         >> summarise(mean_speed=mean(speed))),
        "Landsat: NDVI": on(raster, lambda f: f >> mutate(ndvi=(nir * 1.0 - red) / (nir * 1.0 + red)),
                            field_axis="band"),
        "Landsat: mean per band": on(raster, lambda f: f >> summarise(m=mean(col("values"))), field_axis="band"),
        "Landsat: crop on x": on(raster, lambda f: f >> filter(px > 289000, px < 291000), field_axis="band"),
    }


# ── Measuring ───────────────────────────────────────────────────────────────


def same_result(results: dict) -> None:
    """Both backends must give identical data, types included, before timing means anything."""
    first, second = (results[b].collect() for b in BACKENDS)
    xr.testing.assert_identical(first, second)
    types = [{name: data[name].dtype for name in data.variables} for data in (first, second)]
    if types[0] != types[1]:
        raise AssertionError(f"types differ: {types}")


def time_both(make: Callable, minimum: float = 0.4) -> dict:
    """Median and best milliseconds per backend, alternating backends between rounds."""
    runs = {b: make(b) for b in BACKENDS}
    same_result({b: run() for b, run in runs.items()})
    once = {}
    for b, run in runs.items():
        start = time.perf_counter()
        run()
        once[b] = time.perf_counter() - start
    repeats = int(min(25, max(5, minimum / max(min(once.values()), 1e-6))))
    times: dict[str, list[float]] = {b: [] for b in BACKENDS}
    for _ in range(repeats):
        for b, run in runs.items():
            start = time.perf_counter()
            run()
            times[b].append((time.perf_counter() - start) * 1e3)
    return {b: {"median_ms": statistics.median(t), "best_ms": min(t), "repeats": repeats} for b, t in times.items()}


def environment() -> dict:
    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    cpu = run("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor()
    packages = sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "packages": dict(packages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="results file (default: benchmarks/results/<date>-<commit>.json)")
    parser.add_argument("--quick", action="store_true", help="skip the 10M-point workloads")
    args = parser.parse_args()

    workloads = dict(synthetic(34_688))
    if not args.quick:
        workloads.update(synthetic(10_000_000))
    if os.environ.get("NUSCENES_DATAROOT"):
        workloads.update(nuscenes(Path(os.environ["NUSCENES_DATAROOT"])))
    if os.environ.get("TIDY3A_SAMPLES"):
        workloads.update(samples(Path(os.environ["TIDY3A_SAMPLES"])))

    env = environment()
    print(f"tidy3A {ta.__version__}, NumPy {np.__version__}, xarray {xr.__version__}, Python {env['python']}, "
          f"{env['cpu']}; commit {(env['git_commit'] or '?')[:8]}{' (uncommitted changes)' if env['git_dirty'] else ''}")
    print(f"{'workload':58s} {'xarray':>9s} {'numpy':>9s} {'numpy/xarray':>13s}   (median ms)")
    results = {}
    for name, make in workloads.items():
        results[name] = timing = time_both(make)
        a, b = timing["xarray"]["median_ms"], timing["numpy"]["median_ms"]
        print(f"{name:58s} {a:9.2f} {b:9.2f} {a / b:12.2f}x")

    out = args.out or REPO / "benchmarks" / "results" / (
        f"{env['date'][:10]}-{(env['git_commit'] or 'unknown')[:8]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"environment": env, "results": results}, indent=1))
    print(f"\nresults and environment written to {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


if __name__ == "__main__":
    main()
