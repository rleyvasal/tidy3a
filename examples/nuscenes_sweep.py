"""nuScenes Mini: one LiDAR sweep in the vehicle frame, points within 50 m (plan §7).

    python examples/nuscenes_sweep.py /path/to/nuscenes-mini

The nuScenes tables are read directly as JSON to find a sweep and its
calibration; the nuScenes SDK is not needed. tidy3A does the numerical
work, and the script checks it against an independent NumPy calculation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tidy3a import (
    dot,
    fields,
    filter,
    formula,
    group_by,
    infer,
    mean,
    mutate,
    n,
    pull,
    read_array,
    select,
    summarise,
)

x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
AXES = ["x", "y", "z"]


def table(root: Path, name: str) -> list[dict]:
    return json.loads((root / "v1.0-mini" / f"{name}.json").read_text())


def first_lidar_sweep(root: Path) -> tuple[Path, dict, dict]:
    """The earliest key-frame LIDAR_TOP sweep, its calibrated sensor, and the sample data row."""
    sensors = {row["token"]: row for row in table(root, "sensor")}
    calibrations = {row["token"]: row for row in table(root, "calibrated_sensor")}
    lidar = [
        row for row in table(root, "sample_data")
        if row["is_key_frame"]
        and sensors[calibrations[row["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "LIDAR_TOP"
    ]
    sweep = min(lidar, key=lambda row: row["timestamp"])
    return root / sweep["filename"], calibrations[sweep["calibrated_sensor_token"]], sweep


def rotation_from_wxyz(q) -> np.ndarray:
    """A rotation matrix from a nuScenes quaternion, ordered w, x, y, z."""
    w, a, b, c = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
        [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
        [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)],
    ])


def main(root: Path) -> None:
    path, calibration, sweep = first_lidar_sweep(root)
    print(f"sweep: {sweep['filename']}  ({path.stat().st_size:,} bytes)")
    print(f"calibration: rotation (w, x, y, z) {calibration['rotation']}, translation {calibration['translation']}\n")

    # B. Import through the generic reader: five float32 fields per point.
    cloud = read_array(
        path,
        value_type="float32",
        dims={"point": infer, "feature": 5},
        labels={"feature": ["x", "y", "z", "intensity", "ring"]},
        field_axis="feature",
    )
    print(cloud, "\n")
    print(cloud >> select(x, y, z), "\n")

    # D. A rotation matrix and offset from the calibration.
    R = xr.DataArray(rotation_from_wxyz(calibration["rotation"]), dims=("i", "feature"),
                     coords={"i": AXES, "feature": AXES})
    offset = xr.DataArray(calibration["translation"], dims="i", coords={"i": AXES})

    # E. Transform to the vehicle frame, measure distance, keep points within 50 m.
    nearby = (
        cloud
        >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
        >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
        >> filter(distance <= 50)
    )
    print(nearby, "\n")

    per_ring = nearby >> group_by(ring) >> summarise(points = n(), mean_intensity = mean(intensity))
    rings = per_ring.collect()
    print(f"rings: {rings.sizes['ring']}  (labels {rings.ring.values.min():g} … {rings.ring.values.max():g})")
    print(f"points kept: {int(rings['points'].sum()):,} of {cloud.collect().sizes['point']:,}\n")

    check(path, calibration, R, nearby, rings)


def check(path: Path, calibration: dict, R: xr.DataArray, nearby, rings) -> None:
    """The same calculation in plain NumPy, written independently of tidy3A."""
    points = np.fromfile(path, dtype="<f4").reshape(-1, 5)  # as the nuScenes SDK reads it
    matrix = rotation_from_wxyz(calibration["rotation"])
    vehicle = points[:, :3].astype(float) @ matrix.T + np.asarray(calibration["translation"])
    dist = np.linalg.norm(vehicle, axis=1)
    keep = dist <= 50

    checks = {
        "kept positions": np.array_equal(nearby >> pull("point"), np.flatnonzero(keep) + 1),
        "distances": np.allclose(nearby >> pull(distance), dist[keep], rtol=1e-6),
        "intensities": np.array_equal(nearby >> pull(intensity), points[keep, 3]),
        "points per ring": np.array_equal(
            rings["points"].values, np.unique(points[keep, 4], return_counts=True)[1]),
    }
    try:
        from pyquaternion import Quaternion  # the nuScenes SDK's quaternion library
    except ImportError:
        print("pyquaternion not installed: rotation convention not cross-checked")
    else:
        checks["rotation matches pyquaternion"] = np.allclose(
            Quaternion(calibration["rotation"]).rotation_matrix, R.values)
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit("tidy3A and the independent calculation disagree")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/nuscenes_sweep.py /path/to/nuscenes-mini")
    main(Path(sys.argv[1]))
