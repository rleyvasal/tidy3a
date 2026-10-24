"""The complete point-cloud workflow (plan §7) on a synthetic binary file."""

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


def test_cloud_workflow_from_a_raw_file(tmp_path):
    # Interleaved float32 records x y z intensity ring, as in nuScenes LiDAR files.
    records = np.array(
        [[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="<f4",
    )
    path = tmp_path / "sweep.pcd.bin"
    records.tofile(path)

    cloud = read_array(
        path,
        value_type="float32",
        dims={"point": infer, "feature": 5},
        labels={"feature": ["x", "y", "z", "intensity", "ring"]},
        field_axis="feature",
    )
    assert "inferred: point = 80 bytes / 4 bytes / 5 = 4" in repr(cloud)

    xyz = cloud >> select(x, y, z)
    assert xyz.collect()["values"].feature.values.tolist() == AXES

    R = xr.DataArray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),
                     coords={"i": AXES, "feature": AXES})
    offset = xr.DataArray([1.0, 0, 0], dims="i", coords={"i": AXES})
    nearby = (
        cloud
        >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
        >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
        >> filter(distance <= 50)
    )
    assert (nearby >> pull("point")).tolist() == [1, 2, 4]
    assert (nearby >> pull(intensity)).tolist() == [10.0, 20.0, 40.0]
    np.testing.assert_allclose(nearby >> pull(distance), [49.0, 50.0, 3.0])

    per_ring = (nearby >> group_by(ring) >> summarise(points = n(), mean_intensity = mean(intensity))).collect()
    assert per_ring.ring.values.tolist() == [1.0, 2.0]
    assert per_ring["points"].values.tolist() == [1, 2]
    assert per_ring["mean_intensity"].values.tolist() == [10.0, 30.0]
