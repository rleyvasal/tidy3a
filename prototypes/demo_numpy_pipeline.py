"""Run from the repository root: python prototypes/demo_numpy_pipeline.py.

The point-cloud example of plan §7 on the experimental NumPy backend: rotate
the points, keep those within 50 metres, centre intensity within each ring,
and summarise intensity per ring. The same pipeline gives the same results on
the default xarray backend; only ``backend="numpy"`` differs.
"""

import numpy as np
import xarray as xr

from tidy3a import dot, fields, filter, formula, group_by, infer, mean, mutate, n, pull, read_array, summarise


def main():
    records = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2],
                        [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float32")
    cloud = read_array(records, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature",
                       backend="numpy")
    x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
    xyz = ["x", "y", "z"]
    rotation = xr.DataArray(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="float32"),
                            dims=("i", "feature"), coords={"i": xyz, "feature": xyz})
    offset = xr.DataArray(np.array([1, 0, 0], dtype="float32"), dims="i", coords={"i": xyz})

    nearby = (cloud
              >> mutate(p=dot(rotation, [x, y, z], over="feature") + offset)
              >> mutate(distance=formula(r"\sqrt{\sum_i p_i^2}"))
              >> filter(distance <= 50))
    centered = nearby >> group_by(ring) >> mutate(centered=intensity - mean(intensity), group_size=n())
    per_ring = centered >> summarise(points=n(), mean_intensity=mean(intensity))

    np.testing.assert_array_equal(nearby >> pull("point"), [1, 2, 4])
    np.testing.assert_allclose(nearby >> pull(distance), [49, 50, 3])
    np.testing.assert_allclose(per_ring.collect()["mean_intensity"], [10, 30])
    np.testing.assert_allclose(centered >> pull("centered"), [0, -10, 10])
    print("Original point positions:", nearby >> pull("point"))
    print("Distances:", nearby >> pull(distance))
    print("Intensity centred within each ring:", centered >> pull("centered"))
    print(per_ring)


if __name__ == "__main__":
    main()
