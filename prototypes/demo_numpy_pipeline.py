"""Run from the repository root: python prototypes/demo_numpy_pipeline.py."""

import numpy as np

from numpy_core import labelled
from numpy_pipeline import read_array
from tidy3a import dot, fields, filter, formula, group_by, infer, mean, mutate, n, pull, summarise


def main():
    records = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2],
                        [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float32")
    cloud = read_array(records, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
    xyz = ["x", "y", "z"]
    rotation = labelled(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="float32"),
                         ("i", "feature"), {"i": xyz, "feature": xyz})
    offset = labelled(np.array([1, 0, 0], dtype="float32"), ("i",), {"i": xyz})

    nearby = (cloud
              >> mutate(p=dot(rotation, [x, y, z], over="feature") + offset)
              >> mutate(distance=formula(r"\sqrt{\sum_i p_i^2}"))
              >> filter(distance <= 50))
    grouped = nearby >> group_by(ring)
    cached = grouped._group_cache
    centered = grouped >> mutate(centered=intensity - mean(intensity), group_size=n())
    assert centered._group_cache is cached
    per_ring = centered >> summarise(points=n(), mean_intensity=mean(intensity))

    np.testing.assert_array_equal(nearby >> pull("point"), [1, 2, 4])
    np.testing.assert_allclose(nearby >> pull(distance), [49, 50, 3])
    np.testing.assert_allclose(per_ring.collect()["mean_intensity"], [10, 30])
    np.testing.assert_allclose(centered >> pull("centered"), [0, -10, 10])
    print("Original point positions:", nearby >> pull("point"))
    print("Distances:", nearby >> pull(distance))
    print("Intensity centred within each ring:", centered >> pull("centered"))
    print("Grouping cache reused:", centered._group_cache is cached)
    print(per_ring.collect())


if __name__ == "__main__":
    main()
