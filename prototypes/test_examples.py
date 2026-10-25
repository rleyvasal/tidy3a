"""Milestone 0: the acceptance examples, computed in plain xarray.

Each test is the xarray program that the tidy3a code in
docs/m0-examples.md would compile to, run on small synthetic data with
answers worked out by hand. The tidy3a spellings are in the comments.
"""

import numpy as np
import pytest
import xarray as xr

EXACT = {"arithmetic_join": "exact"}


# ── Example 1: point cloud ──────────────────────────────────────────────────
# Sensor-frame points chosen so that, after a 90° rotation about z and a
# 1 m offset along x, they land at 49 m, 50 m, 51 m, and 3 m from the origin.

R = xr.DataArray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dims=("i", "feature"),
    coords={"i": ["x", "y", "z"], "feature": ["x", "y", "z"]},
)
OFFSET = xr.DataArray([1.0, 0.0, 0.0], dims="i", coords={"i": ["x", "y", "z"]})


def point_cloud():
    # read_array(..., dims={"point": infer, "feature": 5}, labels=..., field_axis="feature")
    values = np.array(
        [
            [0.0, -48.0, 0.0, 10.0, 1.0],
            [0.0, -49.0, 0.0, 20.0, 2.0],
            [51.0, 1.0, 0.0, 30.0, 1.0],
            [0.0, 1.0, 3.0, 40.0, 2.0],
        ],
        dtype="float32",
    )
    return xr.DataArray(
        values,
        dims=("point", "feature"),
        coords={"feature": ["x", "y", "z", "intensity", "ring"]},
    )


def test_point_cloud_select():
    # cloud >> select(x, y, z)
    xyz = point_cloud().sel(feature=["x", "y", "z"])
    assert xyz.dims == ("point", "feature")
    assert xyz.feature.values.tolist() == ["x", "y", "z"]


def test_point_cloud_transform_distance_filter():
    # cloud
    # >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
    # >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
    # >> filter(distance <= 50)
    cloud = point_cloud()
    cloud = cloud.assign_coords(point=np.arange(1, cloud.sizes["point"] + 1))
    with xr.set_options(**EXACT):
        p = xr.dot(R, cloud.sel(feature=["x", "y", "z"]), dim="feature") + OFFSET
        distance = np.sqrt((p**2).sum(dim="i", skipna=False))
    assert distance.values.tolist() == pytest.approx([49.0, 50.0, 51.0, 3.0])

    ds = xr.Dataset({"values": cloud, "p": p, "distance": distance})
    nearby = ds.isel(point=(ds.distance <= 50).values)
    # Original positions are kept as labels; 50 m is inside the boundary.
    assert nearby.point.values.tolist() == [1, 2, 4]
    assert nearby["values"].sel(feature="intensity").values.tolist() == [10.0, 20.0, 40.0]


def test_point_cloud_mean_intensity_per_ring():
    # nearby >> group_by(ring) >> summarise(mean_intensity = mean(intensity))
    cloud = point_cloud().isel(point=[0, 1, 3])
    ring = cloud.sel(feature="ring", drop=True).rename("ring")
    intensity = cloud.sel(feature="intensity", drop=True)
    out = intensity.groupby(ring).mean(skipna=False)
    assert out.ring.values.tolist() == [1.0, 2.0]
    assert out.values.tolist() == [10.0, 30.0]


def test_point_cloud_vector_write_back():
    # cloud >> mutate([x, y, z] = dot(R, [x, y, z], over="feature"))
    cloud = point_cloud()
    with xr.set_options(**EXACT):
        rotated = xr.dot(R, cloud.sel(feature=["x", "y", "z"]), dim="feature")
    # Map axis i onto feature by label, then write back.
    rotated = rotated.rename(i="feature").transpose("point", "feature")
    out = cloud.copy()
    out.loc[{"feature": ["x", "y", "z"]}] = rotated
    assert out.sel(feature="x").values.tolist() == [48.0, 49.0, -1.0, -1.0]
    assert out.sel(feature="intensity").values.tolist() == [10.0, 20.0, 30.0, 40.0]


# ── Example 2: sensor time series ───────────────────────────────────────────


def sensors():
    # read_array(..., dims={"time": infer, "sensor": 2, "feature": 3}, labels=..., field_axis="feature")
    values = np.array(
        [
            [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[3.0, 2.0, 0.0], [30.0, 2.0, 0.0]],
            [[5.0, 4.0, 1.0], [50.0, 4.0, 1.0]],
            [[7.0, 6.0, 1.0], [70.0, 6.0, 1.0]],
        ]
    )
    return xr.DataArray(
        values,
        dims=("time", "sensor", "feature"),
        coords={
            "time": np.arange(1, 5),
            "sensor": ["front", "rear"],
            "feature": ["x", "y", "z"],
        },
    )


def test_sensor_mean_per_sensor():
    # data >> group_by(sensor) >> summarise(mean_x = mean(x))
    out = sensors().sel(feature="x").mean(dim="time", skipna=False)
    assert out.dims == ("sensor",) and out.values.tolist() == [4.0, 40.0]


def test_sensor_summary_keeps_field_axis():
    # data >> group_by(sensor) >> summarise(m = mean(values))
    out = sensors().mean(dim="time", skipna=False)
    assert out.dims == ("sensor", "feature")
    assert out.sel(sensor="front").values.tolist() == [4.0, 3.0, 0.5]


def test_sensor_filter_on_two_axes_keeps_rectangle():
    # data >> filter(sensor == "front", time > 2)
    data = sensors()
    out = data.isel(sensor=(data.sensor == "front").values, time=(data.time > 2).values)
    assert dict(out.sizes) == {"time": 2, "sensor": 1, "feature": 3}


def test_sensor_multi_axis_predicate_reduced_with_any():
    # data >> filter(any(x > 20, over="sensor"))
    data = sensors()
    keep = (data.sel(feature="x") > 20).any(dim="sensor")
    assert keep.dims == ("time",)
    assert data.isel(time=keep.values).time.values.tolist() == [2, 3, 4]


# ── Example 3: geospatial raster ────────────────────────────────────────────
# A 3 × 2 pixel scene with four spectral bands at two dates, the kind of
# data cube stars handles. Reflectances are chosen so NDVI is exact.


def scene():
    # read_array(..., dims={"x": 3, "y": 2, "band": 4, "time": 2}, labels=..., field_axis="band")
    red = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.2]])
    nir = np.array([[0.3, 0.2], [0.3, 0.7], [0.6, 0.2]])
    blue = np.full((3, 2), 0.05)
    green = np.full((3, 2), 0.08)
    date1 = np.stack([blue, green, red, nir], axis=-1)
    date2 = date1 * np.array([1.0, 1.0, 1.0, 0.5])  # nir halves at the second date
    return xr.DataArray(
        np.stack([date1, date2], axis=-1),
        dims=("x", "y", "band", "time"),
        coords={
            "x": [500000.0, 500030.0, 500060.0],
            "y": [4100030.0, 4100000.0],
            "band": ["blue", "green", "red", "nir"],
            "time": np.array(["2026-06-01", "2026-07-01"], dtype="datetime64[ns]"),
        },
    )


def ndvi_of(cube):
    nir = cube.sel(band="nir", drop=True)
    red = cube.sel(band="red", drop=True)
    return (nir - red) / (nir + red)


def test_geospatial_ndvi():
    # scene >> mutate(ndvi = (nir - red) / (nir + red))
    ndvi = ndvi_of(scene())
    assert ndvi.dims == ("x", "y", "time")
    np.testing.assert_allclose(
        ndvi.sel(time="2026-06-01").values, [[0.5, 0.0], [0.0, 0.75], [0.5, 0.0]]
    )


def test_geospatial_crop_like_stars_filter():
    # scene >> filter(x > 500010, y > 4100010)       (stars: filter(x > ..., y > ...))
    cube = scene()
    out = cube.isel(x=(cube.x > 500010).values, y=(cube.y > 4100010).values)
    assert dict(out.sizes) == {"x": 2, "y": 1, "band": 4, "time": 2}


def test_geospatial_mean_over_time_per_pixel():
    # scene >> mutate(ndvi = …) >> group_by(x, y) >> summarise(mean_ndvi = mean(ndvi))
    out = ndvi_of(scene()).mean(dim="time", skipna=False)
    assert out.dims == ("x", "y")
    # Pixel (500000, 4100030): NDVI 0.5 then (0.15 - 0.1) / 0.25 = 0.2.
    assert float(out.isel(x=0, y=0)) == pytest.approx((0.5 + 0.2) / 2)


def test_geospatial_mean_per_band_like_st_apply():
    # scene >> group_by(band) >> summarise(m = mean(values))   (stars: st_apply(x, "band", mean))
    # With field_axis="band" the band axis is kept without group_by: summarise(m = mean(values)).
    out = scene().mean(dim=["x", "y", "time"], skipna=False)
    assert out.dims == ("band",)
    assert float(out.sel(band="blue")) == pytest.approx(0.05)


def test_geospatial_ndvi_bins():
    # scene >> mutate(ndvi = …) >> group_by(bins(ndvi, [-1, 0.25, 1])) >> summarise(n = n())
    ndvi = ndvi_of(scene()).sel(time="2026-06-01").stack(pixel=("x", "y"))
    counts = ndvi.groupby_bins(ndvi, [-1, 0.25, 1], right=False).count()
    assert counts.values.tolist() == [3, 3]
