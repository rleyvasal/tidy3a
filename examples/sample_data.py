"""tidy3A on three public sample datasets, each result checked against plain xarray.

    python examples/sample_data.py /path/to/samples

The directory holds:

- air_temperature.nc: NCEP air temperature, time × lat × lon (xarray tutorial data)
- eraint_uvz.nc: ERA-Interim wind u, v and height z, month × level × latitude × longitude
- L7_ETMs.tif: a Landsat 7 scene, band × y × x (the R stars package's example)

Reading NetCDF needs scipy; reading GeoTIFF needs rioxarray. Neither is a
tidy3A dependency: xarray opens the files, and tidy() wraps them.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from tidy3a import (
    col,
    dot,
    fields,
    filter,
    formula,
    group_by,
    bins,
    mean,
    mutate,
    pack,
    select,
    sum,
    summarise,
    tidy,
)

RESULTS: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    RESULTS.append((name, bool(ok)))
    print(f"{'ok  ' if ok else 'FAIL'} {name}")


def same(a: xr.DataArray, b: xr.DataArray) -> bool:
    try:
        xr.testing.assert_allclose(a.transpose(*b.dims), b)
        return True
    except AssertionError:
        return False


# ── 1. Air temperature: a time × lat × lon grid ─────────────────────────────


def air_temperature(root: Path) -> None:
    print("\n== air_temperature.nc ==")
    ds = xr.open_dataset(root / "air_temperature.nc")
    air_data = tidy(ds)
    print(air_data, "\n")
    air, lat, lon, time = fields("air", "lat", "lon", "time")

    # Keep a latitude band and a longitude window (observation-axis labels).
    band = air_data >> filter(lat >= 30, lat <= 60, lon >= 235, lon <= 290)
    expected = ds.air.sel(lat=slice(60, 30), lon=slice(235, 290))
    check("filter on lat and lon labels", same(band.collect()["air"], expected))

    # Mean per latitude: group_by keeps lat, summarise reduces time and lon.
    per_lat = air_data >> group_by(lat) >> summarise(m = mean(air))
    check("mean per latitude", same(per_lat.collect()["m"], ds.air.mean(["time", "lon"])))

    # Anomalies: each grid point minus its own mean over time (grouped mutate).
    anomalies = air_data >> group_by(lat, lon) >> mutate(anomaly = air - mean(air))
    check("anomaly per grid point", same(anomalies.collect()["anomaly"], ds.air - ds.air.mean("time")))

    # 30-day windows per grid point, as stars' aggregate(by = "30 days").
    # Grouping reduces every observation axis that is not a key, as dplyr
    # does, so lat and lon are keys here to keep one value per grid point.
    windows = (air_data >> group_by(bins(time, "30 days"), lat, lon) >> summarise(m = mean(air))).collect()
    resampled = ds.air.resample(time="30D").mean()
    check(
        "30-day windows per grid point match xarray resample",
        windows.sizes["time_bin"] == resampled.sizes["time"]
        and np.allclose(windows["m"].transpose("time_bin", "lat", "lon").values, resampled.values),
    )
    overall = (air_data >> group_by(bins(time, "30 days")) >> summarise(m = mean(air))).collect()
    check("30-day windows over the whole grid", np.allclose(overall["m"].values, resampled.mean(["lat", "lon"]).values))

    # A formula: Kelvin to Celsius.
    celsius = air_data >> mutate(c = formula("air - 273.15"))
    check("formula air - 273.15", same(celsius.collect()["c"], ds.air - 273.15))


# ── 2. ERA-Interim wind: separate u and v variables ─────────────────────────


def era_wind(root: Path) -> None:
    print("\n== eraint_uvz.nc ==")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the file's _FillValue attributes are non-standard
        ds = xr.open_dataset(root / "eraint_uvz.nc")
    wind = tidy(ds)
    print(wind, "\n")
    u, v, level, latitude, speed = fields("u", "v", "level", "latitude", "speed")

    speed_data = wind >> mutate(speed = formula(r"\sqrt{u^2 + v^2}"))
    check("wind speed formula", same(speed_data.collect()["speed"], np.hypot(ds.u, ds.v)))

    # Pack u and v into one vector and rotate it 90° counter-clockwise.
    turn = xr.DataArray([[0.0, -1.0], [1.0, 0.0]], dims=("rotated", "component"),
                        coords={"rotated": ["u", "v"], "component": ["u", "v"]})
    rotated = (wind >> mutate(r = dot(turn, pack([u, v], along="component"), over="component"))).collect()["r"]
    check("rotated u is -v", same(rotated.sel(rotated="u", drop=True), -ds.v))
    check("rotated v is u", same(rotated.sel(rotated="v", drop=True), ds.u))
    check("rotation keeps speed", same(np.hypot(rotated.sel(rotated="u", drop=True),
                                                rotated.sel(rotated="v", drop=True)), np.hypot(ds.u, ds.v)))

    # Mean speed per pressure level, northern hemisphere only.
    north = (speed_data >> filter(latitude >= 0) >> group_by(level)
             >> summarise(mean_speed = mean(speed))).collect()
    expected = np.hypot(ds.u, ds.v).sel(latitude=slice(90, 0)).mean(["month", "latitude", "longitude"])
    check("northern mean speed per level", same(north["mean_speed"], expected))


# ── 3. Landsat 7: bands as a field axis, as in stars ────────────────────────


def landsat(root: Path) -> None:
    print("\n== L7_ETMs.tif ==")
    import rioxarray

    with rioxarray.open_rasterio(root / "L7_ETMs.tif") as opened:
        raw = opened.load()  # close the file: an open handle prints noise at exit
    names = ["blue", "green", "red", "nir", "swir1", "swir2"]  # Landsat 7 ETM+ bands 1-5 and 7
    scene = tidy(raw.assign_coords(band=names), field_axis="band")
    print(scene, "\n")
    red, nir, x = fields("red", "nir", "x")

    # Bands are uint8, so convert before subtracting: NumPy integer arithmetic wraps around.
    ndvi_data = scene >> mutate(ndvi = (nir * 1.0 - red) / (nir * 1.0 + red))
    r = raw.sel(band=3, drop=True).astype(float)
    n_ = raw.sel(band=4, drop=True).astype(float)
    expected_ndvi = (n_ - r) / (n_ + r)
    check("NDVI", same(ndvi_data.collect()["ndvi"], expected_ndvi))

    wrapped = (scene >> mutate(bad = nir - red)).collect()["bad"]
    check("uint8 nir - red wraps as in NumPy (why NDVI converts first)",
          wrapped.dtype == np.uint8 and int(wrapped.min()) >= 0)

    # One mean per band, as stars' st_apply(x, "band", mean): the field axis is kept.
    per_band = (scene >> summarise(m = mean(col("values")))).collect()["m"]
    check("mean per band keeps the band axis",
          per_band.dims == ("band",) and np.allclose(per_band.values, raw.mean(["x", "y"]).values))

    # Crop as in the stars vignette: filter(x > 289000, x < 291000).
    crop = scene >> filter(x > 289000, x < 291000)
    check("crop on x", same(crop.collect()["values"], raw.assign_coords(band=names).sel(x=slice(289000, 291000))))

    # Two bands as columns.
    check("select red and nir", (scene >> select(red, nir)).collect()["values"].band.values.tolist() == ["red", "nir"])

    # Vegetated pixels per scene: a count over both spatial axes.
    vegetated = (ndvi_data >> summarise(k = sum(col("ndvi") > 0.3))).collect()["k"].item()
    check("vegetated pixel count", vegetated == int((expected_ndvi > 0.3).sum()))
    print(f"     NDVI > 0.3 on {vegetated:,} of {raw.sizes['x'] * raw.sizes['y']:,} pixels")


def main(root: Path) -> None:
    air_temperature(root)
    era_wind(root)
    landsat(root)
    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    if failed:
        raise SystemExit("tidy3A and plain xarray disagree: " + ", ".join(failed))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/sample_data.py /path/to/samples")
    main(Path(sys.argv[1]))
