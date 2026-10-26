"""The xarray backend's float32 sums and means carry what xarray's own reduction would.

Their values come from labelarray's correctly rounded kernel; their axes,
coordinates, name, and attributes are built directly rather than by
reducing a slice, and must match what that reduction gives, under each
``keep_attrs`` setting, for coordinates along kept and reduced axes, scalar
coordinates, and unnamed arrays.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a.xarray_backend import XarrayBackend

pytestmark = pytest.mark.backends("xarray", reason="tests the xarray backend's own method")


def arrays():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 4, 5)).astype("float32")
    yield xr.DataArray(values, dims=("i", "point", "band"), coords={"i": ["x", "y", "z"]}, name="p",
                       attrs={"units": "m"})
    yield xr.DataArray(
        values, dims=("i", "point", "band"),
        coords={"point": np.arange(4) * 2, "band": list("abcde"), "station": "north",
                "lat": ("point", np.linspace(0, 3, 4)), "weight": ("i", [1.0, 2.0, 3.0])},
        attrs={"long_name": "air"},
    )
    yield xr.DataArray(values[0], dims=("point", "band"))


@pytest.mark.parametrize("keep_attrs", ["default", True, False])
@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("axes", [["i"], ["i", "band"], ["point"]])
def test_float32_results_match_xarrays_reduction_metadata(keep_attrs, op, axes):
    backend = XarrayBackend()
    for array in arrays():
        if not set(axes) <= set(array.dims):
            continue
        with xr.set_options(keep_attrs=keep_attrs):
            got = backend.reduce(array, op, axes)
            want = getattr(array.isel({a: slice(0, 1) for a in axes}), op)(dim=axes)
        assert got.dims == want.dims and got.name == want.name and got.attrs == want.attrs
        xr.testing.assert_identical(got.coords.to_dataset(), want.coords.to_dataset())
        assert got.dtype == np.float32
