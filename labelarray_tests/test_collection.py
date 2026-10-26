"""Collections of labelled arrays (``labelarray.Collection``, plan milestone 10, step A5).

Selection, assignment, joins under the exact label rule, nearest-label
matching, and conversion to and from xarray, checked against xarray.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a.labelarray import Array, Collection


@pytest.fixture
def grid() -> Array:
    rng = np.random.default_rng(5)
    return Array(rng.normal(size=(4, 3)).astype("float32"), ("lat", "time"),
                 {"lat": [10.0, 20.0, 30.0, 40.0], "time": np.arange("2026-01-01", "2026-01-04", dtype="datetime64[D]").astype("datetime64[ns]")},
                 attrs={"units": "K"}, name="air")


def _dataset(labels, values, name):
    return xr.Dataset({name: ("k", np.asarray(values, dtype="float64"))}, coords={"k": labels})


@pytest.mark.parametrize("how, xarray_join", [("left", "left"), ("inner", "inner"), ("outer", "outer")])
def test_joins_match_xarray(how, xarray_join):
    left, right = _dataset([1, 3, 5], [1, 2, 3], "a"), _dataset([5, 4, 1], [10, 20, 30], "b")
    got = Collection.from_xarray(left).join(Collection.from_xarray(right), how).to_xarray()
    want = xr.merge([left, right], join=xarray_join)
    xr.testing.assert_identical(got, want)


def test_joins_pair_labels_exactly():
    left = Collection({"a": Array([1.0, 2.0], ("k",))}, {"k": np.array([2**53 + 1, 7])})
    right = Collection({"b": Array([10.0, 20.0], ("k",))}, {"k": np.array([2**53, 7])})
    joined = left.join(right, "inner")
    assert joined.coords["k"].tolist() == [7]
    assert joined.array("b").data.tolist() == [20.0]


def test_nearest_matches_xarray():
    right = _dataset([0.0, 1.0, 2.0, 3.0], [10, 11, 12, 13], "v")
    wanted = np.array([0.1, 1.6, 5.0])
    got = Collection.from_xarray(right).nearest("k", wanted, 0.5).to_xarray()
    want = right.reindex(k=wanted, method="nearest", tolerance=0.5)
    xr.testing.assert_identical(got, want)


def test_collection_selection_assignment_and_dicts(grid):
    data = Collection({"air": grid, "count": Array(np.arange(4), ("lat",))}, grid.coords, attrs={"title": "t"})
    picked = data.isel(lat=[1, 2]).select(["count"])
    assert picked.dims == ("lat",) and picked.array("count").data.tolist() == [1, 2]
    assert data.sel(lat=20.0).dims == ("time",)
    added = data.assign("doubled", grid * 2).rename({"count": "n"})
    assert set(added.variables) == {"air", "n", "doubled"}
    assert data.to_dict()["dims"] == {"lat": 4, "time": 3}
    xr.testing.assert_identical(Collection.from_xarray(data.to_xarray()).to_xarray(), data.to_xarray())
    np.testing.assert_equal(data.to_dict(), data.to_xarray().to_dict(data="array"))
