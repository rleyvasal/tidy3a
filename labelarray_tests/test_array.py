"""Labelled arrays (``labelarray.Array``, plan milestone 10, steps A4 and A5).

Selection by positions and labels, transposing, renaming, reductions, and
contraction are checked against xarray, the reference implementation.
NumPy's ufuncs, ``np.where``, and Python's operators broadcast by axis name
under the exact alignment rule: the same labels in another order are matched
by label, and anything else is an ``AlignError``.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import labelarray
from tidy3a.labelarray import AlignError, Array, LabelError, ShapeError, align, concat


def _xr(array: Array) -> xr.DataArray:
    return xr.DataArray(array.data, dims=array.dims, coords={d: array.labels(d) for d in array.coords})


def _same(got: Array, want: xr.DataArray) -> None:
    assert got.dims == want.dims
    np.testing.assert_array_equal(got.data, want.values)
    for dim in want.indexes:
        np.testing.assert_array_equal(got.labels(dim), want[dim].values)


@pytest.fixture
def grid() -> Array:
    rng = np.random.default_rng(5)
    return Array(rng.normal(size=(4, 3)).astype("float32"), ("lat", "time"),
                 {"lat": [10.0, 20.0, 30.0, 40.0], "time": np.arange("2026-01-01", "2026-01-04", dtype="datetime64[D]").astype("datetime64[ns]")},
                 attrs={"units": "K"}, name="air")


# ── The public API ──────────────────────────────────────────────────────────


def test_the_public_api_is_importable_from_the_package():
    for name in labelarray.__all__:
        assert getattr(labelarray, name) is not None, name
    assert {"Array", "Collection", "Grouping", "factorize", "index_for", "LabelArrayError"} <= set(labelarray.__all__)


# ── Arrays ──────────────────────────────────────────────────────────────────


def test_selection_by_positions_and_labels_matches_xarray(grid):
    _same(grid.isel(lat=[3, 0], time=slice(1, None)), _xr(grid).isel(lat=[3, 0], time=slice(1, None)))
    _same(grid.isel({"lat": 1}), _xr(grid).isel(lat=1))
    _same(grid.sel(lat=[40.0, 10.0]), _xr(grid).sel(lat=[40.0, 10.0]))
    _same(grid.sel(time=np.datetime64("2026-01-02")), _xr(grid).sel(time=np.datetime64("2026-01-02")))
    _same(grid.isel(lat=np.array([True, False, True, False])), _xr(grid).isel(lat=[0, 2]))


def test_label_lookups_are_exact():
    big = Array([1.0, 2.0], ("k",), {"k": [2**53 + 1, 3]})
    assert big.sel(k=2**53 + 1).data == 1.0
    with pytest.raises(LabelError, match="no label"):
        big.sel(k=float(2**53))  # 2**53 + 1 is not 2**53
    with pytest.raises(LabelError, match="no label"):
        big.sel(k=4)


def test_transpose_rename_and_attributes(grid):
    turned = grid.transpose("time", "lat")
    _same(turned, _xr(grid).transpose("time", "lat"))
    assert turned.attrs == {"units": "K"} and turned.name == "air"
    renamed = grid.rename({"lat": "latitude"})
    assert renamed.dims == ("latitude", "time")
    np.testing.assert_array_equal(renamed.labels("latitude"), grid.labels("lat"))


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "median", "std", "var"])
def test_reductions_match_xarray(grid, op):
    got = grid.reduce(op, ["time"])
    want = getattr(_xr(grid).astype("float64"), op)("time").astype("float32") if op in ("sum", "mean") \
        else getattr(_xr(grid), op)("time")
    assert got.dims == ("lat",)
    np.testing.assert_allclose(got.data, want.values, rtol=1e-6)
    np.testing.assert_array_equal(got.labels("lat"), grid.labels("lat"))


def test_float32_sums_are_correctly_rounded_in_any_layout():
    values = np.array([[2.0**24, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -(2.0**24)]], dtype="float32")
    row = Array(values, ("one", "k"))
    column = Array(np.asfortranarray(values.T), ("k", "one"))
    assert row.reduce("sum", ["k"]).data.tolist() == [8.0]
    assert column.reduce("sum", ["k"]).data.tolist() == [8.0]


def test_missing_values_in_reductions(grid):
    data = grid.data.copy()
    data[0, 1] = np.nan
    gappy = Array(data, grid.dims, grid.coords)
    assert np.isnan(gappy.reduce("mean", ["time"]).data[0])
    np.testing.assert_allclose(gappy.reduce("mean", ["time"], na_rm=True).data, _xr(gappy).mean("time").values,
                               rtol=1e-6)  # xarray adds float32 in float32; the library rounds once
    assert np.isnan(gappy.reduce("sum", ["time"], na_rm=True, min_count=3).data[0])


def test_dot_matches_xarray():
    rotation = Array(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float64"), ("i", "feature"),
                     {"i": ["x", "y", "z"], "feature": ["x", "y", "z"]})
    points = Array(np.arange(12.0).reshape(4, 3), ("point", "feature"), {"feature": ["x", "y", "z"]})
    _same(rotation.dot(points, ["feature"]), xr.dot(_xr(rotation), _xr(points), dim="feature"))


def test_operators_align_by_label_and_refuse_what_differs():
    a = Array([1.0, 2.0, 3.0], ("x",), {"x": [1, 2, 3]})
    b = Array([30.0, 10.0, 20.0], ("x",), {"x": [3, 1, 2]})
    assert (a + b).data.tolist() == [11.0, 22.0, 33.0]
    first, second = align(a, b)
    assert first is a and second.labels("x").tolist() == [1, 2, 3]
    with pytest.raises(AlignError):
        a + Array([1.0, 2.0, 3.0], ("x",), {"x": [1, 2, 4]})


def test_concat_matches_xarray(grid):
    other = Array(grid.data * 2, grid.dims, grid.coords)
    got = concat([grid, other], "run", ["a", "b"])
    want = xr.concat([_xr(grid), _xr(other)], dim=xr.DataArray(["a", "b"], dims="run", name="run"))
    _same(got, want.transpose("run", "lat", "time"))


def test_bad_shapes_are_shape_errors():
    with pytest.raises(ShapeError, match="axis names"):
        Array(np.zeros(3), ("a", "b"))
    with pytest.raises(ShapeError, match="labels"):
        Array(np.zeros(3), ("a",), {"a": [1, 2]})


# ── NumPy's protocols ───────────────────────────────────────────────────────


def test_ufuncs_with_two_outputs_give_two_arrays():
    a = Array([7.0, -7.5], ("x",), {"x": [1, 2]})
    whole, part = np.modf(a)
    assert whole.values.tolist() == [0.0, -0.5] and part.values.tolist() == [7.0, -7.0]
    quotient, remainder = divmod(a, 2)
    assert quotient.values.tolist() == [3.0, -4.0] and remainder.values.tolist() == [1.0, 0.5]
    assert quotient.coords["x"].tolist() == [1, 2]


def test_sqrt_keeps_axes_and_labels():
    a = Array([1.0, 4.0, 9.0], ("x",), {"x": [3, 1, 2]}, attrs={"units": "m"}, name="v")
    root = np.sqrt(a)
    assert isinstance(root, Array) and root.dims == ("x",)
    assert root.values.tolist() == [1.0, 2.0, 3.0] and root.coords["x"].tolist() == [3, 1, 2]


def test_broadcasting_is_by_axis_name_not_position():
    a = Array(np.arange(6.0).reshape(2, 3), ("i", "j"))
    b = Array(np.array([10.0, 20.0]), ("i",))
    total = np.add(a, b)
    assert total.dims == ("i", "j") and total.values.tolist() == [[10, 11, 12], [23, 24, 25]]
    assert (b + a).dims == ("i", "j")  # the first operand's axes first, as apply does
    c = Array(np.arange(6.0).reshape(3, 2), ("j", "i"))
    np.testing.assert_array_equal((a - c).values, a.values - c.values.T)


def test_a_reorder_is_matched_by_label():
    a = Array([1.0, 2.0, 3.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    b = Array([20.0, 10.0, 30.0], ("x",), {"x": [np.nan, 0.5, 1.5]})
    total = a + b
    assert total.values.tolist() == [11.0, 22.0, 33.0]
    assert np.asarray(total.coords["x"]).tolist()[::2] == [0.5, 1.5]
    hours = np.array(["2026-01-01T00", "2026-01-01T01"], dtype="datetime64[s]")
    c = Array([1.0, 2.0], ("t",), {"t": hours})
    d = Array([20.0, 10.0], ("t",), {"t": hours[::-1].astype("datetime64[ns]")})
    assert np.add(c, d).values.tolist() == [11.0, 22.0]


def test_what_alignment_refuses_is_an_align_error():
    a = Array([1.0, 2.0], ("k",), {"k": [2**53 + 1, 3]})
    with pytest.raises(AlignError, match="labels differ"):
        np.add(a, Array([1.0, 2.0], ("k",), {"k": [float(2**53), 3.0]}))  # exact: 2**53 + 1 is not 2**53
    with pytest.raises(AlignError, match="labelled on one operand"):
        np.add(a, Array([1.0, 2.0], ("k",)))
    with pytest.raises(AlignError, match="neither has labels"):
        np.add(Array([1.0, 2.0], ("k",)), Array([1.0, 2.0, 3.0], ("k",)))
    with pytest.raises(AlignError, match="repeated labels"):
        np.add(Array([1.0, 2.0, 3.0], ("k",), {"k": [1, 1, 2]}), Array([1.0, 2.0, 3.0], ("k",), {"k": [1, 2, 1]}))
    with pytest.raises(AlignError, match="labels differ"):
        np.where(a > 0, a, Array([1.0, 2.0], ("k",), {"k": [True, False]}))  # True is not 1


def test_repeated_labels_in_the_same_order_pair_by_position():
    a = Array([1.0, 2.0, 3.0], ("k",), {"k": [1, 1, 2]})
    assert (a + a).values.tolist() == [2.0, 4.0, 6.0]


def test_what_has_no_meaning_by_axis_name_is_refused():
    a = Array([1.0, 4.0], ("x",))
    with pytest.raises(TypeError):
        np.add(a, np.array([1.0, 2.0]))  # a plain array has no axis names
    with pytest.raises(TypeError):
        np.add.reduce(a)
    with pytest.raises(TypeError):
        np.sqrt(a, out=np.empty(2))
    with pytest.raises(TypeError):
        np.matmul(a, a)  # core dimensions are positions
    with pytest.raises(TypeError):
        np.mean(a)
    with pytest.raises(TypeError):
        np.where(a > 1)  # positions, not values


def test_comparisons_give_arrays_so_arrays_are_not_hashable():
    a, b = Array([1.0, 2.0], ("x",)), Array([1.0, 3.0], ("x",))
    assert (a == b).values.tolist() == [True, False]
    with pytest.raises(TypeError):
        hash(a)


def test_operators_stay_quiet_and_numpy_functions_warn():
    a = Array([1.0, 0.0], ("x",))
    with np.errstate(all="raise"):
        assert (a / 0.0).values[0] == np.inf  # Python operators do not warn, as on xarray arrays
        with pytest.raises(FloatingPointError):
            np.divide(a, 0.0)  # NumPy functions do


def test_operators_defer_to_objects_that_opt_out_of_ufuncs():
    class Deferring:
        __array_ufunc__ = None

        def __radd__(self, other):
            return "deferred"

    assert Array([1.0], ("x",)) + Deferring() == "deferred"
