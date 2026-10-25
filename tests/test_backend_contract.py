"""The deliberate differences between backends (docs/backends.md), each tested on both."""

import numpy as np
import pytest
import xarray as xr

from tidy3a import AlignError, TidyError, col, mutate, select, set_backend, tidy, where

pytestmark = pytest.mark.backends("xarray", reason="names both backends itself")


@pytest.fixture
def encoded():
    data = xr.Dataset({"t": ("s", np.array([1.0, 2.0], dtype="float32"), {"units": "K"})},
                      coords={"s": [1, 2]}, attrs={"title": "encoded"})
    data.t.encoding = {"dtype": np.dtype("int16"), "scale_factor": 0.5}
    data.encoding = {"unlimited_dims": {"s"}}
    return data


def test_storage_encodings_are_kept_only_on_xarray(encoded):
    on_xarray = tidy(encoded, backend="xarray").collect()
    on_numpy = tidy(encoded, backend="numpy").collect()
    assert on_xarray.t.encoding == encoded.t.encoding and on_xarray.encoding == encoded.encoding
    assert on_numpy.t.encoding == {} and on_numpy.encoding == {}
    for result in (on_xarray, on_numpy):  # attributes follow the same rules on both
        assert result.t.attrs == {"units": "K"} and result.attrs == {"title": "encoded"}


def test_lazy_values_stay_on_xarray():
    dask = pytest.importorskip("dask.array")
    lazy = xr.Dataset({"a": ("t", dask.ones(4, chunks=2))})
    set_backend("numpy")  # the test fixture restores the default afterwards
    assert tidy(lazy)._backend.name == "xarray"  # NumPy is only the default: lazy data stays lazy
    with pytest.raises(TidyError, match='backend="numpy" holds eager, in-memory data only, but a is lazy'):
        tidy(lazy, backend="numpy")


@pytest.mark.parametrize("coords", [
    {"s": [1, 2], "station": "X"},                       # a scalar coordinate
    {"s": [1, 2], "lat": ("s", [10.0, 20.0])},           # an auxiliary coordinate along s
])
def test_other_coordinates_stay_on_xarray(coords):
    data = xr.Dataset({"a": ("s", [1.0, 2.0])}, coords=coords)
    assert tidy(data, backend="xarray")._backend.name == "xarray"
    with pytest.raises(TidyError, match="is a coordinate that is not one axis's labels"):
        tidy(data, backend="numpy")


def test_labels_of_different_number_types_compare_exactly_on_numpy():
    big = 2**53 + 1
    data = xr.Dataset({"a": ("s", [1.0, 2.0])}, coords={"s": np.array([big, 3], dtype="int64")})
    other = xr.DataArray([20.0, 10.0], dims="s", coords={"s": np.array([3.0, float(2**53)])})
    # pandas compares through float64, so the xarray backend pairs big with 2**53.
    paired = (tidy(data, backend="xarray") >> mutate(c=col("a") + other)).collect()
    assert paired["c"].values.tolist() == [11.0, 22.0]
    with pytest.raises(AlignError, match="s labels differ"):
        tidy(data, backend="numpy") >> mutate(c=col("a") + other)


def test_selection_predicates_receive_the_backends_arrays():
    data = xr.Dataset({"a": ("s", [1.0, 2.0]), "b": ("s", [3, 4])})
    seen = {}
    for name in ("xarray", "numpy"):
        types = []
        tidy(data, backend=name) >> select(where(lambda array: types.append(type(array).__name__) or True))
        seen[name] = set(types)
    assert seen == {"xarray": {"DataArray"}, "numpy": {"NumpyArray"}}


# ── The same on both ────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["xarray", "numpy"])
def test_repeated_labels_in_another_order_are_one_clear_error(backend):
    data = tidy(xr.Dataset({"a": ("s", [1.0, 2.0, 3.0])}, coords={"s": [1, 1, 2]}), backend=backend)
    other = xr.DataArray([10.0, 20.0, 30.0], dims="s", coords={"s": [2, 1, 1]})
    with pytest.raises(AlignError, match="s has repeated labels, in a different order"):
        data >> mutate(c=col("a") + other)
    same_order = xr.DataArray([10.0, 20.0, 30.0], dims="s", coords={"s": [1, 1, 2]})
    assert (data >> mutate(c=col("a") + same_order)).collect()["c"].values.tolist() == [11.0, 22.0, 33.0]


@pytest.mark.parametrize("backend", ["xarray", "numpy"])
def test_operators_are_quiet_and_numpy_functions_warn(backend):
    import warnings

    from tidy3a import sqrt

    data = tidy(xr.Dataset({"a": ("s", [1.0, -1.0])}), backend=backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data >> mutate(q=col("a") / 0)  # no warning
    with pytest.warns(RuntimeWarning):
        data >> mutate(r=sqrt(col("a")))


@pytest.mark.parametrize("backend", ["xarray", "numpy"])
def test_wrapping_shares_in_memory_values(backend):
    values = np.arange(4.0)
    frame = tidy(xr.Dataset({"a": ("s", values)}), backend=backend)
    assert np.shares_memory(frame.collect()["a"].values, values)
