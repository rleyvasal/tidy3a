"""NumPy's ufuncs and ``np.where`` on ``NumpyArray`` (plan milestone 10, step A4).

``np.sqrt(array)``, ``np.add(a, b)``, ``a + b``, and ``np.where(c, a, b)``
broadcast by axis name and apply the exact alignment rule (§5.5, decision 3):
the same labels in another order are matched by label, and anything else
that ``align`` refuses is an ``AlignError``. Each result must equal what the
current backend's ``apply`` gives after tidy3a's alignment, so the tests run
on both backends and compare against each, on row-major and column-major
data.
"""

import operator

import numpy as np
import pytest

from tidy3a.align import align_many
from tidy3a.backend import get_backend
from tidy3a.expr import Expr, col
from tidy3a.labelarray.errors import AlignError
from tidy3a.numpy_backend import NumpyArray

LABELS = {
    "a": np.array([10, 20, 30, 40]),
    "b": np.array(["x", "y", "z"], dtype=object),
    "c": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
    "d": np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
}


def _backend_array(value):
    """*value* as the current backend's array, with the same data, axes, and labels."""
    backend = get_backend()
    if not isinstance(value, NumpyArray) or backend.name == "numpy":
        return value
    import xarray as xr

    return xr.DataArray(value.data, dims=value.dims, coords={d: np.asarray(c) for d, c in value.coords.items()})


def _expected(op, *operands):
    backend = get_backend()
    values = [_backend_array(v) for v in operands]
    aligned = align_many(values, [f"operand {k}" for k in range(len(values))], [], backend=backend)
    with np.errstate(all="ignore"):
        return backend.where(*aligned) if op is np.where else backend.apply(op, *aligned)


def _assert_same(got, expected):
    """The same axes (in any order for xarray), labels, and bits."""
    if isinstance(expected, tuple):
        assert isinstance(got, tuple) and len(got) == len(expected)
        for one, other in zip(got, expected):
            _assert_same(one, other)
        return
    assert isinstance(got, NumpyArray)
    assert set(got.dims) == set(expected.dims)
    expected = expected.transpose(*got.dims) if get_backend().name == "xarray" else expected
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got.values, expected.values, strict=True)
    labelled = {d for d in expected.dims if d in expected.coords}
    assert set(got.coords) == labelled
    for dim in labelled:
        np.testing.assert_array_equal(np.asarray(got.coords[dim]), np.asarray(expected.coords[dim]))


def _random_array(rng, dims, labelled, layout, positive=False):
    shape = [len(LABELS[d]) for d in dims]
    data = rng.uniform(0.1 if positive else -3, 3, shape)
    data[rng.random(shape) < 0.1] = np.nan
    if layout == "F":
        data = np.asfortranarray(data)
    coords = {}
    for axis, dim in enumerate(dims):
        if dim in labelled:
            order = rng.permutation(len(LABELS[dim]))  # labels in another order on each operand
            coords[dim] = LABELS[dim][order]
            data = np.take(data, order, axis=axis)
    return NumpyArray(data, dims, coords)


def _random_dims(rng):
    return tuple(rng.permutation(list(LABELS))[: rng.integers(1, 4)])


UNARY = [np.sqrt, np.negative, np.isnan, np.exp, np.floor, np.sign]
BINARY = [np.add, np.subtract, np.multiply, np.true_divide, np.maximum, np.greater, np.arctan2, np.power,
          np.equal]
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow, operator.lt, operator.eq]


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("seed", range(30))
def test_ufuncs_equal_apply_after_alignment(seed, layout):
    rng = np.random.default_rng(seed)
    labelled = {d for d in LABELS if rng.random() < 0.7}
    left = _random_array(rng, _random_dims(rng), labelled, layout, positive=True)
    right = _random_array(rng, _random_dims(rng), labelled, layout, positive=True)
    with np.errstate(all="ignore"):
        for ufunc in UNARY:
            _assert_same(ufunc(left), _expected(ufunc, left))
        for ufunc in BINARY:
            _assert_same(ufunc(left, right), _expected(ufunc, left, right))
            _assert_same(ufunc(left, 2.5), _expected(ufunc, left, 2.5))
            _assert_same(ufunc(np.float32(2), right), _expected(ufunc, np.float32(2), right))
    for op in OPERATORS:
        _assert_same(op(left, right), _expected(op, left, right))
        _assert_same(op(3, right), _expected(op, 3, right))


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("seed", range(20))
def test_where_equals_the_backends_where(seed, layout):
    rng = np.random.default_rng(100 + seed)
    labelled = {d for d in LABELS if rng.random() < 0.7}
    condition = _random_array(rng, _random_dims(rng), labelled, layout) > 0
    true = _random_array(rng, _random_dims(rng), labelled, layout)
    false = _random_array(rng, _random_dims(rng), labelled, layout)
    _assert_same(np.where(condition, true, false), _expected(np.where, condition, true, false))
    _assert_same(np.where(condition, true, 0.0), _expected(np.where, condition, true, 0.0))


def test_ufuncs_with_two_outputs_give_two_arrays():
    a = NumpyArray([7.0, -7.5], ("x",), {"x": [1, 2]})
    whole, part = np.modf(a)
    assert whole.values.tolist() == [0.0, -0.5] and part.values.tolist() == [7.0, -7.0]
    quotient, remainder = divmod(a, 2)
    assert quotient.values.tolist() == [3.0, -4.0] and remainder.values.tolist() == [1.0, 0.5]
    assert quotient.coords["x"].tolist() == [1, 2]


def test_sqrt_keeps_axes_and_labels():
    a = NumpyArray([1.0, 4.0, 9.0], ("x",), {"x": [3, 1, 2]}, attrs={"units": "m"}, name="v")
    root = np.sqrt(a)
    assert isinstance(root, NumpyArray) and root.dims == ("x",)
    assert root.values.tolist() == [1.0, 2.0, 3.0] and root.coords["x"].tolist() == [3, 1, 2]


def test_broadcasting_is_by_axis_name_not_position():
    a = NumpyArray(np.arange(6.0).reshape(2, 3), ("i", "j"))
    b = NumpyArray(np.array([10.0, 20.0]), ("i",))
    total = np.add(a, b)
    assert total.dims == ("i", "j") and total.values.tolist() == [[10, 11, 12], [23, 24, 25]]
    assert (b + a).dims == ("i", "j")  # the first operand's axes first, as apply does
    c = NumpyArray(np.arange(6.0).reshape(3, 2), ("j", "i"))
    np.testing.assert_array_equal((a - c).values, a.values - c.values.T)


def test_a_reorder_is_matched_by_label():
    a = NumpyArray([1.0, 2.0, 3.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    b = NumpyArray([20.0, 10.0, 30.0], ("x",), {"x": [np.nan, 0.5, 1.5]})
    total = a + b
    assert total.values.tolist() == [11.0, 22.0, 33.0]
    assert np.asarray(total.coords["x"]).tolist()[::2] == [0.5, 1.5]
    hours = np.array(["2026-01-01T00", "2026-01-01T01"], dtype="datetime64[s]")
    c = NumpyArray([1.0, 2.0], ("t",), {"t": hours})
    d = NumpyArray([20.0, 10.0], ("t",), {"t": hours[::-1].astype("datetime64[ns]")})
    assert np.add(c, d).values.tolist() == [11.0, 22.0]


def test_what_alignment_refuses_is_an_align_error():
    a = NumpyArray([1.0, 2.0], ("k",), {"k": [2**53 + 1, 3]})
    with pytest.raises(AlignError, match="labels differ"):
        np.add(a, NumpyArray([1.0, 2.0], ("k",), {"k": [float(2**53), 3.0]}))  # exact: 2**53 + 1 is not 2**53
    with pytest.raises(AlignError, match="labelled on one operand"):
        np.add(a, NumpyArray([1.0, 2.0], ("k",)))
    with pytest.raises(AlignError, match="neither has labels"):
        np.add(NumpyArray([1.0, 2.0], ("k",)), NumpyArray([1.0, 2.0, 3.0], ("k",)))
    with pytest.raises(AlignError, match="repeated labels"):
        np.add(NumpyArray([1.0, 2.0, 3.0], ("k",), {"k": [1, 1, 2]}), NumpyArray([1.0, 2.0, 3.0], ("k",), {"k": [1, 2, 1]}))
    with pytest.raises(AlignError, match="labels differ"):
        np.where(a > 0, a, NumpyArray([1.0, 2.0], ("k",), {"k": [True, False]}))  # True is not 1


def test_repeated_labels_in_the_same_order_pair_by_position():
    a = NumpyArray([1.0, 2.0, 3.0], ("k",), {"k": [1, 1, 2]})
    assert (a + a).values.tolist() == [2.0, 4.0, 6.0]


def test_what_has_no_meaning_by_axis_name_is_refused():
    a = NumpyArray([1.0, 4.0], ("x",))
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


def test_operators_defer_to_expressions_and_stay_quiet():
    a = NumpyArray([1.0, 0.0], ("x",))
    assert isinstance(a + col("x"), Expr) and isinstance(a * col("x"), Expr)
    with np.errstate(all="raise"):
        assert (a / 0.0).values[0] == np.inf  # Python operators do not warn, as on the xarray backend
        with pytest.raises(FloatingPointError):
            np.divide(a, 0.0)  # NumPy functions do


def test_comparisons_give_arrays_so_arrays_are_not_hashable():
    a, b = NumpyArray([1.0, 2.0], ("x",)), NumpyArray([1.0, 3.0], ("x",))
    assert (a == b).values.tolist() == [True, False]
    with pytest.raises(TypeError):
        hash(a)
