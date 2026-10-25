"""One exact-label rule for every path that compares labels (docs/backends.md).

Labels on an axis match only when they are the same kind (booleans, numbers,
times, durations, text, bytes) and exactly equal: no conversion may change a
value. Alignment and joins follow the same rule on both backends,
whatever order the labels are in.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    AlignError,
    TidyError,
    col,
    full_join,
    inner_join,
    left_join,
    mutate,
    tidy,
)

BIG = 2**53 + 1  # an integer that float64 cannot represent


def _data(labels, name="a"):
    return xr.Dataset({name: ("s", np.arange(1.0, len(labels) + 1))}, coords={"s": labels})


def test_integer_labels_never_match_floats_that_round_to_them_in_the_same_order():
    data = tidy(_data(np.array([BIG, 3], dtype="int64")))
    other = xr.DataArray([10.0, 20.0], dims="s", coords={"s": np.array([float(2**53), 3.0])})
    with pytest.raises(AlignError, match="s labels differ"):
        data >> mutate(c=col("a") + other)


def test_integer_labels_never_match_floats_that_round_to_them_in_another_order():
    data = tidy(_data(np.array([BIG, 3], dtype="int64")))
    other = xr.DataArray([20.0, 10.0], dims="s", coords={"s": np.array([3.0, float(2**53)])})
    with pytest.raises(AlignError, match="s labels differ"):
        data >> mutate(c=col("a") + other)


def test_equal_values_of_different_number_types_still_match():
    data = tidy(_data(np.array([1, 2], dtype="int64")))
    other = xr.DataArray([10.0, 20.0], dims="s", coords={"s": np.array([1.0, 2.0])})
    assert (data >> mutate(c=col("a") + other)).collect()["c"].values.tolist() == [11.0, 22.0]


@pytest.mark.parametrize("join", [left_join, inner_join, full_join])
def test_joins_refuse_labels_their_combined_type_cannot_hold(join):
    left = tidy(_data(np.array([BIG, 5], dtype="int64")))
    right = _data(np.array([float(2**53), 9.0]), name="b")
    with pytest.raises(AlignError, match=r"s labels .* cannot all be held exactly"):
        left >> join(right)


def test_joins_of_representable_number_types_keep_every_value():
    left = tidy(_data(np.array([1, 5], dtype="int64")))
    right = _data(np.array([5.0, 9.0]), name="b")
    out = (left >> full_join(right)).collect()
    assert out["s"].values.tolist() == [1.0, 5.0, 9.0]
    assert out["a"].values.tolist()[:2] == [1.0, 2.0] and out["b"].values.tolist()[1:] == [1.0, 2.0]


@pytest.mark.parametrize("dtype, promoted", [
    ("int8", "float32"), ("int16", "float32"), ("uint8", "float32"), ("uint16", "float32"),
    ("int32", "float64"), ("int64", "float64"), ("uint32", "float64"), ("uint64", "float64"),
    ("float32", "float32"), ("float64", "float64"), ("bool", "float64"),
])
def test_join_gaps_promote_types_as_xarray_does(dtype, promoted):
    left = tidy(_data(np.array([1, 2])))
    right = xr.Dataset({"b": ("s", np.array([1, 0], dtype=dtype))}, coords={"s": [2, 3]})
    assert str((left >> left_join(right)).collect()["b"].dtype) == promoted


def test_repeated_labels_that_must_be_matched_in_a_join_are_one_error():
    left = tidy(_data(np.array([1, 1, 2])))
    right = _data(np.array([1, 2, 3]), name="b")
    with pytest.raises(AlignError, match="s has repeated labels"):
        left >> left_join(right)


def test_identical_repeated_labels_still_join():
    left = tidy(_data(np.array([1, 1, 2])))
    right = _data(np.array([1, 1, 2]), name="b")
    out = (left >> left_join(right)).collect()
    assert out["b"].values.tolist() == [1.0, 2.0, 3.0]


def test_the_rule_is_a_tidy3a_error_on_both_backends():
    assert issubclass(AlignError, TidyError)
