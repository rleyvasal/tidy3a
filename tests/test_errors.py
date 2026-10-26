"""The library's errors and tidy3a's (milestone 10, step A3).

The library raises its own errors, a family of ``ValueError`` under
``labelarray.errors.LabelArrayError`` (tested with the library, in
``labelarray_tests``). tidy3a's ``LabelError`` and ``AlignError`` inherit from
both ``TidyError`` and the library's classes, and a library error that
escapes through a verb, a reader, ``tidy()``, or ``collect()`` is raised
again as tidy3a's class, so ``except TidyError`` and ``except
labelarray.LabelError`` both keep working.
"""

import inspect

import numpy as np
import pytest
import xarray as xr

from tidy3a import AlignError, TidyError, col, mutate, read_array, tidy
from tidy3a import errors as tidy_errors
from tidy3a.labelarray import errors as labelarray


def test_tidy3as_label_and_align_errors_inherit_from_both():
    assert not issubclass(labelarray.LabelArrayError, TidyError)
    assert issubclass(tidy_errors.LabelError, TidyError) and issubclass(tidy_errors.LabelError, labelarray.LabelError)
    assert issubclass(AlignError, TidyError) and issubclass(AlignError, labelarray.AlignError)


def test_a_library_error_through_a_verb_is_a_tidy3a_error_with_the_same_message():
    data = tidy(xr.Dataset({"a": ("k", [1.0, 2.0])}, coords={"k": np.array([1, "a"], dtype=object)}), backend="numpy")
    other = xr.DataArray([1.0, 2.0], dims="k", coords={"k": np.array(["a", 1], dtype=object)})
    with pytest.raises(TidyError, match=r"align: labels on 'k' mix types \(int, str\)") as caught:
        data >> mutate(c=col("a") + other)
    assert isinstance(caught.value, labelarray.LabelError) and isinstance(caught.value, tidy_errors.LabelError)
    assert caught.value.__cause__ is None  # one error, not a chain
    with pytest.raises(labelarray.LabelError):
        data >> mutate(c=col("a") + other)


def test_a_library_error_through_tidy_is_a_tidy3a_error():
    data = xr.Dataset({"a": ("k", [1.0, 2.0])}, coords={"k": np.array([1, "a"], dtype=object)})
    with pytest.raises(TidyError, match="mix types"):  # checking the field axis looks the labels up
        tidy(data, field_axis="k", backend="numpy")


def test_every_boundary_translates_library_errors():
    """The readers, ``tidy()``, every verb, and the ways out of a frame are wrapped."""
    from tidy3a import read_binary
    from tidy3a.frame import TidyArray
    from tidy3a.verbs import Verb

    for boundary in (read_array, read_binary, tidy, Verb.__call__, TidyArray.collect, TidyArray.to_xarray,
                     TidyArray.to_dict, TidyArray.__repr__):
        assert hasattr(boundary, "__wrapped__") and "translating" in inspect.getsource(boundary.__wrapped__.__globals__["translating"]), boundary
