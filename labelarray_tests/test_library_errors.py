"""The library's errors (plan milestone 10, step A3).

Every error the library raises is a ``ValueError`` in one family under
``labelarray.errors.LabelArrayError``.
"""

import numpy as np
import pytest

from tidy3a.labelarray import Array, align
from tidy3a.labelarray import errors as labelarray


def test_the_library_errors_are_value_errors_in_one_family():
    for name in ("LabelError", "AlignError", "GroupError", "ShapeError"):
        assert issubclass(getattr(labelarray, name), labelarray.LabelArrayError)
    assert issubclass(labelarray.LabelArrayError, ValueError)


def test_arrays_raise_the_library_errors():
    with pytest.raises(labelarray.ShapeError, match="2 axis names"):
        Array(np.zeros(3), ("a", "b"))
    with pytest.raises(labelarray.LabelError, match="no label"):
        Array([1.0, 2.0], ("k",), {"k": [1, 2]}).sel(k=3)
    mixed = Array([1.0, 2.0], ("k",), {"k": np.array([1, "a"], dtype=object)})
    with pytest.raises(labelarray.LabelError, match="mix types"):
        align(mixed, Array([1.0, 2.0], ("k",), {"k": np.array(["a", 1], dtype=object)}))
    with pytest.raises(labelarray.AlignError, match="labelled on one operand"):
        align(Array([1.0], ("k",), {"k": [1]}), Array([1.0], ("k",)))
