"""The library's errors and tidy3A's (milestone 10, step A3).

The NumPy modules that will become a library raise their own errors, a
family of ``ValueError`` under ``tidy3a.core.errors.CoreError``, and know
nothing of tidy3A. tidy3A's ``LabelError`` and ``AlignError`` inherit from
both ``TidyError`` and the library's classes, and a library error that
escapes through a verb, a reader, ``tidy()``, or ``collect()`` is raised
again as tidy3A's class, so ``except TidyError`` and ``except
core.LabelError`` both keep working.
"""

import ast
import inspect

import numpy as np
import pytest
import xarray as xr

from tidy3a import AlignError, TidyError, col, mutate, read_array, tidy
from tidy3a import errors as tidy_errors
from tidy3a.core import errors as core
from tidy3a.numpy_backend import NUMPY, NumpyArray


def test_the_library_errors_are_value_errors_in_one_family():
    for name in ("LabelError", "AlignError", "ShapeError"):
        assert issubclass(getattr(core, name), core.CoreError)
    assert issubclass(core.CoreError, ValueError) and not issubclass(core.CoreError, TidyError)


def test_the_library_errors_module_knows_nothing_of_tidy3a():
    tree = ast.parse(inspect.getsource(core))
    names = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {name for name in names if name.startswith("tidy3a") and not name.startswith("tidy3a.core")}


def test_tidy3as_label_and_align_errors_inherit_from_both():
    assert issubclass(tidy_errors.LabelError, TidyError) and issubclass(tidy_errors.LabelError, core.LabelError)
    assert issubclass(AlignError, TidyError) and issubclass(AlignError, core.AlignError)


def test_the_numpy_modules_raise_the_library_errors():
    with pytest.raises(core.ShapeError, match="2 axis names"):
        NumpyArray(np.zeros(3), ("a", "b"))
    with pytest.raises(core.LabelError, match="no label"):
        NUMPY.sel(NumpyArray([1.0, 2.0], ("k",), {"k": [1, 2]}), {"k": 3})
    mixed = NumpyArray([1.0, 2.0], ("k",), {"k": np.array([1, "a"], dtype=object)})
    with pytest.raises(core.LabelError, match="mix types"):
        NUMPY.compare_labels(mixed, NumpyArray([1.0, 2.0], ("k",), {"k": np.array(["a", 1], dtype=object)}), "k")


def test_a_library_error_through_a_verb_is_a_tidy3a_error_with_the_same_message():
    data = tidy(xr.Dataset({"a": ("k", [1.0, 2.0])}, coords={"k": np.array([1, "a"], dtype=object)}), backend="numpy")
    other = xr.DataArray([1.0, 2.0], dims="k", coords={"k": np.array(["a", 1], dtype=object)})
    with pytest.raises(TidyError, match=r"align: labels on 'k' mix types \(int, str\)") as caught:
        data >> mutate(c=col("a") + other)
    assert isinstance(caught.value, core.LabelError) and isinstance(caught.value, tidy_errors.LabelError)
    assert caught.value.__cause__ is None  # one error, not a chain
    with pytest.raises(core.LabelError):
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
