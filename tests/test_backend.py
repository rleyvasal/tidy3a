"""The backend seam (plan milestone 9): verbs reach stored data only through a backend."""

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tidy3a
from tidy3a import (TidyError, col, filter, group_by, infer, left_join, mean, mutate, read_array, select,
                    slice, summarise, tidy)
from tidy3a.backend import XARRAY, XarrayBackend, get_backend
from tidy3a.frame import TidyArray

PACKAGE = Path(tidy3a.__file__).parent
# The xarray backend, and the readers that build xarray datasets from files.
MAY_IMPORT_XARRAY = {"backend.py", "io.py"}
# The NumPy backend converts to and from xarray only when asked, importing it there.
MAY_IMPORT_XARRAY_IN_FUNCTIONS = {"numpy_backend.py"}


def _xarray_imports(tree: ast.AST) -> list[ast.AST]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "xarray" for a in node.names):
            found.append(node)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "xarray":
            found.append(node)
    return found


@pytest.mark.backends("xarray", reason="reads source files; the backend does not matter")
def test_only_the_backends_and_readers_import_xarray():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        if path.name in MAY_IMPORT_XARRAY or not _xarray_imports(tree):
            continue
        assert path.name in MAY_IMPORT_XARRAY_IN_FUNCTIONS, f"{path.name} imports xarray"
        top_level = [node for node in tree.body if node in _xarray_imports(tree)]
        assert not top_level, f"{path.name} imports xarray at module level"


def cloud():
    values = np.array([[0.0, 1.0, 10.0, 1.0], [2.0, 3.0, 20.0, 2.0], [4.0, 5.0, 30.0, 1.0]], dtype="float32")
    return read_array(values, dims={"point": infer, "feature": 4},
                      labels={"feature": ["x", "y", "intensity", "ring"]}, field_axis="feature")


def test_every_verb_keeps_the_frames_backend():
    frame = cloud()
    backend = get_backend()
    assert frame._backend is backend
    steps = [select("x", "intensity", "ring"), filter(col("x") > 0), mutate(r=col("x") * 2),
             slice(point=[1]), group_by(col("ring")), summarise(m=mean(col("intensity")))]
    for step in steps:
        frame = frame >> step
        assert frame._backend is backend
    assert tidy(xr.Dataset({"a": ("t", [1.0])}))._backend is backend


def test_joining_data_from_two_backends_is_an_error_naming_both():
    class Other(XarrayBackend):
        name = "other"

    right = TidyArray(xr.Dataset({"b": ("point", [1.0, 2.0, 3.0])}), backend=Other())
    with pytest.raises(TidyError, match=f"{get_backend().name} backend and the other data the other backend"):
        cloud() >> left_join(right)
