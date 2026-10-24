"""The backend seam (plan milestone 9): verbs reach stored data only through a backend."""

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tidy3a
from tidy3a import (TidyError, col, filter, group_by, infer, left_join, mean, mutate, read_array, select,
                    slice, summarise, tidy)
from tidy3a.backend import XARRAY, XarrayBackend
from tidy3a.frame import TidyArray

PACKAGE = Path(tidy3a.__file__).parent
# The xarray backend, and the readers that build xarray datasets from files.
MAY_IMPORT_XARRAY = {"backend.py", "io.py"}


def _imports_xarray(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "xarray" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "xarray":
            return True
    return False


def test_only_the_backend_and_readers_import_xarray():
    importing = sorted(p.name for p in PACKAGE.glob("*.py") if _imports_xarray(p))
    assert set(importing) <= MAY_IMPORT_XARRAY, importing


def cloud():
    values = np.array([[0.0, 1.0, 10.0, 1.0], [2.0, 3.0, 20.0, 2.0], [4.0, 5.0, 30.0, 1.0]], dtype="float32")
    return read_array(values, dims={"point": infer, "feature": 4},
                      labels={"feature": ["x", "y", "intensity", "ring"]}, field_axis="feature")


def test_every_verb_keeps_the_frames_backend():
    frame = cloud()
    assert frame._backend is XARRAY
    steps = [select("x", "intensity", "ring"), filter(col("x") > 0), mutate(r=col("x") * 2),
             slice(point=[1]), group_by(col("ring")), summarise(m=mean(col("intensity")))]
    for step in steps:
        frame = frame >> step
        assert frame._backend is XARRAY
    assert tidy(xr.Dataset({"a": ("t", [1.0])}))._backend is XARRAY


def test_joining_data_from_two_backends_is_an_error_naming_both():
    class Other(XarrayBackend):
        name = "other"

    right = TidyArray(xr.Dataset({"b": ("point", [1.0, 2.0, 3.0])}), backend=Other())
    with pytest.raises(TidyError, match="xarray backend and the other data the other backend"):
        cloud() >> left_join(right)
