"""The backend seam (plan milestone 9): verbs reach stored data only through a backend."""

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tidy3a
from tidy3a import (TidyError, col, filter, group_by, infer, left_join, mean, mutate, read_array, select,
                    slice, summarise, tidy)
from tidy3a.backend import Backend, XarrayBackend, get_backend
from tidy3a.numpy_backend import NumpyBackend
from tidy3a.frame import TidyArray

PACKAGE = Path(tidy3a.__file__).parent
OPTIONAL = ("xarray", "pandas")
# The xarray backend is the only module that imports xarray or pandas when it is imported.
MAY_IMPORT_AT_MODULE_LEVEL = {"xarray_backend.py"}
# These use them only when present or asked for: conversion to and from xarray, label
# display and fast grouping in the NumPy backend, pandas' fast paths in expressions, and
# opening a written Zarr store, which only xarray reads.
MAY_IMPORT_IN_FUNCTIONS = {"numpy_backend.py", "expr.py", "write.py"}


def _optional_imports(tree: ast.AST) -> list[ast.AST]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in OPTIONAL for a in node.names):
            found.append(node)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in OPTIONAL:
            found.append(node)
    return found


@pytest.mark.backends("xarray", reason="reads source files; the backend does not matter")
def test_only_the_xarray_backend_imports_xarray_or_pandas_at_import_time():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        found = _optional_imports(tree)
        if path.name in MAY_IMPORT_AT_MODULE_LEVEL or not found:
            continue
        assert path.name in MAY_IMPORT_IN_FUNCTIONS, f"{path.name} imports xarray or pandas"
        top_level = [node for node in tree.body if node in found]
        assert not top_level, f"{path.name} imports xarray or pandas at module level"


NO_XARRAY_OR_PANDAS = textwrap.dedent("""
    import importlib.abc, sys

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("xarray", "pandas"):
                raise ImportError(f"{name} is not installed")

    sys.meta_path.insert(0, Blocked())

    import numpy as np
    import tidy3a
    from tidy3a import (bins, col, dot, fields, filter, formula, group_by, infer, mean, mutate, n, pull,
                        read_array, select, summarise)
    from tidy3a.numpy_backend import NumpyArray

    tidy3a.set_backend("numpy")
    records = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], "float32")
    x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
    xyz = ["x", "y", "z"]
    cloud = read_array(records, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    rotation = NumpyArray(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float32"), ("i", "feature"),
                          {"i": xyz, "feature": xyz})
    offset = NumpyArray(np.array([1, 0, 0], "float32"), ("i",), {"i": xyz})
    nearby = (cloud >> mutate(p=dot(rotation, [x, y, z], over="feature") + offset)
              >> mutate(distance=formula(r"\\sqrt{\\sum_i p_i^2}")) >> filter(distance <= 50))
    per_ring = nearby >> group_by(ring) >> summarise(points=n(), m=mean(intensity))
    assert (nearby >> pull("point")).tolist() == [1, 2, 4]
    assert (nearby >> pull(distance)).tolist() == [49.0, 50.0, 3.0]
    assert (per_ring >> pull("m")).tolist() == [10.0, 30.0]
    assert (cloud >> select(x) >> mutate(x=x * 2) >> pull(x)).tolist() == [0.0, 0.0, 102.0, 0.0]
    times = read_array(np.arange(4.0), dims=["time"],
                       labels={"time": np.arange("2026-06-01", "2026-06-05", dtype="datetime64[D]")})
    two_days = times >> group_by(bins(col("time"), "2 days")) >> summarise(m=mean(col("values")))
    assert (two_days >> pull("m")).tolist() == [0.5, 2.5]
    assert "time_bin: [2026-06-01, 2026-06-03)" in repr(two_days)
    assert not {"xarray", "pandas"} & {name.split(".")[0] for name in sys.modules}
    print(repr(per_ring))
""")


@pytest.mark.backends("xarray", reason="runs its own process on the NumPy backend")
def test_eager_pipelines_run_without_xarray_or_pandas(tmp_path):
    script = tmp_path / "without.py"
    script.write_text(NO_XARRAY_OR_PANDAS)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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


# ── The interface, written down ─────────────────────────────────────────────


def _public_methods(cls):
    return {name for name in dir(cls) if not name.startswith("_") and callable(getattr(cls, name))}


@pytest.mark.parametrize("backend_class", [XarrayBackend, NumpyBackend])
def test_each_backend_implements_exactly_the_protocol(backend_class):
    """The same methods, parameters, and defaults as ``Backend``, and no public method outside it."""
    protocol = _public_methods(Backend)
    assert _public_methods(backend_class) == protocol
    assert isinstance(backend_class(), Backend)
    for name in sorted(protocol):
        expected = inspect.signature(getattr(Backend, name))
        actual = inspect.signature(getattr(backend_class, name))
        described = [(p.name, p.kind, p.default) for p in expected.parameters.values()]
        implemented = [(p.name, p.kind, p.default) for p in actual.parameters.values()]
        assert implemented == described, f"{backend_class.__name__}.{name}{actual} differs from Backend.{name}{expected}"


def test_every_protocol_method_is_used_by_the_shared_modules():
    """The protocol holds what the verbs, expressions, rules, and readers call, and nothing more."""
    source = Path(__file__).resolve().parents[1] / "src" / "tidy3a"
    shared = "".join((source / name).read_text() for name in
                     ("verbs.py", "expr.py", "formula.py", "align.py", "frame.py", "io.py", "linalg.py", "backend.py",
                      "plan.py", "tables.py", "chunked.py"))
    unused = sorted(name for name in _public_methods(Backend) if f".{name}(" not in shared)
    assert unused == []
