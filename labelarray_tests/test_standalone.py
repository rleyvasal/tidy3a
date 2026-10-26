"""The library stands alone (plan milestone 10, steps A3, A5, and A6).

It imports nothing else of tidy3a, runs without xarray and pandas, and its
tests import nothing of tidy3a but the library, so library and tests can
leave together (step A7).
"""

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

from tidy3a import labelarray

LIBRARY = Path(inspect.getfile(labelarray)).parent
TESTS = Path(__file__).parent


def _imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    names |= {f"{node.module}.{alias.name}" for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module == "tidy3a" for alias in node.names}
    return names


def _outside(names: set[str]) -> set[str]:
    return {name for name in names if name.startswith("tidy3a") and name != "tidy3a"
            and not name.startswith("tidy3a.labelarray")}


def test_the_library_imports_nothing_else_of_tidy3a():
    for path in LIBRARY.glob("*.py"):
        assert not _outside(_imported(path)), f"{path.name} imports {_outside(_imported(path))}"


def test_the_library_tests_import_nothing_of_tidy3a_but_the_library():
    paths = list(TESTS.glob("*.py"))
    assert len(paths) >= 10
    for path in paths:
        assert not _outside(_imported(path)), f"{path.name} imports {_outside(_imported(path))}"


WITHOUT_XARRAY_OR_PANDAS = textwrap.dedent("""
    import importlib.abc, sys

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("xarray", "pandas"):
                raise ImportError(f"{name} is not installed")

    sys.meta_path.insert(0, Blocked())

    import numpy as np
    from tidy3a.labelarray import Array, Collection, Grouping, grouped

    a = Array(np.arange(6.0).reshape(3, 2), ("k", "band"), {"k": [3, 1, 2]})
    b = Array([30.0, 10.0, 20.0], ("k",), {"k": [3, 1, 2]})
    left = Collection({"a": a}, a.coords)
    right = Collection({"b": b.isel(k=[1, 2])}, {"k": [1, 2]})
    joined = left.join(right, "inner")
    assert joined.coords["k"].tolist() == [1, 2]
    means = grouped.reduce(a + b, "mean", Grouping(np.array([0, 1, 0]), ["even", "odd"]), "k", "parity")
    assert means.data.tolist() == [[27.0, 28.0], [12.0, 13.0]]
    assert not {"xarray", "pandas"} & {name.split(".")[0] for name in sys.modules}
""")


def test_the_library_runs_without_xarray_or_pandas(tmp_path):
    script = tmp_path / "alone.py"
    script.write_text(WITHOUT_XARRAY_OR_PANDAS)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
