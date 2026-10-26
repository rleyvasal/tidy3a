"""Tables need Polars only when used (plan milestone 8, part 2).

These run with or without Polars installed: importing tidy3a imports neither
Polars nor tidy3, and ``as_table()`` without Polars says how to get it.
"""

import sys

import numpy as np
import pytest

from tidy3a import TidyError, as_table, infer, read_array


def cloud():
    values = np.zeros((2, 3), dtype="float32")
    return read_array(values, dims={"point": infer, "feature": 3}, labels={"feature": ["x", "y", "z"]},
                      field_axis="feature")


def test_without_polars_as_table_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", None)
    with pytest.raises(TidyError, match=r"as_table\(\) gives a Polars DataFrame, and Polars is not installed"):
        cloud() >> as_table()


def test_importing_tidy3a_imports_neither_polars_nor_tidy3():
    import subprocess

    code = "import sys, tidy3a; assert not {'polars', 'tidy3'} & set(sys.modules), sorted({'polars', 'tidy3'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
