"""The README's examples run, and print what their comments say.

Each Python block runs on its own, as a reader would paste it, with no
backend chosen (the package's own default). A line ``print(expr)  # text``
checks that ``expr`` prints as ``text``, whitespace aside.
"""

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

import tidy3a.backend as backend_module

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
PRINTED = re.compile(r"^print\((?P<expr>.*)\)\s+#\s+(?P<text>.+)$", re.MULTILINE)

pytestmark = pytest.mark.backends("xarray", reason="runs on the package's own default backend")


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def test_the_readme_has_its_examples():
    assert len(BLOCKS) >= 4, "the README's Python examples were not found"


@pytest.mark.parametrize("number", range(len(BLOCKS)))
def test_each_readme_example_runs_and_prints_what_it_says(number, monkeypatch):
    monkeypatch.setattr(backend_module, "_default", None)  # no set_backend(), as for a new reader
    code = BLOCKS[number]
    needs = set(re.findall(r"^(?:import|from) (\w+)", code, re.MULTILINE))
    needs |= {"polars"} if "as_table(" in code else set()  # as_table() gives a Polars DataFrame
    for module in sorted(needs):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"README example {number + 1} needs {module}, an optional dependency")
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, f"README example {number + 1}", "exec"), namespace)  # noqa: S102 - the README's own code
    checks = list(PRINTED.finditer(code))
    for check in checks:
        shown = str(eval(check["expr"], namespace))
        assert _same(shown, check["text"]), f"example {number + 1}: {check['expr']} prints {shown!r}, not {check['text']!r}"
