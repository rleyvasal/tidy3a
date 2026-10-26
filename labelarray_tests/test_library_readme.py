"""The library's README: its examples run, and print what their comments say."""

import inspect
import re
from pathlib import Path

import pytest

from tidy3a import labelarray

README = Path(inspect.getfile(labelarray)).parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
PRINTED = re.compile(r"^print\((?P<expr>.*)\)\s+#\s+(?P<text>.+)$", re.MULTILINE)


@pytest.mark.parametrize("number", range(len(BLOCKS)))
def test_each_readme_example_runs_and_prints_what_it_says(number, capsys):
    code = BLOCKS[number]
    namespace: dict = {}
    exec(compile(code, f"labelarray README example {number + 1}", "exec"), namespace)  # noqa: S102 - the README's own code
    capsys.readouterr()
    for check in PRINTED.finditer(code):
        shown = " ".join(str(eval(check["expr"], namespace)).split())
        assert shown == " ".join(check["text"].split()), f"example {number + 1}: {check['expr']} prints {shown!r}"
