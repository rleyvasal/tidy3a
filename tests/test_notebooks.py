import numpy as np
import pytest

pytest.importorskip("IPython")
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402

from tidy3a.jupyter import apply_masking, rewrite_vector_assignments  # noqa: E402

SETUP = (
    "import numpy as np, xarray as xr\n"
    "values = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype=float)\n"
    'cloud = read_array(values, dims={"point": infer, "feature": 5},\n'
    '                   labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")\n'
)


@pytest.fixture
def shell():
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "tidy3a.jupyter")
    yield shell
    shell.run_line_magic("unload_ext", "tidy3a.jupyter")
    InteractiveShell.clear_instance()


def run(shell, source):
    result = shell.run_cell(source)
    assert result.success, result.error_before_exec or result.error_in_exec
    return shell.user_ns


def failure(shell, source):
    result = shell.run_cell(source)
    assert not result.success
    return result.error_before_exec or result.error_in_exec


def test_first_cell_imports_and_uses_bare_names(shell):
    ns = run(shell, "from tidy3a import *\n" + SETUP + "xyz = cloud >> select(x, y, z)")
    assert ns["xyz"].collect()["values"].feature.values.tolist() == ["x", "y", "z"]


def test_filter_mutate_summarise_with_bare_names(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(shell, "kept = cloud >> filter(intensity > 15) >> pull(point)")
    assert ns["kept"].tolist() == [2, 3, 4]
    ns = run(shell, "m = cloud >> group_by(ring) >> summarise(m = mean(intensity)) >> pull(m)")
    assert ns["m"].tolist() == [20.0, 30.0]


def test_python_values_are_used_when_data_has_no_such_name(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(shell, "threshold = 15\nkept = cloud >> filter(intensity > threshold) >> pull(point)")
    assert ns["kept"].tolist() == [2, 3, 4]
    ns = run(shell, "masked = cloud >> mutate(m = if_else(intensity > 15, intensity, nan)) >> pull(m)")
    assert np.isnan(ns["masked"][0])


def test_collision_is_an_error_with_col_and_env_escapes(shell):
    run(shell, "from tidy3a import *\n" + SETUP + "intensity = 99")
    error = failure(shell, "cloud >> filter(intensity > 15)")
    assert 'both data and a notebook variable (intensity = 99)' in str(error)
    assert 'col("intensity")' in str(error) and 'env("intensity")' in str(error)
    ns = run(shell, 'a = cloud >> filter(col("intensity") > 15) >> pull(point)')
    assert ns["a"].tolist() == [2, 3, 4]
    ns = run(shell, 'b = cloud >> filter(col("intensity") > env("intensity")) >> pull(point)')
    assert ns["b"].tolist() == []


def test_vector_assignment_and_notebook_operators(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(
        shell,
        'R = xr.DataArray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),\n'
        '                 coords={"i": ["x", "y", "z"], "feature": ["x", "y", "z"]})\n'
        'out = cloud >> mutate([x, y, z] = dot(R, [x, y, z], over="feature"))\n'
        "new_x = out >> pull(x)",
    )
    assert ns["new_x"].tolist() == [48.0, 49.0, -1.0, -1.0]


def test_distance_example_in_notebook_form(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(
        shell,
        'R = xr.DataArray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),\n'
        '                 coords={"i": ["x", "y", "z"], "feature": ["x", "y", "z"]})\n'
        'offset = xr.DataArray([1.0, 0, 0], dims="i", coords={"i": ["x", "y", "z"]})\n'
        "nearby = (\n"
        "    cloud\n"
        '    >> mutate(p = dot(R, [x, y, z], over="feature") + offset)\n'
        '    >> mutate(distance = formula(r"\\sqrt{\\sum_i p_i^2}"))\n'
        "    >> filter(distance <= 50)\n"
        ")\n"
        "kept = nearby >> pull(point)",
    )
    assert ns["kept"].tolist() == [1, 2, 4]


def test_functions_modules_and_fields_keep_their_meaning(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(shell, 'ds = cloud >> select(where(is_numeric))\nn_vars = len(ds.collect().data_vars)')
    assert ns["n_vars"] == 1
    ns = run(shell, 'xx = fields("x")\nv = cloud >> mutate(q = xx * 2) >> pull(q)')
    assert ns["v"].tolist() == [0.0, 0.0, 102.0, 0.0]


def test_names_inside_functions(shell):
    run(shell, "from tidy3a import *\n" + SETUP)
    ns = run(
        shell,
        "def above(data, limit):\n"
        "    return data >> filter(intensity > limit) >> pull(point)\n"
        "kept = above(cloud, 25)",
    )
    assert ns["kept"].tolist() == [3, 4]


def test_a_notebook_defined_verb_is_left_alone(shell):
    ns = run(shell, "w = 3\ndef select(*names):\n    return names\nout = select(w)")
    assert ns["out"] == (3,)


def test_prefixed_calls_are_rewritten(shell):
    ns = run(shell, "import tidy3a as ta\n" + SETUP.replace("read_array", "ta.read_array").replace("infer", "ta.infer")
             + "xyz = cloud >> ta.select(x, y)")
    assert ns["xyz"].collect()["values"].feature.values.tolist() == ["x", "y"]


def test_source_rewrites():
    assert rewrite_vector_assignments("d >> mutate([x, y] = f(x), k = 1)") == "d >> mutate(__tidy3a_assign__([x, y], f(x)), k = 1)"
    assert rewrite_vector_assignments('mutate(s = "[a] = b")') == 'mutate(s = "[a] = b")'
    assert rewrite_vector_assignments("filter([x] == y)") == "filter([x] == y)"
    text = apply_masking("cloud >> filter(x > limit)", {"filter": __import__("tidy3a").filter})
    assert "__tidy3a_name__('x', lambda: x)" in text and "__tidy3a_name__('limit'" in text
    assert apply_masking("filter(x > 1)", {}) == "filter(x > 1)"  # not tidy3A's filter


# ── tidy3 and tidy3A in one notebook ────────────────────────────────────────


def both_shell():
    pytest.importorskip("tidy3")
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "tidy3.jupyter")
    shell.run_line_magic("load_ext", "tidy3a.jupyter")
    return shell


def close_both(shell):
    shell.run_line_magic("unload_ext", "tidy3a.jupyter")
    shell.run_line_magic("unload_ext", "tidy3.jupyter")
    InteractiveShell.clear_instance()


@pytest.mark.parametrize("first", ["tidy3", "tidy3a"])
def test_tidy3_and_tidy3a_share_a_notebook(first):
    import tidy3.masking

    if not hasattr(tidy3.masking, "_owned_by_tidy3"):
        pytest.skip("needs the tidy3 version whose notebook layer rewrites only tidy3-owned calls")
    shell = both_shell()
    try:
        tidy3_cell = (
            "from tidy3 import *\n"
            'cars = tidy({"mpg": [21.0, 30.0], "wt": [2.6, 1.9]})\n'
            "fast = (cars >> filter(mpg > 25)).collect()['mpg'].to_list()"
        )
        tidy3a_cell = "from tidy3a import *\n" + SETUP + "kept = (cloud >> filter(intensity > 15) >> pull(point)).tolist()"
        order = [tidy3_cell, tidy3a_cell] if first == "tidy3" else [tidy3a_cell, tidy3_cell]
        for cell in order + order:
            result = shell.run_cell(cell)
            assert result.success, result.error_before_exec or result.error_in_exec
        assert shell.user_ns["fast"] == [30.0]
        assert shell.user_ns["kept"] == [2, 3, 4]
    finally:
        close_both(shell)
