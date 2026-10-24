import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    FormulaError,
    RefError,
    col,
    dot,
    fields,
    filter,
    formula,
    infer,
    mutate,
    pull,
    read_array,
    sqrt,
    sum,
    tidy,
)

x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
AXES = ["x", "y", "z"]
FIELDS = ["x", "y", "z", "intensity", "ring"]


def cloud():
    values = np.array(
        [[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float64",
    )
    return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, field_axis="feature")


def rotated():
    R = xr.DataArray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),
                     coords={"i": AXES, "feature": AXES})
    offset = xr.DataArray([1.0, 0, 0], dims="i", coords={"i": AXES})
    return cloud() >> mutate(p = dot(R, [x, y, z], over="feature") + offset)


def values(frame, name):
    return frame.collect()[name]


# ── plain text and LaTeX agree with native expressions ──────────────────────


@pytest.mark.parametrize("text", ["intensity^2", "intensity**2", r"\mathrm{intensity}^2", r"\mathrm{intensity}^{2}"])
def test_power_matches_native(text):
    a = values(cloud() >> mutate(q = formula(text)), "q")
    b = values(cloud() >> mutate(q = intensity ** 2), "q")
    xr.testing.assert_identical(a, b)


def test_plain_text_operators_and_functions():
    out = values(cloud() >> mutate(q = formula("sqrt(intensity) + 2 * ring - abs(y) / 4")), "q")
    expected = values(cloud() >> mutate(q = sqrt(intensity) + 2 * ring - abs(y) / 4), "q")
    xr.testing.assert_allclose(out, expected)


def test_distance_formula_matches_native_reduction_and_filters():
    native = values(rotated() >> mutate(d = sqrt(sum(col("p") ** 2, over="i"))), "d")
    via_formula = values(rotated() >> mutate(d = formula(r"\sqrt{\sum_i p_i^2}")), "d")
    xr.testing.assert_allclose(native, via_formula)
    nearby = rotated() >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}")) >> filter(distance <= 50)
    assert (nearby >> pull("point")).tolist() == [1, 2, 4]


def test_index_notation_rotation_matches_dot():
    R = xr.DataArray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dims=("i", "feature"),
                     coords={"i": AXES, "feature": AXES})
    native = values(cloud() >> mutate(p = dot(R, [x, y, z], over="feature")), "p")
    for text in (r"p_i = \sum_j R_{ij} v_j", r"p_i = \sum_j R_{i,j} v_j"):
        out = values(cloud() >> mutate(formula(text, bind={"j": "feature", "v": [x, y, z]})), "p")
        xr.testing.assert_allclose(out.transpose(*native.dims), native)


def test_text_strings_in_mutate_stay_literals():
    assert values(cloud() >> mutate(note = "x^2"), "note").item() == "x^2"


# ── conventions shared with plot3 ───────────────────────────────────────────


def frame_xy():
    ds = xr.Dataset({"x": ("t", [0.5, 1.0]), "y": ("t", [2.0, 3.0])})
    return tidy(ds)


def test_letters_side_by_side_multiply():
    out = values(frame_xy() >> mutate(q = formula(r"$xy$")), "q")
    assert out.values.tolist() == [1.0, 3.0]
    assert "xy → x · y" in repr(frame_xy() >> mutate(q = formula(r"$xy$")))


def test_letter_run_matching_a_name_is_an_error():
    ds = xr.Dataset({"x": ("t", [1.0]), "y": ("t", [2.0]), "xy": ("t", [5.0])})
    with pytest.raises(FormulaError, match=r"\\mathrm\{xy\}"):
        tidy(ds) >> mutate(q = formula(r"$xy$"))
    assert values(tidy(ds) >> mutate(q = formula(r"\mathrm{xy} + x \cdot y")), "q").item() == 7.0


def test_function_conventions():
    f = frame_xy()
    xs = np.array([0.5, 1.0])
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\sin x^2")), "q"), np.sin(xs ** 2))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\sin^2 x")), "q"), np.sin(xs) ** 2)
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\sin^{-1} x")), "q"), np.arcsin(xs))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"e^{x}")), "q"), np.exp(xs))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\log_{10} y")), "q"), np.log10([2.0, 3.0]))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\ln y")), "q"), np.log([2.0, 3.0]))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\frac{y}{x}")), "q"), [4.0, 3.0])
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\sqrt[3]{y}")), "q"), np.cbrt([2.0, 3.0]))
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\left| x - y \right|")), "q"), [1.5, 2.0])
    np.testing.assert_allclose(values(f >> mutate(q = formula(r"\pi x")), "q"), np.pi * xs)


def test_log_with_other_base_is_unsupported():
    with pytest.raises(FormulaError, match="other than 10"):
        formula(r"\log_2 x")


def test_unsupported_commands_and_eaten_backslashes():
    with pytest.raises(FormulaError, match=r"\\int isn't supported in formula"):
        formula(r"\int x")
    with pytest.raises(FormulaError, match="raw string"):
        formula("\frac{1}{x}")


def test_e_is_a_constant_unless_it_is_a_name():
    ds = xr.Dataset({"e": ("t", [2.0]), "x": ("t", [1.0])})
    out = values(tidy(ds) >> mutate(q = formula(r"e^{x}")), "q")
    assert out.item() == 2.0  # data e to the power x
    assert values(tidy(ds) >> mutate(q = formula(r"\mathrm{e}^{x}")), "q").item() == pytest.approx(np.e)


def test_plain_i_is_an_index_and_mathrm_i_is_imaginary():
    out = values(frame_xy() >> mutate(q = formula(r"x + \mathrm{i} y")), "q")
    assert out.values.tolist() == [0.5 + 2j, 1.0 + 3j]


# ── subscripts and binding ──────────────────────────────────────────────────


def test_label_and_position_subscripts():
    frame = rotated()
    horizontal = values(frame >> mutate(h = formula(r"\sqrt{p_x^2 + p_y^2}")), "h")
    p_data = values(frame, "p")  # not named p: a Python p would collide with the data p
    np.testing.assert_allclose(horizontal, np.hypot(p_data.sel(i="x"), p_data.sel(i="y")))
    first = values(frame >> mutate(f = formula("$p_1$")), "f")
    xr.testing.assert_allclose(first, p_data.sel(i="x", drop=True))


def test_plain_text_subscripts_are_names_as_in_plot3():
    ds = xr.Dataset({"p_1": ("t", [4.0])})
    assert values(tidy(ds) >> mutate(q = formula("p_1 * 2")), "q").item() == 8.0


def test_underscore_names_use_escaped_mathrm():
    ds = xr.Dataset({"x_1": ("t", [3.0])})
    assert values(tidy(ds) >> mutate(q = formula(r"\mathrm{x\_1}^2")), "q").item() == 9.0


def test_unknown_subscript_lists_axes():
    with pytest.raises(FormulaError, match="axes: i, point|axes: point, i"):
        rotated() >> mutate(q = formula(r"$p_w$"))


def test_unbound_index_letter_explains_binding():
    R = xr.DataArray(np.eye(3), dims=("i", "feature"), coords={"i": AXES, "feature": AXES})
    with pytest.raises(FormulaError, match='bind=\\{"j"'):
        cloud() >> mutate(formula(r"p_i = \sum_j R_{ij} v_j", bind={"v": [x, y, z]}))
    del R


def test_bounds_are_checked_one_based_positions():
    frame = rotated()
    p_data = values(frame, "p")
    part = values(frame >> mutate(s = formula(r"\sum_{i=1}^{2} p_i^2")), "s")
    np.testing.assert_allclose(part, (p_data.sel(i=["x", "y"]) ** 2).sum("i"))
    assert "positions 1–2 of 3" in repr(frame >> mutate(s = formula(r"\sum_{i=1}^{2} p_i^2")))
    with pytest.raises(FormulaError, match="outside i"):
        frame >> mutate(s = formula(r"\sum_{i=1}^{4} p_i^2"))
    with pytest.raises(FormulaError, match="not bound to an integer"):
        frame >> mutate(s = formula(r"\sum_{i=1}^{n} p_i^2"))
    whole = values(frame >> mutate(s = formula(r"\sum_{i=1}^{n} p_i^2", bind={"n": 3})), "s")
    np.testing.assert_allclose(whole, (p_data ** 2).sum("i"))


def test_sum_needs_its_index_in_the_body():
    with pytest.raises(FormulaError, match="no subscript in its body uses k"):
        rotated() >> mutate(s = formula(r"\sum_k p_i"))


def test_python_values_and_collisions():
    gain = 2.0
    assert values(cloud() >> mutate(q = formula(r"\mathrm{gain} \cdot \mathrm{intensity}")), "q").values.tolist() == [20.0, 40.0, 60.0, 80.0]
    intensity_value = 5.0
    ds = xr.Dataset({"intensity_value": ("t", [1.0])})
    with pytest.raises(RefError, match="both data and a Python value"):
        tidy(ds) >> mutate(q = formula(r"\mathrm{intensity\_value}"))
    del gain, intensity_value


def test_references_from_fields_are_data_not_collisions():
    # x and y are defined above with fields(); they still mean the data.
    assert values(frame_xy() >> mutate(q = formula("x + y")), "q").values.tolist() == [2.5, 4.0]


def test_unknown_name():
    with pytest.raises(RefError, match="not data, a bound value, or a Python value"):
        frame_xy() >> mutate(q = formula(r"\mathrm{nothing} + 1"))


# ── show-back ───────────────────────────────────────────────────────────────


def test_formula_shows_how_it_was_read():
    f = formula(r"\sqrt{\sum_i p_i^2}")
    assert "summed index letters: i" in repr(f)
    assert f._repr_latex_() == r"$\sqrt{\sum_i p_i^2}$"
    text = repr(rotated() >> mutate(d = f))
    assert 'i → axis "i" of p (labels x  y  z), summed' in text
    assert "result: (point)" in text
    assert 'as array code: sqrt(sum(p ** 2, over="i"))' in text


def test_left_side_must_match_the_right():
    with pytest.raises(FormulaError, match="left side indexes k"):
        rotated() >> mutate(formula(r"q_k = \sum_i p_i"))
