"""Recorded plans, explain(), and reusable pipelines (plan milestone 10, decision 10, stage a).

Each verb runs at once, as before, and records a step on its result: the
verb, and the result's variables, axes, groups, and notes. ``explain()``
lists the source and each step with its effect, computed from those records
without running anything. Verbs compose before any data into a pipeline
(``mutate(...) >> filter(...)``) that runs with ``frame >> steps`` and
previews with ``steps.explain(frame, head=...)``; ``result.plan()`` gives
the recorded steps back as a pipeline to run on other data.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    Pipeline,
    col,
    fields,
    filter,
    group_by,
    infer,
    mean,
    mutate,
    n,
    pull,
    read_array,
    select,
    summarise,
    tidy,
)

x, y, z, intensity, ring, d = fields("x", "y", "z", "intensity", "ring", "d")
POINTS = np.array([[3, 4, 0, 10, 1], [0, 0, 50, 20, 2], [60, 0, 0, 30, 1], [1, 1, 1, 40, 2]], dtype="float32")


def cloud():
    return read_array(POINTS, dims={"point": infer, "feature": 5},
                      labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")


def steps():
    return mutate(d=x + y + z) >> filter(d <= 50) >> group_by(ring) >> summarise(points=n(), m=mean(intensity))


# ── explain() ───────────────────────────────────────────────────────────────


def test_explain_lists_the_source_and_each_step_with_its_effect():
    add, keep, by, total = mutate(d=x + y + z), filter(d <= 50), group_by(ring), summarise(points=n(), m=mean(intensity))
    result = cloud() >> add >> keep >> by >> total
    assert str(result.explain()) == "\n".join([
        "<plan source: NumPy array>",
        "  values (point: 4, feature: 5) float32",
        f"1. {add.text}",
        "     + d (point: 4) float32",
        f"2. {keep.text}",
        "     point: 4 → 3",
        f"3. {by.text}",
        "     groups: ring (2)",
        f"4. {total.text}",
        "     - axis point",
        "     - axis feature",
        "     + axis ring: 2",
        "     - values",
        "     - d",
        "     + points (ring: 2) int64",
        "     + m (ring: 2) float32",
        "     groups: none",
    ])


def test_notes_appear_under_the_step_that_made_them():
    data = tidy(xr.Dataset({"v": ("p", [1.0, 2.0, 3.0]), "k": ("p", [1.0, np.nan, 2.0])}))
    result = data >> group_by(col("k")) >> summarise(m=mean(col("v")))
    text = str(result.explain())
    first, second = text.split("\n2. ")
    assert "dropped 1 position with a missing k" in first and "dropped" not in second


def test_a_step_that_changes_nothing_says_so():
    result = cloud() >> select(x, y, z, intensity, ring)
    assert str(result.explain()).endswith("(no change)")


def test_a_fresh_frame_has_no_steps():
    assert str(cloud().explain()).splitlines()[-1] == "(no steps)"


def test_explain_runs_nothing(monkeypatch):
    result = cloud() >> steps()
    backend = type(result._backend)
    monkeypatch.setattr(backend, "apply", lambda *a, **k: pytest.fail("explain() computed something"))
    monkeypatch.setattr(backend, "group_reduce", lambda *a, **k: pytest.fail("explain() computed something"))
    assert "4. summarise" in str(result.explain())


def test_explain_shows_as_a_value_and_prints_the_same():
    plan = (cloud() >> mutate(d=x * 2)).explain()
    assert repr(plan) == str(plan)


def test_the_inspection_output_is_unchanged():
    result = cloud() >> mutate(d=x * 2)
    assert "plan" not in repr(result) and "1. mutate" not in repr(result)


# ── Pipelines ───────────────────────────────────────────────────────────────


def test_verbs_compose_into_a_pipeline_that_runs_as_the_verbs_would():
    pipeline = steps()
    assert isinstance(pipeline, Pipeline) and len(pipeline) == 4
    by_steps = (cloud() >> pipeline).collect()
    by_verbs = (cloud() >> mutate(d=x + y + z) >> filter(d <= 50) >> group_by(ring)
                >> summarise(points=n(), m=mean(intensity))).collect()
    xr.testing.assert_identical(by_steps, by_verbs)


def test_pipelines_join_flat_and_print_their_steps():
    first = mutate(d=x * 2) >> filter(d > 0)
    joined = first >> (group_by(ring) >> summarise(m=mean(d)))
    assert len(joined) == 4
    assert repr(first) == f"<pipeline {first.verbs[0].text} >> {first.verbs[1].text}>"


def test_a_pipeline_can_end_in_a_value():
    values = cloud() >> (filter(x < 10) >> pull("x"))
    np.testing.assert_array_equal(values, np.array([3, 0, 1], "float32"))


def test_only_verbs_compose():
    with pytest.raises(TypeError, match="verbs and pipelines"):
        mutate(d=x * 2) >> 3


def test_a_pipeline_previews_on_the_first_positions_without_changing_the_data():
    data = cloud()
    plan = steps().explain(data, head=3)
    lines = str(plan).splitlines()
    assert lines[0] == "<plan source: NumPy array; previewed on the first 3 of 4 positions of point>"
    assert lines[1] == "  values (point: 3, feature: 5) float32"
    assert "     point: 3 → 2" in lines  # the third point is 60 m out
    assert "slice" not in str(plan)
    assert (data >> pull("point")).tolist() == [1, 2, 3, 4]


def test_a_preview_of_a_pipeline_ending_in_a_value_says_what_it_gives():
    plan = (filter(x < 10) >> pull("x")).explain(cloud())
    assert str(plan).splitlines()[-1] == "     → NumPy array (3,) float32"


# ── plan() ──────────────────────────────────────────────────────────────────


def test_a_results_plan_runs_again_on_other_data():
    nearby = cloud() >> mutate(d=x + y + z) >> filter(d <= 50)
    plan = nearby.plan()
    assert isinstance(plan, Pipeline) and len(plan) == 2
    other = read_array(POINTS[::-1].copy(), dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    again = other >> plan
    expected = other >> mutate(d=x + y + z) >> filter(d <= 50)
    xr.testing.assert_identical(again.collect(), expected.collect())
    assert str(again.explain()).splitlines()[2:] == str(expected.explain()).splitlines()[2:]


def test_a_fresh_frames_plan_changes_nothing():
    data = cloud()
    assert len(data.plan()) == 0
    xr.testing.assert_identical((data >> data.plan()).collect(), data.collect())


def test_the_plan_survives_a_change_of_backend_and_field_axis():
    result = cloud() >> mutate(d=x * 2)
    moved = tidy(result, backend="xarray")
    assert str(moved.explain()).splitlines()[2:] == str(result.explain()).splitlines()[2:]
