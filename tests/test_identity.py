"""Structural identity (plan decision 5): steps written apart that do the same work share it, and nothing else does.

A missed share costs time; a false share, two steps recorded alike that
behave differently, costs a wrong result. So every setting of every verb,
expression, key, selector, and transform must reach the structure, values
keep their types, arrays compare by identity, and a caller's function is
like no other.
"""

import copy
import dataclasses
import inspect

import numpy as np
import pytest

from tidy3a import (
    any_of,
    as_table,
    bins,
    cells,
    contains,
    ends_with,
    env,
    everything,
    fields,
    filter,
    formula,
    group_by,
    infer,
    last_col,
    matches,
    mean,
    mutate,
    n,
    num_range,
    pull,
    read_array,
    rounded,
    scan,
    select,
    slice_head,
    slice_tail,
    starts_with,
    sum,
    summarise,
    where,
)
from tidy3a import slice as take
from tidy3a.expr import AxisKey, BinOp, Call, Literal, MultiKey, NameRef, Reduce, Ref, UnaryOp, ValueKey, Vector
from tidy3a.identity import structure
from tidy3a.verbs import Verb

x, y, z, ring, d, half = fields("x", "y", "z", "ring", "d", "half")


def same(a, b):
    return a.structure() == b.structure() and hash(a.structure()) == hash(b.structure())


# ── Written apart, the same work ────────────────────────────────────────────


PIPELINES = [
    lambda: mutate(d=x * 2) >> filter(d > 0) >> group_by(ring) >> summarise(points=n(), m=mean(x)),
    lambda: mutate(d=formula(r"\sqrt{x^2 + y^2}")) >> filter(d <= 50) >> select("x", "d"),
    lambda: group_by(cells(x, 1), cells(y, 1), axis="voxel") >> summarise(points=n()),
    lambda: group_by(bins(z, [-1, 0, 1]), dropna=False) >> summarise(total=sum(x, na_rm=True, min_count=2)),
    lambda: select(starts_with("x"), ~ends_with("z")) >> take(point=[1, 2]) >> slice_head(point=1),
]


@pytest.mark.parametrize("build", PIPELINES)
def test_pipelines_written_apart_have_the_same_structure(build):
    first, second = build(), build()
    assert all(same(a, b) for a, b in zip(first.verbs, second.verbs, strict=True))


# ── Every setting reaches the structure ─────────────────────────────────────


# Two calls per named option, which must differ; the coverage test below requires every option of every verb.
OPTIONS = {
    (take, "drop"): (lambda: take(point=1), lambda: take(point=1, drop=True)),
    (group_by, "dropna"): (lambda: group_by(ring), lambda: group_by(ring, dropna=False)),
    (group_by, "axis"): (lambda: group_by(x, ring), lambda: group_by(x, ring, axis="voxel")),
    (summarise, "groups"): (lambda: summarise(m=n()), lambda: summarise(groups="keep", m=n())),
    (pull, "item"): (lambda: pull("x"), lambda: pull("y")),
    (as_table, "long"): (lambda: as_table(), lambda: as_table(long=True)),
}
VERBS = [select, filter, take, slice_head, slice_tail, mutate, group_by, summarise, pull, as_table]


def test_every_named_option_of_every_verb_is_checked():
    """A verb given a new option fails here until OPTIONS shows that the option reaches the structure."""
    for verb in VERBS:
        named = {name for name, parameter in inspect.signature(verb).parameters.items()
                 if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}
        assert named == {option for (owner, option) in OPTIONS if owner is verb}, verb.__name__


@pytest.mark.parametrize("pair", list(OPTIONS.values()), ids=[f"{v.__name__}-{o}" for v, o in OPTIONS])
def test_each_option_changes_the_structure(pair):
    first, second = (build() for build in pair)
    assert not same(first, second)


@pytest.mark.parametrize("first, second", [
    (lambda: select("x"), lambda: select("y")),
    (lambda: select("x", "y"), lambda: select("y", "x")),
    (lambda: filter(x > 0), lambda: filter(x >= 0)),
    (lambda: mutate(d=x), lambda: mutate(e=x)),
    (lambda: mutate(d=x, e=y), lambda: mutate(e=y, d=x)),  # in order: a later step can read an earlier one
    (lambda: slice_head(point=1), lambda: slice_tail(point=1)),
    (lambda: slice_head(point=1), lambda: slice_head(time=1)),
    (lambda: group_by(ring), lambda: group_by(r=ring)),
    (lambda: summarise(m=mean(x)), lambda: summarise(m=mean(x, na_rm=True))),
    (lambda: summarise(m=sum(x)), lambda: summarise(m=sum(x, min_count=1))),
    (lambda: summarise(m=mean(x)), lambda: summarise(m=mean(x, over="time"))),
], ids=lambda build: inspect.getsource(build).strip()[:40])
def test_arguments_change_the_structure(first, second):
    assert not same(first(), second())


@pytest.mark.parametrize("first, second", [
    (lambda: starts_with("x"), lambda: starts_with("x", ignore_case=False)),  # their texts are the same
    (lambda: starts_with("x"), lambda: ends_with("x")),
    (lambda: contains("x"), lambda: contains(["x", "y"])),
    (lambda: matches("x"), lambda: matches("x", ignore_case=False)),
    (lambda: num_range("x", [1, 2]), lambda: num_range("x", [1, 2], width=2)),
    (lambda: any_of(["x"]), lambda: any_of(["y"])),
    (lambda: last_col(), lambda: last_col(1)),
    (lambda: everything(), lambda: ~everything()),
])
def test_selector_settings_change_the_structure(first, second):
    assert not same(select(first()), select(second()))
    assert same(select(first()), select(first()))


def test_any_of_takes_a_generator_once():
    assert same(select(any_of(name for name in ["x", "y"])), select(any_of(["x", "y"])))


@pytest.mark.parametrize("first, second", [
    (lambda: cells(x, 1), lambda: cells(x, 2)),
    (lambda: cells(x, 1), lambda: cells(x, 1, origin=0.5)),
    (lambda: cells(x, 1), lambda: cells(y, 1)),
    (lambda: rounded(x, 1), lambda: rounded(x, 0.5)),
    (lambda: rounded(x, 1), lambda: cells(x, 1)),
    (lambda: bins(x, [0, 1, 2]), lambda: bins(x, [0, 1, 3])),
    (lambda: bins(x, [0, 1]), lambda: bins(x, [0.0, 1.0])),  # integer and float edges label differently
])
def test_key_transform_settings_change_the_structure(first, second):
    assert not same(group_by(first()), group_by(second()))
    assert same(group_by(first()), group_by(first()))


NODES = [
    Ref("x"), Ref("x", "var"), NameRef("x", 1), Literal(1), Literal(2.0, label="two"), Vector([Ref("x"), Ref("y")]),
    BinOp("+", Ref("x"), Literal(1)), UnaryOp("-", Ref("x")), Call("dot", (Ref("x"), Ref("y")), {"over": "i"}),
    Reduce("sum", Ref("x"), "time", True, 2), AxisKey("time"), ValueKey("x", Ref("x")),
    cells(x, 1), MultiKey([cells(x, 1), cells(y, 1)], "voxel", False),
]


@pytest.mark.parametrize("node", NODES, ids=lambda node: type(node).__name__)
def test_every_field_of_every_node_is_in_its_structure(node):
    """Each field changed alone changes the structure, so a field added later cannot be left out."""
    for field in dataclasses.fields(node):
        changed = copy.copy(node)
        object.__setattr__(changed, field.name, object())
        assert structure(changed) != structure(node), (type(node).__name__, field.name)


# ── Types are part of a value ───────────────────────────────────────────────


@pytest.mark.parametrize("first, second", [
    (1, 1.0), (1, True), (0, False), (1.0, np.float64(1.0)), (1.0, np.float32(1.0)), (1, np.int64(1)),
    (np.int32(1), np.int64(1)), (0.0, -0.0), ("1", 1), (1 + 0j, 1.0),
])
def test_values_of_different_types_are_apart(first, second):
    assert structure(first) != structure(second)
    assert not same(mutate(d=x * first), mutate(d=x * second))


def test_x_times_numpy_float64_is_not_x_times_a_python_float():
    """np.float64 is a subclass of float, but NumPy promotes float32 data with it and not with a Python float."""
    values = np.ones((3, 1), dtype="float32")
    frame = read_array(values, dims={"point": infer, "feature": 1}, labels={"feature": ["x"]}, field_axis="feature")
    assert (frame >> mutate(d=x * 1.0) >> pull("d")).dtype != (frame >> mutate(d=x * np.float64(1.0)) >> pull("d")).dtype
    assert not same(mutate(d=x * 1.0), mutate(d=x * np.float64(1.0)))


def test_nan_is_one_value():
    assert structure(float("nan")) == structure(float("nan"))
    assert same(filter(x != float("nan")), filter(x != float("nan")))


# ── Identity, and what is like nothing else ─────────────────────────────────


def test_arrays_compare_by_identity_not_contents():
    values = np.arange(3.0)
    assert same(mutate(d=x * env("values")), mutate(d=x * env("values")))
    assert not same(mutate(d=x * np.arange(3.0)), mutate(d=x * np.arange(3.0)))


def test_a_callers_function_is_like_no_other():
    def anything(data):
        return True

    assert not same(select(where(anything)), select(where(anything)))
    assert not same(select(where(lambda data: True)), select(where(lambda data: True)))
    from tidy3a import is_numeric

    assert same(select(where(is_numeric)), select(where(is_numeric)))
    verb = select(where(anything))
    assert same(verb, verb)  # the same verb is itself


def test_verbs_without_a_recorded_operation_are_like_no_other():
    first, second = (Verb("custom()", lambda frame: frame) for _ in range(2))
    assert not same(first, second) and same(first, first)


def test_formulas_take_the_values_their_names_pick_up():
    k = 2
    two = formula(r"k \cdot x")
    k = 3
    three = formula(r"k \cdot x")
    assert not same(mutate(d=two), mutate(d=three))
    assert same(mutate(d=formula(r"\sqrt{x}")), mutate(d=formula(r"\sqrt{x}")))  # an imported sqrt plays no part
    assert not same(mutate(d=formula(r"a \cdot x", bind={"a": 2})), mutate(d=formula(r"a \cdot x", bind={"a": 2.0})))
    unrelated = 1  # noqa: F841  a name the formula does not use
    assert same(mutate(d=formula(r"k \cdot x")), mutate(d=three))


# ── A scan shares what pipelines written apart have in common ───────────────


def counting(counts):
    """A row-wise verb that counts the blocks it runs on and changes nothing; two of them are the same step (a mutate
    with no assignments, as far as a block, and a structure, can tell)."""

    def apply(frame):
        counts.append(1)
        return frame

    return Verb("counting()", apply, operation="mutate", arguments=[])


def test_a_scan_shares_the_steps_of_pipelines_written_apart():
    rng = np.random.default_rng(0)
    values = (rng.normal(size=(20_000, 5)) * 30).astype("float32")
    values[:, 4] = rng.integers(0, 40, len(values))
    options = dict(dims={"point": infer, "feature": 5}, labels={"feature": ["x", "y", "z", "intensity", "ring"]},
                   field_axis="feature")
    counts = []
    frame = read_array(values, blocks=3_000, **options)
    near = mutate(d=formula(r"\sqrt{x^2 + y^2 + z^2}"))
    rings = frame >> near >> counting(counts) >> filter(d <= 60) >> group_by(ring) >> summarise(points=n())
    grid = (frame >> mutate(d=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> counting(counts) >> filter(d <= 60)
            >> group_by(cells(x, 20.0), cells(y, 20.0), axis="cell") >> summarise(points=n()))
    assert counts == [1, 1]  # each on the first block as it was written
    got_rings, got_grid = scan(rings, grid)
    assert counts == [1] * (2 + 7)  # then once per block for both
    eager = read_array(values, **options) >> near >> filter(d <= 60)
    assert (got_rings >> pull("points")).tolist() == (eager >> group_by(ring) >> summarise(points=n())
                                                      >> pull("points")).tolist()
    assert (got_grid >> pull("points")).tolist() == (eager >> group_by(cells(x, 20.0), cells(y, 20.0), axis="cell")
                                                     >> summarise(points=n()) >> pull("points")).tolist()


def test_a_step_like_no_other_is_not_shared_and_neither_is_what_follows():
    values = np.ones((10, 2), dtype="float32")
    options = dict(dims={"point": infer, "feature": 2}, labels={"feature": ["x", "y"]}, field_axis="feature")
    counts = []
    frame = read_array(values, blocks=5, **options)
    first = frame >> select(where(lambda data: True)) >> counting(counts) >> summarise(points=n())
    second = frame >> select(where(lambda data: True)) >> counting(counts) >> summarise(points=n())
    scan(first, second)
    assert counts == [1] * (2 + 2 * 2)  # the lambdas differ, so both branches run on both blocks
