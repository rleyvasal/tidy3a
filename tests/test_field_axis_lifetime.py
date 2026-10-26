"""A field axis lasts as long as the axis does.

When a verb removes the field axis (a summary of each field over every
observation axis, or keeping only variables without it), the result no
longer names it: ``explain()`` says so, and a new field axis can be declared.
A verb that keeps the axis keeps it as the field axis.
"""

import numpy as np
import pytest

from tidy3a import (
    TidyError,
    col,
    fields,
    group_by,
    infer,
    mean,
    n,
    read_array,
    select,
    summarise,
    tidy,
)

x, intensity, ring = fields("x", "intensity", "ring")


def cloud():
    values = np.array([[1, 10, 0], [2, 20, 0], [3, 30, 1]], dtype="float32")
    return read_array(values, dims={"point": infer, "feature": 3}, labels={"feature": ["x", "intensity", "ring"]},
                      field_axis="feature")


def test_summarising_away_the_field_axis_ends_it():
    out = cloud() >> group_by(ring) >> summarise(points=n(), m=mean(intensity))
    assert out._field_axis is None
    assert "field axis: feature → none" in str(out.explain())


def test_a_new_field_axis_can_be_declared_after():
    out = cloud() >> group_by(ring) >> summarise(points=n(), m=mean(intensity))
    stacked = tidy(out.to_xarray().to_dataarray("stat"), field_axis="stat")
    assert stacked._field_axis == "stat"
    with pytest.raises(TidyError, match="not an axis"):
        tidy(out, field_axis="feature")  # it is gone, so naming it is an error, not "already has"


def test_a_verb_that_keeps_the_axis_keeps_the_field_axis():
    per_field = cloud() >> summarise(m=mean(col("values")))
    assert per_field._field_axis == "feature" and "feature" in per_field._backend.sizes(per_field._data)
    assert (cloud() >> select(x, intensity))._field_axis == "feature"
