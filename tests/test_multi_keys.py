"""Grouping by several value keys at once (plan milestone 10, multi-value keys).

``group_by(cells(x, 1), cells(y, 1), cells(z, 1))`` makes one group per
combination that occurs, as dplyr, pandas, and Polars do: the groups lie
along one new axis, ``group`` unless ``axis=`` names it, sorted by the first
key, then the next; ``summarise`` gives each key's value per group as a
variable named after the key, before the summaries. Empty combinations are
never made, so voxels cost what is occupied, not the bounding box. pandas'
``groupby`` on the same keys is the reference.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidy3a import (
    GroupError,
    SummariseError,
    cells,
    col,
    fields,
    group_by,
    infer,
    mean,
    mutate,
    n,
    pull,
    read_array,
    rounded,
    select,
    summarise,
    tidy,
)

x, y, z, intensity = fields("x", "y", "z", "intensity")


def scan(size=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.column_stack([
        rng.uniform(0, 12, size), rng.uniform(0, 3, size), rng.uniform(0, 2, size), rng.uniform(0, 255, size),
    ]).astype("float32")
    return values, read_array(values, dims={"point": infer, "feature": 4},
                              labels={"feature": ["x", "y", "z", "intensity"]}, field_axis="feature")


def voxels(frame, **options):
    return frame >> group_by(cells(x, 1), cells(y, 1), cells(z, 1), **options)


# ── One row per occupied combination ────────────────────────────────────────


def test_one_group_per_occupied_voxel_with_the_keys_as_variables():
    values, frame = scan()
    out = voxels(frame) >> summarise(points=n(), m=mean(intensity))
    table = pd.DataFrame(values, columns=["x", "y", "z", "i"])
    cells_of = np.floor(table[["x", "y", "z"]]).astype(int)
    want = table.assign(cx=cells_of.x, cy=cells_of.y, cz=cells_of.z).groupby(["cx", "cy", "cz"]).i.agg(["size", "mean"])
    assert out._backend.variable_names(out._data) == ["x_cell", "y_cell", "z_cell", "points", "m"]
    assert (out >> pull("points")).tolist() == want["size"].tolist()
    np.testing.assert_allclose(out >> pull("m"), want["mean"].to_numpy(), rtol=1e-6)
    first = want.index[0]
    assert (out >> pull("x_cell"))[0] == f"[{first[0]}, {first[0] + 1})"
    assert out._backend.sizes(out._data)["group"] == len(want)


def test_groups_sort_by_each_key_in_turn_in_the_keys_own_order():
    data = tidy(xr.Dataset({"x": ("p", [10.5, 2.5, 2.5, 10.5]), "y": ("p", [0.5, 1.5, 0.5, 0.5]),
                            "v": ("p", [1.0, 2.0, 3.0, 4.0])}))
    out = data >> group_by(cells(col("x"), 1), cells(col("y"), 1)) >> summarise(s=mean(col("v")))
    # Cells sort as numbers ([2, 3) before [10, 11)), not as their text.
    assert (out >> pull("x_cell")).tolist() == ["[2, 3)", "[2, 3)", "[10, 11)"]
    assert (out >> pull("y_cell")).tolist() == ["[0, 1)", "[1, 2)", "[0, 1)"]
    assert (out >> pull("s")).tolist() == [3.0, 2.0, 2.5]


def test_plain_value_keys_combine_too():
    data = tidy(xr.Dataset({"ring": ("p", [2, 1, 2, 1, 2]), "kind": ("p", ["b", "a", "a", "a", "b"]),
                            "v": ("p", [1.0, 2.0, 3.0, 4.0, 5.0])}))
    out = data >> group_by(col("ring"), col("kind")) >> summarise(points=n(), m=mean(col("v")))
    assert (out >> pull("ring")).tolist() == [1, 2, 2]
    assert (out >> pull("kind")).tolist() == ["a", "a", "b"]
    assert (out >> pull("points")).tolist() == [2, 1, 2]
    assert (out >> pull("m")).tolist() == [3.0, 3.0, 3.0]


def test_axis_names_the_groups():
    _, frame = scan(200)
    out = voxels(frame, axis="voxel") >> summarise(points=n())
    assert "voxel" in out._backend.sizes(out._data) and "group" not in out._backend.sizes(out._data)
    assert 'axis = "voxel"' in str(voxels(frame, axis="voxel").explain())


def test_grouped_mutate_works_per_combination():
    values, frame = scan(500)
    out = voxels(frame) >> mutate(centred=intensity - mean(intensity))
    table = pd.DataFrame(values, columns=["x", "y", "z", "i"])
    keys = [np.floor(table[c]).astype(int) for c in "xyz"]
    want = table.i - table.i.groupby(keys).transform("mean")
    np.testing.assert_allclose(out >> pull("centred"), want.to_numpy(), rtol=1e-5, atol=1e-4)


def test_an_axis_key_is_kept_alongside_the_combinations():
    data = tidy(xr.Dataset(
        {"v": (("time", "p"), np.arange(8.0).reshape(2, 4)), "a": ("p", [1, 1, 2, 2]), "b": ("p", [1, 2, 1, 1])},
        coords={"time": [10, 20]},
    ))
    out = data >> group_by("time", col("a"), col("b")) >> summarise(m=mean(col("v")))
    assert dict(out._backend.sizes(out._data)) == {"group": 3, "time": 2}
    np.testing.assert_array_equal(out >> pull("m"), np.array([[0.0, 4.0], [1.0, 5.0], [2.5, 6.5]]))  # (group, time)


def test_the_groups_line_and_explain_show_the_keys():
    _, frame = scan(200)
    grouped = voxels(frame)
    assert "groups: cells(x), cells(y), cells(z)" in repr(grouped)
    assert "groups: cells(x), cells(y), cells(z) (" in str(grouped.explain())


# ── Missing keys ────────────────────────────────────────────────────────────


def test_dropna_leaves_out_positions_missing_any_key_with_one_note():
    data = tidy(xr.Dataset({"a": ("p", [1.0, np.nan, 2.0, 1.0]), "b": ("p", [1.0, 1.0, np.nan, 1.0]),
                            "v": ("p", [1.0, 2.0, 3.0, 4.0])}))
    out = data >> group_by(col("a"), col("b")) >> summarise(m=mean(col("v")))
    assert (out >> pull("a")).tolist() == [1.0] and (out >> pull("m")).tolist() == [2.5]
    assert "changed: group_by(a, b) dropped 2 positions with a missing a or b; dropna=False keeps them as groups" in repr(out)


def test_dropna_false_keeps_missing_values_as_groups_sorted_last():
    data = tidy(xr.Dataset({"a": ("p", [1.0, np.nan, 2.0, 1.0]), "b": ("p", [1.0, 1.0, np.nan, 1.0]),
                            "v": ("p", [1.0, 2.0, 3.0, 4.0])}))
    out = data >> group_by(col("a"), col("b"), dropna=False) >> summarise(m=mean(col("v")))
    a, b = out >> pull("a"), out >> pull("b")
    assert a[0] == 1.0 and a[1] == 2.0 and np.isnan(a[2])
    assert b[0] == 1.0 and np.isnan(b[1]) and b[2] == 1.0
    assert (out >> pull("m")).tolist() == [2.5, 3.0, 2.0]


# ── What summarise and select do with the keys ──────────────────────────────


def test_summarise_ends_the_grouping_unless_asked_to_keep_it():
    _, frame = scan(200)
    assert "groups:" not in repr(voxels(frame) >> summarise(points=n()))
    kept = voxels(frame) >> summarise(points=n(), groups="keep")
    assert "groups: group" in repr(kept)


def test_select_keeps_the_key_variables():
    data = tidy(xr.Dataset({"a": ("p", [1, 2]), "b": ("p", [3, 4]), "v": ("p", [1.0, 2.0])}))
    picked = data >> group_by(col("a"), col("b")) >> select(col("v"))
    assert set(picked._backend.variable_names(picked._data)) == {"a", "b", "v"}


# ── Errors ──────────────────────────────────────────────────────────────────


def test_axis_needs_several_value_keys():
    _, frame = scan(50)
    with pytest.raises(GroupError, match=r"axis= names the axis of several value keys; cells\(x\) groups along its own axis, x_cell"):
        frame >> group_by(cells(x, 1), axis="voxel")


def test_keys_must_vary_along_one_axis():
    data = tidy(xr.Dataset({"a": ("p", [1, 2]), "b": ("t", [3, 4, 5])}))
    with pytest.raises(GroupError, match=r"group_by\(a, b\) needs keys along one axis; a varies along p, b along t"):
        data >> group_by(col("a"), col("b")) >> summarise(k=n())


def test_the_axis_name_must_be_free():
    data = tidy(xr.Dataset({"a": ("p", [1, 2]), "b": ("p", [3, 4]), "v": (("p", "t"), np.zeros((2, 2)))}))
    with pytest.raises(GroupError, match='axis="t" is already an axis of this data'):
        data >> group_by(col("a"), col("b"), axis="t") >> summarise(k=n())
    with pytest.raises(GroupError, match='a key is named "a" too'):
        data >> group_by(col("a"), col("b"), axis="a") >> summarise(k=n())


def test_keys_need_different_names_and_summaries_other_names():
    data = tidy(xr.Dataset({"x": ("p", [0.2, 1.4]), "v": ("p", [1.0, 2.0])}))
    with pytest.raises(GroupError, match="two keys named x_cell"):
        data >> group_by(cells(col("x"), 1), cells(col("x"), 0.5)) >> summarise(k=n())
    with pytest.raises(SummariseError, match='"x" is a key of the groups'):
        data >> group_by(col("x"), rounded(col("v"), 1)) >> summarise(x=n())
