"""Summary tables combined in a tree (plan milestone 15): the same as merging the blocks one by one, in order.

Each block's summary is made a table of its own, and the tables are combined
in several tree shapes. What adds exactly (counts, integers, extremes,
flags, the median's counts, the float32 sums' counts and terms) is the same
in every shape; what adds in float64 (float sums, a float32 sum's estimate,
the spreads' moments) is the same within rounding; and the float32 sums an
estimate settles are the same bits.
"""

import copy

import numpy as np
import pytest

from tidy3a import (all, any, cells, chunked, fields, group_by, max, mean, median, min, n, quantile, std, sum,
                    summarise, tidy, var)

v32, v64, i32, g, c, m = fields("v32", "v64", "i32", "g", "c", "m")
SUMMARIES = summarise(k=n(), s32=sum(v32, na_rm=True), m32=mean(v32), s64=sum(v64), mi=mean(i32), si=sum(i32),
                      lo=min(v32), hi=max(i32), sd=std(v64), va=var(v32), yes=any(v32 > 50),
                      every=all(i32 > -900), mid=median(v64, na_rm=True), q=quantile(i32, 0.3))
KEYS = {
    "a value": group_by(g),
    "cells whose range grows from block to block": group_by(cells(c, 5)),
    "a key with missing values, kept": group_by(m, dropna=False),
    "a key with missing values, dropped": group_by(m),
    "two keys": group_by(g, cells(c, 25)),
    "no key": None,
}
EXACT = {"n", "int", "flag", "extreme", "rank"}  # kinds whose pieces add exactly, by name prefix


def data(count=3_000):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(11)
    v32 = (rng.normal(size=count) * 100).astype(np.float32)
    v32[rng.random(count) < 0.03] = np.nan
    v64 = rng.normal(size=count) * 1e3
    m = rng.integers(0, 4, count).astype(float)
    m[rng.random(count) < 0.05] = np.nan
    return xr.Dataset({"v32": ("point", v32), "v64": ("point", v64),
                       "i32": ("point", rng.integers(-1_000, 1_000, count).astype(np.int32)),
                       "g": ("point", rng.integers(0, 9, count)), "c": ("point", np.arange(count) / 10.0),
                       "m": ("point", m)})


def partials(frame):
    """Each block's summary (``_partial``), as the driver makes it."""
    blocks = frame._layout()
    names = chunked._needs(frame)
    steps = [verb for verb in frame._steps if verb._operation in chunked.ROW_WISE]
    found = []
    for span in blocks.every:
        rows = blocks.load(*span, names)
        for verb in steps:
            rows = rows >> verb
        found.append(chunked._partial(frame._plan, rows, blocks.axis))
    return found


def table_of(plan, partial):
    table = chunked._Table(plan)
    table.merge(*copy.deepcopy(partial))
    return table


def combined(plan, found, shape, rng):
    """The blocks' tables combined in a tree of *shape*: pairs by number, left to right, right to left, or random."""
    tables = [table_of(plan, partial) for partial in found]
    while len(tables) > 1:
        if shape == "pairs":
            tables = [tables[at] if at + 1 == len(tables) else _two(tables[at], tables[at + 1])
                      for at in range(0, len(tables), 2)]
        elif shape == "left":
            tables = [_two(tables[0], tables[1]), *tables[2:]]
        elif shape == "right":
            tables = [*tables[:-2], _two(tables[-2], tables[-1])]
        else:
            at = int(rng.integers(0, len(tables) - 1))
            tables = [*tables[:at], _two(tables[at], tables[at + 1]), *tables[at + 2:]]
    return tables[0]


def same_labels(a, b):
    """The same labels in the same order, a missing one (NaN) equal to another."""
    return len(a) == len(b) and all(x == y or (x != x and y != y) for x, y in zip(a.tolist(), b.tolist()))


def _two(first, second):
    first.combine(second)
    return first


@pytest.mark.parametrize("key", list(KEYS))
@pytest.mark.parametrize("rows", [1, 97, 1_000])
def test_tables_combined_in_any_tree_are_the_blocks_merged_in_order(key, rows):
    frame = tidy(data(400 if rows == 1 else 3_000), blocks={"point": rows})  # blocks of one: a block a value
    frame = frame >> KEYS[key] >> SUMMARIES if KEYS[key] is not None else frame >> SUMMARIES
    plan = frame._plan
    found = partials(frame)
    in_order = chunked._Table(plan)
    for partial in copy.deepcopy(found):
        in_order.merge(*partial)
    rng = np.random.default_rng(0)
    for shape in ("pairs", "left", "right", "random", "random"):
        tree = combined(plan, found, shape, rng)
        assert np.array_equal(tree.keys, in_order.keys), shape
        assert np.array_equal(tree.counts, in_order.counts) and tree.counts.dtype == in_order.counts.dtype
        assert (tree.blocks, tree.dropped) == (in_order.blocks, in_order.dropped)
        for mine, theirs in zip(tree.parts, in_order.parts):
            assert same_labels(mine.full(), theirs.full()) and mine.dropped == theirs.dropped
        for hidden, (kind, held) in in_order.pieces.items():
            got_kind, got = tree.pieces[hidden]
            assert got_kind == kind and got.dtype == held.dtype and got.shape == held.shape, (shape, hidden)
            if kind.split(":")[0] in EXACT or held.dtype.kind in "biu":
                assert got.tobytes() == held.tobytes(), (shape, hidden, kind)
            elif kind == "estimate":
                assert np.array_equal(got[2:], held[2:], equal_nan=True), (shape, hidden)  # counts and terms
                assert np.allclose(got[:2], held[:2], rtol=1e-12, atol=1e-9, equal_nan=True), (shape, hidden)
            else:
                assert np.allclose(got, held, rtol=1e-12, atol=1e-6, equal_nan=True), (shape, hidden, kind)
        settled, settled_tree = in_order.settle()[0], tree.settle()[0]
        for hidden, node in plan.reductions.items():  # the float32 sums an estimate settles: the same bits
            if node.op in ("sum", "mean") and in_order.pieces[hidden][0] == "estimate":
                a, b = settled[hidden], settled_tree[hidden]
                assert a.tobytes() == b.tobytes(), (shape, hidden)


def test_combining_an_empty_table_changes_nothing():
    frame = tidy(data(), blocks={"point": 500}) >> group_by(g) >> SUMMARIES
    found = partials(frame)
    table = table_of(frame._plan, found[0])
    before = (table.keys.copy(), table.counts.copy(), {k: v[1].copy() for k, v in table.pieces.items()})
    table.combine(chunked._Table(frame._plan))
    assert np.array_equal(table.keys, before[0]) and np.array_equal(table.counts, before[1])
    assert all(np.array_equal(held, before[2][hidden], equal_nan=True) for hidden, (_, held) in table.pieces.items())
    empty = chunked._Table(frame._plan)
    empty.combine(table_of(frame._plan, found[0]))
    assert np.array_equal(empty.keys, table.keys) and np.array_equal(empty.counts, table.counts)
