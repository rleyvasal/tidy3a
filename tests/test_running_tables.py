"""A thread's running table (plan milestone 20, step 2): where every reduction's result is free of the order of its
additions, each thread's compiled blocks add into one dense table, widened as a block's keys go beyond it, and the
threads' tables merge once, densely; the same bits on any number of threads, as NumPy's blocks and memory give."""

import numpy as np
import pytest

from tidy3a import (all, any, bins, cells, fields, filter, formula, group_by, if_else, infer, max, mean, median, min,
                    mutate, n, nan, read_array, std, sum, summarise, var)
from tidy3a.fused import driver

pytest.importorskip("numba")
names = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring, distance = fields(*names, "distance")
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature")
NEAR = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60)
ORDER_FREE = {
    "voxels": NEAR >> group_by(cells(x, 10), cells(y, 10), cells(z, 10), axis="voxel")
    >> summarise(k=n(), m=mean(intensity), top=max(intensity), low=min(y)),
    "rings, every kind": group_by(ring) >> summarise(k=n(), m=mean(intensity), s=sum(fields("whole")),
                                                     hot=any(intensity > 80), cold=all(intensity > -150),
                                                     mid=median(z)),
    "bins": group_by(bins(x, [-1_000, -10, 0, 10, 1_000])) >> summarise(k=n(), m=mean(y)),
    "a missing key kept": group_by(fields("gap"), dropna=False) >> summarise(k=n(), m=mean(intensity)),
    "a missing key dropped": group_by(ring, fields("gap")) >> summarise(k=n(), m=mean(intensity)),
    "no key": NEAR >> summarise(k=n(), m=mean(intensity), top=max(x)),
}


def cloud(count=60_000, seed=4, sort=False):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=(count, 5)) * 30).astype(np.float32)
    values[:, 4] = rng.integers(0, 6, count)
    if sort:  # each block's cells beyond the last's: the running table widens block after block
        values = values[np.argsort(values[:, 0])]
    return values


def frame(values, name, blocks=None, threads=1):
    data = read_array(values, **OPTIONS) if blocks is None else read_array(values, blocks=blocks, threads=threads,
                                                                           **OPTIONS)
    data = data >> mutate(whole=fields("ring") * 3 - 7, gap=if_else(fields("ring") > 1, fields("ring"), nan))
    return data >> ORDER_FREE[name]


def bits(result):
    data = result.collect()
    values = {key: np.asarray(data[key].values) for key in data.variables}
    return {key: value.tolist() if value.dtype.kind in "OU" else value.tobytes() for key, value in values.items()}


@pytest.fixture
def added(monkeypatch):
    """The blocks added into running tables."""
    tally = []
    add = driver._Running.add

    def spy(self, *args):
        tally.append(1)
        return add(self, *args)

    monkeypatch.setattr(driver._Running, "add", spy)
    return tally


@pytest.mark.parametrize("sort", [False, True], ids=["mixed", "widening block by block"])
@pytest.mark.parametrize("name", list(ORDER_FREE))
def test_running_tables_give_the_same_bits_on_any_number_of_threads(name, sort, added, monkeypatch):
    values = cloud(sort=sort)
    got = [bits(frame(values, name, 4_000, threads)) for threads in (1, 3, 7)]
    assert len(added) == 3 * 15, len(added)
    assert got[0] == got[1] == got[2]
    assert got[0] == bits(frame(values, name))
    with monkeypatch.context() as patch:
        patch.setenv("TIDY3A_COMPILED", "0")
        assert got[0] == bits(frame(values, name, 4_000))


def test_a_running_table_too_large_to_widen_is_given_up_as_a_table(added, monkeypatch):
    """Past the dense limits, the running table so far joins the merge as a block's would, and another starts."""
    values = cloud(sort=True)
    want = bits(frame(values, "voxels"))
    given = []
    table = driver._Running.table

    def spy(self, fused):
        given.append(1)
        return table(self, fused)

    monkeypatch.setattr(driver._Running, "table", spy)
    monkeypatch.setattr(driver, "DENSE_GROUPS", 300)  # a few blocks' cells, not every block's
    assert bits(frame(values, "voxels", 4_000, 3)) == want
    assert len(given) > 3 and added


@pytest.mark.parametrize("verbs", [group_by(ring) >> summarise(v=std(intensity)),
                                   group_by(ring) >> summarise(k=n(), v=var(y), m=mean(intensity))],
                         ids=["a spread", "a spread beside a mean"])
def test_a_summary_whose_result_follows_its_order_keeps_a_table_a_block(verbs, added):
    """A spread adds in float64 block by block, so its last bits follow the blocks' order: a table a block, merged
    in the fixed tree, as before."""
    values = cloud()
    got = [bits(read_array(values, blocks=4_000, threads=threads, **OPTIONS) >> verbs) for threads in (1, 3)]
    assert got[0] == got[1]  # the same bits on any number of threads, as the fixed tree gives
    assert not added
