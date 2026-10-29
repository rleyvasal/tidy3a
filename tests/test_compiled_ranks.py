"""The median's and quantiles' later passes compiled (plan milestone 20, step 1): each pass in one loop over a block's
positions, counting into each thread's running table, with the bits NumPy's passes give, and memory's; a block the
loop cannot place taken back and counted as before; many targets counted sparsely, as before."""

import numpy as np
import pytest

from tidy3a import (cells, fields, filter, formula, group_by, infer, median, mutate, n, quantile, read_array,
                    summarise, tidy)
from tidy3a.fused import driver, ranks

pytest.importorskip("numba")
names = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring, distance = fields(*names, "distance")
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature")
NEAR = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 50)


def cloud(count=60_000, seed=3, rings=8):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=(count, 5)) * 30).astype(np.float32)
    values[:, 4] = rng.integers(0, rings, count)
    return values


CASES = {
    "the median by ring, filtered": NEAR >> group_by(ring) >> summarise(m=median(intensity), k=n()),
    "quantiles by ring": NEAR >> group_by(ring) >> summarise(low=quantile(intensity, 0.1), high=quantile(x, 0.9)),
    "the median of an expression": group_by(ring) >> summarise(m=median(intensity * 2 + x)),
    "two keys, one of cells": group_by(ring, cells(z, 100)) >> summarise(m=median(intensity)),  # 8 × 4 groups
    "no key": NEAR >> summarise(m=median(intensity), q=quantile(y, 0.25)),
    "a NaN among the values, kept out": group_by(ring) >> summarise(m=median(fields("nan_y"), na_rm=True)),
}


def frame(values, name, blocks=None, threads=1):
    data = read_array(values, **OPTIONS) if blocks is None else read_array(values, blocks=blocks, threads=threads,
                                                                           **OPTIONS)
    if name == "a NaN among the values, kept out":
        data = data >> mutate(nan_y=fields("y") / (fields("ring") - 3))  # 0 / 0 where ring is 3 and y is 0
    return data >> CASES[name]


def bits(result):
    data = result.collect()
    values = {key: np.asarray(data[key].values) for key in data.data_vars}
    return {key: value.tolist() if value.dtype.kind == "O" else value.tobytes() for key, value in values.items()}


@pytest.fixture
def counted(monkeypatch):
    """How the later passes were counted: by the one loop, or not (a block it gave back, or none run)."""
    tally = {"loop": 0, "otherwise": 0}
    count = driver.Fused.count

    def spy(self, *args):
        found = count(self, *args)
        tally["loop" if found is not None else "otherwise"] += 1
        return found

    monkeypatch.setattr(driver.Fused, "count", spy)
    return tally


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(CASES))
def test_a_later_pass_in_one_loop_gives_numpys_bits_and_memorys(name, threads, counted, monkeypatch):
    values = cloud()
    values[::997, 1] = 0  # zeros, so the division above makes NaN on ring 3
    got = bits(frame(values, name, 4_000, threads))
    assert counted["loop"] > 0 and counted["otherwise"] == 0, counted
    assert got == bits(frame(values, name))
    with monkeypatch.context() as patch:
        patch.setenv("TIDY3A_COMPILED", "0")
        assert got == bits(frame(values, name, 4_000))


def test_a_block_holding_a_value_the_loop_cannot_place_is_taken_back_and_counted_otherwise(counted):
    """A negative zero as a key is one the kernels do not place (NumPy's order of the zeros is its own): the block's
    loop stops there, takes back what it counted, and the block is counted as NumPy's passes count it."""
    values = cloud()
    values[30_123, 4] = -0.0
    got = bits(frame(values, "the median by ring, filtered", 4_000))
    assert counted["otherwise"] > 0
    assert got == bits(frame(values, "the median by ring, filtered"))


def test_many_targets_are_counted_sparsely_as_before(counted):
    """More targets than a dense table of 65,536 digits each holds within ``DENSE_DIGITS``: the sparse pass."""
    values = cloud(rings=200)
    got = bits(frame(values, "the median by ring, filtered", 4_000))
    assert counted["loop"] == 0 and 200 * 2 * ranks.DIGITS > ranks.DENSE_DIGITS
    assert got == bits(frame(values, "the median by ring, filtered"))


@pytest.mark.parametrize("dtype", ["float64", "int16", "int64", "uint32"])
def test_other_types_of_values(dtype):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(8)
    values = rng.normal(size=(400, 30)) * 1_000
    data = xr.Dataset({"v": (("time", "cell"), (np.abs(values) if dtype[0] == "u" else values).astype(dtype)),
                       "month": ("time", np.arange(400) // 40 % 12)})
    verbs = group_by(fields("month")) >> summarise(m=median(fields("v")), q=quantile(fields("v"), 0.3))
    assert bits(tidy(data, blocks={"time": 64}, threads=2) >> verbs) == bits(tidy(data) >> verbs)


def test_running_tables_are_freed_with_their_pass():
    values = cloud()
    bits(frame(values, "the median by ring, filtered", 4_000, 3))
    import gc

    gc.collect()
    assert not ranks._RUNNING
