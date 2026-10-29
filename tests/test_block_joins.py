"""Joins of data read in blocks with data in memory (plan milestone 19, step 1): each block joins on its own, the
frame staying read in blocks, with the bits the same join gives in memory; a join the blocks cannot make alone runs
on the whole axis and says why."""

import numpy as np
import pytest

from tidy3a import (fields, filter, full_join, group_by, if_else, inner_join, left_join, mean, mutate, n, nan, sum,
                    summarise, tidy)
from tidy3a.chunked import ChunkedFrame
from tidy3a.errors import BlockError

xr = pytest.importorskip("xarray")
t, land, month, quality = fields("t", "land", "month", "quality")
rng = np.random.default_rng(5)
STEPS = 120
GRID = xr.Dataset({"t": (("time", "lat", "lon"), rng.normal(280, 10, (STEPS, 9, 18)).astype(np.float32)),
                   "month": ("time", np.arange(STEPS) // 10 % 12 + 1)},
                  coords={"time": np.arange(STEPS), "lat": np.arange(9.0), "lon": np.arange(18.0)})
MASK = xr.Dataset({"land": (("lat", "lon"), rng.random((9, 18)) > 0.6)},
                  coords={"lat": np.arange(9.0), "lon": np.arange(18.0)})
WIDER = xr.Dataset({"elev": (("lat", "lon"), rng.random((11, 18)))},
                   coords={"lat": np.arange(-1.0, 10.0), "lon": np.arange(18.0)})
INFO = xr.Dataset({"quality": ("time", rng.random(100).astype(np.float32))}, coords={"time": np.arange(10, 110)})
JOINS = {
    "a mask over the other axes": lambda: left_join(MASK) >> mutate(lt=if_else(land, t, nan)) >> group_by(month)
                                          >> summarise(m=mean(fields("lt"), na_rm=True), k=n()),
    "a value per time step, filtered on": lambda: left_join(INFO) >> filter(quality > 0.2) >> group_by(month)
                                                  >> summarise(m=mean(t), k=n()),
    "only the time steps both hold": lambda: inner_join(INFO) >> group_by(month) >> summarise(m=mean(t), s=sum(t)),
    "a full join widening another axis": lambda: full_join(WIDER) >> group_by(month)
                                                 >> summarise(m=mean(t, na_rm=True), k=n()),
    "rows kept": lambda: left_join(INFO) >> mutate(q=quality * 2),
    "two joins": lambda: left_join(MASK) >> inner_join(INFO) >> filter(quality > 0.5) >> group_by(month)
                         >> summarise(k=n(), m=mean(if_else(land, t, nan), na_rm=True)),
}


def same(got, want):
    got, want = got.collect(), want.collect()
    assert sorted(got.data_vars) == sorted(want.data_vars) and dict(got.sizes) == dict(want.sizes)
    for name in want.variables:
        assert np.asarray(got[name].values).tobytes() == np.asarray(want[name].values).tobytes(), name


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("name", list(JOINS))
def test_a_join_with_data_in_memory_runs_on_each_block(name, threads):
    frame = tidy(GRID, blocks={"time": 25}, threads=threads) >> JOINS[name]()
    assert isinstance(frame, ChunkedFrame)  # still read in blocks: nothing gathered in memory
    assert "nothing read yet" in repr(frame)
    same(frame, tidy(GRID) >> JOINS[name]())


def along(first, last, seed=7):
    """A value and a count per time step, for some of the grid's steps and some it lacks."""
    rng = np.random.default_rng(seed)
    size = last - first
    return xr.Dataset({"quality": ("time", rng.random(size).astype(np.float32)),
                       "visits": ("time", rng.integers(0, 9, size))}, coords={"time": np.arange(first, last)})


FULL = {
    "rows": lambda other: full_join(other),
    "rows, then a mutate": lambda other: full_join(other) >> mutate(q=quality * 2),
    "a summary by month, the missing month kept": lambda other: full_join(other) >> group_by(month, dropna=False)
    >> summarise(k=n(), m=mean(t, na_rm=True), v=sum(fields("visits"))),
}


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("span", [(60, 160), (10, 50), (200, 230)],
                         ids=["overlapping", "inside the grid's steps", "beyond them"])
@pytest.mark.parametrize("name", list(FULL))
def test_a_full_join_along_the_blocked_axis_runs_in_one_pass(name, span, threads):
    """Each block full-joins the other data at its own labels, cast to the types the whole join gives (here the
    month and the visits as floats, where gaps are), and a last block of no positions joins the labels no block holds:
    still read in blocks, memory's bits, types, and order (plan milestone 19, after Spark's broadcast nested-loop
    join)."""
    other = along(*span)
    frame = tidy(GRID, blocks={"time": 25}, threads=threads) >> FULL[name](other)
    assert isinstance(frame, ChunkedFrame)
    got, want = frame.collect(), (tidy(GRID) >> FULL[name](other)).collect()
    assert {key: value.dtype for key, value in got.variables.items()} == \
        {key: value.dtype for key, value in want.variables.items()}
    same(frame, tidy(GRID) >> FULL[name](other))


def test_times_and_small_integers_take_the_whole_joins_types():
    """int8 to float32, times in days to seconds: the backend's own filling of gaps, which the blocks follow."""
    days = (np.datetime64("2026-01-01") + np.arange(STEPS).astype("timedelta64[D]"))
    data = GRID.assign(small=("time", (np.arange(STEPS) % 7).astype(np.int8)), day=("time", days))
    other = along(100, 140)
    frame = tidy(data, blocks={"time": 30}) >> full_join(other)
    same(frame, tidy(data) >> full_join(other))
    assert frame.collect()["small"].dtype == np.float32


def test_a_scan_adds_the_last_block_for_every_frame():
    from tidy3a import scan

    other = along(60, 160)
    blocks = tidy(GRID, blocks={"time": 25})
    joined = blocks >> full_join(other) >> group_by(month, dropna=False) >> summarise(k=n())
    plain = blocks >> group_by(month) >> summarise(m=mean(t), k=n())
    got_joined, got_plain = scan(joined, plain)
    same(got_joined, tidy(GRID) >> full_join(other) >> group_by(month, dropna=False) >> summarise(k=n()))
    same(got_plain, tidy(GRID) >> group_by(month) >> summarise(m=mean(t), k=n()))


@pytest.mark.parametrize("verbs, why", [
    (lambda other: filter(month > 3) >> full_join(other), "after a filter or an inner join"),
    (lambda other: full_join(other.rename({"quality": "t"})), "hold variables of the same names"),
], ids=["after a filter", "names in common"])
def test_a_full_join_the_blocks_cannot_make_alone_runs_on_the_whole_axis_and_says_why(verbs, why):
    other = along(60, 160)
    frame = tidy(GRID, blocks={"time": 25}) >> verbs(other)
    assert not isinstance(frame, ChunkedFrame)
    assert any(why in note for note in frame._notes), frame._notes
    same(frame, tidy(GRID) >> verbs(other))


def test_a_join_of_two_frames_read_in_blocks_runs_on_the_whole_axis():
    other = tidy(INFO, blocks={"time": 30})
    frame = tidy(GRID, blocks={"time": 25}) >> left_join(other)
    assert not isinstance(frame, ChunkedFrame)
    assert any("the other data is read in blocks too" in note for note in frame._notes)


def test_a_join_after_group_by_is_refused_as_the_other_grouped_verbs_are():
    with pytest.raises(BlockError, match="after group_by, only summarise"):
        tidy(GRID, blocks={"time": 25}) >> group_by(month) >> left_join(MASK)


def big_grid(lat, lon, steps=96, seed=6):
    rng = np.random.default_rng(seed)
    grid = xr.Dataset({"t": (("time", "lat", "lon"), rng.normal(280, 10, (steps, lat, lon)).astype(np.float32)),
                       "month": ("time", np.arange(steps) // 8 % 12 + 1)},
                      coords={"time": np.arange(steps), "lat": np.arange(float(lat)), "lon": np.arange(float(lon))})
    mask = xr.Dataset({"land": (("lat", "lon"), rng.random((lat, lon)) > 0.6),
                       "weight": (("lat", "lon"), rng.random((lat, lon)).astype(np.float32))},
                      coords={"lat": np.arange(float(lat)), "lon": np.arange(float(lon))})
    info = xr.Dataset({"quality": ("time", rng.random(steps).astype(np.float32))}, coords={"time": np.arange(steps)})
    return grid, mask, info


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("shape", [(9, 18), (20, 20)], ids=["162 values a position", "400, reduced by NumPy first"])
def test_a_summary_after_joins_compiles_with_numpys_bits(shape, threads, monkeypatch):
    """NumPy joins each block first, and the kernels read what it brings: a mask over lat and lon by value, a
    quality per time step by position (plan milestone 19, step 2)."""
    pytest.importorskip("numba")
    from tidy3a.fused import plan_for

    grid, mask, info = big_grid(*shape)
    weight = fields("weight")
    verbs = (left_join(mask) >> inner_join(info) >> filter(quality > 0.2) >> mutate(lt=if_else(land, t, nan))
             >> group_by(month) >> summarise(m=mean(fields("lt"), na_rm=True), w=sum(t * weight), k=n()))
    frame = tidy(grid, blocks={"time": 16}, threads=threads) >> verbs
    assert plan_for(frame).compiled, plan_for(frame).reasons
    frame.collect()
    assert frame._result._notes[-1] == "ran: 6 blocks in a compiled kernel"
    with monkeypatch.context() as patch:
        patch.setenv("TIDY3A_COMPILED", "0")
        want = tidy(grid, blocks={"time": 16}) >> verbs
        want.collect()
    same(frame, want)


def test_joins_that_could_change_what_came_before_stay_on_numpy():
    """A full join may widen the axes, and change the types, of what came before it; a join bringing a name a mutate
    made would meet it differently if made first."""
    from tidy3a.fused import plan_for

    grid, mask, info = big_grid(9, 18)
    widened = tidy(grid, blocks={"time": 16}) >> full_join(WIDER.rename({"elev": "e"})) >> group_by(month) \
        >> summarise(k=n())
    assert "full_join(…): a full join runs on NumPy" in plan_for(widened).reasons
    clash = tidy(grid, blocks={"time": 16}) >> mutate(quality=t * 0) >> left_join(info, suffix=("", "_info")) \
        >> group_by(month) >> summarise(k=n())
    assert any("share names with ones a mutate made" in reason for reason in plan_for(clash).reasons)
