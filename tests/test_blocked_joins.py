"""Joins of two frames read in blocks (plan milestone 21, step 5): each block of this frame reads from the other only
the positions of the labels it holds, puts them through the other's verbs, and joins them as data in memory; a full
join adds the other's labels no block holds last; memory's bits, the frame read in blocks."""

import numpy as np
import pytest

from tidy3a import (fields, filter, full_join, group_by, inner_join, left_join, mean, mutate, n, select, sum,
                    summarise, tidy)
from tidy3a import chunked
from tidy3a.chunked import ChunkedFrame
from tidy3a.errors import AlignError

xr = pytest.importorskip("xarray")
rng = np.random.default_rng(21)
t, h, month = fields("t", "h", "month")
STEPS = 150
GRID = xr.Dataset({"t": (("time", "lat"), rng.normal(280, 10, (STEPS, 6)).astype(np.float32)),
                   "month": ("time", np.arange(STEPS) // 12 % 12 + 1)},
                  coords={"time": np.arange(STEPS) * 2, "lat": np.arange(6.0)})
OTHER = xr.Dataset({"h": (("time", "lat"), rng.random((120, 6)).astype(np.float32)),
                    "w": ("time", rng.integers(0, 9, 120)), "unused": ("time", np.zeros(120))},
                   coords={"time": np.arange(60, 300, 2)[::-1].copy(), "lat": np.arange(6.0)})  # in another order


def same(got, want):
    got, want = got.collect(), want.collect()
    assert sorted(got.data_vars) == sorted(want.data_vars) and dict(got.sizes) == dict(want.sizes)
    for name in want.variables:
        assert np.asarray(got[name].values).tobytes() == np.asarray(want[name].values).tobytes(), name


def their(blocks=None):
    data = tidy(OTHER, blocks={"time": blocks}) if blocks else tidy(OTHER)
    return data >> select("h", "w") >> mutate(h2=h * 2)


JOINS = {
    "left, summarised": lambda other: left_join(other) >> mutate(d=t - h) >> group_by(month)
    >> summarise(k=n(), m=mean(fields("d"), na_rm=True)),
    "inner, rows kept": lambda other: inner_join(other) >> mutate(d=t * fields("h2")),
    "full, rows kept": lambda other: full_join(other),
    "full, summarised": lambda other: full_join(other) >> group_by(month, dropna=False)
    >> summarise(k=n(), v=sum(fields("w"))),
    "after a filter of this data": lambda other: filter(month > 2) >> left_join(other) >> group_by(month)
    >> summarise(m=mean(h, na_rm=True)),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(JOINS))
def test_two_frames_read_in_blocks_join_block_by_block(name, threads):
    frame = tidy(GRID, blocks={"time": 25}, threads=threads) >> JOINS[name](their(blocks=17))
    assert isinstance(frame, ChunkedFrame)  # still read in blocks: nothing gathered in memory
    same(frame, tidy(GRID) >> JOINS[name](their()))


def test_each_block_reads_only_its_part_of_the_other(monkeypatch):
    read = []
    block = chunked.DatasetSource.block

    def counting(self, start, stop, names=None):
        if self.data is not GRID:
            read.append((start, stop, names))
        return block(self, start, stop, names)

    monkeypatch.setattr(chunked.DatasetSource, "block", counting)
    frame = tidy(GRID, blocks={"time": 25}) >> left_join(their(blocks=17)) >> group_by(month) >> summarise(k=n())
    frame.collect()
    rows = builtins_sum(stop - start for start, stop, _ in read)
    assert 0 < rows <= 2 * 120  # what the blocks' labels match, not every block of the other for each
    assert all(names is None or "unused" not in names for _, _, names in read)  # only the variables it brings


def builtins_sum(values):
    total = 0
    for value in values:
        total += value
    return total


def test_unlabelled_frames_pair_their_blocks_positions():
    left = GRID.drop_vars("time")
    right = OTHER.isel(time=slice(0, 120)).drop_vars("time")
    left = left.isel(time=slice(0, 120))
    verbs = lambda other: left_join(other) >> group_by(month) >> summarise(m=mean(h), k=n())  # noqa: E731
    frame = tidy(left, blocks={"time": 25}, threads=2) >> verbs(tidy(right, blocks={"time": 40}))
    assert isinstance(frame, ChunkedFrame)
    same(frame, tidy(left) >> verbs(tidy(right)))
    with pytest.raises(AlignError, match="different lengths"):
        tidy(left, blocks={"time": 25}) >> left_join(tidy(right.isel(time=slice(0, 90)), blocks={"time": 40}))


@pytest.mark.parametrize("other, why", [
    (lambda: tidy(OTHER, blocks={"lat": 2}), "read in blocks along lat, not time"),
    (lambda: tidy(OTHER, blocks={"time": 17}) >> filter(fields("w") > 3), "decides which of its positions"),
], ids=["blocked along another axis", "a filter of the other"])
def test_a_join_the_blocks_cannot_look_up_runs_on_the_whole_axis_and_says_why(other, why):
    frame = tidy(GRID, blocks={"time": 25}) >> left_join(other())
    assert not isinstance(frame, ChunkedFrame)
    assert any(why in note for note in frame._notes), frame._notes


def test_a_summarised_other_joins_as_data_in_memory():
    summary = tidy(OTHER, blocks={"time": 17}) >> summarise(hm=mean(h), wk=sum(fields("w")))
    frame = tidy(GRID, blocks={"time": 25}) >> left_join(summary) >> mutate(u=t * fields("hm"))
    assert isinstance(frame, ChunkedFrame)
    same(frame, tidy(GRID) >> left_join(tidy(OTHER) >> summarise(hm=mean(h), wk=sum(fields("w"))))
         >> mutate(u=t * fields("hm")))


def test_a_summary_after_a_join_of_two_frames_compiles(monkeypatch):
    pytest.importorskip("numba")
    from tidy3a.fused import plan_for

    verbs = left_join(their(blocks=17)) >> group_by(month) >> summarise(k=n(), m=mean(t * h, na_rm=True))
    frame = tidy(GRID, blocks={"time": 25}, threads=2) >> verbs
    assert plan_for(frame).compiled, plan_for(frame).reasons
    frame.collect()
    assert "compiled kernel" in frame._result._notes[-1]
    with monkeypatch.context() as patch:
        patch.setenv("TIDY3A_COMPILED", "0")
        same(frame, tidy(GRID) >> left_join(their()) >> group_by(month) >> summarise(k=n(), m=mean(t * h, na_rm=True)))


@pytest.mark.parametrize("blocked", [False, True], ids=["data in memory", "read in blocks too"])
def test_a_variable_selected_away_before_a_join_does_not_meet_the_one_it_brings(blocked, monkeypatch):
    """This data's own h, dropped by select before the join, which brings another h: a block holds the first only
    because a later verb reads the name, and the compiled path, joining each block before its kernels, joins on what
    the verbs held there, as NumPy's blocks and memory do."""
    pytest.importorskip("numba")
    from tidy3a.fused import plan_for

    data = GRID.assign(h=GRID["t"] * 0 - 1)
    other = tidy(OTHER, blocks={"time": 17}) >> select("h") if blocked else tidy(OTHER) >> select("h")
    verbs = select("t", "month") >> left_join(other) >> mutate(d=t - h) >> group_by(month) >> summarise(
        m=mean(fields("d"), na_rm=True))
    frame = tidy(data, blocks={"time": 25}) >> verbs
    assert plan_for(frame).compiled
    same(frame, tidy(data) >> select("t", "month") >> left_join(tidy(OTHER) >> select("h")) >> mutate(d=t - h)
         >> group_by(month) >> summarise(m=mean(fields("d"), na_rm=True)))
