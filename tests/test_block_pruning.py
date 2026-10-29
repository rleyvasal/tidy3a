"""Blocks a filter on the blocked axis's labels keeps nothing of are not read (plan milestone 19, step 5, as Spark
prunes partitions by their statistics): a block's labels are known before it is read, so the filter is tried on
them alone; the result is memory's, and says how many blocks were skipped."""

import numpy as np
import pytest

from tidy3a import (chunked, fields, filter, full_join, group_by, infer, mean, n, read_array, scan, sum, summarise,
                    tidy)

xr = pytest.importorskip("xarray")
rng = np.random.default_rng(11)
DAYS = np.datetime64("2026-01-01") + np.arange(365).astype("timedelta64[D]")
GRID = xr.Dataset({"t": (("time", "lat"), rng.normal(280, 10, (365, 9)).astype(np.float32)),
                   "month": ("time", DAYS.astype("datetime64[M]").astype(int) % 12 + 1)},
                  coords={"time": DAYS.astype("datetime64[ns]"), "lat": np.arange(9.0)})
names = ["x", "y", "z", "intensity", "ring"]
CLOUD = (rng.normal(size=(10_000, 5)) * 30).astype(np.float32)
CLOUD[:, 4] = rng.integers(0, 8, len(CLOUD))
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature")
time, t, month, point, x, ring = fields("time", "t", "month", "point", "x", "ring")
JUNE, JULY = np.datetime64("2026-06-01"), np.datetime64("2026-07-01")


@pytest.fixture
def reads(monkeypatch):
    loaded = []
    load = chunked.Blocks.load

    def counting(self, start, stop, names=None):
        if stop > start:
            loaded.append((start, stop))
        return load(self, start, stop, names)

    monkeypatch.setattr(chunked.Blocks, "load", counting)
    return loaded


def same(got, want):
    got, want = got.collect(), want.collect()
    assert dict(got.sizes) == dict(want.sizes)
    for name in want.variables:
        assert np.asarray(got[name].values).tobytes() == np.asarray(want[name].values).tobytes(), name


def skipped(frame):
    return [note for note in frame._result._notes if note.startswith("skipped")]


def dates(blocks, threads=1):
    return tidy(GRID, blocks={"time": 30}, threads=threads) if blocks else tidy(GRID)


def points(blocks, threads=1):
    return read_array(CLOUD, blocks=1_000, threads=threads, **OPTIONS) if blocks else read_array(CLOUD, **OPTIONS)


CASES = {  # a frame, and the blocks read of 13 (dates, 30 a block) or 10 (points, 1,000 a block)
    "dates from June": (lambda *a: dates(*a) >> filter(time >= JUNE), 8, 13),
    "June alone, by month": (lambda *a: dates(*a) >> filter((time >= JUNE) & (time < JULY)) >> group_by(month)
                             >> summarise(m=mean(t), k=n()), 2, 13),
    "two filters on the axis, one not": (lambda *a: dates(*a) >> filter(time >= JUNE) >> filter(time < JULY)
                                         >> filter(month == 6), 2, 13),
    "positions": (lambda *a: points(*a) >> filter(point > 9_500), 1, 10),
    "positions, by ring": (lambda *a: points(*a) >> filter(point <= 1_200) >> group_by(ring)
                           >> summarise(k=n(), s=sum(x)), 2, 10),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(CASES))
def test_blocks_a_filter_on_the_axis_keeps_nothing_of_are_not_read(name, threads, reads):
    make, blocks, every = CASES[name]
    frame = make(True, threads)
    frame.collect()
    assert len(reads) == blocks, reads
    assert skipped(frame) == [f"skipped {every - blocks:,} of {every:,} blocks, which "
                              + skipped(frame)[0].split(", which ")[1]]
    same(frame, make(False))


def test_a_filter_reading_more_than_the_axis_reads_every_block(reads):
    frame = read_array(CLOUD, blocks=1_000, **OPTIONS) >> filter((point > 9_500) & (x > 0))
    frame.collect()
    assert len(reads) == 10 and not skipped(frame)


def test_a_compiled_summary_runs_only_the_blocks_kept(reads):
    pytest.importorskip("numba")
    frame = read_array(CLOUD, blocks=1_000, **OPTIONS) >> filter(point > 9_500) >> group_by(ring) >> summarise(k=n())
    frame.collect()
    assert len(reads) == 1 and frame._result._notes[-1] == "ran: 1 block in a compiled kernel"
    same(frame, read_array(CLOUD, **OPTIONS) >> filter(point > 9_500) >> group_by(ring) >> summarise(k=n()))


def test_a_scan_reads_the_blocks_any_of_its_frames_keeps(reads):
    blocks = tidy(GRID, blocks={"time": 30})
    june = blocks >> filter((time >= JUNE) & (time < JULY)) >> summarise(k=n())
    january = blocks >> filter(time < np.datetime64("2026-02-01")) >> summarise(k=n())
    scan(june, january)
    assert len(reads) == 4  # two for June (it straddles), one for January, the week of February 1 in the first two
    reads.clear()
    everything = blocks >> summarise(k=n())
    scan(blocks >> filter(time >= JUNE) >> summarise(k=n()), everything)
    assert len(reads) == 13  # a frame without such a filter needs every block


def test_a_full_joins_own_block_is_never_skipped():
    other = xr.Dataset({"q": ("time", np.ones(3, np.float32))},
                       coords={"time": np.array(["2027-03-01", "2027-03-02", "2027-03-03"], "datetime64[ns]")})
    verbs = filter(time >= JUNE) >> full_join(other) >> summarise(k=n(), q=sum(fields("q")))
    frame = tidy(GRID, blocks={"time": 30}) >> verbs
    same(frame, tidy(GRID) >> verbs)


def test_a_filter_keeping_nothing_reads_nothing(reads):
    frame = tidy(GRID, blocks={"time": 30}) >> filter(time > np.datetime64("2030-01-01")) >> summarise(k=n())
    same(frame, tidy(GRID) >> filter(time > np.datetime64("2030-01-01")) >> summarise(k=n()))
    assert not reads
