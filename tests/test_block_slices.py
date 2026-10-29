"""Slices of data read in blocks (plan milestone 19, step 4): the blocked axis sliced from only the blocks holding
what is kept, in memory, with the rows, labels, and bits the same slice gives in memory; positions a filter decides
found by reading blocks one, then more at a time; other axes sliced on each block, the frame staying read in blocks."""

import numpy as np
import pytest

from tidy3a import chunked, fields, filter, infer, mutate, read_array, slice_head, slice_tail, tidy
from tidy3a import slice as take
from tidy3a.chunked import ChunkedFrame

xr = pytest.importorskip("xarray")
names = ["x", "y", "z", "intensity", "ring"]
x = fields("x")
rng = np.random.default_rng(9)
CLOUD = (rng.normal(size=(10_000, 5)) * 30).astype(np.float32)
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature")
GRID = xr.Dataset({"t": (("time", "lat"), rng.normal(280, 10, (120, 9)).astype(np.float32))},
                  coords={"time": np.arange(1000, 1120), "lat": np.arange(9.0)})


@pytest.fixture
def reads(monkeypatch):
    """The blocks a run loads, with positions."""
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


CLOUD_SLICES = {  # verbs, and the blocks of 1,000 read
    "the first ten": (slice_head(point=10), 1),
    "the last ten": (slice_tail(point=10), 1),
    "the first 2,500": (slice_head(point=2_500), 3),
    "more than there are": (slice_head(point=20_000), 10),
    "none": (slice_tail(point=0), 0),
    "positions, out of order and past the end": (take(point=[9_999, 3, 4_500, 3, 50_000]), 3),
    "one position, its axis dropped": (take(point=4_321, drop=False), 1),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(CLOUD_SLICES))
def test_a_slice_of_the_blocked_axis_reads_only_the_blocks_it_keeps(name, threads, reads):
    verb, blocks = CLOUD_SLICES[name]
    got = read_array(CLOUD, blocks=1_000, threads=threads, **OPTIONS) >> verb
    assert not isinstance(got, ChunkedFrame)
    assert len(reads) == blocks, reads
    same(got, read_array(CLOUD, **OPTIONS) >> verb)


FILTERED = {  # verbs after a filter, the positions of the rows the filter keeps
    "the first ten kept": (filter(x > 30), slice_head(point=10)),
    "the last ten kept": (filter(x > 30), slice_tail(point=10)),
    "positions among those kept": (filter(x > 30), take(point=[40, 2, 700])),
    "a sparse filter, read in rounds": (filter(x > 95), slice_head(point=4)),
    "a sparse filter from the end": (filter(x > 95), slice_tail(point=4)),
    "more than are kept": (filter(x > 95), slice_head(point=1_000)),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(FILTERED))
def test_after_a_filter_blocks_are_read_until_enough_rows_are_kept(name, threads, reads):
    """One block, then more at a time (as Spark's LIMIT scales up): never more blocks than twice those needed, and
    the rows memory's."""
    kept, verb = FILTERED[name]
    got = read_array(CLOUD, blocks=1_000, threads=threads, **OPTIONS) >> kept >> verb
    same(got, read_array(CLOUD, **OPTIONS) >> kept >> verb)
    if name == "a sparse filter, read in rounds":
        matches = np.flatnonzero(CLOUD[:, 0] > 95)
        needed = matches[3] // 1_000 + 1  # the block holding the fourth match
        assert needed < len(reads) <= 10 and len(reads) <= 2 * needed + threads, (needed, reads)


def test_a_labelled_axis_keeps_its_labels():
    for verb in (slice_tail(time=5), take(time=[3, 100])):
        same(tidy(GRID, blocks={"time": 25}) >> verb, tidy(GRID) >> verb)


def test_a_slice_of_another_axis_runs_on_each_block(reads):
    frame = tidy(GRID, blocks={"time": 25}) >> slice_head(lat=3) >> mutate(u=fields("t") * 2)
    assert isinstance(frame, ChunkedFrame) and not reads
    same(frame, tidy(GRID) >> slice_head(lat=3) >> mutate(u=fields("t") * 2))


def test_excluding_positions_reads_every_block(reads):
    got = read_array(CLOUD, blocks=1_000, **OPTIONS) >> take(point=[-1, -2])
    assert len(reads) == 10
    same(got, read_array(CLOUD, **OPTIONS) >> take(point=[-1, -2]))


@pytest.mark.parametrize("verb, spans", [
    (slice_head(point=10), [(0, 10)]),
    (slice_tail(point=10), [(9_990, 10_000)]),
    (slice_head(point=2_500), [(0, 1_000), (1_000, 2_000), (2_000, 2_500)]),
    (take(point=[9_999, 3, 4_500, 3, 50_000]), [(2, 3), (4_499, 4_500), (9_998, 9_999)]),
    (take(point=[5, 8, 900]), [(4, 900)]),
], ids=["head", "tail", "across blocks", "scattered", "near one another"])
def test_known_positions_are_read_alone_not_the_blocks_holding_them(verb, spans, reads):
    """A slice of ten positions reads ten, a block's worth of data never held for it; positions near one another
    are read in one span (milestone 19, step 7)."""
    got = read_array(CLOUD, blocks=1_000, **OPTIONS) >> mutate(w=x * 2) >> verb
    assert sorted(reads) == spans
    same(got, read_array(CLOUD, **OPTIONS) >> mutate(w=x * 2) >> verb)
