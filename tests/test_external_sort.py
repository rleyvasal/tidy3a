"""A sort written block by block (plan milestone 21, step 4): each block's sorted rows spilled to a run of their own,
the runs merged into the file in one streaming pass; the file memory's sort would write, and no spill left behind."""

import os

import numpy as np
import pytest

from tidy3a import (arrange, desc, fields, filter, infer, read_array, slice_head, slice_tail, tidy, write)
from tidy3a import external

xr = pytest.importorskip("xarray")
pytest.importorskip("numba")
names = ["x", "y", "z", "intensity", "ring"]
x, y, ring = fields("x", "y", "ring")
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature")
rng = np.random.default_rng(14)
CLOUD = (rng.normal(size=(12_000, 5)) * 30).astype(np.float32)
CLOUD[::7, 0], CLOUD[::11, 0], CLOUD[::13, 0] = np.nan, 0.0, -0.0  # NaN, and both zeros, which tie
CLOUD[:, 4] = rng.integers(0, 5, len(CLOUD))


@pytest.fixture(autouse=True)
def small_pieces(monkeypatch):
    """Pieces of 97 rows: a run spans many, and the merge takes many rounds."""
    monkeypatch.setattr(external, "PIECE", 97)
    monkeypatch.setattr(external, "WINDOWS", 97 * 2 * 6)  # six blocks of 2,000: pieces of 97


@pytest.fixture
def runs(monkeypatch):
    spilled = []
    add = external.SortedWriter.add

    def spy(self, data):
        spilled.append(1)
        return add(self, data)

    monkeypatch.setattr(external.SortedWriter, "add", spy)
    return spilled


SORTS = {
    "one key": lambda: arrange(x),
    "downwards": lambda: arrange(desc(x)),
    "two keys": lambda: arrange(ring, desc(x)),
    "after a filter": lambda: filter(y > 0) >> arrange(desc(ring), x),
    "the first 1,234": lambda: arrange(x) >> slice_head(point=1_234),
    "the last 777, downwards": lambda: arrange(desc(x)) >> slice_tail(point=777),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(SORTS))
def test_a_sort_written_is_the_file_memorys_sort_writes(name, threads, tmp_path, runs):
    path = tmp_path / "sorted.npy"
    written = read_array(CLOUD, blocks=2_000, threads=threads, **OPTIONS) >> SORTS[name]() >> write(path)
    want = np.asarray((read_array(CLOUD, **OPTIONS) >> SORTS[name]()).collect()["values"].values)
    assert np.load(path).tobytes() == want.tobytes()
    assert len(runs) == 6  # a run a block
    assert not [entry for entry in os.listdir(tmp_path) if entry.startswith(".tidy3a-sort")]
    assert np.asarray(written.collect()["values"].values).tobytes() == want.tobytes()  # read back in blocks


def test_a_dataset_sorted_into_zarr_keeps_its_labels_and_types(tmp_path, runs):
    pytest.importorskip("zarr")
    days = np.datetime64("2026-01-01") + np.arange(500).astype("timedelta64[D]")
    data = xr.Dataset({"t": (("time", "lat"), rng.normal(size=(500, 4)).astype(np.float32)),
                       "when": ("time", rng.permutation(days).astype("datetime64[ns]"))},
                      coords={"time": np.arange(500) * 3, "lat": np.arange(4.0)})
    verbs = arrange(desc(fields("when")))
    tidy(data, blocks={"time": 64}) >> verbs >> write(tmp_path / "sorted.zarr")
    want = (tidy(data) >> verbs).collect()
    got = xr.open_zarr(tmp_path / "sorted.zarr", consolidated=False).load()
    assert got["time"].values.tolist() == want["time"].values.tolist()
    assert got["when"].values.tolist() == want["when"].values.tolist() and got["when"].dtype == want["when"].dtype
    assert np.asarray(got["t"].values).tobytes() == np.asarray(want["t"].values).tobytes()
    assert runs


def test_a_failed_merge_leaves_neither_the_file_nor_its_runs(tmp_path, monkeypatch):
    from tidy3a.write import Writer

    def fail(self, data):
        raise OSError("the disk is full")

    monkeypatch.setattr(Writer, "add", fail)
    with pytest.raises(OSError, match="disk is full"):
        read_array(CLOUD, blocks=2_000, **OPTIONS) >> arrange(x) >> write(tmp_path / "sorted.npy")
    assert os.listdir(tmp_path) == []


def test_a_text_key_writes_the_merged_rows_from_memory(tmp_path, runs):
    pytest.importorskip("zarr")
    words = xr.Dataset({"word": ("p", rng.choice(["ant", "bee", "cat"], 300)), "v": ("p", rng.normal(size=300))})
    tidy(words, blocks={"p": 64}) >> arrange(fields("word"), desc(fields("v"))) >> write(tmp_path / "words.zarr")
    want = (tidy(words) >> arrange(fields("word"), desc(fields("v")))).collect()
    got = xr.open_zarr(tmp_path / "words.zarr", consolidated=False).load()
    assert got["word"].values.tolist() == want["word"].values.tolist() and not runs
