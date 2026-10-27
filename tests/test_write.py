"""write(path): a frame's data to a .npy file or a Zarr store, block by block for data read in blocks (plan
milestone 12, step 2).

What is written reads back as ``collect()`` gives it in memory, bit for bit; a
frame read in blocks writes its rows as the blocks arrive and never joins them
in memory; a failed run leaves nothing under the name given.
"""

import os

import numpy as np
import pytest

from tidy3a import (
    WriteError,
    bins,
    col,
    fields,
    filter,
    formula,
    group_by,
    infer,
    mean,
    mutate,
    n,
    pull,
    read_array,
    scan,
    select,
    summarise,
    tidy,
    write,
)

NAMES = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring = fields(*NAMES)
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")
NEAR = mutate(d=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(col("d") <= 40)


def cloud(rows=10_000):
    rng = np.random.default_rng(0)
    values = (rng.normal(size=(rows, 5)) * 30).astype("float32")
    values[:, 4] = rng.integers(0, 8, rows)
    return values


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OU" or b.dtype.kind in "OU":
        return a.shape == b.shape and a.tolist() == b.tolist()
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture
def where(tmp_path):
    return lambda name: str(tmp_path / name)


@pytest.mark.parametrize("rows", [None, 7, 1_500, 20_000])
def test_a_npy_file_reads_back_as_collect_gives_it(where, rows):
    values = cloud()
    frame = read_array(values, **OPTIONS, **({"blocks": rows} if rows else {}))
    written = frame >> NEAR >> select("values") >> write(where("near.npy"))
    want = (read_array(values, **OPTIONS) >> NEAR >> select("values")).collect()
    stored = np.load(where("near.npy"), mmap_mode="r")  # the header's row count, written last, is the rows'
    assert stored.shape == want["values"].shape and stored.dtype == np.float32
    assert same_bits(written >> pull("values"), want["values"].values)
    assert (written >> pull("feature")).tolist() == NAMES
    assert not [name for name in os.listdir(os.path.dirname(where("near.npy"))) if ".partial-" in name]


@pytest.mark.parametrize("rows", [None, 997, 4_000])  # each block is one append to the store
def test_a_zarr_store_reads_back_as_collect_gives_it(where, rows):
    pytest.importorskip("zarr")
    values = cloud()
    frame = read_array(values, **OPTIONS, **({"blocks": rows} if rows else {}))
    written = frame >> NEAR >> write(where("near.zarr"))
    want = (read_array(values, **OPTIONS) >> NEAR).collect()
    back = written.collect()
    for name in want.data_vars:
        assert same_bits(back[name].values, want[name].values), name
    assert same_bits(back["point"].values, want["point"].values)  # the positions the filter kept, in order
    assert back["feature"].values.tolist() == NAMES


def test_a_frame_read_in_blocks_never_joins_its_rows_in_memory(where, monkeypatch):
    pytest.importorskip("zarr")
    import tidy3a.chunked as chunked

    def refuse(*args):
        raise AssertionError("rows joined in memory")

    monkeypatch.setattr(chunked, "_joined", refuse)
    frame = read_array(cloud(), **OPTIONS, blocks=1_000)
    written = frame >> NEAR >> write(where("near.zarr"))
    assert type(written).__name__ == "ChunkedFrame"  # the store, read back in blocks of the same size
    assert "blocks: 10 of 1,000 positions on point" in repr(frame) and "positions on point" in repr(written)


def test_zarr_keeps_text_of_any_width_times_attributes_and_drops_netcdf_encodings(where):
    pytest.importorskip("zarr")
    xr = pytest.importorskip("xarray")
    stamps = np.datetime64("2024-01-01", "ns") + np.arange(6) * np.timedelta64(1, "h")
    source = xr.Dataset({"name": ("time", np.array(["a", "b", "c", "a much longer name", "e", "f"])),
                         "v": ("time", np.arange(6, dtype="float32"), {"units": "K"})},
                        coords={"time": stamps}, attrs={"title": "t"})
    source["v"].encoding.update({"dtype": "float32", "zlib": True, "chunksizes": (3,), "source": "a.nc"})
    written = tidy(source, blocks={"time": 2}) >> write(where("t.zarr"))
    back = written.collect()
    assert back["name"].values.tolist() == source["name"].values.tolist()  # the longer name is not cut to one letter
    assert np.array_equal(back["time"].values, stamps) and back.attrs == {"title": "t"} and back["v"].attrs == {
        "units": "K"}
    assert same_bits(back["v"].values, source["v"].values)


def test_an_existing_path_wrong_suffix_and_several_variables_are_refused_before_any_block(where):
    values = cloud()
    frame = read_array(values, **OPTIONS, blocks=1_000)
    source, opened = frame._blocks.source, []
    open(where("there.npy"), "wb").close()
    with pytest.raises(WriteError, match="there.npy exists; overwrite=True replaces it"):
        frame >> select("values") >> write(where("there.npy"))
    with pytest.raises(WriteError, match=r"one of \.npy, \.zarr"):
        frame >> write(where("near.csv"))
    with pytest.raises(WriteError, match=r"a \.npy file holds one array, and this data has 2 variables \(values, d\)"):
        frame >> NEAR >> write(where("two.npy"))
    assert opened == [] and os.path.getsize(where("there.npy")) == 0
    written = frame >> select("values") >> write(where("there.npy"), overwrite=True)
    assert (written >> pull("values")).shape == (10_000, 5)


def test_a_failed_block_leaves_no_file(where):
    """A value only block 3 holds fails there; the writer removes its temporary file, and the name stays free."""
    from tidy3a.verbs import Verb

    def fail_late(frame):  # a verb works on the data it is given: the empty block, then each block
        positions = np.asarray(frame._backend.axis_labels(frame._data, "point").values)
        if len(positions) and positions.max() > 2_000:
            raise WriteError("a failure in a later block")
        return frame

    values = cloud()
    frame = read_array(values, **OPTIONS, blocks=1_000)
    failing = frame >> Verb("fail_late()", fail_late, operation="mutate", arguments=[])
    with pytest.raises(WriteError, match="(?s)a failure in a later block.*in block 3 of 10"):
        failing >> write(where("never.npy"))
    assert os.listdir(os.path.dirname(where("never.npy"))) == []


def test_writing_beside_a_summary_reads_the_blocks_once(where):
    pytest.importorskip("zarr")
    values = cloud()
    path = where("cloud.npy")
    np.save(path, values)
    frame = read_array(path, **OPTIONS, blocks=1_000)
    source, opened = frame._blocks.source, []
    reopen = source.reopen
    source.reopen = lambda: opened.append(1) or reopen()
    near = frame >> NEAR
    rings = near >> group_by(ring) >> summarise(points=n(), d=mean(col("d")))
    summary, written = scan(rings, writes=[(near, where("near.zarr"))])
    assert len(opened) == 10  # one read of each block for both
    eager = read_array(values, **OPTIONS) >> NEAR
    assert same_bits(summary >> pull("d"), eager >> group_by(ring) >> summarise(d=mean(col("d"))) >> pull("d"))
    assert same_bits(written.collect()["d"].values, eager.collect()["d"].values)


def test_a_pipeline_goes_on_from_what_it_wrote(where):
    values = cloud()
    frame = read_array(values, **OPTIONS, blocks=1_500)
    per_ring = frame >> NEAR >> select("values") >> write(where("near.npy")) >> group_by(ring) >> summarise(
        points=n(), brightness=mean(intensity))
    eager = read_array(values, **OPTIONS) >> NEAR >> group_by(ring) >> summarise(points=n(),
                                                                              brightness=mean(intensity))
    for name in ("ring", "points", "brightness"):
        assert same_bits(per_ring >> pull(name), eager >> pull(name)), name


def test_a_summary_and_an_empty_result_are_written_whole(where):
    pytest.importorskip("zarr")
    values = cloud()
    frame = read_array(values, **OPTIONS, blocks=1_000)
    summary = frame >> group_by(bins(intensity, [-1_000, 0, 1_000])) >> summarise(points=n())
    back = (summary >> write(where("summary.zarr"))).collect()
    assert back["points"].values.sum() == 10_000
    nothing = frame >> NEAR >> filter(col("d") < 0) >> select("values") >> write(where("nothing.npy"))
    assert np.load(where("nothing.npy")).shape == (0, 5) and (nothing >> pull("values")).shape == (0, 5)
