"""Files read in blocks (plan milestone 10, decision 10, stage b, phase 1).

``read_array(path, ..., blocks=rows)`` holds the first block, runs each verb
on it as it is called, and runs the recorded verbs over every block when a
result is asked for. Results are compared with the same pipeline in memory:
counts, labels, and float32 sums and means bit for bit at every block size,
float64 and integer ones closely, and what a block cannot answer is refused
when the verb is called.
"""

import numpy as np
import pytest

from tidy3a import (
    BlockError,
    GroupError,
    MutateError,
    ReadError,
    SummariseError,
    bins,
    cells,
    fields,
    filter,
    formula,
    group_by,
    infer,
    max,
    mean,
    median,
    min,
    mutate,
    n,
    pull,
    read_array,
    rounded,
    select,
    sqrt,
    std,
    sum,
    summarise,
)
from tidy3a import slice as take

NAMES = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring, distance = fields(*NAMES, "distance")


def cloud(rows=20_000, seed=0):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=(rows, 5)) * 30).astype("float32")
    values[:, 4] = rng.integers(0, 40, rows)
    # A group whose float32 sums sit on ties or cancel exactly, which the error bound cannot settle.
    values[:6, 4] = 99
    values[:6, :3] = 0  # within 60 m, so the filter keeps them
    values[:6, 3] = np.float32(2**24), 1, 1, 2**30, 1, -(2**30)
    return values


def in_memory(values, pipeline, names):
    frame = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")
    out = frame >> pipeline
    return {name: out >> pull(name) for name in names}


def in_blocks(path, pipeline, names, rows, threads=1, **options):
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=rows, threads=threads, **options)
    out = frame >> pipeline
    return out, {name: out >> pull(name) for name in names}


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OU" or b.dtype.kind in "OU":  # text labels: equal as values, as objects or NumPy text
        return a.dtype.kind in "OU" and b.dtype.kind in "OU" and a.shape == b.shape and a.tolist() == b.tolist()
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture
def npy(tmp_path):
    values = cloud()
    path = tmp_path / "cloud.npy"
    np.save(path, values)
    return path, values


PIPELINE = (mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60) >> group_by(ring)
            >> summarise(points=n(), brightness=mean(intensity), total=sum(intensity), near=mean(distance),
                         low=min(z), high=max(z)))
COLUMNS = ["ring", "points", "brightness", "total", "near", "low", "high"]


@pytest.mark.parametrize("rows", [7, 997, 6_000, 20_000, 1_000_000])
@pytest.mark.parametrize("threads", [1, 3])
def test_float32_summaries_match_in_memory_bit_for_bit(npy, rows, threads):
    path, values = npy
    want = in_memory(values, PIPELINE, COLUMNS)
    _, got = in_blocks(path, PIPELINE, COLUMNS, rows, threads)
    for name in COLUMNS:
        assert same_bits(got[name], want[name]), name


def test_one_row_blocks(tmp_path):
    values = cloud(300)
    path = tmp_path / "small.npy"
    np.save(path, values)
    want = in_memory(values, PIPELINE, COLUMNS)
    _, got = in_blocks(path, PIPELINE, COLUMNS, 1, threads=3)
    for name in COLUMNS:
        assert same_bits(got[name], want[name]), name


def test_the_result_is_a_frame_like_the_in_memory_one(npy):
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    result = frame >> PIPELINE
    collected = result.collect()
    reference = (read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES},
                            field_axis="feature") >> PIPELINE).collect()
    assert list(collected.data_vars) == list(reference.data_vars)
    assert collected.sizes == reference.sizes
    assert result.to_dict()["dims"] == reference.to_dict()["dims"]
    text = str(result.explain())
    assert "the first of 7 blocks of 3,000 positions on point" in text
    assert "+ axis ring: 41" in text  # the last step's effect comes from every block


def test_expressions_of_reductions_combine(npy):
    path, values = npy
    pipeline = (filter(intensity > 0) >> group_by(ring)
                >> summarise(points=n(), spread=(max(z) - min(z)) / n(), rms=sqrt(mean(z * z)), half=-mean(z) / 2))
    columns = ["points", "spread", "rms", "half"]
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 2_500)
    for name in columns:
        assert same_bits(got[name], want[name]), name


def test_no_group_by_gives_single_values(npy):
    path, values = npy
    pipeline = select("intensity", "z") >> summarise(points=n(), brightness=mean(intensity), top=max(z))
    columns = ["points", "brightness", "top"]
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 3_000)
    for name in columns:
        assert same_bits(got[name], want[name]), name
        assert np.ndim(got[name]) == 0


def test_a_block_the_filter_empties_and_a_key_missing_in_some_blocks(tmp_path):
    values = cloud(9_000)
    values[3_000:6_000, 0] = 1e6  # the second block of 3,000 is removed whole
    values[:3_000, 4] = 1  # ring 1 only in the first block, the others only later
    values[6_000:, 4] += 50
    path = tmp_path / "gaps.npy"
    np.save(path, values)
    pipeline = filter(x < 1e5) >> group_by(ring) >> summarise(points=n(), brightness=mean(intensity), low=min(z))
    columns = ["ring", "points", "brightness", "low"]
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 3_000)
    for name in columns:
        assert same_bits(got[name], want[name]), name


def test_missing_keys_are_left_out_and_counted_over_every_block(tmp_path):
    values = cloud()
    values[::50, 4] = np.nan
    path = tmp_path / "missing.npy"
    np.save(path, values)
    pipeline = group_by(ring) >> summarise(points=n(), brightness=mean(intensity))
    columns = ["ring", "points", "brightness"]
    want = in_memory(values, pipeline, columns)
    result, got = in_blocks(path, pipeline, columns, 4_000)
    for name in columns:
        assert same_bits(got[name], want[name]), name
    assert "changed: group_by(ring) dropped 400 positions with a missing ring" in repr(result)


def test_float64_and_integer_sums_are_close(tmp_path):
    values = cloud().astype("float64")
    path = tmp_path / "cloud64.npy"
    np.save(path, values)
    pipeline = group_by(ring) >> summarise(points=n(), brightness=mean(intensity), total=sum(intensity),
                                           top=max(intensity, na_rm=True))
    columns = ["ring", "points", "brightness", "total", "top"]
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 3_000)
    assert same_bits(got["ring"], want["ring"]) and same_bits(got["points"], want["points"])
    assert same_bits(got["top"], want["top"])
    for name in ("brightness", "total"):
        assert got[name].dtype == want[name].dtype and np.allclose(got[name], want[name], rtol=1e-12), name

    counts = np.rint(cloud()).astype("int32")
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    want = in_memory(counts, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 3_000)
    for name in columns:
        assert same_bits(got[name], want[name]), name


def test_raw_file_in_blocks_matches_npy(tmp_path, npy):
    path, values = npy
    raw = tmp_path / "cloud.bin"
    values.astype(">f4").tofile(raw)
    frame = read_array(raw, value_type="float32", dims={"point": infer, "feature": 5}, labels={"feature": NAMES},
                       field_axis="feature", byte_order="big", blocks=5_000)
    want = in_memory(values, PIPELINE, COLUMNS)
    out = frame >> PIPELINE
    for name in COLUMNS:
        assert same_bits(out >> pull(name), want[name]), name


def test_verbs_run_on_the_first_block_as_they_are_called(npy):
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    assert "unlabeled, positions 1 … 3,000" in repr(frame)
    assert "blocks: 7 of 3,000 positions on point" in repr(frame)
    assert len(frame >> pull("x")) == 20_000  # every block, joined
    with pytest.raises(MutateError, match='"wrong"'):
        frame >> mutate(d=fields("wrong") + 1)
    assert "1. mutate(d = x + 1)" in str((frame >> mutate(d=x + 1)).explain())


@pytest.mark.parametrize("pipeline, message", [
    (mutate(centred=intensity - mean(intensity)) >> group_by(ring) >> summarise(points=n()), "whole axis"),
    (group_by(ring) >> mutate(centred=intensity - mean(intensity)), "after group_by, only summarise"),
    (group_by("point") >> summarise(points=n()), "keeps an axis"),
    (group_by(ring) >> summarise(points=n(), middle=median(intensity)), "cannot be combined"),
    (group_by(ring) >> summarise(total=sum(intensity, min_count=2)), "min_count"),
    (group_by(ring) >> summarise(spread=std(intensity)), "cannot be combined"),
])
def test_what_a_block_cannot_answer_is_refused_when_the_verb_is_called(npy, pipeline, message):
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    with pytest.raises(BlockError, match=message):
        frame >> pipeline


@pytest.mark.parametrize("pipeline, message", [
    (group_by(ring) >> summarise(first=intensity + 1), "wrap it in a reduction"),
    (group_by(ring) >> summarise(worst=max(mean(intensity))), "cannot be summarised"),
])
def test_summarise_s_own_errors_come_first(npy, pipeline, message):
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    with pytest.raises(SummariseError, match=message):
        frame >> pipeline


@pytest.mark.parametrize("options, message", [
    ({"blocks": 0}, "positive"),
    ({"blocks": 3, "threads": 0}, "positive"),
    ({"blocks": 3, "lazy": True}, "one of them"),
    ({"blocks": 3, "field_axis": "point"}, "first axis"),
])
def test_block_options_are_checked(npy, options, message):
    path, _ = npy
    options = {"field_axis": "feature", **options}
    with pytest.raises(ReadError, match=message):
        read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, **options)
    with pytest.raises(ReadError, match="already loaded"):
        read_array(np.zeros((4, 5), "float32"), dims={"point": infer, "feature": 5}, blocks=2)


def test_an_empty_file_gives_an_empty_summary(tmp_path):
    path = tmp_path / "empty.npy"
    np.save(path, np.zeros((0, 5), "float32"))
    pipeline = group_by(ring) >> summarise(points=n(), brightness=mean(intensity))
    want = in_memory(np.zeros((0, 5), "float32"), pipeline, ["ring", "points", "brightness"])
    _, got = in_blocks(path, pipeline, ["ring", "points", "brightness"], 3)
    for name in ("points", "brightness"):
        assert got[name].shape == want[name].shape == (0,)


# ── Phase 2: several keys, transforms, dropna=False, rows kept ──────────────


VOXELS = (filter(intensity > -100) >> group_by(cells(x, 20.0), rounded(y, 25), cells(z, 30, origin=5), axis="voxel")
          >> summarise(points=n(), brightness=mean(intensity), top=max(z)))
VOXEL_COLUMNS = ["voxel", "x_cell", "y_rounded", "z_cell", "points", "brightness", "top"]


@pytest.mark.parametrize("rows", [7, 997, 6_000, 1_000_000])
def test_several_keys_and_transforms_give_the_in_memory_groups_in_order(npy, rows):
    path, values = npy
    want = in_memory(values, VOXELS, VOXEL_COLUMNS)
    _, got = in_blocks(path, VOXELS, VOXEL_COLUMNS, rows, threads=3)
    for name in VOXEL_COLUMNS:
        assert same_bits(got[name], want[name]), name


@pytest.mark.parametrize("pipeline, columns", [
    (group_by(cells(x, 10)) >> summarise(points=n(), brightness=mean(intensity)), ["x_cell", "points", "brightness"]),
    (group_by(rounded(z, 0.5)) >> summarise(points=n(), low=min(x)), ["z_rounded", "points", "low"]),
    (group_by(bins(intensity, [-2**31, -50, 0, 0.5, 1, 2**31])) >> summarise(points=n(), brightness=mean(intensity)),
     ["intensity_bin", "points", "brightness"]),  # every bin, the empty ones too
    (group_by(ring, cells(x, 40)) >> summarise(points=n(), brightness=mean(intensity)),
     ["group", "ring", "x_cell", "points", "brightness"]),
])
def test_each_kind_of_key(npy, pipeline, columns):
    path, values = npy
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, 2_500)
    for name in columns:
        assert same_bits(got[name], want[name]), name


@pytest.mark.parametrize("pipeline, columns, note", [
    (group_by(ring, dropna=False) >> summarise(points=n(), brightness=mean(intensity)),
     ["ring", "points", "brightness"], None),
    (group_by(cells(x, 20), dropna=False) >> summarise(points=n()), ["x_cell", "points"], None),
    (group_by(rounded(x, 10), dropna=False) >> summarise(points=n()), ["x_rounded", "points"], None),
    (group_by(ring, cells(x, 20), dropna=False) >> summarise(points=n(), top=max(z)),
     ["group", "ring", "x_cell", "points", "top"], None),
    (group_by(ring, cells(x, 20)) >> summarise(points=n()), ["group", "ring", "x_cell", "points"],
     "changed: group_by(ring, cells(x)) dropped 800 positions with a missing ring or x; dropna=False keeps them "
     "as groups"),
    (group_by(cells(x, 20)) >> summarise(points=n()), ["x_cell", "points"],
     "changed: group_by(cells(x)) dropped 400 positions with a missing x; dropna=False keeps them as a group"),
])
def test_missing_keys_kept_as_the_last_group_or_dropped_and_counted(tmp_path, pipeline, columns, note):
    values = cloud()
    values[::50, 4] = np.nan
    values[25::50, 0] = np.nan
    path = tmp_path / "missing.npy"
    np.save(path, values)
    want = in_memory(values, pipeline, columns)
    result, got = in_blocks(path, pipeline, columns, 3_000)
    for name in columns:
        assert same_bits(got[name], want[name]), name
    if note is not None:
        assert note in repr(result)


def test_many_small_groups_merge(tmp_path):
    rng = np.random.default_rng(1)
    values = (rng.uniform(0, 1000, size=(100_000, 5))).astype("float32")
    values[:, 4] = rng.integers(0, 3, 100_000)
    path = tmp_path / "sparse.npy"
    np.save(path, values)
    pipeline = (group_by(cells(x, 10), cells(y, 10), cells(z, 100), axis="voxel")
                >> summarise(points=n(), brightness=mean(intensity)))
    columns = ["voxel", "x_cell", "y_cell", "z_cell", "points", "brightness"]
    want = in_memory(values, pipeline, columns)
    assert len(want["voxel"]) > 50_000
    _, got = in_blocks(path, pipeline, columns, 20_000, threads=2)
    for name in columns:
        assert same_bits(got[name], want[name]), name


def test_values_outside_every_bin_are_group_by_s_own_error(npy):
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    with pytest.raises(GroupError, match="fall outside every bin"):
        frame >> group_by(bins(intensity, [0, 10]))


ROWS = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60) >> select("x", "distance")


@pytest.mark.parametrize("rows", [7, 3_000, 1_000_000])
def test_rows_kept_are_joined_with_their_positions(npy, rows):
    path, values = npy
    want = in_memory(values, ROWS, ["point", "x", "distance"])
    result, got = in_blocks(path, ROWS, ["point", "x", "distance"], rows, threads=2)
    for name in ("point", "x", "distance"):
        assert same_bits(got[name], want[name]), name
    assert "(all removed by" not in repr(result)
    assert "2. filter(distance <= 60)" in str(result.explain())


def test_rows_kept_with_the_axis_s_own_labels_and_a_verb_after(npy):
    path, values = npy
    labels = np.arange(20_000) * 10
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES, "point": labels},
                       field_axis="feature", blocks=3_000)
    reference = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES, "point": labels},
                           field_axis="feature")
    pipeline = ROWS >> take(point=[1, 2, 3])  # slice needs the whole axis: the rows are joined first
    for name in ("point", "x"):
        assert same_bits(frame >> pipeline >> pull(name), reference >> pipeline >> pull(name)), name


def test_no_rows_kept_anywhere(tmp_path, npy):
    path, values = npy
    pipeline = filter(x > 1e9)
    want = in_memory(values, pipeline, ["x"])
    result, got = in_blocks(path, pipeline, ["x"], 3_000)
    assert got["x"].shape == want["x"].shape == (0,)
    assert "point: 0 (all removed by filter(" in repr(result)
