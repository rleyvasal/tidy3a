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
    TidyError,
    all,
    any,
    bins,
    cells,
    col,
    dot,
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
    quantile,
    read_array,
    rounded,
    scan,
    select,
    sqrt,
    std,
    sum,
    summarise,
    tidy,
    var,
    where,
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
        # NaN among objects is equal to NaN: list equality would ask whether both are the same NaN object, which
        # depends on how NumPy and Python made them (they differ on Python 3.11 and the minimum NumPy).
        return (a.dtype.kind in "OU" and b.dtype.kind in "OU" and a.shape == b.shape
                and all(x == y or (x != x and y != y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())))
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
    assert "nothing read yet: 7 blocks of 3,000 positions on point" in text
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


def test_verbs_are_checked_on_the_empty_block_and_nothing_is_read_until_a_result(npy):
    """Plan decision 4: a pipeline is built without a read; each verb's eager checks run on a block of no positions,
    and sizes the data decides show as ?."""
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    source, opened = frame._blocks.source, []
    reopen = source.reopen
    source.reopen = lambda: opened.append(1) or reopen()
    text = repr(frame)
    assert "nothing read yet" in text and "values (point: ?, feature: 5) float32" in text
    assert "blocks: 7 of 3,000 positions on point" in text
    with pytest.raises(MutateError, match='"wrong"'):
        frame >> mutate(d=fields("wrong") + 1)
    summary = frame >> mutate(d=x + 1) >> filter(col("d") > 0) >> group_by(ring) >> summarise(points=n())
    plan = str(summary.explain())
    assert "nothing read yet: 7 blocks of 3,000 positions on point" in plan
    assert "1. mutate(d = x + 1)" in plan and "point: ? → ?" not in plan and "+ axis ring: ?" in plan
    assert "all removed by" not in plan + repr(summary)  # the empty block's emptiness writes no note
    assert opened == []  # nothing read while the pipeline was built
    assert len(frame >> pull("x")) == 20_000  # every block, joined
    assert len(opened) == 7


@pytest.mark.parametrize("pipeline, message", [
    (mutate(centred=intensity - mean(intensity)) >> group_by(ring) >> summarise(points=n()), "whole axis"),
    (group_by(ring) >> mutate(centred=intensity - mean(intensity)), "after group_by, only summarise"),
    (group_by("point") >> summarise(points=n()), "keeps an axis"),
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


def test_values_outside_every_bin_are_group_by_s_own_error_naming_the_block(npy):
    """A value-level error is found when the blocks run (the empty block has no values), and names its block."""
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    grouped = frame >> group_by(bins(intensity, [0, 10])) >> summarise(points=n())
    with pytest.raises(GroupError, match=r"(?s)fall outside every bin.*in block 1 of 7"):
        grouped >> pull("points")


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


# ── Phase 3: what combines closely, not bit for bit ─────────────────────────


@pytest.mark.parametrize("make, rows", [(lambda: cloud(), 3_000), (lambda: cloud().astype("float64"), 997),
                                        (lambda: np.rint(cloud()).astype("int32"), 7_000)])
def test_spread_and_flags_combine(tmp_path, make, rows):
    values = make()
    if values.dtype.kind == "f":
        values[::40, 3] = np.nan
    path = tmp_path / "spread.npy"
    np.save(path, values)
    pipeline = (group_by(ring) >> summarise(
        points=n(), spread=std(intensity), v=var(intensity), spread_present=std(intensity, na_rm=True),
        v_present=var(intensity, na_rm=True), some=any(intensity > 10), every=all(z > -200),
        total=sum(intensity, min_count=3), total_present=sum(intensity, na_rm=True, min_count=600),
        average=mean(intensity, na_rm=True)))
    columns = ["ring", "points", "spread", "v", "spread_present", "v_present", "some", "every", "total",
               "total_present", "average"]
    want = in_memory(values, pipeline, columns)
    _, got = in_blocks(path, pipeline, columns, rows, threads=2)
    for name in ("ring", "points", "some", "every"):
        assert same_bits(got[name], want[name]), name
    for name in ("spread", "v", "spread_present", "v_present", "total", "total_present", "average"):
        assert got[name].dtype == want[name].dtype, name
        assert np.allclose(got[name], want[name], rtol=1e-5, equal_nan=True), name
    assert np.isnan(want["total_present"]).any()  # min_count took effect in memory and on blocks alike
    if values.dtype == np.float32:
        assert same_bits(got["total"], want["total"]) and same_bits(got["average"], want["average"])


def test_a_block_cannot_answer_over(npy):
    path, _ = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    with pytest.raises(BlockError, match="names axes"):
        frame >> summarise(total=sum(intensity, over="point"))


# ── Milestone 11, step 1: blocks over any source ────────────────────────────


def dataset(values):
    import xarray as xr

    return xr.Dataset({name: ("point", values[:, column]) for column, name in enumerate(NAMES)})


def test_an_array_in_memory_runs_in_blocks(npy):
    _, values = npy
    frame = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000, threads=2)
    assert "blocks: 7 of 3,000 positions on point" in repr(frame)
    want = in_memory(values, PIPELINE, COLUMNS)
    out = frame >> PIPELINE
    for name in COLUMNS:
        assert same_bits(out >> pull(name), want[name]), name
    rows = frame >> ROWS
    assert same_bits(rows >> pull("point"), in_memory(values, ROWS, ["point"])["point"])


def test_a_dataset_runs_in_blocks(npy):
    _, values = npy
    pytest.importorskip("xarray")
    source = dataset(values)
    pipeline = (mutate(distance=sqrt(x * x + y * y + z * z)) >> filter(distance <= 60) >> group_by(ring)
                >> summarise(points=n(), brightness=mean(intensity), low=min(z)))
    columns = ["ring", "points", "brightness", "low"]
    want = tidy(source) >> pipeline
    frame = tidy(source, blocks={"point": 3_000}, threads=2)
    assert "nothing read yet: 7 blocks of 3,000 positions on point" in str(frame.explain())
    out = frame >> pipeline
    for name in columns:
        assert same_bits(out >> pull(name), want >> pull(name)), name
    joined = tidy(source, blocks={"point": 3_000}) >> filter(x > 0) >> select("x", "ring")
    assert same_bits(joined >> pull("x"), tidy(source) >> filter(x > 0) >> pull("x"))


def chunked_dataset(values, chunk):
    """Each variable in Dask chunks of *chunk* rows, each chunk read from a delayed function that counts its reads."""
    da = pytest.importorskip("dask.array")
    import xarray as xr
    from dask import delayed

    reads = []

    @delayed
    def read(name, start, stop):
        reads.append((name, start))
        return values[start:stop, NAMES.index(name)]

    variables = {}
    for name in NAMES:
        blocks = [da.from_delayed(read(name, start, min(start + chunk, len(values))),
                                  shape=(min(start + chunk, len(values)) - start,), dtype=values.dtype)
                  for start in range(0, len(values), chunk)]
        variables[name] = ("point", da.concatenate(blocks))
    return xr.Dataset(variables), reads


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("blocks", [{"point": "chunks"}, "chunks"])
def test_a_dask_backed_dataset_reads_each_chunk_once(npy, blocks, threads):
    _, values = npy
    source, reads = chunked_dataset(values, 3_000)
    pipeline = filter(x > 0) >> group_by(ring) >> summarise(points=n(), brightness=mean(intensity))
    frame = tidy(source, blocks=blocks, threads=threads)
    assert "blocks: 7 on point, the data's own chunks" in repr(frame)
    assert reads == []  # nothing is read while the pipeline is built
    out = frame >> pipeline
    brightness = out >> pull("brightness")
    from collections import Counter

    read = ("x", "intensity", "ring")  # the variables the pipeline reads: y and z are never computed (milestone 12)
    every = Counter((name, start) for name in read for start in range(0, 20_000, 3_000))
    assert Counter(reads) == every  # each chunk exactly once
    want = in_memory(values, pipeline, ["brightness"])["brightness"]
    assert same_bits(brightness, want)


def test_unequal_chunks(npy):
    da = pytest.importorskip("dask.array")
    _, values = npy
    source, _ = chunked_dataset(values, 7_000)
    source = source.chunk({"point": (5_000, 9_000, 6_000)})
    assert source.chunks["point"] == (5_000, 9_000, 6_000)
    out = tidy(source, blocks="chunks") >> group_by(ring) >> summarise(points=n(), brightness=mean(intensity))
    for name in ("points", "brightness"):
        assert same_bits(out >> pull(name), in_memory(values, group_by(ring) >> summarise(
            points=n(), brightness=mean(intensity)), [name])[name]), name


def test_gridded_data_blocked_along_time():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    temperature = (rng.normal(size=(40, 5, 6)) * 10 + 280).astype("float32")
    temperature[3, 2, 2] = np.nan
    day = np.repeat(np.arange(8), 5)
    source = xr.Dataset({"temperature": (("time", "lat", "lon"), temperature), "day": ("time", day)},
                        coords={"lat": np.arange(5), "lon": np.arange(6) * 10})
    t, d = fields("temperature", "day")
    grouped = filter(d < 6) >> group_by(d) >> summarise(points=n(), warmth=mean(t, na_rm=True), top=max(t),
                                                         spread=std(t, na_rm=True))
    whole = summarise(points=n(), warmth=mean(t, na_rm=True), coldest=min(t, na_rm=True))
    for pipeline, columns in ((grouped, ["day", "points", "warmth", "top", "spread"]),
                              (whole, ["points", "warmth", "coldest"])):
        want = tidy(source) >> pipeline
        out = tidy(source, blocks={"time": 7}) >> pipeline
        for name in columns:
            got, expected = out >> pull(name), want >> pull(name)
            if name == "spread":
                assert got.dtype == expected.dtype and np.allclose(got, expected, rtol=1e-5), name
            else:
                assert same_bits(got, expected), name


def test_block_options_on_datasets_are_checked(npy):
    xr = pytest.importorskip("xarray")
    _, values = npy
    source = dataset(values)
    with pytest.raises(ReadError, match="not chunked along point"):
        tidy(source, blocks={"point": "chunks"})
    with pytest.raises(TidyError, match="one Dask-chunked axis"):
        tidy(source, blocks="chunks")
    with pytest.raises(TidyError, match='"row", which is not an axis'):
        tidy(source, blocks={"row": 100})
    with pytest.raises(TidyError, match="maps one axis"):
        tidy(source, blocks={"point": 100, "feature": 5})
    with pytest.raises(ReadError, match="positions"):
        tidy(source, blocks={"point": 0})
    path = npy[0]
    with pytest.raises(ReadError, match='blocks="chunks" takes'):
        read_array(path, dims={"point": infer, "feature": 5}, blocks="chunks")


# ── Milestone 11, step 2: several results from one pass ─────────────────────


def counting(counts):
    """A row-wise verb that counts the blocks it runs on and changes nothing."""
    from tidy3a.verbs import Verb

    def apply(frame):
        counts.append(1)
        return frame

    return Verb("counting()", apply, operation="mutate", arguments=[])


RINGS = group_by(ring) >> summarise(points=n(), low=min(z), high=max(z))
GRID = group_by(cells(x, 20.0), rounded(y, 25), axis="cell") >> summarise(points=n(), top=max(intensity))


def test_the_shared_prefix_runs_once_per_block(npy, monkeypatch):
    monkeypatch.setenv("TIDY3A_COMPILED", "0")  # the NumPy walk's sharing: a compiled frame does not walk its verbs
    path, values = npy
    counts = []
    cloud_ = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                        blocks=3_000)
    near = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60)
    shared = cloud_ >> mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> counting(counts) >> filter(distance <= 60)
    assert counts == [1]  # on the first block, as the verb was called
    rings, grid = shared >> RINGS, shared >> GRID
    got_rings, got_grid = scan(rings, grid)
    assert counts == [1] * 8  # once more per block, for both frames
    assert rings._result is got_rings and grid._result is got_grid
    assert "+ axis ring: 41" in str(rings.explain())  # the frames hold their results, as after collect()
    again = scan(rings, grid)  # computed frames are given as they are
    assert again[0] is got_rings and again[1] is got_grid
    assert counts == [1] * 8
    for name in ("ring", "points", "low", "high"):
        assert same_bits(got_rings >> pull(name), in_memory(values, near >> RINGS, [name])[name]), name
    for name in ("cell", "points", "top"):
        assert same_bits(got_grid >> pull(name), in_memory(values, near >> GRID, [name])[name]), name


@pytest.mark.parametrize("threads", [1, 3])
def test_a_joining_frame_beside_summarising_ones(npy, threads):
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000, threads=threads)
    near = frame >> mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60)
    rows = near >> select("x", "distance")  # a leaf that joins its rows
    rings = near >> RINGS  # hangs off the same prefix
    whole = frame >> summarise(points=n(), far=max(x))  # at the root, with no row-wise verb
    got_rows, got_rings, got_whole = scan(rows, rings, whole)
    assert same_bits(got_rows >> pull("point"), in_memory(values, ROWS, ["point"])["point"])
    assert same_bits(got_rows >> pull("distance"), in_memory(values, ROWS, ["distance"])["distance"])
    assert same_bits(got_rings >> pull("points"), in_memory(values, near.plan() >> RINGS, ["points"])["points"])
    assert got_whole >> pull("points") == len(values)
    assert same_bits(got_whole >> pull("far"), values[:, 0].max())


def test_a_frame_that_is_another_s_prefix(npy):
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    near = frame >> mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 60)
    rings = near >> RINGS
    got_near, got_rings, again = scan(near, rings, near)  # the same frame twice runs once
    assert got_near is again
    assert same_bits(got_near >> pull("distance"), in_memory(values, ROWS, ["distance"])["distance"])
    assert same_bits(got_rings >> pull("low"), in_memory(values, near.plan() >> RINGS, ["low"])["low"])


@pytest.mark.parametrize("pairs", [None, 1])
@pytest.mark.parametrize("rows", [97, 3_000])
def test_the_exact_pass_is_shared_and_gives_the_same_bits(npy, monkeypatch, pairs, rows):
    """Two branches whose float32 sums the error bound cannot settle (the tie group) get their exact sums from one
    more pass; with EXACT_PAIRS at 1, a pass takes one combination, so each branch's tie group gets a pass of its own."""
    from labelarray import sums as labelarray_sums

    monkeypatch.setenv("TIDY3A_COMPILED", "0")  # passes counted by the NumPy walk; a compiled first pass is not walked

    if pairs is not None:
        monkeypatch.setattr(labelarray_sums, "EXACT_PAIRS", pairs)
    path, values = npy
    counts = []
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=rows)
    shared = frame >> counting(counts) >> filter(intensity > -(2**31))
    half = fields("half")
    by_ring = group_by(ring) >> summarise(total=sum(intensity), brightness=mean(intensity))
    by_half = mutate(half=intensity * 0.5) >> group_by(ring) >> summarise(total=sum(half), points=n())
    totals, halves = shared >> by_ring, shared >> by_half
    got_totals, got_halves = scan(totals, halves)
    blocks = -(-len(values) // rows)
    passes = (len(counts) - 1) // blocks  # the first block, then each block once per pass
    assert len(counts) == 1 + passes * blocks
    assert passes == 2 if pairs is None else passes >= 3  # one shared second pass, or one per combination
    want = in_memory(values, filter(intensity > -(2**31)) >> by_ring, ["ring", "total", "brightness"])
    for name in ("ring", "total", "brightness"):
        assert same_bits(got_totals >> pull(name), want[name]), name
    want = in_memory(values, filter(intensity > -(2**31)) >> by_half, ["total", "points"])
    for name in ("total", "points"):
        assert same_bits(got_halves >> pull(name), want[name]), name


def test_scan_reads_each_dask_chunk_once_for_several_frames(npy):
    _, values = npy
    source, reads = chunked_dataset(values, 3_000)
    frame = tidy(source, blocks="chunks", threads=2)
    near = frame >> mutate(distance=sqrt(x * x + y * y + z * z)) >> filter(distance <= 60)
    rings = near >> group_by(ring) >> summarise(points=n(), low=min(z))
    grid = near >> group_by(cells(x, 20.0), rounded(y, 25), axis="cell") >> summarise(points=n())
    rows = near >> select("x", "ring")
    del reads[:]
    got_rings, got_grid, got_rows = scan(rings, grid, rows)
    from collections import Counter

    read = ("x", "y", "z", "ring")  # what the three frames read between them; intensity is never computed
    assert Counter(reads) == Counter((name, start) for name in read for start in range(0, 20_000, 3_000))
    whole = tidy(source) >> near.plan()
    assert same_bits(got_rings >> pull("low"), whole >> group_by(ring) >> summarise(low=min(z)) >> pull("low"))
    assert same_bits(got_grid >> pull("points"), whole >> group_by(cells(x, 20.0), rounded(y, 25), axis="cell")
                     >> summarise(points=n()) >> pull("points"))
    assert same_bits(got_rows >> pull("x"), whole >> pull("x"))


def test_scan_refuses_frames_from_different_sources_and_frames_in_memory(npy):
    path, values = npy
    options = dict(dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")
    one = read_array(path, blocks=3_000, **options) >> summarise(points=n())
    other = read_array(path, blocks=5_000, **options) >> summarise(points=n())
    with pytest.raises(BlockError, match="different read_array\\(\\) or tidy\\(\\) calls"):
        scan(one, other)
    with pytest.raises(BlockError, match="frames read with blocks=, not TidyArray"):
        scan(one, read_array(values, **options))
    assert scan() == ()
    (got,) = scan(one)
    assert got >> pull("points") == len(values)


# ── Review of 3c453fd: eager and block equivalence, and bounded memory ───────


def test_positions_are_the_whole_axis_s_in_every_block(npy):
    """A block sees its own positions: col(axis) counts from the block's start, and a filter by position keeps the
    same rows as in memory; without a filter, the joined rows have no labels, as in memory."""
    path, values = npy
    options = dict(dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")
    eager = read_array(values, **options)
    for source in (path, values):
        frame = read_array(source, blocks=3_000, **options)
        assert "values (point: ?, feature: 5)" in repr(frame)  # nothing read yet
        positions = mutate(p=col("point")) >> select("p")
        assert same_bits(frame >> positions >> pull("p"), eager >> positions >> pull("p"))
        assert not (frame >> positions).collect().coords  # no filter ran, so the axis stays unlabelled
        by_position = filter((col("point") > 7_000) & (col("point") <= 9_010)) >> select("x")
        got, want = frame >> by_position, eager >> by_position
        assert same_bits(got >> pull("x"), want >> pull("x")) and same_bits(got >> pull("point"), want >> pull("point"))


@pytest.mark.parametrize("pipeline, message", [
    (mutate(t=dot(x, x)), "dot\\(x, x\\) sums over point"),
    (mutate(t=dot(x, y, over="point")), "sums over point"),
    (filter(x > dot(x, x) / 100), "sums over point"),
    (mutate(t=formula(r"\sum_p x_p", bind={"p": "point"})), "sum\\(x, over=\"point\"\\) needs the whole axis"),
    (mutate(t=formula(r"x - \sum_p x_p / 3", bind={"p": "point"})), "needs the whole axis"),
])
def test_sums_hidden_in_dot_and_formulas_are_refused(npy, pipeline, message):
    frame = read_array(npy[0], dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    with pytest.raises(BlockError, match=message):
        frame >> pipeline


def test_integer_means_add_in_float64_and_integer_sums_wrap_as_in_memory():
    xr = pytest.importorskip("xarray")
    v = fields("v")
    source = xr.Dataset({"v": ("point", np.array([2**62] * 4 + [1, 2, 3], dtype="int64"))})
    pipeline = summarise(m=mean(v), t=sum(v))
    eager, blocked = tidy(source) >> pipeline, tidy(source, blocks={"point": 2}) >> pipeline
    assert same_bits(blocked >> pull("m"), eager >> pull("m"))
    assert same_bits(blocked >> pull("t"), eager >> pull("t"))  # wrapped the same way


def test_joined_rows_keep_coordinates_attributes_and_encodings():
    xr = pytest.importorskip("xarray")
    v = fields("v")
    source = xr.Dataset({"v": ("time", np.arange(6.0), {"units": "K"}), "site": ((), "A", {"role": "name"})},
                        coords={"station": ("time", list("aabbcc"), {"kind": "id"}), "time": np.arange(6)},
                        attrs={"title": "t"})
    source["v"].encoding["dtype"] = "float32"
    pipeline = filter(v > 0) >> mutate(w=v * 2)
    eager, blocked = (tidy(source) >> pipeline).collect(), (tidy(source, blocks={"time": 2}) >> pipeline).collect()
    xr.testing.assert_identical(blocked, eager)
    assert blocked.attrs == eager.attrs == {"title": "t"}
    assert blocked["v"].attrs == {"units": "K"} and blocked["station"].attrs == eager["station"].attrs
    assert blocked["v"].encoding == eager["v"].encoding


def test_a_frame_s_field_axis_and_backend_carry_into_blocks_and_its_groups_are_refused(npy):
    _, values = npy
    frame = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")
    blocked = tidy(frame, blocks={"point": 3_000})
    assert blocked._backend is frame._backend and blocked._field_axis == "feature"
    want = in_memory(values, PIPELINE, COLUMNS)
    out = blocked >> PIPELINE
    for name in COLUMNS:
        assert same_bits(out >> pull(name), want[name]), name
    with pytest.raises(TidyError, match="ungroup\\(\\) first"):
        tidy(frame >> group_by(ring), blocks={"point": 3_000})
    with pytest.raises(TidyError, match='already has field axis "feature"'):
        tidy(frame, blocks={"point": 3_000}, field_axis="point")


def test_a_group_of_missing_values_first_seen_in_a_later_block(tmp_path):
    values = np.array([[1, 1.0], [1, np.nan], [2, np.nan], [2, np.nan], [3, 5.0], [3, 6.0], [4, np.nan]],
                      dtype="float32")
    path = tmp_path / "missing.npy"
    np.save(path, values)
    g, v = fields("g", "v")
    options = dict(dims={"point": infer, "feature": 2}, labels={"feature": ["g", "v"]}, field_axis="feature")
    for na_rm in (True, False):
        pipeline = group_by(g) >> summarise(lo=min(v, na_rm=na_rm), hi=max(v, na_rm=na_rm))
        eager = read_array(values, **options) >> pipeline
        for rows in (1, 2, 3):
            blocked = read_array(path, blocks=rows, **options) >> pipeline
            for name in ("lo", "hi"):
                assert same_bits(blocked >> pull(name), eager >> pull(name)), (na_rm, rows, name)


def test_a_single_bins_key_lists_every_bin_as_in_memory(tmp_path):
    values = np.array([[1, 1.0], [1, np.nan], [2, 5.0], [2, 6.0], [3, 13.0]], dtype="float32")
    path = tmp_path / "bins.npy"
    np.save(path, values)
    g, v = fields("g", "v")
    options = dict(dims={"point": infer, "feature": 2}, labels={"feature": ["g", "v"]}, field_axis="feature")
    summary = summarise(c=n(), s=sum(v), m=mean(v), lo=min(v), hi=max(v), sd=std(v), a=any(v > 0), al=all(v > 0),
                        s2=sum(v, na_rm=True), lo2=min(v, na_rm=True))
    columns = ["v_bin", "c", "s", "m", "lo", "hi", "sd", "a", "al", "s2", "lo2"]
    edges = [0, 2, 4, 6, 8, 10, 12, 14]
    for pipeline in (group_by(bins(v, edges)) >> summary, group_by(bins(v, edges), dropna=False) >> summary):
        eager = read_array(values, **options) >> pipeline
        for rows in (1, 2, 5):
            blocked = read_array(path, blocks=rows, **options) >> pipeline
            for name in columns:
                got, want = blocked >> pull(name), eager >> pull(name)
                if name == "sd":
                    assert got.dtype == want.dtype and np.allclose(got, want, equal_nan=True), (rows, name)
                else:
                    assert same_bits(got, want), (rows, name)
    several = group_by(bins(v, edges), g, axis="cell") >> summarise(c=n())  # several keys: the combinations seen
    assert same_bits(read_array(path, blocks=2, **options) >> several >> pull("c"),
                     read_array(values, **options) >> several >> pull("c"))


def test_variables_with_different_dask_chunks_take_an_explicit_block_size():
    da = pytest.importorskip("dask.array")
    xr = pytest.importorskip("xarray")
    a, b = fields("a", "b")
    source = xr.Dataset({"a": ("point", da.from_array(np.arange(10.0), chunks=2)),
                         "b": ("point", da.from_array(np.arange(10.0), chunks=5))})
    out = tidy(source, blocks={"point": 4}) >> summarise(t=sum(a + b))
    assert out >> pull("t") == 90.0
    with pytest.raises(ReadError, match="not chunked along point"):
        tidy(source, blocks={"point": "chunks"})


def test_blocks_started_ahead_of_the_one_consumed_next_are_bounded():
    """A slow first block does not let the pool run ahead through every block: at most threads + 1 are started before
    the first result is taken, so results waiting to be merged in order stay bounded."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tidy3a.chunked import _in_order

    release, started = threading.Event(), []

    def work(item):
        started.append(item)
        if item == 0:
            release.wait(5)
        return item

    with ThreadPoolExecutor(2) as pool:
        results = _in_order(pool, work, list(range(10)))
        taken = []
        consumer = threading.Thread(target=lambda: taken.extend(results))
        consumer.start()
        deadline = threading.Event()
        deadline.wait(0.3)
        assert sorted(started) == [0, 1, 2]  # threads + 1, the rest waiting on the first
        release.set()
        consumer.join(5)
    assert taken == list(range(10))


def test_exact_sums_are_finished_as_each_pass_ends(npy, monkeypatch):
    from labelarray import sums as labelarray_sums

    monkeypatch.setenv("TIDY3A_COMPILED", "0")  # the exact passes' requests, counted: compiled, the ties need none

    finished = []
    real = labelarray_sums.exact_results

    def counting(buckets, op, na_rm):
        finished.append(buckets.nbytes)
        return real(buckets, op, na_rm)

    monkeypatch.setattr(labelarray_sums, "exact_results", counting)
    frame = read_array(npy[0], dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    totals = frame >> group_by(ring) >> summarise(total=sum(intensity))
    halves = frame >> group_by(ring) >> summarise(total=sum(intensity * 0.5))
    got, _ = scan(totals, halves)
    assert len(finished) == 2  # each request's buckets finished within its pass, not kept until the end
    assert same_bits(got >> pull("total"), in_memory(npy[1], group_by(ring) >> summarise(total=sum(intensity)),
                                                     ["total"])["total"])


# ── Decision 4, step 4: an error while a block runs names the block ─────────


@pytest.mark.parametrize("threads", [1, 3])
def test_an_error_in_a_later_block_names_the_block(tmp_path, threads):
    """The first block passes every check; a value only block 3 holds fails there, and the error says where."""
    values = cloud(4_000)
    values[:, 2] = np.clip(values[:, 2], -50, 50)
    values[2_500, 2] = 500  # outside every bin, in positions 2,001-3,000
    path = tmp_path / "late.npy"
    np.save(path, values)
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=1_000, threads=threads)
    summary = frame >> group_by(bins(z, [-100, 0, 100])) >> summarise(points=n())  # the first block is fine
    with pytest.raises(GroupError, match=r"(?s)fall outside every bin.*\n  in block 3 of 4 \(positions 2,001–3,000 on point\)"):
        summary >> pull("points")
    rows = frame >> mutate(cell=formula(r"x")) >> group_by(bins(z, [-100, 0, 100])) >> summarise(points=n())
    with pytest.raises(GroupError, match="in block 3 of 4"):
        scan(rows, frame >> summarise(points=n()))


# ── Decision 4, step 1: an empty block from each source, without a read ─────


def schema(frame):
    """Each variable's axes and type, and the axes' sizes, of a frame."""
    backend = frame._backend
    return ({str(name): (tuple(map(str, variable.dims)), variable.dtype.str)
             for name, variable in backend.variables(frame._data).items()},
            {str(axis): size for axis, size in backend.sizes(frame._data).items()})


def test_an_empty_file_block_reads_nothing_and_has_the_file_s_schema(npy, tmp_path):
    path, values = npy
    big_endian = tmp_path / "big.npy"
    np.save(big_endian, values.astype(">f4"))
    for source_path in (path, big_endian):
        frame = read_array(source_path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES},
                           field_axis="feature", blocks=3_000)
        source = frame._blocks.source
        opened = []
        reopen = source.reopen
        source.reopen = lambda: opened.append(1) or reopen()
        empty = source.empty()
        assert opened == []  # the file is not even mapped
        variables, sizes = schema(empty)
        first_variables, first_sizes = schema(source.block(0, 3_000))
        assert variables == first_variables and sizes == {**first_sizes, "point": 0}


def test_an_empty_array_block_has_the_array_s_schema(npy):
    _, values = npy
    frame = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    variables, sizes = schema(frame._blocks.source.empty())
    assert variables == schema(frame)[0] and sizes == {"point": 0, "feature": 5}


@pytest.mark.parametrize("as_array", [False, True])
def test_an_empty_dask_block_computes_no_chunk(npy, as_array):
    xr = pytest.importorskip("xarray")
    _, values = npy
    source, reads = chunked_dataset(values, 3_000)
    source = source.assign_coords(when=("point", source["x"].data * 0))  # a lazy coordinate along the axis
    data = source["x"] if as_array else source
    frame = tidy(data, blocks="chunks")
    del reads[:]
    empty = frame._blocks.source.empty()
    assert reads == []  # isel(point=slice(0, 0)).compute() would read the first chunk
    variables, sizes = schema(empty)
    want = schema(frame._blocks.source.block(0, 3_000))
    assert variables == want[0] and sizes == {**want[1], "point": 0}
    assert isinstance(xr, object)


def test_the_block_verbs_on_an_empty_block_give_the_types_of_a_run(npy):
    """What decision 4 checks verbs on: a summary's type on no positions is its type on data. A text key's labels are
    the exception, numbers on no positions, so a key's label type will come from its transform (pinned here, so the
    test fails when that changes)."""
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    empty, first = frame._blocks.source.empty(), frame._blocks.source.block(0, 3_000)
    for pipeline, labels in ((PIPELINE, set()), (ROWS, set()), (VOXELS, {"x_cell", "z_cell"})):
        on_nothing, on_data = schema(empty >> pipeline)[0], schema(first >> pipeline)[0]
        assert {name: kind for name, kind in on_nothing.items() if name not in labels} == \
            {name: kind for name, kind in on_data.items() if name not in labels}
        for name in labels:
            assert on_data[name][1] == "|O" and on_nothing[name][1] != "|O"


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_min_and_max_of_times_text_and_empty_bins_on_blocks(rows):
    """The merge filled unseen groups with np.iinfo's extremes, which times and text have none of, and an integer min
    of an empty bin showed that fill rather than NaN."""
    xr = pytest.importorskip("xarray")
    t, g, w, d, k = fields("t", "g", "when", "d", "k")
    times = np.array(["2024-01-01", "2024-01-02", "NaT", "2024-01-04", "2024-01-05", "2024-01-06"],
                     dtype="datetime64[ns]")
    source = xr.Dataset({"t": ("point", np.array(list("bacdef"))), "g": ("point", [1.0, 1, 2, 2, 5, 5]),
                         "when": ("point", times), "d": ("point", times - times[0]),
                         "k": ("point", np.array([3, 1, 4, 1, 5, 9], dtype="int32"))})
    for pipeline in (group_by(g) >> summarise(lo=min(w), hi=max(w, na_rm=True), short=min(d), first=min(t),
                                              last=max(t), least=min(k)),
                     group_by(bins(g, [0, 1.5, 3, 4, 6])) >> summarise(least=min(k), last=max(t), first=min(w))):
        want = tidy(source) >> pipeline
        got = tidy(source, blocks={"point": rows}) >> pipeline
        for name in want._backend.variables(want._data):
            assert same_bits(got >> pull(name), want >> pull(name)), name


# ── Decision 4, step 3: preview() runs a sample ─────────────────────────────


def test_preview_runs_the_first_middle_and_last_blocks_and_leaves_the_frame_alone(npy, monkeypatch):
    monkeypatch.setenv("TIDY3A_COMPILED", "0")  # the tie's exact pass, counted below: compiled, it is rounded exact
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    source, opened = frame._blocks.source, []
    reopen = source.reopen
    source.reopen = lambda: opened.append(1) or reopen()
    summary = frame >> PIPELINE
    sample = summary.preview()
    assert len(opened) == 3 * 2  # blocks 1, 4, and 7, twice: block 1's tie group needs the exact pass
    rows = np.concatenate([values[0:3_000], values[9_000:12_000], values[18_000:20_000]])
    want = in_memory(rows, PIPELINE, COLUMNS)
    for name in COLUMNS:
        assert same_bits(sample >> pull(name), want[name]), name
    words = "a sample: blocks 1, 4, 7 of 7 (8,000 of 20,000 positions on point); collect() runs every block"
    assert f"preview: {words}" in sample._notes
    assert words in str(sample.explain())
    assert summary._result is None  # a sample is never the frame's result
    assert same_bits(summary >> pull("points"), in_memory(values, PIPELINE, ["points"])["points"])
    assert len(opened) == 3 * 2 + 7 * 2


def test_preview_of_named_blocks_and_of_rows(npy):
    path, values = npy
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=3_000)
    rows = (frame >> ROWS).preview(blocks=[2, 5])
    eager = read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": NAMES},
                       field_axis="feature") >> ROWS
    kept = eager >> pull("point")
    inside = ((kept > 3_000) & (kept <= 6_000)) | ((kept > 12_000) & (kept <= 15_000))
    assert same_bits(rows >> pull("point"), kept[inside])  # the whole axis's positions
    assert same_bits(rows >> pull("distance"), (eager >> pull("distance"))[inside])
    assert same_bits((frame >> PIPELINE).preview(blocks=3) >> pull("points"),
                     in_memory(values[6_000:9_000], PIPELINE, ["points"])["points"])
    for wrong in ([0], [8], [], [1.5]):
        with pytest.raises(BlockError, match="block numbers from 1 to 7"):
            (frame >> ROWS).preview(blocks=wrong)


def test_an_error_in_a_sample_names_the_block_among_every_block(tmp_path):
    values = cloud(4_000)
    values[:, 2] = np.clip(values[:, 2], -50, 50)
    values[2_500, 2] = 500
    path = tmp_path / "late.npy"
    np.save(path, values)
    frame = read_array(path, dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature",
                       blocks=1_000)
    summary = frame >> group_by(bins(z, [-100, 0, 100])) >> summarise(points=n())
    assert (summary.preview(blocks=[1, 2]) >> pull("points")).sum() == 2_000
    with pytest.raises(GroupError, match="in block 3 of 4"):
        summary.preview(blocks=[3])


# ── Milestone 12, step 1: a block holds only the variables read ─────────────


def counted_grid():
    """A gridded dataset of delayed chunk reads, counted per variable: a field along (time, lat, lon), and two along
    time only, as hour and month are in climate data."""
    da = pytest.importorskip("dask.array")
    xr = pytest.importorskip("xarray")
    from dask import delayed

    rng = np.random.default_rng(5)
    values = {"temperature": (rng.normal(size=(48, 4, 5)) * 10 + 280).astype("float32"),
              "wind": rng.normal(size=(48, 4, 5)).astype("float32"),
              "hour": np.tile(np.arange(24), 2).astype("int8"), "day": np.repeat(np.arange(2), 24).astype("int8")}
    reads = []

    @delayed
    def read(name, start, stop):
        reads.append(name)
        return values[name][start:stop]

    variables = {}
    for name, array in values.items():
        dims = ("time", "lat", "lon")[:array.ndim]
        parts = [da.from_delayed(read(name, start, start + 12), shape=(12, *array.shape[1:]), dtype=array.dtype)
                 for start in range(0, 48, 12)]
        variables[name] = (dims, da.concatenate(parts))
    return xr.Dataset(variables, coords={"lat": np.arange(4), "lon": np.arange(5)}), values, reads


def test_a_summary_reads_only_its_variables_and_n_counts_as_in_memory():
    """n() counts the axes every variable shares: here time only, as hour and day lie along it. A block holding
    only temperature would count grid cells; n() takes its axes from the whole schema, so it counts hours."""
    xr = pytest.importorskip("xarray")
    source, values, reads = counted_grid()
    t = fields("temperature")
    pipeline = summarise(points=n(), warmth=mean(t))
    got = tidy(source, blocks={"time": "chunks"}) >> pipeline
    want = tidy(xr.Dataset({name: (source[name].dims, array) for name, array in values.items()},
                           coords=source.coords)) >> pipeline
    del reads[:]
    assert same_bits(got >> pull("points"), want >> pull("points")) and (got >> pull("points")) == 48
    assert same_bits(got >> pull("warmth"), want >> pull("warmth"))
    assert sorted(set(reads)) == ["temperature"] and len(reads) == 4  # one chunk per block, of temperature only


def test_a_count_alone_reads_one_variable_along_the_axis():
    source, _, reads = counted_grid()
    got = tidy(source, blocks={"time": "chunks"}) >> summarise(points=n())
    del reads[:]
    assert (got >> pull("points")) == 48
    assert len(set(reads)) == 1 and set(reads) <= {"hour", "day"}  # the smallest along time


def test_keys_formulas_filters_and_kept_rows_are_read():
    source, values, reads = counted_grid()
    t, w, hour, day = fields("temperature", "wind", "hour", "day")
    frame = tidy(source, blocks={"time": "chunks"})
    summary = (frame >> filter(hour >= 6) >> mutate(gust=formula(r"\sqrt{\mathrm{wind}^2}")) >> group_by(day)
               >> summarise(top=max(col("gust"))))
    del reads[:]
    summary >> pull("top")
    assert set(reads) == {"hour", "wind", "day"}
    rows = frame >> filter(day == 1) >> select("temperature")
    del reads[:]
    rows >> pull("temperature")
    assert set(reads) == {"day", "temperature"}  # the filter's variable, and the variable the rows keep


def test_a_selector_that_looks_at_values_is_refused_before_anything_is_read():
    """A where() of the caller's would answer for no values on the empty block, and each block could pick other
    columns; it is refused when select() is called, and selecting by type still works."""
    source, _, reads = counted_grid()
    frame = tidy(source, blocks={"time": "chunks"})
    del reads[:]
    for item in (where(lambda data: True), ~where(lambda data: True), [where(lambda data: True)]):
        with pytest.raises(BlockError, match=r"picks columns by their values.*preview\(\) a sample to see them"):
            frame >> select(item)
    assert reads == []
    from tidy3a import is_numeric

    picked = frame >> select(where(is_numeric)) >> summarise(points=n())
    assert (picked >> pull("points")) == 48


def test_a_field_label_written_reads_its_holder_whole(npy):
    xr = pytest.importorskip("xarray")
    _, values = npy
    source = xr.Dataset({"values": (("point", "feature"), values), "extra": ("point", values[:, 0] * 2)},
                        coords={"feature": NAMES})
    frame = tidy(source, field_axis="feature", blocks={"point": 3_000})
    pipeline = mutate(x=x * 2) >> group_by(ring) >> summarise(far=max(x))
    want = tidy(source, field_axis="feature") >> pipeline
    got = frame >> pipeline
    assert same_bits(got >> pull("far"), want >> pull("far"))
    assert _needs_of(frame >> pipeline) == {"values"}  # the holder of x and ring; extra is never read


def _needs_of(frame):
    from tidy3a.chunked import _needs

    return set(_needs(frame))


def test_the_backend_is_chosen_once_for_the_whole_data():
    """Without a default, data with storage encodings stays on xarray; a block holding only an unencoded variable
    would go to NumPy if each block chose for itself. Every block takes the backend chosen for the whole data."""
    xr = pytest.importorskip("xarray")
    source = xr.Dataset({"a": ("point", np.arange(10.0)), "b": ("point", np.arange(10.0))})
    source["a"].encoding["dtype"] = "float32"
    b = fields("b")
    frame = tidy(source, blocks={"point": 4})
    out = frame >> summarise(total=sum(b))
    assert (out >> pull("total")) == 45.0
    assert frame._backend is tidy(source)._backend is out._run()._backend
    assert frame._blocks.source.block(0, 4, frozenset({"b"}))._backend is frame._backend


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("na_rm", [False, True])
def test_any_and_all_of_missing_values_on_blocks(rows, na_rm):
    """A missing value is left out as in memory (False for any, True for all), where NaN cast to bool was True; and
    without na_rm a group that had one is missing, float64 as in memory, where blocks gave bool."""
    xr = pytest.importorskip("xarray")
    v, g = fields("v", "g")
    floats = np.array([0.0, np.nan, 1.0, 0.0, 0.0, 2.0])
    durations = np.array([0, 1, 0, 0, 0, 2], dtype="timedelta64[ns]")
    durations[1] = np.timedelta64("NaT", "ns")
    for values in (floats, durations):
        source = xr.Dataset({"v": ("point", values), "g": ("point", [1.0, 1, 2, 2, 5, 5])})
        for op in (any, all):
            for pipeline in (group_by(g) >> summarise(s=op(v, na_rm=na_rm)),
                             group_by(bins(g, [0, 1.5, 3, 4, 6])) >> summarise(s=op(v, na_rm=na_rm))):
                want = tidy(source) >> pipeline >> pull("s")
                got = tidy(source, blocks={"point": rows}) >> pipeline >> pull("s")
                assert same_bits(got, want), (values.dtype, op, got, want)


# ── Milestone 13, step 3: the median on blocks, exactly ─────────────────────


@pytest.mark.parametrize("dtype, passes", [("float32", 2), ("float64", 4), ("int16", 2), ("int64", 4),
                                           ("timedelta64[ns]", 4)])
@pytest.mark.parametrize("na_rm", [False, True])
def test_the_median_on_blocks_is_np_median_s_and_costs_its_passes(tmp_path, dtype, passes, na_rm):
    """The median is found by counting each group's values a 16-bit digit of their key a pass: two passes after the
    first for 32-bit values, four for 64-bit ones; odd and even counts, ties, missing values, an empty bin, and a
    single-value group give np.median's answer, the same bits."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(7)
    count = 3_001
    if dtype.startswith("float"):
        values = (rng.normal(size=count) * 50).astype(dtype)
        values[rng.choice(count, 25, replace=False)] = np.nan
        values[:300] = np.round(values[:300]) + 0.0  # ties, without -0.0 (below)
    elif dtype.startswith("timedelta"):
        values = rng.integers(-10**12, 10**12, count).astype(dtype)
        values[:9] = np.timedelta64("NaT")
    else:
        values = rng.integers(-30_000, 30_000, count).astype(dtype)
    group = rng.integers(0, 5, count).astype(float)
    group[-1] = 40.5  # a group of one value
    source = xr.Dataset({"v": ("point", values), "g": ("point", group), "other": ("point", np.zeros(count))})
    path = tmp_path / "m.zarr"
    v, g = fields("v", "g")
    pipeline = group_by(bins(g, [0, 2, 2.5, 5, 50])) >> summarise(points=n(), middle=median(v, na_rm=na_rm))
    want = tidy(source) >> pipeline
    frame = tidy(source, blocks={"point": 250})
    loads = []
    load = frame._blocks.source.block
    frame._blocks.source.block = lambda start, stop, names=None: loads.append(names) or load(start, stop, names)
    got = frame >> pipeline
    assert same_bits(got >> pull("middle"), want >> pull("middle"))
    assert len(loads) == 13 * (1 + passes)  # the first pass, then one a digit
    assert all(names == frozenset({"v", "g"}) for names in loads)  # only what the median reads, every pass
    assert not path.exists()


def test_medians_of_two_frames_share_their_passes():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(3)
    source = xr.Dataset({"a": ("point", rng.normal(size=2_000).astype("float32")),
                         "b": ("point", rng.normal(size=2_000).astype("float32")),
                         "g": ("point", rng.integers(0, 4, 2_000))})
    a, b, g = fields("a", "b", "g")
    frame = tidy(source, blocks={"point": 500})
    loads = []
    load = frame._blocks.source.block
    frame._blocks.source.block = lambda start, stop, names=None: loads.append(1) or load(start, stop, names)
    first = frame >> group_by(g) >> summarise(m=median(a))
    second = frame >> group_by(g) >> summarise(m=median(b))
    got_first, got_second = scan(first, second)
    assert len(loads) == 4 * (1 + 2)  # both medians' two passes in the same rounds
    assert same_bits(got_first >> pull("m"), tidy(source) >> group_by(g) >> summarise(m=median(a)) >> pull("m"))
    assert same_bits(got_second >> pull("m"), tidy(source) >> group_by(g) >> summarise(m=median(b)) >> pull("m"))


def test_a_median_of_zeros_of_both_signs_is_zero():
    """-0.0 and 0.0 are equal and NumPy does not order them, so which one its partition leaves at the middle depends
    on the data's order; the block median orders -0.0 first. The value is the same; its sign may differ."""
    xr = pytest.importorskip("xarray")
    v = fields("v")
    source = xr.Dataset({"v": ("point", np.array([0.0, -0.0, -0.0, 0.0, 1.0, -1.0], dtype="float32"))})
    got = tidy(source, blocks={"point": 2}) >> summarise(middle=median(v)) >> pull("middle")
    want = tidy(source) >> summarise(middle=median(v)) >> pull("middle")
    assert got == want == 0.0 and got.dtype == want.dtype


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.73, 1.0])
def test_a_quantile_on_blocks_is_np_quantile_s_with_its_edges(q):
    """The two values around NumPy's virtual index q × (n − 1), found as the median finds its middle, give
    np.quantile's answer, the same bits: ties, infinities beside the rank (NaN, as in NumPy), missing values,
    an empty bin, a group of one."""
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(11)
    values = (rng.normal(size=2_001) * 30).astype("float32")
    values[:9] = np.nan
    values[9:200] = np.round(values[9:200]) + 0.0
    values[200:203], values[203] = np.inf, -np.inf
    group = rng.integers(0, 6, 2_001).astype(float)
    group[-1] = 30.5
    source = xr.Dataset({"v": ("point", values), "g": ("point", group)})
    v, g = fields("v", "g")
    for na_rm in (False, True):
        pipeline = group_by(bins(g, [0, 3, 3.5, 6, 40])) >> summarise(s=quantile(v, q, na_rm=na_rm))
        want = tidy(source) >> pipeline >> pull("s")
        frame = tidy(source, blocks={"point": 333})
        loads = []
        load = frame._blocks.source.block
        frame._blocks.source.block = lambda start, stop, names=None: loads.append(1) or load(start, stop, names)
        assert same_bits(frame >> pipeline >> pull("s"), want)
        assert len(loads) == 7 * (1 + 2)  # float32: two passes after the first
