"""``arrange`` and ``desc`` in memory (plan milestone 21, step 1): the positions of the one axis the keys vary along,
sorted by the first key, ties by the next, ties in their order; ``desc`` downwards; missing values last either way;
every variable along the axis reordered with its labels, on both backends."""

import numpy as np
import pytest

from tidy3a import ArrangeError, arrange, desc, fields, filter, group_by, infer, mutate, pull, read_array, tidy
from tidy3a.chunked import ChunkedFrame

xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")
rng = np.random.default_rng(12)
SIZE = 400
DATA = xr.Dataset({
    "f32": ("p", np.where(rng.random(SIZE) < 0.1, np.nan, rng.integers(-3, 4, SIZE)).astype(np.float32)),
    "f64": ("p", rng.normal(size=SIZE).round(1)),
    "i": ("p", rng.integers(-2, 3, SIZE)),
    "b": ("p", rng.random(SIZE) > 0.5),
    "when": ("p", np.where(rng.random(SIZE) < 0.1, np.datetime64("NaT"),
                           np.datetime64("2026-01-01") + rng.integers(0, 5, SIZE).astype("timedelta64[D]"))
             .astype("datetime64[ns]")),
    "word": ("p", rng.choice(["ant", "bee", "cat"], SIZE)),
    "grid": (("p", "q"), rng.normal(size=(SIZE, 3))),
    "other": ("q", np.arange(3.0)),
}, coords={"p": np.arange(SIZE) * 10, "q": ["u", "v", "w"]})
DATA["f32"][::37] = -0.0  # negative zeros, which tie with the positive
f32, f64, i, b, when, word = fields("f32", "f64", "i", "b", "when", "word")


def reference(keys):
    """pandas' stable sort, missing values last: the order every case must give."""
    frame = pd.DataFrame({name: DATA[name].values for name in ("f32", "f64", "i", "b", "when", "word")})
    names = [name for name, _ in keys]
    order = frame.sort_values(names, ascending=[not down for _, down in keys], kind="stable", na_position="last")
    return order.index.to_numpy()


CASES = {
    "a float32 key with NaN and both zeros": ([f32], [("f32", False)]),
    "downwards": ([desc(f32)], [("f32", True)]),
    "two keys": ([i, desc(f64)], [("i", False), ("f64", True)]),
    "three, one boolean": ([desc(b), f32, i], [("b", True), ("f32", False), ("i", False)]),
    "times with NaT, downwards": ([desc(when)], [("when", True)]),
    "text, then times": ([word, when], [("word", False), ("when", False)]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_arrange_gives_the_stable_order_missing_values_last(name):
    keys, ordered = CASES[name]
    got = (tidy(DATA) >> arrange(*keys)).collect()
    want = DATA.isel(p=reference(ordered))
    assert got["p"].values.tolist() == want["p"].values.tolist()  # the labels move with the positions
    for variable in DATA.data_vars:
        assert np.asarray(got[variable].values).tobytes() == np.asarray(want[variable].values).tobytes(), variable
    assert got["q"].values.tolist() == ["u", "v", "w"]  # another axis untouched


def test_an_expression_as_a_key():
    got = (tidy(DATA) >> arrange(desc(f64 * f64), fields("p"))).collect()
    square = DATA["f64"].values ** 2
    want = np.lexsort([DATA["p"].values, -square])
    assert got["p"].values.tolist() == DATA["p"].values[want].tolist()


def test_rows_of_a_field_axis():
    values = rng.normal(size=(50, 3)).astype(np.float32)
    cloud = read_array(values, dims={"point": infer, "feature": 3}, labels={"feature": ["x", "y", "z"]},
                       field_axis="feature")
    got = cloud >> arrange(desc(fields("y")))
    assert np.asarray(got >> pull(fields("y"))).tolist() == sorted(values[:, 1].tolist(), reverse=True)
    assert np.asarray(got >> pull(fields("x"))).tolist() == values[np.argsort(-values[:, 1], kind="stable"), 0].tolist()


def test_no_keys_change_nothing():
    got = (tidy(DATA) >> arrange()).collect()
    assert got["p"].values.tolist() == DATA["p"].values.tolist()


@pytest.mark.parametrize("keys, message", [
    ([fields("grid")], "varies along p and q"),
    ([f32, fields("other")], "the keys before it along p"),
    ([1], "single value"),
], ids=["two axes", "two keys on different axes", "a single value"])
def test_keys_with_no_one_axis_are_refused(keys, message):
    with pytest.raises(ArrangeError, match=message):
        tidy(DATA) >> arrange(*keys)


def test_desc_twice_and_grouped_data_are_refused():
    with pytest.raises(ArrangeError, match="already downwards"):
        desc(desc(f32))
    with pytest.raises(ArrangeError, match="ungroup"):
        tidy(DATA) >> group_by(i) >> arrange(f32)


BLOCKED = {  # verbs on data read in blocks, sorting the blocked axis (milestone 21, step 2)
    "one key": lambda: arrange(f32),
    "one key, downwards": lambda: arrange(desc(f32)),
    "two keys": lambda: arrange(i, desc(f64)),
    "times, downwards": lambda: arrange(desc(when)),
    "text, then a boolean": lambda: arrange(word, b),
    "after a filter, then a mutate": lambda: filter(i > -2) >> arrange(desc(i), f32) >> mutate(twice=f64 * 2),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(BLOCKED))
def test_a_sort_of_the_blocked_axis_sorts_each_block_and_merges_them(name, threads):
    """Each block sorted on its own, the blocks merged, earlier ones first among ties: memory's order and bits."""
    frame = tidy(DATA, blocks={"p": 64}, threads=threads) >> BLOCKED[name]()
    want = (tidy(DATA) >> BLOCKED[name]()).collect()
    got = frame.collect()
    if isinstance(frame, ChunkedFrame):  # still read in blocks until a result is asked for (step 3)
        assert any("each block sorted, then the blocks merged" in note for note in frame._result._notes)
    assert got["p"].values.tolist() == want["p"].values.tolist()
    for variable in want.data_vars:
        assert np.asarray(got[variable].values).tobytes() == np.asarray(want[variable].values).tobytes(), variable


def test_a_sort_of_another_axis_runs_on_each_block():
    """A key along an axis every block holds whole sorts it the same in each: the frame stays read in blocks."""
    grid = xr.Dataset({"t": (("time", "lat"), rng.normal(size=(100, 9))), "w": ("lat", rng.random(9))},
                      coords={"time": np.arange(100), "lat": np.arange(9.0)})
    verbs = arrange(desc(fields("w"))) >> mutate(u=fields("t") * 2)
    got = tidy(grid, blocks={"time": 30}) >> verbs
    assert isinstance(got, ChunkedFrame)
    got, want = got.collect(), (tidy(grid) >> verbs).collect()
    assert got["lat"].values.tolist() == want["lat"].values.tolist()
    assert np.asarray(got["u"].values).tobytes() == np.asarray(want["u"].values).tobytes()


def test_a_sort_of_the_blocked_axis_is_refused_as_in_memory_when_called():
    with pytest.raises(ArrangeError, match="varies along p and q"):
        tidy(DATA, blocks={"p": 64}) >> arrange(fields("grid"))


SLICED = {  # a sort then a slice of the blocked axis: each block's part, then the overall (milestone 21, step 3)
    "the first 10 of a float key with NaN": (lambda: arrange(f32), "slice_head", 10),
    "the last 10, downwards": (lambda: arrange(desc(f32)), "slice_tail", 10),
    "the first 37, where NaN reaches the edge": (lambda: arrange(desc(f32)), "slice_head", 372),
    "the last 50 of an integer key": (lambda: arrange(i), "slice_tail", 50),
    "a boolean key, its first 5": (lambda: arrange(desc(b)), "slice_head", 5),
    "two keys, a whole sort a block": (lambda: arrange(i, desc(f64)), "slice_head", 25),
    "times, a whole sort a block": (lambda: arrange(when), "slice_tail", 30),
    "none": (lambda: arrange(f32), "slice_head", 0),
    "more than there are": (lambda: arrange(desc(f64)), "slice_head", 5_000),
    "after a filter": (lambda: filter(b) >> arrange(f64), "slice_head", 7),
}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(SLICED))
def test_a_slice_after_a_sort_takes_each_blocks_part_then_the_overall(name, threads, monkeypatch):
    """Each block keeps only its first or last n of the order (a partial sort for one key of numbers or booleans,
    ties at the edge by position), merged into the overall n: memory's rows, in memory's order."""
    from tidy3a import slice_head, slice_tail

    verbs, which, count = SLICED[name]
    cut = {"slice_head": slice_head, "slice_tail": slice_tail}[which](p=count)
    frame = tidy(DATA, blocks={"p": 64}, threads=threads) >> verbs() >> cut
    assert isinstance(frame, ChunkedFrame)
    got, want = frame.collect(), (tidy(DATA) >> verbs() >> cut).collect()
    assert got["p"].values.tolist() == want["p"].values.tolist()
    for variable in want.data_vars:
        assert np.asarray(got[variable].values).tobytes() == np.asarray(want[variable].values).tobytes(), variable
    words = "first" if which == "slice_head" else "last"
    assert any(f"each block's {words} {count:,} found" in note for note in frame._result._notes)


def test_other_verbs_after_a_sort_run_on_its_merged_rows():
    from tidy3a import slice_head

    for verbs in (arrange(f32) >> mutate(u=f64 + 1), arrange(f32) >> slice_head(q=2),
                  arrange(f32) >> slice_head(p=4) >> mutate(u=f64 + 1)):
        got = tidy(DATA, blocks={"p": 64}) >> verbs
        want = (tidy(DATA) >> verbs).collect()
        assert got.collect()["p"].values.tolist() == want["p"].values.tolist()


def test_a_sorted_frame_reads_nothing_until_a_result_is_asked_for():
    frame = tidy(DATA, blocks={"p": 64}) >> arrange(f32)
    assert "nothing read yet" in repr(frame)
    assert frame.preview(blocks=[1]).collect()["p"].values.tolist() == sorted(
        DATA["p"].values[:64].tolist(), key=lambda label: (np.isnan(DATA["f32"].values[label // 10]),
                                                          DATA["f32"].values[label // 10]))


def test_a_sort_written_writes_the_merged_rows(tmp_path):
    from tidy3a import write

    pytest.importorskip("zarr")
    path = tmp_path / "sorted.zarr"
    tidy(DATA[["f32", "f64", "i"]], blocks={"p": 64}) >> arrange(desc(f64)) >> write(path)
    want = (tidy(DATA[["f32", "f64", "i"]]) >> arrange(desc(f64))).collect()
    got = xr.open_zarr(path, consolidated=False)
    assert got["p"].values.tolist() == want["p"].values.tolist()
