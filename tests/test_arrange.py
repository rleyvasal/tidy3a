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
    got = tidy(DATA, blocks={"p": 64}, threads=threads) >> BLOCKED[name]()
    assert not isinstance(got, ChunkedFrame)  # the rows in memory, as a slice's
    assert any("each block sorted, then the blocks merged" in note for note in got._notes)
    want = (tidy(DATA) >> BLOCKED[name]()).collect()
    got = got.collect()
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
