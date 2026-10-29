"""A summary's blocks in compiled kernels (plan milestone 16, step 5): the same bits as NumPy's run of the blocks.

Each summary runs over blocks compiled and with ``TIDY3A_COMPILED=0``, on one thread and three: the same variables,
types, labels, and bits, float32 sums settled by labelarray's exact pass where they must be among them. A block whose
values do not agree with NumPy, or whose groups are too many for dense tables, runs on NumPy, the result saying how
many and why; a scan runs compiled frames and NumPy ones over the same blocks; ``repr`` and ``explain()`` say what is
planned, and the result what ran.
"""

import builtins

import numpy as np
import pytest

import tidy3a
from tidy3a import (bins, cells, fields, filter, group_by, if_else, infer, max, mean, median, min, mutate, n,
                    quantile, read_array, rounded, scan, sqrt, std, sum, summarise, tidy, var)

pytest.importorskip("numba")
xr = pytest.importorskip("xarray")

x, y, z, intensity, ring, d, e, f = fields("x", "y", "z", "intensity", "ring", "d", "e", "f")
NAMES = ["x", "y", "z", "intensity", "ring"]
OPTIONS = dict(dims={"point": infer, "feature": 5}, labels={"feature": NAMES}, field_axis="feature")


def cloud_values(size=20_000, seed=0):
    rng = np.random.default_rng(seed)
    values = (rng.normal(size=(size, 5)) * 30).astype("float32")
    values[:, 3] = np.abs(values[:, 3]).round()  # intensities, whole and with ties for the float32 sums
    values[:, 4] = rng.integers(0, 40, size)  # the ring, a whole float32
    values[rng.choice(size, 50, replace=False), 2] = np.nan
    values[rng.choice(size, 20, replace=False), 4] = np.nan  # points without a ring
    return values


NEAR = mutate(d=sqrt(x**2 + y**2 + z**2)) >> filter(d <= 60)
SUMMARIES = {
    "rings": NEAR >> group_by(ring) >> summarise(
        points=n(), total=sum(intensity), avg=mean(z), low=min(z, na_rm=True), high=max(z), spread=std(x),
        wide=var(y, na_rm=True), mid=median(y), q=quantile(intensity, 0.3), lit=tidy3a.any(intensity > 80),
        all_near=tidy3a.all(d < 59)),
    "rings kept": NEAR >> group_by(ring, dropna=False) >> summarise(points=n(), total=sum(intensity)),
    "grid": NEAR >> group_by(cells(x, 20.0), rounded(y, 25), axis="cell") >> summarise(
        points=n(), top=max(intensity), heat=sum(intensity), level=mean(z, na_rm=True)),
    "bins": group_by(bins(x, [-1000, -10, 0, 0.5, 10, 1000])) >> summarise(points=n(), total=sum(intensity)),
    "no key": NEAR >> summarise(points=n(), total=sum(intensity), far=max(d)),
    "expression": mutate(e=if_else(z > 0, x * 2 + 1, -y)) >> group_by(ring) >> summarise(s=sum(e), m=min(e)),
}


def same_bits(a, b):
    a, b = a.collect(), b.collect()
    assert sorted(a.variables) == sorted(b.variables)
    for name in a.variables:
        first, second = np.asarray(a[name].values), np.asarray(b[name].values)
        assert first.dtype == second.dtype and first.shape == second.shape, name
        if first.dtype.kind == "O":
            assert first.tolist() == second.tolist(), name
        else:
            assert first.tobytes() == second.tobytes(), name


def numpy_run(monkeypatch, make):
    """*make*'s frames run with compiling off: run here, as a frame read in blocks runs only when asked."""
    with monkeypatch.context() as patch:
        patch.setenv("TIDY3A_COMPILED", "0")
        made = make()
        for frame in made if isinstance(made, tuple) else (made,):
            if hasattr(frame, "collect"):
                frame.collect()
                result = getattr(frame, "_result", None) or frame
                assert not any(note.startswith("ran:") for note in result._notes), "compiled"
        return made


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("name", list(SUMMARIES))
def test_compiled_blocks_give_numpys_bits(name, threads, monkeypatch):
    values = cloud_values()

    def run():
        frame = read_array(values, blocks=3_000, threads=threads, **OPTIONS) >> SUMMARIES[name]
        frame.collect()
        return frame

    compiled = run()
    assert compiled._result._notes[-1] == "ran: 7 blocks in a compiled kernel"
    same_bits(compiled, numpy_run(monkeypatch, run))


def test_a_grid_of_values_a_position_compiles():
    """Each time step's grid grouped by its month: one code a position, each value of it added."""
    rng = np.random.default_rng(1)
    t = (rng.normal(size=(60, 6, 8)) * 10 + 280).astype("float32")
    t[rng.random(t.shape) < 0.02] = np.nan
    data = xr.Dataset({"t": (("time", "lat", "lon"), t), "month": ("time", np.arange(60) % 12 + 1),
                       "sun": ("time", rng.random(60) > 0.3)})
    temperature, month, sun = fields("t", "month", "sun")
    verbs = (filter(sun) >> group_by(month)
             >> summarise(mean=mean(temperature - 273.15, na_rm=True), top=max(temperature), cells=n()))
    frame = tidy(data, blocks={"time": 7}) >> verbs
    frame.collect()
    assert frame._result._notes[-1] == "ran: 9 blocks in a compiled kernel"
    same_bits(frame, tidy(data) >> verbs)


def test_a_block_that_does_not_agree_runs_on_numpy(monkeypatch):
    values = cloud_values()
    values[7_000] = [1, 2, 3, 4, 2.5]  # a near point whose ring is not whole, in the third block
    verbs = SUMMARIES["rings kept"]
    frame = read_array(values, blocks=3_000, **OPTIONS) >> verbs
    frame.collect()
    assert frame._result._notes[-1] == ("ran: 6 blocks in a compiled kernel; 1 on NumPy, as a value where agreement "
                                        "with NumPy is not established")
    same_bits(frame, numpy_run(monkeypatch, lambda: read_array(values, blocks=3_000, **OPTIONS) >> verbs))


@pytest.mark.parametrize("name", ["rings kept", "grid"])
def test_groups_too_many_for_dense_tables_are_sorted(name, monkeypatch):
    """Past the dense tables' limits, a block's tables are as large as the combinations it holds, found by sorting
    them: still compiled, the same bits; only combinations beyond 64-bit codes run on NumPy."""
    from tidy3a.fused import driver

    values = cloud_values()
    want = numpy_run(monkeypatch, lambda: read_array(values, blocks=3_000, **OPTIONS) >> SUMMARIES[name])
    for limits in ({"DENSE_GROUPS": 30}, {"DENSE_BYTES": 1_000}):  # 41 rings of 88 bytes each; the grid's more
        with monkeypatch.context() as patch:
            for setting, value in limits.items():
                patch.setattr(driver, setting, value)
            frame = read_array(values, blocks=3_000, threads=3, **OPTIONS) >> SUMMARIES[name]
            frame.collect()
            assert frame._result._notes[-1] == "ran: 7 blocks in a compiled kernel", limits
            same_bits(frame, want)
    monkeypatch.setattr(driver, "CODES", 30)
    frame = read_array(values, blocks=3_000, **OPTIONS) >> SUMMARIES[name]
    frame.collect()
    assert frame._result._notes[-1] == ("ran: 7 blocks on NumPy, as its keys' groups make more combinations than "
                                        "64-bit codes number")
    same_bits(frame, want)


def test_a_kernel_numba_refuses_runs_on_numpy(monkeypatch):
    import numba.core.errors

    from tidy3a.fused import driver

    def refuse(frame):
        raise numba.core.errors.TypingError("no such thing")

    monkeypatch.setattr(driver, "build", refuse)
    frame = read_array(cloud_values(), blocks=3_000, **OPTIONS) >> SUMMARIES["rings"]
    frame.collect()
    assert frame._result._notes[-1] == ("ran: 7 blocks on NumPy, as Numba could not compile its kernel (TypingError: "
                                        "no such thing)")


def test_a_scan_runs_compiled_and_numpy_frames_over_the_same_blocks(monkeypatch):
    values = cloud_values()
    frame = read_array(values, blocks=3_000, threads=2, **OPTIONS)
    compiled = frame >> SUMMARIES["grid"]
    slow = frame >> mutate(e=tidy3a.conj(x / 100)) >> group_by(ring) >> summarise(s=sum(e), points=n())
    rows = frame >> NEAR
    got_compiled, got_slow, got_rows = scan(compiled, slow, rows)
    assert got_compiled._notes[-1] == "ran: 7 blocks in a compiled kernel"
    assert not any(note.startswith("ran:") for note in got_slow._notes)
    def alone():
        frame = read_array(values, blocks=3_000, **OPTIONS)
        return scan(*(frame >> verbs for verbs in (
            SUMMARIES["grid"], mutate(e=tidy3a.conj(x / 100)) >> group_by(ring) >> summarise(s=sum(e), points=n()),
            NEAR)))

    alone = numpy_run(monkeypatch, alone)
    for got, want in zip((got_compiled, got_slow, got_rows), alone):
        same_bits(got, want)


def test_repr_and_explain_say_what_is_planned_and_what_ran(monkeypatch):
    frame = read_array(cloud_values(), blocks=3_000, **OPTIONS)
    rings = frame >> SUMMARIES["rings"]
    assert "planned run: a compiled kernel a block" in repr(rings)
    assert str(rings.explain()).endswith("planned run: a compiled kernel a block")
    slow = frame >> mutate(e=tidy3a.conj(x)) >> mutate(f=e * 2) >> group_by(cells(z, 5.0)) \
        >> summarise(s=sum(f), m=mean(tidy3a.conj(intensity + 1)))
    assert "planned run: NumPy, as mutate(e = conj(x)): conj() runs on NumPy (and 1 more: explain() lists them)" in \
        repr(slow)  # f, made from e, is not a reason of its own
    assert str(slow.explain()).splitlines()[-3:] == [
        "planned run: NumPy, as", "  mutate(e = conj(x)): conj() runs on NumPy",
        "  mean(conj(intensity + 1)): conj() runs on NumPy"]
    sample = rings.preview(blocks=[1, 2])
    assert not any(note.startswith("ran:") for note in sample._notes)  # a sample is not the run
    rings.collect()
    assert "ran: 7 blocks in a compiled kernel" in repr(rings)
    assert "ran: 7 blocks in a compiled kernel" in str(rings.explain())
    kept = frame >> NEAR
    assert "planned run" not in repr(kept)  # rows are kept, not summarised
    monkeypatch.setenv("TIDY3A_COMPILED", "0")
    off = frame >> SUMMARIES["rings"]
    assert "planned run: NumPy, as compiling is turned off (TIDY3A_COMPILED=0)" in repr(off)
    off.collect()
    assert not any(note.startswith("ran:") for note in off._result._notes)


def test_auto_sizing_counts_the_dense_tables(monkeypatch):
    """A compiled frame's blocks each hold their dense tables, an eighth of a block's share of the budget at most."""
    from tidy3a.fused.driver import DENSE_BYTES, dense_limit
    from tidy3a.sizing import RESERVE, cost

    assert dense_limit(None, 4) == DENSE_BYTES
    assert dense_limit(RESERVE + 8 * 4 * 1_000_000, 4) == 1_000_000
    assert dense_limit(RESERVE + 2**40, 4) == DENSE_BYTES
    values = cloud_values()
    budget = RESERVE + 400_000_000

    def fixed():
        frame = read_array(values, blocks="auto", memory=budget, threads=2, **OPTIONS)
        plain = NEAR >> group_by(ring) >> summarise(points=n(), top=max(z))
        return cost([frame >> plain]).fixed, cost([frame >> SUMMARIES["rings"]]).fixed

    compiled = fixed()
    numpy = numpy_run(monkeypatch, fixed)
    assert compiled[0] == dense_limit(budget, 2) and numpy[0] == 0
    assert compiled[1] == builtins.max(numpy[1], dense_limit(budget, 2))  # the exact sums' and median's own


@pytest.mark.parametrize("dropna", [True, False])
def test_a_key_far_wider_than_its_values_sorts_them(dropna, monkeypatch):
    """Two keys, one counted over its range and one whose values lie far apart, sorted; missing ones dropped or kept."""
    rng = np.random.default_rng(2)
    size = 6_000
    far = rng.choice(np.array([-(10**12), 5, 10**12, 7 * 10**11]), size).astype("float64")
    far[rng.choice(size, 30, replace=False)] = np.nan
    data = xr.Dataset({"far": ("p", far), "near": ("p", rng.integers(0, 5, size)),
                       "v": ("p", rng.normal(size=size).astype("float32"))})
    near, far_key, v = fields("near", "far", "v")
    verbs = group_by(near, far_key, axis="pair", dropna=dropna) >> summarise(points=n(), total=sum(v), top=max(v))
    frame = tidy(data, blocks={"p": 1_000}, threads=2) >> verbs
    frame.collect()
    assert frame._result._notes[-1] == "ran: 6 blocks in a compiled kernel"
    same_bits(frame, numpy_run(monkeypatch, lambda: tidy(data, blocks={"p": 1_000}) >> verbs))


def ties(size=6_000, seed=7):
    """Groups of two adjacent float32 values, whose means lie on float32 ties the error bound cannot decide."""
    rng = np.random.default_rng(seed)
    first = (rng.normal(size=size // 2) * 100).astype("float32")
    v = np.stack([first, np.nextafter(first, np.float32(np.inf))], axis=1).ravel()
    pair = np.repeat(np.arange(size // 2), 2)
    order = rng.permutation(size)  # the pairs spread over the blocks
    return xr.Dataset({"v": ("p", v[order]), "pair": ("p", pair[order])})


def passes(monkeypatch):
    """The later passes' asks a run makes, counted."""
    from tidy3a import chunked

    asked = []
    original = chunked._passes

    def counting(leaves, blocks, pool, settled, *rest, **options):
        asked.extend(ask for leaf in leaves if id(leaf) in settled for asks in settled[id(leaf)][1].values()
                     for ask in asks)
        return original(leaves, blocks, pool, settled, *rest, **options)

    monkeypatch.setattr(chunked, "_passes", counting)
    return asked


@pytest.mark.skipif(not hasattr(__import__("labelarray").sums, "EXACT"), reason="labelarray marks exact totals from 0.1.7")
def test_exact_totals_need_no_exact_pass(monkeypatch):
    """The kernel marks a total no addition rounded as exact, merges keep the mark where adding the blocks' totals
    was exact, and labelarray rounds those from the total: the ties need no exact pass, with NumPy's bits."""
    data = ties()
    pair, v = fields("pair", "v")
    verbs = group_by(pair) >> summarise(m=mean(v), s=sum(v), k=n())
    asked = passes(monkeypatch)
    compiled = tidy(data, blocks={"p": 1_000}, threads=3) >> verbs
    compiled.collect()
    assert compiled._result._notes[-1] == "ran: 6 blocks in a compiled kernel" and asked == []
    want = numpy_run(monkeypatch, lambda: tidy(data, blocks={"p": 1_000}) >> verbs)
    assert len(asked) > 0  # NumPy's estimate leaves the ties to the exact pass
    same_bits(compiled, want)
    same_bits(compiled, tidy(data) >> verbs)


def test_without_labelarrays_exact_mark_the_exact_pass_settles_the_ties(monkeypatch):
    """With a labelarray that has no exact mark (0.1.6), the kernel's totals carry the bound as before."""
    import types

    from labelarray import sums

    import tidy3a.reductions
    from tidy3a.fused import reductions

    older = types.SimpleNamespace(**{name: getattr(sums, name) for name in dir(sums)
                                     if not name.startswith("__") and name not in ("EXACT", "merge")})
    monkeypatch.setattr(tidy3a.reductions, "sums", older)  # tidy3a sees no mark; labelarray itself is unchanged
    monkeypatch.setattr(reductions, "EXACT", None)
    data = ties()
    pair, v = fields("pair", "v")
    verbs = group_by(pair) >> summarise(m=mean(v), s=sum(v), k=n())
    asked = passes(monkeypatch)
    frame = tidy(data, blocks={"p": 1_000}, threads=3) >> verbs
    frame.collect()
    assert frame._result._notes[-1] == "ran: 6 blocks in a compiled kernel" and len(asked) > 0
    same_bits(frame, tidy(data) >> verbs)


@pytest.mark.parametrize("threads", [1, 3])
def test_positions_of_many_values_are_reduced_first_by_numpy(threads, monkeypatch):
    """Positions of 400 values each (past ``BY_POSITION``): the compiled block places the keys and computes the
    values, NumPy's reductions take each position first, as the NumPy path does: its bits, every reduction."""
    from tidy3a.fused import driver

    rng = np.random.default_rng(3)
    t = (rng.normal(size=(90, 20, 20)) * 10 + 280).astype("float32")
    t[rng.random(t.shape) < 0.01] = np.nan
    data = xr.Dataset({"t": (("time", "lat", "lon"), t), "month": ("time", np.arange(90) % 12 + 1),
                       "sun": ("time", rng.random(90) > 0.3), "w": (("time", "lat", "lon"), rng.random(t.shape))})
    temperature, month, sun, w = fields("t", "month", "sun", "w")
    verbs = (filter(sun) >> mutate(c=temperature - 273.15) >> group_by(month)
             >> summarise(m=mean(fields("c"), na_rm=True), top=max(temperature), low=min(fields("c"), na_rm=True),
                          spread=std(w), s=sum(w), mid=median(temperature, na_rm=True), hot=tidy3a.any(temperature > 300),
                          cells=n()))
    called = []
    original = driver.Fused._by_position
    monkeypatch.setattr(driver.Fused, "_by_position", lambda self, *a: called.append(1) or original(self, *a))
    frame = tidy(data, blocks={"time": 8}, threads=threads) >> verbs
    frame.collect()
    assert frame._result._notes[-1] == "ran: 12 blocks in a compiled kernel" and len(called) == 12
    same_bits(frame, numpy_run(monkeypatch, lambda: tidy(data, blocks={"time": 8}) >> verbs))
    got, want = frame.collect(), (tidy(data) >> verbs).collect()  # float64 sums and spreads: close to memory's
    for name in ("m", "top", "low", "mid", "hot", "cells"):
        assert np.asarray(got[name].values).tobytes() == np.asarray(want[name].values).tobytes(), name


def test_positions_of_many_values_take_the_kernel_from_four_threads_where_order_does_not_matter(monkeypatch):
    """An order-free summary (float32 means, extremes, counts, flags, the median) of 400-value positions runs
    NumPy's reductions below eight threads and the fused kernel from eight on: the same bits on any number, a tie of
    -0.0 and +0.0 included, whose extreme both paths take by the kernel's order. A float64 sum keeps NumPy's
    reductions on every number of threads, as its bits follow the order of its additions."""
    from tidy3a.fused import driver

    rng = np.random.default_rng(4)
    t = (rng.normal(size=(96, 20, 20)) * 10 + 280).astype("float32")
    z = -np.abs(rng.normal(size=t.shape)).astype("float32")
    z[np.arange(96) % 12 == 5] = 0.0  # month 6: zeros only, of both signs
    z[(np.arange(96) % 12 == 5)[:, None, None] & (rng.random(t.shape) < 0.5)] = -0.0
    data = xr.Dataset({"t": (("time", "lat", "lon"), t), "z": (("time", "lat", "lon"), z),
                       "w": (("time", "lat", "lon"), t.astype("float64")), "month": ("time", np.arange(96) % 12 + 1)})
    temperature, month, zz, w = fields("t", "month", "z", "w")
    free = group_by(month) >> summarise(m=mean(temperature), top=max(zz), low=min(zz), k=n(),
                                        hot=tidy3a.any(temperature > 300), mid=median(temperature))
    called = []
    original = driver.Fused._by_position
    monkeypatch.setattr(driver.Fused, "_by_position", lambda self, *a: called.append(1) or original(self, *a))
    runs = {}
    for threads in (1, 3, 8, 12):
        called.clear()
        runs[threads] = tidy(data, blocks={"time": 8}, threads=threads) >> free
        runs[threads].collect()
        assert len(called) == (12 if threads < 8 else 0), threads
    for threads in (3, 8, 12):
        same_bits(runs[threads], runs[1])
    assert np.signbit(runs[1].collect()["top"].values[5]) == False  # max of the zeros: +0.0
    assert np.signbit(runs[1].collect()["low"].values[5]) == True  # min: -0.0
    ordered = group_by(month) >> summarise(m=mean(temperature), s=sum(w))  # a float64 sum: the order matters
    called.clear()
    frame = tidy(data, blocks={"time": 8}, threads=8) >> ordered
    frame.collect()
    assert len(called) == 12



@pytest.mark.parametrize("threads", [1, 3])
def test_islands_numpy_computes_give_numpys_bits(threads, monkeypatch):
    """exp, log, a cube, // and % computed by NumPy for each block, one of a variable an earlier mutate made, which
    NumPy makes again for it, and the kernel reading them: NumPy's bits (milestone 18)."""
    values = cloud_values()
    d, w = fields("d", "w")
    verbs = (mutate(d=sqrt(x**2 + y**2 + z**2), w=tidy3a.exp(-d / 20)) >> filter(d <= 60)
             >> filter(tidy3a.sin(x) > -0.5) >> group_by(ring)
             >> summarise(weighted=sum(w * intensity), glow=mean(tidy3a.log(abs(intensity) + 1)), cube=max(z ** 3),
                          steps=min(x // 7), rest=max(x % 5), k=n()))
    frame = read_array(values, blocks=3_000, threads=threads, **OPTIONS) >> verbs
    assert "with NumPy for exp(-d / 20), sin(x), log(abs(intensity) + 1), z ** 3" in repr(frame)  # // and %: compiled
    frame.collect()
    assert frame._result._notes[-1] == "ran: 7 blocks in a compiled kernel"
    same_bits(frame, numpy_run(monkeypatch, lambda: read_array(values, blocks=3_000, **OPTIONS) >> verbs))
