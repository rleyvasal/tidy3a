"""Float32 sums and means: float64 accumulation, one rounding (plan milestone 10, decisions 4 and 5).

Every grouped float32 sum and mean, and every ungrouped one adding 8 or more
values per result, is correctly rounded: the float32 nearest the exact sum or
mean of the stored values, ties to even. The reference here is exact rational
arithmetic on the stored float32 values and an independent rounding routine,
not NumPy. Ungrouped reductions of fewer than 8 values per result (such as
``sum(p**2, over="i")`` over x, y, and z) add in float64 in a fixed order and
round once. A zero result is +0.0, as NumPy gives it. These check the verbs
on both backends; the kernel itself, compiled and not, is tested with the
library, in labelarray's own tests.
"""

import builtins
import math
from fractions import Fraction

import numpy as np
import pytest
import xarray as xr

from tidy3a import col, group_by, mean, mutate, sum, summarise, tidy

F32 = np.float32
MAX32 = Fraction(float(np.finfo(np.float32).max))


def round32(value: Fraction) -> np.float32:
    """*value* correctly rounded to float32, ties to even, by exact arithmetic."""
    if value == 0:
        return F32(0.0)
    sign, size = (1, value) if value > 0 else (-1, -value)
    if size >= MAX32 + Fraction(2) ** 103:  # half a step above the largest float32
        return F32(sign * math.inf)
    exponent = size.numerator.bit_length() - size.denominator.bit_length()
    while Fraction(2) ** exponent > size:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= size:
        exponent += 1
    step = Fraction(2) ** (max(exponent, -126) - 23)  # spacing of float32 values there, subnormals included
    scaled = size / step
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    return F32(sign * float(whole * step))


def expected(values: np.ndarray, op: str, na_rm: bool) -> np.float32:
    """tidy3a's rules, then the exact result correctly rounded."""
    values = np.asarray(values, F32)
    if np.isnan(values).any():
        if not na_rm:
            return F32(np.nan)
        values = values[~np.isnan(values)]
    plus, minus = bool((values == np.inf).any()), bool((values == -np.inf).any())
    if plus and minus:
        return F32(np.nan)
    if plus or minus:
        return F32(np.inf if plus else -np.inf) if len(values) else F32(np.nan)
    total = builtins.sum((Fraction(float(v)) for v in values), Fraction(0))  # Python's sum: tidy3a's is imported
    if op == "sum":
        return round32(total)
    return round32(total / len(values)) if len(values) else F32(np.nan)


def same_bits(a, b) -> bool:
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a.view(np.uint32) == b.view(np.uint32))))


def corpus():
    """Groups of float32 values: broad random ones and adversarial ones, keyed 0, 1, 2, …"""
    rng = np.random.default_rng(11)
    groups = []
    for size in (1, 2, 3, 5, 8, 13, 100, 3000):
        groups.append(rng.uniform(0, 255, size))
        groups.append(rng.normal(0, 1e4, size))                                   # cancellation
        groups.append(rng.choice([-1, 1], size) * 10.0 ** rng.uniform(-30, 30, size))  # wide range
        groups.append(rng.normal(0, 1e-40, size))                                 # subnormals
    big = 2.0**53
    groups += [
        [big, 1, -big], [1e30, 1, -1e30], [2.0**24, *[1.0] * 1000, -(2.0**24)], [1e8, 1, -1e8] * 300,
        [3e38] * 10,                                       # the sum overflows float32, the mean does not
        [-0.0, -0.0], [np.inf, 1], [np.inf, -np.inf], [-np.inf, 5],
        [1.0, 2.0**-24], [1.0 + 2.0**-23, 2.0**-24],       # exactly halfway between two float32 values
        [0.1] * 10, [np.nan, 1.0, 2.0], [np.nan, np.nan],
    ]
    halfway = rng.uniform(1, 2, 200).astype(F32)           # many pairs summing exactly to halfway points
    for value in halfway:
        groups.append([float(value), float(np.spacing(value)) / 2])
    values = np.concatenate([np.asarray(g, F32) for g in groups])
    keys = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    order = rng.permutation(len(values))
    return values[order], keys[order], [np.asarray(g, F32) for g in groups]


@pytest.mark.parametrize("na_rm", [False, True])
def test_grouped_float32_sums_and_means_are_correctly_rounded(na_rm):
    values, keys, groups = corpus()
    data = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    with np.errstate(all="ignore"):
        out = (tidy(data) >> group_by(col("key"))
               >> summarise(s=sum(col("v"), na_rm=na_rm), m=mean(col("v"), na_rm=na_rm))).collect()
    assert out["s"].dtype == F32 and out["m"].dtype == F32
    for name, op in (("s", "sum"), ("m", "mean")):
        want = np.array([expected(g, op, na_rm) for g in groups], F32)
        wrong = [k for k in range(len(groups)) if not same_bits(out[name].values[k], want[k])]
        assert wrong == [], [(k, groups[k][:4], out[name].values[k], want[k]) for k in wrong[:5]]


def test_grouped_results_broadcast_by_mutate_are_the_same_values():
    values, keys, groups = corpus()
    data = xr.Dataset({"v": ("point", values), "key": ("point", keys)})
    with np.errstate(all="ignore"):
        out = (tidy(data) >> group_by(col("key")) >> mutate(m=mean(col("v"), na_rm=True))).collect()
    want = np.array([expected(g, "mean", True) for g in groups], F32)
    assert same_bits(out["m"].values, want[keys])


@pytest.mark.parametrize("na_rm", [False, True])
def test_ungrouped_float32_reductions_of_many_values_are_correctly_rounded(na_rm):
    rng = np.random.default_rng(5)
    grid = (rng.normal(0, 1e4, (500, 7)) * 10.0 ** rng.uniform(-3, 3, (500, 7))).astype(F32)
    grid[3, 2] = np.nan
    grid[:, 4] = np.tile(np.array([2.0**53, 1, -(2.0**53), 0.5], F32), 125)
    data = xr.Dataset({"v": (("time", "station"), grid)})
    with np.errstate(all="ignore"):
        out = (tidy(data) >> mutate(s=sum(col("v"), over="time", na_rm=na_rm),
                                    m=mean(col("v"), over="time", na_rm=na_rm))).collect()
        whole = (tidy(data) >> summarise(m=mean(col("v"), na_rm=na_rm))).collect()
    for name, op in (("s", "sum"), ("m", "mean")):
        want = np.array([expected(grid[:, k], op, na_rm) for k in range(7)], F32)
        assert same_bits(out[name].values, want), (out[name].values, want)
    assert same_bits(whole["m"].values, expected(grid.ravel(), "mean", na_rm))


def test_few_values_per_result_add_in_float64_in_a_fixed_order():
    rng = np.random.default_rng(2)
    p = rng.uniform(-80, 80, (3, 1000)).astype(F32) ** 2
    data = xr.Dataset({"p": (("i", "point"), p)})
    out = (tidy(data) >> mutate(d=sum(col("p"), over="i"))).collect()
    want = ((p[0].astype(np.float64) + p[1]) + p[2]).astype(F32)
    assert same_bits(out["d"].values, want)
    assert not same_bits(p.sum(axis=0), want)  # NumPy's float32 sum rounds differently for some points


@pytest.mark.parametrize("grouped", [False, True])
def test_results_do_not_depend_on_memory_layout(grouped):
    rng = np.random.default_rng(9)
    rows = (rng.normal(0, 1e4, (4000, 3)) * 10.0 ** rng.uniform(-2, 2, (4000, 3))).astype(F32)
    key = rng.integers(0, 50, 4000)
    by_row = xr.Dataset({"values": (("point", "feature"), rows), "key": ("point", key)},
                        coords={"feature": ["a", "b", "c"]})
    by_field = xr.Dataset({"values": (("feature", "point"), np.ascontiguousarray(rows.T)), "key": ("point", key)},
                          coords={"feature": ["a", "b", "c"]})
    results = []
    for data in (by_row, by_field):
        frame = tidy(data, field_axis="feature")
        frame = frame >> group_by(col("key")) if grouped else frame
        results.append((frame >> summarise(m=mean(col("values")), s=sum(col("values")))).collect())
    for name in ("m", "s"):
        assert same_bits(results[0][name].transpose(*results[1][name].dims).values, results[1][name].values)


def test_float64_inputs_keep_numpys_results():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1e4, 1000)
    key = rng.integers(0, 20, 1000)
    data = xr.Dataset({"v": ("point", values), "key": ("point", key)})
    out = (tidy(data) >> group_by(col("key")) >> summarise(m=mean(col("v")))).collect()
    want = np.array([values[key == k].mean() for k in range(20)])
    assert np.array_equal(out["m"].values.view(np.uint64), want.view(np.uint64))


def test_min_count_and_signed_zero():
    data = xr.Dataset({"v": ("point", np.array([-0.0, -0.0, 1.0, np.nan], F32)), "key": ("point", [0, 0, 1, 1])})
    out = (tidy(data) >> group_by(col("key"))
           >> summarise(s=sum(col("v"), na_rm=True, min_count=2))).collect()
    assert out["s"].values[0] == 0 and not np.signbit(out["s"].values[0])  # a zero sum is +0.0
    assert np.isnan(out["s"].values[1])  # one present value, fewer than min_count


@pytest.mark.backends("xarray", reason="checks imports only")
def test_importing_tidy3a_does_not_import_numba():
    import subprocess
    import sys

    code = "import sys, tidy3a; assert 'numba' not in sys.modules, 'numba imported'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
