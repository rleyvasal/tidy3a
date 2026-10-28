"""The fused planner (plan milestone 16, step 1): whether a summary of data read in blocks compiles, and why not.

Read off the verbs on the block of no positions, so nothing is read; on only when Numba is installed and
``TIDY3A_COMPILED`` is not 0; and importing tidy3a loads no Numba. Planning itself needs no Numba, so the tests plan as
where it is installed, wherever they run, but those about its absence.
"""

import subprocess
import sys

import numpy as np
import pytest

from tidy3a import (all, any, bins, cells, exp, fields, filter, floor, formula, group_by, if_else, infer, max,
                    mean, median, min, mutate, n, quantile, read_array, register_reduction, rounded, select, sqrt, std,
                    sum, summarise, tidy, var)
import tidy3a.fused
from tidy3a.fused import Fusion, plan_for



@pytest.fixture(autouse=True)
def compiling(request, monkeypatch):
    """Compiling on, as where Numba is installed, for every test but those about when it is off."""
    if "without_numba" not in request.node.name and "load_no_numba" not in request.node.name:
        monkeypatch.delenv("TIDY3A_COMPILED", raising=False)
        monkeypatch.setattr(tidy3a.fused, "unavailable", lambda: None)


names = ["x", "y", "z", "intensity", "ring"]
x, y, z, intensity, ring, distance, a, b = fields(*names, "distance", "a", "b")


def cloud(dtype="float32"):
    values = np.zeros((100, 5), dtype)
    return read_array(values, dims={"point": infer, "feature": 5}, labels={"feature": names}, field_axis="feature",
                      blocks=10)


def grid(kind="float32"):
    xr = pytest.importorskip("xarray")
    values = {"float32": np.zeros((12, 4), np.float32), "complex": np.zeros((12, 4), np.complex64)}[kind]
    return tidy(xr.Dataset({"t": (("time", "cell"), values), "m": ("time", np.arange(12) % 3),
                            "when": ("time", np.arange(12).astype("datetime64[s]")),
                            "name": ("time", np.array(list("abcdefghijkl"))),
                            "on": ("time", np.arange(12) % 2 == 0)}), blocks={"time": 4})


NEAR = mutate(distance=formula(r"\sqrt{x^2 + y^2 + z^2}")) >> filter(distance <= 50)
COMPILED = {
    "rings": lambda: cloud() >> NEAR >> group_by(ring) >> summarise(k=n(), m=mean(intensity)),
    "voxels": lambda: cloud() >> NEAR >> group_by(cells(x, 1), cells(y, 1), cells(z, 1))
                      >> summarise(k=n(), m=mean(intensity)),
    "every reduction": lambda: cloud() >> group_by(ring) >> summarise(
        k=n(), s=sum(x), m=mean(y), lo=min(z), hi=max(z), sd=std(x), va=var(y), yes=any(x > 0), every=all(y < 1),
        mid=median(z), q=quantile(z, 0.3)),
    "chained assignments": lambda: cloud() >> mutate(a=x * 2, b=a + 1) >> group_by(ring) >> summarise(m=mean(b)),
    "functions and logic": lambda: cloud() >> mutate(a=if_else((x > 0) & ~(y < 0) | (z == 1), abs(x), floor(-y)),
                                                       b=sqrt(x) / 2 + x ** 0.5 - 1 / x ** -1)
                                   >> group_by(ring) >> summarise(m=mean(a), h=max(b)),
    "a selection": lambda: cloud() >> select(x, ring) >> group_by(ring) >> summarise(m=max(x)),
    "bins and rounding": lambda: cloud() >> group_by(bins(x, [0, 1, 2]), rounded(y, 0.5)) >> summarise(k=n()),
    "a grid": lambda: grid() >> filter(fields("m") > 0) >> mutate(c=fields("t") - 273.15) >> group_by(fields("m"))
                      >> summarise(w=mean(fields("c")), h=max(fields("c"))),
    "a boolean key": lambda: grid() >> group_by(fields("on")) >> summarise(w=mean(fields("t"))),
    "no key": lambda: cloud() >> summarise(m=mean(x)),
}
NUMPY = {
    "exp": (lambda: cloud() >> mutate(a=exp(x)) >> group_by(ring) >> summarise(m=mean(a)), "exp() runs on NumPy"),
    "a cube": (lambda: cloud() >> mutate(a=x ** 3) >> group_by(ring) >> summarise(m=mean(a)),
               "a power of 3 runs on NumPy"),
    "floor division": (lambda: cloud() >> mutate(a=x // 2) >> group_by(ring) >> summarise(m=mean(a)),
                       "// runs on NumPy"),
    "a remainder": (lambda: cloud() >> filter(x % 2 == 0) >> group_by(ring) >> summarise(k=n()),
                    "% runs on NumPy"),
    "a time key": (lambda: grid() >> group_by(fields("when")) >> summarise(w=mean(fields("t"))),
                   "when is times, which runs on NumPy"),
    "a text key": (lambda: grid() >> group_by(fields("name")) >> summarise(w=mean(fields("t"))),
                   "name is text, which runs on NumPy"),
    "complex values": (lambda: grid("complex") >> group_by(fields("m")) >> summarise(w=sum(fields("t"))),
                       "is complex numbers, which runs on NumPy"),
    "half-precision values": (lambda: cloud("float16") >> group_by(ring) >> summarise(m=max(x)),
                              "is half-precision floats, which runs on NumPy"),
    "rows kept": (lambda: cloud() >> NEAR, "its rows are kept, not summarised"),
}


@pytest.mark.parametrize("name", list(COMPILED))
def test_summaries_of_compilable_steps_compile(name):
    assert plan_for(COMPILED[name]()) == Fusion(), name


@pytest.mark.parametrize("name", list(NUMPY))
def test_each_step_that_does_not_compile_is_named_with_its_reason(name):
    make, reason = NUMPY[name]
    fusion = plan_for(make())
    assert not fusion.compiled and any(reason in found for found in fusion.reasons), fusion.reasons
    assert fusion.words().startswith("NumPy, as ")


@pytest.fixture
def registry_kept():
    """A registration is global: the registries restored after the test, in place, since the driver holds them."""
    import tidy3a.reductions as reductions

    saved = [(registry, dict(registry)) for registry in (reductions.REGISTRY, reductions.EAGER)]
    yield
    for registry, before in saved:
        registry.clear()
        registry.update(before)


def test_a_registered_reduction_runs_on_numpy(registry_kept):
    spread = register_reduction("spread", lambda values, axes, na_rm: values.max(axis=axes) - values.min(axis=axes),
                                partial=lambda v, c, g, na_rm: np.zeros((1, g)), merge=lambda h, p: h,
                                finish=lambda h, counts: h[0], fill=np.zeros((1, 1)))
    fusion = plan_for(cloud() >> group_by(ring) >> summarise(s=spread(x)))
    assert fusion.reasons == ("spread(x): a registered reduction runs on NumPy",)


def test_every_reason_is_listed_once():
    fusion = plan_for(cloud() >> mutate(a=exp(x), b=x // 2) >> group_by(ring) >> summarise(m=mean(a), s=sum(b)))
    assert len(fusion.reasons) == len(set(fusion.reasons)) >= 2
    assert "(and " in fusion.words() and "explain() lists them" in fusion.words()
    assert fusion.lines()[0] == "planned run: NumPy, as" and len(fusion.lines()) == len(fusion.reasons) + 1


def test_compiling_is_off_without_numba_or_when_turned_off(monkeypatch):
    import importlib.util

    frame = COMPILED["rings"]()
    monkeypatch.setenv("TIDY3A_COMPILED", "0")
    assert plan_for(frame).reasons == ("compiling is turned off (TIDY3A_COMPILED=0)",)
    monkeypatch.setenv("TIDY3A_COMPILED", "1")
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *rest: None if name == "numba" else
                        find_spec(name, *rest))
    assert plan_for(frame).reasons == ('Numba is not installed (pip install "tidy3a[fast]")',)


def test_planning_reads_nothing(monkeypatch):
    frame = COMPILED["voxels"]()
    monkeypatch.setattr(frame._blocks.source, "block", lambda *args, **kwargs: pytest.fail("a block was read"))
    assert plan_for(frame).compiled


def test_importing_and_planning_load_no_numba():
    script = """
import sys
import numpy as np
from tidy3a import fields, group_by, infer, mean, read_array, summarise
from tidy3a.fused import plan_for
values = np.zeros((10, 2), np.float32)
frame = read_array(values, dims={"point": infer, "f": 2}, labels={"f": ["v", "g"]}, field_axis="f", blocks=5)
v, g = fields("v", "g")
assert plan_for(frame >> group_by(g) >> summarise(m=mean(v))).compiled or True
assert "numba" not in sys.modules, sorted(name for name in sys.modules if name.startswith("numba"))
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
