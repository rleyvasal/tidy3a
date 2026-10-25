"""Work and memory that grow with what a step keeps or shows, not with the whole input."""

import importlib.util
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tidy3a import assign, col, dot, fields, filter, infer, mutate, read_array, tidy

x, y, z = fields("x", "y", "z")
REPO = Path(__file__).resolve().parents[1]


class Counted:
    """A label that counts how often it is turned into text."""

    formatted = 0

    def __init__(self, value):
        self.value = value

    def __str__(self):
        Counted.formatted += 1
        return f"L{self.value}"


def test_printing_formats_only_the_labels_it_shows():
    labels = np.empty(1000, dtype=object)
    labels[:] = [Counted(k) for k in range(1000)]
    frame = tidy(xr.Dataset({"a": ("p", np.zeros(1000))}, coords={"p": labels}))
    Counted.formatted = 0
    text = repr(frame)
    assert "L0  L1  L2  L3  L4  L5  …  L999" in text
    assert Counted.formatted <= 7


def test_printing_many_float_and_time_labels_is_unchanged():
    floats = tidy(xr.Dataset({"a": ("p", np.zeros(100))}, coords={"p": np.arange(100) * 0.1}))
    assert "0  0.1  0.2  0.3  0.4  0.5  …  9.9" in repr(floats)
    days = np.datetime64("2026-01-01") + np.arange(100).astype("timedelta64[D]")
    times = days.astype("datetime64[ns]")
    times[-1] += np.timedelta64(1, "s")  # one label off midnight: every label shows its time
    text = repr(tidy(xr.Dataset({"a": ("t", np.zeros(100))}, coords={"t": times})))
    assert "2026-01-01T00:00:00" in text and "2026-04-10T00:00:01" in text


def _cloud():
    values = np.arange(40.0, dtype="float32").reshape(10, 4)
    return read_array(values, dims={"point": infer, "feature": 4}, labels={"feature": ["x", "y", "z", "i"]},
                      field_axis="feature")


def test_a_vector_write_copies_its_holder_once(monkeypatch):
    frame = _cloud()
    backend = type(frame._backend)
    writes = []
    original = backend.with_values
    monkeypatch.setattr(backend, "with_values", lambda self, *a: writes.append(1) or original(self, *a))
    turn = xr.DataArray(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float32"), dims=("r", "feature"),
                        coords={"r": ["x", "y", "z"], "feature": ["x", "y", "z"]})
    out = (frame >> mutate(assign([x, y, z], dot(turn, [x, y, z], over="feature")))).collect()
    assert len(writes) == 1
    assert out["values"].sel(feature="x").values.tolist() == (-np.arange(1.0, 40, 4)).tolist()
    assert out["values"].sel(feature="y").values.tolist() == np.arange(0.0, 40, 4).tolist()
    assert out["values"].sel(feature="i").values.tolist() == np.arange(3.0, 40, 4).tolist()


def test_a_vector_write_notes_promotion_once_naming_the_first_field():
    frame = read_array(np.zeros((3, 3), dtype="int16"), dims={"point": infer, "feature": 3},
                       labels={"feature": ["x", "y", "z"]}, field_axis="feature")
    vector = xr.DataArray(np.full((3, 3), 0.5), dims=("point", "c"), coords={"c": ["x", "y", "z"]})
    out = frame >> mutate(assign([x, y, z], vector))
    promoted = [note for note in out._notes if "promoted" in note]
    assert promoted == ["changed: values promoted int16 → float64 by mutate(x = …); all fields on feature now float64"]
    assert out.collect()["values"].values.tolist() == [[0.5] * 3] * 3


def test_filtering_an_unlabelled_axis_does_not_build_every_position():
    size = 1_000_000
    frame = tidy(xr.Dataset({"a": ("p", np.arange(size, dtype="float64"))}))
    keep = col("a") >= size // 2
    frame >> filter(keep)  # warm up
    tracemalloc.start()
    out = frame >> filter(keep)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # The mask (1 MB), the kept positions (4 MB), and the kept values (4 MB); not
    # positions 1 … size for the whole axis (8 MB) and a copy of the kept ones.
    assert peak < 13 * 2**20, f"{peak / 2**20:.1f} MB"
    labels = out.collect()["p"].values
    assert labels.dtype == np.int64 and labels[0] == size // 2 + 1 and labels[-1] == size and len(labels) == size // 2


def _bench():
    spec = importlib.util.spec_from_file_location("bench_run", REPO / "benchmarks" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.backends("xarray", reason="times both backends itself")
def test_benchmark_rounds_alternate_which_backend_runs_first():
    run = _bench()
    order = []
    result = tidy(xr.Dataset({"a": ("p", [1.0])}))

    def make(backend):
        return lambda: order.append(backend) or result

    timing = run.time_both(make, minimum=0)
    rounds = [tuple(order[k:k + 2]) for k in range(len(order) - 2 * timing["xarray"]["repeats"], len(order), 2)]
    assert rounds[0::2] == [("xarray", "numpy")] * len(rounds[0::2])
    assert rounds[1::2] == [("numpy", "xarray")] * len(rounds[1::2])


@pytest.mark.backends("xarray", reason="lists workloads only")
def test_benchmarks_include_many_small_groups():
    names = list(_bench().workloads(quick=True))
    assert any("90,000 groups" in name for name in names)
