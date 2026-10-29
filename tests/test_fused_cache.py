"""Compiled kernels kept on disk (plan milestone 16, step 6): written once, read by later runs, named by what decides
their machine code, and compiled in memory where they cannot be kept."""

import builtins
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("numba")
from tidy3a.fused import cache, kernels  # noqa: E402

pytestmark = pytest.mark.backends("numpy", reason="the cache keeps kernels, which see plain arrays on any backend")

SOURCE = "def twice(n, values, out):\n    for i in range(n):\n        out[i] = values[i] * 2\n"

RUN = textwrap.dedent("""
    import json, time, warnings
    import numpy as np
    warnings.simplefilter("ignore")
    from tidy3a import fields, filter, group_by, infer, mean, mutate, n, read_array, sqrt, summarise
    from tidy3a.fused import kernels
    x, y, z, intensity, ring, d = fields("x", "y", "z", "intensity", "ring", "d")
    rng = np.random.default_rng(0)
    values = (rng.normal(size=(20_000, 5)) * 30).astype("float32")
    values[:, 4] = rng.integers(0, 40, len(values))
    frame = read_array(values, blocks=5_000, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    rings = (frame >> mutate(d=sqrt(x**2 + y**2 + z**2)) >> filter(d <= 60) >> group_by(ring)
             >> summarise(points=n(), brightness=mean(intensity)))
    clock = time.perf_counter()
    points = rings.collect()["points"].values.tolist()
    took = time.perf_counter() - clock
    stats = [kernel.stats for kernel in kernels.COMPILED.values()]
    print(json.dumps({"note": rings._result._notes[-1], "points": points, "seconds": took,
                      "hits": sum(sum(s.cache_hits.values()) for s in stats),
                      "misses": sum(sum(s.cache_misses.values()) for s in stats), "kernels": len(stats)}))
""")


def fresh(folder):
    """The rings summary in a new interpreter keeping kernels in *folder*: what it ran, and Numba's cache counts."""
    environment = {**os.environ, "TIDY3A_KERNELS": str(folder)}
    environment.pop("TIDY3A_COMPILED", None)
    done = subprocess.run([sys.executable, "-c", RUN], env=environment, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_later_run_reads_the_kernels_an_earlier_one_kept(tmp_path):
    first = fresh(tmp_path)
    assert first["note"] == "ran: 4 blocks in a compiled kernel"
    assert first["kernels"] == 4 and first["hits"] == 0 and first["misses"] >= 4
    kept = sorted(path.name for path in tmp_path.glob("k*.py"))
    assert len(kept) == 4 and all(len(name) == len("k.py") + 32 for name in kept)
    assert list((tmp_path / "__pycache__").glob("*.nbc")) and not list(tmp_path.glob(".k*"))
    second = fresh(tmp_path)
    assert second["points"] == first["points"] and second["note"] == first["note"]
    assert second["hits"] >= 4 and second["misses"] == 0  # nothing compiled again
    assert sorted(path.name for path in tmp_path.glob("k*.py")) == kept


def test_a_kernel_is_named_by_its_source_and_what_decides_its_code(tmp_path, monkeypatch):
    monkeypatch.setenv("TIDY3A_KERNELS", str(tmp_path))
    twice = cache.load(SOURCE, "twice")
    out = np.zeros(3)
    twice(3, np.arange(3.0), out)
    assert out.tolist() == [0.0, 2.0, 4.0]
    first = {path.name for path in tmp_path.glob("k*.py")}
    assert len(first) == 1
    assert (tmp_path / first.pop()).read_text().startswith("import math\nimport numba\nimport numpy as np\n\ndef twice(")
    cache.load(SOURCE.replace("* 2", "* 3"), "twice")  # another source
    assert len(list(tmp_path.glob("k*.py"))) == 2
    monkeypatch.setattr(cache, "fingerprint", lambda: "numba 99.0")  # another Numba, say
    cache.load(SOURCE, "twice")
    assert len(list(tmp_path.glob("k*.py"))) == 3


def test_the_folder_defaults_under_the_users_cache(monkeypatch):
    monkeypatch.delenv("TIDY3A_KERNELS", raising=False)
    assert cache.directory() == cache.Path.home() / ".cache" / "tidy3a" / "kernels"
    monkeypatch.setenv("TIDY3A_KERNELS", "~/elsewhere")
    assert cache.directory() == cache.Path.home() / "elsewhere"


def test_kernels_compile_in_memory_where_they_cannot_be_kept(tmp_path, monkeypatch):
    blocked = tmp_path / "a file"
    blocked.write_text("")
    monkeypatch.setenv("TIDY3A_KERNELS", str(blocked / "kernels"))  # under a file: no folder can be made
    assert cache.load(SOURCE, "twice") is None
    monkeypatch.setattr(kernels, "COMPILED", {})
    twice = kernels.compile_source(SOURCE, "twice")
    out = np.zeros(2)
    twice(2, np.array([1.0, 4.0]), out)
    assert out.tolist() == [2.0, 8.0]


def test_the_folder_is_held_to_its_limit_least_recently_used_first(tmp_path, monkeypatch):
    """A new kernel removes the least recently used ones, module and Numba's files together, past
    ``TIDY3A_KERNELS_MAX``; loading one records its use."""
    monkeypatch.setenv("TIDY3A_KERNELS", str(tmp_path))
    sources = [SOURCE.replace("* 2", f"* {k}") for k in range(2, 6)]
    for k, source in enumerate(sources):
        kernel = cache.load(source, "twice")
        kernel(1, np.ones(1), np.zeros(1))  # compiled, so Numba keeps its files
        module = next(path for path in tmp_path.glob("k*.py") if f"* {k + 2}" in path.read_text())
        os.utime(module, (1_000 + k, 1_000 + k))  # used in this order
    one = builtins.sum(path.stat().st_size for paths in cache._kept(tmp_path).values() for path in paths) // 4
    assert builtins.all(len(paths) >= 3 for paths in cache._kept(tmp_path).values())  # module, index, code
    cache.load(sources[0], "twice")  # the oldest, used again: now the most recent
    monkeypatch.setenv("TIDY3A_KERNELS_MAX", str(3 * one))
    cache.load(SOURCE.replace("* 2", "* 9"), "twice")  # a new one: the two least recently used go
    left = sorted(path.read_text().split("values[i] * ")[1].split("\n")[0] for path in tmp_path.glob("k*.py"))
    assert left == ["2", "5", "9"]
    assert not [path for path in (tmp_path / "__pycache__").iterdir()
                if path.name.split(".", 1)[0] not in {module.stem for module in tmp_path.glob("k*.py")}]


def test_clear_removes_the_kernels_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setenv("TIDY3A_KERNELS", str(tmp_path))
    cache.load(SOURCE, "twice")(1, np.ones(1), np.zeros(1))
    (tmp_path / "notes.txt").write_text("mine")
    cache.clear()
    assert [path.name for path in tmp_path.iterdir() if path.name != "__pycache__"] == ["notes.txt"]
    assert not list((tmp_path / "__pycache__").iterdir())


def test_the_limit_reads_bytes_or_text(monkeypatch):
    monkeypatch.delenv("TIDY3A_KERNELS_MAX", raising=False)
    assert cache.limit() == 10**9
    monkeypatch.setenv("TIDY3A_KERNELS_MAX", "500MB")
    assert cache.limit() == 500 * 10**6
    monkeypatch.setenv("TIDY3A_KERNELS_MAX", "2048")
    assert cache.limit() == 2048
