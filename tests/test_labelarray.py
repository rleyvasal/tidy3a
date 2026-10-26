"""The labelarray library on its own (plan milestone 10, step A5).

``Array``, ``Collection``, and the grouped reductions are the library's
public API; tidy3a's NumPy backend is an adapter over them. Each operation
is checked against xarray, the reference implementation, and the library
against the import rules: it runs without xarray and pandas, and the
adapter holds no array code of its own (``test_label_index`` checks that
the library imports nothing else of tidy3a).
"""

import ast
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import tidy3a.numpy_backend as adapter
from tidy3a import labelarray
from tidy3a.labelarray import (
    AlignError,
    Array,
    Collection,
    Grouping,
    LabelError,
    ShapeError,
    align,
    concat,
    grouped,
)

pytestmark = pytest.mark.backends("xarray", reason="tests the library itself, which no tidy3a backend chooses")


def _xr(array: Array) -> xr.DataArray:
    return xr.DataArray(array.data, dims=array.dims, coords={d: array.labels(d) for d in array.coords})


def _same(got: Array, want: xr.DataArray) -> None:
    assert got.dims == want.dims
    np.testing.assert_array_equal(got.data, want.values)
    for dim in want.indexes:
        np.testing.assert_array_equal(got.labels(dim), want[dim].values)


@pytest.fixture
def grid() -> Array:
    rng = np.random.default_rng(5)
    return Array(rng.normal(size=(4, 3)).astype("float32"), ("lat", "time"),
                 {"lat": [10.0, 20.0, 30.0, 40.0], "time": np.arange("2026-01-01", "2026-01-04", dtype="datetime64[D]").astype("datetime64[ns]")},
                 attrs={"units": "K"}, name="air")


# ── The public API ──────────────────────────────────────────────────────────


def test_the_public_api_is_importable_from_the_package():
    for name in labelarray.__all__:
        assert getattr(labelarray, name) is not None, name
    assert {"Array", "Collection", "Grouping", "factorize", "index_for", "LabelArrayError"} <= set(labelarray.__all__)


def test_the_adapter_holds_no_array_code_and_builds_on_the_library():
    assert adapter.NumpyArray is Array and adapter.NumpyData is Collection
    tree = ast.parse(inspect.getsource(adapter))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert not functions and classes == ["NumpyBackend"]
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert {"tidy3a.labelarray.array", "tidy3a.labelarray.collection", "tidy3a.labelarray"} <= imported


# ── Arrays ──────────────────────────────────────────────────────────────────


def test_selection_by_positions_and_labels_matches_xarray(grid):
    _same(grid.isel(lat=[3, 0], time=slice(1, None)), _xr(grid).isel(lat=[3, 0], time=slice(1, None)))
    _same(grid.isel({"lat": 1}), _xr(grid).isel(lat=1))
    _same(grid.sel(lat=[40.0, 10.0]), _xr(grid).sel(lat=[40.0, 10.0]))
    _same(grid.sel(time=np.datetime64("2026-01-02")), _xr(grid).sel(time=np.datetime64("2026-01-02")))
    _same(grid.isel(lat=np.array([True, False, True, False])), _xr(grid).isel(lat=[0, 2]))


def test_label_lookups_are_exact():
    big = Array([1.0, 2.0], ("k",), {"k": [2**53 + 1, 3]})
    assert big.sel(k=2**53 + 1).data == 1.0
    with pytest.raises(LabelError, match="no label"):
        big.sel(k=float(2**53))  # 2**53 + 1 is not 2**53
    with pytest.raises(LabelError, match="no label"):
        big.sel(k=4)


def test_transpose_rename_and_attributes(grid):
    turned = grid.transpose("time", "lat")
    _same(turned, _xr(grid).transpose("time", "lat"))
    assert turned.attrs == {"units": "K"} and turned.name == "air"
    renamed = grid.rename({"lat": "latitude"})
    assert renamed.dims == ("latitude", "time")
    np.testing.assert_array_equal(renamed.labels("latitude"), grid.labels("lat"))


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "median", "std", "var"])
def test_reductions_match_xarray(grid, op):
    got = grid.reduce(op, ["time"])
    want = getattr(_xr(grid).astype("float64"), op)("time").astype("float32") if op in ("sum", "mean") \
        else getattr(_xr(grid), op)("time")
    assert got.dims == ("lat",)
    np.testing.assert_allclose(got.data, want.values, rtol=1e-6)
    np.testing.assert_array_equal(got.labels("lat"), grid.labels("lat"))


def test_float32_sums_are_correctly_rounded_in_any_layout():
    values = np.array([[2.0**24, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -(2.0**24)]], dtype="float32")
    row = Array(values, ("one", "k"))
    column = Array(np.asfortranarray(values.T), ("k", "one"))
    assert row.reduce("sum", ["k"]).data.tolist() == [8.0]
    assert column.reduce("sum", ["k"]).data.tolist() == [8.0]


def test_missing_values_in_reductions(grid):
    data = grid.data.copy()
    data[0, 1] = np.nan
    gappy = Array(data, grid.dims, grid.coords)
    assert np.isnan(gappy.reduce("mean", ["time"]).data[0])
    np.testing.assert_allclose(gappy.reduce("mean", ["time"], na_rm=True).data, _xr(gappy).mean("time").values,
                               rtol=1e-6)  # xarray adds float32 in float32; the library rounds once
    assert np.isnan(gappy.reduce("sum", ["time"], na_rm=True, min_count=3).data[0])


def test_dot_matches_xarray():
    rotation = Array(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float64"), ("i", "feature"),
                     {"i": ["x", "y", "z"], "feature": ["x", "y", "z"]})
    points = Array(np.arange(12.0).reshape(4, 3), ("point", "feature"), {"feature": ["x", "y", "z"]})
    _same(rotation.dot(points, ["feature"]), xr.dot(_xr(rotation), _xr(points), dim="feature"))


def test_operators_align_by_label_and_refuse_what_differs():
    a = Array([1.0, 2.0, 3.0], ("x",), {"x": [1, 2, 3]})
    b = Array([30.0, 10.0, 20.0], ("x",), {"x": [3, 1, 2]})
    assert (a + b).data.tolist() == [11.0, 22.0, 33.0]
    first, second = align(a, b)
    assert first is a and second.labels("x").tolist() == [1, 2, 3]
    with pytest.raises(AlignError):
        a + Array([1.0, 2.0, 3.0], ("x",), {"x": [1, 2, 4]})


def test_concat_matches_xarray(grid):
    other = Array(grid.data * 2, grid.dims, grid.coords)
    got = concat([grid, other], "run", ["a", "b"])
    want = xr.concat([_xr(grid), _xr(other)], dim=xr.DataArray(["a", "b"], dims="run", name="run"))
    _same(got, want.transpose("run", "lat", "time"))


def test_bad_shapes_are_shape_errors():
    with pytest.raises(ShapeError, match="axis names"):
        Array(np.zeros(3), ("a", "b"))
    with pytest.raises(ShapeError, match="labels"):
        Array(np.zeros(3), ("a",), {"a": [1, 2]})


# ── Collections ─────────────────────────────────────────────────────────────


def _dataset(labels, values, name):
    return xr.Dataset({name: ("k", np.asarray(values, dtype="float64"))}, coords={"k": labels})


@pytest.mark.parametrize("how, xarray_join", [("left", "left"), ("inner", "inner"), ("outer", "outer")])
def test_joins_match_xarray(how, xarray_join):
    left, right = _dataset([1, 3, 5], [1, 2, 3], "a"), _dataset([5, 4, 1], [10, 20, 30], "b")
    got = Collection.from_xarray(left).join(Collection.from_xarray(right), how).to_xarray()
    want = xr.merge([left, right], join=xarray_join)
    xr.testing.assert_identical(got, want)


def test_joins_pair_labels_exactly():
    left = Collection({"a": Array([1.0, 2.0], ("k",))}, {"k": np.array([2**53 + 1, 7])})
    right = Collection({"b": Array([10.0, 20.0], ("k",))}, {"k": np.array([2**53, 7])})
    joined = left.join(right, "inner")
    assert joined.coords["k"].tolist() == [7]
    assert joined.array("b").data.tolist() == [20.0]


def test_nearest_matches_xarray():
    right = _dataset([0.0, 1.0, 2.0, 3.0], [10, 11, 12, 13], "v")
    wanted = np.array([0.1, 1.6, 5.0])
    got = Collection.from_xarray(right).nearest("k", wanted, 0.5).to_xarray()
    want = right.reindex(k=wanted, method="nearest", tolerance=0.5)
    xr.testing.assert_identical(got, want)


def test_collection_selection_assignment_and_dicts(grid):
    data = Collection({"air": grid, "count": Array(np.arange(4), ("lat",))}, grid.coords, attrs={"title": "t"})
    picked = data.isel(lat=[1, 2]).select(["count"])
    assert picked.dims == ("lat",) and picked.array("count").data.tolist() == [1, 2]
    assert data.sel(lat=20.0).dims == ("time",)
    added = data.assign("doubled", grid * 2).rename({"count": "n"})
    assert set(added.variables) == {"air", "n", "doubled"}
    assert data.to_dict()["dims"] == {"lat": 4, "time": 3}
    xr.testing.assert_identical(Collection.from_xarray(data.to_xarray()).to_xarray(), data.to_xarray())
    np.testing.assert_equal(data.to_dict(), data.to_xarray().to_dict(data="array"))


# ── Grouped reductions ──────────────────────────────────────────────────────


@pytest.mark.parametrize("groups", [3, 40])
@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "count"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_grouped_reductions_match_each_group_alone(groups, op, dtype):
    rng = np.random.default_rng(groups)
    codes = rng.integers(0, groups, 200)
    values = Array((rng.normal(size=(200, 2)) * 100).astype(dtype), ("point", "band"), {"band": ["r", "g"]})
    grouping = Grouping(codes, [f"g{i}" for i in range(groups)])
    got = grouped.reduce(values, op, grouping, "point", "group")
    assert got.dims == ("group", "band") and got.labels("group").tolist() == grouping.labels
    for code in range(groups):
        alone = values.isel(point=np.flatnonzero(codes == code)).reduce(op, ["point"])
        np.testing.assert_array_equal(got.data[code], alone.data)


def test_counts_and_broadcast_back_to_positions():
    grouping = Grouping(np.array([1, 0, 1, 1]), ["a", "b"])
    counts = grouped.counts(grouping, "key", per_position=2)
    assert counts.data.tolist() == [2, 6] and counts.labels("key").tolist() == ["a", "b"]
    stats = Array([10.0, 20.0], ("key",), {"key": ["a", "b"]})
    back = grouped.broadcast(stats, grouping, "point", "key", labels=np.array([5, 6, 7, 8]))
    assert back.dims == ("point",) and back.data.tolist() == [20.0, 10.0, 20.0, 20.0]
    assert back.labels("point").tolist() == [5, 6, 7, 8]


# ── Standing alone ──────────────────────────────────────────────────────────


WITHOUT_XARRAY_OR_PANDAS = textwrap.dedent("""
    import importlib.abc, sys

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("xarray", "pandas"):
                raise ImportError(f"{name} is not installed")

    sys.meta_path.insert(0, Blocked())

    import numpy as np
    from tidy3a.labelarray import Array, Collection, Grouping, grouped

    a = Array(np.arange(6.0).reshape(3, 2), ("k", "band"), {"k": [3, 1, 2]})
    b = Array([30.0, 10.0, 20.0], ("k",), {"k": [3, 1, 2]})
    left = Collection({"a": a}, a.coords)
    right = Collection({"b": b.isel(k=[1, 2])}, {"k": [1, 2]})
    joined = left.join(right, "inner")
    assert joined.coords["k"].tolist() == [1, 2]
    means = grouped.reduce(a + b, "mean", Grouping(np.array([0, 1, 0]), ["even", "odd"]), "k", "parity")
    assert means.data.tolist() == [[27.0, 28.0], [12.0, 13.0]]
    assert not {"xarray", "pandas"} & {name.split(".")[0] for name in sys.modules}
""")


def test_the_library_runs_without_xarray_or_pandas(tmp_path):
    script = tmp_path / "alone.py"
    script.write_text(WITHOUT_XARRAY_OR_PANDAS)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ── The library's README ────────────────────────────────────────────────────


README = Path(inspect.getfile(labelarray)).parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
PRINTED = re.compile(r"^print\((?P<expr>.*)\)\s+#\s+(?P<text>.+)$", re.MULTILINE)


@pytest.mark.parametrize("number", range(len(BLOCKS)))
def test_each_readme_example_runs_and_prints_what_it_says(number, capsys):
    code = BLOCKS[number]
    namespace: dict = {}
    exec(compile(code, f"labelarray README example {number + 1}", "exec"), namespace)  # noqa: S102 - the README's own code
    capsys.readouterr()
    for check in PRINTED.finditer(code):
        shown = " ".join(str(eval(check["expr"], namespace)).split())
        assert shown == " ".join(check["text"].split()), f"example {number + 1}: {check['expr']} prints {shown!r}"
