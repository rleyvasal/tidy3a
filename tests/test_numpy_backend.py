"""The experimental NumPy backend (plan milestone 9, step 3): same verbs, same results."""

import ast
import inspect
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
from labelarray import Array, Collection

import tidy3a
import tidy3a.numpy_backend as adapter
from tidy3a import (
    ReadError,
    TidyError,
    dot,
    fields,
    filter,
    formula,
    group_by,
    infer,
    mean,
    mutate,
    n,
    pull,
    read_array,
    set_backend,
    summarise,
    tidy,
)
from tidy3a.backend import XARRAY, get_backend
from tidy3a.numpy_backend import NUMPY, NumpyData

x, y, z, intensity, ring, distance = fields("x", "y", "z", "intensity", "ring", "distance")
XYZ = ["x", "y", "z"]
FIELDS = ["x", "y", "z", "intensity", "ring"]
# A 90° rotation about z, then 1 m along x: the points land at 49, 50, 51, and 3 m.
RECORDS = np.array([[0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2]], dtype="float32")
R = xr.DataArray(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], "float32"), dims=("i", "feature"),
                 coords={"i": XYZ, "feature": XYZ})
OFFSET = xr.DataArray(np.array([1, 0, 0], "float32"), dims="i", coords={"i": XYZ})
# These tests name or compare backends themselves, so they run once.
pytestmark = pytest.mark.backends("xarray", reason="names or compares backends itself")


@pytest.fixture(autouse=True)
def restore_default_backend():
    before = get_backend().name
    yield
    set_backend(before)


def section_7(backend):
    cloud = read_array(RECORDS, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS},
                       field_axis="feature", backend=backend)
    nearby = (cloud >> mutate(p=dot(R, [x, y, z], over="feature") + OFFSET)
              >> mutate(distance=formula(r"\sqrt{\sum_i p_i^2}")) >> filter(distance <= 50))
    grouped = nearby >> group_by(ring) >> mutate(centered=intensity - mean(intensity))
    return nearby, grouped, grouped >> summarise(points=n(), mean_intensity=mean(intensity))


def test_section_7_gives_the_same_printouts_values_and_types_on_both_backends():
    xarray_frames, numpy_frames = section_7("xarray"), section_7("numpy")
    assert isinstance(numpy_frames[0]._data, NumpyData)
    for a, b in zip(xarray_frames, numpy_frames):
        assert repr(a) == repr(b)
        left, right = a.collect(), b.collect()
        xr.testing.assert_identical(left, right)
        assert {v: left[v].dtype for v in left.variables} == {v: right[v].dtype for v in right.variables}
    nearby, grouped, per_ring = numpy_frames
    assert (nearby >> pull("point")).tolist() == [1, 2, 4]
    assert (nearby >> pull(distance)).tolist() == [49.0, 50.0, 3.0]
    assert (grouped >> pull("centered")).tolist() == [0.0, -10.0, 10.0]
    assert per_ring.collect()["mean_intensity"].values.tolist() == [10.0, 30.0]


def test_the_default_backend_and_backend_names():
    fresh = subprocess.run([sys.executable, "-c", "import tidy3a.backend as b; print(b.get_backend().name)"],
                           capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == "numpy"  # the package's own default, before any set_backend()
    set_backend("numpy")
    assert get_backend() is NUMPY
    assert tidy(xr.Dataset({"a": ("t", [1.0, 2.0])}))._backend is NUMPY
    with pytest.raises(TidyError, match="backend must be one of 'xarray', 'numpy', not 'dask'"):
        set_backend("dask")


def test_lazy_input_stays_on_xarray_unless_numpy_is_asked_for():
    dask = pytest.importorskip("dask.array")
    lazy = xr.Dataset({"a": ("t", dask.ones(4, chunks=2))})
    set_backend("numpy")
    assert tidy(lazy)._backend is XARRAY  # only the default: lazy data stays lazy
    with pytest.raises(TidyError, match='backend="numpy" holds eager, in-memory data only, but a is lazy'):
        tidy(lazy, backend="numpy")


def test_memory_mapped_files_stay_on_xarray(tmp_path):
    path = tmp_path / "values.bin"
    np.arange(6, dtype="<f4").tofile(path)
    set_backend("numpy")
    frame = read_array(path, value_type="float32", dims={"t": infer}, lazy=True)
    assert frame._backend is XARRAY
    with pytest.raises(ReadError, match="lazy, chunked, or memory-mapped"):
        read_array(path, value_type="float32", dims={"t": infer}, lazy=True, backend="numpy")


def test_a_frame_moves_between_backends_keeping_its_roles():
    frame = read_array(RECORDS, dims={"point": infer, "feature": 5}, labels={"feature": FIELDS}, field_axis="feature")
    moved = tidy(frame, backend="numpy")
    assert moved._backend is NUMPY and moved._field_axis == "feature"
    assert repr(moved) == repr(frame)
    back = tidy(moved, backend="xarray")
    xr.testing.assert_identical(back.collect(), frame.collect())


def test_joins_bring_raw_xarray_data_onto_this_datas_backend():
    left = tidy(xr.Dataset({"a": ("t", [1.0, 2.0])}, coords={"t": [1, 2]}), backend="numpy")
    joined = left >> tidy3a.left_join(xr.Dataset({"b": ("t", [5.0, 6.0])}, coords={"t": [2, 3]}))
    assert joined._backend is NUMPY
    assert joined.collect()["b"].values.tolist()[1] == 5.0


def test_the_backend_is_an_adapter_over_the_library():
    """Plan milestone 10, steps A5 and A7: the array code lives in the ``labelarray`` package."""
    assert adapter.NumpyArray is Array and adapter.NumpyData is Collection
    tree = ast.parse(inspect.getsource(adapter))
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert not functions and classes == ["NumpyBackend"]
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert {"labelarray.array", "labelarray.collection", "labelarray"} <= imported


def test_tidy3a_labelarray_names_the_labelarray_package_and_its_modules():
    """tidy3a 0.3.0 shipped the library as ``tidy3a.labelarray``; the name stays."""
    import labelarray
    import labelarray.sums
    import tidy3a.labelarray.sums
    from tidy3a.labelarray.index import index_for

    import tidy3a.labelarray
    from tidy3a.labelarray import grouped, sums

    assert tidy3a.labelarray is labelarray and tidy3a.labelarray.sums is labelarray.sums
    assert sums is labelarray.sums and grouped is labelarray.grouped and index_for is labelarray.index_for
