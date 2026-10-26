"""Array backends (plan milestone 9): where tidy3a's verbs meet an array library.

The verbs, expressions, formulas, alignment rules, and inspection output hold
every tidy3a rule and message. They reach stored data only through a
backend: a collection of variables sharing axes and labels, and the arrays
in it. Arrays expose a small common surface (``dims``, ``sizes``, ``ndim``,
``dtype``, and ``values``, which computes NumPy values); everything else,
from label lookups to arithmetic under the alignment rules, is a backend
method.

``Backend`` below is the interface, written as a protocol that both backends
are checked against. ``NumpyBackend`` (``tidy3a.numpy_backend``) holds
eager, in-memory data and is the default where it holds the data without
loss (``backend_for``). ``XarrayBackend`` (``tidy3a.xarray_backend``) wraps
the xarray code the package used before the interface existed and holds
everything else: lazy, chunked, memory-mapped, and encoded data.
``set_backend`` chooses the default, and ``tidy()`` and the readers take
``backend=``. A backend raises tidy3a errors only where a limitation is its
own (computing a median of a whole chunked array); rule errors and their
hints come from the callers.

This module imports neither xarray nor pandas: each backend is imported when
first used, so eager pipelines on the NumPy backend run without them. Arrays
supplied from outside, such as a rotation matrix, are xarray objects;
``is_xarray`` recognises them without importing xarray, since an xarray object
can exist only once xarray has been imported.
"""

from __future__ import annotations

import mmap
import sys
from collections.abc import Hashable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tidy3a.errors import TidyError


def is_xarray(value: Any) -> bool:
    """An xarray array supplied from outside, e.g. a rotation matrix."""
    xr = sys.modules.get("xarray")
    return xr is not None and isinstance(value, xr.DataArray)


def is_xarray_object(value: Any) -> bool:
    xr = sys.modules.get("xarray")
    return xr is not None and isinstance(value, (xr.DataArray, xr.Dataset))


def memory_mapped(array: Any) -> bool:
    """Whether *array* views a memory-mapped file, however many views lie between.

    Older xarray versions hold a plain view of the ``np.memmap``.
    """
    base: Any = array
    while base is not None:
        if isinstance(base, (np.memmap, mmap.mmap)):
            return True
        base = getattr(base, "base", None)
    return False


def kept_positions(index: Any, size: int) -> tuple[Any, np.ndarray]:
    """An indexer for *index* on an axis of *size* positions (a mask, a slice, or positions),
    and the 0-based positions it keeps, as an int64 array the caller owns and may change
    once the indexer has been used.
    """
    if isinstance(index, slice):
        return index, np.arange(size, dtype=np.int64)[index].copy()
    index = np.asarray(index)
    if index.dtype == bool:
        index = np.flatnonzero(index).astype(np.int64, copy=False)
        return index, index
    index = index.astype(np.int64, copy=True)
    return index, index


@runtime_checkable
class Backend(Protocol):
    """What a backend provides; the verbs, expressions, formulas, and alignment rules use nothing else.

    A collection holds variables sharing axes and labels; an array is one variable
    with named axes. Both backends are checked against this protocol in the tests:
    the same methods, parameters, and defaults, and no public method outside it.
    Arrays also provide ``dims``, ``sizes``, ``shape``, ``ndim``, ``dtype``, and ``values``.
    """

    name: str

    # ── Conversion ────────────────────────────────────────────────────────────

    def wrap(self, data: Any) -> tuple[Any, str]:
        """A collection from an xarray object, and a description of its source."""

    def unsupported(self, data: Any) -> str | None:
        """Why xarray *data* cannot be held by this backend, or None when it can."""

    def outside(self, value: Any) -> Any:
        """An array from the calling code (an xarray object) as this backend's array; other values unchanged."""

    def to_xarray(self, data: Any) -> Any:
        """A shallow xarray Dataset sharing the values."""

    def collect(self, data: Any) -> Any:
        """An xarray Dataset with lazy values computed."""

    def to_dict(self, data: Any) -> dict:
        """NumPy arrays in the layout of xarray's ``Dataset.to_dict(data="array")``."""

    # ── Readers ───────────────────────────────────────────────────────────────

    def from_records(self, variables: Any, coords: Any) -> Any:
        """A collection of separately typed fields, each ``(dims, values)`` (``read_binary``)."""

    def from_array(self, values: Any, dims: Any, coords: Any, name: Any) -> Any:
        """One array as a collection of one variable (``read_array``)."""

    def from_split(self, values: Any, dims: Any, coords: Any, axis: Any) -> Any:
        """One variable per label on *axis* (``read_array(..., variables_from=axis)``)."""

    # ── Collections ───────────────────────────────────────────────────────────

    def variable_names(self, data: Any) -> list[Hashable]:
        """The variable names, in order."""

    def has_variable(self, data: Any, name: Any) -> bool:
        """Whether *name* is a variable."""

    def variable(self, data: Any, name: Any) -> Any:
        """The variable *name* as an array with its labels."""

    def variables(self, data: Any) -> Mapping[Hashable, Any]:
        """Every variable as an array, by name."""

    def layout(self, data: Any) -> list[tuple[Hashable, tuple[Hashable, ...], tuple[int, ...], Any]]:
        """Each variable's name, axes, shape, and NumPy type, read from storage without building arrays."""

    def sizes(self, obj: Any) -> Mapping[Hashable, int]:
        """Axis name to size, for a collection or an array."""

    def axes(self, obj: Any) -> Any:
        """The axis names, in order."""

    def has_axis(self, obj: Any, dim: Any) -> bool:
        """Whether *dim* is an axis; an unhashable name is not."""

    def has_coordinate(self, data: Any, name: Any) -> bool:
        """Whether *name* is a coordinate of the collection."""

    def attrs(self, obj: Any) -> Mapping[Hashable, Any]:
        """The attributes of a collection or an array."""

    def coordinate_attrs(self, data: Any, name: Any) -> Mapping[Hashable, Any]:
        """The attributes of coordinate *name*."""

    def coordinate_names(self, data: Any) -> Any:
        """The coordinate names, supporting set operations."""

    # ── Labels ────────────────────────────────────────────────────────────────

    def has_labels(self, obj: Any, dim: Any) -> bool:
        """Whether *dim* has labels on *obj*."""

    def labelled_axes(self, obj: Any) -> list[Hashable]:
        """The axes of *obj* that have labels."""

    def label_map(self, obj: Any) -> Mapping[Hashable, Any]:
        """Axis name to label list, for every labelled axis."""

    def label_values(self, obj: Any, dim: Any) -> Any:
        """The labels of *dim* as a NumPy array, for formatting and arithmetic."""

    def label_list(self, obj: Any, dim: Any) -> Any:
        """The labels of *dim* as Python values, for messages."""

    def has_label(self, obj: Any, dim: Any, label: Any) -> bool:
        """Whether *label* is among the labels of *dim*."""

    def label_count(self, obj: Any, dim: Any) -> int:
        """How many labels *dim* has."""

    def labels_unique(self, obj: Any, dim: Any) -> bool:
        """Whether no label on *dim* repeats."""

    def label_position(self, obj: Any, dim: Any, label: Any) -> Any:
        """The position of *label* on *dim*; an error when it is absent."""

    def compare_labels(self, left: Any, right: Any, dim: Any) -> tuple[str, int, int]:
        """``("same", 0, 0)``, ``("reordered", 0, 0)``, or ``("different", only in left, only in right)`` under the exact rule."""

    def labels_strict_subset(self, part: Any, whole: Any, dim: Any) -> bool:
        """Whether *part*'s labels on *dim* are some, but not all, of *whole*'s."""

    def reorder_like(self, value: Any, dim: Any, reference: Any) -> Any:
        """*value* with its labels on *dim* put in *reference*'s order."""

    def axis_labels(self, data: Any, axis: Any) -> Any:
        """An axis's labels as data along it; an unlabelled axis gives 1-based positions."""

    def axis_coordinate(self, data: Any, axis: Any) -> Any:
        """An axis's coordinate as an array, attributes included."""

    def keep(self, data: Any, axis: Any, index: Any) -> Any:
        """Positions on *axis* (a mask, a slice, or positions); an unlabelled axis is labelled with the 1-based positions kept."""

    # ── Arrays ────────────────────────────────────────────────────────────────

    def is_array(self, value: Any) -> bool:
        """Whether *value* is this backend's array."""

    def array(self, values: Any, dims: Any = (), coords: Any = None, name: Any = None) -> Any:
        """A new array from values, axis names, and labels."""

    def codes_like(self, codes: Any, values: Any) -> Any:
        """Group codes as an array on the first axis of *values*, sharing its labels."""

    def isel(self, obj: Any, indexers: Any, drop: Any = False) -> Any:
        """Selection by position on named axes; an integer drops the axis."""

    def sel(self, obj: Any, indexers: Any, drop: Any = False) -> Any:
        """Selection by label on named axes."""

    def field(self, data: Any, holder: Any, field_axis: Any, name: Any) -> Any:
        """One field of a holder, the field axis dropped, named after the field."""

    def transpose(self, value: Any, *dims: Any) -> Any:
        """*value* with its axes in the given order."""

    def rename_axes(self, value: Any, names: Any) -> Any:
        """*value* with axes renamed."""

    def concat(self, values: Any, dim: Any, labels: Any, *, exact: Any = False) -> Any:
        """Arrays stacked along a new labelled axis; ``exact`` refuses mismatched labels."""

    def apply(self, op: Any, *operands: Any) -> Any:
        """*op* on operands already aligned, broadcasting by axis name; scalars pass through."""

    def where(self, condition: Any, true: Any, false: Any) -> Any:
        """Values from *true* where *condition* holds, else from *false*."""

    def isin(self, value: Any, choices: Any) -> Any:
        """Whether each value is among *choices*."""

    def dot(self, left: Any, right: Any, over: Any) -> Any:
        """The product of *left* and *right* summed over the axes *over*."""

    def fresh(self, value: Any, keep_attrs: Any) -> Any:
        """*value* without storage encoding, and without attributes unless *keep_attrs*."""

    # ── Reductions and grouping ───────────────────────────────────────────────

    def reduce(self, array: Any, op: Any, axes: Any, na_rm: Any = False, min_count: Any = None) -> Any:
        """*op* over *axes* with tidy3a's missing-value rules (``na_rm``, ``min_count``)."""

    def group_reduce(self, array: Any, op: Any, groups: Any, others: Any, na_rm: Any, min_count: Any) -> Any:
        """*op* per group of *groups* over its axis and *others*."""

    def group_counts(self, groups: Any, per_position: Any) -> Any:
        """Positions per group, times *per_position*."""

    def broadcast(self, stats: Any, groups: Any) -> Any:
        """A per-group result back on each grouped position."""

    # ── Building collections ──────────────────────────────────────────────────

    def select_variables(self, data: Any, names: Any) -> Any:
        """The collection reduced to *names*, dropping labels no variable uses."""

    def assign(self, data: Any, name: Any, value: Any) -> Any:
        """Add or replace a variable whose labels agree with the collection's."""

    def replace(self, data: Any, name: Any, value: Any) -> Any:
        """Replace an existing variable's array."""

    def with_values(self, target: Any, values: Any) -> Any:
        """*target* with new values and no attributes."""

    def has_metadata(self, value: Any) -> bool:
        """Whether *value* carries attributes or encoding a write-back would clear."""

    def drop_position_axis(self, data: Any, axis: Any) -> Any:
        """The collection with the one remaining position on *axis* dropped."""

    def summary(self, results: Any, attrs: Any, coords: Any) -> Any:
        """A collection of summary results with the given attributes and coordinates."""

    def attach(self, data: Any, name: Any, values: Any, along: Any) -> Any:
        """A 1-D variable attached by position along *along*."""

    # ── Joins ─────────────────────────────────────────────────────────────────

    def rename_variables(self, data: Any, names: Any) -> Any:
        """The collection with variables renamed."""

    def nearest(self, right: Any, dim: Any, left: Any, tolerance: Any) -> Any:
        """*right* on *left*'s labels for *dim*, taking the nearest label within *tolerance*."""

    def merge(self, left: Any, right: Any, how: Any) -> Any:
        """Join on shared axes by label: ``left``, ``inner``, or ``outer``; conflicting attributes are dropped."""


def xarray_backend() -> Any:
    """The xarray backend, importing xarray on first use."""
    try:
        from tidy3a.xarray_backend import XARRAY
    except ImportError as error:
        if not _missing_xarray(error):
            raise
        raise TidyError(
            "this needs the xarray backend (xarray inputs, lazy, chunked, or memory-mapped data, or "
            'backend="xarray"), and xarray or pandas is not installed.\n'
            "  Install them: pip install 'tidy3a[xarray]'"
        ) from None
    return XARRAY


def _missing_xarray(error: ImportError) -> bool:
    return error.name in ("xarray", "pandas") or any(name in str(error) for name in ("xarray", "pandas"))




def __getattr__(name: str) -> Any:
    # XARRAY, XarrayBackend, and label_index live in tidy3a.xarray_backend; reaching
    # them from here imports xarray then, not when tidy3a is imported.
    if name in ("XARRAY", "XarrayBackend", "label_index"):
        import tidy3a.xarray_backend as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_BACKENDS = ("xarray", "numpy")
_default: str | None = None  # None: NumPy for data it holds without loss, xarray for the rest


def set_backend(name: str) -> None:
    """Choose the backend new data uses when none is named: ``"xarray"`` or ``"numpy"``.

    Without a choice, new data uses the NumPy backend where it holds the data
    without loss, and xarray otherwise: lazy, chunked, and memory-mapped
    values, coordinates other than an axis's labels, multi-level labels,
    time zones, and xarray storage encodings stay on xarray.
    ``set_backend("numpy")`` uses NumPy wherever it can hold the data,
    dropping storage encodings; lazy and other unsupported data still stay
    on xarray.
    """
    global _default
    _check_name(name)
    _default = name


def get_backend(name: str | None = None) -> Any:
    """The backend called *name*, or the default one (NumPy unless ``set_backend`` chose otherwise)."""
    if name is None:
        name = _default if _default is not None else "numpy"
    _check_name(name)
    if name == "xarray":
        return xarray_backend()
    from tidy3a.numpy_backend import NUMPY

    return NUMPY


def backend_for(data: Any, requested: str | None, error: type[TidyError] = TidyError) -> Any:
    """The backend that holds xarray *data*: the one asked for, or the default.

    Data the NumPy backend cannot hold stays on xarray, unless the NumPy
    backend was asked for by name; then it is an error that says why. With
    no choice made at all, data with xarray storage encodings stays on xarray
    too, since the NumPy backend does not keep them.
    """
    if requested is None and _default is None and storage_encodings(data):
        return xarray_backend()
    backend = get_backend(requested)
    return _fallback(backend, backend.unsupported(data), requested, error)


def storage_encodings(data: Any) -> bool:
    """Whether an xarray object, or any of its variables and coordinates, has a storage encoding."""
    if getattr(data, "encoding", None):
        return True
    variables = data.variables.values() if hasattr(data, "data_vars") else (data[c].variable for c in data.coords)
    return any(variable.encoding for variable in variables)


def reader_backend(values: dict[str, Any], requested: str | None, error: type[TidyError]) -> Any:
    """The backend for arrays a reader produced: memory-mapped ones stay on xarray, as ``backend_for``."""
    backend = get_backend(requested)
    if backend.name == "xarray":
        return backend
    mapped = next((name for name, array in values.items() if memory_mapped(array)), None)
    reason = None if mapped is None else f"{mapped} is lazy, chunked, or memory-mapped"
    return _fallback(backend, reason, requested, error)


def _fallback(backend: Any, reason: str | None, requested: str | None, error: type[TidyError]) -> Any:
    if reason is None:
        return backend
    if requested is not None:
        raise error(
            f'backend="{backend.name}" holds eager, in-memory data only, but {reason}.\n'
            '  Use backend="xarray", or load the values into memory first'
        )
    return xarray_backend()


def _check_name(name: Any) -> None:
    if name not in _BACKENDS:
        raise TidyError(f'backend must be one of {", ".join(map(repr, _BACKENDS))}, not {name!r}')
