"""Array backends (plan milestone 9): where tidy3A's verbs meet an array library.

The verbs, expressions, formulas, alignment rules, and inspection output hold
every tidy3A rule and message. They reach stored data only through a
backend: a collection of variables sharing axes and labels, and the arrays
in it. Arrays expose a small common surface (``dims``, ``sizes``, ``ndim``,
``dtype``, and ``values``, which computes NumPy values); everything else,
from label lookups to arithmetic under the alignment rules, is a backend
method.

``XarrayBackend`` (``tidy3a.xarray_backend``) is the default and wraps the
xarray code the package used before the interface existed. ``NumpyBackend``
(``tidy3a.numpy_backend``) is experimental and holds eager, in-memory data
only. ``set_backend`` chooses the default, and ``tidy()`` and the readers take
``backend=``. A backend raises tidy3A errors only where a limitation is its
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
from typing import Any

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


def xarray_backend() -> Any:
    """The xarray backend, importing xarray on first use."""
    from tidy3a.xarray_backend import XARRAY

    return XARRAY


def __getattr__(name: str) -> Any:
    # XARRAY, XarrayBackend, and label_index live in tidy3a.xarray_backend; reaching
    # them from here imports xarray then, not when tidy3a is imported.
    if name in ("XARRAY", "XarrayBackend", "label_index"):
        import tidy3a.xarray_backend as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_BACKENDS = ("xarray", "numpy")
_default = "xarray"


def set_backend(name: str) -> None:
    """Choose the backend new data uses when none is named: ``"xarray"`` (the default) or ``"numpy"``.

    The NumPy backend is experimental and holds eager, in-memory data only;
    lazy, chunked, and memory-mapped inputs stay on xarray when it is only
    the default.
    """
    global _default
    _check_name(name)
    _default = name


def get_backend(name: str | None = None) -> Any:
    """The backend called *name*, or the default one."""
    name = _default if name is None else name
    _check_name(name)
    if name == "xarray":
        return xarray_backend()
    from tidy3a.numpy_backend import NUMPY

    return NUMPY


def backend_for(data: Any, requested: str | None, error: type[TidyError] = TidyError) -> Any:
    """The backend that holds xarray *data*: the one asked for, or the default.

    Data the NumPy backend cannot hold stays on xarray, unless the NumPy
    backend was asked for by name; then it is an error that says why.
    """
    backend = get_backend(requested)
    if backend.name == "xarray":
        return backend
    return _fallback(backend, backend.unsupported(data), requested, error)


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
