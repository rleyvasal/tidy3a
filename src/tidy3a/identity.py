"""Structural identity: what a step does, as a value that separately written steps share (plan decision 5).

``structure(value)`` gives a hashable record of a verb's or an expression's
settings, built from the settings themselves: two pipelines written apart
that do the same work give equal structures, so a scan shares their work.
A missed share costs time and a false share a wrong result, so whatever a
value cannot capture is kept apart:

- every value carries its type, since ``1 == 1.0 == True`` in Python while
  ``x * 1`` and ``x * 1.0`` differ for integer ``x``; floats are exact
  (``float.hex``), with one NaN, and ``-0.0`` apart from ``0.0``;
- arrays, xarray objects, and frames are compared by identity (``Same``),
  never by contents, and the structure holds the object, so its ``id`` is
  not reused while the structure lives; an array changed in place after a
  step was written keeps its identity, so a structure is valid for one run;
- functions, lambdas, and values of other types are ``Unique``: equal to
  nothing, so anything holding one is shared only with itself;
- a frozen dataclass is its type and every field, so a field added later is
  in the structure without being listed; other classes give
  ``_structure()``, and labelarray's transforms their ``_settings()``.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Hashable, Mapping
from fractions import Fraction
from typing import Any

import numpy as np


class Same:
    """An object compared by identity; held, so its ``id`` stays its own while this lives."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Same) and other.value is self.value

    def __hash__(self) -> int:
        return id(self.value)

    def __repr__(self) -> str:
        return f"Same(<{type(self.value).__name__} at {id(self.value):#x}>)"


class Unique:
    """Equal only to itself: a value whose behaviour a structure cannot capture."""

    __slots__ = ("kind",)

    def __init__(self, kind: str):
        self.kind = kind

    def __repr__(self) -> str:
        return f"Unique({self.kind})"


_PLAIN = (type(None), bool, int, str, bytes)
_TIMES = (datetime.datetime, datetime.date, datetime.time, datetime.timedelta, datetime.timezone)


def structure(value: Any) -> Hashable:
    """*value* as a hashable record of its type and contents (see the module's notes)."""
    if isinstance(value, np.generic):  # first: np.float64 is a float, but x * np.float64(1) is not x * 1.0
        return ("numpy", value.dtype.str, value.tobytes())
    if isinstance(value, _PLAIN):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, Fraction):
        return ("Fraction", value.numerator, value.denominator)
    if isinstance(value, np.dtype):
        return ("dtype", value.str)
    if isinstance(value, range):
        return ("range", value.start, value.stop, value.step)
    if isinstance(value, _TIMES):
        return (type(value).__name__, repr(value))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(structure(item) for item in value))
    if isinstance(value, Mapping):  # in order: summaries and assignments keep theirs
        return ("dict", tuple((structure(key), structure(item)) for key, item in value.items()))
    own = getattr(type(value), "_structure", None)
    if own is not None:
        return own(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        return (type(value).__qualname__,
                tuple((field.name, structure(getattr(value, field.name))) for field in dataclasses.fields(value)))
    settings = getattr(value, "_settings", None)
    if settings is not None and type(value).__module__.startswith("labelarray."):
        return ("transform", type(value).__qualname__, structure(settings()))
    from tidy3a.frame import TidyArray

    if isinstance(value, (np.ndarray, TidyArray)) or hasattr(value, "__array__"):  # xarray objects have __array__
        return ("same", Same(value))
    if callable(value):
        return ("unique", Unique(getattr(value, "__qualname__", type(value).__name__)))
    return ("unique", Unique(type(value).__name__))
