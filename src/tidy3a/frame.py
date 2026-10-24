"""The ``TidyArray`` wrapper and ``tidy()``.

A ``TidyArray`` holds an xarray Dataset plus the few facts xarray does not
keep: which axis is the field axis, where variables made by
``variables_from=`` came from, and what the reader inferred. Verbs receive
and return ``TidyArray`` objects through ``>>``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from tidy3a.errors import TidyError

# Long label lists are shortened in the inspection output.
_LABELS_SHOWN = 6
_ANNOTATION_COLUMN = 40


class TidyArray:
    """Named arrays with tidy3A's axis roles; build one with ``tidy()`` or ``read_array()``."""

    __slots__ = ("_data", "_field_axis", "_origins", "_source", "_notes")

    def __init__(
        self,
        data: xr.Dataset,
        *,
        field_axis: str | None = None,
        origins: Mapping[Hashable, tuple[str, tuple[Any, ...]]] | None = None,
        source: str | None = None,
        notes: Sequence[str] = (),
    ):
        self._data = data
        self._field_axis = field_axis
        # variable name -> (axis it came from, that axis's labels in order)
        self._origins = dict(origins or {})
        self._source = source
        self._notes = tuple(notes)

    def to_xarray(self) -> xr.Dataset:
        """The underlying xarray Dataset (a shallow copy sharing the same values)."""
        return self._data.copy(deep=False)

    def collect(self) -> xr.Dataset:
        """A computed xarray Dataset; eager and memory-mapped values are not copied."""
        return self._data.compute()

    def __rshift__(self, verb: Callable[[TidyArray], Any]) -> Any:
        if not callable(verb):
            raise TypeError(
                f"cannot pipe into {type(verb).__name__}; "
                "the right side of >> must be a tidy3A verb"
            )
        return verb(self)

    def __repr__(self) -> str:
        return _inspect(self)


def tidy(data: Any, *, field_axis: str | None = None) -> TidyArray:
    """Wrap an xarray ``DataArray`` or ``Dataset`` for tidy3A verbs.

    An unnamed DataArray becomes the variable ``values``. ``field_axis=``
    declares the axis whose labels act as column names, as in
    ``read_array``. A ``TidyArray`` is returned unchanged.
    """
    if isinstance(data, TidyArray):
        if field_axis is None or field_axis == data._field_axis:
            return data
        if data._field_axis is not None:
            raise TidyError(
                f'this data already has field axis "{data._field_axis}"; '
                f'it cannot also use field_axis="{field_axis}"'
            )
        check_field_axis(data._data, field_axis, TidyError)
        return TidyArray(
            data._data,
            field_axis=field_axis,
            origins=data._origins,
            source=data._source,
            notes=data._notes,
        )
    if isinstance(data, xr.DataArray):
        name = data.name if data.name is not None else "values"
        dataset = data.to_dataset(name=name)
        source = "xarray DataArray"
    elif isinstance(data, xr.Dataset):
        dataset = data
        source = "xarray Dataset"
    else:
        raise TypeError(
            f"tidy() wraps xarray DataArray and Dataset objects, not {type(data).__name__}; "
            "use read_array() for NumPy arrays and files"
        )
    if field_axis is not None:
        check_field_axis(dataset, field_axis, TidyError)
    return TidyArray(dataset, field_axis=field_axis, source=source)


def check_field_axis(data: xr.Dataset, axis: Any, error: type[TidyError]) -> None:
    """Raise unless *axis* can be a field axis: an existing axis with unique labels."""
    if not isinstance(axis, str):
        raise error(f"field_axis must be an axis name, not {type(axis).__name__}")
    if axis not in data.dims:
        axes = ", ".join(str(dim) for dim in data.dims) or "none"
        raise error(f'field_axis="{axis}" is not an axis of this data (axes: {axes})')
    if axis not in data.indexes:
        raise error(
            f'field_axis="{axis}" has no labels; its labels become column names, '
            "so give it labels first"
        )
    if not data.indexes[axis].is_unique:
        raise error(f'field_axis="{axis}" has repeated labels; column names must be unique')


# ── Inspection output ───────────────────────────────────────────────────────


def _inspect(frame: TidyArray) -> str:
    data = frame._data
    header = "<TidyArray"
    if frame._source:
        header += f" source: {frame._source}"
    lines = [header + ">"]

    for name, variable in data.data_vars.items():
        sizes = ", ".join(f"{dim}: {variable.sizes[dim]}" for dim in variable.dims)
        lines.append(f"{name} ({sizes}) {variable.dtype}")

    by_axis: dict[str, list[Hashable]] = {}
    for name, (axis, _) in frame._origins.items():
        by_axis.setdefault(axis, []).append(name)
    for axis, names in by_axis.items():
        listed = "  ".join(str(name) for name in names)
        lines.append(f"  {listed}    from axis {axis} (variables_from)")

    axes = [dim for dim in _axes_in_order(data)]
    width = max((len(str(axis)) for axis in axes), default=0) + 1
    rows = [_axis_row(frame, axis) for axis in axes]
    column = min(
        max((len(text) for text, note in rows if note), default=0) + 4,
        _ANNOTATION_COLUMN,
    )
    for axis, (text, note) in zip(axes, rows):
        label = f"{axis}:".ljust(width + 1)
        if note:
            lines.append(f"  {label}{text.ljust(column)}{note}")
        else:
            lines.append(f"  {label}{text}")
        if axis == "variable" and axis != frame._field_axis and axis in data.indexes:
            lines.append(
                '  hint: "variable" looks like an axis of variable names '
                "(xarray's to_dataarray default)."
            )
            lines.append(
                "        To select its labels as columns: "
                'tidy(da, field_axis="variable")'
            )

    lines.extend(frame._notes)
    return "\n".join(lines)


def _axes_in_order(data: xr.Dataset) -> list[Hashable]:
    seen: list[Hashable] = []
    for variable in data.data_vars.values():
        for dim in variable.dims:
            if dim not in seen:
                seen.append(dim)
    for dim in data.dims:
        if dim not in seen:
            seen.append(dim)
    return seen


def _axis_row(frame: TidyArray, axis: Hashable) -> tuple[str, str]:
    data = frame._data
    size = data.sizes[axis]
    if axis in data.indexes:
        text = _format_labels(data.indexes[axis].to_numpy())
        if axis == frame._field_axis:
            return text, "field axis (labels act as columns)"
        return text, "observation axis (labels are values)"
    if size == 0:
        return "unlabeled, empty (0 positions)", ""
    if size == 1:
        return "unlabeled, position 1", ""
    return f"unlabeled, positions 1 … {size:,}", ""


def _format_labels(values: np.ndarray) -> str:
    if len(values) == 0:
        return "no labels (0 positions)"
    shown = [str(value) for value in values]
    if len(shown) > _LABELS_SHOWN + 2:
        shown = [*shown[:_LABELS_SHOWN], "…", shown[-1]]
    return "  ".join(shown)
