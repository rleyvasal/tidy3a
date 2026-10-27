"""The ``TidyArray`` wrapper and ``tidy()``.

A ``TidyArray`` holds a backend's collection of variables (an xarray
Dataset with the xarray backend) plus the few facts it does not keep: which
axis is the field axis, where variables made by ``variables_from=`` came
from, and what the reader inferred. Verbs receive and return ``TidyArray``
objects through ``>>``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from tidy3a.backend import backend_for, get_backend, is_xarray_object, xarray_backend
from tidy3a.errors import TidyError, translating
from tidy3a.plan import Pipeline, Plan, recorded, start

# Long label lists are shortened in the inspection output.
_LABELS_SHOWN = 6
_ANNOTATION_COLUMN = 40


class TidyArray:
    """Named arrays with tidy3a's axis roles; build one with ``tidy()`` or ``read_array()``."""

    __slots__ = ("_data", "_backend", "_field_axis", "_origins", "_source", "_notes", "_groups", "_resolved",
                 "_record")

    def __init__(
        self,
        data: Any,
        *,
        backend: Any = None,
        field_axis: str | None = None,
        origins: Mapping[Hashable, tuple[str, tuple[Any, ...]]] | None = None,
        source: str | None = None,
        notes: Sequence[str] = (),
        groups: Sequence[Any] = (),
        resolved: tuple[Any, Any, Any] | None = None,
        record: Any = None,
    ):
        self._data = data
        self._backend = xarray_backend() if backend is None else backend
        self._field_axis = field_axis
        # variable name -> (axis it came from, that axis's labels in order)
        self._origins = dict(origins or {})
        self._source = source
        self._notes = tuple(notes)
        # group_by keys, in order (tidy3a.verbs.AxisKey / ValueKey)
        self._groups = tuple(groups)
        # (data, value key, tidy3a.expr.Groups): the key's groups, found once and valid for that data
        self._resolved = resolved
        # the source and the verbs that made this frame (tidy3a.plan.Record), or None before any verb
        self._record = record

    def _evolve(self, data: Any = None, *, notes: Sequence[str] = (), **changes: Any) -> TidyArray:
        """A copy with *data* and other parts replaced; *notes* are appended."""
        backend = self._backend
        data = self._data if data is None else data
        origins = changes.pop(
            "origins",
            {name: origin for name, origin in self._origins.items() if backend.has_variable(data, name)},
        )
        # What the reader inferred about an axis is stale once a verb resizes or removes it.
        sizes = backend.sizes(data)
        resized = [
            dim for dim, size in backend.sizes(self._data).items()
            if dim not in sizes or sizes[dim] != size
        ]
        new_notes = [
            note for note in self._notes
            if not any(note.startswith(f"inferred: {dim} =") for dim in resized)
        ]
        for note in notes:
            if note not in new_notes:
                new_notes.append(note)
        field_axis = changes.pop("field_axis", self._field_axis)
        if field_axis is not None and not backend.has_axis(data, field_axis):
            field_axis = None  # the verb removed the axis, so there is no field axis to name
        return TidyArray(
            data,
            backend=backend,
            field_axis=field_axis,
            origins=origins,
            source=changes.pop("source", self._source),
            notes=new_notes,
            groups=changes.pop("groups", self._groups),
            resolved=changes.pop("resolved", None),
            record=changes.pop("record", self._record),
        )

    @translating
    def to_xarray(self) -> Any:
        """A shallow Dataset copy sharing values, without computing lazy arrays."""
        return self._backend.to_xarray(self._data)

    @translating
    def collect(self) -> Any:
        """Compute lazy variables into memory as an xarray Dataset; the wrapped data stays lazy.

        Eager and memory-mapped values are not copied.
        """
        return self._backend.collect(self._data)

    @translating
    def to_dict(self) -> dict:
        """The data as NumPy arrays, in the layout of xarray's ``Dataset.to_dict(data="array")``.

        ``{"coords": …, "attrs": …, "dims": …, "data_vars": …}``, each variable
        and coordinate as ``{"dims", "attrs", "data"}``; lazy values are
        computed. Needs NumPy only; ``xr.Dataset.from_dict`` rebuilds the
        Dataset that ``collect()`` gives.
        """
        return self._backend.to_dict(self._data)

    def __rshift__(self, verb: Callable[[TidyArray], Any]) -> Any:
        if not callable(verb):
            raise TypeError(
                f"cannot pipe into {type(verb).__name__}; "
                "the right side of >> must be a tidy3a verb"
            )
        return verb(self)

    def __tidy__(self, **options: Any) -> Any:
        """tidy3's ``tidy()`` calls this: the frame's ``as_table()`` as a tidy3 frame."""
        from tidy3a.tables import to_tidy3

        return to_tidy3(self, **options)

    def _apply_verb(self, verb: Any) -> Any:
        """Execute a verb and record it on the result (``tidy3a.plan``); experimental frames may override this."""
        result = verb._apply(self)
        return recorded(self, verb, result) if isinstance(result, TidyArray) else result

    def explain(self) -> Any:
        """The source and each verb since, with what it changed: axes, variables, groups, and notes.

        Recorded as the verbs ran, so nothing is computed. It prints as text.
        """
        return Plan(start(self))

    def plan(self) -> Any:
        """The verbs that made this data, as a pipeline to run on other data: ``other >> nearby.plan()``."""
        return Pipeline(() if self._record is None else (step.verb for step in self._record.steps))

    @translating
    def __repr__(self) -> str:
        return _inspect(self)


@translating
def tidy(data: Any, *, axis: str | None = None, field_axis: str | None = None,
         fields: Sequence[str] | None = None, backend: str | None = None,
         blocks: Mapping[str, Any] | str | None = None, threads: int = 1) -> TidyArray:
    """Wrap an xarray ``DataArray`` or ``Dataset``, or read a table, for tidy3a verbs.

    An unnamed DataArray becomes the variable ``values``. ``field_axis=``
    declares the axis whose labels act as column names, as in
    ``read_array``. A ``TidyArray`` is returned unchanged, or moved to the
    backend named by ``backend=``.

    Wrapping preserves metadata and backing storage without reading
    measurement values. Keep file-backed sources open until collection.

    ``blocks={"time": 100}`` runs the verbs block by block along that axis,
    each block computed and wrapped on its own, so lazy, file-backed, and
    Dask-backed data is read a block at a time (``read_array``'s
    ``blocks=``, for datasets); ``{"time": "chunks"}``, or ``"chunks"``
    alone when one axis is chunked, takes the data's own Dask chunks as the
    blocks, so each is computed once. ``threads`` blocks run at a time.

    ``backend=`` is ``"xarray"`` or ``"numpy"``; without it, the default
    from ``set_backend()`` is used, or with none chosen, NumPy where it holds
    the data without loss (``docs/backends.md``).

    A Polars DataFrame or LazyFrame, or a tidy3 frame, is read as a table:
    ``axis=`` names the axis the rows run along, each column becomes a
    variable along it, and a column of that name becomes its labels.
    ``fields=`` packs the named columns into one array along ``field_axis=``,
    whose labels they become, as ``read_array(..., field_axis=)`` gives.
    """
    from tidy3a.tables import from_table, is_table

    if is_table(data):
        return from_table(data, axis=axis, field_axis=field_axis, fields=fields, backend=backend)
    if axis is not None or fields is not None:
        raise TidyError("axis= and fields= are for tables (Polars DataFrames and tidy3 frames), "
                        f"not {type(data).__name__}")
    if blocks is not None:
        from tidy3a.chunked import ChunkedFrame, DatasetSource

        source = data
        if isinstance(data, TidyArray):  # its field axis and backend carry over; its groups cannot
            if data._groups:
                raise TidyError("blocks= reads data in blocks, and a grouping needs every position of a group; "
                                "ungroup() first, then group_by() the blocked frame")
            if field_axis is not None and data._field_axis is not None and field_axis != data._field_axis:
                raise TidyError(f'this data already has field axis "{data._field_axis}"; '
                                f'it cannot also use field_axis="{field_axis}"')
            field_axis = data._field_axis if field_axis is None else field_axis
            backend = data._backend.name if backend is None else backend
            source = data.to_xarray()
        blocked, rows = _blocked_axis(blocks, source)
        return ChunkedFrame.open(DatasetSource(source, blocked, field_axis, backend), rows, threads)
    if isinstance(data, TidyArray):
        if backend is not None and get_backend(backend) is not data._backend:
            source = data.to_xarray()
            target = backend_for(source, backend)
            data = TidyArray(
                target.wrap(source)[0], backend=target, field_axis=data._field_axis,
                origins=data._origins, source=data._source, notes=data._notes, groups=data._groups,
                record=data._record,
            )
        if field_axis is None or field_axis == data._field_axis:
            return data
        if data._field_axis is not None:
            raise TidyError(
                f'this data already has field axis "{data._field_axis}"; '
                f'it cannot also use field_axis="{field_axis}"'
            )
        check_field_axis(data._data, field_axis, TidyError, data._backend)
        return data._evolve(field_axis=field_axis)
    if is_xarray_object(data):
        holder = backend_for(data, backend)
        dataset, source = holder.wrap(data)
    else:
        raise TypeError(
            f"tidy() wraps xarray DataArray and Dataset objects and reads Polars and tidy3 tables, "
            f"not {type(data).__name__}; use read_array() for NumPy arrays and files"
        )
    if field_axis is not None:
        check_field_axis(dataset, field_axis, TidyError, holder)
    return TidyArray(dataset, backend=holder, field_axis=field_axis, source=source)


def _blocked_axis(blocks: Any, data: Any) -> tuple[str, Any]:
    """The axis ``blocks=`` names and its block size: one axis mapped to positions or "chunks", or "chunks" alone."""
    if blocks == "chunks":
        from tidy3a.chunked import _chunks_along

        chunked = [str(dim) for dim in getattr(data, "dims", ()) if _chunks_along(data, dim) is not None]
        if len(chunked) != 1:
            listed = ", ".join(chunked) or "none"
            raise TidyError(f'blocks="chunks" needs one Dask-chunked axis to block along (chunked: {listed}); '
                            'name it: blocks={"time": "chunks"}')
        return chunked[0], "chunks"
    if isinstance(blocks, Mapping) and len(blocks) == 1:
        (axis, rows), = blocks.items()
        if isinstance(axis, str):
            return axis, rows
    raise TidyError('blocks= maps one axis to a number of positions or "chunks", e.g. blocks={"time": 100}, '
                    f"not {blocks!r}")


def check_field_axis(data: Any, axis: Any, error: type[TidyError], backend: Any = None) -> None:
    """Raise unless *axis* can be a field axis: an existing axis with unique labels."""
    backend = xarray_backend() if backend is None else backend
    if not isinstance(axis, str):
        raise error(f"field_axis must be an axis name, not {type(axis).__name__}")
    if not backend.has_axis(data, axis):
        axes = ", ".join(str(dim) for dim in backend.axes(data)) or "none"
        raise error(f'field_axis="{axis}" is not an axis of this data (axes: {axes})')
    if not backend.has_labels(data, axis):
        raise error(
            f'field_axis="{axis}" has no labels; its labels become column names, '
            "so give it labels first"
        )
    if not backend.labels_unique(data, axis):
        raise error(f'field_axis="{axis}" has repeated labels; column names must be unique')


# ── Inspection output ───────────────────────────────────────────────────────


def _inspect(frame: TidyArray) -> str:
    data = frame._data
    backend = frame._backend
    header = "<TidyArray"
    if frame._source:
        header += f" source: {frame._source}"
    lines = [header + ">"]

    for name, variable in backend.variables(data).items():
        sizes = ", ".join(f"{dim}: {variable.sizes[dim]}" for dim in variable.dims)
        lines.append(f"{name} ({sizes}) {variable.dtype}")

    by_axis: dict[str, list[Hashable]] = {}
    for name, (axis, _) in frame._origins.items():
        by_axis.setdefault(axis, []).append(name)
    for axis, names in by_axis.items():
        listed = "  ".join(str(name) for name in names)
        lines.append(f"  {listed}    from axis {axis} (variables_from)")

    axes = [dim for dim in _axes_in_order(data, backend)]
    width = max((len(str(axis)) for axis in axes), default=0) + 1
    rows = [_axis_row(frame, axis) for axis in axes]
    column = min(
        max((len(text) for text, note in rows if note), default=0) + 4,
        _ANNOTATION_COLUMN,
    )
    for axis, (text, note) in zip(axes, rows):
        label = f"{axis}:".ljust(width + 1)
        if note:
            lines.append(f"  {label}{text.ljust(column - 4)}    {note}")
        else:
            lines.append(f"  {label}{text}")
        if axis == "variable" and axis != frame._field_axis and backend.has_labels(data, axis):
            lines.append(
                '  hint: "variable" looks like an axis of variable names '
                "(xarray's to_dataarray default)."
            )
            lines.append(
                "        To select its labels as columns: "
                'tidy(da, field_axis="variable")'
            )

    if frame._groups:
        lines.append("groups: " + ", ".join(key.label for key in frame._groups))
    lines.extend(frame._notes)
    return "\n".join(lines)


def _axes_in_order(data: Any, backend: Any) -> list[Hashable]:
    seen: list[Hashable] = []
    for variable in backend.variables(data).values():
        for dim in variable.dims:
            if dim not in seen:
                seen.append(dim)
    for dim in backend.axes(data):
        if dim not in seen:
            seen.append(dim)
    return seen


def _axis_row(frame: TidyArray, axis: Hashable) -> tuple[str, str]:
    data, backend = frame._data, frame._backend
    size = backend.sizes(data)[axis]
    if backend.has_labels(data, axis):
        text = _format_labels(backend.label_values(data, axis))
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
    # Only the labels shown are formatted: the first few and the last.
    cut = len(values) > _LABELS_SHOWN + 2
    picked = np.concatenate([values[:_LABELS_SHOWN], values[-1:]]) if cut else values
    if values.dtype.kind == "M":
        # Times to the second, or dates alone when every label is at midnight.
        seconds = values.astype("datetime64[s]")
        midnight = (seconds == seconds.astype("datetime64[D]")).all()
        shown = list(np.datetime_as_string(picked.astype("datetime64[s]"), unit="D" if midnight else "s"))
    elif values.dtype.kind == "f":
        shown = [format(value, ".10g") for value in picked]  # display only; labels are unchanged
    else:
        shown = [str(value) for value in picked]
    if cut:
        shown = [*shown[:_LABELS_SHOWN], "…", shown[-1]]
    return "  ".join(shown)
