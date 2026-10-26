"""Tables to and from tidy3a: Polars DataFrames and tidy3 frames (plan milestone 8, part 2).

``as_table()`` turns data with one observation axis into a Polars DataFrame,
one row per position; ``tidy(table, axis=...)`` (``from_table``) reads one
back. Polars is optional and imported only here, when a table is made or
read; tidy3a never imports tidy3 except in ``TidyArray.__tidy__``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from tidy3a.backend import reader_backend
from tidy3a.errors import ReadError, TidyError
from tidy3a.expr import axis_labels
from tidy3a.frame import TidyArray, check_field_axis
from tidy3a.verbs import Verb, _call_text

# Polars stores times and durations in these units only (and dates as days).
_POLARS_UNITS = ("ms", "us", "ns")
# Units NumPy converts to milliseconds exactly.
_TO_MILLISECONDS = ("s", "m", "h", "D", "W")


def _polars(what: str) -> Any:
    try:
        import polars
    except ImportError:
        raise TidyError(
            f"{what} gives a Polars DataFrame, and Polars is not installed.\n"
            "  pip install polars, or pip install 'tidy3a[polars]'"
        ) from None
    return polars


# ── as_table ────────────────────────────────────────────────────────────────


def as_table(*, long: bool = False) -> Verb:
    """The data as a Polars DataFrame, one row per position on its observation axis.

    The axis's labels (original positions after a filter, as ``pull`` gives
    them) are a column named after the axis; field-axis labels and the other
    variables along the axis are columns. Data with more than one
    observation axis needs ``long=True``, which makes one row per cell, with
    a column for each axis's labels and variables repeated along the axes
    they lack.
    """
    named = {"long": long} if long else {}

    def apply(frame: TidyArray) -> Any:
        pl = _polars("as_table()")
        data, backend, field_axis = frame._data, frame._backend, frame._field_axis
        sizes = dict(backend.sizes(data))
        axes = [dim for dim in sizes if dim != field_axis]
        if not axes:
            raise TidyError("as_table() needs an observation axis; this data has none besides the field axis")
        if len(axes) > 1 and not long:
            listed = ", ".join(f"{dim} ({sizes[dim]:,})" for dim in axes)
            rows = int(np.prod([sizes[dim] for dim in axes], dtype=np.int64))
            raise TidyError(
                f"as_table() makes one row per position on one observation axis; this data has {listed}.\n"
                f"  as_table(long=True) makes one row per cell: {rows:,} rows"
            )
        columns: dict[str, np.ndarray] = {}
        grids = np.meshgrid(*[np.arange(sizes[dim]) for dim in axes], indexing="ij")
        for number, dim in enumerate(axes):
            labels = np.asarray(axis_labels(data, dim, backend).values)
            _add(columns, str(dim), labels[grids[number].ravel()] if long else labels)
        for name, value in backend.variables(data).items():
            for column, part in _parts(name, value, field_axis, backend):
                dims = tuple(part.dims)
                if not long and dims != tuple(axes):
                    shown = ", ".join(str(dim) for dim in dims) or "none"
                    raise TidyError(
                        f'as_table(): "{name}" has no {axes[0]} axis (its axes: {shown}); '
                        "every column runs along the rows"
                    )
                values = np.asarray(part.values)
                if long:
                    values = _broadcast(values, dims, axes, sizes).ravel()
                _add(columns, column, values)
        return pl.DataFrame([_series(pl, name, values) for name, values in columns.items()])

    return Verb(_call_text("as_table", (), named), apply, operation="as_table", arguments=long)


def _parts(name: Hashable, value: Any, field_axis: Hashable | None, backend: Any) -> list[tuple[str, Any]]:
    """A variable's columns: one per field-axis label for a field holder, else the variable itself."""
    if field_axis is None or field_axis not in value.dims:
        return [(str(name), value)]
    return [(str(label), backend.sel(value, {field_axis: label}))
            for label in backend.label_list(value, field_axis)]


def _add(columns: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    if name in columns:
        raise TidyError(f'as_table() would make the column "{name}" twice; rename the axis or the variable first')
    columns[name] = values


def _broadcast(values: np.ndarray, dims: tuple, axes: list, sizes: dict) -> np.ndarray:
    """*values* (on *dims*) laid out on all of *axes*, repeated along the axes it lacks."""
    order = sorted(range(len(dims)), key=lambda k: axes.index(dims[k]))
    values = np.transpose(values, order)
    shape = [sizes[dim] if dim in dims else 1 for dim in axes]
    return np.broadcast_to(values.reshape(shape), [sizes[dim] for dim in axes])


def _series(pl: Any, name: str, values: np.ndarray) -> Any:
    kind = values.dtype.kind
    if kind == "c":
        raise TidyError(
            f'as_table(): "{name}" holds complex values, which Polars has no type for; '
            "make columns of real() and imag() first"
        )
    if kind in "mM":
        unit = np.datetime_data(values.dtype)[0]
        if unit not in _POLARS_UNITS and not (kind == "M" and unit == "D"):
            if unit not in _TO_MILLISECONDS:
                raise TidyError(f'as_table(): "{name}" has {unit!r} time steps, which Polars cannot hold')
            values = values.astype(f"{'datetime64' if kind == 'M' else 'timedelta64'}[ms]")
    if kind in "OU" and all(isinstance(v, str) for v in values.ravel().tolist()):
        return pl.Series(name, values.tolist(), dtype=pl.String)
    return pl.Series(name, values)


# ── TidyArray.__tidy__ ──────────────────────────────────────────────────────


def to_tidy3(frame: TidyArray, **options: Any) -> Any:
    """``tidy3.tidy(frame)``: the frame's table as a tidy3 frame."""
    try:
        import tidy3
    except ImportError:
        raise TidyError("tidy3's tidy() needs tidy3, which is not installed: pip install tidy3") from None
    return tidy3.tidy(frame >> as_table(**options))


# ── tidy(table, axis=...) ───────────────────────────────────────────────────


def is_table(data: Any) -> bool:
    """A Polars DataFrame or LazyFrame, or a tidy3 frame (anything with ``to_polars()``)."""
    module = type(data).__module__.split(".")[0]
    return (module == "polars" and type(data).__name__ in ("DataFrame", "LazyFrame")) or \
        (module != "tidy3a" and callable(getattr(data, "to_polars", None)))


def from_table(table: Any, *, axis: str | None, field_axis: str | None, fields: Sequence[str] | None,
               backend: str | None) -> TidyArray:
    if not isinstance(axis, str) or not axis:
        raise ReadError(
            "axis= names the axis the rows run along: tidy(table, axis=\"point\"); "
            "a column of that name becomes its labels"
        )
    notes: list[str] = []
    kind = type(table).__name__
    if type(table).__module__.split(".")[0] != "polars":
        table, source = table.to_polars(), "tidy3 frame"
        notes.append("materialized: collected the tidy3 frame")
    elif kind == "LazyFrame":
        table, source = table.collect(), "Polars LazyFrame"
        notes.append("materialized: collected the Polars LazyFrame")
    else:
        source = "Polars DataFrame"
    columns = {name: _column(table[name], name, notes) for name in table.columns}
    coords = {}
    if axis in columns:
        labels = columns.pop(axis)
        if table[axis].null_count():
            count = table[axis].null_count()
            raise ReadError(f'the "{axis}" column has {count:,} missing value{"" if count == 1 else "s"}; '
                            "labels cannot be missing")
        coords[axis] = labels
    variables: dict[Hashable, tuple] = {}
    if fields is not None or field_axis is not None:
        variables["values"] = _packed(table, columns, axis, field_axis, fields)
        coords[field_axis] = np.array(list(fields), dtype=object)
    for name, values in columns.items():
        if name in variables:
            raise ReadError(f'the column "{name}" has the name fields= gives the packed array; rename it first')
        variables[name] = (axis, values)
    holder = reader_backend({str(name): values for name, (_, values) in variables.items()}, backend, ReadError)
    data = holder.from_records(variables, coords)
    if field_axis is not None:
        check_field_axis(data, field_axis, ReadError, holder)
    return TidyArray(data, backend=holder, field_axis=field_axis, source=source, notes=notes)


def _packed(table: Any, columns: dict, axis: str, field_axis: str | None, fields: Sequence[str] | None) -> tuple:
    if fields is None:
        raise ReadError(f'field_axis= needs fields=, the columns to pack along "{field_axis}"')
    if field_axis is None:
        raise ReadError('fields= needs field_axis=, the name of the axis the packed columns run along')
    fields = list(fields)
    for name in fields:
        if name not in columns:
            raise ReadError(f'fields= names no column "{name}"; columns: {", ".join(table.columns)}')
    types = [str(table.schema[name]) for name in fields]
    if len(set(types)) > 1:
        raise ReadError(f"fields {', '.join(fields)} have different types ({', '.join(dict.fromkeys(types))}); "
                        "cast them to one type first")
    return (axis, field_axis), np.stack([columns.pop(name) for name in fields], axis=1)


def _column(series: Any, name: str, notes: list[str]) -> np.ndarray:
    """One column as a NumPy array, under tidy3a's type rules."""
    import polars as pl

    dtype = series.dtype
    for kind, words in ((pl.List, "a list column"), (pl.Struct, "a struct column"), (pl.Binary, "a binary column"),
                        (pl.Array, "a fixed-size array column"), (pl.Object, "a column of Python objects")):
        if dtype == kind or isinstance(dtype, kind):
            raise ReadError(f'column "{name}" is {words}, which has no NumPy type; '
                            "unnest or convert it to numbers, text, or times first")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        raise ReadError(f'column "{name}" has times with a time zone ({dtype.time_zone}); '
                        'convert them to one zone and drop it first (.dt.replace_time_zone(None))')
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return np.array(series.cast(pl.String).to_list(), dtype=object)
    if dtype == pl.Date or isinstance(dtype, (pl.Datetime, pl.Duration)):
        return series.to_numpy()
    if dtype.is_numeric() or dtype == pl.Boolean:
        nulls = series.null_count()
        if not nulls or dtype.is_float():
            return series.to_numpy()
        target = np.float32 if dtype in (pl.Int8, pl.Int16, pl.UInt8, pl.UInt16) else np.float64
        notes.append(f"changed: {name} had {nulls:,} missing value{'' if nulls == 1 else 's'}; "
                     f"stored as {np.dtype(target).name} with NaN")
        return series.cast(pl.Float64).to_numpy().astype(target)
    raise ReadError(f'column "{name}" has the Polars type {dtype}, which tidy3a does not read')
