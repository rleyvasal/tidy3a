"""The tidy-layer verbs (plan §5): select, filter, slice, mutate, grouping, and joins.

Each verb returns a callable that takes a ``TidyArray`` and returns a new one,
for use with ``>>``. Verbs never change the shape of the data on their own
(``docs/decisions.md``).
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from tidy3a.align import align_to, check_join_axes
from tidy3a.errors import (
    AlignError,
    FilterError,
    GroupError,
    MutateError,
    RefError,
    SelectError,
    SliceError,
    SummariseError,
    TidyError,
)
from tidy3a.expr import (
    MISSING,
    AxisKey,
    BinOp,
    Context,
    Expr,
    Literal,
    Ref,
    UnaryOp,
    ValueKey,
    Vector,
    as_expr,
    axis_labels,
    check_name,
    evaluate,
    resolve,
    resolve_value_key,
)
from tidy3a.frame import TidyArray, tidy
from tidy3a.selectors import Column, Exclude, Selector


class Verb:
    """A step in a ``>>`` pipeline."""

    def __init__(self, text: str, apply: Callable[[TidyArray], Any]):
        self.text = text
        self._apply = apply

    def __call__(self, frame: Any) -> Any:
        if not isinstance(frame, TidyArray):
            raise TypeError(
                f"{self.text} works on tidy3A data; wrap xarray objects with tidy() "
                "or read files with read_array()"
            )
        return self._apply(frame)

    def __repr__(self) -> str:
        return f"<verb {self.text}>"


def _call_text(name: str, args: Sequence[Any], named: dict[str, Any] | None = None) -> str:
    parts = [_text(arg) for arg in args]
    parts += [f"{key} = {_text(value)}" for key, value in (named or {}).items()]
    return f"{name}({', '.join(parts)})"


def _text(value: Any) -> str:
    if isinstance(value, (Expr, Selector)):
        return value.text()
    if isinstance(value, ValueKey):
        return value.label
    if isinstance(value, Assignment):
        return value.text()
    if isinstance(value, list):
        return "[" + ", ".join(_text(v) for v in value) + "]"
    return repr(value) if not isinstance(value, str) else f'"{value}"'


def _with_positions(data: xr.Dataset, axis: Hashable) -> xr.Dataset:
    """Attach 1-based positions as labels, so later steps keep original identities."""
    if axis in data.indexes:
        return data
    return data.assign_coords({axis: np.arange(1, data.sizes[axis] + 1)})


# ── select ──────────────────────────────────────────────────────────────────


def select(*items: Any) -> Verb:
    """Keep variables and field-axis labels (columns), in the order given."""

    def apply(frame: TidyArray) -> TidyArray:
        ctx = Context(frame, verb="select", error=SelectError, grouped=False)
        data = ctx.data
        columns = _columns(ctx)
        flat = list(_flatten(items))
        chosen: list[Column] = list(columns) if flat and _is_exclusion(flat[0]) else []
        excluded: set[Hashable] = set()  # variables removed by ~, so labels do not bring them back
        named_labels: list[Hashable] = []
        for item in flat:
            if _is_exclusion(item):
                removed = {(c.kind, c.name) for c in _match(_excluded(item), columns, ctx)}
                chosen = [c for c in chosen if (c.kind, c.name) not in removed]
                excluded |= {name for kind, name in removed if kind == "var"}
            else:
                for column in _match(item, columns, ctx):
                    if column.kind == "var":
                        excluded.discard(column.name)
                    else:
                        named_labels.append(column.name)
                    if not any(c.kind == column.kind and c.name == column.name for c in chosen):
                        chosen.append(column)

        notes: list[str] = []
        for key in frame._groups:
            if isinstance(key, ValueKey) and isinstance(key.expr, Ref):
                kept = any(c.name == key.expr.name for c in chosen)
                match = [c for c in columns if c.name == key.expr.name]
                if not kept and match:
                    chosen.append(match[0])
                    notes.append(f"changed: kept {key.expr.name} because the data is grouped by it")

        labels = [c.name for c in chosen if c.kind == "label"]
        names = [c.name for c in chosen if c.kind == "var"]
        if labels:
            holders = [h for h in _holders(data, ctx.field_axis) if h not in excluded]
            if not holders and named_labels:
                raise SelectError(
                    f'"{named_labels[0]}" is a label on {ctx.field_axis}, but every variable with that '
                    "axis was excluded; keep one of them to select its labels"
                )
            names = [h for h in holders if h not in names] + names
        result = data[names] if names else data.drop_vars(list(data.data_vars))
        if labels and ctx.field_axis in result.dims:
            result = result.sel({ctx.field_axis: labels})
        return frame._evolve(result, notes=notes)

    return Verb(_call_text("select", items), apply)


def _holders(data: xr.Dataset, field_axis: Hashable | None) -> list[Hashable]:
    if field_axis is None:
        return []
    return [v for v in data.data_vars if field_axis in data[v].dims]


def _columns(ctx: Context) -> list[Column]:
    data = ctx.data
    columns = [Column("var", name, data[name]) for name in data.data_vars]
    holders = _holders(data, ctx.field_axis)
    for value in data.indexes[ctx.field_axis] if ctx.field_axis is not None else []:
        view = data[holders[0]].sel({ctx.field_axis: value}) if len(holders) == 1 else None
        columns.append(Column("label", value, view))
    return columns


def _flatten(items: Sequence[Any]):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        elif isinstance(item, Vector):
            yield from item.items
        else:
            yield item


def _is_exclusion(item: Any) -> bool:
    return isinstance(item, Exclude) or (
        isinstance(item, UnaryOp) and item.symbol in ("~", "-") and isinstance(item.operand, (Ref, Vector))
    )


def _excluded(item: Any) -> Any:
    if isinstance(item, Exclude):
        return item.item
    return list(item.operand.items) if isinstance(item.operand, Vector) else item.operand


def _columns_hint(ctx: Context) -> str:
    text = "\n  variables: " + (", ".join(map(str, ctx.data.data_vars)) or "none")
    if ctx.field_axis is not None:
        text += f"\n  labels on {ctx.field_axis}: " + "  ".join(map(str, ctx.data.indexes[ctx.field_axis]))
    return text


def _match(item: Any, columns: list[Column], ctx: Context) -> list[Column]:
    if isinstance(item, (list, tuple)):
        return [c for part in item for c in _match(part, columns, ctx)]
    if isinstance(item, Selector):
        return item.pick(columns)
    if isinstance(item, str):
        item = Ref(item)
    if not isinstance(item, Ref):
        raise TypeError(f"select() takes names, references, and selection helpers, not {_text(item)}")
    if not check_name(item, ctx) and item.python is not MISSING:
        raise SelectError(f'"{item.name}" is a notebook variable, not a column of this data.{_columns_hint(ctx)}')
    variables = [c for c in columns if c.kind == "var" and c.name == item.name]
    labels = [c for c in columns if c.kind == "label" and c.name == item.name]
    if item.kind == "var":
        if not variables:
            resolve(item, ctx)
        return variables
    if item.kind == "label":
        if not labels:
            resolve(item, ctx)
        return labels
    if variables and labels:
        resolve(item, ctx)  # raises the variable-or-label error
    if variables or labels:
        return variables or labels
    if item.name in ctx.data.dims:
        raise SelectError(
            f'"{item.name}" is an axis; select keeps columns. '
            f"To keep positions on {item.name}, use filter() or slice()"
        )
    resolve(item, ctx)  # raises the unknown-name error, with a filter hint for values
    return []


# ── filter ──────────────────────────────────────────────────────────────────


def filter(*predicates: Any) -> Verb:  # noqa: A001
    """Keep positions on observation axes where every predicate holds."""

    text = _call_text("filter", predicates)

    def apply(frame: TidyArray) -> TidyArray:
        if not predicates:
            return frame
        ctx = Context(frame, verb="filter", error=FilterError)
        data = ctx.data
        masks: dict[Hashable, xr.DataArray] = {}
        for predicate in predicates:
            for part in _conjuncts(as_expr(predicate)):
                mask = evaluate(part, ctx)
                part_text = part.text()
                if not isinstance(mask, xr.DataArray) or mask.ndim == 0:
                    raise FilterError(
                        f"{part_text} is a single value; filter needs a condition that varies "
                        "along one axis, e.g. filter(distance <= 50)"
                    )
                if mask.dtype != bool:
                    raise FilterError(f"{part_text} is {mask.dtype}, not true/false; filter needs a condition")
                if mask.ndim > 1:
                    raise FilterError(_multi_axis_message(part_text, mask.dims))
                axis = mask.dims[0]
                if axis == ctx.field_axis:
                    raise FilterError(
                        f"{part_text} varies along {axis}, the field axis; fields are columns, "
                        "so use select() to keep them"
                    )
                mask = align_to(mask, data, part_text, ctx.notes)
                masks[axis] = masks[axis] & mask if axis in masks else mask

        notes = list(ctx.notes)
        for axis, mask in masks.items():
            data = _with_positions(data, axis).isel({axis: mask.values})
            if data.sizes[axis] == 0:
                notes.append(f"changed: {axis}: 0 (all removed by {text})")
        return frame._evolve(data, notes=notes)

    return Verb(text, apply)


def _conjuncts(node: Expr) -> list[Expr]:
    if isinstance(node, BinOp) and node.symbol == "&":
        return _conjuncts(node.left) + _conjuncts(node.right)
    return [node]


def _multi_axis_message(text: str, dims: Sequence[Hashable]) -> str:
    first, second = dims[0], dims[1]
    return (
        f"{text} varies along {' and '.join(str(d) for d in dims)}; "
        "filter keeps whole positions on one axis.\n"
        f'  Keep {first} positions where any {second} matches:  filter(any({text}, over="{second}"))\n'
        f'  Keep {first} positions where all {second} match:    filter(all({text}, over="{second}"))\n'
        f"  Keep the shape, blank failing cells:  mutate(<field> = if_else({text}, <field>, nan))"
    )


# ── slice ───────────────────────────────────────────────────────────────────


def slice(*, drop: bool = False, **axes: Any) -> Verb:  # noqa: A001
    """Keep 1-based positions on named axes; negative positions exclude. Axes are kept."""

    def apply(frame: TidyArray) -> TidyArray:
        _no_groups(frame, "slice")
        data = frame._data
        for axis, spec in axes.items():
            index = _positions(data, axis, spec, frame)
            data = _with_positions(data, axis).isel({axis: _position_indexer(index)})
            if drop:
                if data.sizes[axis] != 1:
                    raise SliceError(
                        f"drop=True removes {axis} only when one position remains; "
                        f"{data.sizes[axis]:,} remain"
                    )
                data = data.isel({axis: 0}).drop_vars(axis, errors="ignore")
        return frame._evolve(data)

    named = {**axes, **({"drop": True} if drop else {})}
    return Verb(_call_text("slice", (), named), apply)


def slice_head(**axes: int) -> Verb:
    """The first *n* positions on each named axis."""
    return _slice_end(axes, head=True)


def slice_tail(**axes: int) -> Verb:
    """The last *n* positions on each named axis."""
    return _slice_end(axes, head=False)


def _slice_end(axes: dict[str, int], *, head: bool) -> Verb:
    name = "slice_head" if head else "slice_tail"

    def apply(frame: TidyArray) -> TidyArray:
        _no_groups(frame, name)
        data = frame._data
        for axis, count in axes.items():
            _check_slice_axis(data, axis, frame)
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise SliceError(f"{name}({axis} = {count!r}) needs a non-negative count")
            size = data.sizes[axis]
            count = min(int(count), size)
            index = list(range(count)) if head else list(range(size - count, size))
            data = _with_positions(data, axis).isel({axis: _position_indexer(index)})
        return frame._evolve(data)

    return Verb(_call_text(name, (), axes), apply)


def _position_indexer(index: list[int]) -> builtins.slice | list[int]:
    """Use basic indexing for regular ranges so file backends stay lazy."""
    if not index:
        return builtins.slice(0, 0)
    step = index[1] - index[0] if len(index) > 1 else 1
    if step and all(b - a == step for a, b in zip(index, index[1:])):
        stop = index[-1] + (1 if step > 0 else -1)
        return builtins.slice(index[0], None if stop < 0 else stop, step)
    return index


def _check_slice_axis(data: xr.Dataset, axis: str, frame: TidyArray) -> None:
    if axis not in data.dims:
        raise SliceError(f"{axis} is not an axis of this data (axes: {', '.join(map(str, data.dims))})")
    if axis == frame._field_axis:
        raise SliceError(f"{axis} is the field axis; fields are columns, so use select() to keep them")


def _positions(data: xr.Dataset, axis: str, spec: Any, frame: TidyArray) -> list[int]:
    _check_slice_axis(data, axis, frame)
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        wanted = [int(spec)]
    elif isinstance(spec, (list, tuple, range, np.ndarray)):
        wanted = [int(p) for p in spec]
    else:
        raise SliceError(f"slice({axis} = …) takes positions: an integer, a list, or a range")
    if 0 in wanted:
        raise SliceError(
            "positions start at 1, as in tidy3 and dplyr; position 0 does not exist.\n"
            f"  For the first item use slice({axis} = 1), or slice_head({axis} = 1)."
        )
    size = data.sizes[axis]
    if all(p > 0 for p in wanted):
        return [p - 1 for p in wanted if p <= size]  # past the end is dropped, as in tidy3
    if all(p < 0 for p in wanted):
        excluded = {-p - 1 for p in wanted}
        return [i for i in range(size) if i not in excluded]
    raise SliceError("slice positions must be all positive (keep) or all negative (exclude)")


def _no_groups(frame: TidyArray, name: str) -> None:
    if frame._groups:
        raise SliceError(f"{name} on grouped data is not supported yet; ungroup() first")


# ── mutate ──────────────────────────────────────────────────────────────────


class Assignment:
    """``assign([x, y, z], value)``: a vector, or a name that is not a Python identifier."""

    def __init__(self, target: Any, value: Any):
        if isinstance(target, (list, tuple)):
            target = as_expr(list(target))
            if not isinstance(target, Vector):
                raise MutateError("assign() targets a name or a list of references, e.g. [x, y, z]")
        elif not isinstance(target, (str, Ref)):
            raise MutateError("assign() targets a name or a list of references, e.g. [x, y, z]")
        self.target = target.name if isinstance(target, Ref) else target
        self.value = value

    def text(self) -> str:
        target = self.target.text() if isinstance(self.target, Vector) else str(self.target)
        return f"{target} = {_text(self.value)}"


def assign(target: Any, value: Any) -> Assignment:
    """A mutate assignment; ``assign([x, y, z], value)`` writes a vector at once."""
    return Assignment(target, value)


def mutate(*assignments: Any, **named: Any) -> Verb:
    """Add or replace variables, in order; field labels are written into their array.

    A formula with a left side, ``formula("p_i = …")``, may be passed on its own.
    """
    from tidy3a.formula import Formula

    positional = []
    for step in assignments:
        if isinstance(step, Formula) and step.target is not None:
            step = Assignment(step.target, step)
        if not isinstance(step, Assignment):
            raise TypeError(
                'mutate() takes name = value, assign(target, value), or formula("name = …")'
            )
        positional.append(step)
    steps = [*positional, *(Assignment(name, value) for name, value in named.items())]

    def apply(frame: TidyArray) -> TidyArray:
        for step in steps:
            ctx = Context(frame, verb="mutate", error=MutateError)
            value_expr = as_expr(step.value)
            value = _result_metadata(evaluate(value_expr, ctx), value_expr)
            notes = list(ctx.notes)
            if isinstance(step.target, Vector):
                data = _store_vector(frame, step.target, value, step.text(), notes)
            else:
                data = _store(frame, step.target, value, value_expr.text(), notes)
            origins = {k: v for k, v in frame._origins.items() if k in data.data_vars}
            if not isinstance(step.target, Vector):
                origins.pop(step.target, None) if step.target in frame._data.data_vars else None
            frame = frame._evolve(data, notes=notes, origins=origins)
        return frame

    return Verb(_call_text("mutate", positional, named), apply)


def _result_metadata(value: Any, node: Expr) -> Any:
    """New values need fresh storage encoding; computed values also need fresh attributes."""
    if isinstance(value, xr.DataArray):
        value = value.copy(deep=False)
        value.encoding = {}
        if not isinstance(node, (Ref, Literal)):
            value.attrs = {}
    return value


def _store(frame: TidyArray, name: Hashable, value: Any, text: str, notes: list[str]) -> xr.Dataset:
    data = frame._data
    field_axis = frame._field_axis if frame._field_axis in data.dims else None
    is_variable = name in data.data_vars
    holders = _holders(data, field_axis)
    is_label = field_axis is not None and name in data.indexes[field_axis] and bool(holders)
    if is_variable and is_label:
        raise MutateError(
            f'"{name}" is both a variable and a label on {field_axis}.\n'
            f'  To write the variable, rename it first; to write the label, the variable must not share its name'
        )
    if is_label:
        return _write_label(data, holders, field_axis, name, value, text, notes)
    if name in data.dims or name in data.coords:
        raise MutateError(f'"{name}" is an axis name; choose another name for the variable')
    if isinstance(value, np.ndarray) and value.ndim:
        raise MutateError(
            f"{text} is a NumPy array without axis names; name its axes with xr.DataArray(…, dims=…) "
            "or attach it with bind_cols(…, along=…)"
        )
    if isinstance(value, xr.DataArray):
        for dim in value.dims:
            if dim in value.indexes and dim in data.indexes:
                theirs, ours = value.indexes[dim], data.indexes[dim]
                if not ours.equals(theirs) and set(theirs) < set(ours):
                    raise MutateError(
                        f"{name} = {text} has {len(theirs)} of the {len(ours)} labels on {dim}, "
                        f"but variables share one {dim} axis.\n"
                        f'  Give it its own axis: rename_axes({text}, {dim}="component")'
                    )
        value = align_to(value, data, text, notes)
    else:
        value = xr.DataArray(value)
    with xr.set_options(arithmetic_join="exact"):
        return data.assign({name: value.drop_vars([c for c in value.coords if c not in value.dims])})


def _write_label(data, holders, field_axis, name, value, text, notes) -> xr.Dataset:
    if len(holders) > 1:
        raise MutateError(
            f'"{name}" is a field label on several variables ({", ".join(map(str, holders))}); '
            "select one first"
        )
    holder = holders[0]
    target = data[holder]
    field_dims = [d for d in target.dims if d != field_axis]
    if isinstance(value, np.ndarray) and value.ndim:
        raise MutateError(f"{text} is a NumPy array without axis names; name its axes with xr.DataArray")
    if isinstance(value, xr.DataArray):
        if set(value.dims) != set(field_dims):
            raise MutateError(
                f"{name} has axes ({', '.join(map(str, field_dims))}) but {text} has "
                f"({', '.join(map(str, value.dims)) or 'none'}); a written field must keep its shape"
            )
        value = align_to(value, data, text, notes).transpose(*field_dims)
        new_dtype = np.result_type(target.dtype, value.dtype)
        raw_value = value.values
    else:
        new_dtype = np.result_type(target.dtype, value)
        raw_value = value
    values = target.values
    if not values.flags.writeable:
        notes.append(f"changed: {holder} copied into memory to write {name} (it was memory-mapped)")
    if new_dtype != target.dtype:
        notes.append(
            f"changed: {holder} promoted {target.dtype} → {new_dtype} by mutate({name} = …); "
            f"all fields on {field_axis} now {new_dtype}"
        )
    values = values.astype(new_dtype, copy=True)
    position = data.indexes[field_axis].get_loc(name)
    axis_number = target.dims.index(field_axis)
    index = (np.s_[:],) * axis_number + (position,)
    values[index] = raw_value
    updated = target.copy(data=values)
    updated.encoding = {}
    updated.attrs = {}
    if target.attrs or target.encoding:
        notes.append(f"metadata: cleared attributes and encoding for {holder} after field write-back")
    return data.assign({holder: updated})


def _store_vector(frame: TidyArray, target: Vector, value: Any, text: str, notes: list[str]) -> xr.Dataset:
    names = [item.name for item in target.items]
    if not isinstance(value, xr.DataArray):
        raise MutateError(f"{text}: a vector assignment needs a vector value, with one axis labelled {names}")
    labelled = [d for d in value.dims if d in value.indexes and list(value.indexes[d]) == names]
    if len(labelled) != 1:
        raise MutateError(
            f"{text}: the value needs exactly one axis labelled {'  '.join(map(str, names))}, in that order; "
            f"its axes are ({', '.join(map(str, value.dims))})"
        )
    component_axis = labelled[0]
    data = frame._data
    for name in names:
        component = value.sel({component_axis: name}, drop=True)
        data = _store(frame._evolve(data), name, component, f"{text} [{name}]", notes)
    return data


# ── grouping ────────────────────────────────────────────────────────────────


def group_by(*keys: Any, **named: Any) -> Verb:
    """Group by axes (kept), a value field, or ``bins(...)`` of one."""

    def apply(frame: TidyArray) -> TidyArray:
        data = frame._data
        resolved: list[Any] = []
        for key in keys:
            if isinstance(key, ValueKey):
                resolved.append(key)
                continue
            name = key.name if isinstance(key, Ref) else key
            if not isinstance(name, Hashable) or isinstance(key, (Expr, list)) and not isinstance(key, Ref):
                raise GroupError("group_by() takes names, bins(...), or name = expression")
            if isinstance(key, Ref) and not check_name(key, Context(frame, verb="group_by", error=GroupError, grouped=False)):
                if key.python is not MISSING:
                    raise GroupError(f'"{name}" is a notebook variable, not an axis or field of this data')
            if name in data.dims:
                if name == frame._field_axis:
                    raise GroupError(f"{name} is the field axis; fields are columns, not groups")
                resolved.append(AxisKey(name))
            else:
                resolved.append(ValueKey(name, Ref(name)))
        for name, expr in named.items():
            resolved.append(ValueKey(name, as_expr(expr)))
        values = [key for key in resolved if isinstance(key, ValueKey)]
        if len(values) > 1:
            raise GroupError(
                "grouping by more than one value field is not supported yet; "
                f"got {', '.join(key.label for key in values)}"
            )
        if values:
            resolve_value_key(values[0], frame, [k.axis for k in resolved if isinstance(k, AxisKey)])
        return frame._evolve(groups=resolved)

    return Verb(_call_text("group_by", keys, named), apply)


def ungroup() -> Verb:
    """Remove grouping."""
    return Verb("ungroup()", lambda frame: frame._evolve(groups=()))


def summarise(*, groups: str | None = None, **named: Any) -> Verb:
    """One value per group for each summary; other observation axes are reduced."""
    if groups not in (None, "drop_last", "drop", "keep"):
        raise SummariseError('summarise() groups must be "drop_last", "drop", "keep", or None')

    def apply(frame: TidyArray) -> TidyArray:
        ctx = Context(frame, verb="summarise", error=SummariseError, mode="summarise")
        allowed = set(ctx.group_axes)
        if ctx.field_axis is not None:
            allowed.add(ctx.field_axis)
        if ctx.value_key is not None:
            allowed.add(ctx.groups().name)
        results: dict[Hashable, xr.DataArray] = {}
        for name, expr in named.items():
            node = as_expr(expr)
            value = evaluate(node, ctx)
            value = _result_metadata(_per_group(value, ctx), node)
            extra = [d for d in value.dims if d not in allowed]
            if extra:
                raise SummariseError(
                    f"{name} = {node.text()} still varies along {', '.join(map(str, extra))}; "
                    f"wrap it in a reduction such as mean({node.text()})"
                )
            results[name] = value
        coords = {axis: ctx.data[axis] for axis in ctx.group_axes if axis in ctx.data.indexes}
        data = xr.Dataset(results, attrs=dict(ctx.data.attrs)).assign_coords(
            {axis: values for axis, values in coords.items() if any(axis in v.dims for v in results.values())}
        )
        keys = list(frame._groups)
        mode = groups or "drop_last"
        kept = [] if mode == "drop" else keys if mode == "keep" else keys[:-1]
        kept = [AxisKey(key.name) if isinstance(key, ValueKey) else key for key in kept]
        return frame._evolve(data, notes=ctx.notes, groups=kept, origins={})

    return Verb(_call_text("summarise", (), named), apply)


summarize = summarise


def _per_group(value: Any, ctx: Context) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(value, np.ndarray) and value.ndim:
        raise SummariseError("a summary must be a single value per group, not a NumPy array")
    shape: dict[Hashable, Any] = {}
    for axis in ctx.group_axes:
        shape[axis] = ctx.data[axis].values if axis in ctx.data.indexes else None
    if ctx.value_key is not None:
        groups = ctx.groups()
        shape[groups.name] = groups.labels
    if not shape:
        return xr.DataArray(value)
    dims = list(shape)
    sizes = [len(v) if v is not None else ctx.data.sizes[d] for d, v in shape.items()]
    coords = {d: v for d, v in shape.items() if v is not None}
    return xr.DataArray(np.full(sizes, value), dims=dims, coords=coords)


# ── pull, bind_cols, joins ──────────────────────────────────────────────────


def pull(item: Any) -> Verb:
    """A variable's values, a field's values, or an axis's labels, as a NumPy array."""

    def apply(frame: TidyArray) -> np.ndarray:
        ctx = Context(frame, verb="pull", error=RefError, grouped=False)
        ref = Ref(item) if isinstance(item, str) else item
        if not isinstance(ref, Ref):
            raise TypeError("pull() takes one name")
        if not check_name(ref, ctx) and ref.python is not MISSING:
            raise RefError(f'"{ref.name}" is a notebook variable, not data to pull')
        if ref.kind == "any" and ref.name in ctx.data.dims and ref.name not in ctx.data.data_vars:
            return axis_labels(ctx.data, ref.name).values
        return resolve(ref, ctx).values

    return Verb(_call_text("pull", (item,)), apply)


def bind_cols(*arrays: Any, along: str, **named: Any) -> Verb:
    """Attach outside 1-D arrays by position along an axis, taking its labels."""

    def apply(frame: TidyArray) -> TidyArray:
        data = frame._data
        if along not in data.dims:
            raise AlignError(f'bind_cols(along="{along}"): {along} is not an axis of this data')
        items = list(named.items())
        for array in arrays:
            if isinstance(array, xr.DataArray) and array.name is not None:
                items.append((array.name, array))
            else:
                raise AlignError(
                    "bind_cols() needs a name for each array: bind_cols(scores=scores, "
                    f'along="{along}")'
                )
        size = data.sizes[along]
        for name, array in items:
            if name in data.data_vars:
                raise AlignError(f"bind_cols(): {name} is already a variable; choose another name")
            if isinstance(array, xr.DataArray):
                if array.dims != (along,):
                    raise AlignError(f"bind_cols(): {name} must be 1-D along {along}, not ({', '.join(map(str, array.dims))})")
                if along in array.indexes:
                    raise AlignError(f"bind_cols(): {name} already has labels on {along}; combine it with left_join()")
                values = array.values
            else:
                values = np.asarray(array)
                if values.ndim != 1:
                    raise AlignError(f"bind_cols(): {name} must be 1-D, not {values.ndim}-D")
            if len(values) != size:
                raise AlignError(
                    f"bind_cols(): {name} has {len(values):,} values but {along} has {size:,} positions; "
                    "attaching by position needs equal lengths"
                )
            coords = {along: data.indexes[along]} if along in data.indexes else None
            data = data.assign({name: xr.DataArray(values, dims=along, coords=coords)})
        return frame._evolve(data)

    return Verb(_call_text("bind_cols", arrays, {**named, "along": along}), apply)


def left_join(other: Any, *, suffix: tuple[str, str] = ("_x", "_y"), tolerance: Any = None) -> Verb:
    """Bring in *other*'s variables, keeping this data's labels on shared axes."""
    return _join("left", other, suffix, tolerance)


def inner_join(other: Any, *, suffix: tuple[str, str] = ("_x", "_y"), tolerance: Any = None) -> Verb:
    """Bring in *other*'s variables, keeping only labels present in both."""
    return _join("inner", other, suffix, tolerance)


def full_join(other: Any, *, suffix: tuple[str, str] = ("_x", "_y"), tolerance: Any = None) -> Verb:
    """Bring in *other*'s variables, keeping all labels; gaps become missing."""
    return _join("full", other, suffix, tolerance)


def _join(how: str, other: Any, suffix: Any, tolerance: Any) -> Verb:
    name = f"{how}_join"
    if (
        not isinstance(suffix, (tuple, list)) or len(suffix) != 2
        or not all(isinstance(part, str) for part in suffix)
    ):
        raise TypeError(f"{name}() suffix must be a pair of strings, e.g. ('_x', '_y')")

    def apply(frame: TidyArray) -> TidyArray:
        right_frame = other if isinstance(other, TidyArray) else tidy(other)
        left, right = frame._data, right_frame._data
        check_join_axes(left, right, "the other data")
        clashes = [v for v in left.data_vars if v in right.data_vars]
        left = left.rename({v: f"{v}{suffix[0]}" for v in clashes if suffix[0]})
        right = right.rename({v: f"{v}{suffix[1]}" for v in clashes if suffix[1]})
        again = [v for v in left.data_vars if v in right.data_vars]
        if again:
            raise AlignError(f"{name}(): {', '.join(map(str, again))} still clash after suffixes {suffix!r}")
        if tolerance is not None:
            if how != "left":
                raise TidyError(f"tolerance= is supported on left_join in this release, not {name}")
            for dim in left.dims:
                if dim in right.dims and dim in left.indexes and dim in right.indexes:
                    right = right.reindex({dim: left.indexes[dim]}, method="nearest", tolerance=tolerance)
        join = {"left": "left", "inner": "inner", "full": "outer"}[how]
        merged = xr.merge([left, right], join=join, compat="no_conflicts", combine_attrs="drop_conflicts").drop_encoding()
        attributes = [("dataset", left.attrs, right.attrs, merged.attrs)]
        for axis in left.coords.keys() & right.coords.keys() & merged.coords.keys():
            attributes.append((f"coordinate {axis}", left[axis].attrs, right[axis].attrs, merged[axis].attrs))
        notes = []
        for subject, first, second, kept in attributes:
            dropped = sorted((set(first) | set(second)) - set(kept), key=str)
            if dropped:
                notes.append(f"metadata: dropped conflicting attributes for {subject}: {', '.join(map(str, dropped))}")
        origins = {**right_frame._origins, **frame._origins}
        field_axis = frame._field_axis or right_frame._field_axis
        return frame._evolve(merged, notes=notes, origins=origins, field_axis=field_axis)

    return Verb(f"{name}(…)", apply)
