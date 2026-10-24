"""Experimental eager execution of the point-cloud workflow on numpy_core.

Use the usual expressions and verbs from tidy3a, with read_array()/tidy()
from this module. Existing readers and collect()/to_xarray() are conversion
boundaries. The pipeline stores only core arrays between those boundaries.

This eager prototype supports selection, mutation (including field write-back),
filtering, positional slices, numeric and logical reductions, and reusable
axis/single-value grouping. Joins, selectors, bins, and lazy inputs remain
outside this prototype's scope.
"""

import operator
from dataclasses import dataclass

import numpy as np
import xarray as xr

import numpy_core as core
import tidy3a as ta
from tidy3a.expr import (
    MISSING, AxisKey, BinOp, Call, Context, Literal, NameRef, Reduce, Ref,
    UnaryOp, ValueKey, Vector, _OPERATORS, _plain, as_expr, walk,
)
from tidy3a.frame import TidyArray
from tidy3a.formula import Formula
from tidy3a.verbs import _conjuncts, _flatten, _position_indexer, _positions as _slice_positions


def _from_array(value):
    if (not value.variable._in_memory or not isinstance(value.data, np.ndarray)
            or isinstance(value.data, np.memmap)):
        raise NotImplementedError("NumPy prototype requires eager NumPy input; collect it explicitly first")
    if any(c not in value.dims for c in value.coords):
        raise NotImplementedError("NumPy prototype currently supports dimension coordinates only")
    return core.labelled(value.data, value.dims,
                         {d: value.coords.variables[d].data for d in value.dims if d in value.coords}, value.attrs)


def tidy(data, *, field_axis=None):
    """Convert eager input to an owned, read-only snapshot for safe group caching."""
    if isinstance(data, NumpyTidyArray):
        if field_axis is not None and field_axis != data._field_axis:
            raise NotImplementedError("Set the field axis before entering the NumPy prototype")
        return data
    frame = ta.tidy(data, field_axis=field_axis)
    source = frame.to_xarray()
    if any(c not in source.dims for c in source.coords):
        raise NotImplementedError("NumPy prototype currently supports dimension coordinates only")
    variables = {name: _from_array(value).var for name, value in source.data_vars.items()}
    result = core.Dataset(variables, {d: source.coords[d].values for d in source.dims if d in source.coords})
    return NumpyTidyArray(result, attrs=source.attrs, field_axis=frame._field_axis,
                         origins=frame._origins, source=frame._source, notes=frame._notes, groups=frame._groups)


def read_array(*args, **kwargs):
    """Use the existing reader, then enter the NumPy execution path."""
    if kwargs.get("lazy", False):
        raise NotImplementedError("NumPy prototype requires eager input; lazy read_array is not implemented")
    return tidy(ta.read_array(*args, **kwargs))


def _array(value):
    if isinstance(value, core.Labelled):
        return value
    if np.ndim(value):
        raise core.CoreError("a bare array has no axis names; wrap it with labelled(..., dims=...)")
    return core.labelled(value, ())


def _group_labels(labels):
    # Match today's tidy3A output contract at this adapter boundary, including
    # float32 keys becoming float64 coordinates and text coordinates using object.
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([], dtype=float)
    if labels.dtype.kind == "O":
        values = np.asarray([_plain(v) for v in labels])
        return values.astype(object) if values.dtype.kind in "UO" else values
    if labels.dtype.kind in "biufc":
        dtype = np.asarray([_plain(labels.min()), _plain(labels.max())]).dtype
        return labels.astype(dtype, copy=False)
    return labels.astype(object) if labels.dtype.kind in "US" else labels


def _own(data, *, copy):
    variables = {}
    for name, var in data.variables.items():
        variables[name] = (core.Variable(core.readonly(var.data, copy=copy), var.dims, var.attrs)
                           if copy or var.data.flags.writeable else var)
    coords = {d: core.readonly(c, copy=copy) if copy or c.flags.writeable else c for d, c in data.coords.items()}
    return core.Dataset(variables, coords)


def _capture(node, ctx):
    """Snapshot external literals in a grouping expression; keep data references live."""
    if isinstance(node, Formula):
        node = node.lower(ctx)[0]
    if isinstance(node, NameRef):
        value = ctx.resolve(node)
        node = Ref(node.name, node.kind) if ctx.is_data_name(node.name) else Literal(value)
    if isinstance(node, Literal):
        value = ctx.evaluate(node)
        if isinstance(value, core.Labelled):
            data = _own(core.Dataset({"value": value.var}, value.coords), copy=True)
            value = data["value"]
        elif isinstance(value, np.ndarray):
            value = core.readonly(value, copy=True)
        return Literal(value, node.label)
    if isinstance(node, BinOp):
        return BinOp(node.symbol, _capture(node.left, ctx), _capture(node.right, ctx))
    if isinstance(node, UnaryOp):
        return UnaryOp(node.symbol, _capture(node.operand, ctx))
    if isinstance(node, Call):
        return Call(node.name, [_capture(arg, ctx) for arg in node.args], node.options)
    if isinstance(node, Reduce):
        return Reduce(node.op, None if node.arg is None else _capture(node.arg, ctx),
                      node.over, node.na_rm, node.min_count)
    return node


@dataclass(frozen=True)
class GroupCache:
    index: core.GroupIndex
    labels: np.ndarray
    variables: dict
    coords: dict
    sizes: dict

    @property
    def nbytes(self):
        extra_labels = 0 if np.shares_memory(self.labels, self.index.labels) else self.labels.nbytes
        return self.index.nbytes + extra_labels

    def valid(self, data):
        return (all(data.variables.get(n) is var for n, var in self.variables.items())
                and all(data.coords.get(d) is labels for d, labels in self.coords.items())
                and all(data.sizes.get(d) == size for d, size in self.sizes.items()))

    @classmethod
    def build(cls, frame, key):
        ctx = NumpyContext(frame, error=ta.GroupError, grouped=False)
        if key.edges is not None:
            raise NotImplementedError("NumPy prototype does not yet implement bins()")
        value = _array(ctx.evaluate(key.expr))
        if value.ndim != 1 or value.dims[0] == ctx.field_axis:
            raise ta.GroupError("a grouping key must vary along one observation axis")
        if value.dims[0] not in ctx.axes:
            raise ta.GroupError("the grouping key's axis is not an axis of the data")
        value = _align_to(value, frame._data, "group_by")
        index = core.GroupIndex.build(value, key.name)
        variables = {}
        axes = set(value.dims)
        for ref in (node for node in walk(key.expr) if isinstance(node, Ref)):
            variables[ref.name] = ctx.variables.get(ref.name)
            if ref.name in ctx.field_labels() and ref.kind != "var":
                variables.update({name: ctx.variables[name] for name in ctx.holders()})
        for var in variables.values():
            if var is not None:
                axes.update(var.dims)
        return cls(index, core.readonly(_group_labels(index.labels)), variables,
                   {d: frame._data.coords.get(d) for d in axes}, {d: frame._data.sizes[d] for d in axes})

    def reduce(self, value, node, others=()):
        result = self.index.reduce(value, node.op, node.na_rm, node.min_count, over=others)
        return core.Labelled(result.var, {**result.coords, self.index.name: self.labels}, result.notes)


def _result_metadata(value, node):
    if isinstance(node, (Ref, Literal)):
        return value
    return core.Labelled(core.Variable(value.var.data, value.dims), value.coords, value.notes)


def _align_to(value, data, op):
    """Match a result to the collection without allocating observation values."""
    dims = value.dims
    for dim in dims:
        if dim not in data.sizes:
            continue
        target = core.labelled(np.broadcast_to(0, (data.sizes[dim],)), (dim,),
                               {dim: data.coords[dim]} if dim in data.coords else {})
        (_, raw), coords, order, notes = core.align([target, value], op, expand=False)
        value = core.Labelled(core.Variable(raw, order, value.var.attrs), coords, value.notes + tuple(notes))
    return core.transpose(value, *dims)


def _positions(value, axis, index, drop=False):
    number = value.var.axis(axis, "position")
    raw = value.var.data[(slice(None),) * number + (index,)]
    dims = tuple(d for d in value.dims if not (drop and d == axis))
    coords = {d: c[index] if d == axis else c for d, c in value.coords.items() if d in dims}
    return core.Labelled(core.Variable(raw, dims, value.var.attrs), coords, value.notes)


class NumpyContext(Context):
    """Evaluate the existing expression tree; share formula binding with Context."""

    def __init__(self, frame, *, error=ta.TidyError, mode="rows", grouped=True):
        self.frame, self.data = frame, frame._data
        self.field_axis, self.origins = frame._field_axis, frame._origins
        self.keys = frame._groups if grouped else ()
        self.error, self.mode, self.notes = error, mode, []

    @property
    def variables(self):
        return self.data.variables

    @property
    def axes(self):
        return self.data.sizes

    def is_data_name(self, name):
        return name in self.variables or name in self.axes or name in self.field_labels()

    def is_array(self, value):
        return isinstance(value, core.Labelled)

    def coords_for(self, value):
        return value.coords

    def pick_label(self, value, axis, label):
        return core.drop_axis(core.sel(value, **{axis: [label]}), axis)

    def pick_position(self, value, axis, position):
        return _positions(value, axis, position, drop=True)

    def inventory(self):
        return "\n  variables: " + ", ".join(map(str, self.variables))

    def field_labels(self):
        return self.data.coords[self.field_axis] if self.field_axis is not None else ()

    def holders(self):
        return [name for name, value in self.variables.items() if self.field_axis in value.dims]

    def resolve(self, ref):
        name = ref.name
        if isinstance(ref, NameRef) and ref.python is not MISSING:
            if self.is_data_name(name):
                raise self.error(f'"{name}" is both data and a notebook variable')
            return ref.python
        variable, label = name in self.variables, name in self.field_labels()
        if variable and label and ref.kind == "any":
            raise self.error(f'"{name}" is both a variable and a label on {self.field_axis}')
        if variable and ref.kind != "label":
            return self.data[name]
        if label and ref.kind != "var":
            holders = self.holders()
            if len(holders) != 1:
                raise self.error(f'"{name}" is a field label on several variables; select one first')
            return self.pick_label(self.data[holders[0]], self.field_axis, name)
        if name in self.axes and ref.kind == "any":
            coords = {name: self.data.coords[name]} if name in self.data.coords else {}
            values = coords[name] if coords else np.arange(1, self.axes[name] + 1)
            return core.labelled(values, (name,), coords)
        raise self.error(f'"{name}" is not a variable, field label, or axis of this data')

    def vector(self, node):
        # Resolve each reference first to retain collision and unknown-name checks.
        values = [self.resolve(ref) for ref in node.items]
        names = [ref.name for ref in node.items]
        if all(name in self.field_labels() and ref.kind != "var" for name, ref in zip(names, node.items)):
            return core.sel(self.data[self.holders()[0]], **{self.field_axis: names})
        origins = [self.origins.get(name) for name in names]
        if all(origin is not None for origin in origins) and len({origin[0] for origin in origins}) == 1:
            return core.concat(values, origins[0][0], labels=_group_labels(np.asarray(names)))
        raise self.error("a vector needs field labels or variables with a shared origin axis")

    def evaluate(self, node):
        if isinstance(node, Ref):
            return self.resolve(node)
        if isinstance(node, Literal):
            value = _from_array(node.value) if isinstance(node.value, xr.DataArray) else node.value
            if isinstance(value, core.Labelled):
                return _own(core.Dataset({"value": value.var}, value.coords), copy=True)["value"]
            return value.copy() if isinstance(value, np.ndarray) else value
        if isinstance(node, Vector):
            return self.vector(node)
        if isinstance(node, BinOp):
            return core.apply(_OPERATORS[node.symbol], self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, UnaryOp):
            op = {"-": operator.neg, "+": operator.pos, "~": operator.invert}[node.symbol]
            return core.apply(op, self.evaluate(node.operand))
        if isinstance(node, Reduce):
            return self.reduction(node)
        if isinstance(node, Call):
            return self.call(node)
        if isinstance(node, Formula):
            compiled, log = node.lower(self)
            value = self.evaluate(compiled)
            dims = ", ".join(map(str, getattr(value, "dims", ()))) or "a single value"
            self.notes.append("\n  ".join([f"formula: {node.source}", *log,
                                           f"result: ({dims})", f"as array code: {compiled.text()}"]))
            return value
        raise NotImplementedError(f"NumPy prototype cannot evaluate {type(node).__name__}")

    def call(self, node):
        args = [self.evaluate(arg) for arg in node.args]
        if node.name in ("sqrt", "abs", "conj", "exp", "log", "log2", "log10", "sin", "cos", "tan",
                         "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "floor",
                         "ceil", "sign", "real", "imag", "angle"):
            return core.apply(getattr(np, node.name), *args)
        if node.name == "is_in":
            return core.apply(lambda x: np.isin(x, args[1]), args[0])
        if node.name == "if_else":
            return core.apply(np.where, *args)
        if node.name == "rename_axes":
            return core.rename_axes(args[0], **node.options["names"])
        if node.name == "pack":
            labels = [arg.name if isinstance(arg, Ref) else arg.text() for arg in node.args]
            if len(set(map(str, labels))) != len(labels):
                raise self.error("pack(): labels repeat; each packed value needs its own name")
            return core.concat([_array(value) for value in args], node.options["along"], _group_labels(np.asarray(labels)))
        if node.name == "dot":
            left, right = map(_array, args)
            over = node.options.get("over")
            if over is None:
                shared = [d for d in left.dims if d in right.dims]
                if len(shared) != 1:
                    raise self.error("dot(): name the axes to sum with over=")
                over = shared
            return core.dot(left, right, over)
        if node.name == "pick":
            return self.pick_label(args[0], node.options["axis"], node.options["label"])
        if node.name == "position":
            return self.pick_position(args[0], node.options["axis"], node.options["position"] - 1)
        if node.name == "positions":
            axis, lo, hi = (node.options[k] for k in ("axis", "lower", "upper"))
            if not 1 <= lo <= hi <= args[0].sizes[axis]:
                raise self.error(f"formula: bounds {lo} … {hi} are outside {axis}")
            return _positions(args[0], axis, slice(lo - 1, hi))
        raise NotImplementedError(f"NumPy prototype does not yet implement {node.name}()")

    def groups(self):
        if self.frame._group_cache is None:
            self.frame._group_cache = GroupCache.build(self.frame, self.value_key)
        return self.frame._group_cache

    def grouped_result(self, result):
        return result if self.mode == "summarise" else self.groups().index.broadcast(result)

    def count(self):
        shared = set(self.axes)
        for value in self.variables.values():
            if value.dims:
                shared &= set(value.dims)
        reduced = [d for d in self.axes if d in shared and d != self.field_axis and d not in self.group_axes]
        if self.value_key is None:
            return int(np.prod([self.axes[d] for d in reduced], dtype=np.int64))
        groups = self.groups()
        dim = groups.index.axis
        factor = int(np.prod([self.axes[d] for d in reduced if d != dim], dtype=np.int64))
        result = core.labelled(groups.index.counts * factor, (self.value_key.name,),
                               {self.value_key.name: groups.labels})
        return self.grouped_result(result)

    def reduction(self, node):
        if node.op == "n":
            return self.count()
        value = self.evaluate(node.arg)
        if not self.is_array(value):
            raise self.error(f"{node.text()} reduces a constant; reductions need data")
        if any(d not in value.dims for d in self.group_axes):
            raise self.error(f"{node.text()} is missing a grouping axis")
        if node.over is not None:
            axes = list(node.over) if isinstance(node.over, (tuple, list)) else [node.over]
            if self.value_key is not None or any(d in self.group_axes for d in axes):
                raise self.error("over= conflicts with group_by; ungroup() first")
        else:
            axes = [d for d in value.dims if d != self.field_axis and d not in self.group_axes]
        if self.value_key is not None:
            groups = self.groups()
            others = [d for d in axes if d != groups.index.axis]
            return self.grouped_result(groups.reduce(value, node, others))
        return core.reduce(value, node.op, axes, node.na_rm, node.min_count) if axes else value

    def summary_value(self, value):
        result = _array(value)
        if result.dims:
            return result
        dims = list(self.group_axes)
        coords = {d: self.data.coords[d] for d in dims if d in self.data.coords}
        sizes = [self.axes[d] for d in dims]
        if self.value_key is not None:
            labels = self.groups().labels
            dims.append(self.value_key.name)
            coords[self.value_key.name] = labels
            sizes.append(len(labels))
        return core.labelled(np.full(sizes, result.values), dims, coords)


class NumpyTidyArray(TidyArray):
    """Experimental frame holding a core.Dataset between conversion boundaries."""

    __slots__ = ("_attrs", "_group_cache")

    def __init__(self, data, *, attrs=None, _owned=False, _group_cache=None, **metadata):
        super().__init__(_own(data, copy=not _owned), **metadata)
        self._attrs = dict(attrs or {})
        self._group_cache = _group_cache if _group_cache is not None and _group_cache.valid(self._data) else None
        if not _owned:
            ctx = NumpyContext(self, grouped=False)
            self._groups = tuple(ValueKey(k.name, _capture(k.expr, ctx), k.edges) if isinstance(k, ValueKey) else k
                                 for k in self._groups)

    def _evolve(self, data=None, *, notes=(), **changes):
        data = self._data if data is None else data
        resized = [d for d, size in self._data.sizes.items() if data.sizes.get(d) != size]
        kept_notes = [n for n in self._notes if not any(n.startswith(f"inferred: {d} =") for d in resized)]
        kept_notes.extend(n for n in notes if n not in kept_notes)
        field_axis = self._field_axis if self._field_axis in data.sizes else None
        return NumpyTidyArray(data, attrs=self._attrs, field_axis=field_axis, source=self._source, _owned=True,
                             _group_cache=None if "groups" in changes else self._group_cache,
                             origins=changes.get("origins", {n: o for n, o in self._origins.items() if n in data.variables}),
                             notes=kept_notes, groups=changes.get("groups", self._groups))

    def to_xarray(self):
        return xr.Dataset({n: (v.dims, v.data, dict(v.attrs)) for n, v in self._data.variables.items()},
                          coords=dict(self._data.coords), attrs=dict(self._attrs))

    def collect(self):
        return self.to_xarray()

    def __repr__(self):
        lines = ["<NumpyTidyArray prototype>"]
        lines += [f"{n}: {v!r}" for n, v in self._data.variables.items()]
        return "\n".join([*lines, *self._notes])

    def _apply_verb(self, verb):
        handlers = {"select": self._select, "mutate": self._mutate, "filter": self._filter,
                    "group_by": self._group_by, "summarise": self._summarise,
                    "pull": self._pull, "ungroup": lambda _: self._evolve(groups=()),
                    "slice": self._slice,
                    "slice_head": lambda axes: self._slice_end(axes, True),
                    "slice_tail": lambda axes: self._slice_end(axes, False)}
        if verb._operation not in handlers:
            raise NotImplementedError(f"NumPy prototype does not yet implement {verb.text}")
        try:
            return handlers[verb._operation](verb._arguments)
        except core.CoreError as exc:
            errors = {"select": ta.SelectError, "mutate": ta.MutateError, "filter": ta.FilterError,
                      "group_by": ta.GroupError, "summarise": ta.SummariseError, "pull": ta.RefError,
                      "slice": ta.SliceError, "slice_head": ta.SliceError, "slice_tail": ta.SliceError}
            raise errors.get(verb._operation, ta.TidyError)(str(exc)) from None

    def _pull(self, item):
        if not isinstance(item, (str, Ref)):
            raise TypeError("pull() takes one name")
        ctx = NumpyContext(self, error=ta.RefError, grouped=False)
        return _array(ctx.resolve(Ref(item) if isinstance(item, str) else item)).values

    def _select(self, items):
        ctx = NumpyContext(self, error=ta.SelectError, grouped=False)
        refs = [Ref(item) if isinstance(item, str) else item for item in _flatten(items)]
        for key in self._groups:
            if isinstance(key, ValueKey) and isinstance(key.expr, Ref) and not any(
                    isinstance(ref, Ref) and ref.name == key.expr.name for ref in refs):
                refs.append(key.expr)
        names, labels = [], []
        for ref in refs:
            if not isinstance(ref, Ref):
                raise NotImplementedError("NumPy prototype select() currently takes explicit names only")
            ctx.resolve(ref)
            if ref.name in ctx.variables and ref.kind != "label":
                if ref.name not in names:
                    names.append(ref.name)
            elif ref.name in ctx.field_labels() and ref.kind != "var":
                if ref.name not in labels:
                    labels.append(ref.name)
            else:
                raise ta.SelectError("select keeps columns; use filter for an observation axis")
        if labels:
            names = list(dict.fromkeys([*ctx.holders(), *names]))
        variables, coords = {}, {}
        for name in names:
            value = self._data[name]
            if labels and self._field_axis in value.dims:
                value = core.sel(value, **{self._field_axis: labels})
            variables[name] = value.var
            coords.update(value.coords)
        return self._evolve(core.Dataset(variables, coords))

    def _mutate(self, steps):
        frame = self
        for step in steps:
            ctx = NumpyContext(frame, error=ta.MutateError)
            node = as_expr(step.value)
            value = _result_metadata(_array(ctx.evaluate(node)), node)
            if isinstance(step.target, Vector):
                names = [ref.name for ref in step.target.items]
                axes = [d for d, labels in value.coords.items() if list(labels) == names]
                if len(axes) != 1:
                    raise ta.MutateError("a vector assignment needs one axis labelled with the target names in order")
                for name in names:
                    frame = frame._store(name, ctx.pick_label(value, axes[0], name), ctx.notes)
            else:
                frame = frame._store(step.target, value, ctx.notes)
        return frame

    def _store(self, name, value, notes):
        ctx = NumpyContext(self, error=ta.MutateError)
        if name in ctx.axes:
            raise ta.MutateError(f'"{name}" is an axis name; choose another name for the variable')
        if name in ctx.variables and name in ctx.field_labels():
            raise ta.MutateError(f'"{name}" is both a variable and a field label')
        value = _align_to(value, self._data, "mutate")
        if name in ctx.field_labels():
            if len(ctx.holders()) != 1:
                raise ta.MutateError("field write-back needs one holder; select one first")
            holder = ctx.holders()[0]
            target = self._data[holder]
            dims = tuple(d for d in target.dims if d != self._field_axis)
            if value.ndim and set(value.dims) != set(dims):
                raise ta.MutateError("a written field must keep its shape")
            raw = core.transpose(value, *dims).values if value.ndim else value.values.item()
            dtype = np.result_type(target.dtype, value.dtype if value.ndim else raw)
            data = target.values.astype(dtype, copy=True)
            axis = target.var.axis(self._field_axis, "mutate")
            position = list(ctx.field_labels()).index(name)
            data[(slice(None),) * axis + (position,)] = raw
            updated = core.Labelled(core.Variable(data, target.dims), target.coords)
            if dtype != target.dtype:
                notes = [*notes, f"changed: {holder} promoted {target.dtype} → {dtype} by mutate({name} = …)"]
            result = self._data.assign(**{holder: updated})
        else:
            result = self._data.assign(**{name: value})
        origins = {n: o for n, o in self._origins.items() if n != name}
        return self._evolve(result, notes=[*notes, *value.notes], origins=origins)

    def _slice(self, arguments):
        drop, axes = arguments
        if self._groups:
            raise ta.SliceError("slice() on grouped data is ambiguous; ungroup() first")
        data = self._data
        for axis, spec in axes.items():
            index = _slice_positions(data, axis, spec, self)
            data = data.isel(**{axis: _position_indexer(index)})
            if drop:
                if data.sizes[axis] != 1:
                    raise ta.SliceError(f"drop=True removes {axis} only when one position remains")
                variables = {n: core.drop_axis(data[n], axis).var if axis in v.dims else v
                             for n, v in data.variables.items()}
                data = core.Dataset(variables, {d: c for d, c in data.coords.items() if d != axis})
        return self._evolve(data)

    def _slice_end(self, axes, head):
        positions = {}
        for axis, count in axes.items():
            if axis not in self._data.sizes:
                raise ta.SliceError(f"{axis} is not an axis of this data")
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ta.SliceError("slice_head/tail needs a non-negative count")
            size = self._data.sizes[axis]
            count = min(int(count), size)
            positions[axis] = range(1, count + 1) if head else range(size - count + 1, size + 1)
        return self._slice((False, positions))

    def _filter(self, predicates):
        ctx = NumpyContext(self, error=ta.FilterError)
        masks = {}
        for predicate in predicates:
            for part in _conjuncts(as_expr(predicate)):
                mask = _array(ctx.evaluate(part))
                if mask.ndim != 1 or mask.dtype != bool:
                    raise ta.FilterError("filter needs a true/false condition along one observation axis")
                axis = mask.dims[0]
                if axis == self._field_axis or axis not in self._data.sizes:
                    raise ta.FilterError(f"filter cannot remove positions on {axis!r}; it is not an observation axis")
                mask = _align_to(mask, self._data, "filter")
                masks[axis] = core.apply(np.logical_and, masks[axis], mask) if axis in masks else mask
        data = self._data.isel(**{d: mask.values for d, mask in masks.items()})
        notes = [*ctx.notes, *(n for mask in masks.values() for n in mask.notes)]
        return self._evolve(data, notes=notes)

    def _group_by(self, arguments):
        keys, named = arguments
        resolved = []
        for key in keys:
            if isinstance(key, ValueKey):
                resolved.append(key)
                continue
            if not isinstance(key, (str, Ref)):
                raise ta.GroupError("group_by() takes names or name = expression")
            name = key.name if isinstance(key, Ref) else key
            if name == self._field_axis:
                raise ta.GroupError("the field axis holds columns, not groups")
            resolved.append(AxisKey(name) if name in self._data.sizes else ValueKey(name, Ref(name)))
        resolved += [ValueKey(name, as_expr(expr)) for name, expr in named.items()]
        if sum(isinstance(key, ValueKey) for key in resolved) > 1:
            raise ta.GroupError("grouping by more than one value field is not supported yet")
        ungrouped = NumpyContext(self, error=ta.GroupError, grouped=False)
        resolved = [ValueKey(k.name, _capture(k.expr, ungrouped), k.edges) if isinstance(k, ValueKey) else k
                    for k in resolved]
        result = self._evolve(groups=resolved)
        ctx = NumpyContext(result, error=ta.GroupError)
        if ctx.value_key is not None:
            ctx.groups()
        return result

    def _summarise(self, arguments):
        groups, named = arguments
        ctx = NumpyContext(self, error=ta.SummariseError, mode="summarise")
        allowed = set(ctx.group_axes)
        if self._field_axis is not None:
            allowed.add(self._field_axis)
        if ctx.value_key is not None:
            allowed.add(ctx.value_key.name)
        variables, coords = {}, {}
        for name, expr in named.items():
            node = as_expr(expr)
            value = _result_metadata(ctx.summary_value(ctx.evaluate(node)), node)
            if any(d not in allowed for d in value.dims):
                raise ta.SummariseError(f"{name} still varies over an ungrouped observation axis")
            result = core.Dataset(variables, coords).assign(**{name: value})
            variables, coords = dict(result.variables), dict(result.coords)
        mode = groups or "drop_last"
        keys = () if mode == "drop" else self._groups if mode == "keep" else self._groups[:-1]
        kept = [AxisKey(key.name) if isinstance(key, ValueKey) else key for key in keys]
        return self._evolve(core.Dataset(variables, coords), notes=ctx.notes, groups=kept, origins={})
