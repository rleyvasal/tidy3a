"""Expressions: references to data, operators, reductions, and their evaluation.

An expression is built eagerly from Python operators (``x > 10``,
``mean(intensity)``) and evaluated later against a ``TidyArray`` by a verb.
Names resolve by plan §5.1–5.2: a variable, then a field-axis label, then an
observation axis's labels. Arithmetic between arrays follows the alignment
rules in ``tidy3a.align``.
"""

from __future__ import annotations

import builtins
import operator
import sys
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from tidy3a.align import align_pair
from tidy3a.errors import GroupError, RefError, TidyError

_EXACT = {"arithmetic_join": "exact"}


# ── Nodes ───────────────────────────────────────────────────────────────────


class Expr:
    """Base class for tidy3A expressions; combine them with Python operators."""

    __hash__ = None  # type: ignore[assignment]  # == builds an expression
    # NumPy values on the left defer to the expression's reflected operators.
    # xarray objects do not defer, so write the expression first: x * R, not R * x.
    __array_ufunc__ = None

    def text(self) -> str:
        raise NotImplementedError

    def children(self) -> Sequence[Expr]:
        return ()

    def __repr__(self) -> str:
        return f"<expr {self.text()}>"

    def __bool__(self) -> bool:
        raise TypeError(
            f"{self.text()} is an expression, not a value; "
            "combine conditions with & | ~ instead of and / or / not"
        )

    def is_in(self, values: Any) -> Expr:
        """True where the value is one of *values*."""
        return Call("is_in", (self, Literal(list(values))))

    def __abs__(self) -> Expr:
        return Call("abs", (self,))


def _binary(symbol: str) -> tuple[Callable, Callable]:
    def forward(self, other):
        return BinOp(symbol, self, as_expr(other))

    def backward(self, other):
        return BinOp(symbol, as_expr(other), self)

    return forward, backward


for _symbol, _name in [
    ("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "truediv"), ("//", "floordiv"),
    ("%", "mod"), ("**", "pow"), ("&", "and"), ("|", "or"), ("^", "xor"),
]:
    _forward, _backward = _binary(_symbol)
    setattr(Expr, f"__{_name}__", _forward)
    setattr(Expr, f"__r{_name}__", _backward)

for _symbol, _name in [("==", "eq"), ("!=", "ne"), ("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge")]:
    setattr(Expr, f"__{_name}__", _binary(_symbol)[0])

Expr.__neg__ = lambda self: UnaryOp("-", self)  # type: ignore[assignment]
Expr.__pos__ = lambda self: UnaryOp("+", self)  # type: ignore[assignment]
Expr.__invert__ = lambda self: UnaryOp("~", self)  # type: ignore[assignment]


class Ref(Expr):
    """A name to resolve against the data: ``kind`` is "any", "var", or "label"."""

    def __init__(self, name: Hashable, kind: str = "any"):
        self.name = name
        self.kind = kind

    def text(self) -> str:
        if self.kind == "any":
            return str(self.name)
        helper = "variable" if self.kind == "var" else self.kind
        return f'{helper}("{self.name}")'


class Literal(Expr):
    def __init__(self, value: Any, label: str | None = None):
        self.value = value
        self.label = label

    def text(self) -> str:
        if self.label is not None:
            return self.label
        value = self.value
        if isinstance(value, xr.DataArray):
            return f"<array {value.name}>" if value.name is not None else "<array>"
        if isinstance(value, np.ndarray) and value.ndim:
            return "<numpy array>"
        if isinstance(value, float) and np.isnan(value):
            return "nan"
        return repr(value)


class Vector(Expr):
    """``[x, y, z]``: field labels kept on their axis, or variables re-packed."""

    def __init__(self, items: Sequence[Ref]):
        self.items = tuple(items)

    def children(self) -> Sequence[Expr]:
        return self.items

    def text(self) -> str:
        return "[" + ", ".join(item.text() for item in self.items) + "]"


class BinOp(Expr):
    def __init__(self, symbol: str, left: Expr, right: Expr):
        self.symbol = symbol
        self.left = left
        self.right = right

    def children(self) -> Sequence[Expr]:
        return (self.left, self.right)

    def text(self) -> str:
        return f"{_wrap(self.left)} {self.symbol} {_wrap(self.right)}"


class UnaryOp(Expr):
    def __init__(self, symbol: str, operand: Expr):
        self.symbol = symbol
        self.operand = operand

    def children(self) -> Sequence[Expr]:
        return (self.operand,)

    def text(self) -> str:
        return f"{self.symbol}{_wrap(self.operand)}"


class Call(Expr):
    """A named function from ``FUNCTIONS``; ``options`` are keyword settings."""

    def __init__(self, name: str, args: Sequence[Expr], options: dict[str, Any] | None = None):
        self.name = name
        self.args = tuple(args)
        self.options = dict(options or {})

    def children(self) -> Sequence[Expr]:
        return self.args

    def text(self) -> str:
        parts = [arg.text() for arg in self.args]
        parts += [f"{key}={value!r}" for key, value in self.options.items() if value is not None]
        return f"{self.name}({', '.join(parts)})"


class Reduce(Expr):
    def __init__(
        self, op: str, arg: Expr | None, over: Any = None, na_rm: bool = False,
        min_count: int | None = None,
    ):
        self.op = op
        self.arg = arg
        self.over = over
        self.na_rm = na_rm
        self.min_count = min_count

    def children(self) -> Sequence[Expr]:
        return () if self.arg is None else (self.arg,)

    def text(self) -> str:
        parts = [] if self.arg is None else [self.arg.text()]
        if self.over is not None:
            over = f'"{self.over}"' if isinstance(self.over, str) else repr(self.over)
            parts.append(f"over={over}")
        if self.na_rm:
            parts.append("na_rm=True")
        if self.min_count is not None:
            parts.append(f"min_count={self.min_count}")
        return f"{self.op}({', '.join(parts)})"


def label_index(labels: Sequence[Any], name: Hashable) -> pd.Index:
    """A pandas index for new labels; text stays object dtype, which older xarray needs."""
    values = np.asarray(labels)
    if values.dtype.kind in "UO":
        values = values.astype(object)
    return pd.Index(values, name=name, dtype=values.dtype)


def _wrap(node: Expr) -> str:
    return f"({node.text()})" if isinstance(node, BinOp) else node.text()


def as_expr(value: Any) -> Expr:
    """Turn a Python value into an expression; ``[x, y, z]`` of references is a vector."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        refs = [item for item in items if isinstance(item, Ref)]
        exprs = [item for item in items if isinstance(item, Expr)]
        if items and len(refs) == len(items):
            return Vector(refs)
        if exprs:
            raise TidyError(
                f"[{', '.join(_item_text(item) for item in items)}] mixes data references "
                "and plain values; a vector holds only references, e.g. [x, y, z]"
            )
        return Literal(np.asarray(items))
    return Literal(value)


def _item_text(item: Any) -> str:
    return item.text() if isinstance(item, Expr) else repr(item)


def walk(node: Expr):
    yield node
    for child in node.children():
        yield from walk(child)


# ── Public constructors ─────────────────────────────────────────────────────


def col(name: Hashable) -> Ref:
    """A reference to a variable, field label, or axis by name, as in scripts."""
    return Ref(name)


def variable(name: Hashable) -> Ref:
    """The variable *name*, when a field label has the same name."""
    return Ref(name, "var")


def label(name: Hashable) -> Ref:
    """The field-axis label *name*, when a variable has the same name."""
    return Ref(name, "label")


def fields(*names: Hashable) -> Ref | tuple[Ref, ...]:
    """References for scripts: ``x, y, z = fields("x", "y", "z")``."""
    refs = tuple(Ref(name) for name in names)
    return refs[0] if len(refs) == 1 else refs


def env(name: str) -> Literal:
    """The Python value *name* from the calling code, when data has the same name."""
    frame = sys._getframe(1)
    for scope in (frame.f_locals, frame.f_globals, vars(builtins)):
        if name in scope:
            return Literal(scope[name], label=f'env("{name}")')
    raise RefError(f'env("{name}"): no Python value named {name} is defined here')


def if_else(condition: Any, true: Any, false: Any) -> Call:
    """*true* where *condition* holds, else *false*; tidy3A's masking function."""
    return Call("if_else", (as_expr(condition), as_expr(true), as_expr(false)))


# ── Reductions ──────────────────────────────────────────────────────────────


def _reduction(op: str, builtin: Callable | None = None) -> Callable:
    def reduce(x: Any = None, *args: Any, over: Any = None, na_rm: bool = False, **kwargs: Any) -> Any:
        if builtin is not None and not _is_data(x):
            return builtin(x, *args, **kwargs)
        if args or kwargs.keys() - {"min_count"}:
            raise TypeError(f"{op}() takes one expression plus over=, na_rm=" + (
                ", min_count=" if op == "sum" else ""))
        if x is None:
            raise TypeError(f"{op}() needs an expression, e.g. {op}(intensity)")
        if not isinstance(na_rm, bool):
            raise TypeError(f"{op}() na_rm must be True or False")
        return Reduce(op, as_expr(x), over, na_rm, kwargs.get("min_count"))

    reduce.__name__ = op
    reduce.__qualname__ = op
    reduce.__doc__ = (
        f"{op.capitalize()} over observation axes, per group when grouped. "
        "``over=`` names the axes instead; missing values propagate unless ``na_rm=True``."
    )
    return reduce


def _is_data(value: Any) -> bool:
    if isinstance(value, (Expr, xr.DataArray)):
        return True
    if isinstance(value, (list, tuple)):
        return builtins.any(isinstance(item, Expr) for item in value)
    return False


mean = _reduction("mean")
median = _reduction("median")
std = _reduction("std")
# Variance, as tidy3's and dplyr's var(); the variable reference is variable("x").
var = _reduction("var")
sum = _reduction("sum", builtins.sum)  # noqa: A001
min = _reduction("min", builtins.min)  # noqa: A001
max = _reduction("max", builtins.max)  # noqa: A001
any = _reduction("any", builtins.any)  # noqa: A001
all = _reduction("all", builtins.all)  # noqa: A001


def n() -> Reduce:
    """The number of observations, per group when grouped."""
    return Reduce("n", None)


# ── Grouping keys ───────────────────────────────────────────────────────────


class AxisKey:
    """``group_by(sensor)``: keep this axis, reduce the others."""

    def __init__(self, axis: Hashable):
        self.axis = axis
        self.label = str(axis)


class ValueKey:
    """``group_by(ring)`` or ``group_by(bins(distance, edges))``: group positions by value."""

    def __init__(self, name: Hashable, expr: Expr, edges: Any = None):
        self.name = name
        self.expr = expr
        self.edges = edges
        self.label = str(name) if edges is None else f"bins({expr.text()})"


class Groups:
    """A value key resolved against current data: one code per position on ``axis``."""

    def __init__(self, name: Hashable, axis: Hashable, codes: xr.DataArray, labels: Sequence[Any]):
        self.name = name
        self.axis = axis
        self.codes = codes
        self.labels = list(labels)


# ── Evaluation ──────────────────────────────────────────────────────────────


class Context:
    """What an expression is evaluated against, and how reductions behave."""

    def __init__(self, frame: Any, *, verb: str, error: type[TidyError] = TidyError,
                 mode: str = "rows", grouped: bool = True):
        self.frame = frame
        self.data: xr.Dataset = frame._data
        field_axis = frame._field_axis
        self.field_axis = field_axis if field_axis in self.data.dims else None
        self.origins = frame._origins
        self.verb = verb
        self.error = error
        self.mode = mode
        self.keys = frame._groups if grouped else ()
        self.notes: list[str] = []
        self._groups: Groups | None = None

    @property
    def group_axes(self) -> list[Hashable]:
        return [key.axis for key in self.keys if isinstance(key, AxisKey)]

    @property
    def value_key(self) -> ValueKey | None:
        return next((key for key in self.keys if isinstance(key, ValueKey)), None)

    def groups(self) -> Groups:
        if self._groups is None:
            key = self.value_key
            assert key is not None
            self._groups = resolve_value_key(key, self.frame, self.group_axes)
        return self._groups

    def observation_axes(self, array: xr.DataArray) -> list[Hashable]:
        return [dim for dim in array.dims if dim != self.field_axis]


def evaluate(node: Expr, ctx: Context) -> Any:
    """Evaluate *node*; arrays come back as ``xr.DataArray``, constants as Python values."""
    if isinstance(node, Ref):
        return resolve(node, ctx)
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Vector):
        return resolve_vector(node, ctx)
    if isinstance(node, BinOp):
        return _evaluate_binary(node, ctx)
    if isinstance(node, UnaryOp):
        value = evaluate(node.operand, ctx)
        op = {"-": operator.neg, "+": operator.pos, "~": operator.invert}[node.symbol]
        return op(value)
    if isinstance(node, Call):
        function = FUNCTIONS.get(node.name)
        if function is None:
            raise TidyError(f"{node.name}() is not a tidy3A function")
        return function(node, ctx)
    if isinstance(node, Reduce):
        return _evaluate_reduce(node, ctx)
    lower = getattr(node, "lower", None)
    if lower is not None:  # a formula: compile against the data, evaluate, and show how it was read
        compiled, log = lower(ctx)
        value = evaluate(compiled, ctx)
        dims = ", ".join(map(str, getattr(value, "dims", ()))) or "a single value"
        lines = [f"formula: {node.source}", *log, f"result: ({dims})", f"as array code: {compiled.text()}"]
        ctx.notes.append("\n  ".join(lines))
        return value
    raise TypeError(f"cannot evaluate {type(node).__name__}")


_OPERATORS: dict[str, Callable] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "//": operator.floordiv, "%": operator.mod, "**": operator.pow, "&": operator.and_,
    "|": operator.or_, "^": operator.xor, "==": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


def _evaluate_binary(node: BinOp, ctx: Context) -> Any:
    left = evaluate(node.left, ctx)
    right = evaluate(node.right, ctx)
    return combine(_OPERATORS[node.symbol], left, right, node.left.text(), node.right.text(), ctx)


def combine(op: Callable, left: Any, right: Any, left_text: str, right_text: str, ctx: Context) -> Any:
    """Apply *op* under tidy3A's alignment rules."""
    for value, text in ((left, left_text), (right, right_text)):
        if isinstance(value, np.ndarray) and value.ndim:
            raise TidyError(
                f"{text} is a NumPy array without axis names; pairing it by position could "
                'misalign it. Name its axes: xr.DataArray(values, dims=["point"])'
            )
    if isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray):
        left, right = align_pair(left, right, left_text, right_text, ctx.notes)
    with xr.set_options(**_EXACT):
        return op(left, right)


def resolve(ref: Ref, ctx: Context) -> xr.DataArray:
    """Resolve a name: variable, then field label, then axis labels (plan §5.1–5.2)."""
    data = ctx.data
    name = ref.name
    field_axis = ctx.field_axis
    is_variable = name in data.data_vars
    holders = []
    if field_axis is not None and name in data.indexes[field_axis]:
        holders = [v for v in data.data_vars if field_axis in data[v].dims]
    is_label = bool(holders)

    if ref.kind == "var":
        if not is_variable:
            raise ctx.error(f'variable("{name}"): there is no variable {name}. {_inventory(ctx)}')
        return data[name]
    if ref.kind == "label":
        if not is_label:
            raise ctx.error(f'label("{name}"): there is no field label {name}. {_inventory(ctx)}')
        return _field_view(data, holders, name, field_axis, ctx)
    if is_variable and is_label:
        raise ctx.error(
            f'"{name}" is both a variable and a label on {field_axis}.\n'
            f'  variable:  variable("{name}")\n'
            f'  label:     label("{name}")'
        )
    if is_variable:
        return data[name]
    if is_label:
        return _field_view(data, holders, name, field_axis, ctx)
    if name in data.dims:
        return axis_labels(data, name)
    raise ctx.error(f'"{name}" is not a variable, field label, or axis of this data. {_inventory(ctx)}'
                    + _observation_hint(data, name, field_axis))


def _field_view(data: xr.Dataset, holders: list, name: Hashable, field_axis: Hashable, ctx: Context):
    if len(holders) > 1:
        listed = ", ".join(str(v) for v in holders)
        raise ctx.error(
            f'"{name}" is a field label on {len(holders)} variables ({listed}); '
            f"select one first, e.g. select({holders[0]})"
        )
    return data[holders[0]].sel({field_axis: name}, drop=True).rename(name)


def axis_labels(data: xr.Dataset, axis: Hashable) -> xr.DataArray:
    """An axis's labels as data along it; unlabelled axes give 1-based positions."""
    if axis in data.indexes:
        return data[axis].reset_coords(drop=True)
    return xr.DataArray(np.arange(1, data.sizes[axis] + 1), dims=axis, name=axis)


def _inventory(ctx: Context) -> str:
    data = ctx.data
    parts = ["variables: " + (", ".join(str(v) for v in data.data_vars) or "none")]
    if ctx.field_axis is not None:
        labels = "  ".join(str(v) for v in data.indexes[ctx.field_axis])
        parts.append(f"labels on {ctx.field_axis}: {labels}")
    return "\n  " + "\n  ".join(parts)


def _observation_hint(data: xr.Dataset, name: Hashable, field_axis: Hashable) -> str:
    for dim, index in data.indexes.items():
        if dim != field_axis and name in index:
            return f'\n  "{name}" is a value on the {dim} axis; to keep it use filter({dim} == {name!r})'
    return ""


def resolve_vector(node: Vector, ctx: Context) -> xr.DataArray:
    """``[x, y, z]``: field labels on one variable, or variables from one recorded axis."""
    data = ctx.data
    names = [item.name for item in node.items]
    field_axis = ctx.field_axis
    kinds = []
    for item in node.items:
        value_is_var = item.name in data.data_vars and item.kind != "label"
        value_is_label = (
            field_axis is not None and item.name in data.indexes[field_axis] and item.kind != "var"
        )
        if value_is_var and value_is_label:
            resolve(item, ctx)  # raises the collision error
        kinds.append("var" if value_is_var else "label" if value_is_label else None)
    if None in kinds:
        resolve(node.items[kinds.index(None)], ctx)  # raises the unknown-name error
    if builtins.all(kind == "label" for kind in kinds):
        holders = [v for v in data.data_vars if field_axis in data[v].dims]
        if len(holders) > 1:
            raise ctx.error(
                f"{node.text()} are field labels on several variables "
                f"({', '.join(str(v) for v in holders)}); select one first"
            )
        return data[holders[0]].sel({field_axis: names})
    if builtins.all(kind == "var" for kind in kinds):
        origins = [ctx.origins.get(name) for name in names]
        axes = {origin[0] for origin in origins if origin is not None}
        if None not in origins and len(axes) == 1:
            axis = axes.pop()
            return xr.concat([data[name] for name in names], dim=label_index(names, axis))
        raise ctx.error(
            f"{node.text()} are variables without a shared origin axis; "
            f'pack them explicitly: pack({node.text()}, along="…")'
        )
    raise ctx.error(f"{node.text()} mixes variables and field labels; a vector holds one kind")


# ── Reductions ──────────────────────────────────────────────────────────────

_FLOAT_REDUCERS = {"mean", "sum", "min", "max", "median", "std", "var"}


def reduce_array(
    array: xr.DataArray, op: str, axes: Sequence[Hashable], na_rm: bool = False,
    min_count: int | None = None,
) -> xr.DataArray:
    """One reduction over *axes*, with missing values propagated unless *na_rm*."""
    axes = list(axes)
    if not axes:
        return array
    if op in ("min", "max") and builtins.any(array.sizes[axis] == 0 for axis in axes):
        # NumPy cannot take min or max of nothing; the result is missing.
        return xr.full_like(array.astype(float).sum(dim=axes), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if op in ("any", "all"):
            return getattr(array, op)(dim=axes)
        if op == "count":
            return array.notnull().sum(dim=axes)
        options = {"skipna": na_rm} if array.dtype.kind in "fcmM" else {}
        result = getattr(array, op)(dim=axes, **options)
        if op == "sum" and min_count is not None:
            present = array.notnull().sum(dim=axes)
            result = result.where(present >= min_count)
        return result


def _evaluate_reduce(node: Reduce, ctx: Context) -> Any:
    error = ctx.error
    if node.op == "n":
        return _count(ctx)
    array = evaluate(node.arg, ctx)
    if not isinstance(array, xr.DataArray):
        raise error(f"{node.text()} reduces a constant; reductions need data, e.g. {node.op}(intensity)")
    if ctx.keys:
        _check_has_group_axes(node, array, ctx)
    if node.over is not None:
        axes = list(node.over) if isinstance(node.over, (list, tuple)) else [node.over]
        for axis in axes:
            if axis not in array.dims:
                raise error(f"{node.text()}: {axis} is not an axis of {node.arg.text()} (axes: "
                            f"{', '.join(str(d) for d in array.dims)})")
        if ctx.value_key is not None:
            raise error(
                f"{node.text()}: over= cannot be combined with grouping by {ctx.value_key.label}; "
                "ungroup() first, or drop over="
            )
        clash = [axis for axis in axes if axis in ctx.group_axes]
        if clash:
            raise error(f"{node.text()}: over= includes {clash[0]}, which group_by keeps")
        return reduce_array(array, node.op, axes, node.na_rm, node.min_count)

    axes = [axis for axis in ctx.observation_axes(array) if axis not in ctx.group_axes]
    if ctx.value_key is None:
        return reduce_array(array, node.op, axes, node.na_rm, node.min_count)
    groups = ctx.groups()
    others = [axis for axis in axes if axis != groups.axis]
    stats = _group_reduce(array, node.op, groups, others, node.na_rm, node.min_count)
    return stats if ctx.mode == "summarise" else _broadcast_back(stats, groups)


def _check_has_group_axes(node: Reduce, array: xr.DataArray, ctx: Context) -> None:
    needed = list(ctx.group_axes)
    if ctx.value_key is not None:
        needed.append(ctx.groups().axis)
    for axis in needed:
        if axis not in array.dims:
            refs = [r for r in walk(node) if isinstance(r, Ref)]
            subject = refs[0].text() if refs else node.arg.text()
            key = next((k.label for k in ctx.keys if isinstance(k, AxisKey) and k.axis == axis),
                       ctx.value_key.label if ctx.value_key else str(axis))
            raise ctx.error(
                f"{subject} has no {axis} axis, so it cannot be summarised by group_by({key}).\n"
                f"  Its summary would be the same for every group.\n"
                f"  Summarise it separately without grouping: summarise(… = {node.text()})"
            )


def _group_reduce(array, op, groups: Groups, others, na_rm, min_count) -> xr.DataArray:
    codes = groups.codes.values
    stats = []
    for code in range(len(groups.labels)):
        part = array.isel({groups.axis: codes == code})
        stats.append(reduce_array(part, op, [groups.axis, *others], na_rm, min_count))
    index = label_index(groups.labels, groups.name)
    if stats:
        return xr.concat(stats, dim=index)
    empty = reduce_array(array.isel({groups.axis: slice(0, 0)}), op, [groups.axis, *others], na_rm, min_count)
    return empty.expand_dims({groups.name: 0}).assign_coords({groups.name: index})


def _broadcast_back(stats: xr.DataArray, groups: Groups) -> xr.DataArray:
    picked = stats.isel({groups.name: groups.codes})
    return picked.drop_vars(groups.name, errors="ignore")


def _count(ctx: Context) -> Any:
    data = ctx.data
    observation = [d for d in data.dims if d != ctx.field_axis and d not in ctx.group_axes]
    if ctx.value_key is None:
        return int(np.prod([data.sizes[d] for d in observation], dtype=np.int64))
    groups = ctx.groups()
    per_position = int(np.prod([data.sizes[d] for d in observation if d != groups.axis], dtype=np.int64))
    counts = np.bincount(groups.codes.values, minlength=len(groups.labels)) * per_position
    stats = xr.DataArray(counts, dims=groups.name, coords={groups.name: groups.labels})
    return stats if ctx.mode == "summarise" else _broadcast_back(stats, groups)


# ── Value keys ──────────────────────────────────────────────────────────────


def resolve_value_key(key: ValueKey, frame: Any, group_axes: Sequence[Hashable]) -> Groups:
    """Evaluate a value key on current data and give each position a group code."""
    ctx = Context(frame, verb="group_by", error=GroupError, grouped=False)
    values = evaluate(key.expr, ctx)
    if not isinstance(values, xr.DataArray) or values.ndim != 1:
        dims = ", ".join(str(d) for d in getattr(values, "dims", ())) or "no axes"
        raise GroupError(
            f"group_by({key.label}) needs a field that varies along one axis; "
            f"{key.expr.text()} varies along {dims}"
        )
    axis = values.dims[0]
    if axis in group_axes:
        raise GroupError(f"group_by({key.label}) groups the {axis} axis, which group_by also keeps")
    if axis == ctx.field_axis:
        raise GroupError(f"group_by({key.label}) would group the field axis {axis}; fields are columns")
    if key.edges is None:
        return _value_groups(key, axis, values)
    return _bin_groups(key, axis, values)


def _value_groups(key: ValueKey, axis: Hashable, values: xr.DataArray) -> Groups:
    raw = values.values
    if raw.dtype.kind in "fc" and np.isnan(raw).any():
        raise GroupError(
            f"group_by({key.label}): {int(np.isnan(raw).sum()):,} positions have a missing "
            f"{key.label}; filter them out first"
        )
    labels, codes = np.unique(raw, return_inverse=True)
    codes_da = xr.DataArray(codes.reshape(-1), dims=axis, coords=values.coords)
    return Groups(key.name, axis, codes_da, [_plain(label) for label in labels])


def _bin_groups(key: ValueKey, axis: Hashable, values: xr.DataArray) -> Groups:
    raw = values.values
    edges = _bin_edges(key, raw)
    codes = np.searchsorted(edges, raw, side="right") - 1
    outside = (codes < 0) | (codes >= len(edges) - 1)
    if raw.dtype.kind in "fc":
        outside |= np.isnan(raw)
    if outside.any():
        low, high = _format_edge(edges[0]), _format_edge(edges[-1])
        raise GroupError(
            f"{int(outside.sum()):,} positions of {key.expr.text()} fall outside every bin "
            f"[{low}, {high}); they are never dropped silently.\n"
            "  Widen the edges, or filter those positions out first."
        )
    labels = [f"[{_format_edge(a)}, {_format_edge(b)})" for a, b in zip(edges[:-1], edges[1:])]
    codes_da = xr.DataArray(codes, dims=axis, coords=values.coords)
    return Groups(key.name, axis, codes_da, labels)


def _bin_edges(key: ValueKey, raw: np.ndarray) -> np.ndarray:
    edges = key.edges
    if isinstance(edges, str):
        if raw.dtype.kind != "M":
            raise GroupError(f'bins({key.expr.text()}, "{edges}"): a duration needs time values')
        step = pd.Timedelta(edges).to_timedelta64()
        if raw.size == 0:
            raise GroupError(f'bins({key.expr.text()}, "{edges}"): there are no times to bin')
        start = raw.min()
        count = int((raw.max() - start) // step) + 1
        return start + np.arange(count + 1) * step
    edges = np.asarray(edges)
    if edges.ndim != 1 or len(edges) < 2 or not (edges[1:] > edges[:-1]).all():
        raise GroupError(f"bins({key.expr.text()}, …) needs at least two increasing edges")
    return edges


def _format_edge(value: Any) -> str:
    if isinstance(value, np.datetime64):
        stamp = pd.Timestamp(value)
        return stamp.date().isoformat() if stamp == stamp.normalize() else stamp.isoformat()
    value = _plain(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) and value.dtype.kind not in "mM" else value


def bins(field: Any, edges: Any) -> ValueKey:
    """Group by ranges: left-closed, right-open bins with *edges*, or a duration for time."""
    expr = as_expr(field)
    name = f"{expr.name}_bin" if isinstance(expr, Ref) else "bin"
    return ValueKey(name, expr, edges)


# ── Elementwise functions ───────────────────────────────────────────────────


def _is_in(node: Call, ctx: Context) -> Any:
    value = evaluate(node.args[0], ctx)
    choices = evaluate(node.args[1], ctx)
    return value.isin(list(choices)) if isinstance(value, xr.DataArray) else value in list(choices)


def _if_else(node: Call, ctx: Context) -> Any:
    condition, true, false = (evaluate(arg, ctx) for arg in node.args)
    texts = [arg.text() for arg in node.args]
    arrays = [(v, t) for v, t in zip((condition, true, false), texts) if isinstance(v, xr.DataArray)]
    for (a, a_text), (b, b_text) in zip(arrays, arrays[1:]):
        align_pair(a, b, a_text, b_text, ctx.notes)
    with xr.set_options(**_EXACT):
        return xr.where(condition, true, false)


def _abs(node: Call, ctx: Context) -> Any:
    return abs(evaluate(node.args[0], ctx))


FUNCTIONS: dict[str, Callable[[Call, Context], Any]] = {
    "is_in": _is_in,
    "if_else": _if_else,
    "abs": _abs,
}
