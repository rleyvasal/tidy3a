"""Expressions: references to data, operators, reductions, and their evaluation.

An expression is built eagerly from Python operators (``x > 10``,
``mean(intensity)``) and evaluated later against a ``TidyArray`` by a verb.
Names resolve by plan §5.1–5.2: a variable, then a field-axis label, then an
observation axis's labels. Arithmetic between arrays follows the alignment
rules in ``tidy3a.align``; stored data is reached through the frame's backend
(``tidy3a.backend``).
"""

from __future__ import annotations

import builtins
import operator
import sys
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

from tidy3a.align import align_many, align_pair
from tidy3a.backend import is_xarray, xarray_backend
from tidy3a.errors import GroupError, RefError, TidyError
from tidy3a.frame import TidyArray, tidy
from tidy3a.grouped import Grouping


# ── Nodes ───────────────────────────────────────────────────────────────────


class Expr:
    """Base class for tidy3a expressions; combine them with Python operators."""

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


class NameRef(Ref):
    """A bare notebook name: data, a Python value, or an error if it is both (§5.2).

    The notebook layer creates these when a cell runs; *python* is the
    Python value the name had then, or ``MISSING``.
    """

    def __init__(self, name: Hashable, python: Any):
        super().__init__(name)
        self.python = python


MISSING = object()


def is_data_name(name: Hashable, ctx: Context) -> bool:
    data, backend = ctx.data, ctx.backend
    return (
        backend.has_variable(data, name)
        or backend.has_axis(data, name)
        or (ctx.field_axis is not None and backend.has_label(data, ctx.field_axis, name))
    )


def check_name(ref: Ref, ctx: Context) -> bool:
    """For a notebook name, raise if it is both data and a Python value; return whether it is data."""
    if not isinstance(ref, NameRef):
        return True
    data = is_data_name(ref.name, ctx)
    if data and ref.python is not MISSING:
        shown = repr(ref.python) if isinstance(ref.python, (int, float, complex, str)) else type(ref.python).__name__
        raise ctx.error(
            f'"{ref.name}" is both data and a notebook variable ({ref.name} = {shown}).\n'
            f'  data:      col("{ref.name}")\n'
            f'  notebook:  env("{ref.name}")\n'
            "  tidy3 lets data win here; tidy3a asks, because array notebooks reuse short names."
        )
    return data


class Literal(Expr):
    def __init__(self, value: Any, label: str | None = None):
        self.value = value
        self.label = label

    def text(self) -> str:
        if self.label is not None:
            return self.label
        value = self.value
        if is_xarray(value):
            return f"<array {value.name}>" if value.name is not None else "<array>"
        if isinstance(value, TidyArray):
            names = value._backend.variable_names(value._data)
            return f"<array {names[0]}>" if len(names) == 1 else "<data>"
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
    """*true* where *condition* holds, else *false*; tidy3a's masking function."""
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
    if isinstance(value, Expr) or is_xarray(value):
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


class Groups(Grouping):
    """A value key resolved against current data: a grouping of the positions on ``axis``.

    ``codes_array`` is the codes as the backend's array along ``axis``, with
    its labels, for broadcasting a per-group result back onto the positions.
    """

    def __init__(self, name: Hashable, axis: Hashable, codes_array: Any, labels: Sequence[Any]):
        super().__init__(codes_array.values, labels)
        self.name = name
        self.axis = axis
        self.codes_array = codes_array


# ── Evaluation ──────────────────────────────────────────────────────────────


class Context:
    """What an expression is evaluated against, and how reductions behave."""

    def __init__(self, frame: Any, *, verb: str, error: type[TidyError] = TidyError,
                 mode: str = "rows", grouped: bool = True):
        self.frame = frame
        self.backend = frame._backend
        self.data = frame._data
        field_axis = frame._field_axis
        self.field_axis = field_axis if self.backend.has_axis(self.data, field_axis) else None
        self.origins = frame._origins
        self.verb = verb
        self.error = error
        self.mode = mode
        self.keys = frame._groups if grouped else ()
        self.notes: list[str] = []
        self._groups: Groups | None = None
        resolved = getattr(frame, "_resolved", None)
        if resolved is not None and resolved[0] is self.data and resolved[1] is self.value_key:
            self._groups = resolved[2]  # found by group_by or an earlier mutate step on this data

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

    def observation_axes(self, array: Any) -> list[Hashable]:
        return [dim for dim in array.dims if dim != self.field_axis]

    # Formula compilation asks the context about arrays, so it can share the
    # parser and binding rules with an experimental evaluator.
    @property
    def variables(self):
        return self.backend.variables(self.data)

    @property
    def axes(self):
        return self.backend.axes(self.data)

    def is_data_name(self, name):
        return is_data_name(name, self)

    def evaluate(self, node):
        return evaluate(node, self)

    def is_array(self, value):
        return self.backend.is_array(value)

    def coords_for(self, value):
        return self.backend.label_map(value)

    def pick_label(self, value, axis, label):
        return self.backend.sel(value, {axis: label}, drop=True)

    def pick_position(self, value, axis, position):
        return self.backend.isel(value, {axis: position}, drop=True)

    def inventory(self):
        return _inventory(self)


def evaluate(node: Expr, ctx: Context) -> Any:
    """Evaluate *node*; arrays come back as the backend's arrays, constants as Python values."""
    if isinstance(node, Ref):
        return resolve(node, ctx)
    if isinstance(node, Literal):
        return _outside(node.value, ctx)
    if isinstance(node, Vector):
        return resolve_vector(node, ctx)
    if isinstance(node, BinOp):
        return _evaluate_binary(node, ctx)
    if isinstance(node, UnaryOp):
        value = evaluate(node.operand, ctx)
        op = {"-": operator.neg, "+": operator.pos, "~": operator.invert}[node.symbol]
        return ctx.backend.apply(op, value)
    if isinstance(node, Call):
        function = FUNCTIONS.get(node.name)
        if function is None:
            raise TidyError(f"{node.name}() is not a tidy3a function")
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
    """Apply *op* under tidy3a's alignment rules."""
    for value, text in ((left, left_text), (right, right_text)):
        if isinstance(value, np.ndarray) and value.ndim:
            raise TidyError(
                f"{text} is a NumPy array without axis names; pairing it by position could "
                'misalign it. Name its axes: read_array(values, dims={"point": infer}), '
                'or xr.DataArray(values, dims=["point"])'
            )
    backend = ctx.backend
    if backend.is_array(left) and backend.is_array(right):
        left, right = align_pair(left, right, left_text, right_text, ctx.notes, backend=backend)
    return backend.apply(op, left, right)


def _outside(value: Any, ctx: Context) -> Any:
    """A value from the calling code as the frame's backend's array, where it is an array.

    An xarray array is converted by the backend. A ``TidyArray`` of one
    variable, such as a rotation matrix from ``read_array``, stands for that
    variable, so arrays can come from outside without xarray; one on the
    other backend is moved to this one.
    """
    if not isinstance(value, TidyArray):
        return ctx.backend.outside(value)
    names = value._backend.variable_names(value._data)
    if len(names) != 1:
        listed = ", ".join(map(str, names))
        raise ctx.error(
            f"a TidyArray in an expression stands for its one variable, but this one holds "
            f"{len(names)} variables ({listed}); select one first, e.g. data >> select({names[0] if names else '…'})"
        )
    if value._backend is not ctx.backend:
        value = tidy(value, backend=ctx.backend.name)
    return ctx.backend.variable(value._data, names[0])


def resolve(ref: Ref, ctx: Context) -> Any:
    """Resolve a name: variable, then field label, then axis labels (plan §5.1–5.2).

    A notebook name that is not data resolves to its Python value.
    """
    if isinstance(ref, NameRef) and not check_name(ref, ctx) and ref.python is not MISSING:
        return _outside(ref.python, ctx)
    data, backend = ctx.data, ctx.backend
    name = ref.name
    field_axis = ctx.field_axis
    is_variable = backend.has_variable(data, name)
    holders = []
    if field_axis is not None and backend.has_label(data, field_axis, name):
        holders = [v for v, value in backend.variables(data).items() if field_axis in value.dims]
    is_label = bool(holders)

    if ref.kind == "var":
        if not is_variable:
            raise ctx.error(f'variable("{name}"): there is no variable {name}. {_inventory(ctx)}')
        return backend.variable(data, name)
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
        return backend.variable(data, name)
    if is_label:
        return _field_view(data, holders, name, field_axis, ctx)
    if backend.has_axis(data, name):
        return backend.axis_labels(data, name)
    raise ctx.error(f'"{name}" is not a variable, field label, or axis of this data. {_inventory(ctx)}'
                    + _observation_hint(data, name, field_axis, backend))


def _field_view(data: Any, holders: list, name: Hashable, field_axis: Hashable, ctx: Context):
    if len(holders) > 1:
        listed = ", ".join(str(v) for v in holders)
        raise ctx.error(
            f'"{name}" is a field label on {len(holders)} variables ({listed}); '
            f"select one first, e.g. select({holders[0]})"
        )
    return ctx.backend.field(data, holders[0], field_axis, name)


def axis_labels(data: Any, axis: Hashable, backend: Any = None) -> Any:
    """An axis's labels as data along it; unlabelled axes give 1-based positions."""
    return (xarray_backend() if backend is None else backend).axis_labels(data, axis)


def _inventory(ctx: Context) -> str:
    data, backend = ctx.data, ctx.backend
    parts = ["variables: " + (", ".join(str(v) for v in backend.variable_names(data)) or "none")]
    if ctx.field_axis is not None:
        labels = "  ".join(str(v) for v in backend.label_list(data, ctx.field_axis))
        parts.append(f"labels on {ctx.field_axis}: {labels}")
    return "\n  " + "\n  ".join(parts)


def _observation_hint(data: Any, name: Hashable, field_axis: Hashable, backend: Any = None) -> str:
    backend = xarray_backend() if backend is None else backend
    for dim in backend.labelled_axes(data):
        if dim != field_axis and backend.has_label(data, dim, name):
            return f'\n  "{name}" is a value on the {dim} axis; to keep it use filter({dim} == {name!r})'
    return ""


def resolve_vector(node: Vector, ctx: Context) -> Any:
    """``[x, y, z]``: field labels on one variable, or variables from one recorded axis."""
    data, backend = ctx.data, ctx.backend
    names = [item.name for item in node.items]
    field_axis = ctx.field_axis
    kinds = []
    for item in node.items:
        check_name(item, ctx)
        value_is_var = backend.has_variable(data, item.name) and item.kind != "label"
        value_is_label = (
            field_axis is not None and backend.has_label(data, field_axis, item.name) and item.kind != "var"
        )
        if value_is_var and value_is_label:
            resolve(item, ctx)  # raises the collision error
        kinds.append("var" if value_is_var else "label" if value_is_label else None)
    if None in kinds:
        resolve(node.items[kinds.index(None)], ctx)  # raises the unknown-name error
    if builtins.all(kind == "label" for kind in kinds):
        holders = [v for v, value in backend.variables(data).items() if field_axis in value.dims]
        if len(holders) > 1:
            raise ctx.error(
                f"{node.text()} are field labels on several variables "
                f"({', '.join(str(v) for v in holders)}); select one first"
            )
        return backend.sel(backend.variable(data, holders[0]), {field_axis: names})
    if builtins.all(kind == "var" for kind in kinds):
        origins = [ctx.origins.get(name) for name in names]
        axes = {origin[0] for origin in origins if origin is not None}
        if None not in origins and len(axes) == 1:
            axis = axes.pop()
            return backend.concat([backend.variable(data, name) for name in names], axis, names)
        raise ctx.error(
            f"{node.text()} are variables without a shared origin axis; "
            f'pack them explicitly: pack({node.text()}, along="…")'
        )
    raise ctx.error(f"{node.text()} mixes variables and field labels; a vector holds one kind")


# ── Reductions ──────────────────────────────────────────────────────────────

_FLOAT_REDUCERS = {"mean", "sum", "min", "max", "median", "std", "var"}


def reduce_array(
    array: Any, op: str, axes: Sequence[Hashable], na_rm: bool = False,
    min_count: int | None = None, backend: Any = None,
) -> Any:
    """One reduction over *axes*, with missing values propagated unless *na_rm*."""
    return (xarray_backend() if backend is None else backend).reduce(array, op, axes, na_rm, min_count)


def _evaluate_reduce(node: Reduce, ctx: Context) -> Any:
    error = ctx.error
    backend = ctx.backend
    if node.op == "n":
        return _count(ctx)
    array = evaluate(node.arg, ctx)
    if not backend.is_array(array):
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
        return backend.reduce(array, node.op, axes, node.na_rm, node.min_count)

    axes = [axis for axis in ctx.observation_axes(array) if axis not in ctx.group_axes]
    if ctx.value_key is None:
        return backend.reduce(array, node.op, axes, node.na_rm, node.min_count)
    groups = ctx.groups()
    others = [axis for axis in axes if axis != groups.axis]
    stats = backend.group_reduce(array, node.op, groups, others, node.na_rm, node.min_count)
    return stats if ctx.mode == "summarise" else backend.broadcast(stats, groups)


def _check_has_group_axes(node: Reduce, array: Any, ctx: Context) -> None:
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


def _count(ctx: Context) -> Any:
    """``n()``: positions on observation axes shared by non-scalar variables.

    Scalars do not change the count. Axes only some non-scalar variables have,
    such as a rotation's component axis, are not observations.
    """
    data, backend = ctx.data, ctx.backend
    axes, sizes = backend.axes(data), backend.sizes(data)
    shared = set(axes)
    for variable in backend.variables(data).values():
        if variable.ndim:
            shared &= set(variable.dims)
    observation = [d for d in axes if d in shared and d != ctx.field_axis and d not in ctx.group_axes]
    if ctx.value_key is None:
        return int(np.prod([sizes[d] for d in observation], dtype=np.int64))
    groups = ctx.groups()
    per_position = int(np.prod([sizes[d] for d in observation if d != groups.axis], dtype=np.int64))
    stats = backend.group_counts(groups, per_position)
    return stats if ctx.mode == "summarise" else backend.broadcast(stats, groups)


# ── Value keys ──────────────────────────────────────────────────────────────


def resolve_value_key(key: ValueKey, frame: Any, group_axes: Sequence[Hashable]) -> Groups:
    """Evaluate a value key on current data and give each position a group code."""
    ctx = Context(frame, verb="group_by", error=GroupError, grouped=False)
    values = evaluate(key.expr, ctx)
    if not ctx.backend.is_array(values) or values.ndim != 1:
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
        return _value_groups(key, axis, values, ctx.backend)
    return _bin_groups(key, axis, values, ctx.backend)


def _value_groups(key: ValueKey, axis: Hashable, values: Any, backend: Any) -> Groups:
    raw = values.values
    if raw.dtype.kind in "fc" and np.isnan(raw).any():
        raise GroupError(
            f"group_by({key.label}): {int(np.isnan(raw).sum()):,} positions have a missing "
            f"{key.label}; filter them out first"
        )
    labels, codes = _factorize(raw)
    return Groups(key.name, axis, backend.codes_like(codes, values), _label_list(labels))


def _pandas() -> Any:
    """pandas when installed; it is optional, with NumPy fallbacks where tidy3a uses it."""
    try:
        import pandas
    except ImportError:
        return None
    return pandas


def _factorize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique values and each position's code, exactly as ``np.unique`` gives them.

    pandas, when installed, finds the groups by hashing, which is much faster
    than sorting every value. It gives missing values such as NaT no code,
    and some object values cannot be sorted; those fall back to ``np.unique``.
    """
    pd = _pandas()
    try:
        codes, labels = pd.factorize(raw, sort=True) if pd is not None else (None, None)
    except TypeError:
        codes = None
    if codes is None or (codes < 0).any():
        labels, codes = np.unique(raw, return_inverse=True)
        return labels, codes.reshape(-1)
    return np.asarray(labels), codes


def _bin_groups(key: ValueKey, axis: Hashable, values: Any, backend: Any) -> Groups:
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
    return Groups(key.name, axis, backend.codes_like(codes, values), labels)


def _bin_edges(key: ValueKey, raw: np.ndarray) -> np.ndarray:
    edges = key.edges
    if isinstance(edges, str):
        if raw.dtype.kind != "M":
            raise GroupError(f'bins({key.expr.text()}, "{edges}"): a duration needs time values')
        step = _duration(edges, key)
        if raw.size == 0:
            raise GroupError(f'bins({key.expr.text()}, "{edges}"): there are no times to bin')
        start = raw.min()
        count = int((raw.max() - start) // step) + 1
        return start + np.arange(count + 1) * step
    edges = np.asarray(edges)
    if edges.ndim != 1 or len(edges) < 2 or not (edges[1:] > edges[:-1]).all():
        raise GroupError(f"bins({key.expr.text()}, …) needs at least two increasing edges")
    return edges


# Duration units a bins() string may use, in nanoseconds. Months and years are not
# fixed lengths, and "m" could be either minutes or months, so none of them are units.
_WEEK, _DAY, _HOUR, _MINUTE, _SECOND = 7 * 86_400 * 10**9, 86_400 * 10**9, 3_600 * 10**9, 60 * 10**9, 10**9
_UNITS = {
    **dict.fromkeys(("w", "week", "weeks"), _WEEK),
    **dict.fromkeys(("d", "day", "days"), _DAY),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), _HOUR),
    **dict.fromkeys(("min", "mins", "minute", "minutes"), _MINUTE),
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), _SECOND),
    **dict.fromkeys(("ms", "millisecond", "milliseconds"), 10**6),
    **dict.fromkeys(("us", "microsecond", "microseconds"), 10**3),
    **dict.fromkeys(("ns", "nanosecond", "nanoseconds"), 1),
}


def _duration(text: str, key: ValueKey) -> np.timedelta64:
    """A duration such as "30 days", "2D", or "1h30min": numbers with units, added exactly.

    tidy3a reads durations itself, so a duration means the same whether or
    not pandas is installed.
    """
    import re
    from fractions import Fraction

    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", text)
    rest = re.sub(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", "", text).strip()
    units = [_UNITS.get(unit.lower()) for _, unit in parts]
    total = sum((Fraction(number) * unit for (number, _), unit in zip(parts, units) if unit), Fraction(0))
    if not parts or rest or not all(units) or total.denominator != 1 or total <= 0:
        raise GroupError(
            f'bins({key.expr.text()}, "{text}"): cannot read the duration; write numbers with units '
            '(weeks, days, hours, minutes, seconds, ms, us, ns), e.g. "30 days" or "1h30min", '
            "a whole number of nanoseconds; months and years have no fixed length"
        )
    return np.timedelta64(int(total), "ns")


def _format_edge(value: Any) -> str:
    if isinstance(value, np.datetime64):
        pd = _pandas()
        if pd is not None:
            stamp = pd.Timestamp(value)
            return stamp.date().isoformat() if stamp == stamp.normalize() else stamp.isoformat()
        nanoseconds = value.astype("datetime64[ns]")
        if nanoseconds == nanoseconds.astype("datetime64[D]"):
            return str(np.datetime_as_string(nanoseconds, unit="D"))
        for unit in ("s", "us"):  # as pandas prints: whole seconds, microseconds, else nanoseconds
            if nanoseconds == nanoseconds.astype(f"datetime64[{unit}]"):
                return str(np.datetime_as_string(nanoseconds, unit=unit))
        return str(np.datetime_as_string(nanoseconds, unit="ns"))
    value = _plain(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _label_list(labels: np.ndarray) -> list:
    """Group labels as ``_plain`` gives them, converted in one call where NumPy can.

    ``tolist()`` gives the same Python values as converting each label; times
    and durations stay NumPy scalars, and object arrays are checked one by one.
    """
    labels = np.asarray(labels)
    if labels.dtype.kind in "mM":
        return list(labels)
    if labels.dtype.kind == "O":
        return [_plain(label) for label in labels]
    return labels.tolist()


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
    return ctx.backend.isin(value, list(choices)) if ctx.backend.is_array(value) else value in list(choices)


def _if_else(node: Call, ctx: Context) -> Any:
    values = [evaluate(arg, ctx) for arg in node.args]
    texts = [arg.text() for arg in node.args]
    values = align_many(values, texts, ctx.notes, backend=ctx.backend)
    return ctx.backend.where(*values)


def _abs(node: Call, ctx: Context) -> Any:
    return ctx.backend.apply(operator.abs, evaluate(node.args[0], ctx))


FUNCTIONS: dict[str, Callable[[Call, Context], Any]] = {
    "is_in": _is_in,
    "if_else": _if_else,
    "abs": _abs,
}
