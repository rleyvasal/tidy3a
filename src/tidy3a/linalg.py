"""The array layer (plan §6): dot, numeric functions, conjugation, and axis helpers.

These helpers name their axes openly. ``dot`` sums over the axes in
``over=`` and keeps every other axis, so batch axes such as ``time`` stay
without special rules. ``dot`` never conjugates; write ``conj()`` where an
inner product needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

from tidy3a.align import align_many, align_pair
from tidy3a.expr import (
    FUNCTIONS,
    Call,
    Context,
    Ref,
    _is_data,
    as_expr,
    evaluate,
)


def _axes(over: Any) -> list[Hashable] | None:
    if over is None:
        return None
    return list(over) if isinstance(over, (list, tuple)) else [over]


# ── dot ─────────────────────────────────────────────────────────────────────


def dot(a: Any, b: Any, *, over: Any = None) -> Call:
    """Multiply matching components and sum over the ``over=`` axes; other axes stay.

    Without ``over=``, *a* and *b* must share exactly one axis, which is summed.
    """
    return Call("dot", (as_expr(a), as_expr(b)), {"over": over})


def _dot(node: Call, ctx: Context) -> Any:
    backend = ctx.backend
    left, right = (evaluate(arg, ctx) for arg in node.args)
    texts = [arg.text() for arg in node.args]
    for value, text in zip((left, right), texts):
        if not backend.is_array(value):
            raise ctx.error(f"dot(): {text} is not an array with named axes")
    left, right = align_pair(left, right, texts[0], texts[1], ctx.notes, backend=backend)
    shared = [d for d in left.dims if d in right.dims]
    over = _axes(node.options.get("over"))
    if over is None:
        if len(shared) != 1:
            listed = ", ".join(map(str, shared)) or "none"
            raise ctx.error(
                f"{node.text()}: {texts[0]} and {texts[1]} share {len(shared)} axes ({listed}); "
                'name the ones to sum with over=, e.g. over="' + (str(shared[0]) if shared else "axis") + '"'
            )
        over = shared
    for axis in over:
        missing = [t for v, t in zip((left, right), texts) if axis not in v.dims]
        if missing:
            raise ctx.error(f"{node.text()}: {axis} is not an axis of {missing[0]}")
    return backend.dot(left, right, over)


# ── elementwise numeric functions ───────────────────────────────────────────


def _elementwise(name: str, ufunc: Callable) -> Callable:
    def function(x: Any) -> Any:
        if not _is_data(x):
            return ufunc(x)
        return Call(name, (as_expr(x),))

    def apply(node: Call, ctx: Context) -> Any:
        return ufunc(evaluate(node.args[0], ctx))

    function.__name__ = name
    function.__qualname__ = name
    function.__doc__ = f"NumPy's {ufunc.__name__}, applied element by element."
    FUNCTIONS[name] = apply
    return function


sqrt = _elementwise("sqrt", np.sqrt)
exp = _elementwise("exp", np.exp)
log = _elementwise("log", np.log)
log2 = _elementwise("log2", np.log2)
log10 = _elementwise("log10", np.log10)
sin = _elementwise("sin", np.sin)
cos = _elementwise("cos", np.cos)
tan = _elementwise("tan", np.tan)
arcsin = _elementwise("arcsin", np.arcsin)
arccos = _elementwise("arccos", np.arccos)
arctan = _elementwise("arctan", np.arctan)
sinh = _elementwise("sinh", np.sinh)
cosh = _elementwise("cosh", np.cosh)
tanh = _elementwise("tanh", np.tanh)
floor = _elementwise("floor", np.floor)
ceil = _elementwise("ceil", np.ceil)
sign = _elementwise("sign", np.sign)
conj = _elementwise("conj", np.conj)
real = _elementwise("real", np.real)
imag = _elementwise("imag", np.imag)
angle = _elementwise("angle", np.angle)


def arctan2(y: Any, x: Any) -> Any:
    """The angle of (x, y), in radians, element by element."""
    if not (_is_data(y) or _is_data(x)):
        return np.arctan2(y, x)
    return Call("arctan2", (as_expr(y), as_expr(x)))


def _arctan2(node: Call, ctx: Context) -> Any:
    backend = ctx.backend
    y, x = (evaluate(arg, ctx) for arg in node.args)
    if backend.is_array(y) and backend.is_array(x):
        y, x = align_pair(y, x, node.args[0].text(), node.args[1].text(), ctx.notes, backend=backend)
    return backend.apply(np.arctan2, y, x)


# ── axis helpers ────────────────────────────────────────────────────────────


def rename_axes(x: Any, **names: str) -> Call:
    """Rename axes, e.g. ``rename_axes(amp1, output="input")``; labels move with them."""
    if not names:
        raise TypeError('rename_axes() needs old="new" pairs, e.g. rename_axes(amp1, output="input")')
    return Call("rename_axes", (as_expr(x),), {"names": names})


def _rename_axes(node: Call, ctx: Context) -> Any:
    value = evaluate(node.args[0], ctx)
    names: dict[str, str] = node.options["names"]
    if not ctx.backend.is_array(value):
        raise ctx.error(f"rename_axes(): {node.args[0].text()} has no axes")
    for old, new in names.items():
        if old not in value.dims:
            raise ctx.error(f"rename_axes(): {old} is not an axis of {node.args[0].text()} "
                            f"(axes: {', '.join(map(str, value.dims))})")
        if new in value.dims and new not in names:
            raise ctx.error(f"rename_axes(): {node.args[0].text()} already has an axis {new}")
    return ctx.backend.rename_axes(value, names)


def pack(items: Sequence[Any], *, along: str) -> Call:
    """Pack separate values into one array along a new axis labelled by their names."""
    if not isinstance(along, str) or not along:
        raise TypeError('pack() needs along="axis name"; the new axis is never named for you')
    if not isinstance(items, (list, tuple)) or not items:
        raise TypeError("pack() takes a list of values, e.g. pack([vx, vy, vz], along=\"feature\")")
    return Call("pack", tuple(as_expr(item) for item in items), {"along": along})


def _pack(node: Call, ctx: Context) -> Any:
    backend = ctx.backend
    along = node.options["along"]
    values = [evaluate(arg, ctx) for arg in node.args]
    labels = [arg.name if isinstance(arg, Ref) else arg.text() for arg in node.args]
    arrays = []
    for value, arg in zip(values, node.args):
        if not backend.is_array(value):
            value = backend.array(value)
        if along in value.dims:
            raise ctx.error(f"pack(): {arg.text()} already has an axis {along}")
        arrays.append(value)
    arrays = align_many(arrays, [arg.text() for arg in node.args], ctx.notes, backend=backend)
    if len(set(map(str, labels))) != len(labels):
        raise ctx.error(f"pack(): the labels {labels} repeat; each packed value needs its own name")
    return backend.concat(arrays, along, labels, exact=True)


FUNCTIONS.update({
    "dot": _dot,
    "arctan2": _arctan2,
    "rename_axes": _rename_axes,
    "pack": _pack,
})


__all__ = [
    "angle", "arccos", "arcsin", "arctan", "arctan2", "ceil", "conj", "cos", "cosh", "dot",
    "exp", "floor", "imag", "log", "log2", "log10", "pack", "real", "rename_axes", "sign",
    "sin", "sinh", "sqrt", "tan", "tanh",
]
