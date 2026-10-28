"""Expressions to Numba source with NumPy's types (plan milestone 16, step 2).

Each step of an expression becomes Python source for one value in a kernel's
loop, its operands cast to the type NumPy computes it in and its result to
the type NumPy gives it, both known from the step run on the block of no
positions. IEEE arithmetic is the same in Numba as in NumPy for each of
``+ - * /``, ``sqrt``, comparisons, ``abs``, and ``floor`` in a given type,
so where every step runs in NumPy's type the bits are NumPy's;
``tests/test_fused_expressions.py`` holds each operation to that over types
and special values. A Python number takes the type NumPy gives it beside an
array (``float32 - 273.15`` is float32), as NumPy 2 does.

A value is per position (``name[i]``) or per value of a position
(``name[i, j]``, a grid's cells flattened), as its variable is; a step of
both is per value.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tidy3a.expr import BinOp, Call, Context, Literal, Ref, UnaryOp, evaluate

ARITHMETIC = {"+", "-", "*", "/"}
COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
LOGIC = {"&", "|"}
FUNCTIONS = {"sqrt", "abs", "floor", "if_else"}


class Unsupported(Exception):
    """A step the compiler does not take, and why; the planner falls back to NumPy for it."""


class Inherited(Unsupported):
    """A step reading a variable an earlier ``mutate`` made in a way that runs on NumPy, whose reason was given
    there."""


@dataclass(frozen=True)
class Source:
    """One step as kernel source: ``text`` gives its value at position ``i`` (and value ``j``), in ``dtype``; a
    Python number has no type of its own (``None``) until a step gives it one."""

    text: str
    dtype: np.dtype | None
    cells: bool = False
    value: Any = None  # a Python number's value, for the type NumPy gives it beside an array


@dataclass
class Compiler:
    """Expressions of one block's verbs as source, the arrays they read named ``a0``, ``a1``, ... in ``inputs``.

    A variable an earlier ``mutate`` of the pipeline made is not read from an array: its defining expression is
    compiled in its place (``define``), in the frame it was defined in, and cast to the type it was stored as, which
    is what NumPy computes in storing it and reading it back. The values of a position (a grid's) must lie the same
    way in every variable read, since a kernel does not broadcast one grid onto another.
    """

    axis: Hashable
    inputs: dict[str, tuple[Ref, bool]] = field(default_factory=dict)  # local name: (reference, per value)
    definitions: dict[Hashable, tuple[Any, Context, dict] | None] = field(default_factory=dict)  # None: refused
    cells: tuple | None = None  # the other axes of every variable read with values per position, and their sizes

    def define(self, name: Hashable, expression: Any, context: Context) -> None:
        """``mutate(name = expression)``, with *context* the frame before it: later steps compile *expression*, with
        the definitions as they stood before it, so ``x = x * 2`` reads the earlier ``x``."""
        self.definitions = {**self.definitions, name: (expression, context, self.definitions)}

    def refuse(self, name: Hashable) -> None:
        """``mutate(name = ...)`` runs on NumPy: a later step reading *name* is refused as ``Inherited``."""
        self.definitions = {**self.definitions, name: None}

    def compile(self, node: Any, context: Context) -> Source:
        lower = getattr(node, "lower", None)
        if lower is not None:  # a formula: what it lowers to on the data
            node, _ = lower(context)
        if isinstance(node, Literal):
            return _literal(node)
        if isinstance(node, Ref):
            return self._ref(node, context)
        if isinstance(node, Call) and node.name not in FUNCTIONS:  # refused before its values are computed
            raise Unsupported(f"{node.name}() runs on NumPy")
        if isinstance(node, BinOp) and node.symbol not in ARITHMETIC | COMPARISONS | LOGIC | {"**"}:
            raise Unsupported(f"{node.text()}: {node.symbol} runs on NumPy")
        result = value_type(evaluate(node, context))
        if result.kind not in "biuf" or result == np.float16:  # Numba has no half precision, and these are numbers
            raise Unsupported(f"{node.text()} gives {kind_words(result)}, which runs on NumPy")
        if isinstance(node, BinOp):
            return self._binary(node, context, result)
        if isinstance(node, UnaryOp):
            return self._unary(node, context, result)
        if isinstance(node, Call):
            return self._call(node, context, result)
        raise Unsupported(f"{node.text()} runs on NumPy")

    def _ref(self, node: Ref, context: Context) -> Source:
        value = evaluate(node, context)
        dtype = value_type(value)
        if dtype.kind not in "biuf" or dtype == np.float16:
            raise Unsupported(f"{node.text()} is {kind_words(dtype)}, which runs on NumPy")
        if node.kind != "label" and node.name in self.definitions:  # made by an earlier mutate: its expression
            if self.definitions[node.name] is None:
                raise Inherited(f"{node.text()} is made on NumPy")
            expression, defined, earlier = self.definitions[node.name]
            now, self.definitions = self.definitions, earlier
            try:
                source = self.compile(expression, defined)
            finally:
                self.definitions = now
            return Source(_as(cast(source, dtype), dtype), dtype, source.cells)
        dims = tuple(getattr(value, "dims", ()))
        cells = any(dim != self.axis for dim in dims)
        if self.axis not in dims:
            raise Unsupported(f"{node.text()} does not vary along {self.axis}, which runs on NumPy")
        if cells:
            if dims[0] != self.axis:
                raise Unsupported(f"{node.text()} does not have {self.axis} first, which runs on NumPy")
            layout = tuple(zip(dims[1:], getattr(value, "shape", ())[1:]))
            if self.cells is not None and layout != self.cells:
                raise Unsupported(f"{node.text()} lies over other axes than the values before it, which runs on NumPy")
            self.cells = layout
        for local, (ref, _) in self.inputs.items():
            if ref.name == node.name and ref.kind == node.kind:
                break
        else:
            local = f"a{len(self.inputs)}"
            self.inputs[local] = (node, cells)
        return Source(f"{local}[i, j]" if cells else f"{local}[i]", dtype, cells)

    def _binary(self, node: BinOp, context: Context, result: np.dtype) -> Source:
        left, right = self.compile(node.left, context), self.compile(node.right, context)
        cells = left.cells or right.cells
        if node.symbol == "**":
            return _power(left, node.right, result, cells)
        if node.symbol in ARITHMETIC | LOGIC:
            computed = result
        elif node.symbol in COMPARISONS:
            computed = np.result_type(_operand(left), _operand(right))
            if computed == np.float16:
                raise Unsupported(f"{node.text()} compares in float16, which runs on NumPy")
        else:
            raise Unsupported(f"{node.text()}: {node.symbol} runs on NumPy")
        text = f"({cast(left, computed)} {node.symbol} {cast(right, computed)})"
        return Source(_as(text, result), result, cells)

    def _unary(self, node: UnaryOp, context: Context, result: np.dtype) -> Source:
        operand = self.compile(node.operand, context)
        if node.symbol == "+":
            return Source(_as(cast(operand, result), result), result, operand.cells)
        if node.symbol == "-":
            return Source(_as(f"(-{cast(operand, result)})", result), result, operand.cells)
        if node.symbol == "~":
            inner = cast(operand, result)
            return Source(_as(f"(not {inner})" if result == np.bool_ else f"(~{inner})", result), result, operand.cells)
        raise Unsupported(f"{node.text()}: {node.symbol} runs on NumPy")

    def _call(self, node: Call, context: Context, result: np.dtype) -> Source:
        if node.options:
            raise Unsupported(f"{node.name}() with options runs on NumPy")
        args = [self.compile(arg, context) for arg in node.args]
        cells = any(arg.cells for arg in args)
        if node.name == "sqrt":
            return Source(_as(f"np.sqrt({cast(args[0], result)})", result), result, cells)
        if node.name == "abs":
            return Source(_as(f"abs({cast(args[0], result)})", result), result, cells)
        if node.name == "floor":
            if result.kind in "iub":  # NumPy gives an integer its own value
                return Source(_as(cast(args[0], result), result), result, cells)
            return Source(_as(f"np.floor({cast(args[0], result)})", result), result, cells)
        if node.name == "if_else":
            condition, true, false = args
            text = f"({cast(true, result)} if {cast(condition, np.dtype(bool))} else {cast(false, result)})"
            return Source(_as(text, result), result, cells)
        raise Unsupported(f"{node.name}() runs on NumPy")


def _power(base: Source, exponent: Any, result: np.dtype, cells: bool) -> Source:
    """``**`` where NumPy computes it exactly: 2 as ``x*x``, 0.5 as ``sqrt``, -1 as ``1/x``, 1 as ``x``; for an
    integer base only 2 and 1, NumPy taking the C library's ``pow`` for the others."""
    value = exponent.value if isinstance(exponent, Literal) else None
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        raise Unsupported("a power whose exponent is not a number runs on NumPy")
    value = float(value)
    exact = {2.0, 1.0} if base.dtype is not None and base.dtype.kind in "iu" else {2.0, 0.5, -1.0, 1.0}
    if value not in exact or result.kind not in "iuf":
        raise Unsupported(f"a power of {value:g} runs on NumPy")
    operand = cast(base, result)
    if value == 2.0:
        text = f"({operand} * {operand})"
    elif value == 0.5:
        text = f"np.sqrt({operand})"
    elif value == -1.0:
        text = f"({_number(1, result)} / {operand})"
    else:
        text = operand
    return Source(_as(text, result), result, cells)


def _literal(node: Literal) -> Source:
    value = node.value
    if isinstance(value, (bool, np.bool_)):
        return Source(repr(bool(value)), None, value=bool(value))
    if isinstance(value, (int, np.integer)):
        return Source(repr(int(value)), None if isinstance(value, int) else np.dtype(type(value)), value=value)
    if isinstance(value, (float, np.floating)) and not isinstance(value, np.float16):
        return Source(repr(float(value)), None if isinstance(value, float) else np.dtype(type(value)), value=value)
    raise Unsupported(f"{node.text()} is not a number, which runs on NumPy")


def _operand(source: Source) -> Any:
    """What NumPy promotes *source* as: its type, or a Python number's value (NumPy 2's weak scalars)."""
    return source.dtype if source.dtype is not None else source.value


def cast(source: Source, dtype: np.dtype) -> str:
    """*source* in *dtype*, as NumPy converts it before computing."""
    if source.dtype is None:
        return _number(source.value, dtype)
    return source.text if source.dtype == dtype else f"{_type(dtype)}({source.text})"


def _as(text: str, dtype: np.dtype) -> str:
    """A step's value made its result type (Numba adds booleans as integers, NumPy as logic)."""
    return f"{_type(dtype)}({text})"


def _number(value: Any, dtype: np.dtype) -> str:
    return f"{_type(dtype)}({value!r})"


def _type(dtype: np.dtype) -> str:
    return "np.bool_" if dtype == np.bool_ else f"np.{np.dtype(dtype).name}"


def value_type(value: Any) -> np.dtype:
    dtype = getattr(value, "dtype", None)
    return np.dtype(dtype) if dtype is not None else np.asarray(value).dtype


def kind_words(dtype: Any) -> str:
    """A type as a reason gives it: "times", "text", "half-precision floats"."""
    dtype = np.dtype(dtype)
    if dtype == np.float16:
        return "half-precision floats"
    return {"M": "times", "m": "durations", "U": "text", "S": "text", "O": "objects",
            "c": "complex numbers"}.get(dtype.kind, f"{dtype} values")
