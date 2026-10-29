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

import re

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tidy3a.expr import BinOp, Call, Context, Literal, Ref, UnaryOp, evaluate

ARITHMETIC = {"+", "-", "*", "/"}
COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
LOGIC = {"&", "|"}
FUNCTIONS = {"sqrt", "abs", "floor", "ceil", "sign", "if_else"}
DIVISIONS = {"//", "%"}  # compiled with NumPy's rules (``kernels.HELPERS``)
# Functions of one value at a time that NumPy computes in a way a kernel cannot match on every machine (its own vector
# routines, chosen by the CPU), so NumPy computes them for each block, as islands the kernel reads (milestone 18).
ISLANDS = {"exp", "log", "log2", "log10", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh",
           "tanh", "real", "imag", "angle", "is_in"}


class Unsupported(Exception):
    """A step the compiler does not take, and why; the planner falls back to NumPy for it."""


class Inherited(Unsupported):
    """A step reading a variable an earlier ``mutate`` made in a way that runs on NumPy, whose reason was given
    there."""


# What NumPy applies to an island's operands, which a kernel computes (milestone 18): its own routines for these.
APPLIED = {**{name: getattr(np, name) for name in ("exp", "log", "log2", "log10", "sin", "cos", "tan", "arcsin",
                                                   "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "ceil",
                                                   "sign", "real", "imag", "angle")},
           "//": np.floor_divide, "%": np.remainder, "**": np.power}


@dataclass(frozen=True, eq=False)
class Island:
    """A part of an expression NumPy computes for each block (milestone 18), its values read by the kernel as an
    input. Where its operands compile (*operands*), a kernel computes them, with NumPy's bits, as IEEE arithmetic
    rounds the same everywhere, and NumPy applies only its function (``APPLIED``) to them; else NumPy evaluates
    *node* in the frame after the ``mutate`` steps it reads (*steps*, run by NumPy in their order)."""

    node: Any
    steps: tuple
    operands: tuple = ()  # each operand's source, or a Python number as it stands
    function: Any = None
    dtype: np.dtype | None = None  # its values' type, as NumPy gives it

    def bytes(self, values: int) -> int:
        """What it holds per position of *values* values while a block runs (``sizing``): its values and its
        operands'; NumPy's temporaries too where it evaluates the island whole, taken as two more of its values and
        one of each step it runs again, which NumPy makes as it would on its own path."""
        size = self.dtype.itemsize if self.dtype is not None else 8
        if self.function is not None:  # an operand that is an input read as it is takes no array of its own
            return values * (size + sum(operand.dtype.itemsize for operand in self.operands
                                        if isinstance(operand, Source) and operand.dtype is not None
                                        and not re.fullmatch(r"a\d+\[i(, j)?\]", operand.text)))
        return values * (3 * size + 8 * len(self.steps))

    def text(self) -> str:
        return self.node.text()


@dataclass(frozen=True, eq=False)
class AsIntegers:
    """A field of times or durations read as the int64 counts it is held in, NaT the smallest (milestone 18): what a
    key places, its labels made times again."""

    ref: Ref

    def text(self) -> str:
        return self.ref.text()


def read(value: Ref | Island | AsIntegers, context: Context) -> Any:
    """What a kernel's input holds on the block of *context*: a variable's values, or an island's, computed by NumPy
    after the ``mutate`` steps it reads. (An island whose operands compile is computed by the driver, which has the
    kernels' other inputs; here, as for one that does not, NumPy evaluates it whole, the same bits.)"""
    if isinstance(value, AsIntegers):
        from types import SimpleNamespace

        held = evaluate(value.ref, context)
        return SimpleNamespace(values=np.asarray(held.values).view(np.int64), dims=tuple(held.dims))
    if not isinstance(value, Island):
        return evaluate(value, context)
    frame = context.frame
    if value.steps:
        from tidy3a.verbs import mutate

        for step in value.steps:
            frame = frame >> mutate(step)
    return evaluate(value.node, Context(frame, verb=context.verb, error=context.error, grouped=False))


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
    inputs: dict[str, tuple[Ref | Island, bool]] = field(default_factory=dict)  # local name: (what it reads, per value)
    # name: (expression, frame before it, the definitions before it, its mutate step, its place), or None: refused
    definitions: dict[Hashable, tuple[Any, Context, dict, Any, int] | None] = field(default_factory=dict)
    cells: tuple | None = None  # the other axes of every variable read with values per position, and their sizes

    def define(self, name: Hashable, expression: Any, context: Context, step: Any = None) -> None:
        """``mutate(name = expression)``, with *context* the frame before it: later steps compile *expression*, with
        the definitions as they stood before it, so ``x = x * 2`` reads the earlier ``x``; *step*, the mutate's
        assignment, is what an island reading *name* has NumPy run first."""
        place = sum(1 for value in self.definitions.values() if value is not None)
        self.definitions = {**self.definitions, name: (expression, context, self.definitions, step, place)}

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
            if node.name in ISLANDS:
                return self._island(node, context)
            raise Unsupported(f"{node.name}() runs on NumPy")
        if isinstance(node, BinOp) and node.symbol not in ARITHMETIC | COMPARISONS | LOGIC | DIVISIONS | {"**"}:
            raise Unsupported(f"{node.text()}: {node.symbol} runs on NumPy")
        if isinstance(node, BinOp) and node.symbol == "**" and not _exact_power(node, context):
            return self._island(node, context)
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
            expression, defined, earlier, _, _ = self.definitions[node.name]
            now, self.definitions = self.definitions, earlier
            try:
                source = self.compile(expression, defined)
            finally:
                self.definitions = now
            return Source(_as(cast(source, dtype), dtype), dtype, source.cells)
        cells = self._layout(node, value)
        for local, (ref, _) in self.inputs.items():
            if isinstance(ref, Ref) and ref.name == node.name and ref.kind == node.kind:
                break
        else:
            local = f"a{len(self.inputs)}"
            self.inputs[local] = (node, cells)
        if self.axis not in tuple(getattr(value, "dims", ())):  # the same for every position: read by value
            return Source(f"{local}[j]", dtype, True)
        return Source(f"{local}[i, j]" if cells else f"{local}[i]", dtype, cells)

    def _layout(self, node: Any, value: Any) -> bool:
        """Whether *value*, which a kernel reads, has values per position (a grid's), checking that it varies along
        the blocked axis, first, and lies as the values read before it; or that, without the blocked axis, it lies
        over exactly the other axes a position's values do (a mask joined over lat and lon, milestone 19), which
        ``_ref`` reads by value alone."""
        dims = tuple(getattr(value, "dims", ()))
        cells = any(dim != self.axis for dim in dims)
        if self.axis not in dims:
            layout = tuple(zip(dims, getattr(value, "shape", ())))
            if dims and (self.cells is None or layout == self.cells):
                self.cells = layout
                return True
            raise Unsupported(f"{node.text()} does not vary along {self.axis}, which runs on NumPy")
        if cells:
            if dims[0] != self.axis:
                raise Unsupported(f"{node.text()} does not have {self.axis} first, which runs on NumPy")
            layout = tuple(zip(dims[1:], getattr(value, "shape", ())[1:]))
            if self.cells is not None and layout != self.cells:
                raise Unsupported(f"{node.text()} lies over other axes than the values before it, which runs on NumPy")
            self.cells = layout
        return cells

    def integers(self, node: Ref, context: Context) -> Source:
        """A field of times or durations as the int64 counts it is held in (``AsIntegers``), for a key to place."""
        cells = self._layout(node, evaluate(node, context))
        for local, (read_, _) in self.inputs.items():
            if isinstance(read_, AsIntegers) and (read_.ref.name, read_.ref.kind) == (node.name, node.kind):
                break
        else:
            local = f"a{len(self.inputs)}"
            self.inputs[local] = (AsIntegers(node), cells)
        return Source(f"{local}[i, j]" if cells else f"{local}[i]", np.dtype(np.int64), cells)

    def _island(self, node: Any, context: Context) -> Source:
        """*node* computed by NumPy for each block (``Island``), the kernel reading its values: the same island once,
        however often it is read."""
        value = evaluate(node, context)
        dtype = value_type(value)
        if dtype.kind not in "biuf" or dtype == np.float16:
            raise Unsupported(f"{node.text()} gives {kind_words(dtype)}, which runs on NumPy")
        cells = self._layout(node, value)
        steps = self._steps(node, context)
        operands, function = self._operands(node, context)
        from tidy3a.identity import structure

        key = (structure(node), tuple(id(step) for step in steps))
        for local, (read, _) in self.inputs.items():
            if isinstance(read, Island) and (structure(read.node), tuple(id(s) for s in read.steps)) == key:
                break
        else:
            local = f"a{len(self.inputs)}"
            self.inputs[local] = (Island(node, steps, operands, function, dtype), cells)
        return Source(f"{local}[i, j]" if cells else f"{local}[i]", dtype, cells)

    def _operands(self, node: Any, context: Context) -> tuple[tuple, Any]:
        """*node*'s operands compiled, for a kernel to compute and NumPy to apply its function to, and that function;
        or none, where an operand does not compile (NumPy then evaluates the whole island)."""
        if isinstance(node, Call):
            key, children = node.name, list(node.args)
            if node.options:
                return (), None
        elif isinstance(node, BinOp):
            key, children = node.symbol, [node.left, node.right]
        else:
            return (), None
        function = APPLIED.get(key)
        if function is None:
            return (), None
        saved = dict(self.inputs), self.cells
        operands: list = []
        try:
            for child in children:
                if isinstance(child, Literal) and isinstance(child.value, (bool, int, float, np.number)):
                    operands.append(child.value)  # a Python number stays one, as NumPy takes it beside an array
                else:
                    operands.append(self.compile(child, context))
        except Unsupported:
            self.inputs, self.cells = saved[0], saved[1]
            return (), None
        if not any(isinstance(operand, Source) for operand in operands):
            return (), None
        return tuple(operands), function

    def _steps(self, node: Any, context: Context) -> tuple:
        """The ``mutate`` steps *node* reads, directly or through the definitions it reads, in their order."""
        found: dict[int, Any] = {}

        def visit(expression: Any, definitions: dict, context: Context) -> None:
            lower = getattr(expression, "lower", None)
            if lower is not None:  # a formula: its names are in what it lowers to on the data
                expression = lower(context)[0]
            if isinstance(expression, Ref):
                if expression.kind != "label" and definitions.get(expression.name) is not None:
                    inner, defined, earlier, step, place = definitions[expression.name]
                    if place not in found:
                        found[place] = step
                        visit(inner, earlier, defined)
                return
            for child in getattr(expression, "children", lambda: ())():
                visit(child, definitions, context)

        visit(node, self.definitions, context)
        if any(step is None for step in found.values()):
            raise Unsupported(f"{node.text()} reads a variable whose step cannot be run again, which runs on NumPy")
        return tuple(found[place] for place in sorted(found))

    def _binary(self, node: BinOp, context: Context, result: np.dtype) -> Source:
        left, right = self.compile(node.left, context), self.compile(node.right, context)
        cells = left.cells or right.cells
        if node.symbol == "**":
            return _power(left, node.right, result, cells)
        if node.symbol in DIVISIONS:  # NumPy's floored division and remainder, step by step
            kind = "i" if result.kind in "iub" else f"f{result.itemsize * 8}"
            if kind not in ("i", "f32", "f64"):
                raise Unsupported(f"{node.text()} gives {kind_words(result)}, which runs on NumPy")
            helper = {"//": "div", "%": "mod"}[node.symbol]
            name = f"_i{helper}" if kind == "i" else f"_f{helper}{kind[1:]}"
            return Source(_as(f"{name}({cast(left, result)}, {cast(right, result)})", result), result, cells)
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
        if node.name in ("floor", "ceil"):
            if result.kind in "iub":  # NumPy gives an integer its own value
                return Source(_as(cast(args[0], result), result), result, cells)
            return Source(_as(f"np.{node.name}({cast(args[0], result)})", result), result, cells)
        if node.name == "sign":  # NumPy's: 1, -1, or 0 for either zero; a NaN stays one
            value = cast(args[0], result)
            one, zero = _number(1, result), _number(0, result)
            text = (f"({one} if {value} > {zero} else (-{one} if {value} < {zero} else "
                    f"({zero} if {value} == {zero} else {value})))")
            return Source(_as(text, result), result, cells)
        if node.name == "if_else":
            condition, true, false = args
            text = f"({cast(true, result)} if {cast(condition, np.dtype(bool))} else {cast(false, result)})"
            return Source(_as(text, result), result, cells)
        raise Unsupported(f"{node.name}() runs on NumPy")


def _exact_power(node: BinOp, context: Context) -> bool:
    """Whether ``_power`` takes *node*: an exponent NumPy computes exactly for its base's type."""
    value = node.right.value if isinstance(node.right, Literal) else None
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        return False
    base = value_type(evaluate(node.left, context))
    exact = {2.0, 1.0} if base.kind in "iu" else {2.0, 0.5, -1.0, 1.0}
    return float(value) in exact


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
        return Source(_float_text(float(value)), None if isinstance(value, float) else np.dtype(type(value)),
                      value=value)
    raise Unsupported(f"{node.text()} is not a number, which runs on NumPy")


def _float_text(value: float) -> str:
    """A float as kernel source: NaN and the infinities by name, as ``repr`` gives no Python they are."""
    if value != value:
        return "np.nan"
    if value in (np.inf, -np.inf):
        return "np.inf" if value > 0 else "-np.inf"
    return repr(value)


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
    return f"{_type(dtype)}({_float_text(float(value)) if isinstance(value, float) else repr(value)})"


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
