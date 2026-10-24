"""``formula()``: mathematical text and a strict LaTeX dialect (plan §6.3).

A formula is parsed once, then compiled against the data when a verb uses
it, into the same expressions as Python syntax: ``formula("x^2")`` becomes
``x ** 2``, and ``\\sum_i p_i^2`` becomes ``sum(p ** 2, over="i")``.

LaTeX does not define meaning, so the conventions are fixed and follow
plot3's LaTeX input (``plot3/latexin.py``): ``\\sin x^2`` is ``sin(x**2)``,
``\\sin^2 x`` is ``(sin x)**2``, ``\\sin^{-1}`` is ``arcsin``, ``e^{x}`` is
``exp(x)`` unless ``e`` is a defined name, ``\\log`` takes only base 10, and
letters written side by side multiply. For arrays, subscripts are indices:
``p_i`` is ``p`` along axis ``i``, ``p_x`` is the label ``x``, and ``p_1`` is
position 1. Index letters bind to axes by name, or through ``bind=``.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

import tidy3a.linalg  # noqa: F401  registers sqrt, exp, … in FUNCTIONS
from tidy3a.errors import RefError, TidyError
from tidy3a.expr import (
    FUNCTIONS,
    BinOp,
    Call,
    Context,
    Expr,
    Literal,
    Reduce,
    Ref,
    UnaryOp,
    as_expr,
    evaluate,
)


class FormulaError(TidyError):
    """A formula cannot be parsed, or its names and indices cannot be bound."""


_EATEN = (
    "This looks like LaTeX with backslashes eaten by Python. "
    'Use a raw string: formula(r"\\frac{1}{x}").'
)
# Python already turned "\f", "\a", "\b", "\v" into control characters (as in plot3).
_CONTROL = set("\a\b\f\v")

_GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
}
# LaTeX command or plain-text name -> tidy3A function
_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan",
    "ln": "log", "log": "log", "exp": "exp",
}
_PLAIN_FUNCS = {**_FUNCS, "sqrt": "sqrt", "abs": "abs", "log10": "log10", "log2": "log2"}
_INVERSE = {"sin": "arcsin", "cos": "arccos", "tan": "arctan"}
_SPACING = {",", ";", ":", "!", " ", "quad", "qquad", "thinspace", "medspace", "thickspace"}
_VALUE_CMDS = {"frac", "dfrac", "tfrac", "sqrt", "left", "mathrm", "operatorname", "text", "pi", "sum"}


# ── Syntax tree ─────────────────────────────────────────────────────────────


@dataclass
class Num:
    value: float | int


@dataclass
class Sym:
    name: str
    upright: bool = False  # written \mathrm{...}


@dataclass
class Sub:
    """One subscript item: an index letter, a 1-based position, or a label."""

    kind: str  # "letter" | "position" | "label"
    value: Any


@dataclass
class Indexed:
    base: Any
    subs: list[Sub]


@dataclass
class Bin:
    op: str
    left: Any
    right: Any
    implicit: bool = False


@dataclass
class Neg:
    operand: Any


@dataclass
class Pow:
    base: Any
    exponent: Any


@dataclass
class Func:
    name: str
    arg: Any


@dataclass
class Sum:
    letter: str
    lower: Any
    upper: Any
    body: Any


@dataclass
class Const:
    name: str  # "pi" | "e" | "i"


@dataclass
class Parsed:
    body: Any
    target: Any = None
    sum_letters: set[str] = field(default_factory=set)


# ── Tokens ──────────────────────────────────────────────────────────────────


def _tokens(source: str, latex: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch == "\\":
            j = i + 1
            if j < len(source) and source[j].isalpha():
                while j < len(source) and source[j].isalpha():
                    j += 1
                name = source[i + 1 : j]
            else:
                name = source[j : j + 1]
                j += 1
            if name not in _SPACING:
                out.append(("cmd", name))
            i = j
        elif ch.isdigit() or (ch == "." and i + 1 < len(source) and source[i + 1].isdigit()):
            j = i
            while j < len(source) and (source[j].isdigit() or source[j] == "."):
                j += 1
            out.append(("num", source[i:j]))
            i = j
        elif ch.isalpha() or (not latex and ch == "_"):
            if latex:
                out.append(("letter", ch))
                i += 1
            else:
                j = i
                while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                    j += 1
                out.append(("name", source[i:j]))
                i = j
        elif source.startswith("**", i):
            out.append(("op", "^"))
            i += 2
        elif ch in "+-*/^_{}()[]|,=":
            out.append(("op", ch))
            i += 1
        else:
            raise FormulaError(f"formula: cannot read {ch!r}")
    return out


# ── Parser ──────────────────────────────────────────────────────────────────


class _Parser:
    def __init__(self, source: str, latex: bool):
        self.tokens = _tokens(source, latex)
        self.index = 0
        self.latex = latex
        self.bar_depth = 0
        self.sum_letters: set[str] = set()

    def parse(self) -> Parsed:
        body = self.expression()
        target = None
        if self.eat("op", "="):
            target, body = body, self.expression()
        if self.peek() is not None:
            self.bad(self.peek())
        return Parsed(body, target, self.sum_letters)

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def eat(self, kind: str, value: str | None = None) -> bool:
        token = self.peek()
        if token is not None and token[0] == kind and (value is None or token[1] == value):
            self.index += 1
            return True
        return False

    def expect(self, kind: str, value: str) -> None:
        if not self.eat(kind, value):
            token = self.peek()
            if token is None:
                raise FormulaError(f"formula: unexpected end, expected {value!r}")
            self.bad(token)

    def bad(self, token: tuple[str, str]) -> None:
        kind, value = token
        if kind == "cmd":
            raise FormulaError(f"\\{value} isn't supported in formula")
        raise FormulaError(f"formula: syntax error at {value!r}")

    def starts_factor(self, *, functions: bool) -> bool:
        if not self.latex:
            return False
        token = self.peek()
        if token is None:
            return False
        kind, value = token
        if kind in ("num", "letter"):
            return True
        if kind == "op":
            return value in "({" or (value == "|" and self.bar_depth == 0)
        if kind != "cmd":
            return False
        if value in _FUNCS:
            return functions
        return value in _VALUE_CMDS or value in _GREEK

    def expression(self) -> Any:
        node = self.term()
        while True:
            if self.eat("op", "+"):
                node = Bin("+", node, self.term())
            elif self.eat("op", "-"):
                node = Bin("-", node, self.term())
            else:
                return node

    def term(self, *, functions: bool = True) -> Any:
        node = self.unary()
        while True:
            if self.eat("op", "*") or self.eat("cmd", "cdot") or self.eat("cmd", "times"):
                node = Bin("*", node, self.unary())
            elif self.eat("op", "/"):
                node = Bin("/", node, self.unary())
            elif self.starts_factor(functions=functions):
                node = Bin("*", node, self.unary(), implicit=True)
            else:
                return node

    def unary(self) -> Any:
        if self.eat("op", "+"):
            return self.unary()
        if self.eat("op", "-"):
            return Neg(self.unary())
        return self.power()

    def power(self) -> Any:
        base = self.postfix()
        if not self.eat("op", "^"):
            return base
        exponent = self.script()
        if self.peek() == ("op", "_"):
            raise FormulaError("formula: write subscripts before superscripts, e.g. p_i^2")
        if isinstance(base, Sym) and base.name == "e" and not base.upright:
            return Pow(Sym("e"), exponent)  # resolved to exp() when e is not a defined name
        return Pow(base, exponent)

    def script(self) -> Any:
        if self.eat("op", "{"):
            inner = self.expression()
            self.expect("op", "}")
            return inner
        if self.latex:
            return self.unary()
        return self.unary()

    def postfix(self) -> Any:
        atom = self.primary()
        if self.latex and self.eat("op", "_"):
            return Indexed(atom, self.subscripts())
        return atom

    def subscripts(self) -> list[Sub]:
        items: list[Sub] = []
        if self.eat("op", "{"):
            while not self.eat("op", "}"):
                if self.peek() is None:
                    raise FormulaError("formula: unclosed subscript")
                if self.eat("op", ","):
                    continue
                items.append(self.one_subscript())
        else:
            items.append(self.one_subscript())
        return items

    def one_subscript(self) -> Sub:
        token = self.peek()
        if token is None:
            raise FormulaError("formula: missing subscript")
        kind, value = token
        self.index += 1
        if kind == "letter":
            return Sub("letter", value)
        if kind == "num":
            return Sub("position", int(value))
        if kind == "cmd" and value in ("mathrm", "text", "operatorname"):
            return Sub("label", self.upright_name())
        if kind == "cmd" and value in _GREEK:
            return Sub("letter", value)
        self.index -= 1
        self.bad(token)
        raise AssertionError

    def upright_name(self) -> str:
        self.expect("op", "{")
        parts: list[str] = []
        while not self.eat("op", "}"):
            token = self.peek()
            if token is None:
                raise FormulaError("formula: unclosed \\mathrm{...}")
            self.index += 1
            kind, value = token
            if kind == "cmd" and value == "_":
                parts.append("_")
            elif kind in ("letter", "num", "name"):
                parts.append(value)
            else:
                raise FormulaError(f"formula: {value!r} cannot be part of a name in \\mathrm{{...}}")
        return "".join(parts)

    def group(self) -> Any:
        self.expect("op", "{")
        inner = self.expression()
        self.expect("op", "}")
        return inner

    def primary(self) -> Any:
        token = self.peek()
        if token is None:
            raise FormulaError("formula: unexpected end")
        kind, value = token
        if kind == "num":
            self.index += 1
            return Num(float(value) if "." in value else int(value))
        if kind == "letter":
            self.index += 1
            return Sym(value)
        if kind == "name":
            self.index += 1
            if value in _PLAIN_FUNCS and self.peek() == ("op", "("):
                self.index += 1
                arg = self.expression()
                self.expect("op", ")")
                return Func(_PLAIN_FUNCS[value], arg)
            return Sym(value)
        if kind == "op":
            if value == "(":
                self.index += 1
                inner = self.expression()
                self.expect("op", ")")
                return inner
            if value == "{" and self.latex:
                return self.group()
            if value == "|":
                self.index += 1
                self.bar_depth += 1
                inner = self.expression()
                self.bar_depth -= 1
                self.expect("op", "|")
                return Func("abs", inner)
            self.bad(token)
        assert kind == "cmd"
        self.index += 1
        if value in ("frac", "dfrac", "tfrac"):
            return Bin("/", self.group(), self.group())
        if value == "sqrt":
            if self.eat("op", "["):
                degree = self.expression()
                self.expect("op", "]")
                return Pow(self.group(), Bin("/", Num(1), degree))
            return Func("sqrt", self.group())
        if value in ("mathrm", "operatorname", "text"):
            name = self.upright_name()
            if name in _FUNCS and value == "operatorname":
                return self.apply(name)
            return Sym(name, upright=True)
        if value == "pi":
            return Const("pi")
        if value in _GREEK:
            return Sym(value)
        if value in _FUNCS:
            return self.apply(value)
        if value == "sum":
            return self.summation()
        if value == "left":
            opener = self.peek()
            self.index += 1
            if opener == ("op", "("):
                inner = self.expression()
                self.expect("cmd", "right")
                self.expect("op", ")")
                return inner
            if opener == ("op", "|"):
                self.bar_depth += 1
                inner = self.expression()
                self.bar_depth -= 1
                self.expect("cmd", "right")
                self.expect("op", "|")
                return Func("abs", inner)
            raise FormulaError("formula: \\left must be followed by ( or |")
        self.index -= 1
        self.bad(token)
        raise AssertionError

    def apply(self, command: str) -> Any:
        name = _FUNCS[command]
        power = None
        for _ in range(2):
            if power is None and self.eat("op", "^"):
                exponent = self.script()
                if isinstance(exponent, Neg) and exponent.operand == Num(1) and name in _INVERSE:
                    name = _INVERSE[name]
                else:
                    power = exponent
            elif command == "log" and self.eat("op", "_"):
                base = self.script()
                if base != Num(10):
                    raise FormulaError("\\log with a subscript other than 10 isn't supported in formula")
                name = "log10"
            else:
                break
        if self.peek() == ("op", "{"):
            arg = self.group()
        elif self.peek() in (("op", "-"), ("op", "+")) or self.starts_factor(functions=True):
            arg = self.unary()
            while self.starts_factor(functions=False):
                arg = Bin("*", arg, self.unary(), implicit=True)
        else:
            raise FormulaError(f"formula: \\{command} needs an argument")
        call = Func(name, arg)
        return Pow(call, power) if power is not None else call

    def summation(self) -> Sum:
        if not self.eat("op", "_"):
            raise FormulaError("formula: \\sum needs an index, e.g. \\sum_i")
        lower = None
        if self.eat("op", "{"):
            letter = self.index_letter()
            if self.eat("op", "="):
                lower = self.expression()
            self.expect("op", "}")
        else:
            letter = self.index_letter()
        upper = None
        if self.eat("op", "^"):
            upper = self.script()
        if (lower is None) != (upper is None):
            raise FormulaError(f"formula: \\sum_{letter} needs both bounds or neither")
        self.sum_letters.add(letter)
        body = self.term(functions=True)
        return Sum(letter, lower, upper, body)

    def index_letter(self) -> str:
        token = self.peek()
        if token is not None and (token[0] == "letter" or (token[0] == "cmd" and token[1] in _GREEK)):
            self.index += 1
            return token[1]
        raise FormulaError("formula: a sum index must be one letter, e.g. \\sum_i")


def is_latex(text: str) -> bool:
    """True for ``$...$`` or any backslash command, as in plot3."""
    body = text.strip()
    return "\\" in body or (len(body) >= 2 and body[0] == "$" and body[-1] == "$")


def parse(source: str) -> tuple[Parsed, bool]:
    if any(ch in _CONTROL for ch in source):
        raise FormulaError(_EATEN)
    latex = is_latex(source)
    body = source.strip()
    if latex and len(body) >= 2 and body[0] == "$" and body[-1] == "$":
        body = body[1:-1]
    return _Parser(body, latex).parse(), latex


# ── The formula expression ──────────────────────────────────────────────────


_PY_VALUE_TYPES = (int, float, complex, np.number, np.ndarray, xr.DataArray)


class Formula(Expr):
    """A parsed formula; compiled against the data when a verb evaluates it."""

    def __init__(self, source: str, bind: Mapping[str, Any] | None, scope: Mapping[str, Any]):
        self.source = source
        self.parsed, self.latex = parse(source)
        self.bind = dict(bind or {})
        self.scope = scope
        target = self.parsed.target
        if target is not None:
            base = target.base if isinstance(target, Indexed) else target
            if not isinstance(base, Sym):
                raise FormulaError("formula: the left side of = must be a name, e.g. p_i = …")
            self.target: str | None = base.name
            self.target_letters = [s.value for s in target.subs if s.kind == "letter"] if isinstance(target, Indexed) else []
        else:
            self.target = None
            self.target_letters = []

    def text(self) -> str:
        return f'formula(r"{self.source}")'

    def lower(self, ctx: Context) -> tuple[Expr, list[str]]:
        compiler = _Compiler(self, ctx)
        expr = compiler.compile(self.parsed.body)
        for letter in self.target_letters:
            if letter not in compiler.axes:
                raise FormulaError(
                    f"formula: the left side indexes {letter}, but no subscript on the right uses it"
                )
            if letter in self.parsed.sum_letters:
                raise FormulaError(f"formula: {letter} is summed on the right, so it cannot index the left side")
        return expr, compiler.log

    def explain(self) -> str:
        """The formula as parsed, with its index letters."""
        letters = sorted(self.parsed.sum_letters)
        lines = [
            f"formula: {self.source}",
            f"  parsed: {_render(self.parsed.body)}",
        ]
        if self.target is not None:
            lines.append(f"  target: {self.target}")
        if letters:
            lines.append("  summed index letters: " + ", ".join(letters))
        for key, value in self.bind.items():
            lines.append(f"  bind: {key} → {value!r}" if isinstance(value, str) else f"  bind: {key} → value")
        lines.append("  bindings to axes and data are shown once it is used in a pipeline")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.explain()

    def _repr_latex_(self) -> str:
        return f"${self.source.strip().strip('$')}$" if self.latex else f"$\\mathrm{{{self.source}}}$"


def formula(source: str, *, bind: Mapping[str, Any] | None = None) -> Formula:
    """Mathematics as text: ``formula("x^2")`` or ``formula(r"\\sqrt{\\sum_i p_i^2}")``.

    ``bind`` maps index letters to axis names (``{"j": "feature"}``) and other
    names to values (``{"v": [x, y, z], "n": 3}``). Other names resolve to
    the data or to Python values in the calling code; a name that is both is
    an error.
    """
    if not isinstance(source, str):
        raise TypeError("formula() takes a string of mathematics")
    caller = sys._getframe(1)
    scope = {**caller.f_globals, **caller.f_locals}
    return Formula(source, bind, scope)


def _render(node: Any) -> str:
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Sym):
        return node.name
    if isinstance(node, Const):
        return {"pi": "π", "e": "e", "i": "i"}[node.name]
    if isinstance(node, Indexed):
        subs = ", ".join(str(s.value) for s in node.subs)
        return f"{_render(node.base)}[{subs}]"
    if isinstance(node, Bin):
        symbol = "·" if node.implicit else node.op
        return f"({_render(node.left)} {symbol} {_render(node.right)})"
    if isinstance(node, Neg):
        return f"-{_render(node.operand)}"
    if isinstance(node, Pow):
        return f"{_render(node.base)} ^ {_render(node.exponent)}"
    if isinstance(node, Func):
        return f"{node.name}({_render(node.arg)})"
    if isinstance(node, Sum):
        bounds = "" if node.lower is None else f" from {_render(node.lower)} to {_render(node.upper)}"
        return f"Σ_{node.letter}{bounds} {_render(node.body)}"
    return repr(node)


# ── Compiling against data ──────────────────────────────────────────────────


class _Compiler:
    def __init__(self, formula: Formula, ctx: Context):
        self.formula = formula
        self.ctx = ctx
        self.bind = formula.bind
        self.scope = formula.scope
        self.log: list[str] = []
        self.bounds: dict[str, tuple[int, int]] = {}
        self.axes: dict[str, Hashable] = {}
        self.index_letters = set(formula.parsed.sum_letters) | set(formula.target_letters)
        for key, value in self.bind.items():
            if key in self.index_letters and not isinstance(value, str):
                raise FormulaError(f'bind: {key} is an index letter, so bind it to an axis name, e.g. {{"{key}": "feature"}}')

    def note(self, line: str) -> None:
        if line not in self.log:
            self.log.append(line)

    def compile(self, node: Any) -> Expr:
        if isinstance(node, Num):
            return Literal(node.value)
        if isinstance(node, Const):
            return self.constant(node.name)
        if isinstance(node, Sym):
            return self.name(node)
        if isinstance(node, Indexed):
            return self.indexed(node)
        if isinstance(node, Bin):
            if node.implicit:
                self.check_run(node)
            return BinOp(node.op, self.compile(node.left), self.compile(node.right))
        if isinstance(node, Neg):
            return UnaryOp("-", self.compile(node.operand))
        if isinstance(node, Pow):
            if isinstance(node.base, Sym) and node.base.name == "e" and not node.base.upright:
                if self.lookup("e") is None:
                    return Call("exp", (self.compile(node.exponent),))
            return BinOp("**", self.compile(node.base), self.compile(node.exponent))
        if isinstance(node, Func):
            return Call(node.name, (self.compile(node.arg),))
        if isinstance(node, Sum):
            return self.summation(node)
        raise FormulaError(f"formula: cannot compile {node!r}")

    # names ------------------------------------------------------------------

    def constant(self, name: str) -> Literal:
        value = {"pi": np.pi, "e": np.e, "i": 1j}[name]
        return Literal(value, label={"pi": "π", "e": "e", "i": "1j"}[name])

    def is_data(self, name: str) -> bool:
        data = self.ctx.data
        field_axis = self.ctx.field_axis
        return (
            name in data.data_vars
            or name in data.dims
            or (field_axis is not None and name in data.indexes[field_axis])
        )

    def lookup(self, name: str) -> Expr | None:
        """Resolve a name to data, a bound value, or a Python value; None if unknown."""
        if name in self.bind:
            value = self.bind[name]
            return as_expr(value) if not isinstance(value, Expr) else value
        python = self.scope.get(name)
        if isinstance(python, Ref):
            return Ref(python.name, python.kind)
        python_value = python if isinstance(python, _PY_VALUE_TYPES) and not isinstance(python, bool) else None
        if self.is_data(name):
            if python_value is not None:
                raise RefError(
                    f'formula: "{name}" is both data and a Python value ({_short(python_value)}).\n'
                    f'  data:    bind={{"{name}": col("{name}")}}\n'
                    f'  Python:  bind={{"{name}": {name}}}'
                )
            return Ref(name)
        if python_value is not None:
            return Literal(python_value, label=name)
        return None

    def name(self, node: Sym) -> Expr:
        if node.upright and node.name in ("i", "j"):
            self.note(f"\\mathrm{{{node.name}}} → imaginary unit")
            return Literal(1j, label="1j")
        if node.upright and node.name == "e":
            return Literal(np.e, label="e")
        found = self.lookup(node.name)
        if found is not None:
            self.describe_name(node.name, found)
            return found
        if node.name == "e":
            return Literal(np.e, label="e")
        if not self.formula.latex and node.name == "pi":
            return Literal(np.pi, label="π")
        raise RefError(
            f'formula: "{node.name}" is not data, a bound value, or a Python value here.'
            + _inventory(self.ctx)
        )

    def describe_name(self, name: str, expr: Expr) -> None:
        if isinstance(expr, Ref):
            try:
                value = evaluate(expr, self.ctx)
            except TidyError:
                return
            if isinstance(value, xr.DataArray):
                self.note(f"{name} → data {name} ({', '.join(map(str, value.dims))})")
        elif name in self.bind:
            self.note(f"{name} → bound value")
        else:
            self.note(f"{name} → Python value {name}")

    def check_run(self, node: Bin) -> None:
        letters = _letter_run(node)
        if letters is None or len(letters) < 2:
            return
        joined = "".join(letters)
        if self.is_data(joined) or joined in self.bind or isinstance(self.scope.get(joined), (*_PY_VALUE_TYPES, Ref)):
            product = " \\cdot ".join(letters)  # Python 3.11 allows no backslash inside f-string braces
            raise FormulaError(
                f'formula: {joined} is read as {" · ".join(letters)}, but "{joined}" is also a name.\n'
                f"  For the name write \\mathrm{{{joined}}}; for the product write {product}"
            )
        self.note(f"{joined} → {' · '.join(letters)} (letters side by side multiply)")

    # indices ----------------------------------------------------------------

    def axis_for(self, letter: str, dims: tuple[Hashable, ...], base: str) -> Hashable | None:
        """The axis an index letter names, or None when the letter is a label."""
        if letter in self.bind and isinstance(self.bind[letter], str):
            axis = self.bind[letter]
        elif letter in self.index_letters or letter in dims:
            axis = letter
        else:
            return None
        if axis not in dims:
            raise FormulaError(
                f'formula: index {letter} names axis "{axis}", which {base} does not have '
                f"(axes: {', '.join(map(str, dims))}).\n"
                f'  Bind it to one of them: bind={{"{letter}": "{dims[-1] if dims else "axis"}"}}'
            )
        previous = self.axes.setdefault(letter, axis)
        if previous != axis:
            raise FormulaError(f"formula: index {letter} names both {previous} and {axis}")
        return axis

    def indexed(self, node: Indexed) -> Expr:
        if not isinstance(node.base, Sym):
            raise FormulaError("formula: only names can take subscripts, e.g. p_i")
        base_name = node.base.name
        expr = self.name(node.base)
        value = evaluate(expr, self.ctx)
        if not isinstance(value, xr.DataArray):
            raise FormulaError(f"formula: {base_name}_… needs an array; {base_name} has no axes")
        dims = value.dims
        for sub in node.subs:
            if sub.kind == "letter":
                axis = self.axis_for(sub.value, dims, base_name)
                if axis is not None:
                    self.describe_index(sub.value, axis, base_name, value)
                    if sub.value in self.bounds:
                        lower, upper = self.bounds[sub.value]
                        expr = Call("positions", (expr,), {"axis": axis, "lower": lower, "upper": upper})
                    continue
                expr, value = self.label(expr, value, sub.value, base_name)
            elif sub.kind == "label":
                expr, value = self.label(expr, value, sub.value, base_name)
            else:
                expr, value = self.position(expr, value, sub.value, base_name)
        return expr

    def describe_index(self, letter: str, axis: Hashable, base: str, value: xr.DataArray) -> None:
        labels = ""
        if axis in value.indexes:
            shown = "  ".join(str(v) for v in list(value.indexes[axis])[:6])
            labels = f" (labels {shown})"
        how = "summed" if letter in self.formula.parsed.sum_letters else "free"
        bound = " (bound)" if letter in self.bind else ""
        if letter in self.bounds:
            lower, upper = self.bounds[letter]
            how += f", positions {lower}–{upper} of {value.sizes[axis]}"
        self.note(f'{letter} → axis "{axis}" of {base}{labels}{bound}, {how}')

    def label(self, expr: Expr, value: xr.DataArray, name: str, base: str) -> tuple[Expr, xr.DataArray]:
        axes = [d for d in value.dims if d in value.indexes and name in value.indexes[d]]
        if len(axes) != 1:
            listed = ", ".join(map(str, value.dims))
            where = "no axis" if not axes else f"several axes ({', '.join(map(str, axes))})"
            raise FormulaError(
                f"formula: {base}_{name}: {name} is not an axis of {base} and is a label on {where} "
                f"(axes: {listed}). Use an axis name, a label, or bind={{\"{name}\": \"axis\"}}"
            )
        axis = axes[0]
        self.note(f'{name} → label "{name}" on axis "{axis}" of {base}')
        return Call("pick", (expr,), {"axis": axis, "label": name}), value.sel({axis: name}, drop=True)

    def position(self, expr: Expr, value: xr.DataArray, position: int, base: str) -> tuple[Expr, xr.DataArray]:
        axis = self.component_axis(value, base)
        if not 1 <= position <= value.sizes[axis]:
            raise FormulaError(f"formula: {base}_{position}: {axis} has positions 1 … {value.sizes[axis]}")
        self.note(f'{position} → position {position} on axis "{axis}" of {base}')
        return (
            Call("position", (expr,), {"axis": axis, "position": position}),
            value.isel({axis: position - 1}, drop=True),
        )

    def component_axis(self, value: xr.DataArray, base: str) -> Hashable:
        """The one axis a numeric subscript can mean: the field axis, or an axis no other variable has."""
        data = self.ctx.data
        candidates = []
        for dim in value.dims:
            if dim == self.ctx.field_axis:
                candidates.append(dim)
                continue
            others = [v for v in data.data_vars if v != base and dim in data[v].dims]
            if dim not in data.dims or not others:
                candidates.append(dim)
        if len(candidates) != 1 and value.ndim == 1:
            candidates = list(value.dims)
        if len(candidates) != 1:
            raise FormulaError(
                f"formula: {base}_<number> is ambiguous on {base} ({', '.join(map(str, value.dims))}); "
                f"use a label ({base}_x) or an index letter"
            )
        return candidates[0]

    def summation(self, node: Sum) -> Expr:
        letter = node.letter
        if node.lower is not None:
            lower, upper = self.bound(node.lower, letter), self.bound(node.upper, letter)
            self.bounds[letter] = (lower, upper)
        body = self.compile(node.body)
        self.bounds.pop(letter, None)
        axis = self.axes.get(letter)
        if axis is None:
            raise FormulaError(
                f"formula: \\sum_{letter} sums over {letter}, but no subscript in its body uses {letter}"
            )
        return Reduce("sum", body, over=axis)

    def bound(self, node: Any, letter: str) -> int:
        if isinstance(node, Num) and isinstance(node.value, int):
            return node.value
        if isinstance(node, Sym):
            value = self.bind.get(node.name, self.scope.get(node.name))
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                return int(value)
            raise FormulaError(
                f"formula: the bound {node.name} of \\sum_{letter} is not bound to an integer; "
                f'give it with bind={{"{node.name}": …}}, or write \\sum_{letter} for the whole axis'
            )
        raise FormulaError(f"formula: bounds of \\sum_{letter} must be integers or bound names")


def _letter_run(node: Any) -> list[str] | None:
    if isinstance(node, Sym) and not node.upright and len(node.name) == 1:
        return [node.name]
    if isinstance(node, Bin) and node.implicit:
        left, right = _letter_run(node.left), _letter_run(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _short(value: Any) -> str:
    if isinstance(value, xr.DataArray):
        return f"an array ({', '.join(map(str, value.dims))})"
    if isinstance(value, np.ndarray):
        return "a NumPy array"
    return repr(value)


def _inventory(ctx: Context) -> str:
    data = ctx.data
    text = "\n  variables: " + (", ".join(map(str, data.data_vars)) or "none")
    if ctx.field_axis is not None:
        text += f"\n  labels on {ctx.field_axis}: " + "  ".join(map(str, data.indexes[ctx.field_axis]))
    return text + "\n  Supply other names with bind={...}."


# ── Evaluation helpers registered with the expression system ────────────────


def _positions(node: Call, ctx: Context) -> xr.DataArray:
    value = evaluate(node.args[0], ctx)
    axis, lower, upper = node.options["axis"], node.options["lower"], node.options["upper"]
    size = value.sizes[axis]
    if not 1 <= lower <= upper <= size:
        raise FormulaError(f"formula: bounds {lower} … {upper} are outside {axis}, which has positions 1 … {size}")
    return value.isel({axis: slice(lower - 1, upper)})


def _pick(node: Call, ctx: Context) -> xr.DataArray:
    value = evaluate(node.args[0], ctx)
    return value.sel({node.options["axis"]: node.options["label"]}, drop=True)


def _position(node: Call, ctx: Context) -> xr.DataArray:
    value = evaluate(node.args[0], ctx)
    return value.isel({node.options["axis"]: node.options["position"] - 1}, drop=True)


FUNCTIONS.update({"positions": _positions, "pick": _pick, "position": _position})
