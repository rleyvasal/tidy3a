"""Fused summary kernels (plan milestone 16): a block's whole summary in one compiled pass, where it can be.

A summary of data read in blocks runs, on each block, its row-wise verbs, its
keys' codes, and each reduction's first pass, a NumPy step at a time, each
streaming the block through memory. Where every one of those can be compiled,
one Numba pass a block does them all, reading each value once and writing no
array, and the block's table joins the merge tree as before.

``plan_for(frame)`` says whether a frame's summary compiles and, where not,
which step does not and why, from the verbs run on the block of no positions,
so every step's type is NumPy's own and nothing is read. Compilation needs
Numba (``tidy3a[fast]``) and is on unless ``TIDY3A_COMPILED=0``; this module
only asks whether Numba is installed, so importing tidy3a loads no Numba.

What compiles (the owner's decisions, milestone 16): ``+ - * /``, ``**`` with
an exponent NumPy computes exactly (2, 0.5, -1, 1), comparisons, ``& | ~`` on
booleans, ``sqrt``, ``abs``, ``floor``, and ``if_else``, on booleans,
integers, and float32 and float64; keys of integer, boolean, and float fields
(a float key's values checked, a block at a time, to be whole) and their
``cells``, ``bins``, and ``rounded``; and the built-in reductions. The rest,
``exp``, ``log``, and the trigonometric functions among it, runs on NumPy.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from tidy3a.errors import FilterError, MutateError, SummariseError, TidyError
from tidy3a.expr import BinOp, Call, Context, Literal, Ref, UnaryOp, evaluate

OPERATORS = frozenset({"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=", "&", "|"})
POWERS = frozenset({2.0, 0.5, -1.0, 1.0})  # the exponents NumPy computes exactly: x*x, sqrt, 1/x, x
SIGNS = frozenset({"-", "+", "~"})
FUNCTIONS = frozenset({"sqrt", "abs", "floor", "if_else"})
REDUCTIONS = frozenset({"n", "sum", "mean", "min", "max", "std", "var", "any", "all", "median", "quantile"})
KINDS = "biuf"  # booleans, integers, and floats; float16 (half) is left out below


@dataclass(frozen=True)
class Fusion:
    """Whether a frame's summary compiles: no *reasons*, or each step that does not, and why."""

    reasons: tuple[str, ...] = ()

    @property
    def compiled(self) -> bool:
        return not self.reasons

    def words(self) -> str:
        """How a run is planned to go: a compiled kernel a block, or NumPy and the first reason."""
        if self.compiled:
            return "a compiled kernel a block"
        more = len(self.reasons) - 1
        return f"NumPy, as {self.reasons[0]}{f' (and {more} more: explain() lists them)' if more else ''}"

    def lines(self) -> list[str]:
        """Every reason, for ``explain()``."""
        return [f"planned run: {self.words()}"] if self.compiled or len(self.reasons) == 1 else [
            "planned run: NumPy, as", *(f"  {reason}" for reason in self.reasons)]


def unavailable() -> str | None:
    """Why no block compiles whatever its verbs, or None: turned off, or no Numba."""
    if os.environ.get("TIDY3A_COMPILED", "1").strip() == "0":
        return "compiling is turned off (TIDY3A_COMPILED=0)"
    if importlib.util.find_spec("numba") is None:
        return 'Numba is not installed (pip install "tidy3a[fast]")'
    return None


def plan_for(frame: Any) -> Fusion:
    """Whether *frame*'s summary compiles, and where not, which step and why (plan milestone 16)."""
    off = unavailable()
    if off is not None:
        return Fusion((off,))
    from tidy3a.chunked import ROW_WISE, _needs

    blocks = frame._blocks
    rows = blocks.source.empty(_needs(frame))
    if not blocks.labelled:
        rows = rows._evolve(rows._backend.label_positions(rows._data, blocks.axis, 0))
    if not frame._summarised:
        return Fusion(("its rows are kept, not summarised",))
    reasons: list[str] = []
    with warnings.catch_warnings():  # a reduction of no values warns; here that says nothing
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            for verb in frame._steps:
                if verb._operation in ROW_WISE:
                    found, rows = _verb(verb, rows)
                    reasons += found
            reasons += _summary(frame._plan, rows)
        except TidyError as error:  # the verbs passed their checks on this block; anything else is not planned for
            return Fusion((f"it could not be planned ({str(error).splitlines()[0]})",))
    return Fusion(tuple(dict.fromkeys(reasons)))


def _summary(plan: Any, rows: Any) -> list[str]:
    reasons: list[str] = []
    for part in plan.parts:
        reasons += _key(part, rows)
    context = Context(rows, verb="summarise", error=SummariseError, grouped=False)
    for node in plan.reductions.values():
        if node.op not in REDUCTIONS:
            reasons.append(f"{node.text()}: a registered reduction runs on NumPy")
        elif node.arg is not None:
            reasons += [f"{node.text()}: {why}" for why in _expression(node.arg, context)]
    return reasons


def _verb(verb: Any, rows: Any) -> tuple[list[str], Any]:
    """Why a row-wise verb does not compile, and the block of no positions after it."""
    if verb._operation == "select":
        return [], rows >> verb
    if verb._operation == "filter":
        context = Context(rows, verb="filter", error=FilterError)
        return [f"{verb.text}: {why}" for predicate in verb._arguments
                for why in _expression(predicate, context)], rows >> verb
    from tidy3a.verbs import mutate

    reasons = []
    for step in verb._arguments:  # mutate's assignments, in order, each seeing the ones before
        context = Context(rows, verb="mutate", error=MutateError)
        held = set(rows._backend.variable_names(rows._data))
        if not isinstance(step.target, str) or (rows._field_axis is not None and step.target not in held
                                                and _is_field(rows, step.target)):
            reasons.append(f"{verb.text}: {step.text()} writes into an array's fields, which runs on NumPy")
        reasons += [f"{verb.text}: {why}" for why in _expression(step.value, context)]
        rows = rows >> mutate(step)
    return reasons, rows


def _is_field(rows: Any, name: str) -> bool:
    backend, data = rows._backend, rows._data
    return backend.has_label(data, rows._field_axis, name)


def _key(part: Any, rows: Any) -> list[str]:
    from labelarray.transforms import Bins, GridCells, Round

    context = Context(rows, verb="group_by", error=TidyError, grouped=False)
    reasons = [f"group_by({part.label}): {why}" for why in _expression(part.expr, context)]
    if reasons:
        return reasons
    kind = _dtype(evaluate(part.expr, context)).kind
    if part.transform is None:
        return [] if kind in KINDS else [f"group_by({part.label}): a key of {_kind_words(kind)} runs on NumPy"]
    if isinstance(part.transform, (GridCells, Bins, Round)):
        return [] if kind in "iuf" else [f"group_by({part.label}): its {_kind_words(kind)} run on NumPy"]
    return [f"group_by({part.label}): {type(part.transform).__name__} keys run on NumPy"]


def _expression(node: Any, context: Context) -> list[str]:
    """Why *node* does not compile: each step that runs on NumPy, from the innermost; none when it compiles."""
    lower = getattr(node, "lower", None)
    if lower is not None:  # a formula: what it lowers to on the data
        node, _ = lower(context)
    if isinstance(node, Literal):
        value = node.value
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)) and not isinstance(value, np.float16):
            return []
        return [f"{node.text()} is not a number, which runs on NumPy"]
    if isinstance(node, Ref):
        return _typed(node, context)
    if isinstance(node, BinOp):
        inner = _expression(node.left, context) + _expression(node.right, context)
        if node.symbol == "**":
            exponent = node.right.value if isinstance(node.right, Literal) else None
            if not isinstance(exponent, (int, float, np.integer, np.floating)) or float(exponent) not in POWERS:
                return inner + [f"{node.text()}: a power other than 2, 0.5, -1, or 1 runs on NumPy"]
        elif node.symbol not in OPERATORS:
            return inner + [f"{node.text()}: {node.symbol} runs on NumPy"]
        return inner or _typed(node, context)
    if isinstance(node, UnaryOp):
        inner = _expression(node.operand, context)
        if node.symbol not in SIGNS:
            return inner + [f"{node.text()}: {node.symbol} runs on NumPy"]
        return inner or _typed(node, context)
    if isinstance(node, Call):
        if node.name not in FUNCTIONS or node.options:
            return [f"{node.name}() runs on NumPy"]
        inner = [why for arg in node.args for why in _expression(arg, context)]
        return inner or _typed(node, context)
    return [f"{node.text()} runs on NumPy"]


def _typed(node: Any, context: Context) -> list[str]:
    """Why *node*'s values do not compile for their type: only booleans, integers, and float32 and float64 do."""
    value = evaluate(node, context)
    dtype = _dtype(value)
    if dtype.kind in KINDS and dtype != np.float16:
        return []
    return [f"{node.text()} is {_kind_words(dtype.kind)}, which runs on NumPy"]


def _dtype(value: Any) -> np.dtype:
    dtype = getattr(value, "dtype", None)
    return np.dtype(dtype) if dtype is not None else np.asarray(value).dtype


def _kind_words(kind: str) -> str:
    return {"M": "times", "m": "durations", "U": "text", "S": "text", "O": "objects", "c": "complex numbers",
            "f": "half-precision floats", "b": "booleans"}.get(kind, f"{kind} values")
