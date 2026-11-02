"""Notebook layer: bare names in tidy3a verbs (plan §5.2, §5.6).

Load it in IPython or Jupyter with ``%load_ext tidy3a.jupyter``. Then::

    cloud >> select(x, y, z)
    cloud >> filter(distance <= 50)
    cloud >> mutate([x, y, z] = dot(R, [x, y, z], over="feature"))
    cloud >> filter(!(x > 50))
    cloud >> group_by(window = bins(time, "30 days"), lat, lon)
    cloud >> filter(`intensity (raw)` > 15) >> mutate(`twice x` = x * 2)

A bare name inside a tidy3a verb becomes a deferred reference, settled when
the verb runs: data if only the data has it, the Python value if only the
notebook has it, and an error if both do (tidy3 lets data win; tidy3a asks).
Functions, modules, and references made with ``fields()`` keep their
Python meaning, except where only data or a value fits (``id > 1``,
``max * 2``, ``mean(type)``): there a variable named like a builtin or a
tidy3a helper is the data's. A function's local variables stay its own,
and ``all_of``/``any_of`` take lists of names from the notebook.

Only calls to tidy3a's own verbs are rewritten, decided from the live
namespace and the cell's own imports, so tidy3 and tidy3a can share a
notebook. Adapted from tidy3's ``masking.py`` and ``jupyter.py`` (MIT),
which tidy3a does not import.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import tidy3syntax
import tidy3syntax.ipython
from tidy3syntax.repairs import backtick_assignments, backticks, is_shell_cell, keywords_before_positional, negation
from tidy3syntax.scanner import apply, calls

from tidy3a.expr import MISSING, Expr, NameRef, col

NAME = "__tidy3a_name__"
ASSIGN = tidy3syntax.ASSIGN  # "__r_assign__", the placeholder tidy3, tidy3a, plot3 and stats3 share
OWN_ASSIGN = "__tidy3a_assign__"  # tidy3a's assign: the AST pass renames the shared placeholder to it in tidy3a's calls
COL = "__tidy3a_col__"  # tidy3a's col(), for backtick names

# Verbs whose arguments may hold bare names. Others take positions, counts,
# outside arrays, or other data (slice, bind_cols, joins) and are left alone.
_REWRITTEN = frozenset({"select", "filter", "mutate", "summarise", "summarize", "group_by", "pull", "arrange", "desc"})
# Verbs that take name = value next to positional arguments, in the order written; their settings stay keywords.
_ORDERED = frozenset({"mutate", "group_by"})
_SETTINGS = frozenset({"dropna", "axis"})


# ── Runtime helper ──────────────────────────────────────────────────────────


def make_name(name: str, getter: Any, operand: bool = False) -> Any:
    """Called by rewritten cells: the reference a bare *name* stands for.

    An *operand* is a place only data or a value fits (``id > 1``, ``max * 2``, ``mutate(z = type)``), so there a
    function, class, or module the notebook has under that name, often a builtin or a tidy3a helper, gives way to
    the data's variable of that name.
    """
    try:
        value = getter()
    except NameError:
        return NameRef(name, MISSING)
    if isinstance(value, Expr):
        return value  # fields(), col(), formula(): already tidy3a expressions
    if isinstance(value, (ModuleType, type)) or callable(value):
        if operand and (isinstance(value, (ModuleType, type)) or inspect.isroutine(value)):
            return NameRef(name, MISSING)
        return value  # functions such as is_numeric keep their Python meaning
    return NameRef(name, value)


# ── Ownership ───────────────────────────────────────────────────────────────


def _owned(value: Any) -> bool:
    module = getattr(value, "__name__" if isinstance(value, ModuleType) else "__module__", "")
    return module == "tidy3a" or str(module).startswith("tidy3a.")


def _star_names(module: str) -> list[str]:
    try:
        imported = importlib.import_module(module)
    except Exception:
        return []
    names = getattr(imported, "__all__", None)
    return list(names) if names is not None else [n for n in dir(imported) if not n.startswith("_")]


def _cell_imports(tree: ast.Module) -> dict[str, str]:
    """Names this cell imports, mapped to their package; the last import wins."""
    statements = [s for s in ast.walk(tree) if isinstance(s, (ast.Import, ast.ImportFrom))]
    statements.sort(key=lambda s: (s.lineno, s.col_offset))
    sources: dict[str, str] = {}
    for statement in statements:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                sources[(alias.asname or alias.name).split(".")[0]] = alias.name.split(".")[0]
        elif statement.module and statement.level == 0:
            package = statement.module.split(".")[0]
            for alias in statement.names:
                if alias.name == "*":
                    for name in _star_names(statement.module):
                        sources[name] = package
                else:
                    sources[alias.asname or alias.name] = package
    return sources


def _scope(nodes: list[ast.AST]) -> tuple[set[str], set[str]]:
    """The names a scope binds, and those it declares ``global`` or ``nonlocal``: its own, not those of the
    functions, classes, lambdas, and comprehensions inside it, which have scopes of their own."""
    bound: set[str] = set()
    declared: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            continue
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
        stack.extend(ast.iter_child_nodes(node))
    return bound, declared


def _defined(tree: ast.Module) -> set[str]:
    """The notebook names this cell binds: at its top level, and where a function declares one ``global``. A
    function's local variables are its own (tidy3 17828e2)."""
    names = _scope(tree.body)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _locals(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """A function's local variables: its parameters and what its body binds, less what it declares global."""
    arguments = node.args
    names = {a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
    names.update(a.arg for a in (arguments.vararg, arguments.kwarg) if a is not None)
    bound, declared = _scope(node.body if isinstance(node.body, list) else [node.body])
    return (names | bound) - declared


# ── The AST pass ────────────────────────────────────────────────────────────


class _Names(ast.NodeTransformer):
    """Rewrite bare names inside one verb argument.

    *tidy3a* says whether a call's function is tidy3a's. Its helpers' arguments (``mean(max)``, ``cells(id, 1)``)
    are operands, as are the whole argument and the sides of operators; ``where``'s predicate keeps its meaning,
    and ``all_of``/``any_of`` take lists of names as Python values (tidy3 b1bab9e).
    """

    def __init__(self, tidy3a: Callable[[ast.AST], bool]):
        self.tidy3a = tidy3a

    def operand(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Name):
            return self._name(node, operand=True)
        return self.visit(node)

    def _name(self, node: ast.Name, operand: bool) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        getter = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.Name(id=node.id, ctx=ast.Load()),
        )
        call = ast.Call(
            func=ast.Name(id=NAME, ctx=ast.Load()),
            args=[ast.Constant(value=node.id), getter, *([ast.Constant(value=True)] if operand else [])],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return self._name(node, operand=False)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node.left, node.right = self.operand(node.left), self.operand(node.right)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node.left = self.operand(node.left)
        node.comparators = [self.operand(value) for value in node.comparators]
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node.values = [self.operand(value) for value in node.values]
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node.operand = self.operand(node.operand)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if _backtick_name(node):
            # `intensity (raw)`: a data name that is not a Python identifier.
            return ast.copy_location(ast.Call(func=ast.Name(id=COL, ctx=ast.Load()), args=node.args, keywords=[]), node)
        # The function itself keeps its name; its arguments may hold data names.
        func = node.func
        helper = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        ours = self.tidy3a(func)
        if ours and helper in ("all_of", "any_of"):
            return node
        visit = self.operand if ours and helper != "where" else self.visit
        node.args = [visit(arg) for arg in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=visit(k.value)) for k in node.keywords]
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return node  # np.pi, frame.attr: Python objects

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node.slice = self.visit(node.slice)
        return node  # R[0]: R is a Python object

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_ListComp(self, node: ast.AST) -> ast.AST:
        return node

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp


def _backtick_name(node: ast.AST) -> bool:
    """Whether *node* is ``__r_name__('a b')``, a backtick name the source pass rewrote."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == tidy3syntax.NAME
        and len(node.args) == 1 and not node.keywords
        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
    )


def _assign_call(node: ast.AST) -> bool:
    """``__r_assign__(target, value)``, under the shared name, tidy3's, or tidy3a's own."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in (ASSIGN, OWN_ASSIGN) and len(node.args) == 2 and not node.keywords
    )


class Tidy3aMaskTransformer(ast.NodeTransformer):
    """Rewrite bare names in calls to tidy3a's verbs, and nowhere else."""

    def __init__(self, namespace: dict[str, Any] | None = None):
        self.namespace = namespace
        self.imports: dict[str, str] = {}
        self.defined: set[str] = set()
        self.scopes: list[set[str]] = []  # the local variables of the functions being visited, innermost last

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.imports = _cell_imports(node)
        self.defined = _defined(node) - set(self.imports)
        self.scopes = []
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        self.scopes.append(_locals(node))
        try:
            return self.generic_visit(node)
        finally:
            self.scopes.pop()

    visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef

    def _tidy3a(self, func: ast.AST) -> bool:
        """Whether *func* names one of tidy3a's functions here: not a local variable, nor one the cell binds."""
        namespace = self.namespace or {}
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name = func.value.id
        elif isinstance(func, ast.Name):
            name = func.id
            if name in (ASSIGN, OWN_ASSIGN):
                return True
        else:
            return False
        if any(name in scope for scope in self.scopes):
            return False
        if name in self.imports:
            return self.imports[name] == "tidy3a"
        if name in self.defined:
            return False
        return name in namespace and _owned(namespace[name])

    def _owns(self, func: ast.AST) -> bool:
        verb = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        return verb in _REWRITTEN and self._tidy3a(func)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self._owns(node.func):
            return self.generic_visit(node)
        for arg in node.args:
            if _assign_call(arg):
                # Inside tidy3a's calls the placeholder is tidy3a's assign, whichever package's
                # pass wrote it: the shared name's value in the namespace is not tidy3a's to set.
                arg.func.id = OWN_ASSIGN
        names = _Names(self._tidy3a)
        node.args = [names.operand(arg) for arg in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=names.operand(k.value)) for k in node.keywords]
        return node


# ── Source passes: R syntax Python cannot parse ─────────────────────────────
#
# The rewrites tidy3, tidy3a, plot3 and stats3 share come from tidy3syntax;
# tidy3a describes which of its verbs take them, and adds its own vector form.

_VECTOR = re.compile(r"(\s*)\[([^\[\]]*)\]\s*=(?!=)(.*)\Z", re.S)


def rewrite_vector_assignments(source: str) -> str:
    """``mutate([x, y, z] = value)`` → ``mutate(__r_assign__([x, y, z], value))``.

    ``[x, y, z] = …`` is not valid as a keyword argument, so this runs on the
    source before Python parses it, as tidy3 does for backtick names.
    """
    if "mutate" not in source or "]" not in source:
        return source
    # One call at a time from the right, so an edit never moves a call still to be handled.
    for index in reversed(range(len(calls(source, {"mutate"})))):
        call = calls(source, {"mutate"})[index]
        edits = []
        for a, b in call.arguments:
            match = _VECTOR.match(source[a:b])
            if match is not None:
                space, names, value = match.groups()
                edits.append((a, b, f"{space}{ASSIGN}([{names}], {value.strip()})"))
        source = apply(source, edits)
    return source


DIALECT = tidy3syntax.Dialect(
    "tidy3a",
    negation_calls=_REWRITTEN,
    ordered_calls=_ORDERED,
    settings=_SETTINGS,
    stages=(rewrite_vector_assignments,),
    pipe_operators={">>"},  # cloud >> filter(...) continued on the next line, as in tidy3
    extension="tidy3a.jupyter:load_ipython_extension",
)


def dialect() -> tidy3syntax.Dialect:
    """tidy3a's dialect for tidy3syntax (its ``tidy3syntax.dialects`` entry point)."""
    return DIALECT


def rewrite_keywords_before_positional(source: str) -> str:
    """``group_by(window = bins(time, "30 days"), lat)`` → ``group_by(__r_assign__('window', (…)), lat)``.

    R takes ``name = value`` before a positional argument and Python does
    not, so in ``mutate`` and ``group_by`` each such keyword becomes the
    ordered placeholder the vector form uses, keeping the order written.
    Keywords after the last positional argument, ``**`` spreads, and settings
    such as ``dropna=`` are left alone, so valid Python is unchanged.
    """
    return keywords_before_positional(source, [DIALECT])


def rewrite_bang_not(source: str) -> str:
    """R's ``!`` → ``~`` inside tidy3a verbs: ``filter(!(x > 50))``, ``select(!z)``.

    Only inside a verb's parentheses, where ``!`` is not valid Python, so a
    shell line (``!pip install …``, ``x = !ls``) outside one stays as it is,
    and so do ``!=``, strings, and comments; a ``!`` that starts a line inside
    a verb call is negation, not a shell escape.
    """
    return negation(source, [DIALECT])


def rewrite_source(source: str) -> str:
    """The source passes, in tidy3syntax's order: ``!``, backtick assignments
    (`` `new col` = x ``), vector assignments, keywords before positional
    arguments, then backtick names (`` `intensity (raw)` ``).

    Backtick names become ``__r_name__('…')``, which tidy3 and plot3 read
    too, since a tidy3a pass may meet their calls first in a shared notebook.
    """
    if is_shell_cell(source):
        return source
    source = backtick_assignments(rewrite_bang_not(source))
    return backticks(rewrite_keywords_before_positional(rewrite_vector_assignments(source)))


def tidy3a_source_transform(lines: list[str]) -> list[str]:
    """IPython input transformer for the source passes: ``!``, ``[x, y, z] = …``, and keywords before positions."""
    text = "".join(lines)
    if "!" not in text and "=" not in text:
        return lines
    rewritten = rewrite_source(text)
    if rewritten == text:
        return lines
    return rewritten.splitlines(keepends=True)


# ── Extension ───────────────────────────────────────────────────────────────


def _is_ours(value: Any) -> bool:
    return isinstance(value, Tidy3aMaskTransformer) or getattr(value, "__name__", "") == "tidy3a_source_transform"


def load_ipython_extension(ipython: Any) -> None:
    """``%load_ext tidy3a.jupyter``: turn on bare names and R syntax in tidy3a verbs.

    The source rewrites go through tidy3syntax's coordinator, one input
    transformer shared with tidy3, plot3 and stats3, whatever the load order.
    """
    from tidy3a.verbs import assign

    _remove_own(ipython)
    ipython.user_ns[NAME] = make_name
    ipython.user_ns[OWN_ASSIGN] = assign
    ipython.user_ns[COL] = col
    tidy3syntax.ipython.register(ipython, DIALECT, ast_transformer=Tidy3aMaskTransformer(ipython.user_ns))


def _remove_own(ipython: Any) -> None:
    """An earlier tidy3a's own transformers, from before the coordinator."""
    ipython.input_transformers_cleanup[:] = [t for t in ipython.input_transformers_cleanup if not _is_ours(t)]
    ipython.ast_transformers[:] = [t for t in ipython.ast_transformers if not _is_ours(t)]


def unload_ipython_extension(ipython: Any) -> None:
    tidy3syntax.ipython.unregister(ipython, "tidy3a")
    _remove_own(ipython)


def apply_masking(source: str, namespace: dict[str, Any] | None = None) -> str:
    """The rewritten source of a cell, for tests and inspection."""
    tree = ast.parse(tidy3syntax.rewrite(source, [DIALECT], namespace))
    tree = Tidy3aMaskTransformer(namespace).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
