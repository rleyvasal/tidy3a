"""Notebook layer: bare names in tidy3A verbs (plan §5.2, §5.6).

Load it in IPython or Jupyter with ``%load_ext tidy3a.jupyter``. Then::

    cloud >> select(x, y, z)
    cloud >> filter(distance <= 50)
    cloud >> mutate([x, y, z] = dot(R, [x, y, z], over="feature"))

A bare name inside a tidy3A verb becomes a deferred reference, settled when
the verb runs: data if only the data has it, the Python value if only the
notebook has it, and an error if both do (tidy3 lets data win; tidy3A asks).
Functions, modules, and references made with ``fields()`` keep their
Python meaning.

Only calls to tidy3A's own verbs are rewritten, decided from the live
namespace and the cell's own imports, so tidy3 and tidy3A can share a
notebook. Adapted from tidy3's ``masking.py`` and ``jupyter.py`` (MIT),
which tidy3A does not import.
"""

from __future__ import annotations

import ast
import importlib
import re
from types import ModuleType
from typing import Any

from tidy3a.expr import MISSING, Expr, NameRef

NAME = "__tidy3a_name__"
ASSIGN = "__tidy3a_assign__"

# Verbs whose arguments may hold bare names. Others take positions, counts,
# outside arrays, or other data (slice, bind_cols, joins) and are left alone.
_REWRITTEN = frozenset({"select", "filter", "mutate", "summarise", "summarize", "group_by", "pull"})


# ── Runtime helper ──────────────────────────────────────────────────────────


def make_name(name: str, getter: Any) -> Any:
    """Called by rewritten cells: the reference a bare *name* stands for."""
    try:
        value = getter()
    except NameError:
        return NameRef(name, MISSING)
    if isinstance(value, Expr):
        return value  # fields(), col(), formula(): already tidy3A expressions
    if isinstance(value, (ModuleType, type)) or callable(value):
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


def _defined(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


# ── The AST pass ────────────────────────────────────────────────────────────


class _Names(ast.NodeTransformer):
    """Rewrite bare names inside one verb argument."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        getter = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.Name(id=node.id, ctx=ast.Load()),
        )
        call = ast.Call(
            func=ast.Name(id=NAME, ctx=ast.Load()),
            args=[ast.Constant(value=node.id), getter],
            keywords=[],
        )
        return ast.copy_location(call, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # The function itself keeps its name; its arguments may hold data names.
        node.args = [self.visit(arg) for arg in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=self.visit(k.value)) for k in node.keywords]
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


class Tidy3aMaskTransformer(ast.NodeTransformer):
    """Rewrite bare names in calls to tidy3A's verbs, and nowhere else."""

    def __init__(self, namespace: dict[str, Any] | None = None):
        self.namespace = namespace
        self.imports: dict[str, str] = {}
        self.defined: set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.imports = _cell_imports(node)
        self.defined = _defined(node) - set(self.imports)
        return self.generic_visit(node)

    def _owns(self, func: ast.AST) -> bool:
        namespace = self.namespace or {}
        if isinstance(func, ast.Name):
            name = func.id
            if name not in _REWRITTEN:
                return False
            if name in self.imports:
                return self.imports[name] == "tidy3a"
            if name in self.defined:
                return False
            return name in namespace and _owned(namespace[name])
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.attr not in _REWRITTEN:
                return False
            module = func.value.id
            if module in self.imports:
                return self.imports[module] == "tidy3a"
            return module in namespace and _owned(namespace[module])
        return False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self._owns(node.func):
            return self.generic_visit(node)
        names = _Names()
        node.args = [names.visit(arg) for arg in node.args]
        node.keywords = [ast.keyword(arg=k.arg, value=names.visit(k.value)) for k in node.keywords]
        return node


# ── Source pass: [x, y, z] = … inside mutate ────────────────────────────────

_MUTATE = re.compile(r"\bmutate\s*\(")
_VECTOR = re.compile(r"(\s*)\[([^\[\]]*)\]\s*=(?!=)(.*)\Z", re.S)


def rewrite_vector_assignments(source: str) -> str:
    """``mutate([x, y, z] = value)`` → ``mutate(__tidy3a_assign__([x, y, z], value))``.

    ``[x, y, z] = …`` is not valid as a keyword argument, so this runs on the
    source before Python parses it, as tidy3 does for backtick names.
    """
    out: list[str] = []
    position = 0
    while True:
        match = _MUTATE.search(source, position)
        if match is None or _in_string_or_comment(source, match.start()):
            if match is None:
                break
            out.append(source[position : match.end()])
            position = match.end()
            continue
        split = _split_arguments(source, match.end())
        if split is None:
            break
        arguments, end = split
        out.append(source[position : match.end()])
        out.append(",".join(_vector_argument(rewrite_vector_assignments(arg)) for arg in arguments))
        out.append(")")
        position = end
    out.append(source[position:])
    return "".join(out)


def _vector_argument(text: str) -> str:
    match = _VECTOR.match(text)
    if match is None:
        return text
    space, names, value = match.groups()
    return f"{space}{ASSIGN}([{names}], {value.strip()})"


def _split_arguments(source: str, start: int) -> tuple[list[str], int] | None:
    """Top-level arguments of a call whose ``(`` ends at *start*, and the index after ``)``."""
    depth = 0
    arguments: list[str] = []
    current = start
    i = start
    quote: str | None = None
    while i < len(source):
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if source.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if ch in "'\"":
            quote = source[i : i + 3] if source[i : i + 3] in ("'''", '"""') else ch
            i += len(quote)
            continue
        if ch == "#":
            newline = source.find("\n", i)
            i = len(source) if newline == -1 else newline
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                arguments.append(source[current:i])
                return arguments, i + 1
            depth -= 1
        elif ch == "," and depth == 0:
            arguments.append(source[current:i])
            current = i + 1
        i += 1
    return None


def _in_string_or_comment(source: str, index: int) -> bool:
    line_start = source.rfind("\n", 0, index) + 1
    prefix = source[line_start:index]
    return "#" in prefix or prefix.count('"') % 2 == 1 or prefix.count("'") % 2 == 1


def tidy3a_vector_transform(lines: list[str]) -> list[str]:
    """IPython input transformer for ``[x, y, z] = …`` inside ``mutate``."""
    text = "".join(lines)
    if "mutate" not in text or "]" not in text:
        return lines
    rewritten = rewrite_vector_assignments(text)
    if rewritten == text:
        return lines
    return rewritten.splitlines(keepends=True)


# ── Extension ───────────────────────────────────────────────────────────────


def _is_ours(value: Any) -> bool:
    return isinstance(value, Tidy3aMaskTransformer) or getattr(value, "__name__", "") == "tidy3a_vector_transform"


def load_ipython_extension(ipython: Any) -> None:
    """``%load_ext tidy3a.jupyter``: turn on bare names in tidy3A verbs."""
    from tidy3a.verbs import assign

    unload_ipython_extension(ipython)
    ipython.user_ns[NAME] = make_name
    ipython.user_ns[ASSIGN] = assign
    ipython.input_transformers_cleanup.append(tidy3a_vector_transform)
    ipython.ast_transformers.append(Tidy3aMaskTransformer(ipython.user_ns))


def unload_ipython_extension(ipython: Any) -> None:
    ipython.input_transformers_cleanup[:] = [t for t in ipython.input_transformers_cleanup if not _is_ours(t)]
    ipython.ast_transformers[:] = [t for t in ipython.ast_transformers if not _is_ours(t)]


def apply_masking(source: str, namespace: dict[str, Any] | None = None) -> str:
    """The rewritten source of a cell, for tests and inspection."""
    tree = ast.parse(rewrite_vector_assignments(source))
    tree = Tidy3aMaskTransformer(namespace).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
