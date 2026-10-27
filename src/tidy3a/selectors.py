"""Selection helpers for ``select``, after tidy3's tidyselect set.

They match columns: variables and field-axis labels (plan §5.1). Labels on
observation axes are values and belong to ``filter``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable
from typing import Any


class Column:
    """A selectable column: a variable or a field-axis label."""

    __slots__ = ("kind", "name", "data")

    def __init__(self, kind: str, name: Hashable, data: Any):
        self.kind = kind
        self.name = name
        self.data = data


class Selector:
    """A rule that picks columns by name or content; ``~`` excludes them.

    ``settings`` records the helper and its arguments, the selector's
    identity (``tidy3a.identity``); without them, as for a predicate of the
    caller's, the selector is like no other.
    """

    def __init__(self, text: str, pick: Callable[[list[Column]], list[Column]], settings: tuple | None = None):
        self._text = text
        self._pick = pick
        self._settings = settings

    def _structure(self) -> Hashable:
        from tidy3a.identity import Unique, structure

        return ("selector", Unique(self._text) if self._settings is None else structure(self._settings))

    def pick(self, columns: list[Column]) -> list[Column]:
        return self._pick(columns)

    def text(self) -> str:
        return self._text

    def __invert__(self) -> Exclude:
        return Exclude(self)

    __neg__ = __invert__

    def __repr__(self) -> str:
        return f"<selector {self._text}>"


class Exclude:
    """``~selector`` or ``~name``: remove these columns from the selection."""

    def __init__(self, item: Any):
        self.item = item

    def text(self) -> str:
        item = self.item
        return f"~{item.text() if hasattr(item, 'text') else (repr(item) if not isinstance(item, str) else item)}"

    def __repr__(self) -> str:
        return f"<selector {self.text()}>"

    def _structure(self) -> Hashable:
        from tidy3a.identity import structure

        return ("exclude", structure(self.item))


def _by_name(text: str, test: Callable[[str], bool], settings: tuple) -> Selector:
    return Selector(text, lambda columns: [c for c in columns if test(str(c.name))], settings)


def _matches(match: str | Iterable[str], ignore_case: bool, how: Callable[[str, str], bool]):
    patterns = [match] if isinstance(match, str) else list(match)
    if ignore_case:
        patterns = [p.lower() for p in patterns]

    def test(name: str) -> bool:
        name = name.lower() if ignore_case else name
        return any(how(name, p) for p in patterns)

    return test


def everything() -> Selector:
    """Every variable and field label."""
    return Selector("everything()", lambda columns: list(columns), ("everything",))


def starts_with(match: str | Iterable[str], *, ignore_case: bool = True) -> Selector:
    return _by_name(f"starts_with({match!r})", _matches(match, ignore_case, str.startswith),
                    ("starts_with", match if isinstance(match, str) else tuple(match), ignore_case))


def ends_with(match: str | Iterable[str], *, ignore_case: bool = True) -> Selector:
    return _by_name(f"ends_with({match!r})", _matches(match, ignore_case, str.endswith),
                    ("ends_with", match if isinstance(match, str) else tuple(match), ignore_case))


def contains(match: str | Iterable[str], *, ignore_case: bool = True) -> Selector:
    return _by_name(f"contains({match!r})", _matches(match, ignore_case, lambda n, p: p in n),
                    ("contains", match if isinstance(match, str) else tuple(match), ignore_case))


def matches(pattern: str, *, ignore_case: bool = True) -> Selector:
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    return _by_name(f"matches({pattern!r})", lambda name: regex.search(name) is not None,
                    ("matches", pattern, ignore_case))


def num_range(prefix: str, values: Iterable[int], *, width: int | None = None) -> Selector:
    wanted = [f"{prefix}{value:0{width}d}" if width else f"{prefix}{value}" for value in values]
    return all_of(wanted, _text=f"num_range({prefix!r}, …)", _missing_ok=True)


def all_of(names: Iterable[Hashable], *, _text: str | None = None, _missing_ok: bool = False) -> Selector:
    """Exactly these names, in this order; a missing name is an error."""
    wanted = list(names)

    def pick(columns: list[Column]) -> list[Column]:
        by_name = {c.name: c for c in columns}
        missing = [name for name in wanted if name not in by_name]
        if missing and not _missing_ok:
            from tidy3a.errors import SelectError

            raise SelectError(f"all_of(): {', '.join(map(str, missing))} not found; any_of() allows missing names")
        return [by_name[name] for name in wanted if name in by_name]

    return Selector(_text or f"all_of({wanted!r})", pick, ("all_of", tuple(wanted), _missing_ok))


def any_of(names: Iterable[Hashable]) -> Selector:
    """These names where they exist, in this order."""
    names = list(names)  # once: a generator would be empty the second time
    return all_of(names, _text=f"any_of({names!r})", _missing_ok=True)


def where(*args: Any) -> Selector:
    """Columns whose values satisfy *predicate*, e.g. ``where(is_numeric)``."""
    if len(args) == 3:
        raise TypeError(
            "where() selects columns by predicate, as in tidy3 (select(where(is_numeric))).\n"
            "  For NumPy/xarray-style masking use if_else(cond, x, y)."
        )
    if len(args) != 1 or not callable(args[0]):
        raise TypeError("where() takes one predicate, e.g. where(is_numeric)")
    predicate = args[0]
    name = getattr(predicate, "__name__", "predicate")
    own = ("where", name) if predicate is is_numeric else None  # the caller's predicates are like no other

    def pick(columns: list[Column]) -> list[Column]:
        if own is None:  # a caller's predicate may look at any column's values
            from tidy3a.expr import EVERYTHING, note_read

            note_read(EVERYTHING)
        return [c for c in columns if predicate(c.data)]

    return Selector(f"where({name})", pick, own)


def last_col(offset: int = 0) -> Selector:
    def pick(columns: list[Column]) -> list[Column]:
        return [columns[-1 - offset]] if offset < len(columns) else []

    return Selector(f"last_col({offset})" if offset else "last_col()", pick, ("last_col", offset))


def is_numeric(data: Any) -> bool:
    """True for numeric values; for ``where(is_numeric)``."""
    return getattr(data, "dtype", None) is not None and data.dtype.kind in "biufc"
