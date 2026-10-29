"""Exceptions raised by tidy3a.

Every tidy3a error is a ``ValueError``, so code that already catches bad
input keeps working; the subclasses say which step rejected it. The array
library tidy3a builds on, ``labelarray``, raises its own family,
``LabelArrayError``; tidy3a's ``LabelError``, ``AlignError``, and
``GroupError`` inherit from both, and ``translating`` raises a library
error that escapes through a verb, a reader, ``tidy()``, or ``collect()``
again as tidy3a's class.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from labelarray import errors as labelarray


class TidyError(ValueError):
    """Base class for tidy3a errors."""


class ReadError(TidyError):
    """An array or record reader cannot read the source as described."""


class RefError(TidyError):
    """A name in an expression is unknown or means more than one thing."""


class SelectError(RefError):
    """``select`` cannot pick what was asked for."""


class FilterError(TidyError):
    """``filter`` cannot keep a rectangle of positions."""


class SliceError(TidyError):
    """``slice`` was given positions that do not exist."""


class ArrangeError(TidyError):
    """``arrange`` has no one axis its keys sort."""


class MutateError(TidyError):
    """``mutate`` cannot store a result."""


class GroupError(TidyError, labelarray.GroupError):
    """``group_by`` cannot group by what was asked for."""


class SummariseError(TidyError):
    """``summarise`` cannot reduce a summary to one value per group."""


class BlockError(TidyError):
    """A verb or summary that cannot run block by block over a file read with ``blocks=``."""


class WriteError(TidyError):
    """``write`` cannot write this data to this path."""


class AlignError(TidyError, labelarray.AlignError):
    """Labels on a shared axis do not match."""


class LabelError(TidyError, labelarray.LabelError):
    """Labels that cannot be ordered, compared, or looked up, or a label that is absent."""


_FROM_LIBRARY = {labelarray.LabelError: LabelError, labelarray.AlignError: AlignError,
                 labelarray.GroupError: GroupError}


def translating(function: Callable) -> Callable:
    """Raise a library error escaping *function* again as the tidy3a class, keeping the message and traceback."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except TidyError:
            raise
        except labelarray.LabelArrayError as error:
            tidy = next((cls for base, cls in _FROM_LIBRARY.items() if isinstance(error, base)), TidyError)
            raise tidy(*error.args).with_traceback(error.__traceback__) from None

    return wrapper
