"""Exceptions raised by tidy3A.

Every tidy3A error is a ``ValueError``, so code that already catches bad
input keeps working; the subclasses say which step rejected it.
"""


class TidyError(ValueError):
    """Base class for tidy3A errors."""


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


class MutateError(TidyError):
    """``mutate`` cannot store a result."""


class GroupError(TidyError):
    """``group_by`` cannot group by what was asked for."""


class SummariseError(TidyError):
    """``summarise`` cannot reduce a summary to one value per group."""


class AlignError(TidyError):
    """Labels on a shared axis do not match."""
