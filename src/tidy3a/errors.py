"""Exceptions raised by tidy3A.

Every tidy3A error is a ``ValueError``, so code that already catches bad
input keeps working; the subclasses say which step rejected it.
"""


class TidyError(ValueError):
    """Base class for tidy3A errors."""


class ReadError(TidyError):
    """``read_array`` cannot read the source as described."""
