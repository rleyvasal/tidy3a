"""Errors of the library: one family of ``ValueError``, so callers catching bad input keep working."""


class CoreError(ValueError):
    """Base class for the library's errors."""


class LabelError(CoreError):
    """Labels that cannot be ordered, compared, or looked up, or a label that is absent."""


class AlignError(CoreError):
    """Labels on a shared axis do not match."""


class ShapeError(CoreError):
    """Axis names, sizes, or ranks that do not fit together."""
