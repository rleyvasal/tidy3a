"""labelarray's errors: one family of ``ValueError``, so callers catching bad input keep working."""


class LabelArrayError(ValueError):
    """Base class for labelarray's errors."""


class LabelError(LabelArrayError):
    """Labels that cannot be ordered, compared, or looked up, or a label that is absent."""


class AlignError(LabelArrayError):
    """Labels on a shared axis do not match."""


class ShapeError(LabelArrayError):
    """Axis names, sizes, or ranks that do not fit together."""
