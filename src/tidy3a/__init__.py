"""tidy3A: tidy-style verbs for named multidimensional arrays."""

from tidy3a.__version__ import __version__
from tidy3a.errors import ReadError, TidyError
from tidy3a.frame import TidyArray, tidy
from tidy3a.io import infer, read_array

__all__ = [
    "ReadError",
    "TidyArray",
    "TidyError",
    "__version__",
    "infer",
    "read_array",
    "tidy",
]
