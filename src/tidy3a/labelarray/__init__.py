"""labelarray: labelled N-D arrays in NumPy, becoming a library of its own (plan milestone 10, track A).

Nothing here imports the rest of tidy3a. The public API:

- ``Array``: a NumPy array with named axes, exact labels, attributes, and a
  name; NumPy's ufuncs, ``np.where``, and Python's operators broadcast by axis
  name. ``Collection``: arrays sharing axes, one label array per axis.
  ``concat``, ``align``, and ``broadcast`` work on several arrays (``array``,
  ``collection``).
- The exact label rule: ``kind``, ``equal``, ``compare``, ``strict_subset``,
  ``exact_cast``, ``lossy_join_type`` (``labels``); label indexes and their
  factory, ``index_for`` and ``register_index`` (``index``).
- Grouping: ``Grouping``, with ``reduce``, ``counts``, and ``broadcast``
  applying it to an array (``grouped``); ``factorize`` and the key transforms
  ``Bins``, ``Duration``, ``Round``, and ``GridCells`` (``transforms``).
- Numbers: float32 sums and means correctly rounded (``sums``) and the
  contraction behind ``Array.dot`` (``contract``).
- Errors: ``LabelArrayError`` and its ``LabelError``, ``AlignError``,
  ``GroupError``, and ``ShapeError`` (``errors``).

The package leaves tidy3a once its API has settled (step A7).
"""

from tidy3a.labelarray.array import Array, align, broadcast, concat, label_index
from tidy3a.labelarray.collection import Collection
from tidy3a.labelarray.errors import (
    AlignError,
    GroupError,
    LabelArrayError,
    LabelError,
    ShapeError,
)
from tidy3a.labelarray.grouped import Grouping
from tidy3a.labelarray.index import (
    GridIndex,
    LabelIndex,
    SortedIndex,
    SortOnceIndex,
    index_for,
    register_index,
)
from tidy3a.labelarray.labels import (
    compare,
    equal,
    exact_cast,
    kind,
    lossy_join_type,
    strict_subset,
)
from tidy3a.labelarray.transforms import Bins, Duration, GridCells, Round, factorize

__all__ = [
    "AlignError", "Array", "Bins", "Collection", "Duration", "GridCells", "GridIndex", "GroupError", "Grouping",
    "LabelArrayError", "LabelError", "LabelIndex", "Round", "ShapeError", "SortOnceIndex", "SortedIndex", "align",
    "broadcast", "compare", "concat", "equal", "exact_cast", "factorize", "index_for", "kind", "label_index",
    "lossy_join_type", "register_index", "strict_subset",
]
