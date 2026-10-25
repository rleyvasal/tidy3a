"""Exact alignment of one axis's labels (plan §5.5; decision 3: exact is the only mode).

Labels on a shared axis must match exactly. The same labels in another
order are matched by label; different labels, or repeated labels in another
order, cannot be paired and are an ``AlignError``. There is no looser mode
(xarray's outer join): a mode can be added later, a changed default cannot.

This module needs NumPy only.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

from tidy3a.labelarray.errors import AlignError
from tidy3a.labelarray.index import LabelIndex, index_for
from tidy3a.labelarray.labels import compare


def order_like(reference: Any, labels: Any, dim: Hashable = None, *,
               index: Callable[..., LabelIndex] = index_for) -> np.ndarray | None:
    """The positions that put *labels* in *reference*'s order, or None when they are already the same.

    *index* gives a label array's index; a caller that keeps indexes on its
    label arrays passes its own, so they are built once.
    """
    state, only_reference, only_labels = compare(reference, labels, "align", dim)
    if state == "same":
        return None
    if state == "reordered":
        ours = index(labels, "align", dim)
        if not (ours.unique and index(reference, "align", dim).unique):
            raise AlignError(f"{dim} has repeated labels in a different order on two operands; "
                             "repeated labels cannot be matched one to one")
        return ours.get_indexer(reference)
    raise AlignError(f"{dim} labels differ between operands: {only_labels:,} only in one, "
                     f"{only_reference:,} only in the other; join them first")
