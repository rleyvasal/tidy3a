"""Alignment rules (plan §5.5).

Labels on a shared axis must match exactly. The same labels in another
order are matched by label and reported; different labels need a join
verb. A labelled axis never pairs with an unlabelled one by length. The
rules and messages live here; label comparisons go through the backend.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

from tidy3a import labels
from tidy3a.backend import xarray_backend
from tidy3a.errors import AlignError


def align_pair(
    left: Any, right: Any, left_text: str, right_text: str, notes: list[str], *, backend: Any = None,
) -> tuple[Any, Any]:
    """Return *left* and *right* ready for exact arithmetic, or raise ``AlignError``."""
    backend = xarray_backend() if backend is None else backend
    for dim in left.dims:
        if dim not in right.dims:
            continue
        right = _match_axis(
            dim, left, right, left_text, right_text, notes, backend,
            context=f"in {left_text} and {right_text}",
        )
    return left, right


def align_many(values: Sequence[Any], texts: Sequence[str], notes: list[str], *, backend: Any = None) -> list[Any]:
    """Align each array to earlier operands on every shared axis; keep scalars."""
    backend = xarray_backend() if backend is None else backend
    aligned = []
    for value, text in zip(values, texts):
        if backend.is_array(value):
            for previous, previous_text in zip(aligned, texts):
                if backend.is_array(previous):
                    _, value = align_pair(previous, value, previous_text, text, notes, backend=backend)
        aligned.append(value)
    return aligned


def align_to(value: Any, target: Any, text: str, notes: list[str], *, backend: Any = None) -> Any:
    """Match *value*'s shared axes to *target*'s, reordering by label when needed."""
    backend = xarray_backend() if backend is None else backend
    for dim in value.dims:
        if not backend.has_axis(target, dim):
            continue
        value = _match_axis(dim, target, value, "this data", text, notes, backend, context=f"for {text}")
    return value


def check_join_axes(left: Any, right: Any, right_text: str, *, backend: Any = None) -> None:
    """Joins align by label; a labelled axis still never pairs with an unlabelled one."""
    backend = xarray_backend() if backend is None else backend
    left_sizes, right_sizes = backend.sizes(left), backend.sizes(right)
    for dim in backend.axes(left):
        if not backend.has_axis(right, dim):
            continue
        labelled = (backend.has_labels(left, dim), backend.has_labels(right, dim))
        if labelled[0] != labelled[1]:
            raise AlignError(_labelled_message(dim, labelled[0], "this data", right_text))
        if not any(labelled) and left_sizes[dim] != right_sizes[dim]:
            raise AlignError(
                f"{dim} is unlabelled on both sides with different lengths "
                f"({left_sizes[dim]:,} and {right_sizes[dim]:,}); positions cannot be paired"
            )


def check_join_labels(left: Any, right: Any, verb: str, *, backend: Any = None) -> None:
    """Joins match labels exactly, one to one, under the same rule as alignment.

    A join holds both sides' labels in their common type, so labels that type
    cannot hold exactly (the integer ``2**53 + 1`` as float64) would be
    matched to other labels; repeated labels cannot be matched one to one
    unless both sides have the very same labels.
    """
    backend = xarray_backend() if backend is None else backend
    for dim in backend.axes(left):
        if not (backend.has_axis(right, dim) and backend.has_labels(left, dim) and backend.has_labels(right, dim)):
            continue
        ours, theirs = backend.label_values(left, dim), backend.label_values(right, dim)
        common = labels.lossy_join_type(ours, theirs)
        if common is not None:
            raise AlignError(
                f"{verb}(): {dim} labels are {ours.dtype} in this data and {theirs.dtype} in the other data, "
                f"and they cannot all be held exactly as {common}, the type a join keeps them in; "
                "different labels could be matched.\n"
                f"  Give {dim} labels of one type on both sides first"
            )
        if not (backend.labels_unique(left, dim) and backend.labels_unique(right, dim)) \
                and not labels.equal(ours, theirs):
            raise AlignError(
                f"{verb}(): {dim} has repeated labels, and the labels differ between this data and the "
                "other data; repeated labels cannot be matched one to one.\n"
                f'  Give {dim} unique labels, or attach values by position: bind_cols(..., along="{dim}")'
            )


def _match_axis(
    dim: Hashable, left: Any, right: Any, left_text: str, right_text: str,
    notes: list[str], backend: Any, *, context: str,
) -> Any:
    left_has = backend.has_labels(left, dim)
    right_has = backend.has_labels(right, dim)
    if left_has != right_has:
        raise AlignError(_labelled_message(dim, left_has, left_text, right_text))
    if not left_has:
        left_size, right_size = backend.sizes(left)[dim], backend.sizes(right)[dim]
        if left_size != right_size:
            raise AlignError(
                f"{dim} has {left_size:,} positions in {left_text} but "
                f"{right_size:,} in {right_text}, and neither has labels to match"
            )
        return right
    state, only_left, only_right = backend.compare_labels(left, right, dim)
    if state == "same":
        return right
    if not (backend.labels_unique(left, dim) and backend.labels_unique(right, dim)):
        raise AlignError(
            f"{dim} has repeated labels, in a different order in {left_text} and {right_text}; "
            "repeated labels cannot be matched one to one.\n"
            f'  Give {dim} unique labels, or attach values by position: bind_cols(..., along="{dim}")'
        )
    if state == "reordered":
        before = "  ".join(str(label) for label in _shown(backend.label_list(right, dim)))
        after = "  ".join(str(label) for label in _shown(backend.label_list(left, dim)))
        notes.append(f"changed: {dim} reordered to match ({before} → {after}) {context}")
        return backend.reorder_like(right, dim, left)
    raise AlignError(
        f"{dim} labels differ between {left_text} ({backend.label_count(left, dim):,}) "
        f"and {right_text} ({backend.label_count(right, dim):,}).\n"
        f"  {only_right:,} labels are only in {right_text}; {only_left:,} only in {left_text}.\n"
        f"  To keep shared {dim} labels:  inner_join(...)\n"
        f"  To keep this data's:          left_join(...)"
        + _rounding_hint(dim, backend.label_values(left, dim), backend.label_values(right, dim))
    )


def _labelled_message(dim: Hashable, left_has: bool, left_text: str, right_text: str) -> str:
    labelled, unlabelled = (left_text, right_text) if left_has else (right_text, left_text)
    return (
        f"{dim} is labelled on {labelled} but unlabelled on {unlabelled}.\n"
        "  Pairing by length could match the wrong positions.\n"
        f'  If {unlabelled} follows the same order, attach it by position: bind_cols(..., along="{dim}")'
    )


def _rounding_hint(dim: Hashable, left_labels: np.ndarray, right_labels: np.ndarray) -> str:
    if len(left_labels) != len(right_labels):
        return ""
    try:
        left_values = np.asarray(left_labels, dtype=float)
        right_values = np.sort(np.asarray(right_labels, dtype=float))
    except (TypeError, ValueError):
        return ""
    gap = float(np.max(np.abs(np.sort(left_values) - right_values), initial=0.0))
    scale = float(np.max(np.abs(left_values), initial=1.0)) or 1.0
    if 0 < gap <= 1e-6 * scale:
        return (
            f"\n  The labels differ only by up to {gap:.2g} (rounding).\n"
            "  To match them within a tolerance:  left_join(..., tolerance=1e-9)"
        )
    return ""


def _shown(labels: list) -> list:
    return labels if len(labels) <= 4 else [*labels[:3], "…"]
