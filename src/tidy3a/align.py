"""Alignment rules (plan §5.5).

Labels on a shared axis must match exactly. The same labels in another
order are matched by label and reported; different labels need a join
verb. A labelled axis never pairs with an unlabelled one by length.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from tidy3a.errors import AlignError


def align_pair(
    left: xr.DataArray, right: xr.DataArray, left_text: str, right_text: str, notes: list[str],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return *left* and *right* ready for exact arithmetic, or raise ``AlignError``."""
    for dim in left.dims:
        if dim not in right.dims:
            continue
        right = _match_axis(
            dim, left, right, left_text, right_text, notes,
            context=f"in {left_text} and {right_text}",
        )
    return left, right


def align_many(values: Sequence[Any], texts: Sequence[str], notes: list[str]) -> list[Any]:
    """Align each array to earlier operands on every shared axis; keep scalars."""
    aligned = []
    for value, text in zip(values, texts):
        if isinstance(value, xr.DataArray):
            for previous, previous_text in zip(aligned, texts):
                if isinstance(previous, xr.DataArray):
                    _, value = align_pair(previous, value, previous_text, text, notes)
        aligned.append(value)
    return aligned


def align_to(
    value: xr.DataArray, target: xr.Dataset, text: str, notes: list[str],
) -> xr.DataArray:
    """Match *value*'s shared axes to *target*'s, reordering by label when needed."""
    for dim in value.dims:
        if dim not in target.dims:
            continue
        value = _match_axis(dim, target, value, "this data", text, notes, context=f"for {text}")
    return value


def check_join_axes(left: xr.Dataset, right: xr.Dataset, right_text: str) -> None:
    """Joins align by label; a labelled axis still never pairs with an unlabelled one."""
    for dim in left.dims:
        if dim not in right.dims:
            continue
        labelled = (dim in left.indexes, dim in right.indexes)
        if labelled[0] != labelled[1]:
            raise AlignError(_labelled_message(dim, labelled[0], "this data", right_text))
        if not any(labelled) and left.sizes[dim] != right.sizes[dim]:
            raise AlignError(
                f"{dim} is unlabelled on both sides with different lengths "
                f"({left.sizes[dim]:,} and {right.sizes[dim]:,}); positions cannot be paired"
            )


def _match_axis(
    dim: Hashable, left, right: xr.DataArray, left_text: str, right_text: str,
    notes: list[str], *, context: str,
) -> xr.DataArray:
    left_has = dim in left.indexes
    right_has = dim in right.indexes
    if left_has != right_has:
        raise AlignError(_labelled_message(dim, left_has, left_text, right_text))
    if not left_has:
        if left.sizes[dim] != right.sizes[dim]:
            raise AlignError(
                f"{dim} has {left.sizes[dim]:,} positions in {left_text} but "
                f"{right.sizes[dim]:,} in {right_text}, and neither has labels to match"
            )
        return right
    left_index = left.indexes[dim]
    right_index = right.indexes[dim]
    if left_index.equals(right_index):
        return right
    if len(left_index) == len(right_index) and left_index.sort_values().equals(right_index.sort_values()):
        before = "  ".join(str(label) for label in _shown(right_index))
        after = "  ".join(str(label) for label in _shown(left_index))
        notes.append(f"changed: {dim} reordered to match ({before} → {after}) {context}")
        return right.sel({dim: left_index})
    only_right = len(right_index.difference(left_index))
    only_left = len(left_index.difference(right_index))
    raise AlignError(
        f"{dim} labels differ between {left_text} ({len(left_index):,}) "
        f"and {right_text} ({len(right_index):,}).\n"
        f"  {only_right:,} labels are only in {right_text}; {only_left:,} only in {left_text}.\n"
        f"  To keep shared {dim} labels:  inner_join(...)\n"
        f"  To keep this data's:          left_join(...)"
        + _rounding_hint(dim, left_index, right_index)
    )


def _labelled_message(dim: Hashable, left_has: bool, left_text: str, right_text: str) -> str:
    labelled, unlabelled = (left_text, right_text) if left_has else (right_text, left_text)
    return (
        f"{dim} is labelled on {labelled} but unlabelled on {unlabelled}.\n"
        "  Pairing by length could match the wrong positions.\n"
        f'  If {unlabelled} follows the same order, attach it by position: bind_cols(..., along="{dim}")'
    )


def _rounding_hint(dim: Hashable, left_index, right_index) -> str:
    if len(left_index) != len(right_index):
        return ""
    try:
        left_values = np.asarray(left_index, dtype=float)
        right_values = np.sort(np.asarray(right_index, dtype=float))
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


def _shown(index) -> list:
    values = list(index)
    return values if len(values) <= 4 else [*values[:3], "…"]
