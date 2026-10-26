"""Alignment on the NumPy backend through ``tidy3a.align``, with tidy3a's notes and messages.

The label lookups themselves are tested in labelarray's own tests, against
pandas; here, alignment of NumPy
backend arrays reports what it reordered in tidy3a's words, NaN labels
included, and names both operands when labels differ.
"""

import numpy as np
import pytest

from tidy3a import AlignError
from tidy3a.align import align_pair
from tidy3a.numpy_backend import NUMPY, NumpyArray

DAYS = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]")


def test_alignment_matches_nan_labels_and_reports_real_differences():
    a = NumpyArray([1.0, 2.0, 3.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    b = NumpyArray([20.0, 10.0, 30.0], ("x",), {"x": [np.nan, 0.5, 1.5]})
    notes = []
    left, right = align_pair(a, b, "a", "b", notes, backend=NUMPY)
    assert NUMPY.apply(np.add, left, right).values.tolist() == [11.0, 22.0, 33.0]
    assert notes == ["changed: x reordered to match (nan  0.5  1.5 → 0.5  nan  1.5) in a and b"]
    same = NumpyArray([1.0, 1.0, 1.0], ("x",), {"x": [0.5, np.nan, 1.5]})
    notes = []
    align_pair(a, same, "a", "same", notes, backend=NUMPY)
    assert notes == []  # equal labels, NaN included: nothing reordered
    c = NumpyArray([1.0, 2.0, 3.0], ("x",), {"x": [0.5, 2.5, 1.5]})
    with pytest.raises(AlignError, match="1 labels are only in c; 1 only in a"):
        align_pair(a, c, "a", "c", [], backend=NUMPY)


def test_alignment_across_time_units():
    a = NumpyArray([1.0, 2.0], ("time",), {"time": DAYS[:2]})
    b = NumpyArray([20.0, 10.0], ("time",), {"time": np.array(["2026-01-02", "2026-01-01"], dtype="datetime64[s]")})
    left, right = align_pair(a, b, "a", "b", [], backend=NUMPY)
    assert NUMPY.apply(np.add, left, right).values.tolist() == [11.0, 22.0]
