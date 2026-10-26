"""Whether an axis has a label, under the exact label rule, on both backends.

Field-axis lookups run several times per verb, so the NumPy backend answers
them for a few text labels without the label index; these tests hold both
paths, and both backends, to the same answers: a label matches only one of
its own kind that is exactly equal.
"""

import numpy as np
import pytest

from tidy3a import infer, read_array

TEXT = ["x", "y", "z", "intensity", "ring"]


def frame(labels):
    values = np.zeros((2, len(labels)), "float32")
    return read_array(values, dims={"point": infer, "feature": len(labels)}, labels={"feature": labels})


@pytest.mark.parametrize("labels", [TEXT, [f"field{k}" for k in range(500)] + TEXT], ids=["few", "many"])
@pytest.mark.parametrize(("label", "found"), [
    ("x", True), ("ring", True), (np.str_("y"), True),  # NumPy's text scalars are text
    ("p", False), ("X", False), ("", False),
    (b"x", False), (1, False), (None, False), (True, False), (("x",), False),
])
def test_text_labels_match_only_equal_text(labels, label, found):
    data = frame(labels)
    assert data._backend.has_label(data._data, "feature", label) is found


@pytest.mark.parametrize(("labels", "label", "found"), [
    ([1, 2, 3], 2, True),
    ([1, 2, 3], 2.0, True),  # equal numbers of different types still match
    ([1, 2, 3], True, False),  # True is not 1
    ([2**53 + 1, 3], float(2**53), False),  # exact: 2**53 + 1 is not 2**53
    ([0.5, 1.5], 1.5, True),
    ([0.5, 1.5], "1.5", False),
])
def test_numeric_labels_keep_the_exact_rule(labels, label, found):
    data = frame(labels)
    assert data._backend.has_label(data._data, "feature", label) is found
