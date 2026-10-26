"""The contraction behind ``dot`` (``labelarray.contract``, plan milestone 10, step B3)."""

import numpy as np
import pytest

from tidy3a.labelarray.contract import contract


def _einsum(left, left_dims, right, right_dims, over):
    names = list(dict.fromkeys([*left_dims, *right_dims]))
    letters = {d: chr(ord("a") + k) for k, d in enumerate(names)}
    kept = [d for d in names if d not in over]
    spec = f"{''.join(letters[d] for d in left_dims)},{''.join(letters[d] for d in right_dims)}->"
    return np.einsum(spec + "".join(letters[d] for d in kept), left, right), tuple(kept)


@pytest.mark.parametrize("seed", range(40))
def test_axes_and_values_match_einsum(seed):
    rng = np.random.default_rng(seed)
    sizes = dict(zip("abcde", rng.integers(1, 5, 5)))
    names = list(sizes)
    left_dims = tuple(rng.permutation(names)[: rng.integers(1, 4)])
    right_dims = tuple(rng.permutation(names)[: rng.integers(1, 4)])
    shared = [d for d in left_dims if d in right_dims]
    over = tuple(d for d in shared if rng.random() < 0.6)
    left = rng.normal(size=[sizes[d] for d in left_dims])
    right = rng.normal(size=[sizes[d] for d in right_dims])
    got, dims = contract(left, left_dims, right, right_dims, over)
    want, want_dims = _einsum(left, left_dims, right, right_dims, over)
    assert dims == want_dims and got.shape == want.shape and got.flags.c_contiguous
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_results_do_not_depend_on_operand_layout():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(3, 3)).astype("float32")
    points = rng.uniform(-80, 80, (10_000, 3)).astype("float32")
    results = []
    for layout in (points, np.asfortranarray(points), np.ascontiguousarray(points.T).T, points[::-1][::-1]):
        for rotation in (matrix, np.asfortranarray(matrix)):
            results.append(contract(rotation, ("i", "feature"), layout, ("point", "feature"), ("feature",))[0])
    for other in results[1:]:
        assert np.array_equal(results[0].view(np.uint32), other.view(np.uint32))


def test_a_full_contraction_is_a_scalar():
    value, dims = contract(np.array([1j, 0]), ("k",), np.array([1j, 0]), ("k",), ("k",))
    assert dims == () and value.shape == () and value == -1


@pytest.mark.parametrize("t", [1, 3])
def test_batch_axes_kept_by_both_operands(t):
    rng = np.random.default_rng(1)
    left = rng.normal(size=(t, 3, 4))
    right = rng.normal(size=(4, t, 5))
    got, dims = contract(left, ("t", "i", "c"), right, ("c", "t", "j"), ("c",))
    want, want_dims = _einsum(left, ("t", "i", "c"), right, ("c", "t", "j"), ("c",))
    assert dims == want_dims == ("t", "i", "j")
    np.testing.assert_allclose(got, want, rtol=1e-12)
