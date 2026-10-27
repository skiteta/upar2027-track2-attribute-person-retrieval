from __future__ import annotations

import numpy as np
import pytest

from upar.attribute_reliability import weighted_l1_distances
from upar.bernoulli_distance import (
    mixed_distances,
    select_lambda,
    weighted_bernoulli_nll_distances,
)


def test_weighted_nll_matches_direct_binary_cross_entropy() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    weights = np.asarray([0.5, 1.5], dtype=np.float32)
    actual = weighted_bernoulli_nll_distances(probabilities, queries, weights=weights)
    expected = np.empty_like(actual)
    for index, query in enumerate(queries):
        expected[index] = (
            -(query * np.log(probabilities) + (1 - query) * np.log1p(-probabilities)) * weights
        ).sum(axis=1)
    assert np.allclose(actual, expected, rtol=0, atol=5e-7)


def test_lambda_zero_is_exact_b011_weighted_l1_control() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    weights = np.asarray([0.5, 1.5], dtype=np.float32)
    expected = weighted_l1_distances(probabilities, queries, weights=weights)
    actual = mixed_distances(probabilities, queries, weights=weights, nll_lambda=0.0)
    assert np.array_equal(actual, expected)


def test_lambda_one_is_exact_weighted_nll() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    weights = np.asarray([0.5, 1.5], dtype=np.float32)
    expected = weighted_bernoulli_nll_distances(probabilities, queries, weights=weights)
    actual = mixed_distances(probabilities, queries, weights=weights, nll_lambda=1.0)
    assert np.array_equal(actual, expected)


def test_selection_prefers_macro_then_smaller_lambda() -> None:
    selected = select_lambda(
        [
            {"lambda": 0.0, "macro_mADM": 0.4},
            {"lambda": 0.25, "macro_mADM": 0.41},
            {"lambda": 0.5, "macro_mADM": 0.41},
        ]
    )
    assert selected["lambda"] == 0.25


@pytest.mark.parametrize("value", [-1.0, 1.1, float("nan"), float("inf")])
def test_invalid_lambda_is_rejected(value: float) -> None:
    probabilities = np.asarray([[0.1]], dtype=np.float32)
    queries = np.asarray([[1]], dtype=np.float32)
    weights = np.asarray([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="nll_lambda"):
        mixed_distances(probabilities, queries, weights=weights, nll_lambda=value)
