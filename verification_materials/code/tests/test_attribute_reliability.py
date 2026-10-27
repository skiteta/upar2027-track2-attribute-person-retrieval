from __future__ import annotations

import numpy as np
import pytest

from upar.attribute_reliability import (
    balanced_brier_reliability,
    reliability_weights,
    select_gamma,
    weighted_l1_distances,
)
from upar.metrics_strict import _official_float32_l1_distances


def test_balanced_brier_gives_equal_class_weight() -> None:
    probabilities = np.asarray([[0.8], [0.6], [0.2], [0.1], [0.1]], dtype=np.float32)
    labels = np.asarray([[1], [1], [0], [0], [0]], dtype=np.uint8)
    actual = balanced_brier_reliability(probabilities, labels)
    positive_mse = ((0.2**2) + (0.4**2)) / 2
    negative_mse = ((0.2**2) + (0.1**2) + (0.1**2)) / 3
    assert actual[0] == pytest.approx(1.0 - 0.5 * (positive_mse + negative_mse))


def test_gamma_zero_is_exact_uniform_l1_control() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    reliability = np.asarray([0.6, 0.9], dtype=np.float32)
    weights = reliability_weights(reliability, gamma=0.0)
    assert np.array_equal(weights, np.ones(2, dtype=np.float32))
    assert np.array_equal(
        weighted_l1_distances(probabilities, queries, weights=weights),
        _official_float32_l1_distances(probabilities, queries),
    )


def test_weighted_l1_matches_direct_absolute_difference() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    weights = np.asarray([0.5, 1.5], dtype=np.float32)
    actual = weighted_l1_distances(probabilities, queries, weights=weights)
    expected = np.empty_like(actual)
    for index, query in enumerate(queries):
        expected[index] = (np.abs(probabilities - query) * weights).sum(axis=1)
    assert np.allclose(actual, expected, rtol=0, atol=2e-7)


def test_reliability_weights_are_positive_and_mean_one() -> None:
    reliability = np.asarray([0.5, 0.75, 0.9], dtype=np.float32)
    weights = reliability_weights(reliability, gamma=2.0)
    assert np.all(weights > 0)
    assert float(weights.mean()) == pytest.approx(1.0)


def test_selection_prefers_macro_then_smaller_gamma() -> None:
    selected = select_gamma(
        [
            {"gamma": 0.0, "macro_mADM": 0.4},
            {"gamma": 1.0, "macro_mADM": 0.41},
            {"gamma": 2.0, "macro_mADM": 0.41},
        ]
    )
    assert selected["gamma"] == 1.0


@pytest.mark.parametrize("gamma", [-1.0, float("nan"), float("inf")])
def test_invalid_gamma_is_rejected(gamma: float) -> None:
    reliability = np.asarray([0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="gamma"):
        reliability_weights(reliability, gamma=gamma)
