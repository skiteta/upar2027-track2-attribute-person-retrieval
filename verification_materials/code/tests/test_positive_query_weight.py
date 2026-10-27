from __future__ import annotations

import numpy as np
import pytest

from upar.metrics_strict import _official_float32_l1_distances
from upar.positive_query_weight import positive_weighted_l1_distances, select_positive_weight


def test_uniform_control_is_bit_exact_official_float32_l1() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9, 0.25], [0.7, 0.2, 0.5], [0.3, 0.4, 0.8]],
        dtype=np.float32,
    )
    queries = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.float32)
    expected = _official_float32_l1_distances(probabilities, queries)
    actual = positive_weighted_l1_distances(probabilities, queries, positive_weight=1.0)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_positive_weight_matches_direct_weighted_absolute_difference() -> None:
    probabilities = np.asarray([[0.1, 0.7], [0.8, 0.2]], dtype=np.float32)
    queries = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    actual = positive_weighted_l1_distances(probabilities, queries, positive_weight=2.0)
    expected = np.empty_like(actual)
    for query_index, query in enumerate(queries):
        weights = np.where(query == 1, np.float32(2.0), np.float32(1.0))
        expected[query_index] = (np.abs(probabilities - query) * weights).sum(axis=1)
    assert np.allclose(actual, expected, rtol=0, atol=2e-7)


def test_selection_prefers_macro_then_uniform_control_proximity() -> None:
    selected = select_positive_weight(
        [
            {"positive_weight": 0.5, "macro_mADM": 0.4},
            {"positive_weight": 1.0, "macro_mADM": 0.41},
            {"positive_weight": 2.0, "macro_mADM": 0.41},
        ]
    )
    assert selected["positive_weight"] == 1.0


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_positive_weight_is_rejected(weight: float) -> None:
    probabilities = np.asarray([[0.5]], dtype=np.float32)
    queries = np.asarray([[1]], dtype=np.float32)
    with pytest.raises(ValueError, match="positive_weight"):
        positive_weighted_l1_distances(probabilities, queries, positive_weight=weight)
