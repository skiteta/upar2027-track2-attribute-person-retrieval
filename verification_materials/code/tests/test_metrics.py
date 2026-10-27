import numpy as np
import pytest

from upar.metrics import evaluate_scores


def _single_query(order_scores: list[float]):
    queries = np.asarray([[1, 1], [1, 0], [0, 0]], dtype=np.uint8)
    gallery = np.asarray([[1, 1], [1, 0], [0, 0]], dtype=np.uint8)
    ids = np.asarray([0, 1, 2], dtype=np.int64)
    return evaluate_scores(
        np.asarray([order_scores], dtype=np.float64),
        direction="distance",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        query_indices=np.asarray([0]),
    )


def test_perfect_ranking() -> None:
    result = _single_query([0.0, 1.0, 2.0])
    assert result.summary["mADM"] == pytest.approx(1.0)
    assert result.summary["mAP"] == pytest.approx(1.0)
    assert result.summary["mINP"] == pytest.approx(1.0)
    assert result.summary["Rank-1"] == pytest.approx(1.0)


def test_perturbed_ranking_degrades_in_published_direction() -> None:
    result = _single_query([1.0, 0.0, 2.0])
    assert result.summary["mADM"] == pytest.approx(0.5)
    assert result.summary["mAP"] == pytest.approx(0.5)
    assert result.summary["mINP"] == pytest.approx(0.5)
    assert result.summary["Rank-1"] == pytest.approx(0.0)
    assert result.summary["Rank-5"] == pytest.approx(1.0)


def test_equal_scores_keep_gallery_order() -> None:
    result = _single_query([0.0, 0.0, 0.0])
    assert result.per_query[0]["first_relevant_rank"] == 1


def test_multiple_matches() -> None:
    queries = np.asarray([[1, 1], [1, 0], [0, 0]], dtype=np.uint8)
    gallery = np.asarray([[1, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint8)
    ids = np.asarray([0, 1, 0, 2], dtype=np.int64)
    result = evaluate_scores(
        np.asarray([[0.0, 1.0, 2.0, 3.0]]),
        direction="distance",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        query_indices=np.asarray([0]),
    )
    assert result.summary["mADM"] == pytest.approx(5.0 / 6.0)
    assert result.summary["mAP"] == pytest.approx(5.0 / 6.0)
    assert result.summary["mINP"] == pytest.approx(2.0 / 3.0)


def test_nonfinite_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        _single_query([0.0, float("nan"), 2.0])


def test_degenerate_all_exact_gallery_uses_local_perfect_convention() -> None:
    queries = np.asarray([[1, 1]], dtype=np.uint8)
    gallery = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
    ids = np.asarray([0, 0], dtype=np.int64)
    result = evaluate_scores(
        np.asarray([[0.0, 1.0]]),
        direction="distance",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
    )
    assert result.summary["mADM"] == pytest.approx(1.0)


def test_published_metric_hand_calculation() -> None:
    queries = np.asarray(
        [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
        dtype=np.uint8,
    )
    gallery = np.asarray(
        [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
        dtype=np.uint8,
    )
    ids = np.asarray([0, 1, 2, 3, 0], dtype=np.int64)
    result = evaluate_scores(
        np.arange(5, dtype=np.float64)[None, :],
        direction="distance",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        query_indices=np.asarray([0]),
    )
    assert result.summary["mADM"] == pytest.approx(0.7375)
    assert result.summary["mAP"] == pytest.approx(0.7)
    assert result.summary["mINP"] == pytest.approx(0.4)
    assert result.summary["Rank-1"] == pytest.approx(1.0)
