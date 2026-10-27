import numpy as np
import pytest

from upar.metrics_strict import (
    _official_float32_l1_distances,
    evaluate_attribute_probabilities,
    evaluate_score_matrix,
    evaluate_scores,
)


def _ground_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    gallery = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    ids = np.asarray([0, 1], dtype=np.int64)
    return queries, gallery, ids


def test_uint8_similarity_is_converted_before_negation() -> None:
    queries, gallery, ids = _ground_truth()
    result = evaluate_scores(
        np.asarray([[255, 0]], dtype=np.uint8),
        direction="similarity",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        query_indices=np.asarray([0]),
    )
    assert result.per_query[0]["first_relevant_rank"] == 1


def test_similarity_ties_preserve_gallery_order() -> None:
    queries, gallery, ids = _ground_truth()
    result = evaluate_scores(
        np.asarray([[7, 7]], dtype=np.uint8),
        direction="similarity",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        query_indices=np.asarray([1]),
    )
    assert result.per_query[0]["first_relevant_rank"] == 2


@pytest.mark.parametrize(
    ("queries", "gallery", "ids", "message"),
    [
        (
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([0]),
            "queries must use an integer dtype",
        ),
        (
            np.asarray([[True, False]]),
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([0]),
            "booleans are not accepted",
        ),
        (
            np.asarray([[256, 0]], dtype=np.uint16),
            np.asarray([[0, 0]], dtype=np.uint8),
            np.asarray([0]),
            "queries must contain only binary values",
        ),
        (
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([0.0]),
            "gallery_query_ids must use an integer dtype",
        ),
        (
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([True]),
            "booleans are not accepted",
        ),
        (
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([[1, 0]], dtype=np.uint8),
            np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
            "outside the int64 range",
        ),
    ],
)
def test_malformed_ground_truth_is_rejected_before_cast(
    queries: np.ndarray,
    gallery: np.ndarray,
    ids: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_scores(
            np.asarray([[0.0]]),
            direction="distance",
            queries=queries,
            gallery_attributes=gallery,
            gallery_query_ids=ids,
        )


def test_gallery_attributes_must_match_query_ids() -> None:
    queries, gallery, ids = _ground_truth()
    with pytest.raises(ValueError, match=r"queries\[gallery_query_ids\]"):
        evaluate_scores(
            np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            direction="distance",
            queries=queries,
            gallery_attributes=gallery[::-1],
            gallery_query_ids=ids,
        )


@pytest.mark.parametrize(
    "query_indices",
    [
        np.asarray([0.9]),
        np.asarray([True]),
        np.asarray([[0]]),
        np.asarray([0, 0]),
        np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
    ],
)
def test_malformed_query_indices_are_rejected(query_indices: np.ndarray) -> None:
    queries, gallery, ids = _ground_truth()
    with pytest.raises(ValueError):
        evaluate_scores(
            np.zeros((query_indices.size, gallery.shape[0]), dtype=np.float32),
            direction="distance",
            queries=queries,
            gallery_attributes=gallery,
            gallery_query_ids=ids,
            query_indices=query_indices,
        )


def test_probability_distance_path_exactly_matches_official_float32_expression() -> None:
    probabilities = np.asarray(
        [[0.1, 0.9, 0.25], [0.7, 0.2, 0.5], [0.3, 0.4, 0.8]],
        dtype=np.float32,
    )
    queries = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.float32)
    expected = probabilities.sum(1)[None, :] + queries @ (1.0 - 2.0 * probabilities).T
    actual = _official_float32_l1_distances(probabilities, queries)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_probability_and_saved_score_paths_agree_across_chunks() -> None:
    queries, gallery, ids = _ground_truth()
    probabilities = np.asarray([[0.8, 0.7], [0.1, 0.2]], dtype=np.float32)
    distances = _official_float32_l1_distances(probabilities, queries.astype(np.float32))
    direct = evaluate_attribute_probabilities(
        probabilities,
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        chunk_size=1,
    )
    saved = evaluate_score_matrix(
        distances,
        direction="distance",
        queries=queries,
        gallery_attributes=gallery,
        gallery_query_ids=ids,
        chunk_size=1,
    )
    assert direct.summary == saved.summary
    assert direct.per_query == saved.per_query


@pytest.mark.parametrize("value", [-1e-50, 1.0 + 1e-10])
def test_probability_range_is_checked_before_float32_cast(value: float) -> None:
    queries, gallery, ids = _ground_truth()
    probabilities = np.asarray([[value, 0.5], [0.5, 0.5]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_attribute_probabilities(
            probabilities,
            queries=queries,
            gallery_attributes=gallery,
            gallery_query_ids=ids,
        )


def test_all_exact_gallery_has_explicit_perfect_madm_convention() -> None:
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
