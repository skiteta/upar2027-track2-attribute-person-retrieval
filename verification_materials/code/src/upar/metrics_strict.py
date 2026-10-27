"""Strict v2 reconstruction of the public UPAR Track 2 retrieval metrics.

This module intentionally lives beside :mod:`upar.metrics` instead of replacing
it.  The v1 module is part of already-recorded experiment evidence; v2 tightens
the input contract and reproduces the official sample submission's float32 L1
distance path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["distance", "similarity"]
METRIC_VERSION = "codabench-18196-public-formula-reconstruction-v2"


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float | int]
    per_query: tuple[dict[str, float | int], ...]


def _require_integer_dtype(values: np.ndarray, *, name: str) -> None:
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype; booleans are not accepted")


def _require_int64_range(values: np.ndarray, *, name: str) -> None:
    if values.size == 0:
        return
    lower = int(values.min())
    upper = int(values.max())
    limits = np.iinfo(np.int64)
    if lower < limits.min or upper > limits.max:
        raise ValueError(f"{name} contains a value outside the int64 range")


def _validate_ground_truth(
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_queries = np.asarray(queries)
    raw_gallery = np.asarray(gallery_attributes)
    raw_ids = np.asarray(gallery_query_ids)

    if raw_queries.ndim != 2 or raw_gallery.ndim != 2:
        raise ValueError("queries and gallery attributes must be matrices")
    if raw_queries.shape[0] == 0:
        raise ValueError("queries are empty")
    if raw_queries.shape[1] == 0 or raw_queries.shape[1] != raw_gallery.shape[1]:
        raise ValueError("query and gallery attribute widths differ")
    if raw_gallery.shape[0] == 0:
        raise ValueError("gallery is empty")
    if raw_ids.shape != (raw_gallery.shape[0],):
        raise ValueError("gallery_query_ids has the wrong shape")

    _require_integer_dtype(raw_queries, name="queries")
    _require_integer_dtype(raw_gallery, name="gallery_attributes")
    _require_integer_dtype(raw_ids, name="gallery_query_ids")
    _require_int64_range(raw_ids, name="gallery_query_ids")

    if not np.all((raw_queries == 0) | (raw_queries == 1)):
        raise ValueError("queries must contain only binary values 0 and 1")
    if not np.all((raw_gallery == 0) | (raw_gallery == 1)):
        raise ValueError("gallery_attributes must contain only binary values 0 and 1")

    query_count = raw_queries.shape[0]
    minimum_id = int(raw_ids.min())
    maximum_id = int(raw_ids.max())
    if minimum_id < 0 or maximum_id >= query_count:
        raise ValueError("gallery_query_ids contains an out-of-range id")

    validated_queries = raw_queries.astype(np.uint8, copy=False)
    validated_gallery = raw_gallery.astype(np.uint8, copy=False)
    validated_ids = raw_ids.astype(np.int64, copy=False)
    expected_gallery = validated_queries[validated_ids]
    if not np.array_equal(validated_gallery, expected_gallery):
        mismatched = np.flatnonzero(np.any(validated_gallery != expected_gallery, axis=1))[:10].tolist()
        raise ValueError(f"gallery attributes do not match queries[gallery_query_ids] at rows {mismatched}")

    return validated_queries, validated_gallery, validated_ids


def _validate_query_indices(query_indices: np.ndarray | None, *, query_count: int) -> np.ndarray:
    if query_indices is None:
        return np.arange(query_count, dtype=np.int64)

    raw_indices = np.asarray(query_indices)
    if raw_indices.ndim != 1:
        raise ValueError("query_indices must be a one-dimensional array")
    if raw_indices.size == 0:
        raise ValueError("query_indices is empty")
    _require_integer_dtype(raw_indices, name="query_indices")
    _require_int64_range(raw_indices, name="query_indices")

    minimum_index = int(raw_indices.min())
    maximum_index = int(raw_indices.max())
    if minimum_index < 0 or maximum_index >= query_count:
        raise ValueError("query_indices contains an out-of-range index")
    validated = raw_indices.astype(np.int64, copy=False)
    if np.unique(validated).size != validated.size:
        raise ValueError("query_indices contains duplicates")
    return validated


def _validate_scores(scores: np.ndarray, *, expected_shape: tuple[int, int]) -> np.ndarray:
    raw_scores = np.asarray(scores)
    if raw_scores.shape != expected_shape:
        raise ValueError("scores must have shape selected_queries x gallery")
    if np.issubdtype(raw_scores.dtype, np.bool_) or not (
        np.issubdtype(raw_scores.dtype, np.integer) or np.issubdtype(raw_scores.dtype, np.floating)
    ):
        raise ValueError("scores must contain real numeric values; booleans are not accepted")

    if np.issubdtype(raw_scores.dtype, np.integer) and raw_scores.size:
        exact_limit = 2**53
        if int(raw_scores.min()) < -exact_limit or int(raw_scores.max()) > exact_limit:
            raise ValueError("integer scores outside [-2**53, 2**53] cannot be converted to float64 exactly")
    with np.errstate(over="ignore", invalid="ignore"):
        validated = raw_scores.astype(np.float64, copy=False)
    if not np.isfinite(validated).all():
        raise ValueError("scores contain NaN, infinity, or values outside the float64 range")
    return validated


def _normalized_degree_of_match(queries: np.ndarray, gallery_attributes: np.ndarray) -> np.ndarray:
    width = gallery_attributes.shape[1]
    query_values = queries.astype(np.int64, copy=False)
    gallery_values = gallery_attributes.astype(np.int64, copy=False)
    agreement_count = (
        width
        - query_values.sum(axis=1, dtype=np.int64)[:, None]
        - gallery_values.sum(axis=1, dtype=np.int64)[None, :]
        + 2 * (query_values @ gallery_values.T)
    )
    degree = agreement_count.astype(np.float64) / width
    mean = degree.mean(axis=1, keepdims=True)
    maximum = degree.max(axis=1, keepdims=True)
    denominator = maximum - mean
    normalized = np.zeros_like(degree, dtype=np.float64)
    regular = denominator[:, 0] > np.finfo(np.float64).eps
    normalized[regular] = np.clip((degree[regular] - mean[regular]) / denominator[regular], 0.0, 1.0)

    # The published expression is 0/0 when every gallery item is an exact
    # match.  Define that sole valid degenerate case as perfect.  A constant,
    # non-perfect gallery is outside the aligned Track 2 ground-truth contract.
    degenerate = ~regular
    invalid_degenerate = degenerate & (maximum[:, 0] != 1.0)
    if np.any(invalid_degenerate):
        rows = np.flatnonzero(invalid_degenerate)[:10].tolist()
        raise ValueError(f"mADM normalization is undefined for non-perfect constant-match rows {rows}")
    normalized[degenerate] = 1.0
    return normalized


def _query_metrics(
    order: np.ndarray,
    normalized_match: np.ndarray,
    relevant: np.ndarray,
    query_index: int,
) -> dict[str, float | int]:
    if order.shape != relevant.shape or normalized_match.shape != relevant.shape:
        raise ValueError("ranking vectors have inconsistent shapes")
    ranked_relevant = relevant[order]
    ranks = np.flatnonzero(ranked_relevant).astype(np.int64) + 1
    if ranks.size == 0:
        raise ValueError(f"query {query_index} has no correct gallery match")

    prefix = np.cumsum(normalized_match[order], dtype=np.float64)
    relevant_count = int(ranks.size)
    precision_at_relevant = np.arange(1, relevant_count + 1, dtype=np.float64) / ranks
    adm_at_relevant = prefix[ranks - 1] / ranks
    first_rank = int(ranks[0])
    last_rank = int(ranks[-1])
    return {
        "query_index": int(query_index),
        "relevant_count": relevant_count,
        "ADM": float(np.mean(adm_at_relevant)),
        "AP": float(np.mean(precision_at_relevant)),
        "Rank-1": int(first_rank <= 1),
        "Rank-5": int(first_rank <= 5),
        "Rank-10": int(first_rank <= 10),
        "INP": float(relevant_count / last_rank),
        "first_relevant_rank": first_rank,
        "last_relevant_rank": last_rank,
    }


def _summarize(per_query: list[dict[str, float | int]], gallery_size: int) -> EvaluationResult:
    if not per_query:
        raise ValueError("no queries were evaluated")
    summary: dict[str, float | int] = {
        "query_count": len(per_query),
        "gallery_size": int(gallery_size),
    }
    for source, target in (
        ("ADM", "mADM"),
        ("AP", "mAP"),
        ("Rank-1", "Rank-1"),
        ("Rank-5", "Rank-5"),
        ("Rank-10", "Rank-10"),
        ("INP", "mINP"),
    ):
        summary[target] = float(np.mean([float(row[source]) for row in per_query]))
    return EvaluationResult(summary=summary, per_query=tuple(per_query))


def _evaluate_validated_scores(
    scores: np.ndarray,
    *,
    direction: Direction,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray,
) -> EvaluationResult:
    if direction not in ("distance", "similarity"):
        raise ValueError("direction must be distance or similarity")
    validated_scores = _validate_scores(scores, expected_shape=(query_indices.size, gallery_attributes.shape[0]))
    selected_queries = queries[query_indices]
    normalized_match = _normalized_degree_of_match(selected_queries, gallery_attributes)

    # Convert before negation so unsigned/signed integer inputs cannot wrap.
    ranked_values = validated_scores if direction == "distance" else -validated_scores
    orders = np.argsort(ranked_values, axis=1, kind="stable")
    per_query: list[dict[str, float | int]] = []
    for local_index, query_index in enumerate(query_indices.tolist()):
        relevant = gallery_query_ids == query_index
        per_query.append(_query_metrics(orders[local_index], normalized_match[local_index], relevant, query_index))
    return _summarize(per_query, gallery_attributes.shape[0])


def evaluate_scores(
    scores: np.ndarray,
    *,
    direction: Direction,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray | None = None,
) -> EvaluationResult:
    validated_queries, validated_gallery, validated_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    validated_indices = _validate_query_indices(query_indices, query_count=validated_queries.shape[0])
    return _evaluate_validated_scores(
        scores,
        direction=direction,
        queries=validated_queries,
        gallery_attributes=validated_gallery,
        gallery_query_ids=validated_ids,
        query_indices=validated_indices,
    )


def _official_float32_l1_distances(probabilities: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Match the arithmetic and dtype used by the organizer's sample ``run.py``."""

    if probabilities.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("official L1 distance inputs must be float32")
    return probabilities.sum(axis=1)[None, :] + queries @ (1.0 - 2.0 * probabilities).T


def evaluate_attribute_probabilities(
    probabilities: np.ndarray,
    *,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray | None = None,
    chunk_size: int = 64,
) -> EvaluationResult:
    validated_queries, validated_gallery, validated_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    raw_probabilities = np.asarray(probabilities)
    if raw_probabilities.shape != validated_gallery.shape:
        raise ValueError("probabilities must have shape gallery x attributes")
    if np.issubdtype(raw_probabilities.dtype, np.bool_) or not (
        np.issubdtype(raw_probabilities.dtype, np.integer)
        or np.issubdtype(raw_probabilities.dtype, np.floating)
    ):
        raise ValueError("probabilities must contain real numeric values; booleans are not accepted")
    if not np.isfinite(raw_probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any(raw_probabilities < 0.0) or np.any(raw_probabilities > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        validated_probabilities = raw_probabilities.astype(np.float32, copy=False)
    if not np.isfinite(validated_probabilities).all():
        raise ValueError("probabilities contain values outside the float32 range")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    validated_indices = _validate_query_indices(query_indices, query_count=validated_queries.shape[0])

    per_query: list[dict[str, float | int]] = []
    for start in range(0, validated_indices.size, int(chunk_size)):
        chunk_indices = validated_indices[start : start + int(chunk_size)]
        chunk_queries = validated_queries[chunk_indices].astype(np.float32, copy=False)
        distances = _official_float32_l1_distances(validated_probabilities, chunk_queries)
        result = _evaluate_validated_scores(
            distances,
            direction="distance",
            queries=validated_queries,
            gallery_attributes=validated_gallery,
            gallery_query_ids=validated_ids,
            query_indices=chunk_indices,
        )
        per_query.extend(result.per_query)
    return _summarize(per_query, validated_gallery.shape[0])


def evaluate_score_matrix(
    scores: np.ndarray,
    *,
    direction: Direction,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray | None = None,
    chunk_size: int = 64,
) -> EvaluationResult:
    """Evaluate a possibly memory-mapped Q x G score matrix in query chunks."""

    validated_queries, validated_gallery, validated_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    validated_indices = _validate_query_indices(query_indices, query_count=validated_queries.shape[0])
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if scores.shape != (validated_queries.shape[0], validated_gallery.shape[0]):
        raise ValueError("scores must have shape all_queries x gallery")

    per_query: list[dict[str, float | int]] = []
    for start in range(0, validated_indices.size, int(chunk_size)):
        chunk_indices = validated_indices[start : start + int(chunk_size)]
        chunk_scores = np.asarray(scores[chunk_indices])
        result = _evaluate_validated_scores(
            chunk_scores,
            direction=direction,
            queries=validated_queries,
            gallery_attributes=validated_gallery,
            gallery_query_ids=validated_ids,
            query_indices=chunk_indices,
        )
        per_query.extend(result.per_query)
    return _summarize(per_query, validated_gallery.shape[0])
