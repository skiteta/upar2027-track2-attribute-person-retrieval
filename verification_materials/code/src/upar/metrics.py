from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["distance", "similarity"]


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float | int]
    per_query: tuple[dict[str, float | int], ...]


def _validate_ground_truth(
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(queries, dtype=np.uint8)
    gallery_attributes = np.asarray(gallery_attributes, dtype=np.uint8)
    gallery_query_ids = np.asarray(gallery_query_ids, dtype=np.int64)
    if queries.ndim != 2 or gallery_attributes.ndim != 2:
        raise ValueError("queries and gallery attributes must be matrices")
    if queries.shape[1] == 0 or queries.shape[1] != gallery_attributes.shape[1]:
        raise ValueError("query and gallery attribute widths differ")
    if gallery_query_ids.shape != (gallery_attributes.shape[0],):
        raise ValueError("gallery_query_ids has the wrong shape")
    if np.any((queries != 0) & (queries != 1)) or np.any((gallery_attributes != 0) & (gallery_attributes != 1)):
        raise ValueError("ground-truth attributes must be binary")
    if gallery_query_ids.size == 0:
        raise ValueError("gallery is empty")
    if gallery_query_ids.min() < 0 or gallery_query_ids.max() >= queries.shape[0]:
        raise ValueError("gallery_query_ids contains an out-of-range id")
    return queries, gallery_attributes, gallery_query_ids


def _query_metrics(order: np.ndarray, normalized_match: np.ndarray, relevant: np.ndarray, query_index: int) -> dict[str, float | int]:
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


def _normalized_degree_of_match(queries: np.ndarray, gallery_attributes: np.ndarray) -> np.ndarray:
    width = gallery_attributes.shape[1]
    agreement_count = (
        width
        - queries.sum(axis=1, dtype=np.int64)[:, None]
        - gallery_attributes.sum(axis=1, dtype=np.int64)[None, :]
        + 2 * (queries.astype(np.int16) @ gallery_attributes.astype(np.int16).T)
    )
    degree = agreement_count.astype(np.float64) / width
    mean = degree.mean(axis=1, keepdims=True)
    maximum = degree.max(axis=1, keepdims=True)
    denominator = maximum - mean
    normalized = np.zeros_like(degree, dtype=np.float64)
    regular = denominator[:, 0] > np.finfo(np.float64).eps
    normalized[regular] = np.maximum(0.0, (degree[regular] - mean[regular]) / denominator[regular])

    # The published formula is 0/0 when the whole gallery is equally perfect.
    # Returning one is the continuous perfect-ranking convention. Real challenge
    # galleries contain non-matches; callers should treat this only as a local edge policy.
    degenerate = ~regular
    if np.any(degenerate):
        normalized[degenerate] = (degree[degenerate] == maximum[degenerate]).astype(np.float64)
    return normalized


def evaluate_scores(
    scores: np.ndarray,
    *,
    direction: Direction,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray | None = None,
) -> EvaluationResult:
    queries, gallery_attributes, gallery_query_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    scores = np.asarray(scores)
    if query_indices is None:
        query_indices = np.arange(queries.shape[0], dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)
    if scores.shape != (query_indices.size, gallery_attributes.shape[0]):
        raise ValueError("scores must have shape selected_queries x gallery")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    if direction not in ("distance", "similarity"):
        raise ValueError("direction must be distance or similarity")
    if query_indices.size == 0 or query_indices.min() < 0 or query_indices.max() >= queries.shape[0]:
        raise ValueError("query_indices is empty or out of range")

    selected_queries = queries[query_indices]
    normalized_match = _normalized_degree_of_match(selected_queries, gallery_attributes)
    ranked_values = scores if direction == "distance" else -scores
    orders = np.argsort(ranked_values, axis=1, kind="stable")
    per_query: list[dict[str, float | int]] = []
    for local_index, query_index in enumerate(query_indices.tolist()):
        relevant = gallery_query_ids == query_index
        per_query.append(_query_metrics(orders[local_index], normalized_match[local_index], relevant, query_index))
    return _summarize(per_query, gallery_attributes.shape[0])


def evaluate_attribute_probabilities(
    probabilities: np.ndarray,
    *,
    queries: np.ndarray,
    gallery_attributes: np.ndarray,
    gallery_query_ids: np.ndarray,
    query_indices: np.ndarray | None = None,
    chunk_size: int = 64,
) -> EvaluationResult:
    queries, gallery_attributes, gallery_query_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != gallery_attributes.shape:
        raise ValueError("probabilities must have shape gallery x attributes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if query_indices is None:
        query_indices = np.arange(queries.shape[0], dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)
    if query_indices.size == 0 or query_indices.min() < 0 or query_indices.max() >= queries.shape[0]:
        raise ValueError("query_indices is empty or out of range")

    probability_sum = probabilities.sum(axis=1, dtype=np.float64)
    transformed = 1.0 - 2.0 * probabilities.astype(np.float64)
    per_query: list[dict[str, float | int]] = []
    for start in range(0, query_indices.size, chunk_size):
        chunk_indices = query_indices[start : start + chunk_size]
        chunk_queries = queries[chunk_indices]
        distances = probability_sum[None, :] + chunk_queries.astype(np.float64) @ transformed.T
        result = evaluate_scores(
            distances,
            direction="distance",
            queries=queries,
            gallery_attributes=gallery_attributes,
            gallery_query_ids=gallery_query_ids,
            query_indices=chunk_indices,
        )
        per_query.extend(result.per_query)
    return _summarize(per_query, gallery_attributes.shape[0])


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
    queries, gallery_attributes, gallery_query_ids = _validate_ground_truth(
        queries, gallery_attributes, gallery_query_ids
    )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if query_indices is None:
        query_indices = np.arange(queries.shape[0], dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)
    if query_indices.size == 0 or query_indices.min() < 0 or query_indices.max() >= queries.shape[0]:
        raise ValueError("query_indices is empty or out of range")
    if scores.shape != (queries.shape[0], gallery_attributes.shape[0]):
        raise ValueError("scores must have shape all_queries x gallery")
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("scores must be numeric")

    per_query: list[dict[str, float | int]] = []
    for start in range(0, query_indices.size, chunk_size):
        chunk_indices = query_indices[start : start + chunk_size]
        chunk_scores = np.asarray(scores[chunk_indices])
        result = evaluate_scores(
            chunk_scores,
            direction=direction,
            queries=queries,
            gallery_attributes=gallery_attributes,
            gallery_query_ids=gallery_query_ids,
            query_indices=chunk_indices,
        )
        per_query.extend(result.per_query)
    return _summarize(per_query, gallery_attributes.shape[0])
