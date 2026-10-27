from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .blend_predictions import SOURCES, atomic_json, load_prediction, sha256_file
from .data import RetrievalSplit, load_retrieval_split
from .metrics_strict import EvaluationResult, _official_float32_l1_distances, evaluate_scores
from .positive_query_weight import _b008_lodo_predictions, _summarize

DEFAULT_GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def balanced_brier_reliability(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or probabilities.dtype != np.float32:
        raise ValueError("probabilities must be a float32 matrix")
    if labels.shape != probabilities.shape or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be an aligned binary matrix")
    if not np.isfinite(probabilities).all() or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError("probabilities must be finite and in [0,1]")

    labels_bool = labels.astype(bool, copy=False)
    reliability = np.empty(probabilities.shape[1], dtype=np.float32)
    for index in range(probabilities.shape[1]):
        positive = labels_bool[:, index]
        negative = ~positive
        if not positive.any() or not negative.any():
            raise ValueError(f"attribute {index} lacks a positive or negative reliability example")
        positive_mse = np.mean((np.float32(1.0) - probabilities[positive, index]) ** 2)
        negative_mse = np.mean(probabilities[negative, index] ** 2)
        balanced_brier = np.float32(0.5) * (positive_mse + negative_mse)
        reliability[index] = max(float(np.float32(1.0) - balanced_brier), 1e-6)
    return reliability


def reliability_weights(reliability: np.ndarray, *, gamma: float) -> np.ndarray:
    reliability = np.asarray(reliability)
    if reliability.ndim != 1 or reliability.dtype != np.float32:
        raise ValueError("reliability must be a float32 vector")
    if not np.isfinite(reliability).all() or np.any(reliability <= 0):
        raise ValueError("reliability must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if gamma == 0.0:
        return np.ones_like(reliability, dtype=np.float32)
    weights = np.power(reliability, np.float32(gamma)).astype(np.float32, copy=False)
    return weights / np.mean(weights, dtype=np.float32)


def weighted_l1_distances(
    probabilities: np.ndarray,
    queries: np.ndarray,
    *,
    weights: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    queries = np.asarray(queries)
    weights = np.asarray(weights)
    if probabilities.ndim != 2 or queries.ndim != 2:
        raise ValueError("probabilities and queries must be matrices")
    if probabilities.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("probabilities and queries must be float32")
    if weights.dtype != np.float32 or weights.shape != (probabilities.shape[1],):
        raise ValueError("weights must be an aligned float32 vector")
    if probabilities.shape[1] != queries.shape[1]:
        raise ValueError("probability and query widths differ")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    if np.array_equal(weights, np.ones_like(weights)):
        return _official_float32_l1_distances(probabilities, queries)
    base = (probabilities * weights).sum(axis=1, dtype=np.float32)[None, :]
    transformed = weights * (np.float32(1.0) - np.float32(2.0) * probabilities)
    return base + queries @ transformed.T


def select_gamma(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("gamma search produced no rows")
    return max(rows, key=lambda row: (float(row["macro_mADM"]), -float(row["gamma"])))


def evaluate_weighted(
    probabilities: np.ndarray,
    *,
    weights: np.ndarray,
    split: RetrievalSplit,
    query_indices: np.ndarray,
    chunk_size: int,
) -> EvaluationResult:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    per_query: list[dict[str, float | int]] = []
    for start in range(0, len(query_indices), chunk_size):
        indices = query_indices[start : start + chunk_size]
        distances = weighted_l1_distances(
            probabilities,
            split.queries[indices].astype(np.float32, copy=False),
            weights=weights,
        )
        result = evaluate_scores(
            distances,
            direction="distance",
            queries=split.queries,
            gallery_attributes=split.gallery_attributes,
            gallery_query_ids=split.gallery_query_ids,
            query_indices=indices,
        )
        per_query.extend(result.per_query)
    return _summarize(per_query, split.gallery_size)


def _write_full_distances(
    path: Path,
    *,
    probabilities: np.ndarray,
    queries: np.ndarray,
    weights: np.ndarray,
    chunk_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), len(probabilities)),
        )
        for start in range(0, len(queries), chunk_size):
            stop = min(start + chunk_size, len(queries))
            output[start:stop] = weighted_l1_distances(
                probabilities,
                queries[start:stop].astype(np.float32, copy=False),
                weights=weights,
            )
        output.flush()
        del output
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_search(
    *,
    split_dir: Path,
    b008_all_path: Path,
    b008_search_path: Path,
    gammas: tuple[float, ...],
    output_search: Path,
    output_distances: Path,
    chunk_size: int,
) -> dict[str, Any]:
    if not gammas or tuple(sorted(set(gammas))) != gammas:
        raise ValueError("gammas must be unique and sorted")
    if 0.0 not in gammas or any(not math.isfinite(x) or x < 0 for x in gammas):
        raise ValueError("gammas must be nonnegative and include control 0")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    started = perf_counter()
    split = load_retrieval_split(split_dir)
    search = json.loads(b008_search_path.read_text(encoding="utf-8"))
    if search.get("status") != "SELECTED" or "selected" not in search:
        raise ValueError("B008 blend search is not a selected result")
    lodo_predictions, lodo_evidence = _b008_lodo_predictions(search, split)

    reliabilities: dict[str, np.ndarray] = {}
    fit_records: dict[str, dict[str, Any]] = {}
    for target in SOURCES:
        fit_mask = split.sources != target
        target_mask = split.sources == target
        reliability = balanced_brier_reliability(
            lodo_predictions[target][fit_mask],
            split.gallery_attributes[fit_mask],
        )
        reliabilities[target] = reliability
        fit_records[target] = {
            "fit_sources": [source for source in SOURCES if source != target],
            "fit_row_count": int(fit_mask.sum()),
            "held_source": target,
            "held_row_count": int(target_mask.sum()),
            "reliability": reliability.tolist(),
        }

    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        source_results: dict[str, dict[str, float | int]] = {}
        source_weight_stats: dict[str, dict[str, float]] = {}
        for source in SOURCES:
            mask = split.sources == source
            source_split = split.subset_gallery(mask)
            weights = reliability_weights(reliabilities[source], gamma=gamma)
            result = evaluate_weighted(
                lodo_predictions[source][mask],
                weights=weights,
                split=source_split,
                query_indices=np.unique(source_split.gallery_query_ids),
                chunk_size=chunk_size,
            )
            source_results[source] = result.summary
            source_weight_stats[source] = {
                "minimum": float(weights.min()),
                "maximum": float(weights.max()),
                "mean": float(weights.mean(dtype=np.float64)),
            }
        rows.append(
            {
                "gamma": gamma,
                "macro_mADM": float(
                    np.mean([float(source_results[source]["mADM"]) for source in SOURCES])
                ),
                "sources": source_results,
                "weight_stats": source_weight_stats,
            }
        )

    selected = select_gamma(rows)
    aggregate_reliability = np.exp(
        np.mean(np.log(np.stack([reliabilities[source] for source in SOURCES])), axis=0)
    ).astype(np.float32)
    selected_weights = reliability_weights(
        aggregate_reliability,
        gamma=float(selected["gamma"]),
    )
    b008_all = load_prediction(b008_all_path)
    if b008_all["sha256"] != search.get("output_prediction", {}).get("sha256"):
        raise ValueError("B008 all-domain prediction differs from its selection record")
    if tuple(b008_all["image_paths"].tolist()) != split.image_paths:
        raise ValueError("B008 all-domain image order differs from the evaluation split")
    if tuple(b008_all["attribute_names"].tolist()) != split.attribute_names:
        raise ValueError("B008 all-domain attributes differ from the evaluation split")
    _write_full_distances(
        output_distances,
        probabilities=b008_all["probabilities"],
        queries=split.queries,
        weights=selected_weights,
        chunk_size=chunk_size,
    )
    elapsed = perf_counter() - started
    report = {
        "schema": 1,
        "status": "SELECTED",
        "candidate": "B011_B008_CROSSFIT_ATTRIBUTE_RELIABILITY",
        "reliability": "1 - balanced Brier, fit on non-target sources only",
        "weight": "reliability ** gamma normalized to mean one",
        "distance": "symmetric attribute-weighted float32 L1",
        "selection_surface": "three held-source B008 LODO mADM with equal source weight",
        "gamma_grid": list(gammas),
        "rows": rows,
        "selected": selected,
        "crossfit": fit_records,
        "full_weight": {
            "aggregation": "attribute-wise geometric mean of three cross-fitted reliability vectors",
            "reliability": aggregate_reliability.tolist(),
            "weights": selected_weights.tolist(),
            "attribute_names": list(split.attribute_names),
        },
        "inputs": {
            "split_dir": str(split_dir),
            "b008_all": {"path": str(b008_all_path), "sha256": b008_all["sha256"]},
            "b008_search": {"path": str(b008_search_path), "sha256": sha256_file(b008_search_path)},
            "lodo_components": lodo_evidence,
        },
        "output_distances": {
            "path": str(output_distances),
            "sha256": sha256_file(output_distances),
            "shape": [split.query_count, split.gallery_size],
            "dtype": "float32",
        },
        "runtime": {"job_seconds": elapsed, "gpu_seconds": 0.0},
        "limitations": [
            "Reliability uses exposed development labels, though each held source is excluded from its own weight fit.",
            "The full weight aggregates cross-fitted vectors and is development-selected, not independent confirmation.",
            "Balanced Brier reliability is one fixed proxy and is not the official retrieval objective.",
            "The fixed gamma grid does not establish optimality beyond the tested values.",
        ],
    }
    atomic_json(output_search, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fit attribute reliability weights for B008")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--b008-all", type=Path, required=True)
    parser.add_argument("--b008-search", type=Path, required=True)
    parser.add_argument("--output-search", type=Path, required=True)
    parser.add_argument("--output-distances", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    result = run_search(
        split_dir=args.split_dir,
        b008_all_path=args.b008_all,
        b008_search_path=args.b008_search,
        gammas=DEFAULT_GAMMAS,
        output_search=args.output_search,
        output_distances=args.output_distances,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({"status": result["status"], "selected": result["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
