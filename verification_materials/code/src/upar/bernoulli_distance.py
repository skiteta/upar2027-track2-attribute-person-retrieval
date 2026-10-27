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

from .attribute_reliability import reliability_weights, weighted_l1_distances
from .blend_predictions import SOURCES, atomic_json, load_prediction, sha256_file
from .data import RetrievalSplit, load_retrieval_split
from .metrics_strict import EvaluationResult, evaluate_scores
from .positive_query_weight import _b008_lodo_predictions, _summarize

DEFAULT_LAMBDAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
DEFAULT_EPSILON = float(np.finfo(np.float32).eps)


def weighted_bernoulli_nll_distances(
    probabilities: np.ndarray,
    queries: np.ndarray,
    *,
    weights: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    queries = np.asarray(queries)
    weights = np.asarray(weights)
    if probabilities.ndim != 2 or queries.ndim != 2:
        raise ValueError("probabilities and queries must be matrices")
    if probabilities.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("probabilities and queries must be float32")
    if probabilities.shape[1] != queries.shape[1]:
        raise ValueError("probability and query widths differ")
    if weights.dtype != np.float32 or weights.shape != (probabilities.shape[1],):
        raise ValueError("weights must be an aligned float32 vector")
    if not np.isfinite(probabilities).all() or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if not np.isfinite(queries).all() or not np.all((queries == 0) | (queries == 1)):
        raise ValueError("queries must be finite and binary")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0,0.5)")

    clipped = np.clip(probabilities, np.float32(epsilon), np.float32(1.0 - epsilon))
    negative = -np.log1p(-clipped).astype(np.float32, copy=False)
    positive = -np.log(clipped).astype(np.float32, copy=False)
    base = (negative * weights).sum(axis=1, dtype=np.float32)[None, :]
    delta = (positive - negative) * weights
    return base + queries @ delta.T


def mixed_distances(
    probabilities: np.ndarray,
    queries: np.ndarray,
    *,
    weights: np.ndarray,
    nll_lambda: float,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    if not math.isfinite(nll_lambda) or not 0 <= nll_lambda <= 1:
        raise ValueError("nll_lambda must be finite and in [0,1]")
    l1 = weighted_l1_distances(probabilities, queries, weights=weights)
    if nll_lambda == 0:
        return l1
    nll = weighted_bernoulli_nll_distances(
        probabilities,
        queries,
        weights=weights,
        epsilon=epsilon,
    )
    if nll_lambda == 1:
        return nll
    return np.float32(1.0 - nll_lambda) * l1 + np.float32(nll_lambda) * nll


def select_lambda(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("lambda search produced no rows")
    return max(rows, key=lambda row: (float(row["macro_mADM"]), -float(row["lambda"])))


def evaluate_mixed(
    probabilities: np.ndarray,
    *,
    weights: np.ndarray,
    nll_lambda: float,
    epsilon: float,
    split: RetrievalSplit,
    query_indices: np.ndarray,
    chunk_size: int,
) -> EvaluationResult:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    per_query: list[dict[str, float | int]] = []
    for start in range(0, len(query_indices), chunk_size):
        indices = query_indices[start : start + chunk_size]
        distances = mixed_distances(
            probabilities,
            split.queries[indices].astype(np.float32, copy=False),
            weights=weights,
            nll_lambda=nll_lambda,
            epsilon=epsilon,
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
    nll_lambda: float,
    epsilon: float,
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
            output[start:stop] = mixed_distances(
                probabilities,
                queries[start:stop].astype(np.float32, copy=False),
                weights=weights,
                nll_lambda=nll_lambda,
                epsilon=epsilon,
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
    b011_search_path: Path,
    lambdas: tuple[float, ...],
    epsilon: float,
    output_search: Path,
    output_distances: Path,
    chunk_size: int,
) -> dict[str, Any]:
    if not lambdas or tuple(sorted(set(lambdas))) != lambdas:
        raise ValueError("lambdas must be unique and sorted")
    if 0.0 not in lambdas or any(not math.isfinite(x) or not 0 <= x <= 1 for x in lambdas):
        raise ValueError("lambdas must be in [0,1] and include control 0")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0,0.5)")

    started = perf_counter()
    split = load_retrieval_split(split_dir)
    b008_search = json.loads(b008_search_path.read_text(encoding="utf-8"))
    b011_search = json.loads(b011_search_path.read_text(encoding="utf-8"))
    if b011_search.get("status") != "SELECTED" or "selected" not in b011_search:
        raise ValueError("B011 reliability search is not selected")
    selected_gamma = float(b011_search["selected"]["gamma"])
    lodo_predictions, lodo_evidence = _b008_lodo_predictions(b008_search, split)

    source_weights: dict[str, np.ndarray] = {}
    for source in SOURCES:
        reliability = np.asarray(b011_search["crossfit"][source]["reliability"], dtype=np.float32)
        source_weights[source] = reliability_weights(reliability, gamma=selected_gamma)

    rows: list[dict[str, Any]] = []
    for nll_lambda in lambdas:
        source_results: dict[str, dict[str, float | int]] = {}
        for source in SOURCES:
            mask = split.sources == source
            source_split = split.subset_gallery(mask)
            result = evaluate_mixed(
                lodo_predictions[source][mask],
                weights=source_weights[source],
                nll_lambda=nll_lambda,
                epsilon=epsilon,
                split=source_split,
                query_indices=np.unique(source_split.gallery_query_ids),
                chunk_size=chunk_size,
            )
            source_results[source] = result.summary
        rows.append(
            {
                "lambda": nll_lambda,
                "macro_mADM": float(
                    np.mean([float(source_results[source]["mADM"]) for source in SOURCES])
                ),
                "sources": source_results,
            }
        )

    selected = select_lambda(rows)
    full_weights = np.asarray(b011_search["full_weight"]["weights"], dtype=np.float32)
    b008_all = load_prediction(b008_all_path)
    if b008_all["sha256"] != b008_search.get("output_prediction", {}).get("sha256"):
        raise ValueError("B008 all-domain prediction differs from its selection record")
    if tuple(b008_all["image_paths"].tolist()) != split.image_paths:
        raise ValueError("B008 all-domain image order differs from the evaluation split")
    if tuple(b008_all["attribute_names"].tolist()) != split.attribute_names:
        raise ValueError("B008 all-domain attributes differ from the evaluation split")
    _write_full_distances(
        output_distances,
        probabilities=b008_all["probabilities"],
        queries=split.queries,
        weights=full_weights,
        nll_lambda=float(selected["lambda"]),
        epsilon=epsilon,
        chunk_size=chunk_size,
    )
    report = {
        "schema": 1,
        "status": "SELECTED",
        "candidate": "B012_B011_BERNOULLI_NLL_MIX",
        "baseline": "B011_B008_CROSSFIT_ATTRIBUTE_RELIABILITY",
        "distance": "(1-lambda) * B011 weighted L1 + lambda * B011 weighted Bernoulli NLL",
        "selection_surface": "three held-source B008 LODO predictions with B011 cross-fitted weights and equal source weight",
        "lambda_grid": list(lambdas),
        "epsilon": epsilon,
        "rows": rows,
        "selected": selected,
        "b011": {
            "selected_gamma": selected_gamma,
            "source_weights": {source: source_weights[source].tolist() for source in SOURCES},
            "full_weights": full_weights.tolist(),
        },
        "inputs": {
            "split_dir": str(split_dir),
            "b008_all": {"path": str(b008_all_path), "sha256": b008_all["sha256"]},
            "b008_search": {"path": str(b008_search_path), "sha256": sha256_file(b008_search_path)},
            "b011_search": {"path": str(b011_search_path), "sha256": sha256_file(b011_search_path)},
            "lodo_components": lodo_evidence,
        },
        "output_distances": {
            "path": str(output_distances),
            "sha256": sha256_file(output_distances),
            "shape": [split.query_count, split.gallery_size],
            "dtype": "float32",
        },
        "runtime": {"job_seconds": perf_counter() - started, "gpu_seconds": 0.0},
        "limitations": [
            "The public sources and B011 weights are exposed development surfaces, not independent confirmation.",
            "Bernoulli NLL assumes calibrated independent attributes and can over-penalize confident domain-shift errors.",
            "The fixed lambda grid tests only one distance-shape family and does not establish global optimality.",
        ],
    }
    atomic_json(output_search, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix B011 weighted L1 with Bernoulli NLL")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--b008-all", type=Path, required=True)
    parser.add_argument("--b008-search", type=Path, required=True)
    parser.add_argument("--b011-search", type=Path, required=True)
    parser.add_argument("--output-search", type=Path, required=True)
    parser.add_argument("--output-distances", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    result = run_search(
        split_dir=args.split_dir,
        b008_all_path=args.b008_all,
        b008_search_path=args.b008_search,
        b011_search_path=args.b011_search,
        lambdas=DEFAULT_LAMBDAS,
        epsilon=DEFAULT_EPSILON,
        output_search=args.output_search,
        output_distances=args.output_distances,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({"status": result["status"], "selected": result["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
