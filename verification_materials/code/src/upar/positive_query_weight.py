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

from .blend_predictions import (
    SOURCES,
    atomic_json,
    blend_probabilities,
    load_prediction,
    sha256_file,
)
from .data import RetrievalSplit, load_retrieval_split
from .metrics_strict import EvaluationResult, evaluate_scores

DEFAULT_POSITIVE_WEIGHTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def select_positive_weight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("positive-weight search produced no rows")
    # Exact ties retain the official uniform L1 distance as closely as possible.
    return max(
        rows,
        key=lambda row: (
            float(row["macro_mADM"]),
            -abs(math.log2(float(row["positive_weight"]))),
            -float(row["positive_weight"]),
        ),
    )


def positive_weighted_l1_distances(
    probabilities: np.ndarray,
    queries: np.ndarray,
    *,
    positive_weight: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    queries = np.asarray(queries)
    if probabilities.ndim != 2 or queries.ndim != 2:
        raise ValueError("probabilities and queries must be matrices")
    if probabilities.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("positive-weighted L1 inputs must be float32")
    if probabilities.shape[1] != queries.shape[1]:
        raise ValueError("probability and query widths differ")
    if not np.isfinite(probabilities).all() or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if not np.isfinite(queries).all() or not np.all((queries == 0) | (queries == 1)):
        raise ValueError("queries must be finite binary values")
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive")

    base = probabilities.sum(axis=1)[None, :]
    if positive_weight == 1.0:
        # Preserve the organizer sample's exact float32 arithmetic for the control.
        transformed = 1.0 - 2.0 * probabilities
    else:
        weight = np.float32(positive_weight)
        transformed = weight - np.float32(weight + np.float32(1.0)) * probabilities
    return base + queries @ transformed.T


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


def evaluate_positive_weight(
    probabilities: np.ndarray,
    *,
    split: RetrievalSplit,
    query_indices: np.ndarray,
    positive_weight: float,
    chunk_size: int,
) -> EvaluationResult:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    per_query: list[dict[str, float | int]] = []
    for start in range(0, len(query_indices), chunk_size):
        indices = query_indices[start : start + chunk_size]
        queries = split.queries[indices].astype(np.float32, copy=False)
        distances = positive_weighted_l1_distances(
            probabilities,
            queries,
            positive_weight=positive_weight,
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


def _verified_prediction(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    prediction = load_prediction(path)
    if prediction["sha256"] != record.get("sha256"):
        raise ValueError(f"prediction hash differs from B008 selection record: {path}")
    return prediction


def _b008_lodo_predictions(
    search: dict[str, Any],
    split: RetrievalSplit,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    inputs = search.get("inputs", {})
    alpha = float(search["b006_alpha"])
    beta = float(search["selected"]["beta"])
    outputs: dict[str, np.ndarray] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        components = {
            name: _verified_prediction(inputs[name][source])
            for name in ("b002_lodo", "b005_lodo", "b007_lodo")
        }
        reference = components["b002_lodo"]
        for name, prediction in components.items():
            if not np.array_equal(reference["image_paths"], prediction["image_paths"]):
                raise ValueError(f"{source} {name} image order differs")
            if not np.array_equal(reference["attribute_names"], prediction["attribute_names"]):
                raise ValueError(f"{source} {name} attribute order differs")
        if tuple(reference["image_paths"].tolist()) != split.image_paths:
            raise ValueError(f"{source} prediction paths differ from the evaluation split")
        if tuple(reference["attribute_names"].tolist()) != split.attribute_names:
            raise ValueError(f"{source} prediction attributes differ from the evaluation split")
        b006 = blend_probabilities(
            components["b002_lodo"]["probabilities"],
            components["b005_lodo"]["probabilities"],
            alpha,
        )
        outputs[source] = blend_probabilities(
            b006,
            components["b007_lodo"]["probabilities"],
            beta,
        )
        evidence[source] = {
            name: {"path": str(prediction["path"]), "sha256": prediction["sha256"]}
            for name, prediction in components.items()
        }
    return outputs, evidence


def _write_full_distances(
    path: Path,
    *,
    probabilities: np.ndarray,
    queries: np.ndarray,
    positive_weight: float,
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
            output[start:stop] = positive_weighted_l1_distances(
                probabilities,
                queries[start:stop].astype(np.float32, copy=False),
                positive_weight=positive_weight,
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
    positive_weights: tuple[float, ...],
    output_search: Path,
    output_distances: Path,
    chunk_size: int,
) -> dict[str, Any]:
    if not positive_weights or tuple(sorted(set(positive_weights))) != positive_weights:
        raise ValueError("positive_weights must be unique and sorted")
    if 1.0 not in positive_weights or any(not math.isfinite(x) or x <= 0 for x in positive_weights):
        raise ValueError("positive_weights must be positive and include the uniform-L1 control 1.0")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    started = perf_counter()
    split = load_retrieval_split(split_dir)
    search = json.loads(b008_search_path.read_text(encoding="utf-8"))
    if search.get("status") != "SELECTED" or "selected" not in search:
        raise ValueError("B008 blend search is not a selected result")
    lodo_predictions, lodo_evidence = _b008_lodo_predictions(search, split)

    rows: list[dict[str, Any]] = []
    for positive_weight in positive_weights:
        source_results: dict[str, dict[str, float | int]] = {}
        for source in SOURCES:
            mask = split.sources == source
            source_split = split.subset_gallery(mask)
            query_indices = np.unique(source_split.gallery_query_ids)
            result = evaluate_positive_weight(
                lodo_predictions[source][mask],
                split=source_split,
                query_indices=query_indices,
                positive_weight=positive_weight,
                chunk_size=chunk_size,
            )
            source_results[source] = result.summary
        macro = float(np.mean([float(source_results[source]["mADM"]) for source in SOURCES]))
        rows.append(
            {
                "positive_weight": positive_weight,
                "macro_mADM": macro,
                "sources": source_results,
            }
        )

    selected = select_positive_weight(rows)
    b008_all = load_prediction(b008_all_path)
    recorded_all = search.get("output_prediction", {})
    if b008_all["sha256"] != recorded_all.get("sha256"):
        raise ValueError("B008 all-domain prediction differs from its selection record")
    if tuple(b008_all["image_paths"].tolist()) != split.image_paths:
        raise ValueError("B008 all-domain image order differs from the evaluation split")
    if tuple(b008_all["attribute_names"].tolist()) != split.attribute_names:
        raise ValueError("B008 all-domain attributes differ from the evaluation split")
    _write_full_distances(
        output_distances,
        probabilities=b008_all["probabilities"],
        queries=split.queries,
        positive_weight=float(selected["positive_weight"]),
        chunk_size=chunk_size,
    )
    elapsed = perf_counter() - started
    positive_counts = split.queries.sum(axis=1)
    report = {
        "schema": 1,
        "status": "SELECTED",
        "candidate": "B009_B008_POSITIVE_QUERY_WEIGHT",
        "distance": "sum_negative(p) + positive_weight * sum_positive(1-p)",
        "selection_surface": "three held-source B008 LODO mADM with equal source weight",
        "tie_break": "closest positive_weight to uniform L1 control 1.0; lower weight breaks symmetric ties",
        "positive_weight_grid": list(positive_weights),
        "rows": rows,
        "selected": selected,
        "query_positive_count": {
            "minimum": int(positive_counts.min()),
            "maximum": int(positive_counts.max()),
            "mean": float(positive_counts.mean()),
            "median": float(np.median(positive_counts)),
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
            "All three LODO sources select one scalar, so they remain development evidence rather than confirmation.",
            "The fixed grid does not establish optimality between or beyond the tested weights.",
            "The official mADM relevance normalization remains unchanged; only submitted ranking distance changes.",
            "Public and Private query sparsity may differ from the three local sources.",
        ],
    }
    atomic_json(output_search, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a positive-query L1 weight for fixed B008 probabilities")
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
        positive_weights=DEFAULT_POSITIVE_WEIGHTS,
        output_search=args.output_search,
        output_distances=args.output_distances,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({"status": result["status"], "selected": result["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
