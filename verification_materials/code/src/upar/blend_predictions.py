from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_retrieval_split
from .metrics_strict import evaluate_attribute_probabilities

SOURCES = ("Market1501", "PA100k", "PETA")
DEFAULT_ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"probabilities", "image_paths", "attribute_names"}
        if not required <= set(saved.files):
            raise ValueError(f"prediction is missing required arrays: {path}")
        probabilities = np.asarray(saved["probabilities"])
        image_paths = np.asarray(saved["image_paths"], dtype=str)
        attribute_names = np.asarray(saved["attribute_names"], dtype=str)
    if probabilities.dtype != np.float32 or probabilities.ndim != 2:
        raise ValueError(f"prediction probabilities must be a float32 matrix: {path}")
    if probabilities.shape != (image_paths.size, attribute_names.size):
        raise ValueError(f"prediction arrays are not aligned: {path}")
    if not np.isfinite(probabilities).all() or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError(f"prediction probabilities must be finite and in [0,1]: {path}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "probabilities": probabilities,
        "image_paths": image_paths,
        "attribute_names": attribute_names,
    }


def aligned_pair(baseline_path: Path, candidate_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = load_prediction(baseline_path)
    candidate = load_prediction(candidate_path)
    for key in ("image_paths", "attribute_names"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"baseline and candidate differ in {key}")
    if baseline["probabilities"].shape != candidate["probabilities"].shape:
        raise ValueError("baseline and candidate probability shapes differ")
    return baseline, candidate


def blend_probabilities(baseline: np.ndarray, candidate: np.ndarray, alpha: float) -> np.ndarray:
    if baseline.shape != candidate.shape or baseline.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("blend inputs must be aligned float32 arrays")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if alpha == 0:
        return baseline.copy()
    if alpha == 1:
        return candidate.copy()
    baseline_weight = np.float32(1.0 - alpha)
    candidate_weight = np.float32(alpha)
    return (baseline * baseline_weight + candidate * candidate_weight).astype(np.float32, copy=False)


def select_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("alpha search produced no rows")
    # Exact macro score first; an exact tie stays closer to the current B005
    # baseline and therefore avoids adding B002 without measured benefit.
    return max(rows, key=lambda row: (float(row["macro_mADM"]), float(row["alpha"])))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_prediction(
    path: Path,
    *,
    probabilities: np.ndarray,
    image_paths: np.ndarray,
    attribute_names: np.ndarray,
    alpha: float,
    baseline_sha256: str,
    candidate_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            schema=np.asarray(1, dtype=np.int64),
            probabilities=probabilities,
            image_paths=image_paths,
            attribute_names=attribute_names,
            scoring=np.asarray("prior_corrected_probability_blend"),
            fit_mode=np.asarray("all"),
            holdout_source=np.asarray(""),
            split_role=np.asarray("eval"),
            blend_alpha=np.asarray(alpha, dtype=np.float64),
            baseline_prediction_sha256=np.asarray(baseline_sha256),
            candidate_prediction_sha256=np.asarray(candidate_sha256),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_search(
    *,
    split_dir: Path,
    baseline_all_path: Path,
    candidate_all_path: Path,
    baseline_lodo_paths: dict[str, Path],
    candidate_lodo_paths: dict[str, Path],
    alphas: tuple[float, ...],
    output_search: Path,
    output_prediction: Path,
    chunk_size: int,
) -> dict[str, Any]:
    if not alphas or tuple(sorted(set(alphas))) != alphas:
        raise ValueError("alphas must be unique and sorted")
    if alphas[0] != 0 or alphas[-1] != 1:
        raise ValueError("alpha grid must include the B002 and B005 endpoints")
    if set(baseline_lodo_paths) != set(SOURCES) or set(candidate_lodo_paths) != set(SOURCES):
        raise ValueError("exactly three held-source prediction pairs are required")

    split = load_retrieval_split(split_dir)
    pairs = {
        source: aligned_pair(baseline_lodo_paths[source], candidate_lodo_paths[source])
        for source in SOURCES
    }
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        source_results: dict[str, dict[str, float | int]] = {}
        for source in SOURCES:
            baseline, candidate = pairs[source]
            if tuple(baseline["image_paths"].tolist()) != split.image_paths:
                raise ValueError(f"{source} prediction paths differ from the evaluation split")
            if tuple(baseline["attribute_names"].tolist()) != split.attribute_names:
                raise ValueError(f"{source} attributes differ from the evaluation split")
            mask = split.sources == source
            source_split = split.subset_gallery(mask)
            probabilities = blend_probabilities(
                baseline["probabilities"], candidate["probabilities"], alpha
            )[mask]
            query_indices = np.unique(source_split.gallery_query_ids)
            result = evaluate_attribute_probabilities(
                probabilities,
                queries=source_split.queries,
                gallery_attributes=source_split.gallery_attributes,
                gallery_query_ids=source_split.gallery_query_ids,
                query_indices=query_indices,
                chunk_size=chunk_size,
            )
            source_results[source] = result.summary
        macro = float(np.mean([float(source_results[source]["mADM"]) for source in SOURCES]))
        rows.append({"alpha": alpha, "macro_mADM": macro, "sources": source_results})

    selected = select_alpha(rows)
    baseline_all, candidate_all = aligned_pair(baseline_all_path, candidate_all_path)
    if tuple(baseline_all["image_paths"].tolist()) != split.image_paths:
        raise ValueError("all-domain prediction paths differ from the evaluation split")
    if tuple(baseline_all["attribute_names"].tolist()) != split.attribute_names:
        raise ValueError("all-domain attributes differ from the evaluation split")
    selected_probabilities = blend_probabilities(
        baseline_all["probabilities"],
        candidate_all["probabilities"],
        float(selected["alpha"]),
    )
    save_prediction(
        output_prediction,
        probabilities=selected_probabilities,
        image_paths=baseline_all["image_paths"],
        attribute_names=baseline_all["attribute_names"],
        alpha=float(selected["alpha"]),
        baseline_sha256=str(baseline_all["sha256"]),
        candidate_sha256=str(candidate_all["sha256"]),
    )
    report = {
        "schema": 1,
        "status": "SELECTED",
        "selection_surface": "three held-source LODO mADM with equal source weight",
        "tie_break": "larger alpha, retaining B005 when scores are exactly tied",
        "alpha_grid": list(alphas),
        "rows": rows,
        "selected": selected,
        "inputs": {
            "baseline_all": {"path": str(baseline_all_path), "sha256": baseline_all["sha256"]},
            "candidate_all": {"path": str(candidate_all_path), "sha256": candidate_all["sha256"]},
            "baseline_lodo": {
                source: {"path": str(baseline_lodo_paths[source]), "sha256": pairs[source][0]["sha256"]}
                for source in SOURCES
            },
            "candidate_lodo": {
                source: {"path": str(candidate_lodo_paths[source]), "sha256": pairs[source][1]["sha256"]}
                for source in SOURCES
            },
        },
        "output_prediction": {
            "path": str(output_prediction),
            "sha256": sha256_file(output_prediction),
            "shape": list(selected_probabilities.shape),
            "dtype": str(selected_probabilities.dtype),
        },
        "limitations": [
            "All three LODO sources are used to select alpha, so they are development evidence, not confirmation.",
            "The alpha grid does not establish optimality outside the tested values.",
            "Probability blending requires both models at submission inference time unless later compressed.",
        ],
    }
    atomic_json(output_search, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a B002/B005 probability blend on held-source LODO")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--baseline-all", type=Path, required=True)
    parser.add_argument("--candidate-all", type=Path, required=True)
    for source in SOURCES:
        parser.add_argument(f"--baseline-{source.lower()}", type=Path, required=True)
        parser.add_argument(f"--candidate-{source.lower()}", type=Path, required=True)
    parser.add_argument("--output-search", type=Path, required=True)
    parser.add_argument("--output-prediction", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    baseline_lodo = {source: getattr(args, f"baseline_{source.lower()}") for source in SOURCES}
    candidate_lodo = {source: getattr(args, f"candidate_{source.lower()}") for source in SOURCES}
    result = run_search(
        split_dir=args.split_dir,
        baseline_all_path=args.baseline_all,
        candidate_all_path=args.candidate_all,
        baseline_lodo_paths=baseline_lodo,
        candidate_lodo_paths=candidate_lodo,
        alphas=DEFAULT_ALPHAS,
        output_search=args.output_search,
        output_prediction=args.output_prediction,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({"status": result["status"], "selected": result["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
