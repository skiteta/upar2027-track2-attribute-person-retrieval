from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .data import RetrievalSplit, load_retrieval_split


def load_submission_module(run_py: str | Path) -> ModuleType:
    run_py = Path(run_py).resolve()
    spec = importlib.util.spec_from_file_location("upar_candidate_submission", run_py)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import submission: {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "rank_gallery", None)):
        raise ValueError("submission must define rank_gallery(sample)")
    return module


def make_fixture(split: RetrievalSplit, *, query_count: int, gallery_size: int, image_root: Path) -> dict[str, Any]:
    if query_count <= 0 or gallery_size <= 0:
        raise ValueError("fixture sizes must be positive")
    query_count = min(query_count, split.query_count)
    gallery_size = min(gallery_size, split.gallery_size)
    return {
        "attribute_names": list(split.attribute_names),
        "queries": split.queries[:query_count].astype(int).tolist(),
        "gallery": [
            {
                "index": index,
                "image": split.image_paths[index],
                "image_path": str((image_root / split.image_paths[index]).resolve()),
            }
            for index in range(gallery_size)
        ],
    }


def validate_output(output: Any, *, query_count: int, gallery_size: int) -> tuple[str, np.ndarray]:
    if not isinstance(output, dict) or len(output) != 1:
        raise ValueError("rank_gallery must return a dict with exactly one key")
    kind = next(iter(output))
    if kind not in {"distances", "similarities", "ranking"}:
        raise ValueError("output key must be distances, similarities, or ranking")
    values = np.asarray(output[kind])
    if values.shape != (query_count, gallery_size):
        raise ValueError(f"output has shape {values.shape}, expected {(query_count, gallery_size)}")
    if kind == "ranking":
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("ranking output must contain integers")
        target = np.arange(gallery_size)
        if any(not np.array_equal(np.sort(row), target) for row in values):
            raise ValueError("each ranking row must be a permutation of gallery indices")
    elif not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{kind} must contain finite numeric values")
    return kind, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a Codabench UPAR code submission on a local fixture")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("starter_kit/data"))
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--gallery", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_py = args.submission_dir / "run.py"
    metadata = args.submission_dir / "metadata.yaml"
    if not run_py.is_file() or not metadata.is_file():
        raise ValueError("submission root must contain run.py and metadata.yaml")
    split = load_retrieval_split(args.split_dir)
    fixture = make_fixture(
        split,
        query_count=args.queries,
        gallery_size=args.gallery,
        image_root=args.image_root,
    )
    module = load_submission_module(run_py)
    load_model = getattr(module, "load_model", None)
    if callable(load_model):
        load_model()
    output = module.rank_gallery(fixture)
    kind, values = validate_output(
        output,
        query_count=len(fixture["queries"]),
        gallery_size=len(fixture["gallery"]),
    )
    report = {
        "status": "PASS",
        "submission_dir": str(args.submission_dir),
        "kind": kind,
        "shape": list(values.shape),
        "finite": True,
        "scope": "local API/schema fixture only; not hidden-data execution, runtime certification, or score parity",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
