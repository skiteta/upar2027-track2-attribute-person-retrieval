from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from .data import load_retrieval_split
from .submission import load_submission_module, make_fixture, validate_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a Track 2 code submission on a complete public split")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    split = load_retrieval_split(args.split_dir)
    missing = [path for path in split.image_paths if not (args.image_root / path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} gallery images are missing; first examples: {missing[:3]}")
    fixture = make_fixture(
        split,
        query_count=split.query_count,
        gallery_size=split.gallery_size,
        image_root=args.image_root,
    )
    module = load_submission_module(args.submission_dir / "run.py")
    load_start = time.perf_counter()
    load_model = getattr(module, "load_model", None)
    if callable(load_model):
        load_model()
    load_seconds = time.perf_counter() - load_start
    infer_start = time.perf_counter()
    result = module.rank_gallery(fixture)
    infer_seconds = time.perf_counter() - infer_start
    kind, values = validate_output(result, query_count=split.query_count, gallery_size=split.gallery_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, values, allow_pickle=False)
    report = {
        "status": "PASS",
        "submission_dir": str(args.submission_dir),
        "split_dir": str(args.split_dir),
        "image_root": str(args.image_root),
        "all_gallery_images_present": True,
        "kind": kind,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "load_seconds": load_seconds,
        "infer_seconds": infer_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "scope": "Complete public-validation execution through run.py; not the private Codabench container or hidden domain.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
