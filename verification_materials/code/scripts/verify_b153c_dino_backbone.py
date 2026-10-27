#!/usr/bin/env python3
"""Re-derive every learned DINOv3 backbone tensor; reuse the original CPU pack comparator.

Unlike the older sampled check, this covers all 32 blocks. No checkpoint is written.
The fixed RoPE periods buffer is not a learned soup tensor and is reported separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pin(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("CPU threads must be in 1..4")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    root = args.workspace.resolve()
    base = root / "artifacts/runs"
    helper = base / "20261022T0400+0900_deliverables_prep/verify_weight_chain_v2.py"
    spec = importlib.util.spec_from_file_location("original_weight_verifier", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.torch.set_num_threads(args.threads)
    started = time.monotonic()
    package = base / "20261021T2250+0900_takeover_knob_search/build_b153c_v1/package/assets/model_1.weights.pt"
    sources = [(base / f"20261008T231500+0900_knob_sweep/{name}_all_trainval_seed20261007_ema0.999/model.safetensors", weight)
               for name, weight in (("d3hpq", .2), ("d3hpqr", .3), ("d3hpqrq", .3), ("d3hpqs", .2))]
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p, _ in sources}
    package_stat = (package.stat().st_size, package.stat().st_mtime_ns)
    pins = [{**pin(path), "weight": weight} for path, weight in sources]
    stored = mod.load_pt(package)
    keys = sorted(k for k in stored if not k.startswith("head.") and k != "backbone.rope.periods")
    blocks = sorted({int(k.split(".")[2]) for k in keys if k.startswith("backbone.blocks.")})
    if blocks != list(range(32)):
        raise ValueError(f"expected all 32 blocks, observed {blocks}")
    result = mod.soup_check(stored, sources, keys, mode="build_weighted_soup")
    if any((p.stat().st_size, p.stat().st_mtime_ns) != source_stats[str(p)] for p, _ in sources):
        raise ValueError("source changed while being verified")
    if (package.stat().st_size, package.stat().st_mtime_ns) != package_stat:
        raise ValueError("packaged weights changed while being verified")
    summary = result["summary"]
    passed = summary["not_exact"] == 0 and not result["skipped_not_in_safetensors"]
    report = {"status": "PASS_ALL_LEARNED_BACKBONE_TENSORS" if passed else "FAIL",
              "candidate": "B153C_B152C_ATTRW6_L1W_RERANK", "member": 1,
              "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "device": "cpu", "threads": args.threads, "blocks": blocks,
              "keys_requested": len(keys), "excluded": ["head.* (existing head proof)", "backbone.rope.periods (non-learned buffer)"],
              "sources": pins, "package": pin(package), "helper": pin(helper), "verifier": pin(Path(__file__)),
              "result": result, "wall_seconds": time.monotonic() - started,
              "scope": "Exact CPU reconstruction and INT8 packing of all learned backbone tensors; no training, accuracy or official runtime claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "summary": summary, "wall_seconds": report["wall_seconds"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
