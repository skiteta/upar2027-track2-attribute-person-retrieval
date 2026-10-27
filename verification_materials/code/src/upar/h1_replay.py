"""Exact-ZIP local replay for H1 packages and the matching runtime acceptance record.

Extract the exact ZIP, run its run.py through upar.execute_submission on the complete public
validation split (33,407 images, more than the hidden 29,248) pinned to six CPU cores with the
submission torch 2.4.1 environment, score the saved distances with the strict-v2 metric, and
compare against the training-environment cached prediction of the same weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from .h1_finetune import ROOT, SPLITS, _probabilities, evaluate_probabilities

KIND = "h1_vit_fp32_runtime_acceptance_v1"
MAX_REPLAY_SECONDS = 3000.0  # 6 cores, 33,407 images; official limit is 3,600 s for 29,248 images
# GPU runners (from 2026-10-08): A10, 20 cores, 300 s for 29,248 images. A local GPU replay must use CUDA, keep the
# first plan, and finish 33,407 images within MAX_GPU_REPLAY_SECONDS (the local GPU is faster than an A10).
GPU_KIND = "h1_vit_gpu_runtime_acceptance_v1"
MAX_GPU_REPLAY_SECONDS = 290.0  # the run.py deadline (time budget <= 280 minus rank reserve) plus process start; under 300 s. The local GPU is shared (~97% busy), so A10 time is estimated separately
MAX_ABS_MADM_DIFF = 0.002


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replay(artifact: Path, manifest: Path, out_dir: Path, cores: str, force_plan: str | None = None, gpu_python: Path | None = None,
           time_scale: float = 1.0) -> dict:
    from .data import load_retrieval_split
    from .metrics_strict import evaluate_scores

    out_dir = out_dir.resolve()
    artifact = artifact.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    man = json.loads(manifest.read_text())
    artifact_sha = sha256(artifact)
    if artifact_sha != man["artifact"]["sha256"]:
        raise ValueError("artifact differs from its package manifest")
    extracted = out_dir / "extracted"
    with zipfile.ZipFile(artifact) as zf:
        names = zf.namelist()
        if "run.py" not in names or "metadata.yaml" not in names:
            raise ValueError("run.py and metadata.yaml must be at the archive root")
        zf.extractall(extracted)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "MKL_", "OPENBLAS_"))}
    env["PYTHONPATH"] = str(ROOT / "src")
    if gpu_python is None:
        env["CUDA_VISIBLE_DEVICES"] = ""  # replay the conservative CPU path
    if force_plan:
        env["UPAR_FORCE_PLAN"] = force_plan  # exercise a later plan of the same ZIP
    if time_scale != 1.0:
        env["UPAR_TIME_SCALE"] = str(time_scale)  # stretch run.py's deadline on a shared (slower) local GPU; never set on the platform
    python = str(gpu_python) if gpu_python else str(ROOT / ".venv/bin/python")
    cmd = ["taskset", "-c", cores, python, "-m", "upar.execute_submission",
           "--submission-dir", str(extracted), "--split-dir", str(SPLITS["val"]), "--image-root", str(ROOT / "starter_kit/data"),
           "--output", str(out_dir / "distances.npy"), "--report", str(out_dir / "execute_report.json")]
    started = time.time()
    with (out_dir / "replay_stdout.txt").open("w") as stdout, (out_dir / "replay_stderr.txt").open("w") as stderr:
        returncode = subprocess.run(cmd, cwd=str(extracted), env=env, stdout=stdout, stderr=stderr).returncode
    wall = time.time() - started
    report = {"schema": 1, "artifact": {"path": str(artifact), "sha256": artifact_sha}, "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
              "cores": cores, "core_count": len(_expand(cores)), "returncode": returncode, "wall_seconds": round(wall, 2), "time_scale": time_scale}
    if returncode == 0:
        execute = json.loads((out_dir / "execute_report.json").read_text())
        split = load_retrieval_split(SPLITS["val"])
        distances = np.load(out_dir / "distances.npy", mmap_mode="r")
        per = []
        for start in range(0, split.query_count, 128):
            idx = np.arange(start, min(start + 128, split.query_count))
            per.extend(evaluate_scores(np.asarray(distances[idx]), direction="distance", queries=split.queries,
                                       gallery_attributes=split.gallery_attributes, gallery_query_ids=split.gallery_query_ids,
                                       query_indices=idx).per_query)
        replay_metrics = {k: float(np.mean([q[k] for q in per])) for k in ("ADM", "AP", "Rank-1", "Rank-5", "Rank-10", "INP")}
        config = json.loads((extracted / "assets/config.json").read_text())
        plans = config.get("plans") or [{"name": "all", "members": list(range(len(config["models"])))}]
        plan = next(p for p in plans if p["name"] == force_plan) if force_plan else plans[0]
        stdout_lines = [json.loads(line) for line in (out_dir / "replay_stdout.txt").read_text().splitlines() if line.startswith('{"upar_h1"')]
        started = [r["plan"] for r in stdout_lines if r["event"] == "plan_start"]
        first_plan = plan["name"]
        if started and not force_plan:  # cascades: the reference follows the plan that actually ran
            plan = next(p for p in plans if p["name"] == started[-1])
        predicted_events = [r for r in stdout_lines if r["event"] == "predicted"]
        if predicted_events:  # tier-drop: the reference follows the members that actually produced the output
            tier_names = predicted_events[-1]["tier"].split("+")
            names_in_plan = [config["models"][i]["name"] for i in plan["members"]]
            if tier_names != names_in_plan and tier_names == names_in_plan[: len(tier_names)]:
                plan = dict(plan, members=plan["members"][: len(tier_names)], name=plan["name"] + f"[tier{len(tier_names)}]")
        stdout_lines = [json.loads(line) for line in (out_dir / "replay_stdout.txt").read_text().splitlines() if line.startswith('{"upar_h1"')]
        # optional trailing members may be dropped on slower hosts; the reference follows the members actually kept
        dropped = sum(1 for r in stdout_lines if r["event"] == "optional_drop")
        if dropped > int(plan.get("optional", 0)):
            raise RuntimeError("more members dropped than the plan marks optional")
        members = [config["models"][i] for i in plan["members"][: len(plan["members"]) - dropped]]
        member_probs = np.stack([_probabilities(np.load(Path(m["training_run"]) / "val_logits.npz"), m["flip_tta"]) for m in members])
        member_w = np.asarray([m.get("member_weights", [1.0] * 40) for m in members], dtype=np.float32)
        reference = (np.einsum("mga,ma->ga", member_probs, member_w) / member_w.sum(axis=0)).astype(np.float32)
        from .postproc import apply_post
        reference = apply_post(reference, config.get("post"))  # the same ensemble post-processing as run.py
        cached = evaluate_probabilities(reference, "all", split, entropy_delta=(config.get("post") or {}).get("entropy_delta"))
        predicted = [r for r in stdout_lines if r["event"] == "predicted"]
        expected_tier = "+".join(m["name"] for m in members)
        report.update(realized_plan=plan["name"], first_plan=first_plan,
                      rows=split.gallery_size, queries=split.query_count, execute=execute, replay_metrics=replay_metrics,
                      cached_reference_metrics=cached, abs_mADM_diff=abs(replay_metrics["ADM"] - cached["ADM"]),
                      tier=predicted[-1]["tier"] if predicted else None, expected_tier=expected_tier,
                      plan=plan["name"], forced_plan=force_plan, optional_dropped=dropped,
                      runtime_log=[r for r in stdout_lines if r["event"] in ("hardware", "loaded", "plan_start", "plan_switch", "optional_drop", "tier_drop", "fallback", "predicted", "ranked")])
        loaded = [r for r in stdout_lines if r["event"] == "loaded"]
        report["device"] = loaded[-1].get("device") if loaded else None
        report["mode"] = "gpu" if gpu_python else "cpu"
        if gpu_python:
            report["pass"] = bool(report["device"] == "cuda" and wall <= MAX_GPU_REPLAY_SECONDS * time_scale and report["abs_mADM_diff"] <= MAX_ABS_MADM_DIFF
                                  and report["tier"] == expected_tier)
        else:
            report["pass"] = bool(wall <= MAX_REPLAY_SECONDS and report["abs_mADM_diff"] <= MAX_ABS_MADM_DIFF
                                  and report["core_count"] <= 6 and report["tier"] == expected_tier)
    else:
        report["pass"] = False
    (out_dir / "replay_report.json").write_text(json.dumps(report, indent=1))
    acceptance = {"kind": GPU_KIND if gpu_python else KIND, "artifact": report["artifact"], "replay_report": {"path": str(out_dir / "replay_report.json"),
                  "sha256": sha256(out_dir / "replay_report.json")}, "limits": {"max_replay_seconds": MAX_GPU_REPLAY_SECONDS if gpu_python else MAX_REPLAY_SECONDS,
                  "max_abs_mADM_diff": MAX_ABS_MADM_DIFF, "max_cores": 6}}
    if report["pass"] and not force_plan:  # acceptance covers the default (first-plan) path only
        (out_dir / "runtime_acceptance.json").write_text(json.dumps(acceptance, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("execute", "runtime_log")}, indent=1))
    return report


def _expand(cores: str) -> list[int]:
    out: list[int] = []
    for part in cores.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cores", default="12-17")
    parser.add_argument("--force-plan", help="replay a named later plan (verification only, no acceptance file)")
    parser.add_argument("--gpu-python", type=Path, help="run with this interpreter and CUDA visible (GPU-runner emulation)")
    parser.add_argument("--time-scale", type=float, default=1.0, help="stretch run.py's deadline by this factor (shared local GPU slower than the runner; <= 4)")
    args = parser.parse_args()
    if not 1.0 <= args.time_scale <= 4.0:
        parser.error("--time-scale must be in [1, 4]")
    replay(args.artifact, args.manifest, args.out_dir, args.cores, args.force_plan, args.gpu_python, args.time_scale)


if __name__ == "__main__":
    main()
