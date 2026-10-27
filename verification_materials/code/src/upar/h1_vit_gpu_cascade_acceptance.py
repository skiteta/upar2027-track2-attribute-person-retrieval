"""Stdlib-only release gate for cascaded GPU-runner packages (kind h1_vit_gpu_cascade_acceptance_v1).

A cascade (plan A -> lighter plans) cannot keep plan A on the shared local GPU, so evidence is split:
1. a default-path GPU replay of the exact artifact: completes on CUDA, the realised plan matches its cached reference;
2. a forced first-plan GPU replay of a verification copy whose bytes equal the artifact except the config time budget:
   plan A runs on CUDA end to end and matches its cached reference;
3. the artifact's config fails closed (require_cuda, >= 2 members, no R18 fallback) and fits the 300 s limit.
It grants no submission authority; the A10's speed (which plan runs officially) is not provable locally.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

KIND = "h1_vit_gpu_cascade_acceptance_v1"
MAX_ABS_MADM_DIFF = 0.002
MAX_DEFAULT_WALL_S = 290.0  # matches h1_replay.MAX_GPU_REPLAY_SECONDS (run.py deadline + process start, under the 300 s limit)
MAX_TIME_BUDGET_S = 292.0  # 2026-10-14: overhead outside predict measured 8.1 s (B110G), 7.3 s (B112G), 7.9 s (B114G) -> deadline 292-6 = 286 s keeps the total near 294 s < 300 s. Was 286: 2026-10-12: B110G ran ingestion 261.3 s for a 253.2 s predict (overhead 7.8 s, ranking 0.3 s); deadline 280 + ~8 s stays under 300 s, and a run past 300 s fails like the internal fail-closed, so a later internal deadline is never worse
MAX_TIME_SCALE = 4.0
FULL_ROWS = 33407
FULL_QUERIES = 3462


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ref(item: dict) -> Path:
    path = Path(item["path"])
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise ValueError(f"evidence changed or missing: {path}")
    return path


def _full(report: dict) -> None:
    if report.get("returncode") != 0 or report.get("device") != "cuda":
        raise ValueError("replay did not complete on CUDA")
    if report.get("rows") != FULL_ROWS or report.get("queries") != FULL_QUERIES:
        raise ValueError("complete public-validation replay required")
    if float(report.get("abs_mADM_diff", 1.0)) > MAX_ABS_MADM_DIFF:
        raise ValueError("replayed ranking differs from the cached prediction")
    if report.get("tier") != report.get("expected_tier"):
        raise ValueError("replay tier differs from the plan it ran")


def verify_release(artifact_path, acceptance_path, **_ignored) -> dict:
    artifact = Path(artifact_path)
    acc = json.loads(Path(acceptance_path).read_text())
    if acc.get("kind") != KIND:
        raise ValueError("not a cascade GPU acceptance record")
    actual = _sha256(artifact)
    if acc["artifact"]["sha256"] != actual:
        raise ValueError("acceptance is bound to different artifact bytes")
    default = json.loads(_ref(acc["default_replay_report"]).read_text())
    if default.get("artifact", {}).get("sha256") != actual or default.get("mode") != "gpu" or default.get("forced_plan"):
        raise ValueError("default replay was not a GPU run of these exact bytes")
    _full(default)
    scale = float(default.get("time_scale", 1.0))  # deadline stretch used on a shared local GPU (run.py hook, unset on the platform)
    if not 1.0 <= scale <= MAX_TIME_SCALE:
        raise ValueError("default replay time scale out of range")
    if not 0 < float(default.get("wall_seconds", 0)) <= MAX_DEFAULT_WALL_S * scale:
        raise ValueError("default GPU replay exceeded the local time bound")
    copy = _ref(acc["verification_copy"])
    forced = json.loads(_ref(acc["forced_first_plan_replay_report"]).read_text())
    if forced.get("artifact", {}).get("sha256") != acc["verification_copy"]["sha256"] or forced.get("mode") != "gpu":
        raise ValueError("forced replay was not a GPU run of the verification copy")
    _full(forced)
    with zipfile.ZipFile(artifact) as za, zipfile.ZipFile(copy) as zc:
        names = {n for n in za.namelist() if not n.endswith("/")}  # directory entries carry no content
        if names != {n for n in zc.namelist() if not n.endswith("/")}:
            raise ValueError("verification copy has different members")
        for n in names - {"assets/config.json"}:
            if za.read(n) != zc.read(n):
                raise ValueError(f"verification copy differs in {n}")
        ca, cc = json.loads(za.read("assets/config.json")), json.loads(zc.read("assets/config.json"))
        if {k: v for k, v in ca.items() if k != "time_budget_s"} != {k: v for k, v in cc.items() if k != "time_budget_s"}:
            raise ValueError("verification copy config differs beyond time_budget_s")
        first = (ca.get("plans") or [{}])[0].get("name")
        if forced.get("forced_plan") != first:
            raise ValueError("forced replay did not run the first plan")
        if not (ca.get("require_cuda") is True and int(ca.get("min_active_members", 1)) >= 2 and ca.get("fail_closed_on_fallback") is True):
            raise ValueError("GPU-runner package must fail closed")
        if float(ca.get("time_budget_s", 1e9)) > MAX_TIME_BUDGET_S:
            raise ValueError("package time budget does not fit the 300 s limit")
        if "gpu_half" not in za.read("run.py").decode("utf-8"):
            raise ValueError("run.py is not the GPU-runner template")
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe archive member path")
    return {"kind": KIND, "artifact_sha256": actual, "default_realized_plan": default.get("realized_plan"),
            "default_abs_mADM_diff": default["abs_mADM_diff"], "forced_first_plan": first,
            "forced_abs_mADM_diff": forced["abs_mADM_diff"], "forced_replay_mADM": forced["replay_metrics"]["ADM"],
            "default_time_scale": scale}
