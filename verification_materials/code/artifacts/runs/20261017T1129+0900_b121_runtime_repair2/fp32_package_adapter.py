"""Prepare exactly one selected FP32 numerical revision; root builds explicitly.

Observed distances are NOT bitwise equivalent to LOCKED3. The fixed four-case
comparison and root selection are required. Construction is not runtime
acceptance, submission authority, universal graph parity or an A10 guarantee.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

import gpu_package_adapter as frozen

HERE, ROOT = frozen.HERE, frozen.ROOT
NAME = "B121G_PE4_FEATURE_AGREEMENT_CUDA32"
POLICY = "cuda_fp32_unit_dot_tf32_off_numeric_revision_v1"
REPORT_SHA = "6f79d47947a85b6ab585005360addfc60754e8c1d3163d2bf48aa22a4c80f3dc"
SELECTION_SHA = "c129d9685f1e207ef4946cc4af807d3dbef70603459daf2b5ca1080e6db31576"
REVIEW_SHA = "461851d5705d3ef768b0d2b52e7cd6fc7578bc8d1806a5195cfc14cecea7c6f4"
SOURCE_SHA = {
    "gpu_package_adapter.py": "5f77fc36ef22031743c654c96b22bde5cfffd79725e185d274de77ea4ad02892",
    "b121_fp32_cosine_graph.py": "4f75b54fb3491201581b522a7b54318a7d3a17d8747016ab0675556892067292",
    "b121_gpu_graph.py": "53c55c8a9d54c56fe91f5ee6f851dbe7b6b701619c2f8d48edf3c1ddb1f3d0ae",
    "b121_observe.py": "d7d286b3af3127989cb8ebdb8d5df5b2bca62180b18f4fd48e812b96172bbc14",
}
OBSERVED = {"production_full": (8, 2, 49), "PA100k": (6, 2, 14),
            "PETA": (0, 0, 0), "Market1501": (4, 2, 16)}
file_sha, pin = frozen.file_sha, frozen.pin


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("unique frozen patch boundary required: " + old)
    return source.replace(old, new, 1)


def validate_sources():
    for name, expected in SOURCE_SHA.items():
        if file_sha(HERE / name) != expected:
            raise ValueError("frozen source changed: " + name)
    return [pin(HERE / name) for name in SOURCE_SHA]


def patch_runner(source):
    validate_sources()
    result = replace_once(frozen.patch_runner(source), "CUDA_FP64_GEMM_only", "CUDA_FP32_unit_dot_TF32_off")
    ast.parse(result)
    return result


def patch_runtime(source):
    validate_sources()
    result = frozen.patch_runtime(source)
    for old, new in (
        ("from b121_gpu_graph import build_feature_graph", "from b121_fp32_cosine_graph import build_feature_graph"),
        ("mandatory CUDA FP64 graph unavailable", "mandatory CUDA FP32 unit-dot graph unavailable"),
        ("device='cuda', dtype='float64'", "device='cuda', dtype='float32', normalization='float64', tf32=False"),
        ("CUDA_FP64_GEMM_CPU_exact_topk", "CUDA_FP32_unit_dot_TF32_off_CPU_deterministic_topk"),
    ):
        result = replace_once(result, old, new)
    ast.parse(result)
    return result


def check_report_fields(report, exit_record, report_sha):
    if (type(exit_record.get("actual_exit_code")) is not int or exit_record["actual_exit_code"] != 0
            or not exit_record.get("exec_session") or exit_record.get("report_sha256") != report_sha):
        raise ValueError("actual successful child exit bound to exact benchmark report required")
    if report.get("status") != "NUMERIC_DIFFERENCE_REVIEW_REQUIRED" or set(report.get("cases", {})) != set(OBSERVED):
        raise ValueError("fixed all4 numeric-difference report required, not an equivalence PASS")
    for name, row in report["cases"].items():
        if (tuple(row.get(k) for k in ("graph_index_differences", "gated_index_differences", "distance_unequal")) != OBSERVED[name]
                or not row.get("normalization_exact")
                or row.get("numeric_recipe") != "FP64_normalization_FP32_unit_dot_TF32_off"
                or any(row.get(k) != 0 for k in ("degree_differences", "kept_degree_differences", "queries_with_different_rank", "production_cpu_reference_unequal"))
                or row.get("distance_max_abs") != (0. if name == "PETA" else 1.9073486328125e-6)):
            raise ValueError("observed fixed comparison changed; review a new revision rather than loosen tolerances")
        if name != "production_full":
            metric = row.get("metric") or {}
            if metric.get("unequal_per_query_ADM") != 0 or metric.get("old_mADM") != metric.get("new_mADM"):
                raise ValueError("all3 per-query ADM parity required")


def validate_bench(report_path, exit_path):
    inputs = validate_sources()
    report_path, exit_path = Path(report_path), Path(exit_path)
    if file_sha(report_path) != REPORT_SHA:
        raise ValueError("only exact selected v3 report accepted")
    report, receipt = (json.loads(p.read_text()) for p in (report_path, exit_path))
    check_report_fields(report, receipt, REPORT_SHA)
    selection_path, review_path = HERE / "fp32_selection_v1.json", HERE / "fp32_graph_order_review_v1.json"
    if file_sha(selection_path) != SELECTION_SHA or file_sha(review_path) != REVIEW_SHA:
        raise ValueError("root numeric-revision selection or original graph review changed")
    selection, review = (json.loads(p.read_text()) for p in (selection_path, review_path))
    if (selection.get("decision") != "PREPARE_ONE_NEW_FP32_NUMERIC_REVISION_AND_TEST_FULL_PACKAGE"
            or selection.get("next_candidate") != NAME):
        raise ValueError("explicit one-candidate selection required")
    for ref in review["pins"]:
        path = ROOT / ref["path"]
        if file_sha(path) != ref["sha256"] or path.stat().st_size != ref["size"]:
            raise ValueError("observed graph/report bytes changed")
        inputs.append(pin(path))
    for name in OBSERVED:
        with np.load(HERE / f"actual_gpu_comparison_v2/graph_{name}.npz", allow_pickle=False) as old, np.load(HERE / f"actual_gpu_comparison_v3/graph_{name}.npz", allow_pickle=False) as new:
            for field in ("neighbours", "gated"):
                if not np.array_equal(np.sort(old[field], axis=1), np.sort(new[field], axis=1)):
                    raise ValueError("unordered observed edge membership differs")
            if any(not np.array_equal(old[f], new[f]) for f in ("counts", "kept")):
                raise ValueError("observed degree differs")
    return inputs + [pin(report_path), pin(exit_path), pin(selection_path), pin(review_path)]


def patch_config(cfg):
    cfg = dict(cfg)
    if (cfg.get("candidate") != "B121G_PE4_FEATURE_AGREEMENT_LOCKED3" or cfg.get("time_budget_s") != 292
            or cfg.get("rank_reserve_s") != 40 or cfg.get("plans") != [{"name": "A", "members": [0, 1, 2]}]
            or cfg.get("min_active_members") != 3):
        raise ValueError("original mandatory3 recipe changed")
    cfg["candidate"], cfg["graph_runtime_policy"] = NAME, POLICY
    return cfg


def build(output, report_path, exit_path):
    inputs = validate_bench(report_path, exit_path)
    if file_sha(frozen.PARENT) != frozen.PARENT_SHA:
        raise ValueError("original submitted ZIP changed")
    output = Path(output).resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("new runtime_repair2-only output required")
    output.mkdir(exist_ok=False)
    package = output / "package"
    package.mkdir()
    started = time.monotonic()
    with zipfile.ZipFile(frozen.PARENT) as old:
        names = old.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe archive entry")
            dest = package / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == "run.py":
                dest.write_text(patch_runner(old.read(name).decode()))
            elif name == "b121_runtime.py":
                dest.write_text(patch_runtime(old.read(name).decode()))
            elif name == "assets/config.json":
                dest.write_text(json.dumps(patch_config(json.loads(old.read(name))), indent=2)+"\n")
            else:
                with old.open(name) as source, dest.open("xb") as target:
                    shutil.copyfileobj(source, target, 8*2**20)
        for name in ("b121_fp32_cosine_graph.py", "b121_gpu_graph.py", "b121_observe.py"):
            with (package / name).open("xb") as target:
                target.write((HERE / name).read_bytes())
        artifact = output / (NAME+".zip")
        with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
        preserved = []
        with zipfile.ZipFile(artifact) as new:
            for name in names:
                if name in ("run.py", "b121_runtime.py", "assets/config.json"):
                    continue
                with old.open(name) as a, new.open(name) as b:
                    if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                        raise ValueError("preserved asset/helper changed: "+name)
                preserved.append(name)
    for ref in inputs:
        if pin(ref["path"]) != ref:
            raise ValueError("source/evidence changed during package construction")
    report = {"status": "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED", "candidate": NAME, "policy": POLICY,
              "zip": pin(artifact), "inputs": inputs+[pin(frozen.PARENT), pin(Path(__file__))],
              "preserved_members": preserved, "elapsed_seconds": time.monotonic()-started,
              "numeric_equivalence_to_parent": False,
              "observed_distance_differences": {name: row[2] for name, row in OBSERVED.items()},
              "next": "Actual complete new-ZIP capture, successful child receipt, input equality bridge and exact FP32-recipe reference required. A10 speed and unknown-gallery graph equivalence remain unknown.",
              "official_failure_cause": "unknown after predicted246.0; timeout/OOM not proved"}
    with (output / "complete.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-report", type=Path, required=True)
    parser.add_argument("--bench-exit", type=Path, required=True)
    parser.add_argument("--build-after-review", action="store_true")
    args = parser.parse_args()
    if not args.build_after_review:
        parser.error("root review and explicit build flag required")
    print(json.dumps(build(args.output, args.bench_report, args.bench_exit)), flush=True)
