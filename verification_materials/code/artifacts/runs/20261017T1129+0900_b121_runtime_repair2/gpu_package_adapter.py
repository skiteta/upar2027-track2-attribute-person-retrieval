"""Strict proposed runtime-only package adapter. Root builds after review only.

The old ZIP/assets are never modified. A passing real saved-feature benchmark
and actual child-exit receipt are required even to construct a new ZIP. Package
construction is NOT runtime acceptance and cannot authorize a submission.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "artifacts/runs/20261017T0850+0900_b121_runtime_repair/package_v1/B121G_PE4_FEATURE_AGREEMENT_LOCKED3.zip"
PARENT_SHA = "e24fff761e7c89b60a32845dfdb0818ba0dffd48a67e293bf80eb9df965c04f6"
RUN_SHA = "627e39bc64b04acaab349838f2304b5e8f820fe06f37ae9d9570f7453e1171ba"
RUNTIME_SHA = "1c7153025c5e89ccafd4ca2af5035b3f7f8497237befc4d063ee414598e17c2d"
GPU_GRAPH_SHA = "53c55c8a9d54c56fe91f5ee6f851dbe7b6b701619c2f8d48edf3c1ddb1f3d0ae"
NAME = "B121G_PE4_FEATURE_AGREEMENT_CUDA64"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha(path), "size": path.stat().st_size}


def replace_function(source, name, replacement):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError("exactly one original target function required")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [replacement]
    patched = "".join(lines)
    # Every original top-level node except the one named function is unchanged.
    before = [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
              if not (isinstance(n, ast.FunctionDef) and n.name == name)]
    after = [ast.dump(n, include_attributes=False) for n in ast.parse(patched).body
             if not (isinstance(n, ast.FunctionDef) and n.name in (name, "_b121_score_distances", "_b121_stage_log"))]
    if before != after:
        raise ValueError("unrelated runner AST changed")
    return patched


def patch_runner(source):
    if digest(source.encode()) != RUN_SHA:
        raise ValueError("only exact locked3 runner bytes accepted")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "rank_gallery")
    lines = source.splitlines(keepends=True)
    budget = [i for i, s in enumerate(node.body) if isinstance(s, ast.If)
              and "runtime_budget_seconds" in ast.unparse(s.test)]
    if len(budget) != 2 or not isinstance(node.body[4], ast.Assign) or ast.unparse(node.body[4].targets[0]) != "names":
        raise ValueError("original score/deadline structure differs")
    def statements(items):
        return "".join("".join(lines[s.lineno-1:s.end_lineno]) for s in items)
    scoring = statements(node.body[4:budget[0]])
    helper = "def _b121_score_distances(probabilities, queries, attribute_names):\n" + scoring + "    return distances\n\n\n"
    helper_node = ast.parse(helper).body[0]
    if [ast.dump(s) for s in helper_node.body[:-1]] != [ast.dump(s) for s in node.body[4:budget[0]]]:
        raise ValueError("score arithmetic changed during extraction")
    replacement = '''def _b121_stage_log(event, **values):
    _log(event, **values, resources=runtime_snapshot())


''' + helper + "def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:\n" + statements(node.body[:4]) + '''    _b121_stage_log("score_enter", queries=len(queries), gallery=len(probabilities))
    with numpy_blas_threads(2) as blas:
        _b121_stage_log("score_blas", numpy_blas=blas)
        distances = _b121_score_distances(probabilities, queries, attribute_names)
    _b121_stage_log("score_done", queries=len(queries), gallery=len(probabilities))
''' + statements([node.body[budget[0]]]) + '''    _b121_stage_log("graph_enter", arithmetic="CUDA_FP64_GEMM_only", no_cpu_fallback=True)
    distances = apply_feature_rerank(distances, _GALLERY_FEATURES, probabilities, _b121_stage_log)
''' + statements(node.body[budget[1]:])
    result = replace_function(source, "rank_gallery", replacement)
    additions = "from b121_blas import numpy_blas_threads\nfrom b121_observe import runtime_snapshot\n"
    anchor = "from PIL import Image\n"
    if result.count(anchor) != 1:
        raise ValueError("unique import anchor required")
    result = result.replace(anchor, anchor+additions)
    ast.parse(result)
    return result


def patch_runtime(source):
    if digest(source.encode()) != RUNTIME_SHA:
        raise ValueError("only exact locked3 runtime bytes accepted")
    replacement = '''def apply_feature_rerank(distances, features, probabilities, log):
    if features is None or features.shape != (len(probabilities), 1024):
        raise RuntimeError('missing complete same-forward PE pooled gallery')
    if not torch.cuda.is_available():
        raise RuntimeError('mandatory CUDA FP64 graph unavailable; no CPU fallback')
    started = time.time()
    log('feature_rerank_enter', device='cuda', dtype='float64')
    graph, original = build_feature_graph(features, k=20, chunk_size=256,
                                          device='cuda', allow_gpu=True, log=log)
    graph_seconds = time.time()-started
    with numpy_blas_threads(2) as blas:
        log('feature_gate_enter', numpy_blas=blas)
        retained, count = retain_identical_predicted_profiles(probabilities, graph, original)
        log('feature_smooth_enter', original_edges=int(original.sum()), kept_edges=int(count.sum()))
        output = smooth_with_original_degree(distances, retained, count, original, alpha=.1)
    if not np.isfinite(output).all() or output.shape != distances.shape:
        raise RuntimeError('invalid final gated distances')
    log('feature_reranked', graph_seconds=round(graph_seconds, 4), total_seconds=round(time.time()-started, 4),
        original_edges=int(original.sum()), kept_edges=int(count.sum()), isolated=int((count==0).sum()),
        extra_encoder_forward=False, extra_pool_head_forward=True, numpy_blas=blas,
        graph_backend='CUDA_FP64_GEMM_CPU_exact_topk')
    return output
'''
    result = replace_function(source, "apply_feature_rerank", replacement)
    anchor = "from b121_blas import numpy_blas_threads\n"
    if result.count(anchor) != 1:
        raise ValueError("unique runtime import anchor required")
    return result.replace(anchor, anchor+"from b121_gpu_graph import build_feature_graph\n")


def validate_bench(report_path, exit_path):
    report, exit_record = (json.loads(Path(p).read_text()) for p in (report_path, exit_path))
    if (type(exit_record.get("actual_exit_code")) is not int or exit_record["actual_exit_code"] != 0
            or not exit_record.get("exec_session") or exit_record.get("report_sha256") != file_sha(report_path)):
        raise ValueError("actual successful child exit bound to exact benchmark report required")
    if report.get("status") != "OBSERVED_ALL4_PARITY_PASS" or set(report.get("cases", {})) != {"production_full", "PA100k", "PETA", "Market1501"}:
        raise ValueError("all4 actual fixed parity cases required")
    for name, result in report["cases"].items():
        keys = ("graph_index_differences", "degree_differences", "gated_index_differences", "kept_degree_differences",
                "distance_unequal", "queries_with_different_rank", "production_cpu_reference_unequal")
        if (not result.get("normalization_exact") or any(result.get(k) != 0 for k in keys)
                or result.get("distance_max_abs") != 0):
            raise ValueError("observed numerical difference is not exact equivalence")
        if name != "production_full" and (result.get("metric") or {}).get("unequal_per_query_ADM") != 0:
            raise ValueError("all3 actual source ADM parity required")
    if file_sha(HERE / "b121_gpu_graph.py") != GPU_GRAPH_SHA:
        raise ValueError("benchmarked GPU graph source changed")
    return [pin(report_path), pin(exit_path)]


def build(output, report_path, exit_path):
    inputs = validate_bench(report_path, exit_path)
    if file_sha(PARENT) != PARENT_SHA:
        raise ValueError("original submitted ZIP changed")
    output = Path(output).resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("new runtime_repair2-only output required")
    output.mkdir(exist_ok=False)
    package = output / "package"
    package.mkdir()
    started = time.monotonic()
    with zipfile.ZipFile(PARENT) as old:
        if len(old.namelist()) != len(set(old.namelist())):
            raise ValueError("duplicate archive entries")
        for name in old.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe archive entry")
            dest = package / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == "run.py":
                dest.write_text(patch_runner(old.read(name).decode()))
            elif name == "b121_runtime.py":
                dest.write_text(patch_runtime(old.read(name).decode()))
            elif name == "assets/config.json":
                cfg = json.loads(old.read(name))
                if (cfg["candidate"] != "B121G_PE4_FEATURE_AGREEMENT_LOCKED3" or cfg["time_budget_s"] != 292
                        or cfg["rank_reserve_s"] != 40 or cfg["plans"] != [{"name": "A", "members": [0, 1, 2]}]
                        or cfg["min_active_members"] != 3):
                    raise ValueError("original mandatory3 recipe changed")
                cfg["candidate"] = NAME
                cfg["graph_runtime_policy"] = "cuda_fp64_gemm_cpu_exact_topk_stage_diagnostics_v1"
                dest.write_text(json.dumps(cfg, indent=2)+"\n")
            else:
                with old.open(name) as source, dest.open("xb") as target:
                    shutil.copyfileobj(source, target, 8*2**20)
        for name in ("b121_gpu_graph.py", "b121_observe.py"):
            (package / name).write_bytes((HERE / name).read_bytes())
            inputs.append(pin(HERE / name))
        artifact = output / (NAME+".zip")
        with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(package).as_posix())
        preserved = []
        with zipfile.ZipFile(artifact) as new:
            for name in old.namelist():
                if name in ("run.py", "b121_runtime.py", "assets/config.json"):
                    continue
                with old.open(name) as a, new.open(name) as b:
                    if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                        raise ValueError("preserved asset/helper changed: "+name)
                preserved.append(name)
    report = {"status": "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED", "zip": pin(artifact), "inputs": inputs+[pin(PARENT), pin(Path(__file__))],
              "preserved_members": preserved, "elapsed_seconds": time.monotonic()-started,
              "next": "Actual complete new-ZIP runtime capture and independent numeric reference, process receipts, stage timing and resource review required before submission. Local speed is not A10 guarantee.",
              "official_failure_cause": "unknown external termination after predicted246.0; timeout/OOM not proved"}
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
