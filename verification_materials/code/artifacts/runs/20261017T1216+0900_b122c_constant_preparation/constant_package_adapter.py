"""One fixed B122C adapter. Root alone builds/replays the new revision.

Only b121_runtime.py's correction block and config identity change; run.py,
encoders, B119 scoring, CUDA32 graph and all parent assets remain byte exact.
The full-package reference is available after a root-admitted actual capture.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import time
import zipfile

import numpy as np

from upar import gallery_feature_replay as independent
from upar.gallery_feature_package import pin, sha, write_new
from upar.numpy_blas_scope import numpy_blas_threads

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAME = "B122C_SOURCE_HELD_CONSTANT_ONEBIT_RESCUE"
CONSTANT = 0.5599229900622915
POLICY = "source_held_equal_mass_prevalence_constant_symmetric_onebit_v1"
OLD = ROOT / "artifacts/runs/20261017T1129+0900_b121_runtime_repair2"
PARENT_DIR = OLD / "package_fp32_v1"
PARENT = PARENT_DIR / "B121G_PE4_FEATURE_AGREEMENT_CUDA32.zip"
PARENT_SHA = "fcaed7e28c3f96c4569a468b23ee9a50311614b6986e8f1dc5b8d17ea57a49ff"
RUNTIME_SHA = "e8a7ef51b0e5b334317b92bdad2ae8a1d6a9b55e3b249375abd150915f10b9d5"
HELPER_SHA = "7c3c99bf2701778fc1483e46526f01a51e248524b93c4b872052196848f7de1b"
ORIGIN = ROOT / "artifacts/runs/20261017T1205+0900_b122_portable_preparation/symmetric_portable_design.py"
ORIGIN_SHA = "9efbe868c595796e9c2b848aa63c467a0790f84007d718180bc062aadbfec914"
MODEL_SHA = "8805a189583d5843110cd9855c531bb1fb42e72478146849ef998339ea92eaac"
FIXED_REFERENCE = {
    OLD / "full_replay_fp32_v1/capture.npz": "ee4567242703c2d0826fa7caf98050ab93fa43a3964b755fd4090df2442bc382",
    OLD / "full_replay_fp32_v1/features.npy": "54123c977274bfde5f6e519b859517e54415bde2dd27c730432e15eb199ebf02",
    OLD / "full_replay_fp32_v1/distances.npy": "dfcb7687b28979b11520eae0a5c66146ede6fb3e1d5f7e8c6f8d66bac627d612",
    OLD / "actual_gpu_comparison_v3/graph_production_full.npz": "e957bf296d937e338991fb560a0fed178977745e1e3e0b454ba71129f8ba82b9",
    Path(independent.__file__): "e30ad4681e6c4cbb8744a7994c10ac1bb8362dec15ee48e5f813888645298e59",
}


def origin_and_inputs():
    if sha(ORIGIN) != ORIGIN_SHA or sha(HERE / "b122c_constant.py") != HELPER_SHA:
        raise ValueError("frozen origin validator or helper changed")
    spec = importlib.util.spec_from_file_location("_b122c_origin_validator", ORIGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, refs = module.validate_origin()
    if model.get("fit_prevalence") != CONSTANT:
        raise ValueError("exact saved OOF prevalence differs; no rounding or HPO")
    return refs + [pin(ORIGIN), pin(HERE / "b122c_constant.py"), pin(Path(__file__))]


def patch_config(config):
    if (config.get("candidate") != "B121G_PE4_FEATURE_AGREEMENT_CUDA32"
            or config.get("graph_runtime_policy") != "cuda_fp32_unit_dot_tf32_off_numeric_revision_v1"
            or config.get("plans") != [{"name": "A", "members": [0, 1, 2]}]
            or config.get("min_active_members") != 3 or config.get("time_budget_s") != 292
            or config.get("rank_reserve_s") != 40 or config.get("require_cuda") is not True
            or config.get("fail_closed_on_fallback") is not True):
        raise ValueError("exact mandatory3 CUDA32 parent recipe required")
    result = dict(config)
    result.update(candidate=NAME, onebit_rescue_policy=POLICY,
                  onebit_rescue_constant=CONSTANT, onebit_prevalence_origin_sha256=MODEL_SHA)
    return result


def patch_runtime(source):
    if hashlib.sha256(source.encode()).hexdigest() != RUNTIME_SHA:
        raise ValueError("exact immutable CUDA32 runtime required")
    anchor = "from b121_fp32_cosine_graph import build_feature_graph\n"
    original = """        log('feature_gate_enter', numpy_blas=blas)
        retained, count = retain_identical_predicted_profiles(probabilities, graph, original)
        log('feature_smooth_enter', original_edges=int(original.sum()), kept_edges=int(count.sum()))
        output = smooth_with_original_degree(distances, retained, count, original, alpha=.1)
"""
    replacement = """        log('feature_gate_enter', numpy_blas=blas)
        retained, count = retain_identical_predicted_profiles(probabilities, graph, original)
        rescue = constant_rescue(probabilities, graph, original)
        log('feature_smooth_enter', original_edges=int(original.sum()), kept_edges=int(count.sum()),
            rescued_edges=int(np.count_nonzero(rescue)), rescue_constant=CONSTANT, symmetric=True, fitting=False)
        output = smooth_with_rescue(distances, probabilities, graph, original, rescue, alpha=.1)
"""
    if source.count(anchor) != 1 or source.count(original) != 1:
        raise ValueError("unique frozen runtime patch boundaries required")
    result = source.replace(anchor, anchor + "from b122c_constant import constant_rescue, smooth_with_rescue, CONSTANT\n", 1)
    result = result.replace(original, replacement, 1)
    before = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after = {n.name: ast.dump(n) for n in ast.parse(result).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if set(before) != set(after) or [name for name in before if before[name] != after[name]] != ["apply_feature_rerank"]:
        raise ValueError("unrelated runtime function changed")
    return result


def build(output):
    inputs = origin_and_inputs()
    output = Path(output).resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("new private B122C output only")
    output.mkdir(exist_ok=False)
    package = output / "package"
    package.mkdir()
    started = time.monotonic()
    with zipfile.ZipFile(PARENT) as old:
        names = old.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate parent entries")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe archive entry")
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "b121_runtime.py":
                target.write_text(patch_runtime(old.read(name).decode()))
            elif name == "assets/config.json":
                target.write_text(json.dumps(patch_config(json.loads(old.read(name))), indent=2)+"\n")
            else:
                with old.open(name) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, 8*2**20)
        with (package / "b122c_constant.py").open("xb") as dest:
            dest.write((HERE / "b122c_constant.py").read_bytes())
        artifact = output / (NAME + ".zip")
        with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
        preserved = []
        with zipfile.ZipFile(artifact) as new:
            if set(new.namelist()) != set(names) | {"b122c_constant.py"}:
                raise ValueError("unexpected new member set")
            for name in names:
                if name in ("b121_runtime.py", "assets/config.json"):
                    continue
                with old.open(name) as a, new.open(name) as b:
                    if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                        raise ValueError("parent member changed: " + name)
                preserved.append(name)
    for ref in inputs:
        if pin(ref["path"]) != ref:
            raise ValueError("origin/source changed during build")
    report = {"status": "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED", "candidate": NAME,
              "policy": POLICY, "constant": CONSTANT, "zip": pin(artifact), "inputs": inputs,
              "parent": pin(PARENT), "preserved_members": preserved,
              "changed_members": ["b121_runtime.py", "assets/config.json"],
              "new_members": ["b122c_constant.py"], "elapsed_seconds": time.monotonic()-started,
              "new_fit_updates": 0, "new_features": 0, "coefficient_model_transferred": False,
              "next": "Root-admitted actual complete ZIP replay, actual exit, exact independent reference; not automatic submission."}
    write_new(output / "complete.json", report)
    return report


def checked_package(report_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if (report.get("status") != "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED"
            or report.get("candidate") != NAME or report.get("constant") != CONSTANT
            or report.get("policy") != POLICY or pin(report["zip"]["path"]) != report["zip"]
            or report.get("parent", {}).get("sha256") != PARENT_SHA):
        raise ValueError("exact completed B122C package required")
    for ref in report["inputs"]:
        if pin(ref["path"]) != ref:
            raise ValueError("package source/input changed")
    return report


def replay(package_report, output, time_scale):
    report = checked_package(package_report)
    independent.NAME = NAME
    result = independent.execute(report["zip"]["path"], Path(package_report).parent / "package", output,
                                 report["zip"]["sha256"], time_scale=time_scale)
    write_new(Path(output) / "revision_adapter_receipt.json", {
        "status": "NEW_ZIP_CAPTURE_COMPLETE_NOT_OFFICIAL_RUNTIME_ACCEPTED", "adapter": pin(Path(__file__)),
        "package_report": pin(package_report), "capture_report": pin(Path(output) / "execute_report.json"),
        "artifact": report["zip"], "official_A10_runtime_guaranteed": False})
    return result


def independent_constant_smooth(base, probabilities, neighbours, counts):
    """Independent row loop; fixed width/strides retain declared FP32 reduction.

    Does not import the portable or training rescue implementation. Reads only
    the original base distances, never updates a neighbor from the result.
    """
    if len(base) > 64:
        return np.concatenate([independent_constant_smooth(base[start:start+64], probabilities, neighbours, counts)
                               for start in range(0, len(base), 64)], axis=0)
    out = independent.independent_smooth(base, probabilities, neighbours, counts)
    bits, width = probabilities >= .5, neighbours.shape[1]
    for row, degree in enumerate(counts):
        if not degree:
            continue
        selected = neighbours[row, :degree]
        one = np.count_nonzero(bits[selected] != bits[row], axis=1) == 1
        if not np.any(one):
            continue
        delta = np.zeros((width, len(base)), dtype=base.dtype).T
        # Keep original edge positions and width; don't compact the one-bit
        # edges, which would silently change the floating reduction order.
        delta[:, :degree] = (base[:, selected]-base[:, row, None]) * (one.astype(base.dtype)*np.asarray(CONSTANT, base.dtype))[None, :]
        factor = np.asarray(.1, base.dtype) / np.asarray(degree, base.dtype)
        out[:, row] += factor * delta.sum(axis=1, dtype=base.dtype)
    return out


def reference(capture_dir, process_receipt, output):
    capture_dir, output = Path(capture_dir).resolve(), Path(output).resolve()
    if not capture_dir.is_relative_to(HERE) or not output.is_relative_to(HERE):
        raise ValueError("new private B122C capture/reference only")
    report_path = capture_dir / "execute_report.json"
    report, process = json.loads(report_path.read_text()), json.loads(Path(process_receipt).read_text())
    adapter = json.loads((capture_dir / "revision_adapter_receipt.json").read_text())
    if (type(process.get("actual_exit_code")) is not int or process["actual_exit_code"] != 0
            or not process.get("exec_session") or process.get("execute_report_sha256") != sha(report_path)
            or process.get("artifact_sha256") != report["artifact"]["sha256"]
            or report.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING"
            or adapter.get("capture_report") != pin(report_path) or adapter.get("artifact") != report["artifact"]):
        raise ValueError("successful actual child exit bound to capture and exact ZIP required")
    package = checked_package(adapter["package_report"]["path"])
    if (pin(adapter["package_report"]["path"]) != adapter["package_report"] or package["zip"] != report["artifact"]
            or report.get("device") != "cuda" or (report.get("rows"), report.get("queries")) != (33407, 3462)):
        raise ValueError("complete actual CUDA B122C package capture required")
    parent_cfg = json.loads((PARENT_DIR / "package/assets/config.json").read_text())
    if report.get("config") != patch_config(parent_cfg):
        raise ValueError("only fixed rescue config change permitted")
    refs = [pin(report_path), pin(process_receipt), pin(Path(__file__)), pin(capture_dir / "revision_adapter_receipt.json"), adapter["package_report"]]
    if {r["path"] for r in report["outputs"]} != {str(capture_dir / n) for n in ("capture.npz", "features.npy", "distances.npy")} or len(report["outputs"]) != 3:
        raise ValueError("exact three captured arrays required")
    refs += report["outputs"] + report["sources"] + report["annotations"] + [report["artifact"]]
    for path, expected in FIXED_REFERENCE.items():
        if sha(path) != expected:
            raise ValueError("frozen saved graph/reference changed")
        refs.append(pin(path))
    for ref in refs:
        if pin(ref["path"]) != ref:
            raise ValueError("capture/source changed")
    # Same feature bytes imply the frozen CUDA32 graph can be reused. This
    # checks exact selected-recipe smoothing, not independent GPU graph code.
    if sha(capture_dir / "features.npy") != FIXED_REFERENCE[OLD / "full_replay_fp32_v1/features.npy"]:
        raise ValueError("graph reuse requires exact original same-forward features")
    fields = {"probabilities", "queries", "image_paths", "attribute_names"}
    with np.load(capture_dir / "capture.npz", allow_pickle=False) as new, np.load(OLD / "full_replay_fp32_v1/capture.npz", allow_pickle=False) as old:
        if set(new.files) != fields or set(old.files) != fields:
            raise ValueError("capture vocabulary changed")
        for name in fields:
            a, b = new[name], old[name]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError("same producer input bridge failed: " + name)
        p, q, names = new["probabilities"], new["queries"], new["attribute_names"]
    with np.load(OLD / "actual_gpu_comparison_v3/graph_production_full.npz", allow_pickle=False) as graph:
        n, c = graph["neighbours"], graph["counts"]
    actual = np.load(capture_dir / "distances.npy", mmap_mode="r", allow_pickle=False)
    if actual.shape != (3462, 33407) or actual.dtype != np.float32:
        raise ValueError("complete actual FP32 distance matrix required")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {"inputs": refs, "role": "independent fixed-constant CPU reference, not quality/rate/runtime acceptance"})
    started, cpu = time.monotonic(), time.process_time()
    saved = np.lib.format.open_memmap(output / "independent_distances.npy", mode="w+", dtype=np.float32, shape=actual.shape)
    unequal, rank_diff, max_abs = 0, 0, 0.
    with numpy_blas_threads(2) as blas:
        for start in range(0, len(q), 64):
            end = min(start+64, len(q))
            base = independent.independent_base(p, q[start:end], names, report["config"])
            expected = independent_constant_smooth(base, p, n, c)
            saved[start:end] = expected
            observed = np.asarray(actual[start:end])
            if not np.isfinite(observed).all() or not np.isfinite(expected).all():
                raise ValueError("nonfinite distances")
            unequal += int(np.count_nonzero(observed.view(np.uint32) != expected.view(np.uint32)))
            rank_diff += sum(not np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable")) for a, b in zip(observed, expected))
            max_abs = max(max_abs, float(np.abs(observed-expected).max()))
            if start == 0 or end == len(q) or (start//64) % 8 == 0:
                print(json.dumps({"queries": end, "total": len(q), "unequal": unequal, "rank_diff": rank_diff}), flush=True)
    saved.flush()
    for ref in refs:
        if pin(ref["path"]) != ref:
            raise ValueError("source/input changed during reference")
    result = {"status": "PASS" if unequal == 0 and rank_diff == 0 else "FAIL", "candidate": NAME,
              "kind": "actual_CUDA32_saved_graph_exact_inputs_independent_fixed_constant_base_and_smooth",
              "constant": CONSTANT, "inputs": refs, "reference": pin(output / "independent_distances.npy"),
              "unequal_distance_values": unequal, "queries_with_different_ranks": rank_diff, "distance_max_abs": max_abs,
              "wall_seconds": time.monotonic()-started, "CPU_seconds": time.process_time()-cpu,
              "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "GPU_seconds": 0,
              "numpy_blas": blas, "quality_or_gold_claim": False, "official_A10_runtime_guaranteed": False,
              "graph_algorithm_independently_reimplemented": False, "actual_process_exit_confirmed": False}
    write_new(output / "report.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("exact reference failed; no tolerance waiver or automatic retry")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "replay", "reference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-reviewed", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--package-report", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--process-receipt", type=Path)
    parser.add_argument("--time-scale", type=float, default=1.)
    args = parser.parse_args()
    if not args.root_reviewed:
        parser.error("explicit root review required; never launch implicitly")
    if args.action == "build":
        value = build(args.output)
    elif args.action == "replay":
        if not args.allow_gpu or args.package_report is None:
            parser.error("fresh root GPU admission and exact package report required")
        value = replay(args.package_report, args.output, args.time_scale)
    else:
        if args.capture_dir is None or args.process_receipt is None:
            parser.error("actual capture and successful native child receipt required")
        value = reference(args.capture_dir, args.process_receipt, args.output)
    print(json.dumps({"status": value["status"], "candidate": NAME}), flush=True)
