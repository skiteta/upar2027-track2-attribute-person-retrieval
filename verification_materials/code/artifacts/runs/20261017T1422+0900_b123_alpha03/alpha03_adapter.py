"""Thin alpha .3 revision of immutable B122C; root-only build/replay/reference.

No old artifact validator globals are rewritten. Parent validation remains B122
validation; the new identity is explicit in the separate capture helper.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from upar import fixed_revision_replay as capture
from upar.gallery_feature_package import pin, sha, write_new, replace_once

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAME = "B123C_CONSTANT_RESCUE_ALPHA03"
ALPHA = .3
OLD = ROOT / "artifacts/runs/20261017T1216+0900_b122c_constant_preparation"
PARENT = OLD / "package_v1/B122C_SOURCE_HELD_CONSTANT_ONEBIT_RESCUE.zip"
PARENT_SHA = "68c0f96dff6b9e2b355402220dd9d364142817fce9b6cd9d0cf66b2a147ee4a1"
RUNTIME_SHA = "ea5a4767f8b7d819391e018186c5e83f12fb9dd274f0db2b5cea4867f5ee1bbe"
SCIENCE = ROOT / "artifacts/runs/20261017T1418+0900_rescue_strength"
FIXED = {
    OLD / "constant_package_adapter.py": "7572f2f3ac5789718ebb36de2fa1c8ff0b649e8870038e8c1f55d9718d7627d5",
    OLD / "package_v1/complete.json": "132a1f86a43125e5f35fceb836e9313cccfefb66bac963ecfd987690c0bec7b4",
    SCIENCE / "protocol.json": "5cd8fd90c9249162653e84ed64a45f5c5b2b54d7d1a5879597ecdb27404ce762",
    SCIENCE / "evaluation_v1/report.json": "2fdd5badfaf615e900c57988fbc50660871eaffe6559588767bbdd226e958010",
    SCIENCE / "actual_exit_v2.json": "7602eaf4be21ff115ebfa288e4531a1893b2dfadf02e5b1bc832ee652c863729",
}


def parent_module():
    path = OLD / "constant_package_adapter.py"
    if sha(path) != FIXED[path]:
        raise ValueError("immutable B122 validator changed")
    spec = importlib.util.spec_from_file_location("_immutable_b122_alpha03_parent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    for path, digest in FIXED.items():
        if sha(path) != digest:
            raise ValueError("fixed science/parent input changed")
    exit_record = json.loads((SCIENCE / "actual_exit_v2.json").read_text())
    if type(exit_record.get("actual_exit_code")) is not int or exit_record["actual_exit_code"] != 0:
        raise ValueError("actual alpha comparison exit required")
    parent = parent_module().checked_package(OLD / "package_v1/complete.json")
    if parent["zip"] != pin(PARENT) or parent["zip"]["sha256"] != PARENT_SHA:
        raise ValueError("exact B122 parent ZIP required")
    refs = parent["inputs"] + [pin(p) for p in FIXED] + [pin(PARENT)]
    refs += [pin(Path(__file__)), pin(HERE / "alpha03_reference.py"), pin(Path(capture.__file__)),
             pin(ROOT / "tests/test_b123_alpha03.py")]
    refs += [pin(ROOT / "src/upar" / name) for name in ("gallery_feature_package.py", "numpy_blas_scope.py")]
    return refs


def patch_config(config):
    old = parent_module()
    expected = json.loads((old.PARENT_DIR / "package/assets/config.json").read_text())
    if config != old.patch_config(expected) or config["gallery_rerank"]["alpha"] != .1:
        raise ValueError("complete exact B122 parent config required")
    value = copy.deepcopy(config)
    value["candidate"] = NAME
    value["gallery_rerank"]["alpha"] = ALPHA
    value["constant_rescue_diffusion_alpha"] = ALPHA
    return value


def patch_runtime(source):
    if hashlib.sha256(source.encode()).hexdigest() != RUNTIME_SHA:
        raise ValueError("exact B122 runtime required")
    old_call = "output = smooth_with_rescue(distances, probabilities, graph, original, rescue, alpha=.1)"
    new = replace_once(source, old_call, old_call.replace("alpha=.1", "alpha=.3"))
    old_log = "rescued_edges=int(np.count_nonzero(rescue)), rescue_constant=CONSTANT, symmetric=True, fitting=False)"
    new = replace_once(new, old_log, old_log[:-1] + ", diffusion_alpha=.3)")
    before, after = ast.parse(source), ast.parse(new)
    a = {n.name: ast.dump(n) for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    b = {n.name: ast.dump(n) for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if a.keys() != b.keys() or [k for k in a if a[k] != b[k]] != ["apply_feature_rerank"]:
        raise ValueError("unrelated function changed")
    return new


def build(output):
    refs = inputs()
    output = Path(output).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError("new private output only; preserve partial outputs")
    if shutil.disk_usage(HERE).free < 3 * PARENT.stat().st_size + 20 * 2**30:
        raise ValueError("preserve20GiB disk reserve before building")
    output.mkdir()
    package = output / "package"
    package.mkdir()
    started = time.monotonic()
    with zipfile.ZipFile(PARENT) as old:
        names = old.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate parent ZIP members")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe ZIP entry")
            dest = package / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as stream:
                if name == "b121_runtime.py":
                    stream.write(patch_runtime(old.read(name).decode()).encode())
                elif name == "assets/config.json":
                    stream.write((json.dumps(patch_config(json.loads(old.read(name))), indent=2)+"\n").encode())
                else:
                    with old.open(name) as source:
                        shutil.copyfileobj(source, stream, 8*2**20)
        artifact = output / (NAME + ".zip")
        with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(package.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(package).as_posix())
        preserved = []
        with zipfile.ZipFile(artifact) as new:
            if set(new.namelist()) != set(names) or len(new.namelist()) != len(names):
                raise ValueError("new ZIP member set changed")
            for name in names:
                if name not in ("b121_runtime.py", "assets/config.json"):
                    with old.open(name) as a, new.open(name) as b:
                        if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                            raise ValueError("unchanged member differs")
                    preserved.append(name)
    for ref in refs:
        if pin(ref["path"]) != ref:
            raise ValueError("source/input changed during build")
    report = {"status": "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED", "candidate": NAME,
        "alpha": ALPHA, "zip": pin(artifact), "parent": pin(PARENT), "inputs": refs,
        "config": json.loads((package / "assets/config.json").read_text()),
        "changed_members": ["b121_runtime.py", "assets/config.json"], "new_members": [],
        "preserved_members": preserved, "elapsed_seconds": time.monotonic()-started,
        "new_fit_updates": 0, "new_features": 0, "official_A10_runtime_guaranteed": False}
    write_new(output / "complete.json", report)
    return report


def checked_package(report_path):
    report_path = Path(report_path).resolve()
    if not report_path.is_relative_to(HERE):
        raise ValueError("new private B123 report only")
    report = json.loads(report_path.read_text())
    expected = patch_config(json.loads((OLD / "package_v1/package/assets/config.json").read_text()))
    if (report.get("candidate") != NAME or report.get("alpha") != ALPHA
            or report.get("status") != "PACKAGE_BUILT_NOT_FULL_RUNTIME_ACCEPTED"
            or report.get("parent", {}).get("sha256") != PARENT_SHA or report.get("config") != expected
            or Path(report.get("zip", {}).get("path", "")).resolve() != report_path.parent / (NAME + ".zip")
            or report.get("changed_members") != ["b121_runtime.py", "assets/config.json"]
            or report.get("new_members") != []):
        raise ValueError("exact new alpha.3 package report required")
    for ref in report["inputs"] + [report["zip"], report["parent"]]:
        if pin(ref["path"]) != ref:
            raise ValueError("package/input bytes changed")
    return report


def replay(package_report, output, time_scale):
    report = checked_package(package_report)
    result = capture.execute(report["zip"]["path"], Path(package_report).parent / "package", output,
                             report["zip"]["sha256"], report["config"], time_scale=time_scale)
    write_new(Path(output) / "revision_adapter_receipt.json", {"status": "NEW_ZIP_CAPTURED_NOT_ACCEPTED",
        "candidate": NAME, "adapter": pin(Path(__file__)), "package_report": pin(package_report),
        "capture_report": pin(Path(output) / "execute_report.json"), "artifact": report["zip"]})
    return result


def independent_alpha_smooth(base, probabilities, neighbours, counts):
    """Independent row loops, both terms alpha.3; fixed FP32 reduction order."""
    if len(base) > 64:
        return np.concatenate([independent_alpha_smooth(base[s:s+64], probabilities, neighbours, counts)
                               for s in range(0, len(base), 64)])
    constant = .5599229900622915
    bits, width, out = probabilities >= .5, neighbours.shape[1], base.copy()
    for row, degree in enumerate(counts):
        if not degree:
            continue
        selected = neighbours[row, :degree]
        hamming = np.count_nonzero(bits[selected] != bits[row], axis=1)
        same = selected[hamming == 0]
        if len(same):
            values = np.zeros((width, len(base)), dtype=base.dtype).T
            values[:, :len(same)] = base[:, same]
            mean = values.sum(axis=1, dtype=base.dtype) / np.asarray(len(same), base.dtype)
            mass = np.asarray(ALPHA, base.dtype) * (np.asarray(len(same), base.dtype) / np.asarray(degree, base.dtype))
            out[:, row] = base[:, row] + mass * (mean-base[:, row])
        if np.any(hamming == 1):
            delta = np.zeros((width, len(base)), dtype=base.dtype).T
            weights = (hamming == 1).astype(base.dtype) * np.asarray(constant, base.dtype)
            delta[:, :degree] = (base[:, selected]-base[:, row, None]) * weights[None, :]
            out[:, row] += (np.asarray(ALPHA, base.dtype)/np.asarray(degree, base.dtype)) * delta.sum(axis=1, dtype=base.dtype)
    return out


def reference(capture_dir, process_receipt, output):
    from alpha03_reference import run_reference
    return run_reference(capture_dir, process_receipt, output)


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
        parser.error("root review required; no implicit build/GPU")
    if args.action == "build":
        result = build(args.output)
    elif args.action == "replay":
        if not args.allow_gpu or args.package_report is None:
            parser.error("fresh root admission and exact new package required")
        result = replay(args.package_report, args.output, args.time_scale)
    else:
        if args.capture_dir is None or args.process_receipt is None:
            parser.error("actual capture/exit required")
        result = reference(args.capture_dir, args.process_receipt, args.output)
    print(json.dumps({"status": result["status"], "candidate": NAME}), flush=True)
