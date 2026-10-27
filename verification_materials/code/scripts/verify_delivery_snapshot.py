#!/usr/bin/env python3
"""Verify a frozen delivery, then test a relocated copy without touching its sources.

This checks packaging and CPU unit tests, not retraining, official runtime or quality.
Large external artifacts are size-checked unless --hash-external is explicitly used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def under(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe manifest path: {relative}")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"manifest path leaves root: {relative}")
    return path


def check_file(root: Path, item: dict, *, full_hash: bool) -> dict:
    path = under(root, item["path"])
    row = {"path": item["path"], "check": "sha256" if full_hash else "size_only"}
    if not path.is_file():
        return {**row, "status": "MISSING"}
    row["bytes"] = path.stat().st_size
    if row["bytes"] != item["bytes"]:
        return {**row, "status": "SIZE_MISMATCH"}
    if full_hash:
        row["sha256"] = sha(path)
        if row["sha256"] != item["sha256"]:
            return {**row, "status": "HASH_MISMATCH"}
    return {**row, "status": "PASS"}


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-test", action="append", default=[])
    parser.add_argument("--fixture-manifest", type=Path,
                        help="Explicit hash-pinned small test inputs copied from the workspace")
    parser.add_argument("--hash-external", action="store_true")
    args = parser.parse_args()
    delivery, workspace = args.delivery.resolve(), args.workspace.resolve()
    # Keep the venv launcher path: resolving its symlink would select bare Python.
    python = args.python.absolute()
    output = args.output.resolve()
    if output == delivery or output.is_relative_to(delivery):
        raise ValueError("output must be outside the frozen delivery")
    if output.exists():
        raise FileExistsError("use a new immutable output directory")
    output.mkdir(parents=True)
    manifest_path = delivery / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "delivery": str(delivery), "manifest_sha256": sha(manifest_path),
        "verifier_sha256": sha(Path(__file__)),
        "candidate": manifest["candidate"],
        "scope": "Frozen-file integrity and relocated CPU unit tests only; no training, GPU inference, network or submission.",
        "files": [check_file(delivery, row, full_hash=True) for row in manifest["files"]],
        "external": [check_file(workspace, row, full_hash=args.hash_external)
                     for row in manifest.get("external", []) if row.get("sha256") is not None],
        "external_aggregate_rows": [row for row in manifest.get("external", []) if row.get("sha256") is None],
        "external_b124c_package": [check_file(workspace, row, full_hash=args.hash_external)
                                   for row in manifest.get("external_b124c_package", [])],
    }
    write_json(output / "integrity.json", report)
    if any(row["status"] != "PASS" for row in report["files"]):
        print(json.dumps({"status": "FROZEN_DELIVERY_CHANGED", "report": str(output / "integrity.json")}))
        return 2
    relocated = output / "relocated"
    relocated.mkdir()
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.parts[0] != "code_snapshot":
            continue
        target = under(relocated, str(Path(*rel.parts[1:])))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(delivery / rel, target)
    if args.fixture_manifest:
        fixtures = json.loads(args.fixture_manifest.read_text())
        report["fixture_manifest_sha256"] = sha(args.fixture_manifest)
        report["fixtures"] = [check_file(workspace, row, full_hash=True) for row in fixtures["files"]]
        if any(row["status"] != "PASS" for row in report["fixtures"]):
            report.update(status="FAIL_FIXTURE_INTEGRITY", returncode=2)
            write_json(output / "verification.json", report)
            return 2
        for row in fixtures["files"]:
            target = under(relocated, row["path"])
            if target.exists():
                raise FileExistsError(f"fixture would overwrite snapshot: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(under(workspace, row["path"]), target)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(relocated / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    probe = subprocess.run([str(python), "-c",
        "import importlib.metadata as m,json,upar; from pathlib import Path; "
        "p=Path(upar.__file__).resolve(); assert p.is_relative_to(Path.cwd()); "
        "names=['torch','torchvision','timm','safetensors','numpy','pillow','pytest']; "
        "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
        "print(json.dumps({'upar_path':str(p),'observed_packages':{n:installed.get(n) for n in names}}))"],
        cwd=relocated, env=env, capture_output=True, text=True)
    if probe.returncode:
        report.update(status="FAIL_ENVIRONMENT_PROBE", returncode=probe.returncode,
                      probe_stdout=probe.stdout, probe_stderr=probe.stderr)
        write_json(output / "verification.json", report)
        print(json.dumps({"status": report["status"], "stderr": probe.stderr}))
        return probe.returncode
    report["environment"] = json.loads(probe.stdout)
    tests = ["tests", *args.extra_test]
    for test in tests:
        if not under(relocated, test).exists():
            raise FileNotFoundError(test)
    command = [str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--junitxml={output / 'tests.xml'}", *tests]
    report["command"] = command
    with (output / "tests.log").open("x") as log:
        completed = subprocess.run(command, cwd=relocated, env=env, stdout=log, stderr=subprocess.STDOUT)
    report["returncode"] = completed.returncode
    xml = output / "tests.xml"
    report["test_suites"] = [dict(suite.attrib) for suite in ET.parse(xml).getroot().iter("testsuite")] if xml.exists() else []
    report["status"] = "PASS_SCOPED_CPU_TESTS" if completed.returncode == 0 else "FAIL_CPU_TESTS"
    report["external_missing_or_changed"] = [row for key in ("external", "external_b124c_package")
                                               for row in report[key] if row["status"] != "PASS"]
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["limitations"] = ["Existing interpreter, not a freshly installed environment.",
        "External models are not copied; these unit tests do not exercise full inference or training.",
        "Old absolute-path reconstruction scripts remain documented gaps, not certified portable."]
    write_json(output / "verification.json", report)
    print(json.dumps({k: report[k] for k in ("status", "returncode", "test_suites", "external_missing_or_changed")}))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
