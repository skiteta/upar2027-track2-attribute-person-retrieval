#!/usr/bin/env python3
"""Assemble a verified, code-only B153C delivery revision locally; never upload it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--factsheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, delivery = args.workspace.resolve(), args.delivery.resolve()
    report_path = args.verification.resolve()
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS_SCOPED_CPU_TESTS" or report["returncode"] != 0:
        raise ValueError("passing relocated test report required")
    if report["external_missing_or_changed"]:
        raise ValueError("external artifact inventory changed")
    if digest((root / "scripts/verify_delivery_snapshot.py").read_bytes()) != report["verifier_sha256"]:
        raise ValueError("relocation verifier changed; retain the exact verified source")
    manifest = json.loads((delivery / "manifest.json").read_text())
    if digest((delivery / "manifest.json").read_bytes()) != report["manifest_sha256"]:
        raise ValueError("delivery manifest changed")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("new output directory required")
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}

    def add(name: str, data: bytes, mode: int = 0o644):
        if name in files or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("duplicate or unsafe archive name: " + name)
        if any(p in {".secrets", ".os_state", "work_os", "__pycache__", ".git"} for p in Path(name).parts):
            raise ValueError("private/control path in delivery: " + name)
        files[name] = data
        modes[name] = 0o755 if mode & 0o111 else 0o644

    relocated = report_path.parent / "relocated"
    for row in manifest["files"]:
        src = delivery / row["path"]
        data = src.read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError("frozen source changed: " + row["path"])
        if row["path"].startswith("code_snapshot/"):
            rel = row["path"].removeprefix("code_snapshot/")
            if (relocated / rel).read_bytes() != data:
                raise ValueError("tested copy changed: " + rel)
            add("code/" + rel, data, src.stat().st_mode)
        else:
            add("original_delivery/" + row["path"], data)
    for row in report.get("fixtures", []):
        data = (relocated / row["path"]).read_bytes()
        if row["status"] != "PASS" or digest(data) != row["sha256"]:
            raise ValueError("tested fixture changed")
        add("code/" + row["path"], data)
    add("original_delivery/manifest.json", (delivery / "manifest.json").read_bytes())
    add("original_delivery/MANIFEST.md", (delivery / "MANIFEST.md").read_bytes())
    add("DELIVERY_STATUS.json", args.status.read_bytes())
    add("FACT_SHEET_DRAFT.md", args.factsheet.read_bytes())
    for name in ("verify_delivery_snapshot.py", "verify_b153c_dino_backbone.py", "assemble_b153c_delivery.py"):
        add("code/scripts/" + name, (root / "scripts" / name).read_bytes())
    for name in ("verification.json", "tests.log", "tests.xml"):
        add("verification/relocated_cpu/" + name, (report_path.parent / name).read_bytes())
    proof = report_path.parent.parent / "dino_all_blocks_v1.json"
    proof_data = json.loads(proof.read_text())
    if proof_data["status"] != "PASS_ALL_LEARNED_BACKBONE_TENSORS":
        raise ValueError("all-block DINO proof required")
    add("verification/dino_all_blocks.json", proof.read_bytes())
    add("verification/test_fixtures.json", (report_path.parent.parent / "test_fixtures_v1.json").read_bytes())
    rows = [{"path": k, "bytes": len(v), "sha256": digest(v), "mode": oct(modes[k])} for k, v in sorted(files.items())]
    inventory = {"candidate": manifest["candidate"], "status": "LOCAL_ONLY_NOT_SENT",
                 "files": rows, "count": len(rows),
                 "scope": "Code + official validation test fixtures + evidence, no models/images/credentials.",
                 "limitations": "See DELIVERY_STATUS.json. Unit tests use the existing interpreter; not a clean install or full GPU reproduction."}
    add("MANIFEST.json", (json.dumps(inventory, indent=2) + "\n").encode())
    output.mkdir(parents=True)
    zip_path = output / "B153C_code_delivery_v2.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 23, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | modes[name]) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError("archive verification failed")
        if any(archive.read(name) != data for name, data in files.items()):
            raise ValueError("archive bytes differ")
        if any((archive.getinfo(name).external_attr >> 16) & 0o777 != modes[name] for name in files):
            raise ValueError("archive permissions differ")
    receipt = {"status": "LOCAL_ARCHIVE_VERIFIED_NOT_SENT", "candidate": manifest["candidate"],
               "zip": {"path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": digest(zip_path.read_bytes())},
               "member_count": len(files), "uncompressed_bytes": sum(map(len, files.values())),
               "crc_and_member_sha_verified": True, "executable_modes_preserved": True,
               "model_zip_modified": False, "uploaded": False,
               "contains_original_home_paths": True, "team_details_complete": False,
               "scope": inventory["scope"], "limitations": inventory["limitations"]}
    with (output / "archive_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
