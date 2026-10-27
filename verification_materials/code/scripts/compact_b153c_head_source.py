#!/usr/bin/env python3
"""Extract the needed hp6 head, then reproduce B153C's stored head without a GPU.

Never changes the parent checkpoint or the approved submission. The compact
file replaces hp6_all ONLY for head reconstruction, never for backbone use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def pin(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def byte_equal(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(byte_equal(a[k], b[k]) for k in a))
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def extract_heads(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        heads = {k: handle.get_tensor(k).clone().contiguous()
                 for k in handle.keys() if k.startswith("head.")}
    if not heads:
        raise ValueError("source contains no head tensors")
    save_file(heads, str(destination), metadata={"scope": "head-only; not a backbone checkpoint"})
    restored = load_file(str(destination), device="cpu")
    if not byte_equal(heads, restored):
        raise ValueError("head extraction changed tensor bytes")
    return {"keys": sorted(heads), "tensor_bytes": sum(v.numel() * v.element_size() for v in heads.values()),
            "all_tensor_bytes_equal": True}


def reconstruct(mod, hp, qm, h6):
    r224 = {}
    for k in hp["d3hpq"]:
        if hp["d3hpq"][k].is_floating_point():
            sum4 = hp["d3hpq"][k].float() + hp["d3hpqr"][k].float() + hp["d3hpqrq"][k].float() + hp["d3hpqs"][k].float()
            r224[k] = 3.0 * h6[k].float() - 0.5 * sum4
        else:
            r224[k] = h6[k]
    mems = {"q": hp["d3hpq"], "qr": hp["d3hpqr"], "qrq": hp["d3hpqrq"], "qs": hp["d3hpqs"], "qm": qm, "r224": r224}
    w4 = {"q": .20, "qr": .30, "qrq": .30, "qs": .20, "qm": 0., "r224": 0.}
    w610 = {"q": .18, "qr": .27, "qrq": .27, "qs": .18, "qm": 0., "r224": .10}
    w5m15 = {"q": .17, "qr": .255, "qrq": .255, "qs": .17, "qm": .15, "r224": 0.}
    w7m15 = {"q": .15, "qr": .225, "qrq": .225, "qs": .15, "qm": .15, "r224": .10}
    overrides = {c: w7m15 for c in (10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34)}
    overrides.update({c: w4 for c in (3, 7, 20)})
    overrides.update({c: w5m15 for c in (33, 34)})
    head = mod.colselective(mems, w610, overrides, out_scale=1.03)
    packed = {k: mod.pack(v) for k, v in head.items()}
    packed["head.weight"] = mod.pack(head["head.base.weight"])
    packed["head.bias"] = mod.pack(head["head.base.bias"])
    return packed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-source", type=Path,
                        help="Reconstruct using an existing compact source; never opens the full hp6_all parent")
    parser.add_argument("--origin-receipt", type=Path,
                        help="Original full-parent extraction receipt, required with --compact-source")
    args = parser.parse_args()
    if bool(args.compact_source) != bool(args.origin_receipt):
        parser.error("--compact-source and --origin-receipt must be supplied together")
    root, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runs = root / "artifacts/runs"
    helper = runs / "20261022T0400+0900_deliverables_prep/verify_weight_chain_v2.py"
    spec = importlib.util.spec_from_file_location("original_head_verifier", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.set_num_threads(2)
    ks = runs / "20261008T231500+0900_knob_sweep"
    parent = runs / "20261017T0345+0900_hplus6/hp6_all/model.safetensors"
    compact = args.compact_source.resolve() if args.compact_source else out / "hp6_all_heads.safetensors"
    sources = {n: ks / f"{n}_all_trainval_seed20261007_ema0.999/model.safetensors"
               for n in ("d3hpq", "d3hpqr", "d3hpqrq", "d3hpqs")}
    backup = runs / "20261018T0300+0900_gpu_representation_and_soup_cycle/extracted_features_v1/extra_heads_backup_v1.pt"
    pq = ks / "d3hpq_all_trainval_seed20261007_ema0.999/val_logits.npz"
    pm = ks / "d3hpqm_all_trainval_seed20261007_ema0.999/val_logits.npz"
    package = runs / "20261021T2250+0900_takeover_knob_search/build_b153c_v1/package/assets/model_1.weights.pt"
    paths = [*sources.values(), backup, pq, pm, package, helper, Path(__file__).resolve()]
    origin = None
    if args.compact_source:
        origin = json.loads(args.origin_receipt.read_text())
        actual = pin(compact)
        if (origin.get("status") != "PASS" or not origin["extraction"]["all_tensor_bytes_equal"]
                or actual["sha256"] != origin["compact"]["sha256"] or actual["bytes"] != origin["compact"]["bytes"]):
            raise ValueError("compact source does not match its full-parent extraction receipt")
        paths.extend([compact, args.origin_receipt.resolve()])
    else:
        paths.append(parent)
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    pins = [pin(p) for p in paths]
    extraction = (origin["extraction"] if origin else extract_heads(parent, compact))
    hp = {n: mod.heads(p) for n, p in sources.items()}
    qm = dict(torch.load(str(backup), map_location="cpu", weights_only=True)["d3hpqm_all_trainval_seed20261007_ema0.999"])
    with np.load(pq, allow_pickle=False) as q, np.load(pm, allow_pickle=False) as m:
        shift = torch.from_numpy((np.log(q["pos_weight"]) - np.log(m["pos_weight"])).astype(np.float32))
    for key in ("head.base.bias", "head.bias"):
        if key in qm:
            qm[key] = (qm[key].float() + shift).to(qm[key].dtype).contiguous()
    new = reconstruct(mod, hp, qm, mod.heads(compact))
    stored_all = torch.load(str(package), map_location="cpu", mmap=True, weights_only=True)
    stored = {k: v for k, v in stored_all.items() if k.startswith("head.")}
    checks = {"same_keyset": stored.keys() == new.keys(), "compact_vs_packaged": byte_equal(new, stored)}
    if origin:
        checks["compact_origin_receipt_matched"] = True
    else:
        old = reconstruct(mod, hp, qm, mod.heads(parent))
        checks["original_vs_compact"] = byte_equal(old, new)
    checks["source_metadata_unchanged"] = all((p.stat().st_size, p.stat().st_mtime_ns) == before[str(p)] for p in paths)
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Exact head-only replacement of hp6_all for reconstruction; no accuracy change or backbone substitution.",
              "checks": checks, "extraction": extraction, "packaged_head_keys": sorted(stored),
              "sources": pins, "compact": pin(compact),
              "bytes_avoided_per_transfer": (origin["bytes_avoided_per_transfer"] if origin else parent.stat().st_size - compact.stat().st_size),
              "mode": "compact_source_only" if origin else "extract_from_full_parent",
              "full_hp6_parent_opened_this_run": origin is None,
              "source_deleted": False, "submission_modified": False, "uploaded": False,
              "limitations": ["Deleted 224-px parent checkpoints remain unavailable; extraction does not repair their ancestry.",
                              "Other old verifiers still reference full hp6_all. This script's --compact-source/--origin-receipt path does not require it."]}
    with (out / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "compact", "bytes_avoided_per_transfer")}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
