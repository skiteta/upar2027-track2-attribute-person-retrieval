#!/usr/bin/env python3
"""Freeze H+6 provenance, then build the four CPU weight soups sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from upar.streaming_weight_soup import build_atomic_soup, sha256_file, tensor_schema, file_identity


RECIPES = ("d3hpq", "d3hpqr", "d3hpqrq", "d3hpqs", "d3hp224q", "d3hp224qr")
FOLDS = ("PA100k", "PETA", "Market1501", "all")
PARENT_ROOT = Path("artifacts/runs/20261008T231500+0900_knob_sweep")
DEFAULT_ROOT = Path("artifacts/runs/20261017T0345+0900_hplus6")
PROGRESS_LOG = None


def progress(record):
    line = json.dumps({"at": datetime.now(timezone.utc).isoformat(), **record})
    print(line, flush=True)
    if PROGRESS_LOG is not None:
        with PROGRESS_LOG.open("a") as stream:
            stream.write(line + "\n")


def freeze(root: Path):
    protocol_path = root / "protocol_v1.json"
    manifest_path = root / "parent_manifest_v1.json"
    if protocol_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace an already frozen protocol/parent manifest")
    root.mkdir(parents=True, exist_ok=True)
    groups = {}
    for fold in FOLDS:
        suffix = fold if fold != "all" else "all_trainval_seed20261007"
        parents = []
        for recipe in RECIPES:
            run = PARENT_ROOT / f"{recipe}_{suffix}_ema0.999"
            config = json.loads((run / "config.json").read_text())
            report = json.loads((run / "train_report.json").read_text())
            model = run / "model.safetensors"
            if config["seed"] != 20261007 or config["head"] != "attrq" or config["ema_used"] != .999:
                raise ValueError(f"seed/head/EMA differs: {run}")
            if config["fold"] != fold or config["include_val"] != (fold == "all"):
                raise ValueError(f"fit scope differs: {run}")
            expected_key = "dinov3_hp16_224" if "224" in recipe else "dinov3_hp16_192"
            if config["backbone"] != expected_key or report["status"] != "TRAIN_DONE":
                raise ValueError(f"backbone/finished status differs: {run}")
            if config.get("init_from") or config.get("label_mask") or config.get("coarse_positive"):
                raise ValueError(f"unexpected upstream fit recipe: {run}")
            with np.load(run / "val_logits.npz") as prediction:
                pos_weight = prediction["pos_weight"].astype(np.float32)
            if not np.array_equal(pos_weight, np.asarray(report["pos_weight"], dtype=np.float32)):
                raise ValueError(f"pos_weight report mismatch: {run}")
            before = file_identity(model)
            digest = sha256_file(model)
            if file_identity(model) != before:
                raise ValueError(f"parent changed while hashing: {run}")
            schema = tensor_schema(model)
            if any(any(word in key for word in ("pos_embed", "rope", "freq")) for key in schema):
                raise ValueError(f"unexpected positional state tensor: {run}")
            parent = {"recipe": recipe, "run": str(run), "weight": 1 / 6,
                      "checkpoint": {"path": str(model), "sha256": digest, **before},
                      "config_sha256": sha256_file(run / "config.json"),
                      "train_report_sha256": sha256_file(run / "train_report.json"),
                      "config": config, "fit_rows": report["fit_rows"],
                      "fit_rows_by_source": report["fit_rows_by_source"],
                      "pos_weight": pos_weight.tolist(), "tensor_schema": schema}
            parents.append(parent)
            progress({"event": "PARENT_FROZEN", "fold": fold, "recipe": recipe, "sha256": digest})
        first = parents[0]
        for parent in parents[1:]:
            for key in ("tensor_schema", "fit_rows", "fit_rows_by_source", "pos_weight"):
                if parent[key] != first[key]:
                    raise ValueError(f"parent {key} mismatch in {fold}")
            if parent["config"]["cache_dir"] != first["config"]["cache_dir"]:
                raise ValueError("different cache")
        if fold != "all" and fold in first["fit_rows_by_source"]:
            raise ValueError("held source in reported fit rows")
        cfg = dict(first["config"], backbone="dinov3_hp16_192",
                   output_dir=str(root / f"hp6_{fold}"), soup_of=[p["run"] for p in parents],
                   soup_weights=[1 / 6] * 6, pos_weight=first["pos_weight"],
                   soup_recipe="hplus6_uniform_192_v1", parent_order=list(RECIPES))
        groups[fold] = {"parents": parents, "output": str(root / f"hp6_{fold}"), "config": cfg}
    manifest = {"schema": 1, "frozen_at": datetime.now(timezone.utc).isoformat(), "groups": groups}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    meta_path = Path(groups["PA100k"]["config"]["cache_dir"]) / "meta.npz"
    protocol = {
        "schema": 1, "id": "20261017_hplus6_uniform192_v1", "frozen_at": manifest["frozen_at"],
        "parent_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "cache_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        "recipe_order": list(RECIPES), "weight_per_recipe": 1 / 6,
        "averaging": "actual FP32 state_dict average in fixed parent order; not probability averaging",
        "inference_input_hw": [192, 96], "inference_views": ["original"],
        "pretrained_base": "pretrained/_h1/vit_huge_plus_patch16_dinov3.lvd1689m/model.safetensors",
        "compatibility": {"same_architecture": "vit_huge_plus_patch16_dinov3.lvd1689m",
            "positional_mechanism": "DINOv3 RoPE, dynamic_img_size=True, use_abs_pos_emb=False",
            "state_schema_exact_match": True, "same_seed": 20261007, "ema": .999,
            "head": "attrq", "same_basin": "supported by shared initialization recipe and prior successful cross-resolution soups; not guaranteed for six-parent soup"},
        "comparison": {"baseline": "B119G", "folds": list(FOLDS[:-1]),
            "external_weights": [.4, .35, .25], "pe_eva": "unchanged B119G",
            "post": {"renorm": [[8, 20, .5]], "clip": 1e-5, "entropy_delta": .1875},
            "reliability": "unchanged B011 gamma=.5", "tta": False,
            "fit": "same target-source-excluded parent models for each fold; no target-label fit",
            "evaluation_role": "already exposed LODO development; not fresh confirmation or Private guarantee",
            "no_hpo": True, "primary": "paired 3-source macro mADM difference vs B119G",
            "report": ["each fold", "paired conditional query bootstrap uncertainty", "regressions", "actual full candidate saved artifacts"],
            "adoption": "mixed source signs or negligible gain do not establish best candidate; no automatic submission"},
        "resources": {"cpu_threads": 2, "sequential_fold_order": list(FOLDS),
            "minimum_available_ram_gib": 16, "new_gpu_execution": "root only", "new_training": False},
        "source_sha256": {"src/upar/streaming_weight_soup.py": sha256_file(Path("src/upar/streaming_weight_soup.py")),
                          "scripts/prepare_hplus6.py": sha256_file(Path(__file__))},
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    progress({"event": "PROTOCOL_FROZEN", "path": str(protocol_path), "sha256": sha256_file(protocol_path)})


def build(root: Path):
    protocol_path = root / "protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    manifest_path = Path(protocol["parent_manifest"]["path"])
    if sha256_file(manifest_path) != protocol["parent_manifest"]["sha256"]:
        raise ValueError("frozen manifest changed")
    for path, expected in protocol["source_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"implementation changed after protocol freeze: {path}")
    groups = json.loads(manifest_path.read_text())["groups"]
    for fold in FOLDS:
        group = groups[fold]
        output = Path(group["output"])
        if output.exists():
            receipt = json.loads((output / "weights_ready.json").read_text())
            expected_config = dict(group["config"], output_dir=str(output))
            expected_parents = [p["checkpoint"]["sha256"] for p in group["parents"]]
            if (receipt["status"] != "WEIGHTS_READY"
                    or receipt["protocol_ref"] != str(protocol_path)
                    or json.loads((output / "config.json").read_text()) != expected_config
                    or sha256_file(output / "config.json") != receipt["config_sha256"]
                    or [p["sha256"] for p in receipt["arithmetic"]["parents"]] != expected_parents
                    or sha256_file(output / "model.safetensors") != receipt["model"]["sha256"]):
                raise ValueError(f"existing output invalid: {output}")
            progress({"event": "REUSE_COMPLETE_WEIGHTS", "fold": fold})
            continue
        if shutil.disk_usage(root).free < 20 * 2**30:
            raise RuntimeError("less than 20 GiB disk headroom")
        for parent in group["parents"]:
            run = Path(parent["run"])
            for name, key in (("config.json", "config_sha256"), ("train_report.json", "train_report_sha256")):
                if sha256_file(run / name) != parent[key]:
                    raise ValueError(f"parent metadata changed: {run / name}")
        build_atomic_soup(parents=group["parents"], output=output, config=group["config"],
                          protocol_ref=str(protocol_path), progress=progress)


def main():
    global PROGRESS_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "build"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG = args.root / "cpu_progress_v1.jsonl"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    (freeze if args.command == "freeze" else build)(args.root)


if __name__ == "__main__":
    main()
