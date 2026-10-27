"""Read-only PE4 pooled features from the existing B119 source-held model.

Rebuild deleted fold soup weights without retraining, then capture the existing
base-head input in a forward pre-hook. No image/label optimization or graph fit.
Each invocation writes a new exclusive stage; failed stages are not retried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import h1_finetune as h
from .rank_sibling_pipeline import ROOT, exclusive_json, fingerprint, read_json, validate_run
from .streaming_weight_soup import build_atomic_soup, sha256_file

RUN = ROOT / "artifacts/runs/20261017T0448+0900_pe_gallery_features"
FOLDS = ("PA100k", "PETA", "Market1501")


class PooledCapture:
    """Capture the tensor already passed to head.base, without changing output."""
    def __init__(self, head):
        self.value = None
        self.calls = 0
        self.handle = head.register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 2:
            raise ValueError("unexpected base-head input")
        self.value = inputs[0].detach()
        self.calls += 1
        return None

    def close(self):
        self.handle.remove()


def verify_pins(pins):
    for item in pins:
        if fingerprint(ROOT / item["path"]) != item:
            raise ValueError(f"source evidence changed: {item['path']}")


def prepare(run):
    run.mkdir(exist_ok=True)
    if (run / "protocol.json").exists():
        raise FileExistsError("protocol already exists; inspect and do not reprepare")
    baseline = ROOT / "artifacts/runs/20261016T1710+0900_b119g/candidate_b119g.json"
    spec = read_json(baseline)
    rows = []
    for fold in FOLDS:
        old = ROOT / spec["lodo"]["members"][0]["runs"][fold][0]
        cfg = read_json(old / "config.json")
        if cfg.get("soup_weights") != [.25] * 4:
            raise ValueError("B119 PE equal soup changed")
        parents = [ROOT / name for name in cfg["soup_of"]]
        old_mask_cfg = read_json(parents[2] / "config.json")
        aligned = ROOT / f"artifacts/runs/20261016T1730+0900_rank_scope/maskv2_pw_{fold}"
        aligned_cfg = read_json(aligned / "config.json")
        for key in ("pw_aligned_of", "pw_ref", "fold", "include_val", "backbone", "head", "seed"):
            if aligned_cfg.get(key) != old_mask_cfg.get(key):
                raise ValueError(f"alternative prior-aligned copy differs: {fold}/{key}")
        with np.load(parents[2] / "val_logits.npz") as z0, np.load(aligned / "val_logits.npz") as z1:
            for key in z0.files:
                if not np.array_equal(z0[key], z1[key]):
                    raise ValueError(f"prior-aligned copy prediction mismatch: {fold}/{key}")
        parents[2] = aligned
        audit = [validate_run(parent, fold, deep_weights=True) for parent in parents]
        # Parent order and arithmetic are unchanged; only a deleted derived copy
        # is resolved to its already-existing, verified equivalent recipe copy.
        row = {"fold": fold, "old_config": fingerprint(old / "config.json"),
               "old_prediction": fingerprint(old / "val_logits.npz"),
               "parents": [{"checkpoint": entry["weights"], "weight": .25} for entry in audit],
               "fit_audit": audit, "config": cfg,
               "aligned_resolution": {"old": str((ROOT / cfg['soup_of'][2]).relative_to(ROOT)),
                                      "new": str(aligned.relative_to(ROOT)), "all_saved_arrays_equal": True}}
        rows.append(row)
    protocol = {"schema": 1, "baseline": fingerprint(baseline), "rows": rows,
                "sources": [fingerprint(Path(__file__)), fingerprint(ROOT / "src/upar/h1_finetune.py"),
                            fingerprint(ROOT / "src/upar/streaming_weight_soup.py"),
                            fingerprint(ROOT / "src/upar/rank_sibling_pipeline.py")],
                "feature": "Existing head.base input, pooled1024, original view; no extra backbone forward",
                "evaluation_rows": "only held-source official validation images, original cache order",
                "precision": "existing FP32 parameters / BF16 CUDA autocast; no TF32 override",
                "training": False, "hidden_data": False,
                "scope": "Local-only representation for fixed inference re-ranking research, not a new trained model or accepted submission",
                "limits": ["Batch16 inference may differ numerically from saved batch256 logits",
                           "Reconstructed float32 soup is the original arithmetic, but deleted original full-state SHA cannot be compared",
                           "Runtime quantized exported features require separate verification before deployment"]}
    exclusive_json(run / "protocol.json", protocol)
    return protocol


def build(run, fold):
    protocol = read_json(run / "protocol.json")
    verify_pins(protocol["sources"] + [protocol["baseline"]])
    row = next(r for r in protocol["rows"] if r["fold"] == fold)
    verify_pins([row["old_config"], row["old_prediction"]])
    return build_atomic_soup(parents=row["parents"], output=run / f"pe4_{fold}",
                             config=row["config"], protocol_ref=str((run / "protocol.json").relative_to(ROOT)),
                             minimum_available_gib=16, progress=lambda x: print(json.dumps(x), flush=True))


def infer(run, fold):
    from safetensors.torch import load_file

    protocol = read_json(run / "protocol.json")
    verify_pins(protocol["sources"] + [protocol["baseline"]])
    row = next(r for r in protocol["rows"] if r["fold"] == fold)
    model_dir = run / f"pe4_{fold}"
    ready = read_json(model_dir / "weights_ready.json")
    if sha256_file(model_dir / "model.safetensors") != ready["model"]["sha256"]:
        raise ValueError("reconstructed weights changed")
    cfg = read_json(model_dir / "config.json")
    if sha256_file(model_dir / "config.json") != ready["config_sha256"]:
        raise ValueError("reconstructed config changed")
    stage = run / f"features_{fold}"
    stage.mkdir()
    started = time.monotonic()
    exclusive_json(stage / "started.json", {"fold": fold, "training": False, "protocol": fingerprint(run / "protocol.json")})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info()
        host = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if free < 2.5*2**30 or host < 16*2**30:
            raise RuntimeError("GPU free2.5GiB and host16GiB required; guard is not a reservation")
        cache = ROOT / cfg["cache_dir"]
        meta = dict(np.load(cache / "meta.npz", allow_pickle=False))
        images = np.load(cache / cfg.get("cache_file", "images_u8_256x128.npy"), mmap_mode="r")
        selected = np.flatnonzero((meta["split"] == "val") & (meta["sources"] == fold))
        val = np.flatnonzero(meta["split"] == "val")
        val_positions = np.flatnonzero(meta["sources"][val] == fold)
        if not np.array_equal(val[val_positions], selected):
            raise ValueError("held gallery order mismatch")
        model, mean, std = h._build_model(cfg["backbone"], pretrained=False, head=cfg["head"])
        model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        model.cuda().eval()
        capture = PooledCapture(model.head.base)
        torch.cuda.reset_peak_memory_stats()
        features, logits = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(selected), 16):
                x = torch.from_numpy(np.ascontiguousarray(images[selected[start:start+16]])).cuda()
                x = h._gpu_augment(x, train=False, out_hw=h.BACKBONES[cfg["backbone"]]["img_hw"], mean=mean, std=std)
                value = model(x)
                if capture.value.shape != (len(x), 1024):
                    raise ValueError("PE pooled width or rows changed")
                features.append(capture.value.float().cpu().numpy())
                logits.append(value.float().cpu().numpy())
                if start % 1600 == 0:
                    print(json.dumps({"fold":fold,"rows":min(start+16,len(selected)),"total":len(selected),"seconds":time.monotonic()-started}), flush=True)
        capture.close()
        features, logits = np.concatenate(features), np.concatenate(logits)
        if not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise ValueError("nonfinite inference")
        with np.load(ROOT / row["old_prediction"]["path"]) as previous:
            diff = np.abs(previous["logits_original"][val_positions].astype(np.float32)-logits)
            pw = previous["pos_weight"]
        # Save exactly the unnormalized pooled feature. The later fixed graph
        # owns normalization, so representation and re-ranking stay separable.
        with (stage / "features.npz").open("xb") as stream:
            np.savez(stream, features=features, logits=logits, pos_weight=pw,
                     cache_rows=selected, val_positions=val_positions,
                     image_paths=meta["image_paths"][selected], attribute_names=meta["attribute_names"])
        verify_pins(protocol["sources"] + [protocol["baseline"],row["old_config"],row["old_prediction"]])
        if sha256_file(model_dir / "model.safetensors") != ready["model"]["sha256"]:
            raise ValueError("model changed during inference")
        report = {"status":"FEATURES_COMPLETE","fold":fold,"rows":len(selected),"shape":list(features.shape),
                  "feature_hook_calls":capture.calls,"outputs":fingerprint(stage / "features.npz"),
                  "model":fingerprint(model_dir / "model.safetensors"),"protocol":fingerprint(run / "protocol.json"),
                  "max_abs_logit_vs_saved_batch256":float(diff.max()),"mean_abs_logit_vs_saved_batch256":float(diff.mean()),
                  "runtime_seconds":time.monotonic()-started,"gpu_free_at_start":free,"gpu_total":total,
                  "peak_gpu_allocated":torch.cuda.max_memory_allocated(),"peak_gpu_reserved":torch.cuda.max_memory_reserved(),
                  "training":False,"extra_backbone_forward":False,"deployment_acceptance":False}
        exclusive_json(stage / "completed.json", report)
        return report
    except BaseException as exc:
        exclusive_json(stage / "failure.json", {"error":str(exc),"type":type(exc).__name__,"retry":False,"partial_preserved":True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "infer"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action != "prepare" and args.fold is None:
        parser.error("build/infer requires --fold")
    result = prepare(args.run) if args.action == "prepare" else {"build":build,"infer":infer}[args.action](args.run,args.fold)
    print(json.dumps({"action":args.action,"fold":args.fold,"result":result if args.action!='prepare' else 'FROZEN_NOT_EXECUTED'}, indent=2),flush=True)


if __name__ == "__main__":
    main()
