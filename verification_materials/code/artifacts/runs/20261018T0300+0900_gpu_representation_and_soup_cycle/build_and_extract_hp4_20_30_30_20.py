from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from safetensors.torch import load_file, save_file
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from upar.h1_finetune import _build_model, _gpu_augment

FOLDS = ("PA100k", "PETA", "Market1501")
WEIGHTS = {"d3hpq": 0.20, "d3hpqr": 0.30, "d3hpqrq": 0.30, "d3hpqs": 0.20}


def _forward_logits_and_feat(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feats = model.backbone.forward_features(x)
    feat_vec = feats[:, 5:].mean(dim=1)
    pooled = model.backbone.forward_head(feats, pre_logits=True)
    logits = model.head.base(pooled)
    patches = feats[:, model.npt:]
    att = torch.softmax(torch.einsum("qd,bpd->bqp", model.head.q, model.head.k(patches)) / 16.0, dim=-1)
    z = torch.einsum("bqp,bpc->bqc", att, patches)
    logits = logits + (z * model.head.out).sum(-1)
    return logits, feat_vec


def build_weighted_soup(dirs_and_weights: list[tuple[Path, float]]) -> dict[str, torch.Tensor]:
    sds = [(load_file(str(d / "model.safetensors")), float(w)) for d, w in dirs_and_weights]
    sd0 = sds[0][0]
    out = {}
    for k in sd0:
        if sd0[k].is_floating_point():
            acc = torch.zeros_like(sd0[k], dtype=torch.float32)
            for sd, w in sds:
                acc += w * sd[k].float()
            out[k] = acc.to(sd0[k].dtype)
        else:
            out[k] = sd0[k]
    return out


def main() -> None:
    t0 = time.monotonic()
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    rs = ROOT / "artifacts/runs/20261013T1500+0900_recipe_soup"
    out_all_dir = ROOT / "artifacts/runs/20261017T0345+0900_hplus6/hp4_20_30_30_20_all"
    out_all_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build production all_trainval soup checkpoint
    all_safetensors = out_all_dir / "model.safetensors"
    if not all_safetensors.exists():
        t_b = time.monotonic()
        dirs_all = [
            (ks / f"{name}_all_trainval_seed20261007_ema0.999", w)
            for name, w in WEIGHTS.items()
        ]
        sd_all = build_weighted_soup(dirs_all)
        save_file(sd_all, str(all_safetensors))
        del sd_all
        ref_dir = ROOT / "artifacts/runs/20261016T1710+0900_b119g/soup_hp_q_qr_qrq_s_all_s7"
        shutil.copy2(ref_dir / "val_logits.npz", out_all_dir / "val_logits.npz")
        cfg_all = json.loads((ref_dir / "config.json").read_text())
        cfg_all["output_dir"] = str(out_all_dir.relative_to(ROOT))
        cfg_all["soup_weights"] = [0.20, 0.30, 0.30, 0.20]
        (out_all_dir / "config.json").write_text(json.dumps(cfg_all, indent=2))
        print(f"Built {all_safetensors} in {time.monotonic() - t_b:.1f}s", flush=True)

    # 2. Preload all 33,407 val images once
    cache = ROOT / "artifacts/runs/20261007T020500+0900_gold_cycle/cache_256x128"
    images = np.load(cache / "images_u8_256x128.npy", mmap_mode="r")
    meta = dict(np.load(cache / "meta.npz"))
    val = np.flatnonzero(meta["split"] == "val")
    t_ld = time.monotonic()
    val_imgs = np.ascontiguousarray(images[val])
    print(f"Preloaded {len(val_imgs)} val images in {time.monotonic() - t_ld:.1f}s", flush=True)

    model, mean, std = _build_model("dinov3_hp16_192", pretrained=False, head="attrq")
    model = model.cuda().eval()
    feat_dir = HERE / "extracted_features_v1"

    for fold in FOLDS:
        out_npz = feat_dir / f"hp4_2030_192_full_val_{fold}.npz"
        if out_npz.exists():
            print(f"[{fold}] {out_npz.name} already exists, skipping.", flush=True)
            continue
        tf = time.monotonic()
        dirs_fold = [
            (ks / f"{name}_{fold}_ema0.999", w)
            for name, w in WEIGHTS.items()
        ]
        sd_fold = build_weighted_soup(dirs_fold)
        model.load_state_dict(sd_fold)
        del sd_fold
        model.eval()

        lo, feats_list = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(val_imgs), 256):
                x = torch.from_numpy(val_imgs[s:s + 256]).cuda()
                inp = _gpu_augment(x, train=False, out_hw=(192, 96), mean=mean, std=std)
                logits_o, feat_o = _forward_logits_and_feat(model, inp)
                lo.append(logits_o.float().cpu().numpy())
                feats_list.append(feat_o.float().cpu().numpy())
        lo_arr = np.concatenate(lo, axis=0)
        f_arr = np.concatenate(feats_list, axis=0)
        pw = np.load(rs / f"hp_q_qr_qrq_s_{fold}/val_logits.npz")["pos_weight"]
        np.savez(
            out_npz,
            logits_original=lo_arr,
            logits_flip=lo_arr,
            pos_weight=pw,
            features=f_arr,
        )
        print(f"[{fold}] extracted {out_npz.name} in {time.monotonic() - tf:.1f}s", flush=True)

    print(f"All done in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

