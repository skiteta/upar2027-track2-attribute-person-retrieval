from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"))

from upar.data import load_retrieval_split
from upar.h1_finetune import _build_model, _gpu_augment

FOLDS = ("PA100k", "PETA", "Market1501")


def load_pe_members(fold: str) -> tuple[dict[str, dict[str, torch.Tensor]], np.ndarray]:
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    p_attrq = ks / f"peL280_attrq_{fold}_ema0.999"
    p_rankq = ks / f"pLq_rankq_{fold}_ema0.999"
    p_maskv2 = ks / f"pLq_maskv2w_{fold}_ema0.999"
    p_pswap = ks / f"pLq_pswap03_{fold}_ema0.999"

    pw_ref = np.load(p_attrq / "val_logits.npz")["pos_weight"]
    pw_mask = np.load(p_maskv2 / "val_logits.npz")["pos_weight"]
    bias_delta = torch.from_numpy((np.log(pw_ref) - np.log(pw_mask)).astype(np.float32))

    sd_mask = load_file(str(p_maskv2 / "model.safetensors"))
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_mask:
            sd_mask[bk] = (sd_mask[bk].float() + bias_delta).to(sd_mask[bk].dtype).contiguous()

    mems = {
        "attrq": load_file(str(p_attrq / "model.safetensors")),
        "rankq": load_file(str(p_rankq / "model.safetensors")),
        "maskv2": sd_mask,
        "pswap": load_file(str(p_pswap / "model.safetensors")),
    }
    return mems, pw_ref


def load_hp_members(fold: str) -> tuple[dict[str, dict[str, torch.Tensor]], np.ndarray]:
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    hp6_dir = ROOT / f"artifacts/runs/20261017T0345+0900_hplus6/hp6_{fold}"

    p_q = ks / f"d3hpq_{fold}_ema0.999"
    p_qr = ks / f"d3hpqr_{fold}_ema0.999"
    p_qrq = ks / f"d3hpqrq_{fold}_ema0.999"
    p_qs = ks / f"d3hpqs_{fold}_ema0.999"
    p_qm = ks / f"d3hpqm_{fold}_ema0.999"

    pw_ref = np.load(p_q / "val_logits.npz")["pos_weight"]
    pw_qm = np.load(p_qm / "val_logits.npz")["pos_weight"]
    bias_delta_qm = torch.from_numpy((np.log(pw_ref) - np.log(pw_qm)).astype(np.float32))

    sd_q = load_file(str(p_q / "model.safetensors"))
    sd_qr = load_file(str(p_qr / "model.safetensors"))
    sd_qrq = load_file(str(p_qrq / "model.safetensors"))
    sd_qs = load_file(str(p_qs / "model.safetensors"))
    if (p_qm / "model.safetensors").exists():
        sd_qm = load_file(str(p_qm / "model.safetensors"))
    else:
        bkp = torch.load(HERE / "extracted_features_v1/extra_heads_backup_v1.pt", map_location="cpu")
        sd_qm = dict(bkp[p_qm.name])
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_qm:
            sd_qm[bk] = (sd_qm[bk].float() + bias_delta_qm).to(sd_qm[bk].dtype).contiguous()

    # Recover exact 224x112 pair average head.* from hp6 = (q + qr + qrq + qs + 224q + 224qr) / 6
    sd_hp6 = load_file(str(hp6_dir / "model.safetensors"))
    sd_224pair = {}
    for k in sd_q:
        if k.startswith("head.") and sd_q[k].is_floating_point():
            sum4 = sd_q[k].float() + sd_qr[k].float() + sd_qrq[k].float() + sd_qs[k].float()
            sd_224pair[k] = 3.0 * sd_hp6[k].float() - 0.5 * sum4
        else:
            sd_224pair[k] = sd_hp6[k]
    del sd_hp6

    mems = {
        "q": sd_q,
        "qr": sd_qr,
        "qrq": sd_qrq,
        "qs": sd_qs,
        "qm": sd_qm,
        "r224": sd_224pair,
    }
    return mems, pw_ref


def make_soup_state_dict(mems: dict[str, dict[str, torch.Tensor]], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    sd_out: dict[str, torch.Tensor] = {}
    for m, w in weights.items():
        if w == 0.0:
            continue
        sd = mems[m]
        for k, ten in sd.items():
            if ten.is_floating_point():
                val = float(w) * ten.float()
                sd_out[k] = sd_out[k] + val if k in sd_out else val
            elif k not in sd_out:
                sd_out[k] = ten
    return sd_out


def make_custom_head(
    member_sds: dict[str, dict[str, torch.Tensor]],
    head_weights: dict[str, float],
    tau_qk: float = 0.0,
    tau_out: float = 0.0,
    tau_base: float = 0.0,
    q_scale: float = 1.0,
    out_scale: float = 1.0,
    base_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    out = {}
    for k, tau, sc in [
        ("head.q", tau_qk, q_scale),
        ("head.k.weight", tau_qk, 1.0),
        ("head.out", tau_out, out_scale),
        ("head.base.weight", tau_base, base_scale),
        ("head.base.bias", 0.0, 1.0),
    ]:
        base_t = sum(float(w) * member_sds[m][k].float() for m, w in head_weights.items() if w > 0)
        if tau != 0.0 and base_t.ndim == 2:
            soup_norm = base_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mem_norm = sum(float(w) * member_sds[m][k].float().norm(dim=-1, keepdim=True) for m, w in head_weights.items() if w > 0)
            ratio = (mem_norm / soup_norm) ** float(tau)
            base_t = base_t * ratio
        if sc != 1.0:
            base_t = base_t * float(sc)
        out[k] = base_t
    return out


def eval_head_on_tokens(pooled_cpu: torch.Tensor, patches_cpu: torch.Tensor, head_params: dict[str, torch.Tensor]) -> np.ndarray:
    w_base = head_params["head.base.weight"].cuda().to(torch.bfloat16)
    b_base = head_params["head.base.bias"].cuda().to(torch.bfloat16)
    q_p = head_params["head.q"].cuda().to(torch.bfloat16)
    k_w = head_params["head.k.weight"].cuda().to(torch.bfloat16)
    out_p = head_params["head.out"].cuda().to(torch.bfloat16)

    lo = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(pooled_cpu), 1024):
            pooled = pooled_cpu[s:s + 1024].cuda()
            patches = patches_cpu[s:s + 1024].cuda()
            logits_base = torch.nn.functional.linear(pooled, w_base, b_base)
            k_proj = torch.nn.functional.linear(patches, k_w)
            att = torch.softmax(torch.einsum("qd,bpd->bqp", q_p, k_proj) / 16.0, dim=-1)
            z = torch.einsum("bqp,bpc->bqc", att, patches)
            logits = logits_base + (z * out_p).sum(-1)
            lo.append(logits.float().cpu().numpy())
    return np.concatenate(lo, axis=0)


def main() -> None:
    t0 = time.monotonic()
    feat_dir = HERE / "extracted_features_v1"
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    gt = split.gallery_attributes.astype(np.float64)

    cache = ROOT / "artifacts/runs/20261007T020500+0900_gold_cycle/cache_256x128"
    images = np.load(cache / "images_u8_256x128.npy", mmap_mode="r")
    meta = dict(np.load(cache / "meta.npz"))
    val = np.flatnonzero(meta["split"] == "val")
    val_imgs = np.ascontiguousarray(images[val])
    print(f"Loaded {len(val_imgs)} val images in {time.monotonic() - t0:.1f}s", flush=True)

    pe_w_uni = {"attrq": 0.25, "rankq": 0.25, "maskv2": 0.25, "pswap": 0.25}
    pe_w_2030 = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
    pe_w_2025 = {"attrq": 0.20, "rankq": 0.25, "maskv2": 0.25, "pswap": 0.30}

    pe_configs = {
        "pe_base": (pe_w_uni, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        "pe_b136c": (pe_w_2030, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out00_base00": (pe_w_2030, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out05_base00": (pe_w_2030, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out05_base05": (pe_w_2030, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out075_base025": (pe_w_2030, 1.0, 0.75, 0.25, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out075_base05": (pe_w_2030, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
        "pe_2030_qk100_out100_base05": (pe_w_2030, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0),
        "pe_2030_qk115_out075_base05": (pe_w_2030, 1.15, 0.75, 0.5, 1.0, 1.0, 1.0),
        "pe_2030_qk115_out100_base05": (pe_w_2030, 1.15, 1.0, 0.5, 1.0, 1.0, 1.0),
        "pe_2030_qk100_t95": (pe_w_2030, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95),
        "pe_2030_qk100_t92": (pe_w_2030, 1.0, 1.0, 1.0, 1.0, 0.92, 0.92),
        "pe_2025_qk100_out075_base05": (pe_w_2025, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
        "pe_2025_qk100_out100_base100": (pe_w_2025, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    }
    pe_oof = {k: np.empty((split.gallery_size, 40), dtype=np.float32) for k in pe_configs}

    # Step 1: PE-Core-L/14 (extract pe_full_val_{fold}.npz + evaluate all PE head configs on held-out fold)
    model_pe, mean_pe, std_pe = _build_model("pe_core_l14_280", pretrained=False, head="attrq")
    model_pe = model_pe.cuda().eval()

    for fold in FOLDS:
        tf = time.monotonic()
        mems_pe, pw_pe = load_pe_members(fold)
        sd_pe_uni = make_soup_state_dict(mems_pe, pe_w_uni)
        model_pe.load_state_dict(sd_pe_uni, strict=True)
        model_pe.eval()

        pe_npz = feat_dir / f"pe_full_val_{fold}.npz"
        need_full = not pe_npz.exists()
        hp_idx = np.flatnonzero(split.sources == fold)

        if need_full:
            all_pooled, all_lo = [], []
            hp_patches = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, len(val_imgs), 256):
                    x = torch.from_numpy(val_imgs[s:s + 256]).cuda()
                    inp = _gpu_augment(x, train=False, out_hw=(280, 140), mean=mean_pe, std=std_pe)
                    feats = model_pe.backbone.forward_features(inp)
                    pooled = model_pe.backbone.forward_head(feats, pre_logits=True)
                    logits = model_pe.head.base(pooled)
                    patches = feats[:, 1:]
                    att = torch.softmax(torch.einsum("qd,bpd->bqp", model_pe.head.q, model_pe.head.k(patches)) / 16.0, dim=-1)
                    z = torch.einsum("bqp,bpc->bqc", att, patches)
                    logits = logits + (z * model_pe.head.out).sum(-1)
                    all_pooled.append(pooled.float().cpu().numpy())
                    all_lo.append(logits.float().cpu().numpy())
            full_pooled = np.concatenate(all_pooled, axis=0)
            full_lo = np.concatenate(all_lo, axis=0)
            np.savez(pe_npz, features=full_pooled, logits_original=full_lo, pos_weight=pw_pe)
            print(f"[{fold}] saved {pe_npz.name} in {time.monotonic() - tf:.1f}s", flush=True)

        # Now extract held-out fold pooled + patches in bf16 for fast head evaluation
        imgs_hp = val_imgs[hp_idx]
        pooled_list, patches_list = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(imgs_hp), 256):
                x = torch.from_numpy(imgs_hp[s:s + 256]).cuda()
                inp = _gpu_augment(x, train=False, out_hw=(280, 140), mean=mean_pe, std=std_pe)
                feats = model_pe.backbone.forward_features(inp)
                pooled = model_pe.backbone.forward_head(feats, pre_logits=True)
                patches = feats[:, 1:]
                pooled_list.append(pooled.to(torch.bfloat16).cpu())
                patches_list.append(patches.to(torch.bfloat16).cpu())
        pooled_cpu = torch.cat(pooled_list, dim=0)
        patches_cpu = torch.cat(patches_list, dim=0)

        corr = np.log(pw_pe).astype(np.float32)
        for cname, (w_dict, tqk, tout, tbase, qsc, osc, bsc) in pe_configs.items():
            hp_params = make_custom_head(mems_pe, w_dict, tqk, tout, tbase, qsc, osc, bsc)
            lo = eval_head_on_tokens(pooled_cpu, patches_cpu, hp_params)
            pe_oof[cname][hp_idx] = (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)

        del pooled_cpu, patches_cpu, mems_pe, sd_pe_uni
        torch.cuda.empty_cache()

    del model_pe
    torch.cuda.empty_cache()

    print("\n--- PE-Core-L/14 OOF Calibration (33,407 val images) ---")
    for cname, p0 in pe_oof.items():
        q = np.clip(p0.astype(np.float64), 1e-5, 1.0 - 1e-5)
        ll = -np.mean(gt * np.log(q) + (1.0 - gt) * np.log1p(-q))
        br = np.mean((q - gt) ** 2)
        print(f"  {cname:32s}: LogLoss={ll:.6f}, Brier={br:.6f}", flush=True)

    # Step 2: DINOv3-H+/16 @ 192x96 (evaluate Decoupled Spherical + 5-recipe + 7-recipe multi-res heads)
    hp_w_4 = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20}
    hp_w_5 = {"q": 0.16, "qr": 0.26, "qrq": 0.26, "qs": 0.16, "qm": 0.16}
    hp_w_6_r224 = {"q": 0.16, "qr": 0.24, "qrq": 0.24, "qs": 0.16, "r224": 0.20}
    hp_w_7_all = {"q": 0.14, "qr": 0.22, "qrq": 0.22, "qs": 0.14, "qm": 0.12, "r224": 0.16}

    hp_configs = {
        "hp_base": (hp_w_4, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        "hp_b136c": (hp_w_4, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
        "hp_n100": (hp_w_4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "hp_4_qk100_out00_base00": (hp_w_4, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        "hp_4_qk100_out05_base025": (hp_w_4, 1.0, 0.5, 0.25, 1.0, 1.0, 1.0),
        "hp_4_qk100_out05_base05": (hp_w_4, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0),
        "hp_4_qk100_out075_base05": (hp_w_4, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
        "hp_4_qk100_out100_base05": (hp_w_4, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0),
        "hp_4_qk115_out075_base05": (hp_w_4, 1.15, 0.75, 0.5, 1.0, 1.0, 1.0),
        "hp_4_qk100_t96": (hp_w_4, 1.0, 1.0, 1.0, 1.0, 0.96, 0.96),
        "hp_5_qk100_out100_base100": (hp_w_5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "hp_5_qk100_out075_base05": (hp_w_5, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
        "hp_6_r224_qk100_out100_base100": (hp_w_6_r224, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "hp_6_r224_qk100_out075_base05": (hp_w_6_r224, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
        "hp_7_all_qk100_out100_base100": (hp_w_7_all, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "hp_7_all_qk100_out075_base05": (hp_w_7_all, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0),
    }
    hp_oof = {k: np.empty((split.gallery_size, 40), dtype=np.float32) for k in hp_configs}

    model_hp, mean_hp, std_hp = _build_model("dinov3_hp16_192", pretrained=False, head="attrq")
    model_hp = model_hp.cuda().eval()

    for fold in FOLDS:
        tf = time.monotonic()
        mems_hp, pw_hp = load_hp_members(fold)
        sd_hp4 = make_soup_state_dict({k: mems_hp[k] for k in hp_w_4}, hp_w_4)
        model_hp.load_state_dict(sd_hp4, strict=True)
        model_hp.eval()

        hp_idx = np.flatnonzero(split.sources == fold)
        imgs_hp = val_imgs[hp_idx]
        pooled_list, patches_list = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(imgs_hp), 256):
                x = torch.from_numpy(imgs_hp[s:s + 256]).cuda()
                inp = _gpu_augment(x, train=False, out_hw=(192, 96), mean=mean_hp, std=std_hp)
                feats = model_hp.backbone.forward_features(inp)
                patches = feats[:, 5:]
                pooled = patches.mean(dim=1)
                pooled_list.append(pooled.to(torch.bfloat16).cpu())
                patches_list.append(patches.to(torch.bfloat16).cpu())
        pooled_cpu = torch.cat(pooled_list, dim=0)
        patches_cpu = torch.cat(patches_list, dim=0)

        corr = np.log(pw_hp).astype(np.float32)
        for cname, (w_dict, tqk, tout, tbase, qsc, osc, bsc) in hp_configs.items():
            hp_params = make_custom_head(mems_hp, w_dict, tqk, tout, tbase, qsc, osc, bsc)
            lo = eval_head_on_tokens(pooled_cpu, patches_cpu, hp_params)
            hp_oof[cname][hp_idx] = (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)

        del pooled_cpu, patches_cpu, mems_hp, sd_hp4
        torch.cuda.empty_cache()
        print(f"[{fold}] evaluated all HP head configs in {time.monotonic() - tf:.1f}s", flush=True)

    del model_hp
    torch.cuda.empty_cache()

    print("\n--- DINOv3-H+/16 OOF Calibration (33,407 val images) ---")
    for cname, p1 in hp_oof.items():
        q = np.clip(p1.astype(np.float64), 1e-5, 1.0 - 1e-5)
        ll = -np.mean(gt * np.log(q) + (1.0 - gt) * np.log1p(-q))
        br = np.mean((q - gt) ** 2)
        print(f"  {cname:32s}: LogLoss={ll:.6f}, Brier={br:.6f}", flush=True)

    out_npz = feat_dir / "spherical_head_oof_v2.npz"
    np.savez_compressed(out_npz, **pe_oof, **hp_oof)
    print(f"\nSaved {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB) in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

