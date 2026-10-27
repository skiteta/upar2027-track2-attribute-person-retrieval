from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"))

from extract_pe_and_spherical_suite import (
    FOLDS,
    eval_head_on_tokens,
    load_hp_members,
    load_pe_members,
    make_soup_state_dict,
)
import test_tri_graph_combined_knobs as tk
from test_hp6_224_stack import build_variant_contexts, pool_custom
from test_hp6_224_final_comb import build_anat_moe
from calibrate_public_mevid_synth import (
    build_embedded_synth_and_mevid_contexts,
    eval_sliced_ctx,
    eval_synth_ctxs,
)
from test_eva_head_and_exact_b137c import compute_eva_probs
from upar.data import load_retrieval_split
from upar.h1_finetune import _build_model, _gpu_augment
from upar.postproc import apply_post
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC


def dict_to_device(d: dict, device: str) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = dict_to_device(v, device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def make_colselective_head(
    member_sds: dict[str, dict[str, torch.Tensor]],
    head_weights: dict[str, float],
    col_overrides: dict[int, dict[str, float]] | None = None,
    tau_qk: float = 0.0,
    tau_out: float = 0.0,
    tau_base: float = 0.0,
    q_scale: float = 1.0,
    out_scale: float = 1.0,
    base_scale: float = 1.0,
    k_weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    out = {}
    kw_use = k_weights if k_weights is not None else head_weights
    for k, tau, sc in [
        ("head.q", tau_qk, q_scale),
        ("head.k.weight", tau_qk, 1.0),
        ("head.out", tau_out, out_scale),
        ("head.base.weight", tau_base, base_scale),
        ("head.base.bias", 0.0, 1.0),
    ]:
        w_default = kw_use if k == "head.k.weight" else head_weights
        base_t = sum(float(w) * member_sds[m][k].float() for m, w in w_default.items() if w > 0).clone()
        if tau != 0.0 and base_t.ndim == 2:
            soup_norm = base_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mem_norm = sum(float(w) * member_sds[m][k].float().norm(dim=-1, keepdim=True) for m, w in w_default.items() if w > 0)
            base_t = base_t * ((mem_norm / soup_norm) ** float(tau))

        if col_overrides and k != "head.k.weight":
            for j, w_col in col_overrides.items():
                row_t = sum(float(w) * member_sds[m][k][j].float() for m, w in w_col.items() if w > 0)
                if tau != 0.0 and base_t.ndim == 2:
                    s_norm = row_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    m_norm = sum(float(w) * member_sds[m][k][j].float().norm(dim=-1, keepdim=True) for m, w in w_col.items() if w > 0)
                    row_t = row_t * ((m_norm / s_norm) ** float(tau))
                base_t[j] = row_t

        if sc != 1.0:
            base_t = base_t * float(sc)
        out[k] = base_t
    return out


def main() -> None:
    t0 = time.monotonic()
    feat_dir = HERE / "extracted_features_v1"
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    gt = split.gallery_attributes.astype(np.float64)

    out_npz4 = feat_dir / "spherical_head_oof_v4.npz"
    out_npz5 = feat_dir / "spherical_head_oof_v5.npz"
    z4 = dict(np.load(out_npz4)) if out_npz4.exists() else {}
    if out_npz5.exists():
        print(f"Loading existing {out_npz5}...", flush=True)
        z4.update(dict(np.load(out_npz5)))
    else:
        cache = ROOT / "artifacts/runs/20261007T020500+0900_gold_cycle/cache_256x128"
        images = np.load(cache / "images_u8_256x128.npy", mmap_mode="r")
        meta = dict(np.load(cache / "meta.npz"))
        val = np.flatnonzero(meta["split"] == "val")
        val_imgs = np.ascontiguousarray(images[val])

        MASKED_15 = [4, 5, 6, 10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
        UNMASKED_24 = [c for c in range(40) if c not in MASKED_15 and c != 19]

        pe_w_uni = {"attrq": 0.25, "rankq": 0.25, "maskv2": 0.25, "pswap": 0.25}
        pe_w_2030 = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
        pe_w_m20 = {"attrq": 0.225, "rankq": 0.225, "maskv2": 0.20, "pswap": 0.35}
        pe_w_m15 = {"attrq": 0.24, "rankq": 0.24, "maskv2": 0.15, "pswap": 0.37}

        pe_ov_unm24_m20 = {c: pe_w_m20 for c in UNMASKED_24}
        pe_ov_unm24_m15 = {c: pe_w_m15 for c in UNMASKED_24}

        pe_configs = {
            "pe_unmask24_m20_qk120_b060": (pe_w_2030, pe_ov_unm24_m20, 1.20, 1.0, 0.60, 1.0, 1.0, 1.0),
            "pe_unmask24_m15_qk120_b060": (pe_w_2030, pe_ov_unm24_m15, 1.20, 1.0, 0.60, 1.0, 1.0, 1.0),
        }
        pe_oof = {k: np.empty((split.gallery_size, 40), dtype=np.float32) for k in pe_configs}

        model_pe, mean_pe, std_pe = _build_model("pe_core_l14_280", pretrained=False, head="attrq")
        model_pe = model_pe.cuda().eval()

        for fold in FOLDS:
            tf = time.monotonic()
            mems_pe, pw_pe = load_pe_members(fold)
            sd_pe_uni = make_soup_state_dict(mems_pe, pe_w_uni)
            model_pe.load_state_dict(sd_pe_uni, strict=True)
            model_pe.eval()

            hp_idx = np.flatnonzero(split.sources == fold)
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
            for cname, (w_dict, ov_dict, tqk, tout, tbase, qsc, osc, bsc) in pe_configs.items():
                hp_params = make_colselective_head(mems_pe, w_dict, ov_dict, tqk, tout, tbase, qsc, osc, bsc)
                lo = eval_head_on_tokens(pooled_cpu, patches_cpu, hp_params)
                pe_oof[cname][hp_idx] = (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)

            del pooled_cpu, patches_cpu, mems_pe, sd_pe_uni
            torch.cuda.empty_cache()
            print(f"[PE {fold}] extracted in {time.monotonic() - tf:.1f}s", flush=True)

        del model_pe
        torch.cuda.empty_cache()

        hp_w_4 = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20}
        hp_w_6_10 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "qm": 0.0, "r224": 0.10}
        hp_w_7_m10_r10 = {"q": 0.16, "qr": 0.24, "qrq": 0.24, "qs": 0.16, "qm": 0.10, "r224": 0.10}
        hp_w_7_m12_r10 = {"q": 0.156, "qr": 0.234, "qrq": 0.234, "qs": 0.156, "qm": 0.12, "r224": 0.10}
        hp_w_7_m15_r10 = {"q": 0.15, "qr": 0.225, "qrq": 0.225, "qs": 0.15, "qm": 0.15, "r224": 0.10}
        hp_w_5_m15_r00 = {"q": 0.17, "qr": 0.255, "qrq": 0.255, "qs": 0.17, "qm": 0.15, "r224": 0.0}

        hp_ov_m15_m10_r10 = {c: hp_w_7_m10_r10 for c in MASKED_15}
        hp_ov_m15_m12_r10 = {c: hp_w_7_m12_r10 for c in MASKED_15}
        hp_ov_m15_m15_r10 = {c: hp_w_7_m15_r10 for c in MASKED_15}
        hp_ov_m15_m15_r00 = {c: hp_w_5_m15_r00 for c in MASKED_15}

        hp_configs = {
            "hp_mask15_m10_r10_o103": (hp_w_6_10, hp_ov_m15_m10_r10, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
            "hp_mask15_m12_r10_o103": (hp_w_6_10, hp_ov_m15_m12_r10, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
            "hp_mask15_m15_r10_o103": (hp_w_6_10, hp_ov_m15_m15_r10, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
            "hp_mask15_m15_r00_o103": (hp_w_6_10, hp_ov_m15_m15_r00, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
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
            for cname, (w_dict, ov_dict, tqk, tout, tbase, qsc, osc, bsc) in hp_configs.items():
                hp_params = make_colselective_head(mems_hp, w_dict, ov_dict, tqk, tout, tbase, qsc, osc, bsc)
                lo = eval_head_on_tokens(pooled_cpu, patches_cpu, hp_params)
                hp_oof[cname][hp_idx] = (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)

            del pooled_cpu, patches_cpu, mems_hp, sd_hp4
            torch.cuda.empty_cache()
            print(f"[HP {fold}] extracted in {time.monotonic() - tf:.1f}s", flush=True)

        del model_hp, val_imgs
        torch.cuda.empty_cache()

        z5 = {**pe_oof, **hp_oof}
        np.savez_compressed(out_npz5, **z5)
        z4.update(z5)
        print(f"Saved {out_npz5} in {time.monotonic() - t0:.1f}s", flush=True)

    print("\n--- Column-Selective Head Calibration (33,407 val images) ---")
    for cname, p in z4.items():
        q = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
        ll_all = -(gt * np.log(q) + (1.0 - gt) * np.log(1.0 - q)).mean(axis=0)
        br = np.mean((q - gt) ** 2)
        print(f"  {cname:30s}: LogLoss={ll_all.mean():.6f}, Brier={br:.6f}, Col19_LL={ll_all[19]:.6f}", flush=True)

    # Now evaluate retrieval across all 3 surfaces (OOF-33k, Mixed-Gallery, SimPub)!
    plan = json.loads((ROOT / "artifacts/runs/20261017T1618+0900_mixed_gallery_context/protocol.json").read_text())
    spec_obj = read_json(SPEC)
    train_proto = read_json(ROOT / "artifacts/runs/20261017T1105+0900_cycle_continuation/frontier/sourcefit_edge_v2/protocol.json")
    post_cfg = spec_obj["lodo"]["post"]

    qi_oof = np.unique(split.gallery_query_ids)
    queries_oof = split.queries[qi_oof].astype(np.float32)
    oof_ctx = tk.precompute_eval_ctx(split, qi_oof)
    synth_ctxs, mevid_ctx = build_embedded_synth_and_mevid_contexts(split, qi_oof, n_seeds=5)

    folds_base, oof_base = build_variant_contexts("base_192", None, False, feat_dir, split, plan, spec_obj, train_proto)
    oof_base = dict_to_device(oof_base, "cpu")
    for fold in tk.FOLDS:
        folds_base[fold]["mixed"] = dict_to_device(folds_base[fold]["mixed"], "cpu")
    torch.cuda.empty_cache()

    w_oof = np.mean([folds_base[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)
    w_anat10 = build_anat_moe([0.425, 0.340, 0.235], [0.420, 0.345, 0.235], [0.370, 0.395, 0.235], [0.355, 0.410, 0.235])

    MASKED_15 = [4, 5, 6, 10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
    UNMASKED_24 = [c for c in range(40) if c not in MASKED_15 and c != 19]

    # Preload per-fold val_logits for fold-consistent Mixed-Gallery cp alignment
    pe_fold_logits = {}
    hp_fold_logits = {}
    for fold in tk.FOLDS:
        pe_runs = {
            "attrq": f"peL280_attrq_{fold}_ema0.999",
            "rankq": f"pLq_rankq_{fold}_ema0.999",
            "maskv2": f"pLq_maskv2w_{fold}_ema0.999",
            "pswap": f"pLq_pswap03_{fold}_ema0.999",
        }
        pe_fold_logits[fold] = {}
        for k, rdir in pe_runs.items():
            z = np.load(ks / rdir / "val_logits.npz")
            pw = z["pos_weight"].astype(np.float64)
            pe_fold_logits[fold][k] = 0.5 * (z["logits_original"] + z["logits_flip"]) - np.log(pw)

        hp_runs = {
            "q": ks / f"d3hpq_{fold}_ema0.999/val_logits.npz",
            "qr": ks / f"d3hpqr_{fold}_ema0.999/val_logits.npz",
            "qrq": ks / f"d3hpqrq_{fold}_ema0.999/val_logits.npz",
            "qs": ks / f"d3hpqs_{fold}_ema0.999/val_logits.npz",
            "qm": ks / f"d3hpqm_{fold}_ema0.999/val_logits.npz",
            "r224": feat_dir / f"hp6_224_full_val_{fold}.npz",
        }
        hp_fold_logits[fold] = {}
        for k, rpath in hp_runs.items():
            z = np.load(rpath)
            pw = z["pos_weight"].astype(np.float64)
            hp_fold_logits[fold][k] = 0.5 * (z["logits_original"] + z["logits_flip"]) - np.log(pw)

    def make_fold_logit_probs(fold_logits: dict, w_default: dict, col_ov: dict | None = None, scale: float = 1.0) -> dict[str, np.ndarray]:
        out = {}
        for fold in tk.FOLDS:
            fl = fold_logits[fold]
            lo = sum(float(w) * fl[m] for m, w in w_default.items() if w > 0).copy()
            if col_ov:
                for c, wc in col_ov.items():
                    lo[:, c] = sum(float(w) * fl[m][:, c] for m, w in wc.items() if w > 0)
            lo = lo * float(scale)
            out[fold] = (1.0 / (1.0 + np.exp(-lo))).astype(np.float32)
        return out

    u_pe_oof = np.empty((split.gallery_size, 1024), dtype=np.float64)
    u_eva_oof = np.empty((split.gallery_size, 1024), dtype=np.float64)
    uhp_2030 = np.empty((split.gallery_size, 1280), dtype=np.float64)
    uhp_fold_2030 = {}
    for fold in tk.FOLDS:
        bf = folds_base[fold]
        hp = bf["hp"]
        u_pe_oof[hp] = bf["u_pe_m"][hp]
        u_eva_oof[hp] = bf["u_eva_m"][hp]
        z_hp2030 = np.load(feat_dir / f"hp4_2030_192_full_val_{fold}.npz")
        f_hp_all = z_hp2030["features"].astype(np.float64)
        u_hp_all = f_hp_all / np.maximum(np.linalg.norm(f_hp_all, axis=1, keepdims=True), 1e-12)
        uhp_fold_2030[fold] = u_hp_all
        uhp_2030[hp] = u_hp_all[hp]

    md_oof_uni = oof_base
    md_oof_2030 = dict_to_device(
        tk.precompute_fold_mode(
            u_pe_oof, u_eva_oof, uhp_2030, oof_base["p0"], oof_base["raw_probs"], oof_base["flip_probs"], post_cfg, oof_base["m_std"]
        ),
        "cpu",
    )
    torch.cuda.empty_cache()
    md_mix_2030 = {}
    for fold in tk.FOLDS:
        bf = folds_base[fold]
        bm = bf["mixed"]
        md_mix_2030[fold] = dict_to_device(
            tk.precompute_fold_mode(
                bf["u_pe_m"], bf["u_eva_m"], uhp_fold_2030[fold], bm["p0"], bf["raw_base"], bf["flip_base"], post_cfg, bm["m_std"]
            ),
            "cpu",
        )
        torch.cuda.empty_cache()

    b5_tw07 = [
        ("pe", 0.275, 1.0, 0.0),
        ("eva", 0.23, 0.0, 1.0),
        ("j25_k28", 0.235, 0.75, 0.25),
        ("j50_k28", 0.19, 0.50, 0.50),
        ("tri25_k28", 0.07, 0.5625, 0.1875),
    ]
    aw_8spec = {
        "pe": 0.10, "eva": 0.10, "j25": 0.18, "j50": 0.18,
        "tri25": 0.16, "tri40": 0.14, "pe_hp": 0.07, "eva_hp": 0.07,
    }
    cfg_b131c = {
        "branches": b5_tw07, "anchor_spec_weights": aw_8spec, "ham_tau": 0.17, "dist_kappa": 0.20,
        "sim_diff_alpha": 0.34, "sim_diff_alpha2": 0.04, "exact_seed_blend": 0.72, "ex_vis_alpha": 0.120,
        "anchor_alpha_q": 0.126, "anchor_temp_q": 0.052, "head_beta": 0.044, "head_tau": 0.225,
        "anchor_top_m": 11, "tail_mul_kap": 0.04, "head_dq_blend": 0.15, "seed_disagree_kap": 0.030,
        "seed_kap_weight": False, "exact_branch_blend": 0.22, "ex_diff_alpha": 0.07, "ex_diff_head_blend": 0.50,
        "ex_kappa": 0.24, "head_local_lam": 0.12, "head_local_k": 12, "body_alpha": 0.20, "body_k0": 10,
        "body_tau": 0.40, "body_pow": 1.0, "body_w_uni": 1.0, "body_head_keep": 1.0, "body_err_gamma": 0.85, "body_nll_lam": 0.05,
    }
    cfg_b06 = {
        **cfg_b131c,
        "nll_lam": 0.275, "nll_lam_neg": 0.215, "err_gamma_neg": 0.97, "base_dich_sym_loss": 1.28, "dich_pos_nll_add": 0.08,
        "exact_nll_lam": 0.43, "exact_nll_lam_neg": 0.30, "exact_dich_sym_loss": 1.25, "seed_coh_beta": 0.25,
        "ms_k1": 6, "ms_temp1": 0.035, "ms_blend": 0.22, "body_nll_lam_neg": 0.00, "body_vis_scale": 1.35,
        "body_head_keep": 3.5, "body_head_tau": 0.55, "body_ent_delta": 0.15, "body_pos_q_mul": 0.70,
        "body_err_gamma": 0.90, "body_err_gamma_neg": 0.75, "body_alpha": 0.33, "body_tau": 0.355,
        "cos_c_hp": 0.20, "d_alpha": 0.205, "body_d_alpha": 0.235,
    }
    cfg_recal = {**cfg_b06, "ham_tau": 0.175, "margin_gamma": 1.86}

    p2_t12, p2f_t12 = compute_eva_probs(ks, feat_dir, folds_base, w252=0.50, tau=1.2, scale=1.0)

    def eval_oof_and_simpub(p0_in: np.ndarray, p1_in: np.ndarray, p2_oof: np.ndarray | None, use_2030_feat: bool, kw: dict, w_m: np.ndarray | None):
        p2_o = p2_oof if p2_oof is not None else oof_base["raw_probs"][2]
        raw_oof = [p0_in, p1_in, p2_o]
        flip_oof = [p0_in, p1_in, p2_o if p2_oof is not None else oof_base["flip_probs"][2]]
        post_probs = [apply_post(r, post_cfg) for r in raw_oof]
        m_std_oof = np.std(np.stack(post_probs, axis=0), axis=0).astype(np.float32)
        p_oof = pool_custom(raw_oof, flip_oof, post_cfg, w_m) if w_m is not None else tk.pool_members(raw_oof, post_cfg)
        md_o = dict_to_device(md_oof_2030 if use_2030_feat else md_oof_uni, "cuda")
        md_o.update(p0=p_oof, raw_probs=raw_oof, flip_probs=flip_oof, m_std=m_std_oof)
        oof_dist = tk.run_pipeline(md_o, queries_oof, w_oof, None, **kw)
        del md_o
        torch.cuda.empty_cache()
        ro = tk.eval_with_ctx(oof_dist, oof_ctx)
        r_mev = eval_sliced_ctx(oof_dist, mevid_ctx)
        r_syn = eval_synth_ctxs(oof_dist, synth_ctxs)
        del oof_dist
        torch.cuda.empty_cache()
        sim_pub = {
            k: 0.50 * r_mev[k] + 0.50 * r_syn[k]
            for k in ("mADM", "mE", "mAP", "R1", "R5", "R10", "mINP")
        }
        return ro, sim_pub

    def eval_mix(p0_in: np.ndarray, p1_in: np.ndarray, p0_fold: dict | None, p1_fold: dict | None, p2_fold: dict | None, use_2030_feat: bool, kw: dict, w_m: np.ndarray | None):
        mix_res = {}
        for fold in tk.FOLDS:
            bf = folds_base[fold]
            hp = bf["hp"]
            r0 = p0_fold[fold].copy() if p0_fold is not None else bf["raw_base"][0].copy()
            r1 = p1_fold[fold].copy() if p1_fold is not None else bf["raw_base"][1].copy()
            r2 = p2_fold[fold].copy() if p2_fold is not None else bf["raw_base"][2].copy()
            fl0 = p0_fold[fold].copy() if p0_fold is not None else bf["flip_base"][0].copy()
            fl1 = p1_fold[fold].copy() if p1_fold is not None else bf["flip_base"][1].copy()
            fl2 = p2_fold[fold].copy() if p2_fold is not None else bf["flip_base"][2].copy()
            r0[hp] = p0_in[hp]
            fl0[hp] = p0_in[hp]
            r1[hp] = p1_in[hp]
            fl1[hp] = p1_in[hp]
            r_f = [r0, r1, r2]
            fl_f = [fl0, fl1, fl2]
            pp_f = [apply_post(r, post_cfg) for r in r_f]
            ms_f = np.std(np.stack(pp_f, axis=0), axis=0).astype(np.float32)
            p_f = pool_custom(r_f, fl_f, post_cfg, w_m) if w_m is not None else tk.pool_members(r_f, post_cfg)
            md_m = dict_to_device(md_mix_2030[fold] if use_2030_feat else bf["mixed"], "cuda")
            md_m.update(p0=p_f, raw_probs=r_f, flip_probs=fl_f, m_std=ms_f)
            d_m = tk.run_pipeline(md_m, bf["queries"], bf["w"], bf["hp"], **kw)
            del md_m
            torch.cuda.empty_cache()
            mix_res[fold] = tk.eval_with_ctx(d_m, bf["eval_ctx"])
            del d_m
            torch.cuda.empty_cache()
        return mix_res

    ro_b131, rsp_b131 = eval_oof_and_simpub(oof_base["raw_probs"][0], oof_base["raw_probs"][1], None, False, cfg_b131c, None)
    rm_b131 = eval_mix(oof_base["raw_probs"][0], oof_base["raw_probs"][1], None, None, None, False, cfg_b131c, None)
    b131_oof = ro_b131["mADM"]
    b131_mix = float(np.mean([rm_b131[f]["mADM"] for f in tk.FOLDS]))
    b131_sp = rsp_b131["mADM"]

    cands_p1 = [
        ("00_B137C (baseline)", z4["pe_b137c"], z4["hp_b137c"], cfg_b06),
        ("06_hp7_fix19_m08_r10", z4["pe_b137c"], z4["hp7_fix19_m08_r10_o103"], cfg_b06),
        ("07_hp7_fix19_m10_r10", z4["pe_b137c"], z4["hp7_fix19_m10_r10_o103"], cfg_b06),
        ("08_hp7_fix19_m12_r10", z4["pe_b137c"], z4["hp7_fix19_m12_r10_o103"], cfg_b06),
        ("09_hp7_fix19_m12_r12", z4["pe_b137c"], z4["hp7_fix19_m12_r12_o103"], cfg_b06),
        ("10_hp5_fix19_m12_o103", z4["pe_b137c"], z4["hp5_fix19_m12_o103"], cfg_b06),
        ("11_hp_mask15_m10_r10", z4["pe_b137c"], z4["hp_mask15_m10_r10_o103"], cfg_b06),
        ("12_hp_mask15_m12_r10", z4["pe_b137c"], z4["hp_mask15_m12_r10_o103"], cfg_b06),
        ("13_hp_mask15_m15_r10", z4["pe_b137c"], z4["hp_mask15_m15_r10_o103"], cfg_b06),
        ("14_hp_mask15_m15_r00", z4["pe_b137c"], z4["hp_mask15_m15_r00_o103"], cfg_b06),
        ("15_pe_unmask24_m20", z4["pe_unmask24_m20_qk120_b060"], z4["hp_b137c"], cfg_b06),
        ("16_pe_unmask24_m15", z4["pe_unmask24_m15_qk120_b060"], z4["hp_b137c"], cfg_b06),
        ("17_pe_unm24_m20 + hp7_fix19_m10", z4["pe_unmask24_m20_qk120_b060"], z4["hp7_fix19_m10_r10_o103"], cfg_b06),
        ("18_pe_unm24_m20 + hp_mask15_m12", z4["pe_unmask24_m20_qk120_b060"], z4["hp_mask15_m12_r10_o103"], cfg_b06),
        ("19_pe_unm24_m15 + hp_mask15_m12", z4["pe_unmask24_m15_qk120_b060"], z4["hp_mask15_m12_r10_o103"], cfg_b06),
        ("20_pe_unm24_m15 + hp_mask15_m15", z4["pe_unmask24_m15_qk120_b060"], z4["hp_mask15_m15_r10_o103"], cfg_b06),
        ("21_hp7_fix19_m08 + recal", z4["pe_b137c"], z4["hp7_fix19_m08_r10_o103"], cfg_recal),
        ("22_hp7_fix19_m10 + recal", z4["pe_b137c"], z4["hp7_fix19_m10_r10_o103"], cfg_recal),
        ("23_hp7_fix19_m12 + recal", z4["pe_b137c"], z4["hp7_fix19_m12_r10_o103"], cfg_recal),
        ("24_hp_mask15_m10 + recal", z4["pe_b137c"], z4["hp_mask15_m10_r10_o103"], cfg_recal),
        ("25_hp_mask15_m12 + recal", z4["pe_b137c"], z4["hp_mask15_m12_r10_o103"], cfg_recal),
        ("26_hp_mask15_m15 + recal", z4["pe_b137c"], z4["hp_mask15_m15_r10_o103"], cfg_recal),
        ("27_pe_unm24_m20 + hp_m15_m12 + recal", z4["pe_unmask24_m20_qk120_b060"], z4["hp_mask15_m12_r10_o103"], cfg_recal),
        ("28_pe_unm24_m15 + hp_m15_m15 + recal", z4["pe_unmask24_m15_qk120_b060"], z4["hp_mask15_m15_r10_o103"], cfg_recal),
    ]

    print(f"\n=== PHASE 1: OOF-33k & SimPub Fast Sweep ===")
    print(f"{'Candidate':40s} | {'dOOF':>6s} {'dAP':>6s} {'dR1':>6s} {'dR5':>6s} {'dR10':>6s} | {'dSimP':>6s} {'dSpAP':>6s} {'dSpR1':>6s}")
    print("-" * 105)
    for name, p0_c, p1_c, kw_c in cands_p1:
        ro, rsp = eval_oof_and_simpub(p0_c, p1_c, p2_t12, True, kw_c, w_anat10)
        d_oof = (ro["mADM"] - b131_oof) * 10000.0
        d_ap = (ro["mAP"] - ro_b131["mAP"]) * 10000.0
        d_r1 = (ro["R1"] - ro_b131["R1"]) * 10000.0
        d_r5 = (ro["R5"] - ro_b131["R5"]) * 10000.0
        d_r10 = (ro["R10"] - ro_b131["R10"]) * 10000.0
        d_sp = (rsp["mADM"] - b131_sp) * 10000.0
        d_sp_ap = (rsp["mAP"] - rsp_b131["mAP"]) * 10000.0
        d_sp_r1 = (rsp["R1"] - rsp_b131["R1"]) * 10000.0
        print(
            f"{name:40s} | {d_oof:+6.2f} {d_ap:+6.2f} {d_r1:+6.2f} {d_r5:+6.2f} {d_r10:+6.2f} | "
            f"{d_sp:+6.2f} {d_sp_ap:+6.2f} {d_sp_r1:+6.2f}",
            flush=True,
        )

    # Phase 2: Fold-consistent Mixed-Gallery evaluation for key candidates
    pe_w_2030 = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
    pe_w_m20 = {"attrq": 0.225, "rankq": 0.225, "maskv2": 0.20, "pswap": 0.35}
    pe_w_m15 = {"attrq": 0.24, "rankq": 0.24, "maskv2": 0.15, "pswap": 0.37}
    pe_ov_unm24_m20 = {c: pe_w_m20 for c in UNMASKED_24}
    pe_ov_unm24_m15 = {c: pe_w_m15 for c in UNMASKED_24}

    hp_w_6_10 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "qm": 0.0, "r224": 0.10}
    hp_w_7_m08_r10 = {"q": 0.164, "qr": 0.246, "qrq": 0.246, "qs": 0.164, "qm": 0.08, "r224": 0.10}
    hp_w_7_m10_r10 = {"q": 0.16, "qr": 0.24, "qrq": 0.24, "qs": 0.16, "qm": 0.10, "r224": 0.10}
    hp_w_7_m12_r10 = {"q": 0.156, "qr": 0.234, "qrq": 0.234, "qs": 0.156, "qm": 0.12, "r224": 0.10}
    hp_w_7_m15_r10 = {"q": 0.15, "qr": 0.225, "qrq": 0.225, "qs": 0.15, "qm": 0.15, "r224": 0.10}

    p0f_b137c = make_fold_logit_probs(pe_fold_logits, pe_w_2030, None, 1.0)
    p0f_unm24_m20 = make_fold_logit_probs(pe_fold_logits, pe_w_2030, pe_ov_unm24_m20, 1.0)
    p0f_unm24_m15 = make_fold_logit_probs(pe_fold_logits, pe_w_2030, pe_ov_unm24_m15, 1.0)

    p1f_b137c = make_fold_logit_probs(hp_fold_logits, hp_w_6_10, None, 1.03)
    p1f_fix19_m08 = make_fold_logit_probs(hp_fold_logits, hp_w_7_m08_r10, {19: hp_w_6_10}, 1.03)
    p1f_fix19_m10 = make_fold_logit_probs(hp_fold_logits, hp_w_7_m10_r10, {19: hp_w_6_10}, 1.03)
    p1f_fix19_m12 = make_fold_logit_probs(hp_fold_logits, hp_w_7_m12_r10, {19: hp_w_6_10}, 1.03)
    p1f_mask15_m10 = make_fold_logit_probs(hp_fold_logits, hp_w_6_10, {c: hp_w_7_m10_r10 for c in MASKED_15}, 1.03)
    p1f_mask15_m12 = make_fold_logit_probs(hp_fold_logits, hp_w_6_10, {c: hp_w_7_m12_r10 for c in MASKED_15}, 1.03)
    p1f_mask15_m15 = make_fold_logit_probs(hp_fold_logits, hp_w_6_10, {c: hp_w_7_m15_r10 for c in MASKED_15}, 1.03)

    cands_p2 = [
        ("00_B137C (fold-matched cp)", z4["pe_b137c"], z4["hp_b137c"], p0f_b137c, p1f_b137c, cfg_b06),
        ("06_hp7_fix19_m08_r10", z4["pe_b137c"], z4["hp7_fix19_m08_r10_o103"], p0f_b137c, p1f_fix19_m08, cfg_b06),
        ("07_hp7_fix19_m10_r10", z4["pe_b137c"], z4["hp7_fix19_m10_r10_o103"], p0f_b137c, p1f_fix19_m10, cfg_b06),
        ("08_hp7_fix19_m12_r10", z4["pe_b137c"], z4["hp7_fix19_m12_r10_o103"], p0f_b137c, p1f_fix19_m12, cfg_b06),
        ("11_hp_mask15_m10_r10", z4["pe_b137c"], z4["hp_mask15_m10_r10_o103"], p0f_b137c, p1f_mask15_m10, cfg_b06),
        ("12_hp_mask15_m12_r10", z4["pe_b137c"], z4["hp_mask15_m12_r10_o103"], p0f_b137c, p1f_mask15_m12, cfg_b06),
        ("13_hp_mask15_m15_r10", z4["pe_b137c"], z4["hp_mask15_m15_r10_o103"], p0f_b137c, p1f_mask15_m15, cfg_b06),
        ("18_pe_unm24_m20 + hp_mask15_m12", z4["pe_unmask24_m20_qk120_b060"], z4["hp_mask15_m12_r10_o103"], p0f_unm24_m20, p1f_mask15_m12, cfg_b06),
        ("20_pe_unm24_m15 + hp_mask15_m15", z4["pe_unmask24_m15_qk120_b060"], z4["hp_mask15_m15_r10_o103"], p0f_unm24_m15, p1f_mask15_m15, cfg_b06),
        ("22_hp7_fix19_m10 + recal", z4["pe_b137c"], z4["hp7_fix19_m10_r10_o103"], p0f_b137c, p1f_fix19_m10, cfg_recal),
        ("25_hp_mask15_m12 + recal", z4["pe_b137c"], z4["hp_mask15_m12_r10_o103"], p0f_b137c, p1f_mask15_m12, cfg_recal),
        ("27_pe_unm24_m20 + hp_m15_m12 + recal", z4["pe_unmask24_m20_qk120_b060"], z4["hp_mask15_m12_r10_o103"], p0f_unm24_m20, p1f_mask15_m12, cfg_recal),
    ]

    print(f"\n=== PHASE 2: Fold-Consistent Mixed-Gallery Evaluation ===")
    print(f"{'Candidate':40s} | {'dMix':>6s} ({'PA':>5s}/{'PE':>5s}/{'Mk':>5s}) {'dMxAP':>6s} {'dMxR1':>6s}")
    print("-" * 90)
    for name, p0_c, p1_c, p0f_c, p1f_c, kw_c in cands_p2:
        rm = eval_mix(p0_c, p1_c, p0f_c, p1f_c, p2f_t12, True, kw_c, w_anat10)
        m_avg = float(np.mean([rm[f]["mADM"] for f in tk.FOLDS]))
        d_mix = (m_avg - b131_mix) * 10000.0
        d_pa = (rm["PA100k"]["mADM"] - rm_b131["PA100k"]["mADM"]) * 10000.0
        d_pe = (rm["PETA"]["mADM"] - rm_b131["PETA"]["mADM"]) * 10000.0
        d_mk = (rm["Market1501"]["mADM"] - rm_b131["Market1501"]["mADM"]) * 10000.0
        d_mx_ap = (float(np.mean([rm[f]["mAP"] for f in tk.FOLDS])) - float(np.mean([rm_b131[f]["mAP"] for f in tk.FOLDS]))) * 10000.0
        d_mx_r1 = (float(np.mean([rm[f]["R1"] for f in tk.FOLDS])) - float(np.mean([rm_b131[f]["R1"] for f in tk.FOLDS]))) * 10000.0
        print(
            f"{name:40s} | {d_mix:+6.2f} ({d_pa:+5.1f}/{d_pe:+5.1f}/{d_mk:+5.1f}) {d_mx_ap:+6.2f} {d_mx_r1:+6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
