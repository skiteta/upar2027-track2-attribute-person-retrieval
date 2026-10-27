from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/package_v3/B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR_V3.zip"))

import test_tri_graph_combined_knobs as tk
from calibrate_public_mevid_synth import (
    build_embedded_synth_and_mevid_contexts,
    eval_sliced_ctx,
    eval_synth_ctxs,
)
from upar.data import load_retrieval_split
from upar.postproc import apply_post
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC


def compute_me(madm: float, map_v: float) -> float:
    return (madm - map_v) / max(1.0 - map_v, 1e-12)


def pool_custom(
    raw_probs: list[np.ndarray],
    flip_probs: list[np.ndarray],
    post_cfg: dict,
    w_mat: np.ndarray,  # shape (3, 40) or (3,)
    flip_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gamma_mem: float = 0.0,
    conf_beta: float = 0.0,
) -> np.ndarray:
    eff = []
    for i in range(3):
        fw = float(flip_weights[i])
        if fw > 0.0:
            eff.append((1.0 - fw) * raw_probs[i].astype(np.float64) + fw * flip_probs[i].astype(np.float64))
        else:
            eff.append(raw_probs[i].astype(np.float64))
    stack = np.stack(eff, axis=0)  # (3, G, 40)
    w = np.asarray(w_mat, dtype=np.float64)
    if w.ndim == 1:
        w = w[:, None, None]
    elif w.ndim == 2:
        w = w[:, None, :]
    if conf_beta != 0.0:
        conf = np.abs(2.0 * stack - 1.0)
        w = w * np.exp(float(conf_beta) * conf)
    if gamma_mem > 0.0:
        cert = np.exp(float(gamma_mem) * (1.0 - 4.0 * stack * (1.0 - stack)))
        w = w * cert
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    out = (w * stack).sum(axis=0).astype(np.float32)
    return apply_post(out, post_cfg)


def build_variant_contexts(
    variant_name: str,
    hp_npz_prefix: str | None,
    use_224_features: bool,
    feat_dir: Path,
    split,
    plan: dict,
    spec_obj: dict,
    train_proto: dict,
    base_folds: dict | None = None,
) -> tuple[dict, dict]:
    post_cfg = spec_obj["lodo"]["post"]
    n_tot = split.gallery_size
    u_pe_oof = np.empty((n_tot, 1024), dtype=np.float64)
    u_eva_oof = np.empty((n_tot, 1024), dtype=np.float64)
    u_hp_oof = np.empty((n_tot, 1280), dtype=np.float64)
    p_oof = np.empty((n_tot, 40), dtype=np.float32)
    raw_oof = [np.empty((n_tot, 40), dtype=np.float32) for _ in range(3)]
    flip_oof = [np.empty((n_tot, 40), dtype=np.float32) for _ in range(3)]
    m_std_oof = np.empty((n_tot, 40), dtype=np.float32)
    fold_out = {}

    for fold in tk.FOLDS:
        if base_folds is not None:
            bf = base_folds[fold]
            hp = bf["hp"]
            sub = bf["sub"]
            qi = bf["qi"]
            queries = bf["queries"]
            w = bf["w"]
            eval_ctx = bf["eval_ctx"]
            u_pe_m = bf["u_pe_m"]
            u_eva_m = bf["u_eva_m"]
            u_hp_192 = bf["u_hp_192"]
            raw_base = [r.copy() for r in bf["raw_base"]]
            flip_base = [r.copy() for r in bf["flip_base"]]
            positions = bf["positions"]
        else:
            pe_npz = feat_dir / f"pe_full_val_{fold}.npz"
            if pe_npz.exists():
                hp = np.flatnonzero(split.sources == fold)
                sub = split.subset_gallery(split.sources == fold)
                old_prob_npz = ROOT / f"artifacts/runs/20261017T0446+0900_gallery_reciprocal_rerank/comparison_v1/graph_{fold}.npz"
                with np.load(old_prob_npz, allow_pickle=False) as zp:
                    w = zp["reliability"].copy()
                pf_feats = np.load(pe_npz)["features"]
            else:
                d = tk.mod_mix.load_fold(fold, split, plan)
                hp = d["hp"]
                sub = d["sub"]
                w = d["w"]
                pf_feats = d["features"]
            qi = np.unique(sub.gallery_query_ids)
            queries = sub.queries[qi].astype(np.float32)
            eval_ctx = tk.precompute_eval_ctx(sub, qi)
            row_meta = next(r for r in train_proto["folds"] if r["fold"] == fold)
            with np.load(ROOT / row_meta["meta"]["path"], allow_pickle=False) as zm:
                meta = {k: zm[k].copy() for k in ("split", "sources", "image_paths", "attribute_names")}
            val_idx = np.flatnonzero(meta["split"] == "val")
            positions = np.searchsorted(val_idx, val_idx)
            raw_base, flip_base, _ = tk.load_member_probs_raw(spec_obj, fold, meta, val_idx)
            ef = np.load(feat_dir / f"eva_full_val_{fold}.npz")
            hf = np.load(feat_dir / f"hp_full_val_{fold}.npz")
            u_pe_m = tk.unit_features_fp64(pf_feats)
            u_eva_m = tk.unit_features_fp64(ef["features"])
            u_hp_192 = tk.unit_features_fp64(hf["features"])

        raw_probs = [r.copy() for r in raw_base]
        flip_probs = [r.copy() for r in flip_base]
        u_hp_m = u_hp_192

        if hp_npz_prefix is not None:
            if hp_npz_prefix == "hp6_4060_224":
                z6 = np.load(feat_dir / f"hp6_224_full_val_{fold}.npz")
                z50 = np.load(feat_dir / f"hp6_5050_224_full_val_{fold}.npz")
                corr = np.log(z6["pos_weight"]).astype(np.float32)
                lo = 0.60 * z6["logits_original"] + 0.40 * z50["logits_original"]
                lf = 0.60 * z6["logits_flip"] + 0.40 * z50["logits_flip"]
                raw_probs[1] = (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)[positions]
                flip_probs[1] = (1.0 / (1.0 + np.exp(-(lf - corr)))).astype(np.float32)[positions]
                if use_224_features:
                    f_blend = (0.60 * z6["features"] + 0.40 * z50["features"]).astype(np.float32)
                    u_hp_m = tk.unit_features_fp64(f_blend)
            else:
                zhp = np.load(feat_dir / f"{hp_npz_prefix}_full_val_{fold}.npz")
                corr = np.log(zhp["pos_weight"]).astype(np.float32)
                raw_probs[1] = (1.0 / (1.0 + np.exp(-(zhp["logits_original"] - corr)))).astype(np.float32)[positions]
                flip_probs[1] = (1.0 / (1.0 + np.exp(-(zhp["logits_flip"] - corr)))).astype(np.float32)[positions]
                if use_224_features:
                    u_hp_m = tk.unit_features_fp64(zhp["features"])

        post_probs = [apply_post(r, post_cfg) for r in raw_probs]
        m_std_all = np.std(np.stack(post_probs, axis=0), axis=0).astype(np.float32)
        p0_m = tk.pool_members(raw_probs, post_cfg)

        u_pe_oof[hp] = u_pe_m[hp]
        u_eva_oof[hp] = u_eva_m[hp]
        u_hp_oof[hp] = u_hp_m[hp]
        p_oof[hp] = p0_m[hp]
        m_std_oof[hp] = m_std_all[hp]
        for i in range(3):
            raw_oof[i][hp] = raw_probs[i][hp]
            flip_oof[i][hp] = flip_probs[i][hp]

        mixed_md = tk.precompute_fold_mode(
            u_pe_m, u_eva_m, u_hp_m, p0_m, raw_probs, flip_probs, post_cfg, m_std_all
        )
        fold_out[fold] = {
            "hp": hp,
            "sub": sub,
            "qi": qi,
            "queries": queries,
            "w": w,
            "mixed": mixed_md,
            "eval_ctx": eval_ctx,
            "u_pe_m": u_pe_m,
            "u_eva_m": u_eva_m,
            "u_hp_192": u_hp_192,
            "raw_base": raw_base,
            "flip_base": flip_base,
            "positions": positions,
        }

    oof_md = tk.precompute_fold_mode(
        u_pe_oof, u_eva_oof, u_hp_oof, p_oof, raw_oof, flip_oof, post_cfg, m_std_oof
    )
    return fold_out, oof_md


def main() -> None:
    t0 = time.monotonic()
    feat_dir = HERE / "extracted_features_v1"
    plan = json.loads((ROOT / "artifacts/runs/20261017T1618+0900_mixed_gallery_context/protocol.json").read_text())
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    spec_obj = read_json(SPEC)
    train_proto = read_json(ROOT / "artifacts/runs/20261017T1105+0900_cycle_continuation/frontier/sourcefit_edge_v2/protocol.json")

    qi_oof = np.unique(split.gallery_query_ids)
    queries_oof = split.queries[qi_oof].astype(np.float32)
    oof_ctx = tk.precompute_eval_ctx(split, qi_oof)
    synth_ctxs, mevid_ctx = build_embedded_synth_and_mevid_contexts(split, qi_oof, n_seeds=5)

    # Build contexts for:
    # 1. base_192 (hp4 @ 192x96)
    # 2. hp6_224_prob_only (hp6 @ 224x112 probs, 192x96 features)
    # 3. hp6_224_full (hp6 @ 224x112 probs + 224x112 features)
    # 4. hp6_5050_224_prob_only (hp6_5050 @ 224x112 probs, 192x96 features)
    # 5. hp6_5050_224_full (hp6_5050 @ 224x112 probs + 224x112 features)
    # 6. hp6_4060_224_full (0.6*hp6 + 0.4*hp6_5050 @ 224x112 probs + features)
    t_c = time.monotonic()
    folds_base, oof_base = build_variant_contexts("base_192", None, False, feat_dir, split, plan, spec_obj, train_proto)
    folds_hp6_p, oof_hp6_p = build_variant_contexts("hp6_224_p", "hp6_224", False, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    folds_hp6_f, oof_hp6_f = build_variant_contexts("hp6_224_f", "hp6_224", True, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    folds_hp50_p, oof_hp50_p = build_variant_contexts("hp50_224_p", "hp6_5050_224", False, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    folds_hp50_f, oof_hp50_f = build_variant_contexts("hp50_224_f", "hp6_5050_224", True, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    folds_hp40_f, oof_hp40_f = build_variant_contexts("hp40_224_f", "hp6_4060_224", True, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    w_oof = np.mean([folds_base[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)
    print(f"Built all 6 representation contexts in {time.monotonic() - t_c:.1f}s", flush=True)

    # Check single-member OOF33k LogLoss & Brier improvement of DINOv3-H+ across variants!
    gt_attr = split.gallery_attributes.astype(np.float64)
    print("\n=== DINOv3-H+ Member-1 OOF33k Calibration & Accuracy Across Resolutions/Soups ===")
    for vname, o_md in [
        ("HP4 @ 192x96 (B131C-B133C)", oof_base),
        ("HP6 @ 224x112 (6-recipe)", oof_hp6_f),
        ("HP6_4060 @ 224x112", oof_hp40_f),
        ("HP6_5050 @ 224x112 (50/50)", oof_hp50_f),
    ]:
        pm = np.clip(o_md["raw_probs"][1].astype(np.float64), 1e-5, 1.0 - 1e-5)
        pf = np.clip((0.80 * o_md["raw_probs"][1] + 0.20 * o_md["flip_probs"][1]).astype(np.float64), 1e-5, 1.0 - 1e-5)
        ll_o = -(gt_attr * np.log(pm) + (1.0 - gt_attr) * np.log(1.0 - pm)).mean()
        br_o = ((pm - gt_attr) ** 2).mean()
        ll_f = -(gt_attr * np.log(pf) + (1.0 - gt_attr) * np.log(1.0 - pf)).mean()
        br_f = ((pf - gt_attr) ** 2).mean()
        print(f"  {vname:30s} | Orig LogLoss={ll_o:.6f}, Brier={br_o:.6f} | +20%Flip LogLoss={ll_f:.6f}, Brier={br_f:.6f}")

    # LODO-clean per-attribute MoE weights keeping EVA fixed at 0.25 and routing PE vs HP (total PE+HP = 0.75)
    # Let's compute LODO logloss difference between PE (member 0) and HP50_224 (member 1) on the 2 training folds!
    lodo_pe_hp_diff = {}
    for f in tk.FOLDS:
        other_hp = np.concatenate([folds_base[f2]["hp"] for f2 in tk.FOLDS if f2 != f])
        gt_tr = gt_attr[other_hp]
        p_pe = np.clip(oof_hp50_f["raw_probs"][0][other_hp].astype(np.float64), 1e-5, 1.0 - 1e-5)
        p_hp = np.clip(oof_hp50_f["raw_probs"][1][other_hp].astype(np.float64), 1e-5, 1.0 - 1e-5)
        ll_pe = -(gt_tr * np.log(p_pe) + (1.0 - gt_tr) * np.log(1.0 - p_pe)).mean(axis=0)
        ll_hp = -(gt_tr * np.log(p_hp) + (1.0 - gt_tr) * np.log(1.0 - p_hp)).mean(axis=0)
        # Relative logloss advantage: positive means HP has lower (better) logloss than PE
        rel_adv_hp = (ll_pe - ll_hp) / np.maximum(0.5 * (ll_pe + ll_hp), 1e-6)
        lodo_pe_hp_diff[f] = np.clip(rel_adv_hp, -0.25, 0.25)

    def make_pe_hp_moe_weights(base_pe: float, base_hp: float, base_eva: float, slope: float, adv_vec: np.ndarray) -> np.ndarray:
        # Shifts weight between PE and HP according to per-attribute LODO relative advantage, preserving PE+HP = base_pe+base_hp
        delta = np.clip(slope * adv_vec, -0.08, 0.08)
        w_pe = base_pe - delta
        w_hp = base_hp + delta
        w_eva = np.full_like(w_pe, base_eva)
        return np.stack([w_pe, w_hp, w_eva], axis=0)

    b5_tw07 = [
        ("pe", 0.275, 1.0, 0.0),
        ("eva", 0.23, 0.0, 1.0),
        ("j25_k28", 0.235, 0.75, 0.25),
        ("j50_k28", 0.19, 0.50, 0.50),
        ("tri25_k28", 0.07, 0.5625, 0.1875),
    ]
    b6_tw10 = [
        ("pe", 0.27, 1.0, 0.0),
        ("eva", 0.225, 0.0, 1.0),
        ("j25_k28", 0.225, 0.75, 0.25),
        ("j50_k28", 0.18, 0.50, 0.50),
        ("tri25_k28", 0.06, 0.5625, 0.1875),
        ("tri40_k28", 0.04, 0.40, 0.30),
    ]
    aw_8spec = {
        "pe": 0.10, "eva": 0.10, "j25": 0.18, "j50": 0.18,
        "tri25": 0.16, "tri40": 0.14, "pe_hp": 0.07, "eva_hp": 0.07,
    }
    cfg_b131c = {
        "branches": b5_tw07,
        "anchor_spec_weights": aw_8spec,
        "ham_tau": 0.17,
        "dist_kappa": 0.20,
        "sim_diff_alpha": 0.34,
        "sim_diff_alpha2": 0.04,
        "exact_seed_blend": 0.72,
        "ex_vis_alpha": 0.120,
        "anchor_alpha_q": 0.126,
        "anchor_temp_q": 0.052,
        "head_beta": 0.044,
        "head_tau": 0.225,
        "anchor_top_m": 11,
        "tail_mul_kap": 0.04,
        "head_dq_blend": 0.15,
        "seed_disagree_kap": 0.030,
        "seed_kap_weight": False,
        "exact_branch_blend": 0.22,
        "ex_diff_alpha": 0.07,
        "ex_diff_head_blend": 0.50,
        "ex_kappa": 0.24,
        "head_local_lam": 0.12,
        "head_local_k": 12,
        "body_alpha": 0.20,
        "body_k0": 10,
        "body_tau": 0.40,
        "body_pow": 1.0,
        "body_w_uni": 1.0,
        "body_head_keep": 1.0,
        "body_err_gamma": 0.85,
        "body_nll_lam": 0.05,
    }
    c60 = {
        **cfg_b131c,
        "nll_lam": 0.275,
        "nll_lam_neg": 0.215,
        "err_gamma_neg": 0.97,
        "base_dich_sym_loss": 1.28,
        "dich_pos_nll_add": 0.08,
        "exact_nll_lam": 0.43,
        "exact_nll_lam_neg": 0.30,
        "exact_dich_sym_loss": 1.25,
        "seed_coh_beta": 0.25,
        "ms_k1": 6,
        "ms_temp1": 0.035,
        "ms_blend": 0.22,
        "body_nll_lam_neg": 0.00,
        "body_vis_scale": 1.35,
        "body_head_keep": 3.5,
        "body_head_tau": 0.55,
        "body_ent_delta": 0.15,
        "body_pos_q_mul": 0.70,
        "body_err_gamma": 0.90,
        "body_err_gamma_neg": 0.75,
        "body_alpha": 0.32,
        "body_tau": 0.36,
    }
    c80 = {
        **c60,
        "branches": b6_tw10,
        "dyn_conf": True,
        "d_alpha": 0.22,
        "body_d_alpha": 0.25,
        "margin_scale": 0.95,
    }
    c81 = {
        **c60,
        "dyn_conf": True,
        "d_alpha": 0.22,
        "body_d_alpha": 0.25,
        "margin_scale": 0.95,
    }

    # Candidate list: (name, (folds_ctx, oof_ctx_md), kw_dict, custom_pool_spec)
    candidates = [
        ("00_B131C_anchor", (folds_base, oof_base), cfg_b131c, None),
        ("01_c60_base_192", (folds_base, oof_base), c60, None),
        ("02_c80_base_192", (folds_base, oof_base), c80, None),
        # HP6 @ 224x112 (2/3 W_192 + 1/3 W_224)
        ("10_c60_hp6_224_probOnly", (folds_hp6_p, oof_hp6_p), c60, None),
        ("11_c60_hp6_224_full", (folds_hp6_f, oof_hp6_f), c60, None),
        ("12_c80_hp6_224_full", (folds_hp6_f, oof_hp6_f), c80, None),
        ("13_c81_hp6_224_full", (folds_hp6_f, oof_hp6_f), c81, None),
        # HP6_4060 @ 224x112 (0.60 W_192 + 0.40 W_224)
        ("15_c60_hp4060_224_full", (folds_hp40_f, oof_hp40_f), c60, None),
        ("16_c80_hp4060_224_full", (folds_hp40_f, oof_hp40_f), c80, None),
        ("17_c81_hp4060_224_full", (folds_hp40_f, oof_hp40_f), c81, None),
        # HP6_5050 @ 224x112 (0.50 W_192 + 0.50 W_224)
        ("20_c60_hp5050_224_probOnly", (folds_hp50_p, oof_hp50_p), c60, None),
        ("21_c60_hp5050_224_full", (folds_hp50_f, oof_hp50_f), c60, None),
        ("22_c80_hp5050_224_full", (folds_hp50_f, oof_hp50_f), c80, None),
        ("23_c81_hp5050_224_full", (folds_hp50_f, oof_hp50_f), c81, None),
        # Member weights & Flip TTA on HP6_5050 @ 224x112
        ("30_c81_hp5050_mw39_36_25", (folds_hp50_f, oof_hp50_f), {**c81, "member_weights": (0.39, 0.36, 0.25)}, None),
        ("31_c81_hp5050_mw38_37_25", (folds_hp50_f, oof_hp50_f), {**c81, "member_weights": (0.38, 0.37, 0.25)}, None),
        ("32_c81_hp5050_mw37_38_25", (folds_hp50_f, oof_hp50_f), {**c81, "member_weights": (0.37, 0.38, 0.25)}, None),
        ("33_c81_hp5050_flip15", (folds_hp50_f, oof_hp50_f), {**c81, "flip_weights": (0.15, 0.15, 0.15)}, None),
        ("34_c81_hp5050_flip20", (folds_hp50_f, oof_hp50_f), {**c81, "flip_weights": (0.20, 0.20, 0.20)}, None),
        ("35_c81_hp5050_mw38_37_flip15", (folds_hp50_f, oof_hp50_f), {**c81, "member_weights": (0.38, 0.37, 0.25), "flip_weights": (0.15, 0.15, 0.15)}, None),
        ("36_c80_hp5050_mw38_37_flip15", (folds_hp50_f, oof_hp50_f), {**c80, "member_weights": (0.38, 0.37, 0.25), "flip_weights": (0.15, 0.15, 0.15)}, None),
        # Per-attribute LODO MoE routing (keeping EVA=0.25, routing PE vs HP by attribute accuracy)
        ("40_c81_hp5050_moe_s025", (folds_hp50_f, oof_hp50_f), c81, ("pe_hp_moe", 0.39, 0.36, 0.25, 0.25, (0.0, 0.0, 0.0), 0.0)),
        ("41_c81_hp5050_moe_s040", (folds_hp50_f, oof_hp50_f), c81, ("pe_hp_moe", 0.39, 0.36, 0.25, 0.40, (0.0, 0.0, 0.0), 0.0)),
        ("42_c81_hp5050_moe_s025_flip15", (folds_hp50_f, oof_hp50_f), c81, ("pe_hp_moe", 0.39, 0.36, 0.25, 0.25, (0.15, 0.15, 0.15), 0.0)),
        ("43_c81_hp5050_moe_s025_cb02", (folds_hp50_f, oof_hp50_f), c81, ("pe_hp_moe", 0.39, 0.36, 0.25, 0.25, (0.15, 0.15, 0.15), 0.20)),
        # Also test HP6 @ 224x112 with mw + flip + moe
        ("50_c81_hp6_mw38_37_flip15", (folds_hp6_f, oof_hp6_f), {**c81, "member_weights": (0.38, 0.37, 0.25), "flip_weights": (0.15, 0.15, 0.15)}, None),
        ("51_c81_hp4060_mw38_37_flip15", (folds_hp40_f, oof_hp40_f), {**c81, "member_weights": (0.38, 0.37, 0.25), "flip_weights": (0.15, 0.15, 0.15)}, None),
    ]

    print("\n=== Comprehensive Stack Evaluation ===")
    ref = None
    for name, (f_ctx, o_md), kw, c_spec in candidates:
        m_map = {}
        for f in tk.FOLDS:
            fd = f_ctx[f]
            md_f = dict(fd["mixed"])
            if c_spec is not None and c_spec[0] == "pe_hp_moe":
                _, b_pe, b_hp, b_eva, slope, fw, cb = c_spec
                w_m = make_pe_hp_moe_weights(b_pe, b_hp, b_eva, slope, lodo_pe_hp_diff[f])
                md_f["p0"] = pool_custom(md_f["raw_probs"], md_f["flip_probs"], md_f["post_cfg"], w_m, flip_weights=fw, conf_beta=cb)
            m_map[f] = tk.run_pipeline(md_f, fd["queries"], fd["w"], fd["hp"], **kw)

        md_o = dict(o_md)
        if c_spec is not None and c_spec[0] == "pe_hp_moe":
            _, b_pe, b_hp, b_eva, slope, fw, cb = c_spec
            p0_o = np.empty_like(o_md["p0"])
            for f in tk.FOLDS:
                hp = f_ctx[f]["hp"]
                w_m = make_pe_hp_moe_weights(b_pe, b_hp, b_eva, slope, lodo_pe_hp_diff[f])
                raw_sub = [r[hp] for r in o_md["raw_probs"]]
                flp_sub = [r[hp] for r in o_md["flip_probs"]]
                p0_o[hp] = pool_custom(raw_sub, flp_sub, o_md["post_cfg"], w_m, flip_weights=fw, conf_beta=cb)
            md_o["p0"] = p0_o
        oof_dist = tk.run_pipeline(md_o, queries_oof, w_oof, None, **kw)

        rm = tk.eval_dist_map(m_map, f_ctx)
        ro = tk.eval_with_ctx(oof_dist, oof_ctx)
        r_mev = eval_sliced_ctx(oof_dist, mevid_ctx)
        r_syn = eval_synth_ctxs(oof_dist, synth_ctxs)

        om_a, om_p = ro["mADM"], ro["mAP"]
        om_e = compute_me(om_a, om_p)
        mm_a, mm_p = rm["macro"], rm["mAP"]
        mm_e = compute_me(mm_a, mm_p)
        sim_pub = 0.5 * (r_mev["mADM"] + r_syn["mADM"])

        if ref is None:
            ref = (om_a, om_p, om_e, mm_a, mm_p, mm_e, sim_pub, r_mev["mADM"], r_syn["mADM"])
        dom_a = (om_a - ref[0]) * 10000
        dom_p = (om_p - ref[1]) * 10000
        dom_e = (om_e - ref[2]) * 10000
        dmm_a = (mm_a - ref[3]) * 10000
        dmm_p = (mm_p - ref[4]) * 10000
        dmm_e = (mm_e - ref[5]) * 10000
        dsim = (sim_pub - ref[6]) * 10000
        dmev = (r_mev["mADM"] - ref[7]) * 10000
        dsyn = (r_syn["mADM"] - ref[8]) * 10000

        pred_pub_oof = 0.699485 + (0.0804 * dom_p + 0.1938 * dom_e) / 10000.0
        pred_pub_mix = 0.699485 + (0.0725 * dmm_p + 0.2193 * dmm_e) / 10000.0
        d_pred_oof = (pred_pub_oof - 0.699485) * 10000
        d_pred_mix = (pred_pub_mix - 0.699485) * 10000

        print(
            f"{name:32s} | OOF:{om_a:.6f}({dom_a:+6.2f}) [dP:{dom_p:+6.2f}, dE:{dom_e:+6.2f}] | "
            f"Mix:{mm_a:.6f}({dmm_a:+6.2f}) [dP:{dmm_p:+6.2f}, dE:{dmm_e:+6.2f}] | "
            f"dSim:{dsim:+5.2f}(M:{dmev:+5.2f},S:{dsyn:+5.2f}) | "
            f"Pred(O/M): {pred_pub_oof:.6f}({d_pred_oof:+5.2f}) / {pred_pub_mix:.6f}({d_pred_mix:+5.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()

