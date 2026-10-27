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
from test_hp6_224_stack import build_variant_contexts, compute_me, pool_custom
from calibrate_public_mevid_synth import (
    build_embedded_synth_and_mevid_contexts,
    eval_sliced_ctx,
    eval_synth_ctxs,
)
from upar.data import load_retrieval_split
from upar.postproc import apply_post
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC


def build_anat_moe(w_bag, w_age_acc, w_global, w_u_len_gen=None):
    w = np.zeros((3, 40), dtype=np.float64)
    for j in range(40):
        if j in [4, 5, 6]:
            w[:, j] = w_bag
        elif j in [0, 1, 2, 3, 36, 37, 38, 39]:
            w[:, j] = w_age_acc
        elif j in [7, 20] and w_u_len_gen is not None:
            w[:, j] = w_u_len_gen
        else:
            w[:, j] = w_global
    return w


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

    folds_base, oof_base = build_variant_contexts("base_192", None, False, feat_dir, split, plan, spec_obj, train_proto)
    folds_hp6, oof_hp6 = build_variant_contexts("hp6_224_f", "hp6_224", True, feat_dir, split, plan, spec_obj, train_proto, folds_base)
    w_oof = np.mean([folds_base[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)

    w_anat5 = build_anat_moe([0.415, 0.340, 0.245], [0.415, 0.340, 0.245], [0.365, 0.390, 0.245])
    w_anat7 = build_anat_moe([0.420, 0.335, 0.245], [0.405, 0.350, 0.245], [0.365, 0.390, 0.245], [0.345, 0.410, 0.245])
    w_anat8 = build_anat_moe([0.415, 0.340, 0.245], [0.410, 0.345, 0.245], [0.365, 0.390, 0.245], [0.350, 0.405, 0.245])
    w_anat9 = build_anat_moe([0.415, 0.345, 0.240], [0.415, 0.345, 0.240], [0.365, 0.395, 0.240])

    b5_tw07 = [
        ("pe", 0.275, 1.0, 0.0),
        ("eva", 0.23, 0.0, 1.0),
        ("j25_k28", 0.235, 0.75, 0.25),
        ("j50_k28", 0.19, 0.50, 0.50),
        ("tri25_k28", 0.07, 0.5625, 0.1875),
    ]
    b6_tw09 = [
        ("pe", 0.272, 1.0, 0.0),
        ("eva", 0.228, 0.0, 1.0),
        ("j25_k28", 0.230, 0.75, 0.25),
        ("j50_k28", 0.185, 0.50, 0.50),
        ("tri25_k28", 0.060, 0.5625, 0.1875),
        ("tri40_k28", 0.025, 0.40, 0.30),
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

    candidates = [
        ("00_B131C_anchor", (folds_base, oof_base), cfg_b131c, None),
        ("01_c60_base_192", (folds_base, oof_base), c60, None),
        ("E05_c60_hp6_anatMoE5", (folds_hp6, oof_hp6), c60, w_anat5),
        ("E07_c60_hp6_anatMoE7", (folds_hp6, oof_hp6), c60, w_anat7),
        ("E08_c60_hp6_anatMoE8", (folds_hp6, oof_hp6), c60, w_anat8),
        ("E09_c60_hp6_anatMoE9", (folds_hp6, oof_hp6), c60, w_anat9),
        # Combinations with E05 (w_anat5):
        ("F01_E05_cosCHp020", (folds_hp6, oof_hp6), {**c60, "cos_c_hp": 0.20}, w_anat5),
        ("F02_E05_b6_tw09", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09}, w_anat5),
        ("F03_E05_b6_tw09_cosCHp020", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09, "cos_c_hp": 0.20}, w_anat5),
        ("F04_E05_bodyA034_tau035", (folds_hp6, oof_hp6), {**c60, "body_alpha": 0.34, "body_tau": 0.35}, w_anat5),
        ("F05_E05_b6_tw09_bodyA034_tau035", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09, "body_alpha": 0.34, "body_tau": 0.35}, w_anat5),
        ("F06_E05_b6_tw09_cos020_bodyA034", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09, "cos_c_hp": 0.20, "body_alpha": 0.34, "body_tau": 0.35}, w_anat5),
        ("F07_E08_b6_tw09_cos020_bodyA034", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09, "cos_c_hp": 0.20, "body_alpha": 0.34, "body_tau": 0.35}, w_anat8),
        ("F08_E05_cos020_bodyA033_da21_24", (folds_hp6, oof_hp6), {**c60, "cos_c_hp": 0.20, "body_alpha": 0.33, "d_alpha": 0.21, "body_d_alpha": 0.24}, w_anat5),
        ("F09_E05_b6_cos020_bodyA033_da21_24", (folds_hp6, oof_hp6), {**c60, "branches": b6_tw09, "cos_c_hp": 0.20, "body_alpha": 0.33, "d_alpha": 0.21, "body_d_alpha": 0.24}, w_anat5),
    ]

    ref = None
    for name, (f_ctx, o_md), kw, w_m in candidates:
        m_map = {}
        for f in tk.FOLDS:
            fd = f_ctx[f]
            md_f = dict(fd["mixed"])
            if w_m is not None:
                md_f["p0"] = pool_custom(md_f["raw_probs"], md_f["flip_probs"], md_f["post_cfg"], w_m)
            m_map[f] = tk.run_pipeline(md_f, fd["queries"], fd["w"], fd["hp"], **kw)

        md_o = dict(o_md)
        if w_m is not None:
            md_o["p0"] = pool_custom(md_o["raw_probs"], md_o["flip_probs"], md_o["post_cfg"], w_m)
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
            ref = (om_a, om_p, om_e, mm_a, mm_p, mm_e, sim_pub, r_mev["mADM"], r_syn["mADM"], rm["PA100k"], rm["PETA"], rm["Market1501"])
        dom_a = (om_a - ref[0]) * 10000
        dom_p = (om_p - ref[1]) * 10000
        dom_e = (om_e - ref[2]) * 10000
        dmm_a = (mm_a - ref[3]) * 10000
        dmm_p = (mm_p - ref[4]) * 10000
        dmm_e = (mm_e - ref[5]) * 10000
        dsim = (sim_pub - ref[6]) * 10000
        dmev = (r_mev["mADM"] - ref[7]) * 10000
        dsyn = (r_syn["mADM"] - ref[8]) * 10000
        d_pa = (rm["PA100k"] - ref[9]) * 10000
        d_pe = (rm["PETA"] - ref[10]) * 10000
        d_mk = (rm["Market1501"] - ref[11]) * 10000

        pred_pub_oof = 0.699485 + (0.0804 * dom_p + 0.1938 * dom_e) / 10000.0
        pred_pub_mix = 0.699485 + (0.0725 * dmm_p + 0.2193 * dmm_e) / 10000.0
        d_pred_oof = (pred_pub_oof - 0.699485) * 10000
        d_pred_mix = (pred_pub_mix - 0.699485) * 10000

        print(
            f"{name:34s} | OOF:{om_a:.6f}({dom_a:+6.2f}) [dP:{dom_p:+6.2f}, dE:{dom_e:+6.2f}] | "
            f"Mix:{mm_a:.6f}({dmm_a:+6.2f}) [dP:{dmm_p:+6.2f}, dE:{dmm_e:+6.2f}] (PA:{d_pa:+5.1f},PE:{d_pe:+5.1f},MK:{d_mk:+5.1f}) | "
            f"dSim:{dsim:+5.2f}(M:{dmev:+5.2f},S:{dsyn:+5.2f}) | "
            f"Pred(O/M): {pred_pub_oof:.6f}({d_pred_oof:+5.2f}) / {pred_pub_mix:.6f}({d_pred_mix:+5.2f})",
            flush=True,
        )


if __name__ == "__main__":
    main()

