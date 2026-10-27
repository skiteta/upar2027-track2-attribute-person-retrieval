from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"))

import test_tri_graph_combined_knobs as tk
from test_hp6_224_stack import build_variant_contexts, pool_custom
from test_hp6_224_final_comb import build_anat_moe
from calibrate_public_mevid_synth import (
    build_embedded_synth_and_mevid_contexts,
    eval_sliced_ctx,
    eval_synth_ctxs,
)
from upar.data import load_retrieval_split
from upar.postproc import apply_post
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC


def compute_eva_probs(
    ks: Path,
    feat_dir: Path,
    folds_base: dict,
    w252: float = 0.50,
    tau: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    w224 = 1.0 - w252
    p_oof = np.empty((33407, 40), dtype=np.float32)
    p_by_fold = {}
    for fold in tk.FOLDS:
        hp = folds_base[fold]["hp"]
        sd224 = load_file(str(ks / f"eva02L224_{fold}_ema0.999/model.safetensors"))
        sd252 = load_file(str(ks / f"eva02L252_{fold}_ema0.999/model.safetensors"))
        W1, B1 = sd224["head.weight"].float(), sd224["head.bias"].float()
        W2, B2 = sd252["head.weight"].float(), sd252["head.bias"].float()
        W = w224 * W1 + w252 * W2
        B = w224 * B1 + w252 * B2
        if tau != 0.0:
            mn = w224 * W1.norm(dim=-1, keepdim=True) + w252 * W2.norm(dim=-1, keepdim=True)
            sn = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            W = W * ((mn / sn) ** tau)
        if scale != 1.0:
            W = W * scale
        pw = np.load(ks / f"eva02L252_{fold}_ema0.999/val_logits.npz")["pos_weight"]
        z = np.load(feat_dir / f"eva_full_val_{fold}.npz")
        feats = torch.from_numpy(z["features"]).bfloat16()
        logits = torch.nn.functional.linear(feats, W.bfloat16(), B.bfloat16()).float().numpy() - np.log(pw).astype(np.float32)
        probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        p_by_fold[fold] = probs
        p_oof[hp] = probs[hp]
    return p_oof, p_by_fold


def main() -> None:
    feat_dir = HERE / "extracted_features_v1"
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    plan = json.loads((ROOT / "artifacts/runs/20261017T1618+0900_mixed_gallery_context/protocol.json").read_text())
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    spec_obj = read_json(SPEC)
    train_proto = read_json(ROOT / "artifacts/runs/20261017T1105+0900_cycle_continuation/frontier/sourcefit_edge_v2/protocol.json")
    post_cfg = spec_obj["lodo"]["post"]

    qi_oof = np.unique(split.gallery_query_ids)
    queries_oof = split.queries[qi_oof].astype(np.float32)
    oof_ctx = tk.precompute_eval_ctx(split, qi_oof)
    synth_ctxs, mevid_ctx = build_embedded_synth_and_mevid_contexts(split, qi_oof, n_seeds=5)

    folds_base, oof_base = build_variant_contexts("base_192", None, False, feat_dir, split, plan, spec_obj, train_proto)
    w_oof = np.mean([folds_base[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)
    w_anat10 = build_anat_moe([0.425, 0.340, 0.235], [0.420, 0.345, 0.235], [0.370, 0.395, 0.235], [0.355, 0.410, 0.235])

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
    md_oof_2030 = tk.precompute_fold_mode(
        u_pe_oof, u_eva_oof, uhp_2030, oof_base["p0"], oof_base["raw_probs"], oof_base["flip_probs"], post_cfg, oof_base["m_std"]
    )
    md_mix_2030 = {}
    for fold in tk.FOLDS:
        bf = folds_base[fold]
        bm = bf["mixed"]
        md_mix_2030[fold] = tk.precompute_fold_mode(
            bf["u_pe_m"], bf["u_eva_m"], uhp_fold_2030[fold], bm["p0"], bf["raw_base"], bf["flip_base"], post_cfg, bm["m_std"]
        )

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

    z3 = np.load(feat_dir / "spherical_head_oof_v3.npz")

    def eval_cand(p0_in: np.ndarray, p1_in: np.ndarray, p2_oof: np.ndarray | None, p2_fold: dict | None, use_2030_feat: bool, kw: dict, w_m: np.ndarray | None):
        p2_o = p2_oof if p2_oof is not None else oof_base["raw_probs"][2]
        raw_oof = [p0_in, p1_in, p2_o]
        flip_oof = [p0_in, p1_in, p2_o if p2_oof is not None else oof_base["flip_probs"][2]]
        post_probs = [apply_post(r, post_cfg) for r in raw_oof]
        m_std_oof = np.std(np.stack(post_probs, axis=0), axis=0).astype(np.float32)
        p_oof = pool_custom(raw_oof, flip_oof, post_cfg, w_m) if w_m is not None else tk.pool_members(raw_oof, post_cfg)
        md_o = dict(md_oof_2030 if use_2030_feat else md_oof_uni)
        md_o.update(p0=p_oof, raw_probs=raw_oof, flip_probs=flip_oof, m_std=m_std_oof)
        oof_dist = tk.run_pipeline(md_o, queries_oof, w_oof, None, **kw)
        ro = tk.eval_with_ctx(oof_dist, oof_ctx)
        r_mev = eval_sliced_ctx(oof_dist, mevid_ctx)
        r_syn = eval_synth_ctxs(oof_dist, synth_ctxs)
        r_sp = {k: 0.5 * (r_mev[k] + r_syn[k]) for k in r_mev}

        mix_by_fold = {}
        for fold in tk.FOLDS:
            bf = folds_base[fold]
            hp = bf["hp"]
            r_f = [r.copy() for r in bf["raw_base"]]
            fl_f = [r.copy() for r in bf["flip_base"]]
            r_f[0][hp] = p0_in[hp]
            r_f[1][hp] = p1_in[hp]
            fl_f[0][hp] = p0_in[hp]
            fl_f[1][hp] = p1_in[hp]
            if p2_fold is not None:
                r_f[2] = p2_fold[fold].copy()
                fl_f[2] = p2_fold[fold].copy()
            pp_f = [apply_post(r, post_cfg) for r in r_f]
            ms_f = np.std(np.stack(pp_f, axis=0), axis=0).astype(np.float32)
            p_f = pool_custom(r_f, fl_f, post_cfg, w_m) if w_m is not None else tk.pool_members(r_f, post_cfg)
            md_f = dict(md_mix_2030[fold] if use_2030_feat else bf["mixed"])
            md_f.update(p0=p_f, raw_probs=r_f, flip_probs=fl_f, m_std=ms_f)
            d_f = tk.run_pipeline(md_f, bf["queries"], bf["w"], bf["hp"], **kw)
            mix_by_fold[fold] = tk.eval_with_ctx(d_f, bf["eval_ctx"])
        torch.cuda.empty_cache()
        return ro, mix_by_fold, r_sp

    ro_131, mx_131, sp_131 = eval_cand(oof_base["raw_probs"][0], oof_base["raw_probs"][1], None, None, False, cfg_b131c, None)
    mx_adm_131 = float(np.mean([mx_131[f]["mADM"] for f in tk.FOLDS]))
    mx_r1_131 = float(np.mean([mx_131[f]["R1"] for f in tk.FOLDS]))

    p2_w40_t0, p2f_w40_t0 = compute_eva_probs(ks, feat_dir, folds_base, w252=0.40, tau=0.0, scale=1.0)
    p2_w50_t1, p2f_w50_t1 = compute_eva_probs(ks, feat_dir, folds_base, w252=0.50, tau=1.0, scale=1.0)
    p2_w50_s97, p2f_w50_s97 = compute_eva_probs(ks, feat_dir, folds_base, w252=0.50, tau=0.0, scale=0.97)

    candidates = [
        ("00_B136C (Pub 0.701494)", z3["pe_b136c"], z3["hp_b136c"], None, None),
        ("01_pe_qk115_b060 + hp_b136c", z3["pe_qk115_o100_b060"], z3["hp_b136c"], None, None),
        ("02_pe_qk115_b060 + hp6_10_b100", z3["pe_qk115_o100_b060"], z3["hp6_10_qk100_o103_b100"], None, None),
        ("03_pe_qk115_b060 + hp6_10_b075", z3["pe_qk115_o100_b060"], z3["hp6_10_qk110_o103_b075"], None, None),
        ("04_pe_qk120_b050 + hp6_10_b100", z3["pe_qk120_o100_b050"], z3["hp6_10_qk100_o103_b100"], None, None),
        ("05_pe_qk120_b060 + hp6_10_b100", z3["pe_qk120_o100_b060"], z3["hp6_10_qk100_o103_b100"], None, None),
        ("06_pe_qk115_b050 + hp_b136c + eva_w40", z3["pe_qk115_o100_b050"], z3["hp_b136c"], p2_w40_t0, p2f_w40_t0),
        ("07_pe_qk115_b050 + hp_b136c + eva_t10", z3["pe_qk115_o100_b050"], z3["hp_b136c"], p2_w50_t1, p2f_w50_t1),
        ("08_pe_qk115_b050 + hp_b136c + eva_s97", z3["pe_qk115_o100_b050"], z3["hp_b136c"], p2_w50_s97, p2f_w50_s97),
    ]

    print(
        f"{'Candidate':40s} | {'dOOF':>6s} {'dAP':>6s} {'dR1':>6s} {'dR5':>6s} {'dR10':>6s} | "
        f"{'dMix':>6s} ({'PA':>4s}/{'PE':>4s}/{'Mk':>4s}) {'dMxR1':>6s} | {'dSimP':>6s} {'dSpAP':>6s} {'dSpR1':>6s}",
        flush=True,
    )
    print("-" * 149, flush=True)
    for cname, p0, p1, p2o, p2f in candidates:
        ro, mx, rsp = eval_cand(p0, p1, p2o, p2f, True, cfg_b06, w_anat10)
        mx_adm = float(np.mean([mx[f]["mADM"] for f in tk.FOLDS]))
        mx_r1 = float(np.mean([mx[f]["R1"] for f in tk.FOLDS]))
        dpa = (mx["PA100k"]["mADM"] - mx_131["PA100k"]["mADM"]) * 1e4
        dpe = (mx["PETA"]["mADM"] - mx_131["PETA"]["mADM"]) * 1e4
        dmk = (mx["Market1501"]["mADM"] - mx_131["Market1501"]["mADM"]) * 1e4
        print(
            f"{cname:40s} | {(ro['mADM']-ro_131['mADM'])*1e4:+6.2f} {(ro['mAP']-ro_131['mAP'])*1e4:+6.2f} "
            f"{(ro['R1']-ro_131['R1'])*1e4:+6.2f} {(ro['R5']-ro_131['R5'])*1e4:+6.2f} {(ro['R10']-ro_131['R10'])*1e4:+6.2f} | "
            f"{(mx_adm-mx_adm_131)*1e4:+6.2f} ({dpa:+4.1f}/{dpe:+4.1f}/{dmk:+4.1f}) {(mx_r1-mx_r1_131)*1e4:+6.2f} | "
            f"{(rsp['mADM']-sp_131['mADM'])*1e4:+6.2f} {(rsp['mAP']-sp_131['mAP'])*1e4:+6.2f} {(rsp['R1']-sp_131['R1'])*1e4:+6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

