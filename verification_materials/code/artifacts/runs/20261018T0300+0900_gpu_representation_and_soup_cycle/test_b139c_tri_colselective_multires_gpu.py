from __future__ import annotations

import json
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


def make_col_blend(base_p: np.ndarray, col_sources: list[tuple[list[int], np.ndarray]]) -> np.ndarray:
    out = base_p.copy()
    for cols, src in col_sources:
        out[:, cols] = src[:, cols]
    return out


def compute_eva_colselective_probs(
    ks: Path,
    feat_dir: Path,
    folds_base: dict,
    w252_default: float = 0.50,
    col_w252: dict[int, float] | None = None,
    tau: float = 1.2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    w252_vec = torch.full((40, 1), float(w252_default), dtype=torch.float32)
    if col_w252:
        for c, wv in col_w252.items():
            w252_vec[c, 0] = float(wv)
    w224_vec = 1.0 - w252_vec

    p_oof = np.empty((33407, 40), dtype=np.float32)
    p_by_fold = {}
    for fold in tk.FOLDS:
        hp = folds_base[fold]["hp"]
        sd224 = load_file(str(ks / f"eva02L224_{fold}_ema0.999/model.safetensors"))
        sd252 = load_file(str(ks / f"eva02L252_{fold}_ema0.999/model.safetensors"))
        W1, B1 = sd224["head.weight"].float(), sd224["head.bias"].float()
        W2, B2 = sd252["head.weight"].float(), sd252["head.bias"].float()
        W = w224_vec * W1 + w252_vec * W2
        B = w224_vec[:, 0] * B1 + w252_vec[:, 0] * B2
        if tau != 0.0:
            mn = w224_vec * W1.norm(dim=-1, keepdim=True) + w252_vec * W2.norm(dim=-1, keepdim=True)
            sn = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            W = W * ((mn / sn) ** float(tau))
        pw = np.load(ks / f"eva02L252_{fold}_ema0.999/val_logits.npz")["pos_weight"]
        z = np.load(feat_dir / f"eva_full_val_{fold}.npz")
        feats = torch.from_numpy(z["features"]).bfloat16()
        logits = torch.nn.functional.linear(feats, W.bfloat16(), B.bfloat16()).float().numpy() - np.log(pw).astype(np.float32)
        probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        p_by_fold[fold] = probs
        p_oof[hp] = probs[hp]
    return p_oof, p_by_fold


def main() -> None:
    t0 = time.monotonic()
    feat_dir = HERE / "extracted_features_v1"
    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")

    z3 = dict(np.load(feat_dir / "spherical_head_oof_v3.npz"))
    z4 = dict(np.load(feat_dir / "spherical_head_oof_v4.npz"))
    z4.update(dict(np.load(feat_dir / "spherical_head_oof_v5.npz")))

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
    cfg_mid = {**cfg_b06, "ham_tau": 0.172, "margin_gamma": 1.83}

    p2_base, p2f_base = compute_eva_probs(ks, feat_dir, folds_base, w252=0.50, tau=1.2, scale=1.0)

    def eval_all(p0_in: np.ndarray, p1_in: np.ndarray, p2_oof: np.ndarray | None, p2_fold: dict | None, use_2030_feat: bool, kw: dict, w_m: np.ndarray | None):
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
        mix_res = {}
        for fold in tk.FOLDS:
            bf = folds_base[fold]
            hp = bf["hp"]
            r_f = [bf["raw_base"][0].copy(), bf["raw_base"][1].copy(), p2_fold[fold].copy() if p2_fold is not None else bf["raw_base"][2].copy()]
            fl_f = [bf["flip_base"][0].copy(), bf["flip_base"][1].copy(), p2_fold[fold].copy() if p2_fold is not None else bf["flip_base"][2].copy()]
            r_f[0][hp] = p0_in[hp]
            fl_f[0][hp] = p0_in[hp]
            r_f[1][hp] = p1_in[hp]
            fl_f[1][hp] = p1_in[hp]
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
        return ro, mix_res, sim_pub

    ro_b131, rm_b131, rsp_b131 = eval_all(oof_base["raw_probs"][0], oof_base["raw_probs"][1], None, None, False, cfg_b131c, None)
    b131_oof = ro_b131["mADM"]
    b131_mix = float(np.mean([rm_b131[f]["mADM"] for f in tk.FOLDS]))
    b131_sp = rsp_b131["mADM"]

    # Base B138C components
    BODY_12 = [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
    DICH_3 = [3, 7, 20]
    FG_UNMASK_4 = [4, 23, 35, 36]
    BAG_2 = [35, 36]

    pe_base = z4["pe_b137c"]
    pe_unm20 = z4["pe_unmask24_m20_qk120_b060"]
    pe_unm15 = z4["pe_unmask24_m15_qk120_b060"]

    hp_base = z4["hp_b137c"]
    hp_r00 = z3["hp_b136c"]
    hp_r15 = z3["hp6_15_qk100_o103_b100"]
    hp_m15_r10 = z4["hp_mask15_m15_r10_o103"]
    hp_m15_r00 = z4["hp_mask15_m15_r00_o103"]

    hp_b138c = make_col_blend(hp_base, [(BODY_12, hp_m15_r10)])

    # Axis 1: DINOv3-H+ column-selective r224 resolution routing
    hp_dich3_r00 = make_col_blend(hp_b138c, [(DICH_3, hp_r00)])
    hp_dich3_c34_r00 = make_col_blend(hp_b138c, [(DICH_3, hp_r00), ([34], hp_m15_r00)])
    hp_c20_r00 = make_col_blend(hp_b138c, [([20], hp_r00)])
    hp_c7_c20_r00 = make_col_blend(hp_b138c, [([7, 20], hp_r00)])
    hp_fg4_r15 = make_col_blend(hp_b138c, [(FG_UNMASK_4, hp_r15)])
    hp_bag2_r15 = make_col_blend(hp_b138c, [(BAG_2, hp_r15)])
    hp_dich3_r00_fg4_r15 = make_col_blend(hp_b138c, [(DICH_3, hp_r00), (FG_UNMASK_4, hp_r15)])
    hp_dich3_c34_r00_fg4_r15 = make_col_blend(hp_b138c, [(DICH_3, hp_r00), ([34], hp_m15_r00), (FG_UNMASK_4, hp_r15)])

    # Axis 2: EVA02-L14 column-selective w252 resolution routing
    # Columns where 224 beat 252 across 3/3 folds: [2, 7, 8, 11, 25, 27, 35, 36]
    # Columns where 252 beat 224 across 3/3 folds: [9, 26]
    EVA_224_WINS8 = [2, 7, 8, 11, 25, 27, 35, 36]
    EVA_252_WINS2 = [9, 26]
    p2_bag35, p2f_bag35 = compute_eva_colselective_probs(ks, feat_dir, folds_base, 0.50, {35: 0.35, 36: 0.35}, 1.2)
    p2_bag25, p2f_bag25 = compute_eva_colselective_probs(ks, feat_dir, folds_base, 0.50, {35: 0.25, 36: 0.25}, 1.2)
    ov_eva_8_2 = {c: 0.35 for c in EVA_224_WINS8}
    ov_eva_8_2.update({c: 0.65 for c in EVA_252_WINS2})
    p2_colsel35, p2f_colsel35 = compute_eva_colselective_probs(ks, feat_dir, folds_base, 0.50, ov_eva_8_2, 1.2)

    # Axis 3: PE-Core-L14 column-selective maskv2 reduction on 3/3 winning unmasked subsets
    PE_DICH_BAG4 = [2, 3, 36, 38]
    PE_UNM_COLORS8 = [8, 12, 17, 21, 22, 23, 25, 30]
    PE_WIN12 = PE_DICH_BAG4 + PE_UNM_COLORS8
    pe_db4_m20 = make_col_blend(pe_base, [(PE_DICH_BAG4, pe_unm20)])
    pe_uc8_m20 = make_col_blend(pe_base, [(PE_UNM_COLORS8, pe_unm20)])
    pe_win12_m20 = make_col_blend(pe_base, [(PE_WIN12, pe_unm20)])
    pe_win12_m15 = make_col_blend(pe_base, [(PE_WIN12, pe_unm15)])

    cands = [
        ("00_B137C (baseline)", pe_base, hp_base, p2_base, p2f_base, cfg_b06),
        ("01_B138C (champion baseline)", pe_base, hp_b138c, p2_base, p2f_base, cfg_mid),
        ("02_B138C + hp_dich3_r00", pe_base, hp_dich3_r00, p2_base, p2f_base, cfg_mid),
        ("03_B138C + hp_dich3_c34_r00", pe_base, hp_dich3_c34_r00, p2_base, p2f_base, cfg_mid),
        ("04_B138C + hp_c20_r00", pe_base, hp_c20_r00, p2_base, p2f_base, cfg_mid),
        ("05_B138C + hp_c7_c20_r00", pe_base, hp_c7_c20_r00, p2_base, p2f_base, cfg_mid),
        ("06_B138C + hp_fg4_r15", pe_base, hp_fg4_r15, p2_base, p2f_base, cfg_mid),
        ("07_B138C + hp_bag2_r15", pe_base, hp_bag2_r15, p2_base, p2f_base, cfg_mid),
        ("08_B138C + hp_dich3_r00_fg4_r15", pe_base, hp_dich3_r00_fg4_r15, p2_base, p2f_base, cfg_mid),
        ("09_B138C + hp_dich3_c34_r00_fg4_r15", pe_base, hp_dich3_c34_r00_fg4_r15, p2_base, p2f_base, cfg_mid),
        ("10_B138C + eva_bag35", pe_base, hp_b138c, p2_bag35, p2f_bag35, cfg_mid),
        ("11_B138C + eva_bag25", pe_base, hp_b138c, p2_bag25, p2f_bag25, cfg_mid),
        ("12_B138C + eva_colsel35", pe_base, hp_b138c, p2_colsel35, p2f_colsel35, cfg_mid),
        ("13_B138C + pe_db4_m20", pe_db4_m20, hp_b138c, p2_base, p2f_base, cfg_mid),
        ("14_B138C + pe_uc8_m20", pe_uc8_m20, hp_b138c, p2_base, p2f_base, cfg_mid),
        ("15_B138C + pe_win12_m20", pe_win12_m20, hp_b138c, p2_base, p2f_base, cfg_mid),
        ("16_B138C + pe_win12_m15", pe_win12_m15, hp_b138c, p2_base, p2f_base, cfg_mid),
    ]

    print(f"\n{'Candidate':40s} | {'dOOF':>6s} {'dAP':>6s} {'dR1':>6s} {'dR5':>6s} {'dR10':>6s} | {'dMix':>6s} ({'PA':>4s}/{'PE':>4s}/{'Mk':>4s}) {'dMxR1':>6s} | {'dSimP':>6s} {'dSpAP':>6s} {'dSpR1':>6s}")
    print("-" * 135)
    for name, p0_c, p1_c, p2_c, p2f_c, kw_c in cands:
        ro, rm, rsp = eval_all(p0_c, p1_c, p2_c, p2f_c, True, kw_c, w_anat10)
        d_oof = (ro["mADM"] - b131_oof) * 10000.0
        d_ap = (ro["mAP"] - ro_b131["mAP"]) * 10000.0
        d_r1 = (ro["R1"] - ro_b131["R1"]) * 10000.0
        d_r5 = (ro["R5"] - ro_b131["R5"]) * 10000.0
        d_r10 = (ro["R10"] - ro_b131["R10"]) * 10000.0
        m_avg = float(np.mean([rm[f]["mADM"] for f in tk.FOLDS]))
        d_mix = (m_avg - b131_mix) * 10000.0
        d_pa = (rm["PA100k"]["mADM"] - rm_b131["PA100k"]["mADM"]) * 10000.0
        d_pe = (rm["PETA"]["mADM"] - rm_b131["PETA"]["mADM"]) * 10000.0
        d_mk = (rm["Market1501"]["mADM"] - rm_b131["Market1501"]["mADM"]) * 10000.0
        d_mx_r1 = (float(np.mean([rm[f]["R1"] for f in tk.FOLDS])) - float(np.mean([rm_b131[f]["R1"] for f in tk.FOLDS]))) * 10000.0
        d_sp = (rsp["mADM"] - b131_sp) * 10000.0
        d_sp_ap = (rsp["mAP"] - rsp_b131["mAP"]) * 10000.0
        d_sp_r1 = (rsp["R1"] - rsp_b131["R1"]) * 10000.0
        print(
            f"{name:40s} | {d_oof:+6.2f} {d_ap:+6.2f} {d_r1:+6.2f} {d_r5:+6.2f} {d_r10:+6.2f} | "
            f"{d_mix:+6.2f} ({d_pa:+4.1f}/{d_pe:+4.1f}/{d_mk:+4.1f}) {d_mx_r1:+6.2f} | "
            f"{d_sp:+6.2f} {d_sp_ap:+6.2f} {d_sp_r1:+6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
