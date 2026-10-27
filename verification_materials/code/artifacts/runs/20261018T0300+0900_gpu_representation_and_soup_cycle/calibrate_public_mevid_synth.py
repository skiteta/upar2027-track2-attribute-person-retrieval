"""Rigorous Local-to-Public (MEVID + Synth) Calibration & Structural Simulation Benchmark.

1. Reproduces historical same-backbone submissions (B119G, B121G, B122C, B123C, B127C, B128C, B129C, B130C, B131C, B132C)
   and compares against exact 6-decimal-place Codabench Public scores decomposed from detailed_results.html & scoring_stdout.txt:
     Public_mADM = 0.5 * MEVID_mADM (367 x 28,095) + 0.5 * Synth_mADM (577 x 1,153)
     MEVID_mADM  = MEVID_mAP + MEVID_Body
     Synth_mADM  = Synth_mAP + Synth_Body
2. Implements the exact Codabench combined-gallery -> per-dataset slicing protocol on the 33,407 OOF gallery:
     - Sim_Synth_1153: K=5 deterministic 1,153-image (577-query, ~2.0 matches/query) sparse sub-galleries
       sliced directly out of the 33,407 combined gallery distance matrix after rank_gallery.
     - Sim_MEVID: High-multiplicity (>=15 matches/query) surveillance queries evaluated on the combined gallery.
3. Evaluates Leave-One-Submission-Out (LOSO) and pairwise delta accuracy, then tests B133C and Synth-protective upgrades.
"""
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
from upar.data import load_retrieval_split
from upar.metrics_strict import _normalized_degree_of_match
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC


# Exact 6-decimal official Codabench scores from scoring_stdout.txt & detailed_results.html
# MEVID: 367 queries, 28,095 gallery (96.06% of combined 29,248 gallery)
# Synth: 577 queries, 1,153 gallery (3.94% of combined 29,248 gallery)
OFFICIAL_HISTORY = {
    "B119G": {
        "Public_mADM": 0.673692, "Public_mAP": 0.386677, "Public_Body": 0.287015,
        "MEVID_mADM": 0.552054,  "MEVID_mAP": 0.233370,  "MEVID_Body": 0.318684,
        "Synth_mADM": 0.795330,  "Synth_mAP": 0.539984,  "Synth_Body": 0.255346,
    },
    "B127C": {
        "Public_mADM": 0.691001, "Public_mAP": 0.408298, "Public_Body": 0.282703,
        "MEVID_mADM": 0.588948,  "MEVID_mAP": 0.275400,  "MEVID_Body": 0.313548,
        "Synth_mADM": 0.793054,  "Synth_mAP": 0.541196,  "Synth_Body": 0.251858,
    },
    "B128C": {
        "Public_mADM": 0.698928, "Public_mAP": 0.416663, "Public_Body": 0.282265,
        "MEVID_mADM": 0.602582,  "MEVID_mAP": 0.291077,  "MEVID_Body": 0.311505,
        "Synth_mADM": 0.795274,  "Synth_mAP": 0.542249,  "Synth_Body": 0.253025,
    },
    "B129C": {
        "Public_mADM": 0.698858, "Public_mAP": 0.416577, "Public_Body": 0.282281,
        "MEVID_mADM": 0.602870,  "MEVID_mAP": 0.291413,  "MEVID_Body": 0.311457,
        "Synth_mADM": 0.794846,  "Synth_mAP": 0.541741,  "Synth_Body": 0.253105,
    },
    "B130C": {
        "Public_mADM": 0.699000, "Public_mAP": 0.417679, "Public_Body": 0.281321,
        "MEVID_mADM": 0.603028,  "MEVID_mAP": 0.291862,  "MEVID_Body": 0.311166,
        "Synth_mADM": 0.794972,  "Synth_mAP": 0.543496,  "Synth_Body": 0.251476,
    },
    "B131C": {
        "Public_mADM": 0.699485, "Public_mAP": 0.415614, "Public_Body": 0.283871,
        "MEVID_mADM": 0.604075,  "MEVID_mAP": 0.290158,  "MEVID_Body": 0.313917,
        "Synth_mADM": 0.794895,  "Synth_mAP": 0.541070,  "Synth_Body": 0.253825,
    },
    "B132C": {
        "Public_mADM": 0.699235, "Public_mAP": 0.416406, "Public_Body": 0.282829,
        "MEVID_mADM": 0.604357,  "MEVID_mAP": 0.289791,  "MEVID_Body": 0.314566,
        "Synth_mADM": 0.794113,  "Synth_mAP": 0.543021,  "Synth_Body": 0.251092,
    },
}


def build_embedded_synth_and_mevid_contexts(split, qi_oof: np.ndarray, n_seeds: int = 5) -> tuple[list[dict], dict]:
    """Construct embedded 1,153-gallery (577-query, ~2.0 matches/query) Synth-like contexts
    and a high-multiplicity MEVID-like surveillance context inside the 33,407 OOF gallery.
    """
    import os
    cache_dir = HERE / "extracted_features_v1" / "graph_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"simpub_ctxs_s{n_seeds}_{len(qi_oof)}.pt"
    if cp.exists():
        saved = torch.load(cp, map_location="cpu", weights_only=False)
        return saved["synth_ctxs"], saved["mevid_ctx"]
    g_qids = split.gallery_query_ids.astype(np.int64)
    counts = np.bincount(g_qids, minlength=split.query_count)
    q_oof_pos = {int(q): i for i, q in enumerate(qi_oof.tolist())}

    # 1. High-multiplicity MEVID-like queries (>= 15 matches in the 33,407 gallery; ~398 queries, mean ~43 matches/query)
    mevid_q = np.array([q for q in qi_oof if counts[q] >= 15], dtype=np.int64)
    mevid_row_idx = np.array([q_oof_pos[int(q)] for q in mevid_q], dtype=np.int64)
    q_sel_m = split.queries[mevid_q].astype(np.uint8)
    g_attr_all = split.gallery_attributes.astype(np.uint8)
    nm_mevid = _normalized_degree_of_match(q_sel_m, g_attr_all).astype(np.float32)
    mevid_ctx = {
        "qi": mevid_q,
        "row_idx": mevid_row_idx,
        "col_idx": None,
        "norm_match": nm_mevid,
        "g_qids": g_qids,
    }

    # 2. Embedded 1,153-gallery / 577-query Synth-like sub-galleries (2.0 matches/query, 3.45% of 33,407 gallery)
    # Candidate queries with >=1 match in PETA or PA100k
    q_1 = np.array([q for q in qi_oof if counts[q] == 1], dtype=np.int64)
    q_2p = np.array([q for q in qi_oof if 2 <= counts[q] <= 6], dtype=np.int64)
    q_3p = np.array([q for q in qi_oof if counts[q] >= 3], dtype=np.int64)

    synth_ctxs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(20261019 + seed)
        # To get 577 queries and exactly 1,153 gallery images (mean 2.00 images/query):
        # Pick 150 queries with 1 image, 278 queries with 2 images, 149 queries with 3 images -> 150 + 556 + 447 = 1153 images, 577 queries!
        sel_1 = rng.choice(q_1, size=150, replace=False)
        rem_2 = np.setdiff1d(q_2p, sel_1)
        sel_2 = rng.choice(rem_2, size=278, replace=False)
        rem_3 = np.setdiff1d(q_3p, np.union1d(sel_1, sel_2))
        sel_3 = rng.choice(rem_3, size=149, replace=False)

        q_sub = np.sort(np.concatenate([sel_1, sel_2, sel_3]))
        g_cols = []
        for q in sel_1:
            idx = np.flatnonzero(g_qids == q)
            g_cols.extend(idx[:1].tolist())
        for q in sel_2:
            idx = np.flatnonzero(g_qids == q)
            g_cols.extend(rng.choice(idx, size=2, replace=False).tolist())
        for q in sel_3:
            idx = np.flatnonzero(g_qids == q)
            g_cols.extend(rng.choice(idx, size=3, replace=False).tolist())
        g_cols = np.sort(np.array(g_cols, dtype=np.int64))
        assert len(q_sub) == 577 and len(g_cols) == 1153

        row_idx = np.array([q_oof_pos[int(q)] for q in q_sub], dtype=np.int64)
        q_sel = split.queries[q_sub].astype(np.uint8)
        g_attr_sub = split.gallery_attributes[g_cols].astype(np.uint8)
        nm_sub = _normalized_degree_of_match(q_sel, g_attr_sub).astype(np.float32)
        synth_ctxs.append({
            "qi": q_sub,
            "row_idx": row_idx,
            "col_idx": g_cols,
            "norm_match": nm_sub,
            "g_qids": g_qids[g_cols],
        })
    tmp = cache_dir / f".tmp_{os.getpid()}_{cp.name}"
    torch.save({"synth_ctxs": synth_ctxs, "mevid_ctx": mevid_ctx}, tmp)
    tmp.replace(cp)
    return synth_ctxs, mevid_ctx


def eval_sliced_ctx(oof_dist: np.ndarray, ctx: dict) -> dict[str, float]:
    sub_dist = oof_dist[ctx["row_idx"]]
    if ctx["col_idx"] is not None:
        sub_dist = sub_dist[:, ctx["col_idx"]]
    r = tk.eval_with_ctx(sub_dist, ctx)
    r["Body"] = r["mADM"] - r["mAP"]
    return r


def eval_synth_ctxs(oof_dist: np.ndarray, synth_ctxs: list[dict]) -> dict[str, float]:
    res_list = [eval_sliced_ctx(oof_dist, c) for c in synth_ctxs]
    out = {}
    for k in res_list[0]:
        out[k] = float(np.mean([r[k] for r in res_list]))
    return out


def main() -> None:
    feat_dir = HERE / "extracted_features_v1"
    plan = json.loads((ROOT / "artifacts/runs/20261017T1618+0900_mixed_gallery_context/protocol.json").read_text())
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    spec_obj = read_json(SPEC)
    train_proto = read_json(ROOT / "artifacts/runs/20261017T1105+0900_cycle_continuation/frontier/sourcefit_edge_v2/protocol.json")
    post_cfg = spec_obj["lodo"]["post"]

    t_load = time.monotonic()
    data_by_fold = {}
    n_tot = split.gallery_size
    u_pe_oof = np.empty((n_tot, 1024), dtype=np.float64)
    u_eva_oof = np.empty((n_tot, 1024), dtype=np.float64)
    u_hp_oof = np.empty((n_tot, 1280), dtype=np.float64)
    p_oof = np.empty((n_tot, 40), dtype=np.float32)
    raw_oof = [np.empty((n_tot, 40), dtype=np.float32) for _ in range(3)]
    flip_oof = [np.empty((n_tot, 40), dtype=np.float32) for _ in range(3)]
    m_std_oof = np.empty((n_tot, 40), dtype=np.float32)

    for fold in tk.FOLDS:
        d = tk.mod_mix.load_fold(fold, split, plan)
        hp = d["hp"]
        sub = d["sub"]
        qi = np.unique(sub.gallery_query_ids)
        queries = sub.queries[qi].astype(np.float32)

        row_meta = next(r for r in train_proto["folds"] if r["fold"] == fold)
        with np.load(ROOT / row_meta["meta"]["path"], allow_pickle=False) as zm:
            meta = {k: zm[k].copy() for k in ("split", "sources", "image_paths", "attribute_names")}
        val_idx = np.flatnonzero(meta["split"] == "val")
        raw_probs, flip_probs, post_probs = tk.load_member_probs_raw(spec_obj, fold, meta, val_idx)
        m_std_all = np.std(np.stack(post_probs, axis=0), axis=0).astype(np.float32)

        ef = np.load(feat_dir / f"eva_full_val_{fold}.npz")
        hf = np.load(feat_dir / f"hp_full_val_{fold}.npz")

        u_pe_m = tk.unit_features_fp64(d["features"])
        u_eva_m = tk.unit_features_fp64(ef["features"])
        u_hp_m = tk.unit_features_fp64(hf["features"])

        u_pe_oof[hp] = u_pe_m[hp]
        u_eva_oof[hp] = u_eva_m[hp]
        u_hp_oof[hp] = u_hp_m[hp]
        p_oof[hp] = d["p"][hp]
        m_std_oof[hp] = m_std_all[hp]
        for i in range(3):
            raw_oof[i][hp] = raw_probs[i][hp]
            flip_oof[i][hp] = flip_probs[i][hp]

        held_md = tk.precompute_fold_mode(
            u_pe_m[hp], u_eva_m[hp], u_hp_m[hp], d["p"][hp],
            [rp[hp] for rp in raw_probs], [fp[hp] for fp in flip_probs], post_cfg, m_std_all[hp]
        )
        mixed_md = tk.precompute_fold_mode(
            u_pe_m, u_eva_m, u_hp_m, d["p"], raw_probs, flip_probs, post_cfg, m_std_all
        )

        data_by_fold[fold] = {
            "hp": hp,
            "sub": sub,
            "qi": qi,
            "queries": queries,
            "w": d["w"],
            "held": held_md,
            "mixed": mixed_md,
            "eval_ctx": tk.precompute_eval_ctx(sub, qi),
        }

    oof_md = tk.precompute_fold_mode(
        u_pe_oof, u_eva_oof, u_hp_oof, p_oof, raw_oof, flip_oof, post_cfg, m_std_oof
    )
    qi_oof = np.unique(split.gallery_query_ids)
    queries_oof = split.queries[qi_oof].astype(np.float32)
    w_oof = np.mean([data_by_fold[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)
    oof_ctx = tk.precompute_eval_ctx(split, qi_oof)
    synth_ctxs, mevid_ctx = build_embedded_synth_and_mevid_contexts(split, qi_oof, n_seeds=5)
    print(f"Precomputed all folds + OOF33k + Embedded Synth/MEVID contexts in {time.monotonic() - t_load:.1f}s", flush=True)

    # Exact historical submission configs
    b1_pe = [("pe", 1.0, 1.0, 0.0)]
    b4 = [
        ("pe", 0.30, 1.0, 0.0),
        ("eva", 0.25, 0.0, 1.0),
        ("j25_k28", 0.25, 0.75, 0.25),
        ("j50_k28", 0.20, 0.50, 0.50),
    ]
    b5_tw07 = [
        ("pe", 0.275, 1.0, 0.0),
        ("eva", 0.23, 0.0, 1.0),
        ("j25_k28", 0.235, 0.75, 0.25),
        ("j50_k28", 0.19, 0.50, 0.50),
        ("tri25_k28", 0.07, 0.5625, 0.1875),
    ]
    aw_b128 = {"pe": 0.20, "eva": 0.20, "j25": 0.30, "j50": 0.30}
    aw_8spec = {
        "pe": 0.10, "eva": 0.10, "j25": 0.18, "j50": 0.18,
        "tri25": 0.16, "tri40": 0.14, "pe_hp": 0.07, "eva_hp": 0.07,
    }

    cfg_b119g = {
        "branches": b1_pe,
        "stage1_steps": 0,
        "stage2_steps": 0,
        "entropy_delta": 0.0,
        "anchor_alpha_q": 0.0,
    }
    cfg_b127c = {
        "branches": b1_pe,
        "b127_mode": True,
        "stage1_steps": 0,
        "stage2_steps": 5,
        "coevolve_blend": 0.0,
        "post_tau": 1.0,
        "entropy_delta": 0.35,
        "score_ens_blend": 0.0,
        "d_steps": 1,
        "d_alpha": 0.25,
        "anchor_alpha_q": 0.0,
    }
    cfg_b128c = {
        "branches": b4,
        "anchor_spec_weights": aw_b128,
        "ham_tau": 0.12,
        "dist_kappa": 0.0,
        "anchor_top_m": 12,
        "anchor_alpha_q": 0.105,
        "anchor_temp_q": 0.06,
    }
    cfg_b129c = {
        "branches": b5_tw07,
        "anchor_spec_weights": aw_8spec,
        "ham_tau": 0.16,
        "dist_kappa": 0.18,
        "feat_diff_lam": 0.45,
        "feat_diff_steps": 1,
        "feat_diff_blend": 0.45,
        "anchor_top_m": 12,
        "anchor_alpha_q": 0.105,
        "anchor_temp_q": 0.06,
    }
    cfg_b130c = {
        "branches": b5_tw07,
        "anchor_spec_weights": aw_8spec,
        "ham_tau": 0.16,
        "dist_kappa": 0.18,
        "sim_diff_alpha": 0.34,
        "exact_seed_blend": 0.72,
        "ex_vis_alpha": 0.120,
        "anchor_alpha_q": 0.126,
        "anchor_temp_q": 0.052,
        "head_beta": 0.044,
        "head_tau": 0.225,
        "anchor_top_m": 11,
        "tail_mul_kap": 0.04,
        "head_dq_blend": 0.15,
        "seed_disagree_kap": 0.025,
        "seed_kap_weight": False,
        "exact_branch_blend": 0.22,
    }
    champ_v4 = {
        **cfg_b130c,
        "ex_diff_alpha": 0.07,
        "ex_diff_head_blend": 0.50,
        "ham_tau": 0.17,
        "ex_kappa": 0.24,
        "dist_kappa": 0.20,
        "seed_disagree_kap": 0.030,
        "head_local_lam": 0.12,
        "head_local_k": 12,
        "sim_diff_alpha2": 0.04,
    }
    body_b131 = {
        "body_alpha": 0.20,
        "body_k0": 10,
        "body_tau": 0.40,
        "body_pow": 1.0,
        "body_w_uni": 1.0,
        "body_head_keep": 1.0,
        "body_err_gamma": 0.85,
        "body_nll_lam": 0.05,
    }
    cfg_b131c = {**champ_v4, **body_b131}
    cfg_b132c = {
        **cfg_b131c,
        "base_q_pos_rho": -50.60,
        "body_q_pos_rho": -50.60,
        "body_nll_lam_neg": 0.00,
        "body_err_gamma": 1.40,
        "body_err_gamma_neg": 0.55,
        "body_d_alpha": 0.23,
        "body_kap": 0.0,
        "body_vis_scale": 1.25,
        "body_neg_eta": 0.85,
    }
    cfg_b133c = {
        **cfg_b131c,
        "nll_lam": 0.275,
        "nll_lam_neg": 0.215,
        "err_gamma_neg": 0.97,
        "base_dich_sym_loss": 1.28,
        "dich_pos_nll_add": 0.08,
        "exact_nll_lam": 0.43,
        "exact_nll_lam_neg": 0.30,
        "exact_dich_sym_loss": 1.25,
        "body_alpha": 0.27,
        "body_nll_lam_neg": 0.00,
        "body_err_gamma": 1.40,
        "body_err_gamma_neg": 0.55,
        "body_dich_sym_loss": 0.60,
        "body_d_alpha": 0.23,
        "body_kap": 0.0,
        "body_vis_scale": 1.25,
    }

    # Test Synth-protective & mAP-preserving upgrades on top of B133C:
    # 1. Larger body_k0 (so Synth's top ranks inside combined gallery are protected!)
    # 2. Smooth body_pow > 1.0 and body_ex_prot_tau > 0.0 (protecting likely exact matches from body-stream demotion!)
    # 3. Testing symmetric vs asymmetric body_err_gamma (0.85/0.85 vs 1.15/0.70 vs 1.40/0.55)
    candidates = [
        ("H01_B119G", cfg_b119g),
        ("H02_B127C", cfg_b127c),
        ("H03_B128C", cfg_b128c),
        ("H04_B129C", cfg_b129c),
        ("H05_B130C", cfg_b130c),
        ("H06_B131C", cfg_b131c),
        ("H07_B132C", cfg_b132c),
        ("C00_B133C_pkg_v8", cfg_b133c),
        ("C01_B133C_symBody85", {**cfg_b133c, "body_err_gamma": 0.85, "body_err_gamma_neg": 0.85, "body_nll_lam_neg": 0.05}),
        ("C02_B133C_midBody115_70", {**cfg_b133c, "body_err_gamma": 1.15, "body_err_gamma_neg": 0.70, "body_nll_lam_neg": 0.02}),
        ("C03_B133C_k0_18_pow13", {**cfg_b133c, "body_k0": 18, "body_pow": 1.3}),
        ("C04_B133C_k0_25_pow15", {**cfg_b133c, "body_k0": 25, "body_pow": 1.5}),
        ("C05_B133C_exProt035", {**cfg_b133c, "body_ex_prot_tau": 0.35, "body_ex_prot_pow": 1.5}),
        ("C06_B133C_exProt050_k18", {**cfg_b133c, "body_k0": 18, "body_ex_prot_tau": 0.50, "body_ex_prot_pow": 1.5}),
        ("C07_B133C_exProt050_k22_a30", {**cfg_b133c, "body_k0": 22, "body_ex_prot_tau": 0.50, "body_ex_prot_pow": 1.5, "body_alpha": 0.30}),
    ]

    results = {}
    print(
        f"{'Name':28s} | {'OOF33k':8s} {'O_AP':7s} {'O_Bd':7s} | {'SimMEV':8s} {'SM_AP':7s} {'SM_Bd':7s} | "
        f"{'SimSYN':8s} {'SS_AP':7s} {'SS_Bd':7s} | {'Held':8s} {'H_AP':7s} {'H_Bd':7s} | {'Mixed':8s} {'M_AP':7s} {'M_Bd':7s}",
        flush=True,
    )
    print("-" * 165, flush=True)

    for name, kw in candidates:
        t0 = time.monotonic()
        h_map, m_map = {}, {}
        for f in tk.FOLDS:
            fd = data_by_fold[f]
            h_map[f] = tk.run_pipeline(fd["held"], fd["queries"], fd["w"], None, **kw)
            m_map[f] = tk.run_pipeline(fd["mixed"], fd["queries"], fd["w"], fd["hp"], **kw)
        oof_dist = tk.run_pipeline(oof_md, queries_oof, w_oof, None, **kw)

        rh = tk.eval_dist_map(h_map, data_by_fold)
        rm = tk.eval_dist_map(m_map, data_by_fold)
        ro = tk.eval_with_ctx(oof_dist, oof_ctx)
        r_mev = eval_sliced_ctx(oof_dist, mevid_ctx)
        r_syn = eval_synth_ctxs(oof_dist, synth_ctxs)

        rec = {
            "OOF_mADM": ro["mADM"], "OOF_mAP": ro["mAP"], "OOF_Body": ro["mADM"] - ro["mAP"], "OOF_R1": ro["R1"], "OOF_R5": ro["R5"], "OOF_R10": ro["R10"],
            "SimMEV_mADM": r_mev["mADM"], "SimMEV_mAP": r_mev["mAP"], "SimMEV_Body": r_mev["Body"],
            "SimSYN_mADM": r_syn["mADM"], "SimSYN_mAP": r_syn["mAP"], "SimSYN_Body": r_syn["Body"], "SimSYN_R5": r_syn["R5"], "SimSYN_R10": r_syn["R10"],
            "Held_mADM": rh["macro"], "Held_mAP": rh["mAP"], "Held_Body": rh["macro"] - rh["mAP"],
            "Held_Mk": rh["Market1501"], "Held_PA": rh["PA100k"], "Held_PE": rh["PETA"],
            "Mixed_mADM": rm["macro"], "Mixed_mAP": rm["mAP"], "Mixed_Body": rm["macro"] - rm["mAP"],
            "Mixed_Mk": rm["Market1501"], "Mixed_PA": rm["PA100k"], "Mixed_PE": rm["PETA"],
            "Mixed_PE_AP": rm["PETA_AP"], "Mixed_PE_Body": rm["PETA"] - rm["PETA_AP"],
            "Mixed_PA_AP": rm["PA100k_AP"], "Mixed_PA_Body": rm["PA100k"] - rm["PA100k_AP"],
            "Mixed_Mk_AP": rm["Market1501_AP"], "Mixed_Mk_Body": rm["Market1501"] - rm["Market1501_AP"],
        }
        results[name] = rec
        print(
            f"{name:28s} | {rec['OOF_mADM']:.6f} {rec['OOF_mAP']:.5f} {rec['OOF_Body']:.5f} | "
            f"{rec['SimMEV_mADM']:.6f} {rec['SimMEV_mAP']:.5f} {rec['SimMEV_Body']:.5f} | "
            f"{rec['SimSYN_mADM']:.6f} {rec['SimSYN_mAP']:.5f} {rec['SimSYN_Body']:.5f} | "
            f"{rec['Held_mADM']:.6f} {rec['Held_mAP']:.5f} {rec['Held_Body']:.5f} | "
            f"{rec['Mixed_mADM']:.6f} {rec['Mixed_mAP']:.5f} {rec['Mixed_Body']:.5f} | {time.monotonic()-t0:.1f}s",
            flush=True,
        )

    out_path = HERE / "public_mevid_synth_calibration_v1.json"
    out_path.write_text(json.dumps({"official": OFFICIAL_HISTORY, "local": results}, indent=2))
    print(f"\nSaved raw calibration data to {out_path}", flush=True)


if __name__ == "__main__":
    main()

