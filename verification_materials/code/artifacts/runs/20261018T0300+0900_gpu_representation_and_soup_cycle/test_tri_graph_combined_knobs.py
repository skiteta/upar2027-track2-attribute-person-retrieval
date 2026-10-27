"""Systematic GPU evaluation of:
  1. Certainty-Modulated Member Pooling (gamma_mem in {0.15, 0.30, 0.45}) across PE-Core, DINOv3-H+, EVA02
  2. Scale-Preserved Reliability-Weighted Attribute Margin (margin_w_rho with compensated margin_gamma) + ham_tau=0.16
  3. 2-Pass Progressive Visual Anchor (Pass 1: 0-hop raw visual features -> Pass 2: 1-hop manifold-diffused visual features on updated top-M)
  4. Full Stack with Asymmetric Soft-Min Distance Diffusion (dist_kappa=0.18) and Branch-Std Weighting (branch_std_delta=0.20)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/package_v3/B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR_V3.zip"))

from b121_fp32_cosine_graph import strict_fp32, unit_features_fp64
from b127_jaccard_prob_diffusion import build_quad_graphs_and_hubs
from upar.data import load_retrieval_split
from upar.postproc import apply_post, entropy_weight_matrix
from upar.rank_sibling_pipeline import read_json
from upar.source_fitted_gallery_edges import SPEC

spec_mix = importlib.util.spec_from_file_location(
    "mixed_ctx", ROOT / "artifacts/runs/20261017T1618+0900_mixed_gallery_context/compare_v1.py"
)
mod_mix = importlib.util.module_from_spec(spec_mix)
spec_mix.loader.exec_module(mod_mix)
FOLDS = mod_mix.FOLDS


def load_member_probs_raw(
    spec: dict, fold: str, meta: dict, selected: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    val = np.flatnonzero(meta["split"] == "val")
    positions = np.searchsorted(val, selected)
    raw_probs = []
    flip_probs = []
    post_probs = []
    for member in spec["lodo"]["members"]:
        arrays = []
        arrays_f = []
        for name in member["runs"][fold]:
            with np.load(ROOT / name / "val_logits.npz", allow_pickle=False) as z:
                correction = np.log(z["pos_weight"]).astype(np.float32)
                p_o = (1.0 / (1.0 + np.exp(-(z["logits_original"] - correction)))).astype(np.float32)[positions]
                p_f = (1.0 / (1.0 + np.exp(-(z["logits_flip"] - correction)))).astype(np.float32)[positions]
                arrays.append(p_o)
                arrays_f.append(p_f)
        p_raw = np.mean(arrays, axis=0).astype(np.float32)
        p_flp = np.mean(arrays_f, axis=0).astype(np.float32)
        raw_probs.append(p_raw)
        flip_probs.append(p_flp)
        post_probs.append(apply_post(p_raw, spec["lodo"]["post"]))
    return raw_probs, flip_probs, post_probs


def pool_members(
    raw_probs: list[np.ndarray],
    post_cfg: dict,
    gamma_mem: float = 0.0,
    member_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    flip_probs: list[np.ndarray] | None = None,
    flip_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    eff_probs = []
    for i, p_r in enumerate(raw_probs):
        fw = float(flip_weights[i]) if flip_probs is not None else 0.0
        if fw > 0.0 and flip_probs is not None:
            eff_probs.append((1.0 - fw) * p_r + fw * flip_probs[i])
        else:
            eff_probs.append(p_r)
    stack = np.stack(eff_probs, axis=0).astype(np.float64)  # (3, N, 40)
    w_m = np.array(member_weights, dtype=np.float64)[:, None, None]
    if gamma_mem > 0.0:
        cert = np.exp(float(gamma_mem) * (1.0 - 4.0 * stack * (1.0 - stack)))
        w_eff = w_m * cert
        p_pool = (stack * w_eff).sum(axis=0) / w_eff.sum(axis=0)
    else:
        p_pool = (stack * w_m).sum(axis=0) / w_m.sum(axis=0)
    return apply_post(p_pool.astype(np.float32), post_cfg)


from upar.metrics_strict import _normalized_degree_of_match


def precompute_eval_ctx(sub, qi: np.ndarray) -> dict:
    import hashlib
    import os
    q_sel = sub.queries[qi].astype(np.uint8)
    g_attr = sub.gallery_attributes.astype(np.uint8)
    g_qids = sub.gallery_query_ids.astype(np.int64)
    h = hashlib.sha1()
    h.update(str(q_sel.shape).encode())
    h.update(str(g_attr.shape).encode())
    h.update(qi.tobytes())
    h.update(g_qids[:64].tobytes())
    cache_dir = HERE / "extracted_features_v1" / "graph_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"eval_ctx_{len(qi)}_{len(g_attr)}_{h.hexdigest()[:16]}.npy"
    if cp.exists():
        norm_match = np.load(cp)
    else:
        norm_match = _normalized_degree_of_match(q_sel, g_attr).astype(np.float32)
        tmp = cache_dir / f".tmp_{os.getpid()}_{cp.name}"
        np.save(tmp, norm_match)
        tmp.replace(cp)
    return {
        "qi": qi,
        "norm_match": norm_match,
        "g_qids": g_qids,
    }


def eval_with_ctx(dist: np.ndarray, ctx: dict) -> dict[str, float]:
    qi = ctx["qi"]
    norm_match = ctx["norm_match"]
    g_qids = ctx["g_qids"]
    n_q = len(qi)
    adms, aps, r1s, r5s, r10s, inps = [], [], [], [], [], []
    with torch.inference_mode():
        g_qids_t = torch.from_numpy(g_qids).to("cuda", dtype=torch.int64)
        qi_t = torch.from_numpy(qi).to("cuda", dtype=torch.int64)
        for off in range(0, n_q, 256):
            end = min(off + 256, n_q)
            d_t = torch.from_numpy(dist[off:end].astype(np.float64)).to("cuda")
            orders_t = torch.argsort(d_t, dim=1, stable=True)
            nm_t = torch.from_numpy(norm_match[off:end]).to("cuda", dtype=torch.float64)
            nm_ord_t = torch.gather(nm_t, 1, orders_t)
            prefix_t = torch.cumsum(nm_ord_t, dim=1)
            rel_t = (g_qids_t[orders_t] == qi_t[off:end, None])
            counts = rel_t.sum(dim=1).cpu().numpy()
            _, col_idx_t = torch.nonzero(rel_t, as_tuple=True)
            pref_vals = prefix_t[rel_t].cpu().numpy()
            ranks_all = col_idx_t.cpu().numpy().astype(np.int64) + 1
            pos = 0
            for rc_val in counts:
                rc = int(rc_val)
                ranks = ranks_all[pos : pos + rc]
                pref_r = pref_vals[pos : pos + rc]
                pos += rc
                prec = np.arange(1, rc + 1, dtype=np.float64) / ranks
                adm_r = pref_r / ranks
                f_rk = int(ranks[0])
                l_rk = int(ranks[-1])
                adms.append(float(np.mean(adm_r)))
                aps.append(float(np.mean(prec)))
                r1s.append(float(f_rk <= 1))
                r5s.append(float(f_rk <= 5))
                r10s.append(float(f_rk <= 10))
                inps.append(float(rc / l_rk))
    madm = float(np.mean(adms))
    map_v = float(np.mean(aps))
    return {
        "mADM": madm,
        "mAP": map_v,
        "mE": (madm - map_v) / (1.0 - map_v),
        "R1": float(np.mean(r1s)),
        "R5": float(np.mean(r5s)),
        "R10": float(np.mean(r10s)),
        "mINP": float(np.mean(inps)),
    }


def eval_dist_map(dist_by_fold: dict[str, np.ndarray], data_by_fold: dict[str, dict], ctx_key: str = "eval_ctx") -> dict[str, float]:
    res = {}
    metrics_per_fold = {k: [] for k in ("mADM", "mAP", "R1", "R5", "R10", "mINP")}
    for fold in FOLDS:
        d = data_by_fold[fold]
        r_f = eval_with_ctx(dist_by_fold[fold], d[ctx_key])
        res[fold] = r_f["mADM"]
        res[f"{fold}_AP"] = r_f["mAP"]
        res[f"{fold}_mE"] = r_f["mE"]
        for k in metrics_per_fold:
            metrics_per_fold[k].append(r_f[k])
    res["macro"] = float(np.mean(metrics_per_fold["mADM"]))
    res["mAP"] = float(np.mean(metrics_per_fold["mAP"]))
    res["R1"] = float(np.mean(metrics_per_fold["R1"]))
    res["R5"] = float(np.mean(metrics_per_fold["R5"]))
    res["R10"] = float(np.mean(metrics_per_fold["R10"]))
    res["mINP"] = float(np.mean(metrics_per_fold["mINP"]))
    return res





SPECTRA_DEFS = {
    "pe": (1.0, 0.0, 0.0),
    "eva": (0.0, 1.0, 0.0),
    "hp": (0.0, 0.0, 1.0),
    "j25": (0.75, 0.25, 0.0),
    "j50": (0.50, 0.50, 0.0),
    "tri25": (0.5625, 0.1875, 0.25),
    "tri40": (0.40, 0.30, 0.30),
    "pe_hp": (0.65, 0.0, 0.35),
    "eva_hp": (0.0, 0.65, 0.35),
}


def compute_hubs_gpu(x_pe: torch.Tensor, x_eva: torch.Tensor, x_hp: torch.Tensor, spec_names: list[str], gamma_q: float = 2.6) -> dict[str, np.ndarray]:
    n = len(x_pe)
    hubs_g = {k: torch.zeros(n, dtype=torch.float64, device="cuda") for k in spec_names}
    with strict_fp32(), torch.inference_mode():
        for s in range(0, n, 512):
            e = min(n, s + 512)
            d_pe = x_pe[s:e] @ x_pe.T
            d_eva = x_eva[s:e] @ x_eva.T
            d_hp = x_hp[s:e] @ x_hp.T
            for sk in spec_names:
                w1, w2, w3 = SPECTRA_DEFS[sk]
                d_s = np.float32(w1) * d_pe + np.float32(w2) * d_eva + np.float32(w3) * d_hp
                hubs_g[sk] += (torch.clamp(d_s, min=0.0) ** gamma_q).to(torch.float64).sum(dim=0)
    return {k: (v / float(n)).cpu().numpy().astype(np.float32) for k, v in hubs_g.items()}


def precompute_fold_mode(
    u_pe: np.ndarray,
    u_eva: np.ndarray,
    u_hp: np.ndarray,
    p0: np.ndarray,
    raw_probs: list[np.ndarray],
    flip_probs: list[np.ndarray],
    post_cfg: dict,
    m_std: np.ndarray,
) -> dict:
    import hashlib
    import os
    from b121_gpu_graph import _mutual_from_directed

    h = hashlib.sha1()
    h.update(str(u_pe.shape).encode())
    h.update(u_pe[:16, :16].tobytes())
    h.update(u_eva[:16, :16].tobytes())
    h.update(u_hp[:16, :16].tobytes())
    h.update(u_pe[-16:, -16:].tobytes())
    h.update(u_hp[-16:, -16:].tobytes())
    cache_dir = HERE / "extracted_features_v1" / "graph_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"graph_{u_pe.shape[0]}_{h.hexdigest()[:16]}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return {
            "u_pe": u_pe,
            "u_eva": u_eva,
            "u_hp": u_hp,
            "p0": p0,
            "raw_probs": raw_probs,
            "flip_probs": flip_probs,
            "post_cfg": post_cfg,
            "m_std": m_std,
            "hubs": cached["hubs"],
            "branch_meta": cached["branch_meta"],
        }

    graphs, hubs = build_quad_graphs_and_hubs(u_pe, u_eva, gamma_q=2.6, chunk_size=512)
    n = len(u_pe)
    with strict_fp32(), torch.inference_mode():
        x_pe = torch.from_numpy(u_pe.astype(np.float32)).to("cuda")
        x_eva = torch.from_numpy(u_eva.astype(np.float32)).to("cuda")
        x_hp = torch.from_numpy(u_hp.astype(np.float32)).to("cuda")
        width = min(28, n - 1)
        pre_k = min(96, n - 1)
        dir_tri25 = np.empty((n, width), dtype=np.int64)
        dir_tri40 = np.empty((n, width), dtype=np.int64)
        extra_hubs = {k: torch.zeros(n, dtype=torch.float64, device="cuda") for k in ("hp", "tri25", "tri40", "pe_hp", "eva_hp")}
        diag = torch.arange(512, device="cuda")
        for s in range(0, n, 512):
            e = min(n, s + 512)
            m = e - s
            d_pe = x_pe[s:e] @ x_pe.T
            d_eva = x_eva[s:e] @ x_eva.T
            d_hp = x_hp[s:e] @ x_hp.T
            for sk, (w1, w2, w3) in SPECTRA_DEFS.items():
                if sk in extra_hubs:
                    d_s = np.float32(w1) * d_pe + np.float32(w2) * d_eva + np.float32(w3) * d_hp
                    extra_hubs[sk] += (torch.clamp(d_s, min=0.0) ** 2.6).to(torch.float64).sum(dim=0)
                    if sk in ("tri25", "tri40"):
                        neg = -d_s
                        neg[diag[:m], torch.arange(s, e, device="cuda")] = float("inf")
                        vals, idx = torch.topk(neg, k=pre_k, dim=1, largest=False, sorted=True)
                        v_np, i_np = vals.cpu().numpy(), idx.cpu().numpy()
                        order = np.lexsort((i_np, v_np), axis=1)[:, :width]
                        top_arr = np.take_along_axis(i_np, order, axis=1)
                        if sk == "tri25":
                            dir_tri25[s:e] = top_arr
                        else:
                            dir_tri40[s:e] = top_arr
        graphs["tri25_k28"] = _mutual_from_directed(dir_tri25)
        graphs["tri40_k28"] = _mutual_from_directed(dir_tri40)
        for sk, hv in extra_hubs.items():
            hubs[sk] = (hv / float(n)).cpu().numpy().astype(np.float32)

    branch_meta = {}
    with strict_fp32(), torch.inference_mode():
        u_pe_g = torch.from_numpy(u_pe).to("cuda", dtype=torch.float64)
        u_eva_g = torch.from_numpy(u_eva).to("cuda", dtype=torch.float64)
        u_hp_g = torch.from_numpy(u_hp).to("cuda", dtype=torch.float64)
        for b_name in ("pe", "eva", "j25_k28", "j50_k28", "tri25_k28", "tri40_k28"):
            n_b, c_b = graphs[b_name]
            rows, width = n_b.shape
            n_t = torch.from_numpy(n_b).to("cuda", dtype=torch.long)
            c_t = torch.from_numpy(c_b).to("cuda", dtype=torch.long)
            col_t = torch.arange(width, device="cuda", dtype=torch.long)[None, :]
            valid_t = (col_t < c_t[:, None]) & (n_t >= 0)
            safe_t = torch.where(valid_t, n_t, torch.zeros_like(n_t))
            n_neg2_t = torch.where(valid_t, n_t, torch.full_like(n_t, -2))

            dot_pe = torch.empty((rows, width), dtype=torch.float64, device="cuda")
            dot_eva = torch.empty((rows, width), dtype=torch.float64, device="cuda")
            dot_hp = torch.empty((rows, width), dtype=torch.float64, device="cuda")
            jac_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")
            for s in range(0, rows, 2048):
                e = min(rows, s + 2048)
                st = safe_t[s:e]
                dot_pe[s:e] = (u_pe_g[s:e, None, :] * u_pe_g[st]).sum(dim=2)
                dot_eva[s:e] = (u_eva_g[s:e, None, :] * u_eva_g[st]).sum(dim=2)
                dot_hp[s:e] = (u_hp_g[s:e, None, :] * u_hp_g[st]).sum(dim=2)
                ni = torch.where(valid_t[s:e], n_t[s:e], torch.full_like(n_t[s:e], -1))
                nj = n_neg2_t[st]
                shared = (ni[:, :, None, None] == nj[:, None, :, :]).sum(dim=(1, 3)).to(torch.float64)
                common = 2.0 + shared
                union = (c_t[s:e, None].to(torch.float64) + 1.0) + (c_t[st].to(torch.float64) + 1.0) - common
                jac_t[s:e] = torch.where(valid_t[s:e], common / torch.clamp(union, min=1.0), 0.0)

            branch_meta[b_name] = {
                "n_b": n_b,
                "c_b": c_b,
                "dot_pe": dot_pe.cpu().numpy(),
                "dot_eva": dot_eva.cpu().numpy(),
                "dot_hp": dot_hp.cpu().numpy(),
                "jac": jac_t.cpu().numpy(),
            }
            del n_t, c_t, col_t, valid_t, safe_t, n_neg2_t, dot_pe, dot_eva, dot_hp, jac_t
        del u_pe_g, u_eva_g, u_hp_g
        torch.cuda.empty_cache()

    tmp_path = cache_dir / f".tmp_{os.getpid()}_{cache_path.name}"
    torch.save({"hubs": hubs, "branch_meta": branch_meta}, tmp_path)
    tmp_path.replace(cache_path)

    return {
        "u_pe": u_pe,
        "u_eva": u_eva,
        "u_hp": u_hp,
        "p0": p0,
        "raw_probs": raw_probs,
        "flip_probs": flip_probs,
        "post_cfg": post_cfg,
        "m_std": m_std,
        "hubs": hubs,
        "branch_meta": branch_meta,
    }


def score_gpu(
    p_cal: np.ndarray,
    queries: np.ndarray,
    w_base: np.ndarray,
    entropy_delta: float = 0.35,
    b_std: np.ndarray | None = None,
    branch_std_delta: float = 0.0,
    m_std: np.ndarray | None = None,
    mstd_weight_delta: float = 0.0,
    q_mstd_beta: float = 0.0,
    eps_pos: float = 0.0,
    eps_neg: float = 0.0,
    nll_lam: float = 0.25,
    err_gamma: float = 1.0,
    nll_cap: float = 0.0,
    err_gamma_neg: float = -1.0,
    nll_lam_neg: float = -1.0,
    pos_q_mul: float = 1.0,
    w_base_neg: np.ndarray | None = None,
    q_pos_rho: float = 0.0,
    dich_sym_loss: float = 0.0,
    dich_q0_sym: float = 0.0,
    dich_q1_sym: float = 1.0,
    dich_pos_nll_add: float = 0.0,
) -> np.ndarray:
    eps = float(np.finfo(np.float32).eps)
    lam_pos = np.float32(nll_lam)
    lam_neg = np.float32(nll_lam if nll_lam_neg < 0.0 else nll_lam_neg)
    eg_pos = float(err_gamma)
    eg_neg = float(err_gamma if err_gamma_neg < 0.0 else err_gamma_neg)
    pmul = np.float32(pos_q_mul)
    W = entropy_weight_matrix(p_cal, w_base, entropy_delta)
    W0 = entropy_weight_matrix(p_cal, w_base_neg, entropy_delta) if w_base_neg is not None else None
    if (b_std is not None and branch_std_delta > 0.0) or (m_std is not None and mstd_weight_delta > 0.0):
        q64 = np.clip(p_cal.astype(np.float64), 1e-6, 1.0 - 1e-6)
        bern_sd = np.sqrt(q64 * (1.0 - q64) + 0.0025)
        mod = np.ones_like(q64)
        if b_std is not None and branch_std_delta > 0.0:
            bs_norm = b_std.astype(np.float64) / bern_sd
            mod = mod * np.exp(-float(branch_std_delta) * bs_norm)
        if m_std is not None and mstd_weight_delta > 0.0:
            ms_norm = m_std.astype(np.float64) / bern_sd
            mod = mod * np.exp(-float(mstd_weight_delta) * ms_norm)
        w64 = W.astype(np.float64)
        w_row_sum = w64.sum(axis=1, keepdims=True)
        w64_new = w64 * mod
        w64_new = w64_new * (w_row_sum / np.maximum(w64_new.sum(axis=1, keepdims=True), 1e-12))
        W = w64_new.astype(np.float32)
        if W0 is not None:
            w0_64 = W0.astype(np.float64)
            w0_row_sum = w0_64.sum(axis=1, keepdims=True)
            w0_new = w0_64 * mod
            w0_new = w0_new * (w0_row_sum / np.maximum(w0_new.sum(axis=1, keepdims=True), 1e-12))
            W0 = w0_new.astype(np.float32)
    with strict_fp32(), torch.inference_mode():
        p_t = torch.from_numpy(p_cal).to("cuda", dtype=torch.float32)
        w_t = torch.from_numpy(W).to("cuda", dtype=torch.float32)
        w0_t = torch.from_numpy(W0).to("cuda", dtype=torch.float32) if W0 is not None else w_t
        q_t = torch.from_numpy(queries).to("cuda", dtype=torch.float32)
        clipped = torch.clamp(p_t, min=eps, max=1.0 - eps)
        neg = -torch.log1p(-clipped)
        pos = -torch.log(clipped)
        if nll_cap > 0.0:
            cap = float(nll_cap)
            neg = cap * torch.tanh(neg / cap)
            pos = cap * torch.tanh(pos / cap)
        if eg_pos != 1.0 or eg_neg != 1.0 or lam_pos != lam_neg or pmul != 1.0 or W0 is not None or q_pos_rho != 0.0 or dich_sym_loss != 0.0 or dich_pos_nll_add != 0.0:
            f0_l1 = clipped ** eg_neg if eg_neg != 1.0 else p_t
            f1_l1 = (1.0 - clipped) ** eg_pos if eg_pos != 1.0 else (1.0 - p_t)
            loss0 = (1.0 - lam_neg) * f0_l1 + lam_neg * neg
            if dich_sym_loss > 0.0:
                s_d = np.float32(dich_sym_loss)
                lam_d0 = np.float32(lam_pos + dich_pos_nll_add)
                f0_pos_l1 = clipped[:, [3, 7, 20]] ** eg_pos if eg_pos != 1.0 else p_t[:, [3, 7, 20]]
                loss0_pos = pmul * ((1.0 - lam_d0) * f0_pos_l1 + lam_d0 * neg[:, [3, 7, 20]])
                loss0 = loss0.clone()
                loss0[:, [3, 7, 20]] = (1.0 - s_d) * loss0[:, [3, 7, 20]] + s_d * loss0_pos
            loss1 = pmul * ((1.0 - lam_pos) * f1_l1 + lam_pos * pos)
            if dich_sym_loss < -10.0:
                w_d1 = np.float32(-dich_sym_loss - 10.0)
                loss1 = loss1.clone()
                loss1[:, [3, 7, 20]] = w_d1 * ((1.0 - lam_pos) * f1_l1[:, [3, 7, 20]] + lam_pos * pos[:, [3, 7, 20]])
            elif dich_sym_loss < 0.0:
                w_d = np.float32(-dich_sym_loss)
                eg_d = 0.5 * (eg_pos + eg_neg)
                f0_d = clipped[:, [3, 7, 20]] ** eg_d if eg_d != 1.0 else p_t[:, [3, 7, 20]]
                f1_d = (1.0 - clipped[:, [3, 7, 20]]) ** eg_d if eg_d != 1.0 else (1.0 - p_t[:, [3, 7, 20]])
                loss0 = loss0.clone()
                loss1 = loss1.clone()
                loss0[:, [3, 7, 20]] = w_d * ((1.0 - lam_pos) * f0_d + lam_pos * neg[:, [3, 7, 20]])
                loss1[:, [3, 7, 20]] = w_d * ((1.0 - lam_pos) * f1_d + lam_pos * pos[:, [3, 7, 20]])
            if dich_pos_nll_add != 0.0:
                dl_p = np.float32(dich_pos_nll_add)
                loss1 = loss1.clone()
                loss1[:, [3, 7, 20]] = loss1[:, [3, 7, 20]] + pmul * dl_p * (pos[:, [3, 7, 20]] - f1_l1[:, [3, 7, 20]])
            base = (loss0 * w0_t).sum(dim=1)
            if q_pos_rho != 0.0:
                if abs(q_pos_rho) > 200.0:
                    rho_val = float(q_pos_rho + 200.0) if q_pos_rho < 0 else float(q_pos_rho - 200.0)
                    k_q = torch.clamp(q_t.sum(dim=1, keepdim=True) - 0.5 * q_t[:, 3:4] - q_t[:, 7:8] - q_t[:, 20:21], min=1.0)
                    k_ref = torch.tensor(6.15, dtype=torch.float32, device="cuda")
                elif abs(q_pos_rho) > 150.0:
                    rho_val = float(q_pos_rho + 150.0) if q_pos_rho < 0 else float(q_pos_rho - 150.0)
                    k_q = torch.clamp(q_t.sum(dim=1, keepdim=True) - q_t[:, 3:4] - q_t[:, 7:8] - q_t[:, 20:21], min=1.0)
                    k_ref = torch.tensor(5.75, dtype=torch.float32, device="cuda")
                elif abs(q_pos_rho) > 100.0:
                    rho_val = float(q_pos_rho + 100.0) if q_pos_rho < 0 else float(q_pos_rho - 100.0)
                    k_q = torch.clamp(q_t.sum(dim=1, keepdim=True) - q_t[:, 3:4] - q_t[:, 7:8] - q_t[:, 20:21], min=1.0)
                    k_ref = torch.tensor(6.05, dtype=torch.float32, device="cuda")
                elif abs(q_pos_rho) > 50.0:
                    rho_val = float(q_pos_rho + 50.0) if q_pos_rho < 0 else float(q_pos_rho - 50.0)
                    k_q = torch.clamp(q_t.sum(dim=1, keepdim=True) - q_t[:, 3:4] - q_t[:, 7:8] - q_t[:, 20:21], min=1.0)
                    k_ref = torch.tensor(5.90, dtype=torch.float32, device="cuda")
                else:
                    rho_val = float(q_pos_rho)
                    k_q = torch.clamp(q_t.sum(dim=1, keepdim=True), min=1.0)
                    k_ref = torch.clamp(k_q.mean(), min=1.0)
                q_mul = (k_ref / k_q) ** rho_val
                d1 = w_t * loss1
                d0 = w0_t * loss0
                if m_std is not None and q_mstd_beta != 0.0:
                    ms_t = torch.from_numpy(m_std).to("cuda", dtype=torch.float32)
                    d1 = d1 + np.float32(q_mstd_beta) * w_t * ms_t
                dist = base[None, :] - (q_t @ d0.T) + q_mul * (q_t @ d1.T)
                if dich_q0_sym != 0.0:
                    q0_b = 1.0 - q_t[:, [3, 7, 20]]
                    d0_b = d0[:, [3, 7, 20]]
                    dist = dist + float(dich_q0_sym) * (q_mul - 1.0) * (q0_b @ d0_b.T)
                if dich_q1_sym != 1.0:
                    q1_b = q_t[:, [3, 7, 20]]
                    d1_b = d1[:, [3, 7, 20]]
                    dist = dist - (1.0 - float(dich_q1_sym)) * (q_mul - 1.0) * (q1_b @ d1_b.T)
                return dist.cpu().numpy()
            delta = w_t * loss1 - w0_t * loss0
        else:
            base = (1.0 - lam_pos) * (p_t * w_t).sum(dim=1) + lam_pos * (neg * w_t).sum(dim=1)
            delta = (1.0 - lam_pos) * w_t * (1.0 - 2.0 * p_t) + lam_pos * w_t * (pos - neg)
        if m_std is not None and q_mstd_beta != 0.0:
            ms_t = torch.from_numpy(m_std).to("cuda", dtype=torch.float32)
            delta = delta + np.float32(q_mstd_beta) * w_t * ms_t
        dist = base[None, :] + q_t @ delta.T
        return dist.cpu().numpy()


def diffuse_visual_features_gpu(
    x_in: torch.Tensor,
    gpu_branches: list,
    b_states: list,
    lam_u: float = 0.45,
    steps_u: int = 1,
) -> torch.Tensor:
    if lam_u <= 0.0 or steps_u <= 0:
        return x_in
    num_g, _ = x_in.shape
    x_cur = x_in.clone()
    for _ in range(int(steps_u)):
        x_acc = torch.zeros_like(x_cur)
        for idx_b, (bw32, safe_t, w_t, _, _) in enumerate(gpu_branches):
            st = b_states[idx_b]
            valid_t = st["valid_t"]
            w_val = torch.where(valid_t, w_t, torch.zeros_like(w_t))
            w_sum = w_val.sum(dim=1, keepdim=True)
            denom = torch.clamp(w_sum + 1.5, min=1.0)
            w_norm = w_val / denom
            nb_sum = w_norm.sum(dim=1, keepdim=True)
            x_nb = torch.zeros_like(x_cur)
            for s in range(0, num_g, 4096):
                e = min(num_g, s + 4096)
                x_nb[s:e] = torch.einsum("gk,gkd->gd", w_norm[s:e], x_cur[safe_t[s:e]])
            x_b = (1.0 - float(lam_u) * nb_sum) * x_cur + float(lam_u) * x_nb
            x_acc = x_acc + bw32 * x_b
        x64 = x_acc.to(torch.float64)
        norm = torch.sqrt(torch.clamp((x64 * x64).sum(dim=1, keepdim=True), min=1e-24))
        x_cur = (x64 / norm).to(torch.float32)
    return x_cur


def run_pipeline(
    md: dict,
    queries: np.ndarray,
    w_base: np.ndarray,
    hp_slice: np.ndarray | None,
    *,
    branches: list[tuple[str, float, float, float]],
    gamma_mem: float = 0.0,
    margin_w_rho: float = 0.0,
    margin_gamma: float = 1.85,
    margin_scale: float = 1.0,
    ham_tau: float = 0.0,
    max_ham: int = 5,
    branch_std_delta: float = 0.0,
    bstd_tau: float = 0.0,
    post_tau: float = 1.05,
    entropy_delta: float = 0.35,
    err_gamma: float = 1.0,
    dist_kappa: float = 0.0,
    feat_diff_lam: float = 0.0,
    feat_diff_steps: int = 1,
    feat_diff_blend: float = 0.45,
    cos_c_hp: float = 0.0,
    anchor_spec_weights: dict[str, float] | None = None,
    anchor_alpha_q: float = 0.105,
    anchor_top_m: int = 12,
    anchor_temp_q: float = 0.06,
    anchor_gamma_q: float = 2.6,
    anchor_hub_norm: float = 0.35,
    d_steps: int = 2,
    d_alpha: float = 0.20,
    eps_pos: float = 0.0,
    eps_neg: float = 0.0,
    nll_lam: float = 0.25,
    sim_diff_alpha: float = 0.0,
    sim_diff_steps: int = 1,
    sim_kappa: float = 0.0,
    sim_diff_alpha2: float = 0.0,
    sim_kappa2: float = 0.0,
    sim_hub_rho: float = 0.0,
    dist_hub_rho: float = 0.0,
    vis_ex_gate: float = 0.0,
    mstd_weight_delta: float = 0.0,
    q_mstd_beta: float = 0.0,
    exact_q_mstd_beta: float = 0.0,
    seed_coh_beta: float = 0.0,
    ms_k1: int = 0,
    ms_temp1: float = 0.044,
    ms_blend: float = 0.0,
    seed_disagree_kap: float = 0.0,
    tail_dis_abs: float = 0.0,
    tail_dis_pos: float = 0.0,
    tail_dis_neg: float = 0.0,
    tail_mul_kap: float = 0.0,
    head_dq_blend: float = 0.0,
    head_mul_kap: float = 0.0,
    head_local_lam: float = 0.0,
    head_local_k: int = 12,
    seed_local_lam: float = 0.0,
    ex_diff_alpha2: float = 0.0,
    exact_p0_blend: float = 0.0,
    exact_p0_tau: float = 0.60,
    exact_branch_blend: float = 0.0,
    seed_kap_weight: bool = True,
    dyn_conf: bool = False,
    two_pass_anchor: bool = False,
    pass1_frac: float = 0.55,
    pass1_top_m: int = 8,
    pass1_temp_q: float = 0.045,
    pass1_vis_pull: float = 0.50,
    pass2_top_m: int = 16,
    pass2_temp_q: float = 0.05,
    multiscale_anchor: bool = False,
    norm_top_d_by_std: bool = False,
    seed_hub_norm: float = 0.0,
    head_beta: float = 0.0,
    head_tau: float = 0.25,
    exact_nll_lam: float = 0.40,
    exact_err_gamma: float = 1.25,
    exact_ent_delta: float = 0.45,
    exact_bstd_delta: float = -1.0,
    exact_mstd_delta: float = -1.0,
    exact_seed_blend: float = 0.0,
    seed_weight_blend: float = -1.0,
    exact_diff_steps: int = 0,
    ex_diff_alpha: float = 0.0,
    ex_kappa: float = 0.18,
    ex_diff_seed_only: bool = False,
    ex_diff_head_blend: float = 1.0,
    ex_diff_branch_init: bool = False,
    score_ens_blend: float = 0.45,
    exact_branch_diff: bool = False,
    ex_vis_alpha: float = 0.0,
    per_attr_nll_cap: float = 0.0,
    exact_nll_cap: float = 0.0,
    body_alpha: float = 0.0,
    body_k0: int = 10,
    body_tau: float = 0.40,
    body_pow: float = 1.0,
    body_nll_lam: float = 0.05,
    body_ent_delta: float = 0.15,
    body_w_uni: float = 1.0,
    body_nll_cap: float = 0.0,
    body_err_gamma: float = 1.0,
    body_err_gamma_neg: float = -1.0,
    body_nll_lam_neg: float = -1.0,
    body_pos_q_mul: float = 1.0,
    err_gamma_neg: float = -1.0,
    nll_lam_neg: float = -1.0,
    exact_err_gamma_neg: float = -1.0,
    exact_nll_lam_neg: float = -1.0,
    exact_pos_q_mul: float = 1.0,
    base_pos_q_mul: float = 1.0,
    post_renorm_alpha: float = 0.0,
    post_renorm_lbody: float = 0.0,
    body_head_keep: float = 0.0,
    body_neg_eta: float = 0.0,
    body_floor_c: float = 0.02,
    body_vis_scale: float = 1.0,
    body_kap: float = -1.0,
    body_d_alpha: float = -1.0,
    body_bstd_delta: float = 0.0,
    body_mstd_delta: float = 0.0,
    body_q_mstd_beta: float = 0.0,
    body_w_uni_pos: float = -1.0,
    base_w_uni_neg: float = 0.0,
    body_head_tau: float = -1.0,
    body_ex_prot_tau: float = 0.0,
    body_ex_prot_pow: float = 1.0,
    body_q_pos_rho: float = 0.0,
    base_q_pos_rho: float = 0.0,
    exact_q_pos_rho: float = 0.0,
    body_dich_sym_loss: float = 0.0,
    base_dich_sym_loss: float = 0.0,
    exact_dich_sym_loss: float = 0.0,
    dich_q0_sym: float = 0.0,
    dich_q1_sym: float = 1.0,
    dich_pos_nll_add: float = 0.0,
    anchor_q_rho: float = 0.0,
    head_q_rho: float = 0.0,
    body_alpha_q_rho: float = 0.0,
    head_local_scale_g: bool = False,
    seed_l1_blend: float = 0.0,
    member_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    flip_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    branch_member_tilt: float = 0.0,
    stage1_steps: int = 3,
    stage2_steps: int = 6,
    coevolve_blend: float = 0.55,
    b127_mode: bool = False,
    body_p0_blend: float = 0.0,
    body_clean_c_alpha: float = 0.0,
    body_clean_c_tau: float = 0.25,
    clean_c_nll_lam: float = 0.25,
    clean_c_nll_lam_neg: float = 0.25,
    clean_c_err_gamma: float = 1.0,
    clean_c_err_gamma_neg: float = 1.0,
    clean_c_p0_blend: float = 0.0,
    clean_c_d_alpha: float = -1.0,
    clean_c_dich_sym_loss: float = -1.0,
    clean_c_pos_q_mul: float = 1.0,
    clean_c_ent_delta: float = -999.0,
    clean_c_vis_scale: float = 1.0,
    clean_c_head_keep: float = 1.0,
) -> np.ndarray:
    if stage1_steps == 0 and stage2_steps == 0 and anchor_alpha_q <= 0.0:
        d0 = score_gpu(md["p0"], queries, w_base, entropy_delta=entropy_delta)
        return d0[:, hp_slice] if hp_slice is not None else d0
    need_repool = (
        gamma_mem > 0.0
        or member_weights != (0.40, 0.35, 0.25)
        or any(fw > 0.0 for fw in flip_weights)
    )
    p0 = (
        pool_members(
            md["raw_probs"],
            md["post_cfg"],
            gamma_mem=gamma_mem,
            member_weights=member_weights,
            flip_probs=md["flip_probs"],
            flip_weights=flip_weights,
        )
        if need_repool
        else md["p0"]
    )
    m_std = md["m_std"]
    hubs = md["hubs"]
    bmeta = md["branch_meta"]

    if anchor_spec_weights is None:
        anchor_spec_weights = {
            "pe": 0.10, "eva": 0.10, "j25": 0.18, "j50": 0.18,
            "tri25": 0.16, "tri40": 0.14, "pe_hp": 0.07, "eva_hp": 0.07,
        }

    with strict_fp32(), torch.inference_mode():
        p0_g = torch.from_numpy(p0).to("cuda", dtype=torch.float64)
        ms_g = torch.from_numpy(m_std).to("cuda", dtype=torch.float64)
        wb_g = torch.from_numpy(w_base.astype(np.float64)).to("cuda", dtype=torch.float64)

        b_states = []
        for b_name, bw, w_pe, w_eva in branches:
            w_hp = max(0.0, 1.0 - w_pe - w_eva)
            if branch_member_tilt > 0.0:
                t = float(branch_member_tilt)
                mw_b = (
                    (1.0 - t) * member_weights[0] + t * w_pe,
                    (1.0 - t) * member_weights[1] + t * w_hp,
                    (1.0 - t) * member_weights[2] + t * w_eva,
                )
                p0_b_np = pool_members(
                    md["raw_probs"],
                    md["post_cfg"],
                    gamma_mem=gamma_mem,
                    member_weights=mw_b,
                    flip_probs=md["flip_probs"],
                    flip_weights=flip_weights,
                )
                p0_b_g = torch.from_numpy(p0_b_np).to("cuda", dtype=torch.float64)
            else:
                p0_b_g = p0_g
            bm = bmeta[b_name]
            n_b, c_b = bm["n_b"], bm["c_b"]
            rows, width = n_b.shape
            n_t = torch.from_numpy(n_b).to("cuda", dtype=torch.long)
            c_t = torch.from_numpy(c_b).to("cuda", dtype=torch.long)
            col_t = torch.arange(width, device="cuda", dtype=torch.long)[None, :]
            valid_t = (col_t < c_t[:, None]) & (n_t >= 0)
            safe_t = torch.where(valid_t, n_t, torch.zeros_like(n_t))

            dot_pe = torch.from_numpy(bm["dot_pe"]).to("cuda", dtype=torch.float64)
            dot_eva = torch.from_numpy(bm["dot_eva"]).to("cuda", dtype=torch.float64)
            dot_hp = torch.from_numpy(bm["dot_hp"]).to("cuda", dtype=torch.float64)
            jac_t = torch.from_numpy(bm["jac"]).to("cuda", dtype=torch.float64)

            cos_b = torch.clamp(w_pe * dot_pe + w_eva * dot_eva + w_hp * dot_hp, 0.0, 1.0)
            if b127_mode:
                cos = cos_b
            elif cos_c_hp > 0.0:
                rem = 0.5 * (1.0 - float(cos_c_hp))
                cos_c = torch.clamp(rem * dot_pe + rem * dot_eva + float(cos_c_hp) * dot_hp, 0.0, 1.0)
                cos = 0.65 * cos_b + 0.35 * cos_c
            else:
                cos_c = torch.clamp(0.50 * dot_pe + 0.50 * dot_eva, 0.0, 1.0)
                cos = 0.65 * cos_b + 0.35 * cos_c
            sp_t = torch.where(valid_t, cos * (jac_t ** 0.80), 0.0)

            if margin_w_rho > 0.0:
                r_base = (wb_g ** float(margin_w_rho))[None, None, :]
                r_base = float(margin_scale) * (r_base / r_base.mean(dim=2, keepdims=True))
            else:
                r_base = None

            def _ew(p_in: torch.Tensor, sp=sp_t, vt=valid_t, st=safe_t, rb=r_base):
                bits = p_in >= 0.5
                dp = torch.abs(p_in[:, None, :] - p_in[st])
                if ham_tau > 0.0:
                    ham = ((bits[:, None, :] != bits[st]) & (dp >= float(ham_tau))).sum(dim=2)
                else:
                    ham = (bits[:, None, :] != bits[st]).sum(dim=2)
                term = torch.clamp(1.0 - (dp ** float(margin_gamma)), min=1e-12)
                if rb is not None:
                    log_w = (rb * torch.log(term)).sum(dim=2)
                else:
                    log_w = float(margin_scale) * torch.log(term).sum(dim=2)
                w_attr = torch.exp(log_w)
                return torch.where(vt & (ham <= int(max_ham)), w_attr * sp, 0.0), ham

            def _diff(p_in: torch.Tensor, ew_in: torch.Tensor, steps: int, vt=valid_t, ct=c_t, st=safe_t):
                p_cur = p_in.clone()
                w_valid = torch.where(vt, ew_in, 0.0)
                c_denom = torch.clamp(ct, min=1).to(torch.float64)[:, None]
                if b127_mode:
                    w_norm = w_valid / c_denom
                    w_sum = w_norm.sum(dim=1, keepdim=True)
                    for _ in range(steps):
                        p_nb = (w_norm[:, :, None] * p_cur[st]).sum(dim=1)
                        p_cur = (1.0 - 0.95 * w_sum) * p_cur + 0.95 * p_nb
                    return torch.clamp(p_cur, 1e-5, 1.0 - 1e-5)
                w_raw_sum = w_valid.sum(dim=1, keepdim=True)
                denom = 0.65 * c_denom + 0.35 * torch.clamp(w_raw_sum + 5.0, min=1.0)
                w_norm = w_valid / denom
                w_sum = w_norm.sum(dim=1, keepdim=True)
                step_scale = torch.minimum(
                    torch.tensor(0.95, dtype=torch.float64, device="cuda"),
                    0.95 / torch.clamp(w_sum, min=1e-12),
                )
                ms_term = torch.exp(-0.40 * ms_g)
                conf = torch.exp(0.25 * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * ms_term
                for _ in range(steps):
                    if dyn_conf:
                        conf = torch.exp(0.25 * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * ms_term
                    w_a = w_norm[:, :, None] * conf[st]
                    w_a_sum = w_a.sum(dim=1)
                    scale_a = torch.where(w_a_sum > 1e-12, w_sum / torch.clamp(w_a_sum, min=1e-12), 0.0)
                    p_nb = (w_a * p_cur[st]).sum(dim=1) * scale_a
                    p_cur = (1.0 - step_scale * w_sum) * p_cur + step_scale * p_nb
                return torch.clamp(p_cur, 1e-5, 1.0 - 1e-5)

            ew0_t, ham0_t = _ew(p0_b_g)
            p_mid_t = _diff(p0_b_g, ew0_t, int(stage1_steps))
            b_states.append({
                "name": b_name,
                "bw": bw,
                "p0_b_g": p0_b_g,
                "n_t": n_t,
                "c_t": c_t,
                "valid_t": valid_t,
                "safe_t": safe_t,
                "jac_t": jac_t,
                "sp_t": sp_t,
                "ew0_t": ew0_t,
                "ham0_t": ham0_t,
                "p_mid_t": p_mid_t,
                "_ew": _ew,
                "_diff": _diff,
            })

        raw_pds = []
        for st in b_states:
            if coevolve_blend <= 0.0 or b127_mode:
                ew_t = st["ew0_t"]
            else:
                ew2_t, _ = st["_ew"](st["p_mid_t"])
                ew2_t = torch.where(st["ham0_t"] <= int(max_ham), ew2_t, 0.0)
                s0_t = st["ew0_t"].sum(dim=1, keepdim=True)
                s2_t = ew2_t.sum(dim=1, keepdim=True)
                ew2_norm_t = torch.where(s2_t > 1e-12, ew2_t * (s0_t / torch.clamp(s2_t, min=1e-12)), 0.0)
                gate_t = torch.clamp((st["jac_t"] - 0.24) / 0.30, 0.0, 1.0)
                cb = float(coevolve_blend)
                ew_t = (1.0 - cb) * st["ew0_t"] + cb * ((1.0 - gate_t) * ew2_norm_t + gate_t * ew2_t)
            pd_t = st["_diff"](st["p0_b_g"], ew_t, int(stage2_steps))
            raw_pds.append((st, ew_t, pd_t))

        if bstd_tau > 0.0:
            pd_stack = torch.stack([pd_t for _, _, pd_t in raw_pds], dim=0)
            pd_std = pd_stack.std(dim=0, unbiased=False)
            tau_mat = float(post_tau) * (1.0 + float(bstd_tau) * pd_std)
        else:
            tau_mat = float(post_tau)

        p_cals = []
        branch_gpu_info = []
        for st, ew_t, pd_t in raw_pds:
            q_t = torch.clamp(pd_t, 1e-6, 1.0 - 1.0e-6)
            p_cal_np = (
                torch.clamp(torch.sigmoid(torch.log(q_t / (1.0 - q_t)) / tau_mat), 1e-5, 1.0 - 1e-5)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            if post_renorm_alpha > 0.0:
                a_rn = float(post_renorm_alpha)
                g_u = p_cal_np[:, 8:20]
                s_u = np.maximum(g_u.sum(axis=1, keepdims=True), np.float32(1e-12))
                p_cal_np[:, 8:20] = (1.0 - a_rn) * g_u + a_rn * (g_u / s_u)
            if post_renorm_lbody > 0.0:
                a_rl = float(post_renorm_lbody)
                g_l = p_cal_np[:, 21:33]
                s_l = np.maximum(g_l.sum(axis=1, keepdims=True), np.float32(1e-12))
                p_cal_np[:, 21:33] = (1.0 - a_rl) * g_l + a_rl * (g_l / s_l)
            if post_renorm_alpha > 0.0 or post_renorm_lbody > 0.0:
                p_cal_np = np.clip(p_cal_np, 1e-5, 1.0 - 1e-5).astype(np.float32)
            p_cals.append((st["bw"], p_cal_np))
            safe_pos_t = torch.clamp(st["n_t"], min=0)
            fac_t = (np.float32(d_alpha) / torch.clamp(st["c_t"], min=1).to(torch.float32))[None, :]
            branch_gpu_info.append((np.float32(st["bw"]), safe_pos_t, ew_t.to(torch.float32), fac_t, p_cal_np))
            for k_del in ("p0_b_g", "jac_t", "sp_t", "ew0_t", "ham0_t", "p_mid_t", "_ew", "_diff"):
                st.pop(k_del, None)
        del raw_pds, ms_g, wb_g
        if exact_p0_blend <= 0.0:
            del p0_g
        torch.cuda.empty_cache()

        p_ens = sum(bw * pc for bw, pc in p_cals).astype(np.float32)
        ex_bs_d = branch_std_delta if exact_bstd_delta < 0.0 else exact_bstd_delta
        ex_ms_d = mstd_weight_delta if exact_mstd_delta < 0.0 else exact_mstd_delta
        b_std = np.std(np.stack([pc for _, pc in p_cals], axis=0), axis=0).astype(np.float32) if (branch_std_delta > 0.0 or ex_bs_d > 0.0) else None
        w_mean_vec = np.full_like(w_base, float(w_base.mean()), dtype=np.float32)
        w_base_neg_vec = (
            ((1.0 - float(base_w_uni_neg)) * w_base + float(base_w_uni_neg) * w_mean_vec).astype(np.float32)
            if base_w_uni_neg > 0.0 else None
        )

        base_ens = score_gpu(
            p_ens, queries, w_base, entropy_delta=entropy_delta, b_std=b_std, branch_std_delta=branch_std_delta,
            m_std=m_std, mstd_weight_delta=mstd_weight_delta, q_mstd_beta=q_mstd_beta,
            eps_pos=eps_pos, eps_neg=eps_neg, nll_lam=nll_lam, err_gamma=err_gamma, nll_cap=per_attr_nll_cap,
            err_gamma_neg=err_gamma_neg, nll_lam_neg=nll_lam_neg, pos_q_mul=base_pos_q_mul,
            w_base_neg=w_base_neg_vec, q_pos_rho=base_q_pos_rho,
            dich_sym_loss=base_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
            dich_pos_nll_add=dich_pos_nll_add,
        )
        kap = np.float32(dist_kappa)
        num_q, num_g = len(queries), len(p0)

        if head_beta > 0.0 or exact_seed_blend > 0.0:
            if exact_p0_blend > 0.0:
                q0_t = torch.clamp(p0_g, 1e-6, 1.0 - 1e-6)
                p0_cal_np = (
                    torch.clamp(torch.sigmoid(torch.log(q0_t / (1.0 - q0_t)) / float(exact_p0_tau)), 1e-5, 1.0 - 1e-5)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                p_for_exact = ((1.0 - float(exact_p0_blend)) * p_ens + float(exact_p0_blend) * p0_cal_np).astype(np.float32)
            else:
                p_for_exact = p_ens
            exact_ens = score_gpu(
                p_for_exact, queries, w_base, entropy_delta=exact_ent_delta, b_std=b_std, branch_std_delta=ex_bs_d,
                m_std=m_std, mstd_weight_delta=ex_ms_d, q_mstd_beta=exact_q_mstd_beta,
                nll_lam=exact_nll_lam, err_gamma=exact_err_gamma, nll_cap=exact_nll_cap,
                err_gamma_neg=exact_err_gamma_neg, nll_lam_neg=exact_nll_lam_neg, pos_q_mul=exact_pos_q_mul,
                q_pos_rho=exact_q_pos_rho, dich_sym_loss=exact_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
            )
            exact_ens_0 = exact_ens
            ex_b_list = []
            if exact_branch_blend > 0.0:
                ex_b_sum = np.zeros_like(exact_ens, dtype=np.float32)
                for bw32, _, _, _, p_cal_np in branch_gpu_info:
                    ex_b = score_gpu(
                        p_cal_np, queries, w_base, entropy_delta=exact_ent_delta, b_std=b_std, branch_std_delta=ex_bs_d,
                        m_std=m_std, mstd_weight_delta=ex_ms_d, q_mstd_beta=exact_q_mstd_beta,
                        nll_lam=exact_nll_lam, err_gamma=exact_err_gamma, nll_cap=exact_nll_cap,
                        err_gamma_neg=exact_err_gamma_neg, nll_lam_neg=exact_nll_lam_neg, pos_q_mul=exact_pos_q_mul,
                        q_pos_rho=exact_q_pos_rho, dich_sym_loss=exact_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
                    )
                    ex_b_list.append(ex_b)
                    ex_b_sum += bw32 * ex_b
                exact_ens = ((1.0 - float(exact_branch_blend)) * exact_ens_0 + float(exact_branch_blend) * ex_b_sum).astype(np.float32)
            exact_ens_t = torch.from_numpy(exact_ens).to("cuda", dtype=torch.float32)
            exact_head_t = exact_ens_t
        else:
            exact_ens_0 = base_ens
            ex_b_list = []
            exact_ens_t = torch.from_numpy(base_ens).to("cuda", dtype=torch.float32)
            exact_head_t = exact_ens_t

        gpu_branches = [
            (
                bw32, safe_pos_t, w_t, fac_t,
                score_gpu(
                    p_cal_np, queries, w_base, entropy_delta=entropy_delta, b_std=b_std, branch_std_delta=branch_std_delta,
                    m_std=m_std, mstd_weight_delta=mstd_weight_delta, q_mstd_beta=q_mstd_beta,
                    eps_pos=eps_pos, eps_neg=eps_neg, nll_lam=nll_lam, err_gamma=err_gamma, nll_cap=per_attr_nll_cap,
                    err_gamma_neg=err_gamma_neg, nll_lam_neg=nll_lam_neg, pos_q_mul=base_pos_q_mul,
                    w_base_neg=w_base_neg_vec, q_pos_rho=base_q_pos_rho,
                    dich_sym_loss=base_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
                    dich_pos_nll_add=dich_pos_nll_add,
                ),
            )
            for bw32, safe_pos_t, w_t, fac_t, p_cal_np in branch_gpu_info
        ]

        if ex_diff_alpha > 0.0:
            ex_kap = np.float32(ex_kappa)
            ex_acc = torch.zeros_like(exact_ens_t)
            for idx_b, (bw32, safe_t, w_t, _, _) in enumerate(gpu_branches):
                c_clamped = torch.clamp(b_states[idx_b]["c_t"], min=1).to(torch.float32)
                fac_ex = (np.float32(ex_diff_alpha) / c_clamped)[None, :]
                if ex_diff_branch_init and len(ex_b_list) == len(gpu_branches):
                    eb_blend = float(exact_branch_blend)
                    cur_ex_np = ((1.0 - eb_blend) * exact_ens_0 + eb_blend * ex_b_list[idx_b]).astype(np.float32)
                    cur_ex = torch.from_numpy(cur_ex_np).to("cuda", dtype=torch.float32)
                else:
                    cur_ex = exact_ens_t.clone()
                nxt_ex = torch.empty_like(cur_ex)
                for s in range(0, num_g, 1024):
                    e = min(s + 1024, num_g)
                    before_ex = cur_ex[:, s:e]
                    diff_ex = cur_ex[:, safe_t[s:e]] - before_ex[:, :, None]
                    if ex_kap != 0.0:
                        diff_ex = diff_ex - ex_kap * torch.abs(diff_ex)
                    delta_ex = (diff_ex * w_t[None, s:e, :]).sum(dim=2)
                    nxt_ex[:, s:e] = before_ex + fac_ex[:, s:e] * delta_ex
                if ex_diff_alpha2 > 0.0:
                    fac_ex2 = (np.float32(ex_diff_alpha2) / c_clamped)[None, :]
                    nxt_ex2 = torch.empty_like(nxt_ex)
                    for s in range(0, num_g, 1024):
                        e = min(s + 1024, num_g)
                        before_ex = nxt_ex[:, s:e]
                        diff_ex = nxt_ex[:, safe_t[s:e]] - before_ex[:, :, None]
                        if ex_kap != 0.0:
                            diff_ex = diff_ex - ex_kap * torch.abs(diff_ex)
                        delta_ex = (diff_ex * w_t[None, s:e, :]).sum(dim=2)
                        nxt_ex2[:, s:e] = before_ex + fac_ex2[:, s:e] * delta_ex
                    nxt_ex = nxt_ex2
                ex_acc = ex_acc + bw32 * nxt_ex
                del cur_ex, nxt_ex
            exact_ens_t = ex_acc
            del ex_acc
            if ex_diff_seed_only:
                pass
            elif ex_diff_head_blend < 1.0:
                hb = float(ex_diff_head_blend)
                exact_head_t = (1.0 - hb) * exact_head_t + hb * exact_ens_t
            else:
                exact_head_t = exact_ens_t
        exact_ens_cpu = exact_ens_t.cpu()
        exact_head_cpu = exact_head_t.cpu()
        del exact_ens_t, exact_head_t
        torch.cuda.empty_cache()

        seb = np.float32(score_ens_blend)
        sbb = np.float32(1.0 - score_ens_blend)
        d_comb_all = torch.zeros((num_q, num_g), dtype=torch.float32, device="cuda")
        for bw32, safe_t, w_t, fac_t, d_b in gpu_branches:
            cur = torch.from_numpy((sbb * d_b + seb * base_ens).astype(np.float32)).to("cuda")
            for _ in range(int(d_steps)):
                nxt = torch.empty_like(cur)
                for s in range(0, num_g, 1024):
                    e = min(s + 1024, num_g)
                    before = cur[:, s:e]
                    diff = cur[:, safe_t[s:e]] - before[:, :, None]
                    if kap != 0.0:
                        diff = diff - kap * torch.abs(diff)
                    delta = (diff * w_t[None, s:e, :]).sum(dim=2)
                    nxt[:, s:e] = before + fac_t[:, s:e] * delta
                cur = nxt
            d_comb_all = d_comb_all + bw32 * cur
            del cur, nxt
        all_d_std_mean = float(d_comb_all.std(dim=1).mean().item())
        d_comb_cpu = d_comb_all.cpu()
        del d_comb_all
        torch.cuda.empty_cache()

        if body_clean_c_alpha > 0.0:
            p_ens_cc = ((1.0 - float(clean_c_p0_blend)) * p_ens + float(clean_c_p0_blend) * p0).astype(np.float32) if clean_c_p0_blend > 0.0 else p_ens
            cc_ent = entropy_delta if clean_c_ent_delta < -100.0 else clean_c_ent_delta
            cc_dsl = base_dich_sym_loss if clean_c_dich_sym_loss < 0.0 else clean_c_dich_sym_loss
            base_clean_c = score_gpu(
                p_ens_cc, queries, w_base, entropy_delta=cc_ent, b_std=b_std, branch_std_delta=branch_std_delta,
                m_std=m_std, mstd_weight_delta=mstd_weight_delta, q_mstd_beta=q_mstd_beta,
                nll_lam=clean_c_nll_lam, err_gamma=clean_c_err_gamma, err_gamma_neg=clean_c_err_gamma_neg, nll_lam_neg=clean_c_nll_lam_neg,
                pos_q_mul=clean_c_pos_q_mul, dich_sym_loss=cc_dsl, dich_pos_nll_add=dich_pos_nll_add,
            )
            da_cc = np.float32(d_alpha if clean_c_d_alpha < 0.0 else clean_c_d_alpha)
            d_clean_c_all = torch.zeros((num_q, num_g), dtype=torch.float32, device="cuda")
            for idx_b, (bw32, safe_t, w_t, fac_t, p_cal_np) in enumerate(branch_gpu_info):
                fac_cc_t = fac_t if clean_c_d_alpha < 0.0 else (da_cc / torch.clamp(b_states[idx_b]["c_t"], min=1).to(torch.float32))[None, :]
                p_cal_cc = ((1.0 - float(clean_c_p0_blend)) * p_cal_np + float(clean_c_p0_blend) * p0).astype(np.float32) if clean_c_p0_blend > 0.0 else p_cal_np
                d_b_cc = score_gpu(
                    p_cal_cc, queries, w_base, entropy_delta=cc_ent, b_std=b_std, branch_std_delta=branch_std_delta,
                    m_std=m_std, mstd_weight_delta=mstd_weight_delta, q_mstd_beta=q_mstd_beta,
                    nll_lam=clean_c_nll_lam, err_gamma=clean_c_err_gamma, err_gamma_neg=clean_c_err_gamma_neg, nll_lam_neg=clean_c_nll_lam_neg,
                    pos_q_mul=clean_c_pos_q_mul, dich_sym_loss=cc_dsl, dich_pos_nll_add=dich_pos_nll_add,
                )
                cur_cc = torch.from_numpy((sbb * d_b_cc + seb * base_clean_c).astype(np.float32)).to("cuda")
                for _ in range(int(d_steps)):
                    nxt_cc = torch.empty_like(cur_cc)
                    for s in range(0, num_g, 1024):
                        e = min(s + 1024, num_g)
                        before_cc = cur_cc[:, s:e]
                        diff_cc = cur_cc[:, safe_t[s:e]] - before_cc[:, :, None]
                        if kap != 0.0:
                            diff_cc = diff_cc - kap * torch.abs(diff_cc)
                        delta_cc = (diff_cc * w_t[None, s:e, :]).sum(dim=2)
                        nxt_cc[:, s:e] = before_cc + fac_cc_t[:, s:e] * delta_cc
                    cur_cc = nxt_cc
                d_clean_c_all = d_clean_c_all + bw32 * cur_cc
                del cur_cc, nxt_cc
            d_clean_c_cpu = d_clean_c_all.cpu()
            del d_clean_c_all
            torch.cuda.empty_cache()
        else:
            d_clean_c_cpu = None

        if body_alpha > 0.0 or seed_l1_blend > 0.0:
            w_u_pos = float(body_w_uni if body_w_uni_pos < 0.0 else body_w_uni_pos)
            w_body_pos_vec = ((1.0 - w_u_pos) * w_base + w_u_pos * w_mean_vec).astype(np.float32)
            w_body_neg_vec = (
                ((1.0 - float(body_w_uni)) * w_base + float(body_w_uni) * w_mean_vec).astype(np.float32)
                if body_w_uni_pos >= 0.0 else None
            )
            p_ens_b = ((1.0 - float(body_p0_blend)) * p_ens + float(body_p0_blend) * p0).astype(np.float32) if body_p0_blend > 0.0 else p_ens
            base_l1 = score_gpu(
                p_ens_b, queries, w_body_pos_vec, entropy_delta=body_ent_delta,
                b_std=b_std, branch_std_delta=body_bstd_delta, m_std=m_std, mstd_weight_delta=body_mstd_delta, q_mstd_beta=body_q_mstd_beta,
                nll_lam=body_nll_lam, err_gamma=body_err_gamma, nll_cap=body_nll_cap,
                err_gamma_neg=body_err_gamma_neg, nll_lam_neg=body_nll_lam_neg, pos_q_mul=body_pos_q_mul,
                w_base_neg=w_body_neg_vec, q_pos_rho=body_q_pos_rho,
                dich_sym_loss=body_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
            )
            kap_l = np.float32(dist_kappa if body_kap < 0.0 else body_kap)
            da_l = np.float32(d_alpha if body_d_alpha < 0.0 else body_d_alpha)
            d_l1_all = torch.zeros((num_q, num_g), dtype=torch.float32, device="cuda")
            for idx_b, (bw32, safe_t, w_t, fac_t, p_cal_np) in enumerate(branch_gpu_info):
                fac_l_t = fac_t if body_d_alpha < 0.0 else (da_l / torch.clamp(b_states[idx_b]["c_t"], min=1).to(torch.float32))[None, :]
                p_cal_b = ((1.0 - float(body_p0_blend)) * p_cal_np + float(body_p0_blend) * p0).astype(np.float32) if body_p0_blend > 0.0 else p_cal_np
                d_b_l1 = score_gpu(
                    p_cal_b, queries, w_body_pos_vec, entropy_delta=body_ent_delta,
                    b_std=b_std, branch_std_delta=body_bstd_delta, m_std=m_std, mstd_weight_delta=body_mstd_delta, q_mstd_beta=body_q_mstd_beta,
                    nll_lam=body_nll_lam, err_gamma=body_err_gamma, nll_cap=body_nll_cap,
                    err_gamma_neg=body_err_gamma_neg, nll_lam_neg=body_nll_lam_neg, pos_q_mul=body_pos_q_mul,
                    w_base_neg=w_body_neg_vec, q_pos_rho=body_q_pos_rho,
                    dich_sym_loss=body_dich_sym_loss, dich_q0_sym=dich_q0_sym, dich_q1_sym=dich_q1_sym,
                )
                cur_l = torch.from_numpy((sbb * d_b_l1 + seb * base_l1).astype(np.float32)).to("cuda")
                for _ in range(int(d_steps)):
                    nxt_l = torch.empty_like(cur_l)
                    for s in range(0, num_g, 1024):
                        e = min(s + 1024, num_g)
                        before_l = cur_l[:, s:e]
                        diff_l = cur_l[:, safe_t[s:e]] - before_l[:, :, None]
                        if kap_l != 0.0:
                            diff_l = diff_l - kap_l * torch.abs(diff_l)
                        delta_l = (diff_l * w_t[None, s:e, :]).sum(dim=2)
                        nxt_l[:, s:e] = before_l + fac_l_t[:, s:e] * delta_l
                    cur_l = nxt_l
                d_l1_all = d_l1_all + bw32 * cur_l
                del cur_l, nxt_l
            d_l1_cpu = d_l1_all.cpu()
            del d_l1_all
            torch.cuda.empty_cache()
        else:
            d_l1_cpu = None

        if anchor_alpha_q <= 0.0:
            res_np = d_comb_cpu.numpy()
            return res_np[:, hp_slice] if hp_slice is not None else res_np

        x_pe = torch.from_numpy(md["u_pe"].astype(np.float32)).to("cuda")
        x_eva = torch.from_numpy(md["u_eva"].astype(np.float32)).to("cuda")
        x_hp = torch.from_numpy(md["u_hp"].astype(np.float32)).to("cuda")

        active_specs = [sk for sk, sw in anchor_spec_weights.items() if sw > 0.0]

        def _make_scales(h_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
            sc = {}
            for h_name in active_specs:
                h_vec = h_dict[h_name]
                if anchor_hub_norm > 0.0:
                    s_vec = ((h_vec.mean() ** anchor_hub_norm) / (h_vec ** anchor_hub_norm + 1e-6)).astype(np.float32)
                else:
                    s_vec = np.ones_like(h_vec, dtype=np.float32)
                sc[h_name] = torch.from_numpy(s_vec).to("cuda")[None, :]
            return sc

        scales_0 = _make_scales(hubs)
        m = min(int(anchor_top_m), num_g)
        out_d = np.empty((num_q, num_g), dtype=np.float32)

        skap = np.float32(sim_kappa)
        skap2 = np.float32(sim_kappa2)
        def _diffuse_sim_on_graph(sim_in: torch.Tensor) -> torch.Tensor:
            if sim_diff_alpha <= 0.0 or sim_diff_steps <= 0:
                return sim_in
            sim_acc_b = torch.zeros_like(sim_in)
            for idx_b, (bw32, safe_t, w_t, _, _) in enumerate(gpu_branches):
                c_clamped = torch.clamp(b_states[idx_b]["c_t"], min=1).to(torch.float32)
                if sim_hub_rho > 0.0:
                    w_bar = w_t.sum(dim=1) / c_clamped
                    w_bar_mean = torch.clamp(w_bar.mean(), min=1e-6)
                    hub_fac = ((w_bar / w_bar_mean + 1e-6) ** float(sim_hub_rho))
                    fac_s = (np.float32(sim_diff_alpha) / (c_clamped * hub_fac))[None, :]
                else:
                    fac_s = (np.float32(sim_diff_alpha) / c_clamped)[None, :]
                cur_s = sim_in.clone()
                for _ in range(int(sim_diff_steps)):
                    nxt_s = torch.empty_like(cur_s)
                    for s in range(0, num_g, 8192):
                        e = min(s + 8192, num_g)
                        before_s = cur_s[:, s:e]
                        diff_s = cur_s[:, safe_t[s:e]] - before_s[:, :, None]
                        if skap != 0.0:
                            diff_s = diff_s + skap * torch.abs(diff_s)
                        delta_s = (diff_s * w_t[None, s:e, :]).sum(dim=2)
                        nxt_s[:, s:e] = before_s + fac_s[:, s:e] * delta_s
                    cur_s = nxt_s
                if sim_diff_alpha2 > 0.0:
                    fac_s2 = (np.float32(sim_diff_alpha2) / c_clamped)[None, :]
                    nxt_s2 = torch.empty_like(cur_s)
                    for s in range(0, num_g, 8192):
                        e = min(s + 8192, num_g)
                        before_s = cur_s[:, s:e]
                        diff_s = cur_s[:, safe_t[s:e]] - before_s[:, :, None]
                        if skap2 != 0.0:
                            diff_s = diff_s + skap2 * torch.abs(diff_s)
                        delta_s = (diff_s * w_t[None, s:e, :]).sum(dim=2)
                        nxt_s2[:, s:e] = before_s + fac_s2[:, s:e] * delta_s
                    cur_s = nxt_s2
                sim_acc_b = sim_acc_b + bw32 * cur_s
                del cur_s, nxt_s
            return sim_acc_b

        q_all_t = torch.from_numpy(queries).to("cuda", dtype=torch.float32)
        k_q_multi_all = torch.clamp(q_all_t.sum(dim=1, keepdim=True) - q_all_t[:, 3:4] - q_all_t[:, 7:8] - q_all_t[:, 20:21], min=1.0)
        q_ratio_all = 5.90 / k_q_multi_all

        for q0 in range(0, num_q, 64):
            q1 = min(q0 + 64, num_q)
            dq = d_comb_cpu[q0:q1].to("cuda", non_blocking=True)
            ex_q = exact_ens_cpu[q0:q1].to("cuda", non_blocking=True)
            ex_h_q = exact_head_cpu[q0:q1].to("cuda", non_blocking=True)
            d_std = dq.std(dim=1, keepdim=True)
            ex_std = torch.clamp(ex_q.std(dim=1, keepdim=True), min=1e-6)
            ex_h_std = torch.clamp(ex_h_q.std(dim=1, keepdim=True), min=1e-6)
            q_rat = q_ratio_all[q0:q1]

            def _sim_from_feats(xp, xe, xh, sc_map, tidx, wm):
                c_pe = torch.einsum("qmd,gd->qmg", xp[tidx], xp)
                c_eva = torch.einsum("qmd,gd->qmg", xe[tidx], xe)
                need_hp = any(SPECTRA_DEFS[sk][2] > 0.0 for sk in active_specs)
                c_hp = torch.einsum("qmd,gd->qmg", xh[tidx], xh) if need_hp else None
                if seed_coh_beta > 0.0:
                    xp_s = xp[tidx]
                    xe_s = xe[tidx]
                    xh_s = xh[tidx]
                    mm_pe = torch.einsum("qmd,qnd->qmn", xp_s, xp_s)
                    mm_eva = torch.einsum("qmd,qnd->qmn", xe_s, xe_s)
                    mm_hp = torch.einsum("qmd,qnd->qmn", xh_s, xh_s)
                    mm_joint = torch.clamp(0.45 * mm_pe + 0.35 * mm_eva + 0.20 * mm_hp, min=0.0)
                    coh = torch.einsum("qn,qmn->qm", wm, mm_joint)
                    wm = wm * torch.exp(float(seed_coh_beta) * coh)
                    wm = wm / torch.clamp(wm.sum(dim=1, keepdim=True), min=1e-12)
                sim_acc = torch.zeros_like(dq)
                for sk in active_specs:
                    sw = anchor_spec_weights[sk]
                    w1, w2, w3 = SPECTRA_DEFS[sk]
                    c_s = w1 * c_pe + w2 * c_eva
                    if w3 > 0.0 and c_hp is not None:
                        c_s = c_s + w3 * c_hp
                    kern = torch.clamp(c_s, min=0.0) ** float(anchor_gamma_q)
                    if seed_hub_norm > 0.0:
                        s_seed = (sc_map[sk][0, tidx]) ** float(seed_hub_norm / max(anchor_hub_norm, 1e-6))
                        wm_eff = wm * s_seed
                        wm_eff = wm_eff / torch.clamp(wm_eff.sum(dim=1, keepdim=True), min=1e-12)
                    else:
                        wm_eff = wm
                    sim_s = torch.einsum("qm,qmg->qg", wm_eff, kern) * sc_map[sk]
                    sim_acc = sim_acc + float(sw) * sim_s
                return sim_acc

            if exact_seed_blend > 0.0:
                dq_z = (dq - dq.mean(dim=1, keepdim=True)) / torch.clamp(d_std, min=1e-6)
                ex_z = (ex_q - ex_q.mean(dim=1, keepdim=True)) / ex_std
                z_base = (1.0 - float(exact_seed_blend)) * dq_z + float(exact_seed_blend) * ex_z
                if seed_l1_blend > 0.0 and d_l1_cpu is not None:
                    dl_s_q = d_l1_cpu[q0:q1].to("cuda", non_blocking=True)
                    dl_s_std = torch.clamp(dl_s_q.std(dim=1, keepdim=True), min=1e-6)
                    dl_z = (dl_s_q - dl_s_q.mean(dim=1, keepdim=True)) / dl_s_std
                    z_base = (1.0 - float(seed_l1_blend)) * z_base + float(seed_l1_blend) * dl_z
                z_comb = z_base
                if seed_disagree_kap > 0.0:
                    z_comb = z_comb + float(seed_disagree_kap) * torch.abs(dq_z - ex_z)
                delta_dq = (dq - dq.min(dim=1, keepdim=True).values) / torch.clamp(d_std, min=1e-6)
                delta_ex = (ex_q - ex_q.min(dim=1, keepdim=True).values) / ex_std
                if tail_dis_abs > 0.0:
                    z_comb = z_comb + float(tail_dis_abs) * torch.abs(delta_dq - delta_ex)
                if tail_dis_pos > 0.0:
                    z_comb = z_comb + float(tail_dis_pos) * torch.relu(delta_ex - delta_dq)
                if tail_dis_neg > 0.0:
                    z_comb = z_comb + float(tail_dis_neg) * torch.relu(delta_dq - delta_ex)
                if tail_mul_kap > 0.0:
                    z_comb = z_comb + float(tail_mul_kap) * (delta_dq * delta_ex)
                seed_score = z_comb * d_std
                if seed_kap_weight:
                    weight_score = seed_score
                else:
                    z_w = z_base + (float(tail_mul_kap) * (delta_dq * delta_ex) if tail_mul_kap > 0.0 else 0.0)
                    weight_score = z_w * d_std
            else:
                seed_score = dq
                weight_score = dq

            _, top_idx = torch.topk(seed_score, k=m, dim=1, largest=False, sorted=True)
            top_d = torch.gather(weight_score, 1, top_idx)
            if seed_local_lam > 0.0:
                seed_spread = torch.clamp(top_d[:, -1:] - top_d[:, :1], min=1e-6)
                seed_loc_std = seed_spread / 0.35
                temp_scale = (1.0 - float(seed_local_lam)) + float(seed_local_lam) * (seed_loc_std / torch.clamp(d_std, min=1e-6))
                w_m = torch.softmax(-top_d / (float(anchor_temp_q) * temp_scale), dim=1)
            elif norm_top_d_by_std:
                top_d_scaled = top_d * (all_d_std_mean / torch.clamp(d_std, min=1e-6))
                w_m = torch.softmax(-top_d_scaled / float(anchor_temp_q), dim=1)
            else:
                w_m = torch.softmax(-top_d / float(anchor_temp_q), dim=1)
            if ms_k1 > 0 and ms_blend > 0.0 and ms_k1 < m:
                w_m1 = torch.zeros_like(w_m)
                w_m1[:, :ms_k1] = torch.softmax(-top_d[:, :ms_k1] / float(ms_temp1), dim=1)
                w_m = (1.0 - float(ms_blend)) * w_m + float(ms_blend) * w_m1
            sim_tot = _sim_from_feats(x_pe, x_eva, x_hp, scales_0, top_idx, w_m)
            sim_tot = _diffuse_sim_on_graph(sim_tot)

            if vis_ex_gate > 0.0:
                d_ex0 = (ex_q - ex_q.min(dim=1, keepdim=True).values) / ex_std
                g_ve = torch.exp(-float(vis_ex_gate) * d_ex0)
                sim_g = sim_tot * g_ve
                sim_eff = sim_g * (sim_tot.sum(dim=1, keepdim=True) / torch.clamp(sim_g.sum(dim=1, keepdim=True), min=1e-12))
            else:
                sim_eff = sim_tot

            a_alpha_eff = float(anchor_alpha_q) * (q_rat ** float(anchor_q_rho)) if anchor_q_rho != 0.0 else float(anchor_alpha_q)
            d_out_q = dq - a_alpha_eff * d_std * sim_eff
            head_exp = None

            if head_beta > 0.0:
                if head_dq_blend > 0.0:
                    hdq = float(head_dq_blend)
                    base_ex = (1.0 - hdq) * ex_h_q + hdq * (ex_h_std / torch.clamp(d_std, min=1e-6)) * dq
                else:
                    base_ex = ex_h_q
                if head_mul_kap > 0.0:
                    d_dq_h = (dq - dq.min(dim=1, keepdim=True).values) / torch.clamp(d_std, min=1e-6)
                    d_ex_h = (ex_h_q - ex_h_q.min(dim=1, keepdim=True).values) / ex_h_std
                    base_ex = base_ex + float(head_mul_kap) * ex_h_std * (d_dq_h * d_ex_h)
                if ex_vis_alpha > 0.0:
                    ex_q_eff = base_ex - float(ex_vis_alpha) * ex_h_std * sim_tot
                    ex_eff_std = torch.clamp(ex_q_eff.std(dim=1, keepdim=True), min=1e-6)
                else:
                    ex_q_eff = base_ex
                    ex_eff_std = ex_h_std
                ex_min = ex_q_eff.min(dim=1, keepdim=True).values
                if head_local_lam > 0.0:
                    if head_local_scale_g:
                        hk = min(max(int(head_local_k), int(round(float(head_local_k) * (num_g / 10000.0)))), num_g)
                    else:
                        hk = min(int(head_local_k), num_g)
                    ex_topk = torch.topk(ex_q_eff, k=hk, dim=1, largest=False, sorted=True).values
                    loc_spread = torch.clamp(ex_topk[:, -1:] - ex_min, min=1e-6)
                    loc_norm = loc_spread / 0.35
                    eff_scale = (1.0 - float(head_local_lam)) * ex_eff_std + float(head_local_lam) * loc_norm
                else:
                    eff_scale = ex_eff_std
                delta_ex = (ex_q_eff - ex_min) / eff_scale
                head_exp = torch.exp(-delta_ex / float(head_tau))
                h_beta_eff = float(head_beta) * (q_rat ** float(head_q_rho)) if head_q_rho != 0.0 else float(head_beta)
                d_out_q = d_out_q - h_beta_eff * d_std * head_exp

            if body_alpha > 0.0 and d_l1_cpu is not None:
                dl_q = d_l1_cpu[q0:q1].to("cuda", non_blocking=True)
                dl_std = torch.clamp(dl_q.std(dim=1, keepdim=True), min=1e-6)
                d_l1_out_q = dl_q - float(body_vis_scale) * a_alpha_eff * dl_std * sim_eff
                if body_head_keep > 0.0 and head_exp is not None:
                    hb_exp = torch.exp(-delta_ex / float(body_head_tau)) if body_head_tau > 0.0 else head_exp
                    h_beta_eff = float(head_beta) * (q_rat ** float(head_q_rho)) if head_q_rho != 0.0 else float(head_beta)
                    d_l1_out_q = d_l1_out_q - float(body_head_keep) * h_beta_eff * dl_std * hb_exp
                c_min = d_out_q.min(dim=1, keepdim=True).values
                l_min = d_l1_out_q.min(dim=1, keepdim=True).values
                sc = torch.clamp(d_out_q.std(dim=1, keepdim=True), min=1e-6)
                sl = torch.clamp(d_l1_out_q.std(dim=1, keepdim=True), min=1e-6)
                zc = (d_out_q - c_min) / sc
                zl = (d_l1_out_q - l_min) / sl
                bk = min(max(int(body_k0), int(round(float(body_k0) * (num_g / 10000.0)))), num_g)
                z0_c = torch.topk(zc, k=bk, dim=1, largest=False, sorted=True).values[:, -1:]
                z0_l = torch.topk(zl, k=bk, dim=1, largest=False, sorted=True).values[:, -1:]
                uc = torch.clamp(zc - z0_c, min=0.0)
                if d_clean_c_cpu is not None:
                    dcc_q = d_clean_c_cpu[q0:q1].to("cuda", non_blocking=True)
                    dcc_std = torch.clamp(dcc_q.std(dim=1, keepdim=True), min=1e-6)
                    d_cc_out_q = dcc_q - float(clean_c_vis_scale) * a_alpha_eff * dcc_std * sim_eff
                    if clean_c_head_keep > 0.0 and head_exp is not None:
                        hb_cc_exp = torch.exp(-delta_ex / float(body_head_tau)) if body_head_tau > 0.0 else head_exp
                        h_beta_eff = float(head_beta) * (q_rat ** float(head_q_rho)) if head_q_rho != 0.0 else float(head_beta)
                        d_cc_out_q = d_cc_out_q - float(clean_c_head_keep) * h_beta_eff * dcc_std * hb_cc_exp
                    cc_min = d_cc_out_q.min(dim=1, keepdim=True).values
                    sc_cc = torch.clamp(d_cc_out_q.std(dim=1, keepdim=True), min=1e-6)
                    z_cc = (d_cc_out_q - cc_min) / sc_cc
                    z0_cc = torch.topk(z_cc, k=bk, dim=1, largest=False, sorted=True).values[:, -1:]
                    u_cc = torch.clamp(z_cc - z0_cc, min=1e-5)
                    g_cc = float(body_clean_c_alpha) * (1.0 - torch.exp(-uc / float(body_clean_c_tau)))
                    if body_ex_prot_tau > 0.0 and head_exp is not None:
                        prot_cc = 1.0 - torch.exp(-((delta_ex / float(body_ex_prot_tau)) ** float(body_ex_prot_pow)))
                        g_cc_eff = torch.where(u_cc > uc, g_cc * prot_cc, g_cc)
                    else:
                        g_cc_eff = g_cc
                    uc_body = (1.0 - g_cc_eff) * uc + g_cc_eff * u_cc
                else:
                    uc_body = uc
                b_alpha_eff = float(body_alpha) * (q_rat ** float(body_alpha_q_rho)) if body_alpha_q_rho != 0.0 else float(body_alpha)
                if body_pow != 1.0:
                    gate = b_alpha_eff * (1.0 - torch.exp(-((uc / float(body_tau)) ** float(body_pow))))
                else:
                    gate = b_alpha_eff * (1.0 - torch.exp(-uc / float(body_tau)))
                if body_neg_eta > 0.0:
                    raw_ul = zl - z0_l
                    ul = torch.where(raw_ul >= 0.0, raw_ul, float(body_neg_eta) * raw_ul)
                    if body_ex_prot_tau != 0.0 and head_exp is not None:
                        ptau = abs(float(body_ex_prot_tau))
                        prot = 1.0 - torch.exp(-((delta_ex / ptau) ** float(body_ex_prot_pow)))
                        gate_eff = gate * prot if body_ex_prot_tau < 0.0 else torch.where(ul > uc_body, gate * prot, gate)
                    else:
                        gate_eff = gate
                    u_comb = torch.maximum((1.0 - gate_eff) * uc_body + gate_eff * ul, float(body_floor_c) * uc_body)
                    z_final = torch.where(zc <= z0_c, zc, z0_c + u_comb)
                else:
                    ul = torch.clamp(zl - z0_l, min=1e-5)
                    if body_ex_prot_tau != 0.0 and head_exp is not None:
                        ptau = abs(float(body_ex_prot_tau))
                        prot = 1.0 - torch.exp(-((delta_ex / ptau) ** float(body_ex_prot_pow)))
                        gate_eff = gate * prot if body_ex_prot_tau < 0.0 else torch.where(ul > uc_body, gate * prot, gate)
                    else:
                        gate_eff = gate
                    z_final = torch.where(zc <= z0_c, zc, z0_c + (1.0 - gate_eff) * uc_body + gate_eff * ul)
                d_out_q = c_min + z_final * sc

            out_d[q0:q1] = d_out_q.cpu().numpy()

        return out_d[:, hp_slice] if hp_slice is not None else out_d


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

    for fold in FOLDS:
        d = mod_mix.load_fold(fold, split, plan)
        hp = d["hp"]
        sub = d["sub"]
        qi = np.unique(sub.gallery_query_ids)
        queries = sub.queries[qi].astype(np.float32)

        row_meta = next(r for r in train_proto["folds"] if r["fold"] == fold)
        with np.load(ROOT / row_meta["meta"]["path"], allow_pickle=False) as zm:
            meta = {k: zm[k].copy() for k in ("split", "sources", "image_paths", "attribute_names")}
        val_idx = np.flatnonzero(meta["split"] == "val")
        raw_probs, flip_probs, post_probs = load_member_probs_raw(spec_obj, fold, meta, val_idx)
        m_std_all = np.std(np.stack(post_probs, axis=0), axis=0).astype(np.float32)

        ef = np.load(feat_dir / f"eva_full_val_{fold}.npz")
        hf = np.load(feat_dir / f"hp_full_val_{fold}.npz")

        u_pe_m = unit_features_fp64(d["features"])
        u_eva_m = unit_features_fp64(ef["features"])
        u_hp_m = unit_features_fp64(hf["features"])

        u_pe_oof[hp] = u_pe_m[hp]
        u_eva_oof[hp] = u_eva_m[hp]
        u_hp_oof[hp] = u_hp_m[hp]
        p_oof[hp] = d["p"][hp]
        m_std_oof[hp] = m_std_all[hp]
        for i in range(3):
            raw_oof[i][hp] = raw_probs[i][hp]
            flip_oof[i][hp] = flip_probs[i][hp]

        held_md = precompute_fold_mode(
            u_pe_m[hp], u_eva_m[hp], u_hp_m[hp], d["p"][hp],
            [rp[hp] for rp in raw_probs], [fp[hp] for fp in flip_probs], post_cfg, m_std_all[hp]
        )
        mixed_md = precompute_fold_mode(
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
            "eval_ctx": precompute_eval_ctx(sub, qi),
        }

    oof_md = precompute_fold_mode(
        u_pe_oof, u_eva_oof, u_hp_oof, p_oof, raw_oof, flip_oof, post_cfg, m_std_oof
    )
    qi_oof = np.unique(split.gallery_query_ids)
    queries_oof = split.queries[qi_oof].astype(np.float32)
    w_oof = np.mean([data_by_fold[f]["w"] for f in FOLDS], axis=0).astype(np.float32)
    oof_ctx = precompute_eval_ctx(split, qi_oof)
    print(f"Precomputed all 3 folds + 33,407 OOF gallery in {time.monotonic() - t_load:.1f}s", flush=True)

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

    base_clean = {
        "branches": b5_tw07,
        "anchor_spec_weights": aw_8spec,
        "ham_tau": 0.16,
        "dist_kappa": 0.18,
        "sim_diff_alpha": 0.34,
    }
    s08 = {
        **base_clean,
        "exact_seed_blend": 0.72,
        "ex_vis_alpha": 0.120,
        "anchor_alpha_q": 0.126,
        "anchor_temp_q": 0.052,
        "head_beta": 0.042,
        "head_tau": 0.22,
    }
    m11_champ = {
        **s08,
        "anchor_top_m": 11,
        "tail_mul_kap": 0.04,
        "head_dq_blend": 0.15,
        "seed_disagree_kap": 0.025,
        "seed_kap_weight": False,
    }
    m11_v2 = {
        **m11_champ,
        "exact_branch_blend": 0.22,
        "head_tau": 0.225,
    }
    b130c = {
        **m11_v2,
        "head_beta": 0.044,
    }
    ex06_k20 = {
        **b130c,
        "ex_diff_alpha": 0.06,
        "ex_kappa": 0.18,
        "dist_kappa": 0.20,
    }
    champ_v3 = {
        **ex06_k20,
        "ex_diff_alpha": 0.07,
        "ex_diff_head_blend": 0.50,
        "ham_tau": 0.17,
        "ex_kappa": 0.24,
        "seed_disagree_kap": 0.030,
    }
    champ_v4 = {
        **champ_v3,
        "head_local_lam": 0.12,
        "head_local_k": 12,
        "sim_diff_alpha2": 0.04,
    }

    b128_base = {"branches": b4, "anchor_spec_weights": aw_b128}
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
    b131c = {**champ_v4, **body_b131}
    b132c_exact = {
        **b131c,
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

    asym_b = {
        **body_b131,
        "body_nll_lam_neg": 0.00,
        "body_err_gamma": 1.40,
        "body_err_gamma_neg": 0.55,
        "body_dich_sym_loss": 0.60,
        "body_vis_scale": 1.25,
        "body_d_alpha": 0.23,
        "body_kap": 0.0,
    }

    ex_asym = {
        "exact_nll_lam": 0.44,
        "exact_nll_lam_neg": 0.26,
        "exact_dich_sym_loss": 1.30,
    }

    c04 = {**b131c, **asym_b, "body_alpha": 0.26, "nll_lam": 0.275, "nll_lam_neg": 0.215, "base_dich_sym_loss": 1.28, "dich_pos_nll_add": 0.08}
    c05 = {**b131c, **asym_b, "body_alpha": 0.26, "nll_lam": 0.28, "nll_lam_neg": 0.21, "base_dich_sym_loss": 1.28, "dich_pos_nll_add": 0.07}

    configs = [
        ("00_B131C_Pub6995", b131c),
        ("01_c04_base_only", c04),
        ("02_c05_base_only", c05),
        ("03_c04+ex42_34_ds120", {**c04, "exact_nll_lam": 0.42, "exact_nll_lam_neg": 0.34, "exact_dich_sym_loss": 1.20}),
        ("04_c04+ex43_30_ds125", {**c04, "exact_nll_lam": 0.43, "exact_nll_lam_neg": 0.30, "exact_dich_sym_loss": 1.25}),
        ("05_c04+ex44_26_ds130", {**c04, "exact_nll_lam": 0.44, "exact_nll_lam_neg": 0.26, "exact_dich_sym_loss": 1.30}),
        ("06_c05+ex43_30_ds125", {**c05, "exact_nll_lam": 0.43, "exact_nll_lam_neg": 0.30, "exact_dich_sym_loss": 1.25}),
        ("07_c05+ex44_26_ds130", {**c05, "exact_nll_lam": 0.44, "exact_nll_lam_neg": 0.26, "exact_dich_sym_loss": 1.30}),
        ("08_c04+ex43_30+a28_egneg52", {**c04, "exact_nll_lam": 0.43, "exact_nll_lam_neg": 0.30, "exact_dich_sym_loss": 1.25, "body_alpha": 0.28, "body_err_gamma_neg": 0.52}),
        ("09_c04+ex44_26+a28_egneg52", {**c04, "exact_nll_lam": 0.44, "exact_nll_lam_neg": 0.26, "exact_dich_sym_loss": 1.30, "body_alpha": 0.28, "body_err_gamma_neg": 0.52}),
        ("10_c04+ex43_30+egn97+a27", {**c04, "exact_nll_lam": 0.43, "exact_nll_lam_neg": 0.30, "exact_dich_sym_loss": 1.25, "body_alpha": 0.27, "err_gamma_neg": 0.97}),
    ]

    base_h, base_m, base_o = None, None, None
    for name, kw in configs:
        t0 = time.monotonic()
        h_map, m_map = {}, {}
        for f in FOLDS:
            fd = data_by_fold[f]
            h_map[f] = run_pipeline(fd["held"], fd["queries"], fd["w"], None, **kw)
            m_map[f] = run_pipeline(fd["mixed"], fd["queries"], fd["w"], fd["hp"], **kw)
        oof_dist = run_pipeline(oof_md, queries_oof, w_oof, None, **kw)
        rh = eval_dist_map(h_map, data_by_fold)
        rm = eval_dist_map(m_map, data_by_fold)
        ro = eval_with_ctx(oof_dist, oof_ctx)
        if base_h is None:
            base_h, base_m, base_o = rh, rm, ro
        dh = rh["macro"] - base_h["macro"]
        dm = rm["macro"] - base_m["macro"]
        do = ro["mADM"] - base_o["mADM"]
        h_meff = (rh["macro"] - rh["mAP"]) / (1.0 - rh["mAP"])
        m_meff = (rm["macro"] - rm["mAP"]) / (1.0 - rm["mAP"])
        print(
            f"{name:30s} | H:{rh['macro']:.6f}({dh:+.6f}) [Mk:{rh['Market1501']:.4f} PA:{rh['PA100k']:.4f} PE:{rh['PETA']:.4f}] AP={rh['mAP']:.4f} mE={h_meff:.4f} | "
            f"M:{rm['macro']:.6f}({dm:+.6f}) AP={rm['mAP']:.4f} mE={m_meff:.4f} | "
            f"OOF33k:{ro['mADM']:.6f}({do:+.6f}) AP={ro['mAP']:.4f} mE={ro['mE']:.4f} R1={ro['R1']:.4f} R5={ro['R5']:.4f} R10={ro['R10']:.4f} | {time.monotonic()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()








