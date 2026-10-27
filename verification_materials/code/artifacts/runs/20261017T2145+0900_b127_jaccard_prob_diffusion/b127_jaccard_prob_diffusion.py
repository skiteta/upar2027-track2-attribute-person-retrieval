"""B127C Jaccard-structural continuous power-margin probability diffusion and distance smoothing.

Zero-fit, strictly symmetric gallery-graph inference module on the mutual-k20
PE-Core-L/14 feature graph:
  1. Closed-neighborhood mutual-k Jaccard similarity:
       common_ij = 2 + |N(i) cap N(j)|
       union_ij  = (c_i + 1) + (c_j + 1) - common_ij
       J_ij      = common_ij / union_ij
  2. Continuous 40-attribute power-margin kernel combined with visual cosine
     and Jaccard structural overlap:
       w_ij = I[Hamming(b_i, b_j) <= 5] * max(cos(u_i, u_j), 0) * (J_ij ** 0.80)
              * prod_{a=1}^{40} (1 - |p_ia - p_ja| ** 1.85)
     using only the gallery's own predicted probabilities, FP64 unit features,
     and mutual-k20 adjacency lists.
  3. Multi-step probability diffusion (steps=5, alpha_p=0.95) normalized by original
     mutual-k20 degree c_i, clipped to [1e-5, 1 - 1e-5].
  4. Consensus entropy-weighted Bernoulli-NLL + L1 distance scoring (entropy_delta=0.35)
     via the existing runtime scorer, followed by 1-step weighted distance smoothing
     (alpha_d=0.25) with original-degree denominator.
"""
from __future__ import annotations

import numpy as np


MAX_HAMMING = 5
MARGIN_GAMMA = 1.85
JACCARD_POWER = 0.80
PROB_DIFFUSION_ALPHA = 0.95
PROB_DIFFUSION_STEPS = 5
PROB_CLIP_EPS = 1e-5
ENTROPY_DELTA = 0.35
DIST_DIFFUSION_ALPHA = 0.25


def jaccard_cosine_margin_weights(
    probabilities: np.ndarray,
    unit_features: np.ndarray,
    neighbours: np.ndarray,
    counts: np.ndarray,
    max_hamming: int = MAX_HAMMING,
    margin_gamma: float = MARGIN_GAMMA,
    jaccard_power: float = JACCARD_POWER,
    gallery_chunk_size: int = 1024,
) -> np.ndarray:
    """Compute zero-fit symmetric Jaccard + cosine + continuous power-margin weights on mutual-k edges."""
    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(unit_features, dtype=np.float64)
    n = np.asarray(neighbours, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64)
    if (
        p.ndim != 2
        or p.shape[1] != 40
        or u.ndim != 2
        or len(u) != len(p)
        or n.ndim != 2
        or len(n) != len(p)
        or c.shape != (len(p),)
    ):
        raise ValueError("aligned probabilities, unit_features, neighbours, and counts required")
    if not np.isfinite(p).all() or not np.isfinite(u).all():
        raise ValueError("finite probabilities and unit_features required")
    rows, width = n.shape
    out = np.zeros((rows, width), dtype=np.float64)
    if width == 0 or rows == 0:
        return out
    bits = p >= 0.5
    col_idx = np.arange(width, dtype=np.int64)[None, :]
    valid_all = (col_idx < c[:, None]) & (n >= 0)
    n_neg2 = np.where(valid_all, n, -2)
    gamma = float(margin_gamma)
    j_pow = float(jaccard_power)
    for start in range(0, rows, gallery_chunk_size):
        end = min(start + gallery_chunk_size, rows)
        valid = valid_all[start:end]
        n_sub = n[start:end]
        c_sub = c[start:end, None]
        safe = np.where(valid, n_sub, 0)
        cos = np.clip(np.sum(u[start:end, None, :] * u[safe], axis=2, dtype=np.float64), 0.0, 1.0)
        diff = bits[start:end, None, :] != bits[safe]
        ham = np.count_nonzero(diff, axis=2)
        dp = np.abs(p[start:end, None, :] - p[safe])
        margin = 1.0 - (dp ** gamma if gamma != 1.0 else dp)
        w_attr = np.prod(margin, axis=2, dtype=np.float64)
        if j_pow > 0.0:
            ni = np.where(valid, n_sub, -1)
            nj = n_neg2[safe]
            shared = np.count_nonzero(ni[:, :, None, None] == nj[:, None, :, :], axis=(1, 3))
            common = 2.0 + shared.astype(np.float64)
            union = (c_sub.astype(np.float64) + 1.0) + (c[safe].astype(np.float64) + 1.0) - common
            jac = np.where(valid, common / np.maximum(union, 1.0), 0.0)
            j_term = jac ** j_pow
        else:
            j_term = 1.0
        out[start:end] = np.where(valid & (ham <= int(max_hamming)), w_attr * cos * j_term, 0.0)
    if not np.isfinite(out).all() or np.any(out < 0.0) or np.any(out > 1.0 + 1e-12):
        raise RuntimeError("invalid Jaccard-cosine-margin edge weights")
    return out


def diffuse_probabilities(
    probabilities: np.ndarray,
    neighbours: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    alpha_p: float = PROB_DIFFUSION_ALPHA,
    steps: int = PROB_DIFFUSION_STEPS,
    clip_eps: float = PROB_CLIP_EPS,
) -> np.ndarray:
    """Multi-step probability diffusion normalized by original mutual-k degree."""
    p_cur = np.array(probabilities, dtype=np.float64, copy=True)
    n = np.asarray(neighbours, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if p_cur.ndim != 2 or n.shape != w.shape or len(p_cur) != len(n) or c.shape != (len(n),):
        raise ValueError("aligned probabilities, graph, and weights required")
    valid = (np.arange(n.shape[1], dtype=np.int64)[None, :] < c[:, None]) & (n >= 0)
    safe = np.where(valid, n, 0)
    denom = np.maximum(c, 1).astype(np.float64)[:, None]
    w_norm = np.where(valid, w / denom, 0.0)
    w_sum = np.sum(w_norm, axis=1, keepdims=True, dtype=np.float64)
    a = float(alpha_p)
    if np.any(a * w_sum > 1.0 + 1e-12):
        raise RuntimeError("non-convex probability diffusion step")
    for _ in range(int(steps)):
        p_nb = np.sum(w_norm[:, :, None] * p_cur[safe], axis=1, dtype=np.float64)
        p_cur = (1.0 - a * w_sum) * p_cur + a * p_nb
    eps = float(clip_eps)
    out = np.clip(p_cur, eps, 1.0 - eps).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("nonfinite diffused probabilities")
    return out


def smooth_distances_weighted(
    distances: np.ndarray,
    neighbours: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    alpha_d: float = DIST_DIFFUSION_ALPHA,
    gallery_chunk_size: int = 256,
    query_chunk_size: int = 64,
) -> np.ndarray:
    """1-step weighted distance diffusion with deterministic 64-query chunk reduction."""
    d = np.asarray(distances, dtype=np.float32)
    n = np.asarray(neighbours, dtype=np.int64)
    c = np.asarray(counts, dtype=np.int64)
    W = np.asarray(weights, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != len(n) or n.shape != W.shape or c.shape != (len(n),):
        raise ValueError("aligned distances, graph, and weights required")
    out = np.array(d, dtype=np.float32, copy=True)
    if float(alpha_d) == 0.0:
        return out
    for start in range(0, d.shape[1], gallery_chunk_size):
        end = min(start + gallery_chunk_size, d.shape[1])
        active = np.flatnonzero(W[start:end].sum(axis=1) > 0) + start
        if not len(active):
            continue
        safe = np.maximum(n[active], 0)
        weight = W[active].astype(np.float32)
        factor = np.asarray(alpha_d, dtype=np.float32) / np.maximum(c[active], 1).astype(np.float32)
        for q0 in range(0, len(d), query_chunk_size):
            q1 = min(q0 + query_chunk_size, len(d))
            before = d[q0:q1, active]
            delta = ((d[q0:q1, safe] - before[:, :, None]) * weight[None, :, :]).sum(axis=2, dtype=np.float32)
            out[q0:q1, active] = before + factor[None, :] * delta
    if not np.isfinite(out).all():
        raise RuntimeError("nonfinite weighted distance smoothing")
    return out


# --- B132C PentJ ChampV4 + Asymmetric Body Rescue & Query-Cardinality Specificity Reranking ---

SPECTRA_DEFS = {
    "pe": (1.0, 0.0, 0.0),
    "eva": (0.0, 1.0, 0.0),
    "j25": (0.75, 0.25, 0.0),
    "j50": (0.50, 0.50, 0.0),
    "tri25": (0.5625, 0.1875, 0.25),
    "tri40": (0.40, 0.30, 0.30),
    "pe_hp": (0.65, 0.0, 0.35),
    "eva_hp": (0.0, 0.65, 0.35),
}

ANCHOR_SPEC_WEIGHTS = {
    "pe": 0.10,
    "eva": 0.10,
    "j25": 0.18,
    "j50": 0.18,
    "tri25": 0.16,
    "tri40": 0.14,
    "pe_hp": 0.07,
    "eva_hp": 0.07,
}

B134C_CONFIG = {
    "branches": [
        {"name": "pe", "k": 20, "w_pe": 1.0, "w_eva": 0.0, "w_hp": 0.0, "weight": 0.275},
        {"name": "eva", "k": 20, "w_pe": 0.0, "w_eva": 1.0, "w_hp": 0.0, "weight": 0.230},
        {"name": "j25_k28", "k": 28, "w_pe": 0.75, "w_eva": 0.25, "w_hp": 0.0, "weight": 0.235},
        {"name": "j50_k28", "k": 28, "w_pe": 0.50, "w_eva": 0.50, "w_hp": 0.0, "weight": 0.190},
        {"name": "tri25_k28", "k": 28, "w_pe": 0.5625, "w_eva": 0.1875, "w_hp": 0.25, "weight": 0.070},
    ],
    "cross_cos_alpha": 0.35,
    "cos_c_hp": 0.20,
    "max_hamming": 5,
    "ham_tau": 0.172,
    "margin_gamma": 1.83,
    "jaccard_power": 0.80,
    "prob_diffusion_alpha": 0.95,
    "stage1_steps": 3,
    "stage2_steps": 6,
    "deg_beta": 0.35,
    "tau_w": 5.0,
    "trust_gamma": 0.25,
    "std_trust_gamma": 0.40,
    "coevolve_jac_low": 0.24,
    "coevolve_jac_span": 0.30,
    "coevolve_blend": 0.55,
    "post_tau": 1.05,
    "prob_clip_eps": 1e-5,
    "entropy_delta": 0.35,
    "nll_lam": 0.275,
    "nll_lam_neg": 0.215,
    "err_gamma": 1.0,
    "err_gamma_neg": 0.97,
    "base_dich_sym_loss": 1.28,
    "dich_pos_nll_add": 0.08,
    "dich_attrs": ["Gender-Female", "UpperBody-Length-Short", "LowerBody-Length-Short"],
    "score_ens_blend": 0.45,
    "dist_diffusion_steps": 2,
    "dist_diffusion_alpha": 0.205,
    "dist_kappa": 0.20,
    "exact_ent_delta": 0.45,
    "exact_nll_lam": 0.43,
    "exact_nll_lam_neg": 0.30,
    "exact_err_gamma": 1.25,
    "exact_err_gamma_neg": 1.25,
    "exact_dich_sym_loss": 1.25,
    "exact_branch_blend": 0.22,
    "ex_diff_alpha": 0.07,
    "ex_kappa": 0.24,
    "ex_diff_head_blend": 0.50,
    "exact_seed_blend": 0.72,
    "tail_mul_kap": 0.04,
    "seed_disagree_kap": 0.030,
    "seed_kap_weight": False,
    "seed_coh_beta": 0.25,
    "ms_k1": 6,
    "ms_temp1": 0.035,
    "ms_blend": 0.22,
    "sim_diff_alpha": 0.34,
    "sim_diff_steps": 1,
    "sim_diff_alpha2": 0.04,
    "head_dq_blend": 0.15,
    "ex_vis_alpha": 0.120,
    "head_local_lam": 0.12,
    "head_local_k": 12,
    "head_beta": 0.044,
    "head_tau": 0.225,
    "body_alpha": 0.33,
    "body_k0": 10,
    "body_tau": 0.355,
    "body_pow": 1.0,
    "body_w_uni": True,
    "body_ent_delta": 0.15,
    "body_nll_lam": 0.05,
    "body_nll_lam_neg": 0.00,
    "body_err_gamma": 0.90,
    "body_err_gamma_neg": 0.75,
    "body_pos_q_mul": 0.70,
    "body_dich_sym_loss": 0.0,
    "body_d_alpha": 0.235,
    "body_kap": 0.20,
    "body_vis_scale": 1.35,
    "body_head_keep": 3.5,
    "body_head_tau": 0.55,
    "anchor_spectra": [
        {"name": k, "w_pe": SPECTRA_DEFS[k][0], "w_eva": SPECTRA_DEFS[k][1], "w_hp": SPECTRA_DEFS[k][2], "weight": ANCHOR_SPEC_WEIGHTS[k]}
        for k in ("pe", "eva", "j25", "j50", "tri25", "tri40", "pe_hp", "eva_hp")
    ],
    "anchor_top_m": 11,
    "anchor_alpha_q": 0.126,
    "anchor_temp_q": 0.052,
    "anchor_gamma_q": 2.6,
    "anchor_hub_norm": 0.35,
    "symmetric_zero_fit": True,
}
B133C_CONFIG = B134C_CONFIG

_LAST_EVA_FEATURES: np.ndarray | None = None
_LAST_HP_FEATURES: np.ndarray | None = None
_LAST_MEMBER_STD: np.ndarray | None = None


def build_pent_graphs_and_hubs(
    u_pe: np.ndarray,
    u_eva: np.ndarray,
    u_hp: np.ndarray,
    gamma_q: float = 2.6,
    chunk_size: int = 512,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Fused zero-sync strict-FP32 CUDA builder for all 5 mutual-k graphs and all 8 gallery hubness vectors."""
    import torch
    from b121_fp32_cosine_graph import strict_fp32
    from b121_graph import _deterministic_smallest

    n = len(u_pe)
    if len(u_eva) != n or len(u_hp) != n or u_pe.ndim != 2 or u_eva.ndim != 2 or u_hp.ndim != 2:
        raise ValueError("aligned 2D unit feature matrices required")
    specs = [("pe", 20), ("eva", 20), ("j25_k28", 28), ("j50_k28", 28), ("tri25_k28", 28)]
    spec_names = tuple(SPECTRA_DEFS.keys())
    if n <= 1:
        graphs = {name: (np.empty((n, 0), np.int64), np.zeros(n, np.int64)) for name, _ in specs}
        hubs = {name: np.ones(n, dtype=np.float64) for name in spec_names}
        return graphs, hubs

    pre_k = min(96, n - 1)
    g_pow = float(gamma_q)
    num_chunks = (n + chunk_size - 1) // chunk_size
    with strict_fp32(), torch.inference_mode():
        x_pe = torch.from_numpy(u_pe.astype(np.float32)).to(device="cuda", dtype=torch.float32)
        x_eva = torch.from_numpy(u_eva.astype(np.float32)).to(device="cuda", dtype=torch.float32)
        x_hp = torch.from_numpy(u_hp.astype(np.float32)).to(device="cuda", dtype=torch.float32)
        dirs_gpu = {name: torch.empty((n, min(k, n - 1)), dtype=torch.int64, device="cuda") for name, k in specs}
        has_tie_gpu = {name: torch.zeros(num_chunks, dtype=torch.bool, device="cuda") for name, _ in specs}
        hubs_gpu = {name: torch.empty(n, dtype=torch.float64, device="cuda") for name in spec_names}

        for cid, start in enumerate(range(0, n, chunk_size)):
            end = min(n, start + chunk_size)
            d_pe = x_pe[start:end] @ x_pe.T
            d_eva = x_eva[start:end] @ x_eva.T
            d_hp = x_hp[start:end] @ x_hp.T
            for h_name, (w1, w2, w3) in SPECTRA_DEFS.items():
                mat = np.float32(w1) * d_pe + np.float32(w2) * d_eva
                if w3 > 0.0:
                    mat = mat + np.float32(w3) * d_hp
                hubs_gpu[h_name][start:end] = (torch.clamp(mat, min=0.0) ** g_pow).mean(dim=1, dtype=torch.float64)

            d_j25 = np.float32(0.75) * d_pe + np.float32(0.25) * d_eva
            d_j50 = np.float32(0.50) * d_pe + np.float32(0.50) * d_eva
            d_tri25 = np.float32(0.5625) * d_pe + np.float32(0.1875) * d_eva + np.float32(0.25) * d_hp
            for g_name, mat, k in (
                ("pe", d_pe, 20),
                ("eva", d_eva, 20),
                ("j25_k28", d_j25, 28),
                ("j50_k28", d_j50, 28),
                ("tri25_k28", d_tri25, 28),
            ):
                w = min(k, n - 1)
                neg = -mat.clone()
                neg[torch.arange(end - start, device="cuda"), torch.arange(start, end, device="cuda")] = float("inf")
                vals, idx = torch.topk(neg, k=pre_k, dim=1, largest=False, sorted=True)
                dirs_gpu[g_name][start:end] = idx[:, :w]
                has_tie_gpu[g_name][cid] = torch.any(
                    (vals[:, w - 1] >= vals[:, pre_k - 1]) | torch.any(vals[:, :w] >= vals[:, 1 : w + 1], dim=1)
                )

        tie_flags = {name: has_tie_gpu[name].cpu().numpy() for name, _ in specs}
        for g_name, k in specs:
            w = min(k, n - 1)
            for cid in np.flatnonzero(tie_flags[g_name]):
                start = int(cid) * chunk_size
                end = min(n, start + chunk_size)
                if g_name == "pe":
                    mat = x_pe[start:end] @ x_pe.T
                elif g_name == "eva":
                    mat = x_eva[start:end] @ x_eva.T
                elif g_name == "j25_k28":
                    mat = np.float32(0.75) * (x_pe[start:end] @ x_pe.T) + np.float32(0.25) * (x_eva[start:end] @ x_eva.T)
                elif g_name == "j50_k28":
                    mat = np.float32(0.50) * (x_pe[start:end] @ x_pe.T) + np.float32(0.50) * (x_eva[start:end] @ x_eva.T)
                else:
                    mat = (
                        np.float32(0.5625) * (x_pe[start:end] @ x_pe.T)
                        + np.float32(0.1875) * (x_eva[start:end] @ x_eva.T)
                        + np.float32(0.25) * (x_hp[start:end] @ x_hp.T)
                    )
                neg = -mat.clone()
                neg[torch.arange(end - start, device="cuda"), torch.arange(start, end, device="cuda")] = float("inf")
                vals, idx = torch.topk(neg, k=pre_k, dim=1, largest=False, sorted=True)
                if bool(torch.all(vals[:, w - 1] < vals[:, pre_k - 1])):
                    v_np = vals.cpu().numpy()
                    i_np = idx.cpu().numpy()
                    order = np.lexsort((i_np, v_np), axis=1)[:, :w]
                    dirs_gpu[g_name][start:end] = torch.from_numpy(np.take_along_axis(i_np, order, axis=1)).to("cuda")
                else:
                    dirs_gpu[g_name][start:end] = torch.from_numpy(_deterministic_smallest(neg.cpu().numpy(), w)).to("cuda")

        row_arange = torch.arange(n, device="cuda", dtype=torch.int64)[:, None, None]
        graphs = {}
        for name, k in specs:
            w = min(k, n - 1)
            d_t = dirs_gpu[name]
            mut_t = (d_t[d_t] == row_arange).any(dim=2)
            c_t = mut_t.sum(dim=1, dtype=torch.int64)
            col_idx = torch.arange(w, device="cuda", dtype=torch.int64)[None, :]
            key = torch.where(mut_t, col_idx, torch.full_like(col_idx, w))
            order = torch.argsort(key, dim=1, stable=True)
            packed = torch.gather(d_t, 1, order)
            n_t = torch.where(col_idx < c_t[:, None], packed, torch.full_like(packed, -1))
            graphs[name] = (n_t.cpu().numpy(), c_t.cpu().numpy())
        hubs = {name: hubs_gpu[name].cpu().numpy() for name in hubs_gpu}
    return graphs, hubs


def run_b132c_pentj_body_rerank(
    probabilities: np.ndarray,
    pe_features: np.ndarray,
    eva_features: np.ndarray,
    hp_features: np.ndarray,
    member_std: np.ndarray,
    queries: np.ndarray,
    attribute_names: list[str],
    score_fn,
    log,
) -> np.ndarray:
    """Execute full GPU-accelerated B132C PentJ ChampV4 + Asymmetric Body Rescue & Query-Cardinality Reranking."""
    import time
    import torch
    from b121_fp32_cosine_graph import strict_fp32, unit_features_fp64

    global _LAST_EVA_FEATURES, _LAST_HP_FEATURES, _LAST_MEMBER_STD
    _LAST_EVA_FEATURES = np.asarray(eva_features, dtype=np.float32).copy()
    _LAST_HP_FEATURES = np.asarray(hp_features, dtype=np.float32).copy()
    _LAST_MEMBER_STD = np.asarray(member_std, dtype=np.float32).copy()

    t0 = time.monotonic()
    u_pe = unit_features_fp64(pe_features)
    u_eva = unit_features_fp64(eva_features)
    u_hp = unit_features_fp64(hp_features)

    graphs, hubs = build_pent_graphs_and_hubs(u_pe, u_eva, u_hp, gamma_q=2.6, chunk_size=512)
    t_graphs = time.monotonic() - t0
    log("b132c_graphs_done", graph_seconds=round(t_graphs, 4), rows=len(u_pe))

    branch_specs = [
        ("pe", 0.275, 1.0, 0.0, 0.0),
        ("eva", 0.230, 0.0, 1.0, 0.0),
        ("j25_k28", 0.235, 0.75, 0.25, 0.0),
        ("j50_k28", 0.190, 0.50, 0.50, 0.0),
        ("tri25_k28", 0.070, 0.5625, 0.1875, 0.25),
    ]
    p_cals: list[tuple[float, np.ndarray]] = []
    branch_gpu_info: list[tuple[str, np.float32, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]] = []
    total_active_edges = 0

    with strict_fp32(), torch.inference_mode():
        u_pe_g = torch.from_numpy(u_pe).to("cuda", dtype=torch.float64)
        u_eva_g = torch.from_numpy(u_eva).to("cuda", dtype=torch.float64)
        u_hp_g = torch.from_numpy(u_hp).to("cuda", dtype=torch.float64)
        p0_g = torch.from_numpy(probabilities).to("cuda", dtype=torch.float64)
        ms_g = torch.from_numpy(member_std).to("cuda", dtype=torch.float64)

        for b_name, bw, w_pe, w_eva, w_hp in branch_specs:
            n_b, c_b = graphs[b_name]
            rows, width = n_b.shape
            n_t = torch.from_numpy(n_b).to("cuda", dtype=torch.long)
            c_t = torch.from_numpy(c_b).to("cuda", dtype=torch.long)
            col_t = torch.arange(width, device="cuda", dtype=torch.long)[None, :]
            valid_t = (col_t < c_t[:, None]) & (n_t >= 0)
            safe_t = torch.where(valid_t, n_t, torch.zeros_like(n_t))
            n_neg2_t = torch.where(valid_t, n_t, torch.full_like(n_t, -2))

            jac_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")
            sp_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")
            for s in range(0, rows, 2048):
                e = min(rows, s + 2048)
                st_sub = safe_t[s:e]
                vt_sub = valid_t[s:e]
                dot_pe = (u_pe_g[s:e, None, :] * u_pe_g[st_sub]).sum(dim=2)
                dot_eva = (u_eva_g[s:e, None, :] * u_eva_g[st_sub]).sum(dim=2)
                dot_hp = (u_hp_g[s:e, None, :] * u_hp_g[st_sub]).sum(dim=2)
                cos_b = torch.clamp(w_pe * dot_pe + w_eva * dot_eva + w_hp * dot_hp, 0.0, 1.0)
                cos_c = torch.clamp(0.40 * dot_pe + 0.40 * dot_eva + 0.20 * dot_hp, 0.0, 1.0)
                cos_sub = 0.65 * cos_b + 0.35 * cos_c
                ni = torch.where(vt_sub, n_t[s:e], torch.full_like(n_t[s:e], -1))
                nj = n_neg2_t[st_sub]
                shared = (ni[:, :, None, None] == nj[:, None, :, :]).sum(dim=(1, 3)).to(torch.float64)
                common = 2.0 + shared
                union = (c_t[s:e, None].to(torch.float64) + 1.0) + (c_t[st_sub].to(torch.float64) + 1.0) - common
                j_sub = torch.where(vt_sub, common / torch.clamp(union, min=1.0), 0.0)
                jac_t[s:e] = j_sub
                sp_t[s:e] = torch.where(vt_sub, cos_sub * (j_sub ** 0.80), 0.0)

            def _gpu_ew(p_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                bits = p_in >= 0.5
                dp = torch.abs(p_in[:, None, :] - p_in[safe_t])
                ham = ((bits[:, None, :] != bits[safe_t]) & (dp >= 0.172)).sum(dim=2)
                w_attr = torch.prod(1.0 - (dp ** 1.83), dim=2)
                return torch.where(valid_t & (ham <= 5), w_attr * sp_t, 0.0), ham

            def _gpu_diff(p_in: torch.Tensor, ew_in: torch.Tensor, steps: int) -> torch.Tensor:
                p_cur = p_in.clone()
                w_valid = torch.where(valid_t, ew_in, 0.0)
                c_denom = torch.clamp(c_t, min=1).to(torch.float64)[:, None]
                w_raw_sum = w_valid.sum(dim=1, keepdim=True)
                denom = 0.65 * c_denom + 0.35 * torch.clamp(w_raw_sum + 5.0, min=1.0)
                w_norm = w_valid / denom
                w_sum = w_norm.sum(dim=1, keepdim=True)
                step_scale = torch.minimum(
                    torch.tensor(0.95, dtype=torch.float64, device="cuda"),
                    0.95 / torch.clamp(w_sum, min=1e-12),
                )
                conf = torch.exp(0.25 * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * torch.exp(-0.40 * ms_g)
                for _ in range(steps):
                    w_a = w_norm[:, :, None] * conf[safe_t]
                    w_a_sum = w_a.sum(dim=1)
                    scale_a = torch.where(w_a_sum > 1e-12, w_sum / torch.clamp(w_a_sum, min=1e-12), 0.0)
                    p_nb = (w_a * p_cur[safe_t]).sum(dim=1) * scale_a
                    p_cur = (1.0 - step_scale * w_sum) * p_cur + step_scale * p_nb
                return torch.clamp(p_cur, 1e-5, 1.0 - 1e-5).to(torch.float32)

            ew0_t, ham0_t = _gpu_ew(p0_g)
            p_mid_t = _gpu_diff(p0_g, ew0_t, 3).to(torch.float64)
            ew2_t, _ = _gpu_ew(p_mid_t)
            ew2_t = torch.where(ham0_t <= 5, ew2_t, 0.0)
            s0_t = ew0_t.sum(dim=1, keepdim=True)
            s2_t = ew2_t.sum(dim=1, keepdim=True)
            ew2_norm_t = torch.where(s2_t > 1e-12, ew2_t * (s0_t / torch.clamp(s2_t, min=1e-12)), 0.0)
            gate_t = torch.clamp((jac_t - 0.24) / 0.30, 0.0, 1.0)
            ew_t = 0.45 * ew0_t + 0.55 * ((1.0 - gate_t) * ew2_norm_t + gate_t * ew2_t)
            total_active_edges += int(torch.count_nonzero(ew_t > 0).item())
            pd_t = _gpu_diff(p0_g, ew_t, 6).to(torch.float64)
            q_t = torch.clamp(pd_t, 1e-6, 1.0 - 1e-6)
            p_cal_np = (
                torch.clamp(torch.sigmoid(torch.log(q_t / (1.0 - q_t)) / 1.05), 1e-5, 1.0 - 1e-5)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            p_cals.append((bw, p_cal_np))
            safe_pos_t = torch.clamp(n_t, min=0)
            c_clamp_f32 = torch.clamp(c_t, min=1).to(torch.float32)
            fac_d_t = (np.float32(0.205) / c_clamp_f32)[None, :]
            fac_dl_t = (np.float32(0.235) / c_clamp_f32)[None, :]
            fac_s_t = (np.float32(0.34) / c_clamp_f32)[None, :]
            fac_s2_t = (np.float32(0.04) / c_clamp_f32)[None, :]
            fac_ex_t = (np.float32(0.07) / c_clamp_f32)[None, :]
            branch_gpu_info.append(
                (b_name, np.float32(bw), safe_pos_t, ew_t.to(torch.float32), fac_d_t, fac_dl_t, fac_s_t, fac_s2_t, fac_ex_t, p_cal_np)
            )
            del jac_t, sp_t, ew0_t, ham0_t, p_mid_t, ew2_t, s0_t, s2_t, ew2_norm_t, gate_t, pd_t, q_t

        del u_pe_g, u_eva_g, u_hp_g, p0_g, ms_g
        torch.cuda.empty_cache()

        t_diff = time.monotonic() - t0
        log("b132c_diff_done", diff_seconds=round(t_diff - t_graphs, 4), total_active_edges=total_active_edges)

        x_pe = torch.from_numpy(u_pe.astype(np.float32)).to("cuda", dtype=torch.float32)
        x_eva = torch.from_numpy(u_eva.astype(np.float32)).to("cuda", dtype=torch.float32)
        x_hp = torch.from_numpy(u_hp.astype(np.float32)).to("cuda", dtype=torch.float32)

        p_ens = sum(bw * pc for bw, pc in p_cals).astype(np.float32)

        def _score_triple(p_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            eps = np.float32(1.1920928955078125e-07)
            q64 = np.clip(p_mat.astype(np.float64), 1e-6, 1 - 1e-6)
            one_minus_ent = 1.0 - (-(q64 * np.log(q64) + (1 - q64) * np.log1p(-q64)) / np.log(2.0))
            clipped = np.clip(p_mat, eps, np.float32(1.0) - eps)
            neg = -np.log1p(-clipped).astype(np.float32, copy=False)
            pos = -np.log(clipped).astype(np.float32, copy=False)
            pre = (one_minus_ent, clipped, neg, pos)
            try:
                d_base = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    lam_override=0.275,
                    lam_neg_override=0.215,
                    err_gamma=1.0,
                    err_gamma_neg=0.97,
                    dich_sym_loss=1.28,
                    dich_pos_nll_add=0.08,
                    _pre=pre,
                )
                d_ex = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    ent_override=0.45,
                    lam_override=0.43,
                    lam_neg_override=0.30,
                    err_gamma=1.25,
                    err_gamma_neg=1.25,
                    dich_sym_loss=1.25,
                    _pre=pre,
                )
                d_l1 = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    ent_override=0.15,
                    lam_override=0.05,
                    lam_neg_override=0.0,
                    err_gamma=0.90,
                    err_gamma_neg=0.75,
                    uniform_weights=True,
                    dich_sym_loss=0.0,
                    pos_q_mul=0.70,
                    _pre=pre,
                )
            except TypeError:
                d_base = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    lam_override=0.275,
                    lam_neg_override=0.215,
                    err_gamma=1.0,
                    err_gamma_neg=0.97,
                    dich_sym_loss=1.28,
                    dich_pos_nll_add=0.08,
                )
                d_ex = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    ent_override=0.45,
                    lam_override=0.43,
                    lam_neg_override=0.30,
                    err_gamma=1.25,
                    err_gamma_neg=1.25,
                    dich_sym_loss=1.25,
                )
                d_l1 = score_fn(
                    p_mat,
                    queries,
                    attribute_names,
                    ent_override=0.15,
                    lam_override=0.05,
                    lam_neg_override=0.0,
                    err_gamma=0.90,
                    err_gamma_neg=0.75,
                    uniform_weights=True,
                    dich_sym_loss=0.0,
                    pos_q_mul=0.70,
                )
            return d_base, d_ex, d_l1

        base_ens, exact_ens, base_l1 = _score_triple(p_ens)
        ex_b_sum = np.zeros_like(exact_ens, dtype=np.float32)
        gpu_branches = []
        for _b_name, bw32, safe_pos_t, w_t, fac_d_t, fac_dl_t, fac_s_t, fac_s2_t, fac_ex_t, p_cal_np in branch_gpu_info:
            d_b, ex_b, d_b_l1 = _score_triple(p_cal_np)
            ex_b_sum = ex_b_sum + bw32 * ex_b
            gpu_branches.append((bw32, safe_pos_t, w_t, fac_d_t, fac_dl_t, fac_s_t, fac_s2_t, fac_ex_t, d_b, d_b_l1))
        exact_ens = (np.float32(0.78) * exact_ens + np.float32(0.22) * ex_b_sum).astype(np.float32)
        del ex_b_sum
        t_score = time.monotonic() - t0
        log("b132c_score_done", score_seconds=round(t_score - t_diff, 4))

        num_q, num_g = len(queries), len(probabilities)
        output = np.empty((num_q, num_g), dtype=np.float32)
        top_m = min(11, num_g)
        pre_m = min(64, num_g)
        hk = min(12, num_g)
        bk = min(max(10, int(round(10.0 * (num_g / 10000.0)))), num_g)

        scales_0 = {}
        for h_name in SPECTRA_DEFS:
            hv0 = hubs[h_name]
            sv0 = ((hv0.mean() ** 0.35) / (hv0 ** 0.35 + 1e-6)).astype(np.float32)
            scales_0[h_name] = torch.from_numpy(sv0).to("cuda", dtype=torch.float32)[None, :]

        kap = np.float32(0.20)
        ex_kap = np.float32(0.24)
        for q0 in range(0, num_q, 64):
            q1 = min(q0 + 64, num_q)
            d_ens_q = base_ens[q0:q1]
            d_l1_ens_q = base_l1[q0:q1]
            ex_q0 = torch.from_numpy(exact_ens[q0:q1]).to("cuda", dtype=torch.float32)
            ex_diff_gpu = torch.zeros((q1 - q0, num_g), dtype=torch.float32, device="cuda")
            d_comb_gpu = torch.zeros((q1 - q0, num_g), dtype=torch.float32, device="cuda")
            d_l1_gpu = torch.zeros((q1 - q0, num_g), dtype=torch.float32, device="cuda")
            for bw32, safe_t, w_t, fac_d_t, fac_dl_t, _fac_s_t, _fac_s2_t, fac_ex_t, d_b, d_b_l1 in gpu_branches:
                cur_ex = ex_q0.clone()
                nxt_ex = torch.empty_like(cur_ex)
                for s in range(0, num_g, 8192):
                    e = min(s + 8192, num_g)
                    before_ex = cur_ex[:, s:e]
                    diff_ex = cur_ex[:, safe_t[s:e]] - before_ex[:, :, None]
                    diff_ex = diff_ex - ex_kap * torch.abs(diff_ex)
                    delta_ex = (diff_ex * w_t[None, s:e, :]).sum(dim=2)
                    nxt_ex[:, s:e] = before_ex + fac_ex_t[:, s:e] * delta_ex
                ex_diff_gpu = ex_diff_gpu + bw32 * nxt_ex

                cur = torch.from_numpy((np.float32(0.55) * d_b[q0:q1] + np.float32(0.45) * d_ens_q).astype(np.float32)).to("cuda")
                for _ in range(2):
                    nxt = torch.empty_like(cur)
                    for s in range(0, num_g, 8192):
                        e = min(s + 8192, num_g)
                        before = cur[:, s:e]
                        diff = cur[:, safe_t[s:e]] - before[:, :, None]
                        diff = diff - kap * torch.abs(diff)
                        delta = (diff * w_t[None, s:e, :]).sum(dim=2)
                        nxt[:, s:e] = before + fac_d_t[:, s:e] * delta
                    cur = nxt
                d_comb_gpu = d_comb_gpu + bw32 * cur

                cur_l = torch.from_numpy((np.float32(0.55) * d_b_l1[q0:q1] + np.float32(0.45) * d_l1_ens_q).astype(np.float32)).to("cuda")
                for _ in range(2):
                    nxt_l = torch.empty_like(cur_l)
                    for s in range(0, num_g, 8192):
                        e = min(s + 8192, num_g)
                        before_l = cur_l[:, s:e]
                        diff_l = cur_l[:, safe_t[s:e]] - before_l[:, :, None]
                        diff_l = diff_l - kap * torch.abs(diff_l)
                        delta_l = (diff_l * w_t[None, s:e, :]).sum(dim=2)
                        nxt_l[:, s:e] = before_l + fac_dl_t[:, s:e] * delta_l
                    cur_l = nxt_l
                d_l1_gpu = d_l1_gpu + bw32 * cur_l

            ex_q = ex_diff_gpu
            ex_h_q = 0.50 * ex_q0 + 0.50 * ex_diff_gpu

            d_std = d_comb_gpu.std(dim=1, keepdim=True)
            ex_std = torch.clamp(ex_q.std(dim=1, keepdim=True), min=1e-6)
            ex_h_std = torch.clamp(ex_h_q.std(dim=1, keepdim=True), min=1e-6)
            dq_z = (d_comb_gpu - d_comb_gpu.mean(dim=1, keepdim=True)) / torch.clamp(d_std, min=1e-6)
            ex_z = (ex_q - ex_q.mean(dim=1, keepdim=True)) / ex_std
            delta_dq = (d_comb_gpu - d_comb_gpu.min(dim=1, keepdim=True).values) / torch.clamp(d_std, min=1e-6)
            delta_ex_s = (ex_q - ex_q.min(dim=1, keepdim=True).values) / ex_std
            z_base = (1.0 - 0.72) * dq_z + 0.72 * ex_z
            z_w = z_base + 0.04 * (delta_dq * delta_ex_s)
            z_comb = z_w + 0.030 * torch.abs(dq_z - ex_z)
            seed_score = z_comb * d_std
            weight_score = z_w * d_std

            if pre_m > top_m:
                vals_q, idx_q = torch.topk(seed_score, k=pre_m, dim=1, largest=False, sorted=True)
                has_q_tie = torch.any(
                    (vals_q[:, top_m - 1] >= vals_q[:, pre_m - 1])
                    | torch.any(vals_q[:, :top_m] >= vals_q[:, 1 : top_m + 1], dim=1)
                )
            else:
                has_q_tie = torch.tensor(True, device="cuda")
            if not bool(has_q_tie):
                top_idx = idx_q[:, :top_m]
            else:
                seed_np = seed_score.cpu().numpy()
                part = np.argpartition(seed_np, top_m - 1, axis=1)[:, :top_m]
                vals = np.take_along_axis(seed_np, part, axis=1)
                order = np.lexsort((part, vals), axis=1)
                top_idx_np = np.take_along_axis(part, order, axis=1)
                top_idx = torch.from_numpy(top_idx_np).to("cuda", dtype=torch.long)

            top_d = torch.gather(weight_score, 1, top_idx)
            w_m = torch.softmax(-top_d / 0.052, dim=1)
            ms_k1 = min(6, top_m)
            if ms_k1 > 0 and ms_k1 < top_m:
                w_m1 = torch.zeros_like(w_m)
                w_m1[:, :ms_k1] = torch.softmax(-top_d[:, :ms_k1] / 0.035, dim=1)
                w_m = 0.78 * w_m + 0.22 * w_m1

            xp_s = x_pe[top_idx]
            xe_s = x_eva[top_idx]
            xh_s = x_hp[top_idx]
            mm_pe = torch.einsum("qmd,qnd->qmn", xp_s, xp_s)
            mm_eva = torch.einsum("qmd,qnd->qmn", xe_s, xe_s)
            mm_hp = torch.einsum("qmd,qnd->qmn", xh_s, xh_s)
            mm_joint = torch.clamp(0.45 * mm_pe + 0.35 * mm_eva + 0.20 * mm_hp, min=0.0)
            coh = torch.einsum("qn,qmn->qm", w_m, mm_joint)
            w_m = w_m * torch.exp(0.25 * coh)
            w_m = w_m / torch.clamp(w_m.sum(dim=1, keepdim=True), min=1e-12)

            c_pe = torch.einsum("qmd,gd->qmg", xp_s, x_pe)
            c_eva = torch.einsum("qmd,gd->qmg", xe_s, x_eva)
            c_hp = torch.einsum("qmd,gd->qmg", xh_s, x_hp)
            sim_0 = torch.zeros_like(d_comb_gpu)
            for sk, sw in ANCHOR_SPEC_WEIGHTS.items():
                w1, w2, w3 = SPECTRA_DEFS[sk]
                c_s = np.float32(w1) * c_pe + np.float32(w2) * c_eva
                if w3 > 0.0:
                    c_s = c_s + np.float32(w3) * c_hp
                sim_s = torch.einsum("qm,qmg->qg", w_m, torch.clamp(c_s, min=0.0) ** 2.6) * scales_0[sk]
                sim_0 = sim_0 + np.float32(sw) * sim_s

            sim_tot = torch.zeros_like(sim_0)
            for bw32, safe_t, w_t, _fac_d_t, _fac_dl_t, fac_s_t, fac_s2_t, _fac_ex_t, _d_b, _d_b_l1 in gpu_branches:
                cur_s = sim_0.clone()
                nxt_s = torch.empty_like(cur_s)
                for s in range(0, num_g, 8192):
                    e = min(s + 8192, num_g)
                    before_s = cur_s[:, s:e]
                    diff_s = cur_s[:, safe_t[s:e]] - before_s[:, :, None]
                    delta_s = (diff_s * w_t[None, s:e, :]).sum(dim=2)
                    nxt_s[:, s:e] = before_s + fac_s_t[:, s:e] * delta_s
                nxt_s2 = torch.empty_like(nxt_s)
                for s in range(0, num_g, 8192):
                    e = min(s + 8192, num_g)
                    before_s = nxt_s[:, s:e]
                    diff_s = nxt_s[:, safe_t[s:e]] - before_s[:, :, None]
                    delta_s = (diff_s * w_t[None, s:e, :]).sum(dim=2)
                    nxt_s2[:, s:e] = before_s + fac_s2_t[:, s:e] * delta_s
                sim_tot = sim_tot + bw32 * nxt_s2

            d_out_q = d_comb_gpu - 0.126 * d_std * sim_tot
            base_ex = (1.0 - 0.15) * ex_h_q + 0.15 * (ex_h_std / torch.clamp(d_std, min=1e-6)) * d_comb_gpu
            ex_q_eff = base_ex - 0.120 * ex_h_std * sim_tot
            ex_eff_std = torch.clamp(ex_q_eff.std(dim=1, keepdim=True), min=1e-6)
            ex_min = ex_q_eff.min(dim=1, keepdim=True).values
            ex_topk = torch.topk(ex_q_eff, k=hk, dim=1, largest=False, sorted=True).values
            loc_spread = torch.clamp(ex_topk[:, -1:] - ex_min, min=1e-6)
            loc_norm = loc_spread / 0.35
            eff_scale = (1.0 - 0.12) * ex_eff_std + 0.12 * loc_norm
            delta_ex_h = (ex_q_eff - ex_min) / eff_scale
            head_exp = torch.exp(-delta_ex_h / 0.225)
            d_out_q = d_out_q - 0.044 * d_std * head_exp

            dl_std = torch.clamp(d_l1_gpu.std(dim=1, keepdim=True), min=1e-6)
            hb_exp = torch.exp(-delta_ex_h / 0.55)
            d_l1_out_q = d_l1_gpu - (1.35 * 0.126) * dl_std * sim_tot - (3.5 * 0.044) * dl_std * hb_exp
            c_min = d_out_q.min(dim=1, keepdim=True).values
            l_min = d_l1_out_q.min(dim=1, keepdim=True).values
            sc = torch.clamp(d_out_q.std(dim=1, keepdim=True), min=1e-6)
            sl = torch.clamp(d_l1_out_q.std(dim=1, keepdim=True), min=1e-6)
            zc = (d_out_q - c_min) / sc
            zl = (d_l1_out_q - l_min) / sl
            z0_c = torch.topk(zc, k=bk, dim=1, largest=False, sorted=True).values[:, -1:]
            z0_l = torch.topk(zl, k=bk, dim=1, largest=False, sorted=True).values[:, -1:]
            uc = torch.clamp(zc - z0_c, min=0.0)
            ul = torch.clamp(zl - z0_l, min=1e-5)
            gate = 0.33 * (1.0 - torch.exp(-uc / 0.355))
            z_final = torch.where(zc <= z0_c, zc, z0_c + (1.0 - gate) * uc + gate * ul)
            d_out_q = c_min + z_final * sc
            output[q0:q1] = d_out_q.cpu().numpy()

    if not np.isfinite(output).all() or output.shape != (num_q, num_g):
        raise RuntimeError("nonfinite or misshapen B132C output distances")
    log(
        "b132c_pentj_body_rerank",
        graph_seconds=round(t_graphs, 4),
        total_seconds=round(time.monotonic() - t0, 4),
        total_active_edges=total_active_edges,
        branches=5,
    )
    return output




