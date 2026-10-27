"""Independent bit-exact reference for B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR capture.

Verifies:
  1. Captured inputs (probabilities, queries, image_paths, attribute_names) and features.npy
     are bit-for-bit identical to the pinned B121/B124C reference capture.
  2. Captured eva_features.npy and member_std.npy match extra_capture.json.
  3. Independently recomputes FP64 unit features, 4-branch mutual-k graphs (verifying the PE graph
     is bit-identical to the pinned reference_graph), 4-branch gallery hubness, cross-backbone
     structural terms, Hybrid Co-Evolution (hybrid_v1) probability diffusion, consensus-entropy
     base distances (via gallery_feature_replay.independent_base in 64-query chunks), 2-step weighted
     distance smoothing, and hubness-normalized multi-anchor visual kernel reranking without importing
     the packaged b127_jaccard_prob_diffusion.py module.
  4. Compares bit-for-bit against the captured FP32 distances (PASS only if 0 unequal float32
     distance values and 0 queries with different ranks across all 3,462 x 33,407 entries).

usage: reference_b127c.py CAPTURE_DIR OUTPUT_DIR
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/package_v1/package"))

from b121_fp32_cosine_graph import strict_fp32
from b121_gpu_graph import _mutual_from_directed
from b121_graph import _deterministic_smallest
from upar import gallery_feature_replay
from upar.gallery_feature_package import pin, sha, write_new
from upar.numpy_blas_scope import numpy_blas_threads


def unit_features_ref(features: np.ndarray) -> np.ndarray:
    x = np.array(features, dtype=np.float64, copy=True, order="C")
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    np.divide(x, scale, out=x)
    np.divide(x, np.sqrt(np.sum(x * x, axis=1, keepdims=True)), out=x)
    return x


SPECTRA_REF = {
    "pe": (1.0, 0.0, 0.0),
    "eva": (0.0, 1.0, 0.0),
    "j25": (0.75, 0.25, 0.0),
    "j50": (0.50, 0.50, 0.0),
    "tri25": (0.5625, 0.1875, 0.25),
    "tri40": (0.40, 0.30, 0.30),
    "pe_hp": (0.65, 0.0, 0.35),
    "eva_hp": (0.0, 0.65, 0.35),
}

ANCHOR_WEIGHTS_REF = {
    "pe": 0.10,
    "eva": 0.10,
    "j25": 0.18,
    "j50": 0.18,
    "tri25": 0.16,
    "tri40": 0.14,
    "pe_hp": 0.07,
    "eva_hp": 0.07,
}


def compute_hubs_ref(x_pe: torch.Tensor, x_eva: torch.Tensor, x_hp: torch.Tensor, gamma_q: float = 2.6) -> dict[str, np.ndarray]:
    n = len(x_pe)
    hubs = {name: np.empty(n, dtype=np.float64) for name in SPECTRA_REF}
    with strict_fp32(), torch.inference_mode():
        for s in range(0, n, 512):
            e = min(s + 512, n)
            d_pe = x_pe[s:e] @ x_pe.T
            d_eva = x_eva[s:e] @ x_eva.T
            d_hp = x_hp[s:e] @ x_hp.T
            for h_name, (w1, w2, w3) in SPECTRA_REF.items():
                mat = np.float32(w1) * d_pe + np.float32(w2) * d_eva
                if w3 > 0.0:
                    mat = mat + np.float32(w3) * d_hp
                hubs[h_name][s:e] = (torch.clamp(mat, min=0.0) ** gamma_q).mean(dim=1, dtype=torch.float64).cpu().numpy()
    return hubs


def build_graphs_and_hubs_ref(
    u_pe: np.ndarray, u_eva: np.ndarray, u_hp: np.ndarray, gamma_q: float = 2.6
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    n = len(u_pe)
    specs = [("pe", 20), ("eva", 20), ("j25_k28", 28), ("j50_k28", 28), ("tri25_k28", 28)]
    dirs = {name: np.empty((n, k), np.int64) for name, k in specs}
    hubs = {name: np.empty(n, dtype=np.float64) for name in SPECTRA_REF}
    pre_k = 96
    with strict_fp32(), torch.inference_mode():
        x_pe = torch.from_numpy(u_pe.astype(np.float32)).to("cuda", dtype=torch.float32)
        x_eva = torch.from_numpy(u_eva.astype(np.float32)).to("cuda", dtype=torch.float32)
        x_hp = torch.from_numpy(u_hp.astype(np.float32)).to("cuda", dtype=torch.float32)
        for s in range(0, n, 512):
            e = min(s + 512, n)
            d_pe = x_pe[s:e] @ x_pe.T
            d_eva = x_eva[s:e] @ x_eva.T
            d_hp = x_hp[s:e] @ x_hp.T
            for h_name, (w1, w2, w3) in SPECTRA_REF.items():
                mat = np.float32(w1) * d_pe + np.float32(w2) * d_eva
                if w3 > 0.0:
                    mat = mat + np.float32(w3) * d_hp
                hubs[h_name][s:e] = (
                    (torch.clamp(mat, min=0.0) ** gamma_q).mean(dim=1, dtype=torch.float64).cpu().numpy()
                )
            d_j25 = np.float32(0.75) * d_pe + np.float32(0.25) * d_eva
            d_j50 = np.float32(0.50) * d_pe + np.float32(0.50) * d_eva
            d_tri25 = np.float32(0.5625) * d_pe + np.float32(0.1875) * d_eva + np.float32(0.25) * d_hp
            for g_name, mat, w in (
                ("pe", d_pe, 20),
                ("eva", d_eva, 20),
                ("j25_k28", d_j25, 28),
                ("j50_k28", d_j50, 28),
                ("tri25_k28", d_tri25, 28),
            ):
                neg = -mat.clone()
                neg[torch.arange(e - s, device="cuda"), torch.arange(s, e, device="cuda")] = float("inf")
                vals, idx = torch.topk(neg, k=pre_k, dim=1, largest=False, sorted=True)
                if bool(torch.all(vals[:, w - 1] < vals[:, pre_k - 1])):
                    v_np, i_np = vals.cpu().numpy(), idx.cpu().numpy()
                    order = np.lexsort((i_np, v_np), axis=1)[:, :w]
                    dirs[g_name][s:e] = np.take_along_axis(i_np, order, axis=1)
                else:
                    dirs[g_name][s:e] = _deterministic_smallest(neg.cpu().numpy(), w)
    return {name: _mutual_from_directed(dirs[name]) for name, _ in specs}, hubs


def struct_terms_ref(
    u_pe: np.ndarray,
    u_eva: np.ndarray,
    u_hp: np.ndarray,
    n_b: np.ndarray,
    c_b: np.ndarray,
    w_pe: float,
    w_eva: float,
    w_hp: float,
    cross_alpha: float,
    j_pow: float,
    cos_c_hp: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    rows, width = n_b.shape
    col = np.arange(width, dtype=np.int64)[None, :]
    valid_all = (col < c_b[:, None]) & (n_b >= 0)
    n_neg2 = np.where(valid_all, n_b, -2)
    with torch.inference_mode():
        n_t = torch.from_numpy(n_b).to("cuda", dtype=torch.long)
        c_t = torch.from_numpy(c_b).to("cuda", dtype=torch.long)
        n_neg2_t = torch.from_numpy(n_neg2).to("cuda", dtype=torch.long)
        valid_t = torch.from_numpy(valid_all).to("cuda")
        safe_t = torch.where(valid_t, n_t, torch.zeros_like(n_t))
        upe_g = torch.from_numpy(u_pe).to("cuda", dtype=torch.float64)
        ueva_g = torch.from_numpy(u_eva).to("cuda", dtype=torch.float64)
        uhp_g = torch.from_numpy(u_hp).to("cuda", dtype=torch.float64)
        dot_pe = torch.empty((rows, width), dtype=torch.float64, device="cuda")
        dot_eva = torch.empty((rows, width), dtype=torch.float64, device="cuda")
        dot_hp = torch.empty((rows, width), dtype=torch.float64, device="cuda")
        for s in range(0, rows, 4096):
            e = min(s + 4096, rows)
            st = safe_t[s:e]
            dot_pe[s:e] = (upe_g[s:e, None, :] * upe_g[st]).sum(dim=2)
            dot_eva[s:e] = (ueva_g[s:e, None, :] * ueva_g[st]).sum(dim=2)
            dot_hp[s:e] = (uhp_g[s:e, None, :] * uhp_g[st]).sum(dim=2)
        cos_b = torch.clamp(w_pe * dot_pe + w_eva * dot_eva + w_hp * dot_hp, 0.0, 1.0)
        rem = 0.5 * (1.0 - float(cos_c_hp))
        cos_c = torch.clamp(rem * dot_pe + rem * dot_eva + float(cos_c_hp) * dot_hp, 0.0, 1.0)
        cos_t = (1.0 - cross_alpha) * cos_b + cross_alpha * cos_c
        jac_t = torch.zeros((rows, width), dtype=torch.float64, device="cuda")
        for s in range(0, rows, 2048):
            e = min(s + 2048, rows)
            v_sub = valid_t[s:e]
            s_t = safe_t[s:e]
            ni_t = torch.where(v_sub, n_t[s:e], torch.full_like(n_t[s:e], -1))
            nj_t = n_neg2_t[s_t]
            shared = (ni_t[:, :, None, None] == nj_t[:, None, :, :]).sum(dim=(1, 3)).to(torch.float64)
            common = 2.0 + shared
            union = (c_t[s:e, None].to(torch.float64) + 1.0) + (c_t[s_t].to(torch.float64) + 1.0) - common
            jac_t[s:e] = torch.where(v_sub, common / torch.clamp(union, min=1.0), 0.0)
        sp_t = torch.where(valid_t, cos_t * (jac_t ** j_pow), 0.0)
        return jac_t.cpu().numpy(), sp_t.cpu().numpy()


def edge_weights_ref(
    p: np.ndarray, n: np.ndarray, c: np.ndarray, sp: np.ndarray, max_ham: int, ham_tau: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    p64 = np.asarray(p, dtype=np.float64)
    valid = (np.arange(n.shape[1], dtype=np.int64)[None, :] < c[:, None]) & (n >= 0)
    safe = np.where(valid, n, 0)
    bits = p64 >= 0.5
    dp = np.abs(p64[:, None, :] - p64[safe])
    ham = np.count_nonzero((bits[:, None, :] != bits[safe]) & (dp >= float(ham_tau)), axis=2)
    w_attr = np.prod(1.0 - (dp ** gamma), axis=2, dtype=np.float64)
    return np.where(valid & (ham <= max_ham), w_attr * sp, 0.0), ham


def diffuse_damped_ref(
    p: np.ndarray,
    n: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    alpha_p: float,
    steps: int,
    deg_beta: float,
    tau_w: float,
    trust_gamma: float,
    m_std: np.ndarray,
    std_trust_gamma: float,
    clip_eps: float,
) -> np.ndarray:
    p_cur = np.array(p, dtype=np.float64, copy=True)
    valid = (np.arange(n.shape[1], dtype=np.int64)[None, :] < c[:, None]) & (n >= 0)
    safe = np.where(valid, n, 0)
    w_valid = np.where(valid, w, 0.0)
    c_denom = np.maximum(c, 1).astype(np.float64)[:, None]
    w_raw_sum = np.sum(w_valid, axis=1, keepdims=True, dtype=np.float64)
    denom = (1.0 - deg_beta) * c_denom + deg_beta * np.maximum(w_raw_sum + tau_w, 1.0)
    w_norm = w_valid / denom
    w_sum = np.sum(w_norm, axis=1, keepdims=True, dtype=np.float64)
    step_scale = np.minimum(alpha_p, 0.95 / np.maximum(w_sum, 1e-12))
    conf = np.exp(trust_gamma * (1.0 - 4.0 * p_cur * (1.0 - p_cur))) * np.exp(
        -std_trust_gamma * np.asarray(m_std, dtype=np.float64)
    )
    for _ in range(steps):
        w_attr = w_norm[:, :, None] * conf[safe]
        w_attr_sum = np.sum(w_attr, axis=1, dtype=np.float64)
        scale_attr = np.where(w_attr_sum > 1e-12, w_sum / np.maximum(w_attr_sum, 1e-12), 0.0)
        p_nb = np.sum(w_attr * p_cur[safe], axis=1, dtype=np.float64) * scale_attr
        p_cur = (1.0 - step_scale * w_sum) * p_cur + step_scale * p_nb
    return np.clip(p_cur, clip_eps, 1.0 - clip_eps).astype(np.float32)


def calibrate_ref(p: np.ndarray, tau: float, clip_eps: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(q / (1.0 - q)) / tau
    return np.clip(1.0 / (1.0 + np.exp(-logit)), clip_eps, 1.0 - clip_eps).astype(np.float32)


def independent_exact_base(
    probabilities: np.ndarray,
    queries: np.ndarray,
    attribute_names: np.ndarray,
    config: dict,
    ent: float = 0.45,
    lam: float = 0.40,
    err_gamma: float = 1.25,
    uniform_weights: bool = False,
    lam_neg: float | None = None,
    err_gamma_neg: float | None = None,
    q_pos_rho: float = 0.0,
    dich_sym_loss: float = 0.0,
    dich_pos_nll_add: float = 0.0,
    pos_q_mul: float = 1.0,
) -> np.ndarray:
    p, q = np.asarray(probabilities, np.float32), np.asarray(queries, np.float32)
    order = [config["attribute_names"].index(str(name)) for name in attribute_names]
    if uniform_weights:
        weights = np.ones(len(attribute_names), dtype=np.float32)
    else:
        weights = np.asarray(config["attribute_distance_weights"], np.float32)[order]
    f64 = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    entropy = -(f64 * np.log(f64) + (1 - f64) * np.log1p(-f64)) / np.log(2.0)
    factor = np.exp((1.0 - entropy) * float(ent))
    weights = (weights.astype(np.float64)[None, :] * (factor / factor.mean(axis=1, keepdims=True))).astype(np.float32)
    eps, lam32 = np.float32(config["bernoulli_nll_epsilon"]), np.float32(lam)
    lam0_32 = lam32 if lam_neg is None else np.float32(lam_neg)
    clipped = np.clip(p, eps, np.float32(1) - eps)
    neg, pos = -np.log1p(-clipped).astype(np.float32), -np.log(clipped).astype(np.float32)
    eg_pos = float(err_gamma)
    eg_neg = eg_pos if err_gamma_neg is None else float(err_gamma_neg)
    pmul = np.float32(pos_q_mul)
    f0 = (clipped ** np.float32(eg_neg)).astype(np.float32) if eg_neg != 1.0 else p
    f1 = ((np.float32(1) - clipped) ** np.float32(eg_pos)).astype(np.float32) if eg_pos != 1.0 else (np.float32(1) - p)
    loss0 = (np.float32(1) - lam0_32) * f0 + lam0_32 * neg
    loss1 = pmul * ((np.float32(1) - lam32) * f1 + lam32 * pos)
    if float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:
        dich_idx = [
            i
            for i, n in enumerate(attribute_names)
            if str(n) in ("Gender-Female", "UpperBody-Length-Short", "LowerBody-Length-Short")
        ]
        if float(dich_sym_loss) > 0.0 and dich_idx:
            s_d = np.float32(dich_sym_loss)
            lam_d0 = np.float32(float(lam32) + float(dich_pos_nll_add))
            f0_pos = (
                (clipped[:, dich_idx] ** np.float32(eg_pos)).astype(np.float32)
                if eg_pos != 1.0
                else p[:, dich_idx]
            )
            loss0_pos = pmul * ((np.float32(1) - lam_d0) * f0_pos + lam_d0 * neg[:, dich_idx])
            loss0 = loss0.copy()
            loss0[:, dich_idx] = (np.float32(1) - s_d) * loss0[:, dich_idx] + s_d * loss0_pos
        if float(dich_pos_nll_add) != 0.0 and dich_idx:
            dl_p = np.float32(dich_pos_nll_add)
            loss1 = loss1.copy()
            loss1[:, dich_idx] = loss1[:, dich_idx] + pmul * dl_p * (pos[:, dich_idx] - f1[:, dich_idx])
    d0 = weights * loss0
    d1 = weights * loss1
    constant = d0.sum(axis=1, dtype=np.float32)
    change = d1 - d0
    return constant[None, :] + q @ change.T


def main() -> None:
    spec = json.loads((ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/recipe_v1.json").read_text())
    spec = dict(spec, candidate="B139C_ANAT_MULTIRES_SPHERICAL_HEAD_RERANK")
    package = json.loads((HERE / "package_v14/complete.json").read_text())
    capture_dir, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    assert capture_dir.parent == HERE and output.parent == HERE and not output.exists()
    report = json.loads((capture_dir / "execute_report.json").read_text())
    extra = json.loads((capture_dir / "extra_capture.json").read_text())
    if (
        report.get("candidate") != spec["candidate"]
        or report.get("artifact", {}).get("sha256") != package["zip"]["sha256"]
        or report.get("config") != package["config"]
        or report.get("device") != "cuda"
        or report.get("forced_plan") is not None
        or report.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING"
        or (report.get("rows"), report.get("queries")) != (33407, 3462)
    ):
        raise SystemExit("exact complete new CUDA capture of the built B134C ZIP required")
    for rel, expected in spec["reference_inputs"].items():
        if sha(ROOT / rel) != expected:
            raise SystemExit("changed reference input: " + rel)
    if sha(capture_dir / "features.npy") != spec["reference_inputs"][spec["reference_features"]]:
        raise SystemExit("graph reuse requires bit-identical PE feature file, no tolerance")
    if sha(capture_dir / "eva_features.npy") != extra["eva_features"]["sha256"]:
        raise SystemExit("eva_features.npy hash mismatch")
    if sha(capture_dir / "hp_features.npy") != extra["hp_features"]["sha256"]:
        raise SystemExit("hp_features.npy hash mismatch")
    if sha(capture_dir / "member_std.npy") != extra["member_std"]["sha256"]:
        raise SystemExit("member_std.npy hash mismatch")
    with (
        np.load(capture_dir / "capture.npz", allow_pickle=False) as new,
        np.load(ROOT / spec["reference_capture"], allow_pickle=False) as old,
    ):
        fields = {"probabilities", "queries", "image_paths", "attribute_names"}
        assert set(new.files) == fields and set(old.files) == fields, "capture schema differs"
        for f in ("queries", "image_paths", "attribute_names"):
            if new[f].dtype != old[f].dtype or new[f].shape != old[f].shape or new[f].tobytes() != old[f].tobytes():
                raise SystemExit("exact input identity differs: " + f)
        p, q, names = new["probabilities"], new["queries"], new["attribute_names"]
        if p.dtype != np.float32 or p.shape != (33407, 40) or not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
            raise SystemExit("invalid captured HP6@224+AnatMoE probabilities")
    features = np.load(capture_dir / "features.npy", allow_pickle=False)
    eva_features = np.load(capture_dir / "eva_features.npy", allow_pickle=False)
    hp_features = np.load(capture_dir / "hp_features.npy", allow_pickle=False)
    member_std = np.load(capture_dir / "member_std.npy", allow_pickle=False)
    with np.load(ROOT / spec["reference_graph"], allow_pickle=False) as graph_ref:
        ref_n_pe, ref_c_pe = graph_ref["neighbours"], graph_ref["counts"]
    actual = np.load(capture_dir / "distances.npy", mmap_mode="r", allow_pickle=False)
    assert actual.dtype == np.float32 and actual.shape == (3462, 33407), "complete FP32 distances required"

    output.mkdir()
    started, cpu = time.monotonic(), time.process_time()
    cfg_b132 = package["config"]["b139_anat_multires_spherical_head_rerank"]
    with numpy_blas_threads(2):
        u_pe = unit_features_ref(features)
        u_eva = unit_features_ref(eva_features)
        u_hp = unit_features_ref(hp_features)
        graphs, hubs = build_graphs_and_hubs_ref(u_pe, u_eva, u_hp, float(cfg_b132["anchor_gamma_q"]))
        if not np.array_equal(graphs["pe"][0], ref_n_pe) or not np.array_equal(graphs["pe"][1], ref_c_pe):
            raise SystemExit("PE graph differs from pinned reference_graph")

        branch_defs = [
            ("pe", 0.275, 1.0, 0.0, 0.0),
            ("eva", 0.230, 0.0, 1.0, 0.0),
            ("j25_k28", 0.235, 0.75, 0.25, 0.0),
            ("j50_k28", 0.190, 0.50, 0.50, 0.0),
            ("tri25_k28", 0.070, 0.5625, 0.1875, 0.25),
        ]
        p_cals = []
        branch_info = []
        total_active = 0
        for b_name, bw, w_pe, w_eva, w_hp in branch_defs:
            n_b, c_b = graphs[b_name]
            jac_b, sp_b = struct_terms_ref(
                u_pe, u_eva, u_hp, n_b, c_b, w_pe, w_eva, w_hp,
                float(cfg_b132["cross_cos_alpha"]), float(cfg_b132["jaccard_power"]),
                float(cfg_b132["cos_c_hp"]),
            )
            ew0, ham0 = edge_weights_ref(
                p, n_b, c_b, sp_b,
                int(cfg_b132["max_hamming"]), float(cfg_b132["ham_tau"]), float(cfg_b132["margin_gamma"]),
            )
            p_mid = diffuse_damped_ref(
                p, n_b, c_b, ew0,
                float(cfg_b132["prob_diffusion_alpha"]),
                int(cfg_b132["stage1_steps"]),
                float(cfg_b132["deg_beta"]),
                float(cfg_b132["tau_w"]),
                float(cfg_b132["trust_gamma"]),
                member_std,
                float(cfg_b132["std_trust_gamma"]),
                float(cfg_b132["prob_clip_eps"]),
            )
            ew2, _ = edge_weights_ref(
                p_mid, n_b, c_b, sp_b,
                int(cfg_b132["max_hamming"]), float(cfg_b132["ham_tau"]), float(cfg_b132["margin_gamma"]),
            )
            ew2 = np.where(ham0 <= int(cfg_b132["max_hamming"]), ew2, 0.0)
            s0 = np.sum(ew0, axis=1, keepdims=True, dtype=np.float64)
            s2 = np.sum(ew2, axis=1, keepdims=True, dtype=np.float64)
            ew2_norm = np.where(s2 > 1e-12, ew2 * (s0 / np.maximum(s2, 1e-12)), 0.0)
            gate = np.clip(
                (jac_b - float(cfg_b132["coevolve_jac_low"])) / float(cfg_b132["coevolve_jac_span"]), 0.0, 1.0
            )
            ew_co = (1.0 - gate) * ew2_norm + gate * ew2
            blend_c = float(cfg_b132["coevolve_blend"])
            ew = (1.0 - blend_c) * ew0 + blend_c * ew_co
            total_active += int(np.count_nonzero(ew > 0))
            pd = diffuse_damped_ref(
                p, n_b, c_b, ew,
                float(cfg_b132["prob_diffusion_alpha"]),
                int(cfg_b132["stage2_steps"]),
                float(cfg_b132["deg_beta"]),
                float(cfg_b132["tau_w"]),
                float(cfg_b132["trust_gamma"]),
                member_std,
                float(cfg_b132["std_trust_gamma"]),
                float(cfg_b132["prob_clip_eps"]),
            )
            p_cal = calibrate_ref(pd, float(cfg_b132["post_tau"]), float(cfg_b132["prob_clip_eps"]))
            p_cals.append((bw, p_cal))
            branch_info.append((b_name, bw, n_b, c_b, ew, p_cal))

        p_ens = sum(bw * pc for bw, pc in p_cals).astype(np.float32)
        saved = np.lib.format.open_memmap(
            output / "independent_distances.npy", mode="w+", dtype=np.float32, shape=actual.shape
        )
        unequal = rank_diff = 0
        maximum = 0.0
        num_g = len(p)
        top_m = int(cfg_b132["anchor_top_m"])
        hub_p = float(cfg_b132["anchor_hub_norm"])
        hk = min(int(cfg_b132["head_local_k"]), num_g)
        bk_raw = float(cfg_b132["body_k0"])
        bk = min(max(int(bk_raw), int(round(bk_raw * (num_g / 10000.0)))), num_g)

        with strict_fp32(), torch.inference_mode():
            gpu_b = []
            for b_name, bw, n_b, c_b, ew, p_cal in branch_info:
                n_t = torch.from_numpy(n_b).to("cuda", dtype=torch.long)
                c_t = torch.from_numpy(c_b).to("cuda", dtype=torch.long)
                safe_t = torch.clamp(n_t, min=0)
                w_t = torch.from_numpy(ew.astype(np.float32)).to("cuda", dtype=torch.float32)
                c_clamp = torch.clamp(c_t, min=1).to(torch.float32)
                fac_d_t = (np.float32(cfg_b132["dist_diffusion_alpha"]) / c_clamp)[None, :]
                fac_dl_t = (np.float32(cfg_b132["body_d_alpha"]) / c_clamp)[None, :]
                fac_s_t = (np.float32(cfg_b132["sim_diff_alpha"]) / c_clamp)[None, :]
                fac_s2_t = (np.float32(cfg_b132["sim_diff_alpha2"]) / c_clamp)[None, :]
                fac_ex_t = (np.float32(cfg_b132["ex_diff_alpha"]) / c_clamp)[None, :]
                gpu_b.append((np.float32(bw), safe_t, w_t, fac_d_t, fac_dl_t, fac_s_t, fac_s2_t, fac_ex_t, p_cal))

            x_pe = torch.from_numpy(u_pe.astype(np.float32)).to("cuda", dtype=torch.float32)
            x_eva = torch.from_numpy(u_eva.astype(np.float32)).to("cuda", dtype=torch.float32)
            x_hp = torch.from_numpy(u_hp.astype(np.float32)).to("cuda", dtype=torch.float32)

            scales_0 = {}
            for h_name in SPECTRA_REF:
                hv0 = hubs[h_name]
                sv0 = ((hv0.mean() ** hub_p) / (hv0 ** hub_p + 1e-6)).astype(np.float32)
                scales_0[h_name] = torch.from_numpy(sv0).to("cuda", dtype=torch.float32)[None, :]

            kap = np.float32(cfg_b132["dist_kappa"])
            b_kap = np.float32(cfg_b132["body_kap"])
            ex_kap = np.float32(cfg_b132["ex_kappa"])
            gp = float(cfg_b132["anchor_gamma_q"])
            es_blend = float(cfg_b132["exact_seed_blend"])
            ex_br = np.float32(cfg_b132["exact_branch_blend"])
            ex_hb = float(cfg_b132["ex_diff_head_blend"])
            tm_kap = float(cfg_b132["tail_mul_kap"])
            sk_kap = float(cfg_b132["seed_disagree_kap"])
            hdq_blend = float(cfg_b132["head_dq_blend"])
            h_loc_lam = float(cfg_b132["head_local_lam"])
            temp_q = float(cfg_b132["anchor_temp_q"])
            alp_q = float(cfg_b132["anchor_alpha_q"])
            ex_vis = float(cfg_b132["ex_vis_alpha"])
            h_beta = float(cfg_b132["head_beta"])
            h_tau = float(cfg_b132["head_tau"])
            b_alpha = float(cfg_b132["body_alpha"])
            b_tau = float(cfg_b132["body_tau"])
            b_hk = float(cfg_b132["body_head_keep"])
            b_htau = float(cfg_b132["body_head_tau"])
            b_vis_sc = float(cfg_b132["body_vis_scale"])
            ms_k1 = min(int(cfg_b132["ms_k1"]), top_m)
            ms_temp1 = float(cfg_b132["ms_temp1"])
            ms_blend = float(cfg_b132["ms_blend"])
            coh_beta = float(cfg_b132["seed_coh_beta"])

            for start in range(0, len(q), 64):
                end = min(start + 64, len(q))
                q_sub = q[start:end]
                base_ens_q = independent_exact_base(
                    p_ens, q_sub, names, package["config"],
                    ent=float(cfg_b132["entropy_delta"]),
                    lam=float(cfg_b132["nll_lam"]),
                    lam_neg=float(cfg_b132["nll_lam_neg"]),
                    err_gamma=float(cfg_b132["err_gamma"]),
                    err_gamma_neg=float(cfg_b132["err_gamma_neg"]),
                    uniform_weights=False,
                    dich_sym_loss=float(cfg_b132["base_dich_sym_loss"]),
                    dich_pos_nll_add=float(cfg_b132["dich_pos_nll_add"]),
                )
                exact_ens_q = independent_exact_base(
                    p_ens, q_sub, names, package["config"],
                    ent=float(cfg_b132["exact_ent_delta"]),
                    lam=float(cfg_b132["exact_nll_lam"]),
                    lam_neg=float(cfg_b132["exact_nll_lam_neg"]),
                    err_gamma=float(cfg_b132["exact_err_gamma"]),
                    err_gamma_neg=float(cfg_b132["exact_err_gamma_neg"]),
                    uniform_weights=False,
                    dich_sym_loss=float(cfg_b132["exact_dich_sym_loss"]),
                )
                base_l1_q = independent_exact_base(
                    p_ens, q_sub, names, package["config"],
                    ent=float(cfg_b132["body_ent_delta"]),
                    lam=float(cfg_b132["body_nll_lam"]),
                    lam_neg=float(cfg_b132["body_nll_lam_neg"]),
                    err_gamma=float(cfg_b132["body_err_gamma"]),
                    err_gamma_neg=float(cfg_b132["body_err_gamma_neg"]),
                    uniform_weights=bool(cfg_b132["body_w_uni"]),
                    dich_sym_loss=float(cfg_b132["body_dich_sym_loss"]),
                    pos_q_mul=float(cfg_b132["body_pos_q_mul"]),
                )
                ex_b_sum_q = np.zeros_like(exact_ens_q, dtype=np.float32)
                d_comb_gpu = torch.zeros((end - start, num_g), dtype=torch.float32, device="cuda")
                d_l1_gpu = torch.zeros((end - start, num_g), dtype=torch.float32, device="cuda")
                for bw32, safe_t, w_t, fac_d_t, fac_dl_t, _fac_s_t, _fac_s2_t, _fac_ex_t, p_cal in gpu_b:
                    base_b_q = independent_exact_base(
                        p_cal, q_sub, names, package["config"],
                        ent=float(cfg_b132["entropy_delta"]),
                        lam=float(cfg_b132["nll_lam"]),
                        lam_neg=float(cfg_b132["nll_lam_neg"]),
                        err_gamma=float(cfg_b132["err_gamma"]),
                        err_gamma_neg=float(cfg_b132["err_gamma_neg"]),
                        uniform_weights=False,
                        dich_sym_loss=float(cfg_b132["base_dich_sym_loss"]),
                        dich_pos_nll_add=float(cfg_b132["dich_pos_nll_add"]),
                    )
                    ex_b_q = independent_exact_base(
                        p_cal, q_sub, names, package["config"],
                        ent=float(cfg_b132["exact_ent_delta"]),
                        lam=float(cfg_b132["exact_nll_lam"]),
                        lam_neg=float(cfg_b132["exact_nll_lam_neg"]),
                        err_gamma=float(cfg_b132["exact_err_gamma"]),
                        err_gamma_neg=float(cfg_b132["exact_err_gamma_neg"]),
                        uniform_weights=False,
                        dich_sym_loss=float(cfg_b132["exact_dich_sym_loss"]),
                    )
                    l1_b_q = independent_exact_base(
                        p_cal, q_sub, names, package["config"],
                        ent=float(cfg_b132["body_ent_delta"]),
                        lam=float(cfg_b132["body_nll_lam"]),
                        lam_neg=float(cfg_b132["body_nll_lam_neg"]),
                        err_gamma=float(cfg_b132["body_err_gamma"]),
                        err_gamma_neg=float(cfg_b132["body_err_gamma_neg"]),
                        uniform_weights=bool(cfg_b132["body_w_uni"]),
                        dich_sym_loss=float(cfg_b132["body_dich_sym_loss"]),
                        pos_q_mul=float(cfg_b132["body_pos_q_mul"]),
                    )
                    ex_b_sum_q = ex_b_sum_q + bw32 * ex_b_q
                    cur = torch.from_numpy((np.float32(0.55) * base_b_q + np.float32(0.45) * base_ens_q).astype(np.float32)).to("cuda")
                    for _ in range(int(cfg_b132["dist_diffusion_steps"])):
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

                    cur_l = torch.from_numpy((np.float32(0.55) * l1_b_q + np.float32(0.45) * base_l1_q).astype(np.float32)).to("cuda")
                    for _ in range(int(cfg_b132["dist_diffusion_steps"])):
                        nxt_l = torch.empty_like(cur_l)
                        for s in range(0, num_g, 8192):
                            e = min(s + 8192, num_g)
                            before_l = cur_l[:, s:e]
                            diff_l = cur_l[:, safe_t[s:e]] - before_l[:, :, None]
                            diff_l = diff_l - b_kap * torch.abs(diff_l)
                            delta_l = (diff_l * w_t[None, s:e, :]).sum(dim=2)
                            nxt_l[:, s:e] = before_l + fac_dl_t[:, s:e] * delta_l
                        cur_l = nxt_l
                    d_l1_gpu = d_l1_gpu + bw32 * cur_l

                exact_ens_q = ((np.float32(1.0) - ex_br) * exact_ens_q + ex_br * ex_b_sum_q).astype(np.float32)
                ex_q0 = torch.from_numpy(exact_ens_q).to("cuda", dtype=torch.float32)
                ex_diff_gpu = torch.zeros_like(ex_q0)
                for bw32, safe_t, w_t, _fac_d_t, _fac_dl_t, _fac_s_t, _fac_s2_t, fac_ex_t, _p_cal in gpu_b:
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

                ex_q = ex_diff_gpu
                ex_h_q = (1.0 - ex_hb) * ex_q0 + ex_hb * ex_diff_gpu

                d_std = d_comb_gpu.std(dim=1, keepdim=True)
                ex_std = torch.clamp(ex_q.std(dim=1, keepdim=True), min=1e-6)
                ex_h_std = torch.clamp(ex_h_q.std(dim=1, keepdim=True), min=1e-6)
                dq_z = (d_comb_gpu - d_comb_gpu.mean(dim=1, keepdim=True)) / torch.clamp(d_std, min=1e-6)
                ex_z = (ex_q - ex_q.mean(dim=1, keepdim=True)) / ex_std
                delta_dq = (d_comb_gpu - d_comb_gpu.min(dim=1, keepdim=True).values) / torch.clamp(d_std, min=1e-6)
                delta_ex_s = (ex_q - ex_q.min(dim=1, keepdim=True).values) / ex_std
                z_base = (1.0 - es_blend) * dq_z + es_blend * ex_z
                z_w = z_base + tm_kap * (delta_dq * delta_ex_s)
                z_comb = z_w + sk_kap * torch.abs(dq_z - ex_z)
                seed_score = z_comb * d_std
                weight_score = z_w * d_std

                seed_np = seed_score.cpu().numpy()
                part = np.argpartition(seed_np, top_m - 1, axis=1)[:, :top_m]
                vals = np.take_along_axis(seed_np, part, axis=1)
                order = np.lexsort((part, vals), axis=1)
                top_idx = torch.from_numpy(np.take_along_axis(part, order, axis=1)).to("cuda", dtype=torch.long)

                top_d = torch.gather(weight_score, 1, top_idx)
                w_m = torch.softmax(-top_d / temp_q, dim=1)
                if ms_k1 > 0 and ms_blend > 0.0 and ms_k1 < top_m:
                    w_m1 = torch.zeros_like(w_m)
                    w_m1[:, :ms_k1] = torch.softmax(-top_d[:, :ms_k1] / ms_temp1, dim=1)
                    w_m = (1.0 - ms_blend) * w_m + ms_blend * w_m1

                xp_s = x_pe[top_idx]
                xe_s = x_eva[top_idx]
                xh_s = x_hp[top_idx]
                mm_pe = torch.einsum("qmd,qnd->qmn", xp_s, xp_s)
                mm_eva = torch.einsum("qmd,qnd->qmn", xe_s, xe_s)
                mm_hp = torch.einsum("qmd,qnd->qmn", xh_s, xh_s)
                mm_joint = torch.clamp(0.45 * mm_pe + 0.35 * mm_eva + 0.20 * mm_hp, min=0.0)
                coh = torch.einsum("qn,qmn->qm", w_m, mm_joint)
                w_m = w_m * torch.exp(coh_beta * coh)
                w_m = w_m / torch.clamp(w_m.sum(dim=1, keepdim=True), min=1e-12)

                c_pe = torch.einsum("qmd,gd->qmg", xp_s, x_pe)
                c_eva = torch.einsum("qmd,gd->qmg", xe_s, x_eva)
                c_hp = torch.einsum("qmd,gd->qmg", xh_s, x_hp)
                sim_0 = torch.zeros_like(d_comb_gpu)
                for sk, sw in ANCHOR_WEIGHTS_REF.items():
                    w1, w2, w3 = SPECTRA_REF[sk]
                    c_s = np.float32(w1) * c_pe + np.float32(w2) * c_eva
                    if w3 > 0.0:
                        c_s = c_s + np.float32(w3) * c_hp
                    sim_s = torch.einsum("qm,qmg->qg", w_m, torch.clamp(c_s, min=0.0) ** gp) * scales_0[sk]
                    sim_0 = sim_0 + np.float32(sw) * sim_s

                sim_tot = torch.zeros_like(sim_0)
                for bw32, safe_t, w_t, _fac_d_t, _fac_dl_t, fac_s_t, fac_s2_t, _fac_ex_t, _p_cal in gpu_b:
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

                d_out_q = d_comb_gpu - alp_q * d_std * sim_tot
                base_ex = (1.0 - hdq_blend) * ex_h_q + hdq_blend * (ex_h_std / torch.clamp(d_std, min=1e-6)) * d_comb_gpu
                ex_q_eff = base_ex - ex_vis * ex_h_std * sim_tot
                ex_eff_std = torch.clamp(ex_q_eff.std(dim=1, keepdim=True), min=1e-6)
                ex_min = ex_q_eff.min(dim=1, keepdim=True).values
                ex_topk = torch.topk(ex_q_eff, k=hk, dim=1, largest=False, sorted=True).values
                loc_spread = torch.clamp(ex_topk[:, -1:] - ex_min, min=1e-6)
                loc_norm = loc_spread / 0.35
                eff_scale = (1.0 - h_loc_lam) * ex_eff_std + h_loc_lam * loc_norm
                delta_ex_h = (ex_q_eff - ex_min) / eff_scale
                head_exp = torch.exp(-delta_ex_h / h_tau)
                d_out_q = d_out_q - h_beta * d_std * head_exp

                dl_std = torch.clamp(d_l1_gpu.std(dim=1, keepdim=True), min=1e-6)
                hb_exp = torch.exp(-delta_ex_h / b_htau)
                d_l1_out_q = d_l1_gpu - (b_vis_sc * alp_q) * dl_std * sim_tot - b_hk * h_beta * dl_std * hb_exp
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
                gate = b_alpha * (1.0 - torch.exp(-uc / b_tau))
                z_final = torch.where(zc <= z0_c, zc, z0_c + (1.0 - gate) * uc + gate * ul)
                expected = (c_min + z_final * sc).cpu().numpy()

                observed = np.asarray(actual[start:end])
                assert np.isfinite(expected).all() and np.isfinite(observed).all(), "nonfinite distance"
                saved[start:end] = expected
                unequal += int(np.count_nonzero(expected.view(np.uint32) != observed.view(np.uint32)))
                rank_diff += sum(
                    not np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable"))
                    for a, b in zip(expected, observed)
                )
                maximum = max(maximum, float(np.abs(observed - expected).max()))
        saved.flush()

    result = {
        "status": "PASS" if unequal == rank_diff == 0 else "FAIL",
        "candidate": spec["candidate"],
        "kind": "exact_new_b139c_anat_multires_spherical_head_same_input_CUDA32_graph_independent_reference",
        "b139_anat_multires_spherical_head_rerank": cfg_b132,
        "active_edges_total": total_active,
        "pid": os.getpid(),
        "unequal_distance_values": unequal,
        "queries_with_different_ranks": rank_diff,
        "distance_max_abs": maximum,
        "capture": {
            name: pin(capture_dir / name)
            for name in (
                "capture.npz",
                "features.npy",
                "eva_features.npy",
                "hp_features.npy",
                "member_std.npy",
                "distances.npy",
                "execute_report.json",
                "extra_capture.json",
            )
        },
        "reference": pin(output / "independent_distances.npy"),
        "package_report": pin(HERE / "package_v14/complete.json"),
        "reference_inputs": {rel: pin(ROOT / rel) for rel in spec["reference_inputs"]},
        "wall_seconds": time.monotonic() - started,
        "CPU_seconds": time.process_time() - cpu,
        "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "quality_or_gold_claim": False,
        "official_A10_runtime_guaranteed": False,
    }
    write_new(output / "report.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "unequal_distance_values",
                    "queries_with_different_ranks",
                    "distance_max_abs",
                    "wall_seconds",
                )
            }
        ),
        flush=True,
    )
    if result["status"] != "PASS":
        raise SystemExit("bit-exact independent reference failed")


if __name__ == "__main__":
    main()


