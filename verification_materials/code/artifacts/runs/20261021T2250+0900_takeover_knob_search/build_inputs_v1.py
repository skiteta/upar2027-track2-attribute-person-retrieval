"""Cache the OOF33k + Mixed(3 folds) rerank inputs for B136C / B137C / B139C (fold models only, source-clean).

Reuses the previous worker's context builders verbatim; writes one npz per variant and one shared npz with features,
queries, attribute reliability weights and the eval contexts' query ids, so knob sweeps can call the *deployed*
rerank function (package_v14 b127_jaccard_prob_diffusion.run_b132c_pentj_body_rerank) without rebuilding contexts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
S = ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle"
OUT = Path(__file__).resolve().parent / "inputs_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(S))
sys.path.insert(0, str(ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"))

import test_tri_graph_combined_knobs as tk  # noqa: E402
from test_hp6_224_stack import build_variant_contexts, pool_custom  # noqa: E402
from test_hp6_224_final_comb import build_anat_moe  # noqa: E402
from test_eva_head_and_exact_b137c import compute_eva_probs  # noqa: E402
from test_b139c_tri_colselective_multires_gpu import make_col_blend  # noqa: E402
from upar.data import load_retrieval_split  # noqa: E402
from upar.postproc import apply_post  # noqa: E402
from upar.rank_sibling_pipeline import read_json  # noqa: E402
from upar.source_fitted_gallery_edges import SPEC  # noqa: E402


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    feat_dir = S / "extracted_features_v1"
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
    folds_base, oof_base = build_variant_contexts("base_192", None, False, feat_dir, split, plan, spec_obj, train_proto)
    torch.cuda.empty_cache()
    w_oof = np.mean([folds_base[f]["w"] for f in tk.FOLDS], axis=0).astype(np.float32)
    w_anat10 = build_anat_moe([0.425, 0.340, 0.235], [0.420, 0.345, 0.235], [0.370, 0.395, 0.235], [0.355, 0.410, 0.235])
    w_c36_hp41 = w_anat10.copy()
    w_c36_hp41[:, 36] = np.asarray([0.355, 0.410, 0.235])

    n = split.gallery_size
    u_pe = np.empty((n, 1024), np.float64)
    u_eva = np.empty((n, 1024), np.float64)
    u_hp = np.empty((n, 1280), np.float64)
    fold_rows = {}
    for fold in tk.FOLDS:
        bf = folds_base[fold]
        hp = bf["hp"]
        u_pe[hp] = bf["u_pe_m"][hp]
        u_eva[hp] = bf["u_eva_m"][hp]
        f = np.load(feat_dir / f"hp4_2030_192_full_val_{fold}.npz")["features"].astype(np.float64)
        u_hp[hp] = (f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12))[hp]
        fold_rows[fold] = hp
    # Mixed surface: each fold's gallery = that fold's mixed context (its own rows + other sources' rows predicted by
    # the same fold's models); keep the per-fold raw member probs, features, queries, weights and eval qids.
    shared = {
        "qi_oof": qi_oof, "queries_oof": split.queries[qi_oof].astype(np.float32), "w_oof": w_oof,
        "u_pe": u_pe, "u_eva": u_eva, "u_hp": u_hp,
        "attribute_names": np.asarray(split.attribute_names),
    }
    for fold in tk.FOLDS:
        bf = folds_base[fold]
        f = np.load(feat_dir / f"hp4_2030_192_full_val_{fold}.npz")["features"].astype(np.float64)
        shared[f"{fold}_u_pe"] = np.asarray(bf["u_pe_m"], np.float64)
        shared[f"{fold}_u_eva"] = np.asarray(bf["u_eva_m"], np.float64)
        shared[f"{fold}_u_hp"] = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        shared[f"{fold}_hp"] = np.asarray(bf["hp"])
        shared[f"{fold}_queries"] = np.asarray(bf["queries"], np.float32)
        shared[f"{fold}_w"] = np.asarray(bf["w"], np.float32)
        shared[f"{fold}_qi"] = np.asarray(bf["eval_ctx"]["qi"])
        shared[f"{fold}_g_qids"] = np.asarray(bf["eval_ctx"]["g_qids"])
        shared[f"{fold}_norm_match"] = np.asarray(bf["eval_ctx"]["norm_match"], np.float32)
    np.savez(OUT / "shared.npz", **shared)
    print("shared saved", round(time.time() - t0, 1), flush=True)

    p2_base, p2f_base = compute_eva_probs(ks, feat_dir, folds_base, w252=0.50, tau=1.2, scale=1.0)
    BODY_12 = [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
    DICH_3 = [3, 7, 20]
    LTYPE_2 = [33, 34]
    hp_b138c = make_col_blend(z4["hp_b137c"], [(BODY_12, z4["hp_mask15_m15_r10_o103"])])
    hp_ltype2 = make_col_blend(hp_b138c, [(DICH_3, z3["hp_b136c"]), (LTYPE_2, z4["hp_mask15_m15_r00_o103"])])
    variants = {
        "b136c": (z3["pe_b136c"], z3["hp_b136c"], None, None, w_anat10),
        "b137c": (z4["pe_b137c"], z4["hp_b137c"], p2_base, p2f_base, w_anat10),
        "b139c": (z4["pe_b137c"], hp_ltype2, p2_base, p2f_base, w_c36_hp41),
    }
    for name, (p0_in, p1_in, p2_oof, p2_fold, w_m) in variants.items():
        p2_o = p2_oof if p2_oof is not None else oof_base["raw_probs"][2]
        raw = [p0_in, p1_in, p2_o]
        out = {"oof_p": pool_custom(raw, raw, post_cfg, w_m),
               "oof_mstd": np.std(np.stack([apply_post(r, post_cfg) for r in raw]), axis=0).astype(np.float32)}
        for fold in tk.FOLDS:
            bf = folds_base[fold]
            hp = bf["hp"]
            r_f = [bf["raw_base"][0].copy(), bf["raw_base"][1].copy(),
                   p2_fold[fold].copy() if p2_fold is not None else bf["raw_base"][2].copy()]
            r_f[0][hp] = p0_in[hp]
            r_f[1][hp] = p1_in[hp]
            out[f"{fold}_p"] = pool_custom(r_f, r_f, post_cfg, w_m)
            out[f"{fold}_mstd"] = np.std(np.stack([apply_post(r, post_cfg) for r in r_f]), axis=0).astype(np.float32)
        np.savez(OUT / f"{name}.npz", **out)
        print("variant saved", name, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
