"""Evaluate the deployed B139C-family rerank (package_v14 code, parametrised copy) on the fold-model surfaces.

Surfaces (all source-clean fold models, cached by build_inputs_v1.py):
  oof    : 33,407-image OOF gallery, 3,462 queries (each image predicted by the fold model that never saw its source)
  mixed  : per held-out source, the fold model predicts all 33,407 images; held-out queries vs held-out rows
  simpub : MEVID-like (>=15 matches) slice + 5 Synth-like 577x1,153 slices of the oof distances (previous worker's
           cached contexts)
Per-query ADM / AP / R1 arrays are saved so paired bootstrap comparisons can be made later.

usage:
  knob_eval.py equiv                      # deployed vs parametrised (bit-exact) vs previous worker's harness (B139C)
  knob_eval.py sweep CONFIGS.json OUT.jsonl [--surfaces oof,mixed] [--shard i/n]
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True  # never write __pycache__ into package dirs (validate_package compares file sets)
HERE = Path(__file__).resolve().parent
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
S = ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle"
PKG = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/package_v14/package"
import os
INP = Path(os.environ.get("KNOB_INPUTS", str(HERE / "inputs_v1")))
FOLDS = ("Market1501", "PA100k", "PETA")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PKG))

CFG = json.loads((PKG / "assets/config.json").read_text())


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEP = _load("deployed_b127", PKG / "b127_jaccard_prob_diffusion.py")
PAR = _load("rerank_param", HERE / os.environ.get("KNOB_PARAM", "rerank_param.py"))


def make_score_fn(weights_vec: np.ndarray):
    """_b121_score_distances from package_v14 run.py with the attribute weights supplied by the surface."""
    eps0 = np.float32(CFG["bernoulli_nll_epsilon"])
    lam_cfg = CFG["bernoulli_nll_lambda"]
    ent_cfg = CFG["post"]["entropy_delta"]

    def score(probabilities, queries, attribute_names, ent_override=None, lam_override=None, err_gamma=1.0,
              uniform_weights=False, lam_neg_override=None, err_gamma_neg=None, q_pos_rho=0.0, dich_sym_loss=0.0,
              dich_pos_nll_add=0.0, pos_q_mul=1.0, _pre=None):
        if uniform_weights:
            weights = np.ones(len(attribute_names), dtype=np.float32)
        else:
            weights = np.asarray(weights_vec, dtype=np.float32)
        lam = np.float32(lam_cfg if lam_override is None else lam_override)
        lam0 = lam if lam_neg_override is None else np.float32(lam_neg_override)
        ent = ent_cfg if ent_override is None else ent_override
        if _pre is not None:
            one_minus_ent, clipped, neg, pos = _pre
        else:
            one_minus_ent = None
            clipped = np.clip(probabilities, eps0, np.float32(1.0) - eps0)
            neg = -np.log1p(-clipped).astype(np.float32, copy=False)
            pos = -np.log(clipped).astype(np.float32, copy=False)
        if ent:
            if one_minus_ent is None:
                q = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
                one_minus_ent = 1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))
            h = np.exp(one_minus_ent * float(ent))
            weights = (weights.astype(np.float64)[None, :] * (h / h.mean(axis=1, keepdims=True))).astype(np.float32)
        eg_pos = float(err_gamma)
        eg_neg = eg_pos if err_gamma_neg is None else float(err_gamma_neg)
        pmul = np.float32(pos_q_mul)
        if eg_pos != 1.0 or eg_neg != 1.0 or lam != lam0 or pmul != 1.0 or float(q_pos_rho) != 0.0 or float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:
            f0 = (clipped ** np.float32(eg_neg)).astype(np.float32, copy=False) if eg_neg != 1.0 else probabilities
            f1 = ((np.float32(1.0) - clipped) ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else (np.float32(1.0) - probabilities)
            loss0 = (np.float32(1.0) - lam0) * f0 + lam0 * neg
            loss1 = pmul * ((np.float32(1.0) - lam) * f1 + lam * pos)
            if float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:
                dich_idx = [i for i, n in enumerate(attribute_names) if str(n) in ("Gender-Female", "UpperBody-Length-Short", "LowerBody-Length-Short")]
                if float(dich_sym_loss) > 0.0 and dich_idx:
                    s_d = np.float32(dich_sym_loss)
                    lam_d0 = np.float32(float(lam) + float(dich_pos_nll_add))
                    f0_pos = (clipped[:, dich_idx] ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else probabilities[:, dich_idx]
                    loss0_pos = pmul * ((np.float32(1.0) - lam_d0) * f0_pos + lam_d0 * neg[:, dich_idx])
                    loss0 = loss0.copy()
                    loss0[:, dich_idx] = (np.float32(1.0) - s_d) * loss0[:, dich_idx] + s_d * loss0_pos
                if float(dich_pos_nll_add) != 0.0 and dich_idx:
                    dl_p = np.float32(dich_pos_nll_add)
                    loss1 = loss1.copy()
                    loss1[:, dich_idx] = loss1[:, dich_idx] + pmul * dl_p * (pos[:, dich_idx] - f1[:, dich_idx])
            d0 = weights * loss0
            d1 = weights * loss1
            base = d0.sum(axis=1, dtype=np.float32)
            delta = d1 - d0
        else:
            base = (np.float32(1.0) - lam) * (probabilities * weights).sum(axis=1, dtype=np.float32) \
                + lam * (neg * weights).sum(axis=1, dtype=np.float32)
            delta = (np.float32(1.0) - lam) * weights * (np.float32(1.0) - np.float32(2.0) * probabilities) \
                + lam * weights * (pos - neg)
        distances = np.empty((len(queries), len(probabilities)), dtype=np.float32)
        for start in range(0, len(queries), 64):
            distances[start:start + 64] = base[None, :] + queries[start:start + 64] @ delta.T
        return distances

    return score


def eval_q(dist: np.ndarray, qi: np.ndarray, norm_match: np.ndarray, g_qids: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query ADM / AP / R1 (same definition as the previous worker's eval_with_ctx)."""
    adm, ap, r1 = [], [], []
    with torch.inference_mode():
        g_t = torch.from_numpy(g_qids).to("cuda", dtype=torch.int64)
        qi_t = torch.from_numpy(qi).to("cuda", dtype=torch.int64)
        for off in range(0, len(qi), 256):
            end = min(off + 256, len(qi))
            d_t = torch.from_numpy(dist[off:end].astype(np.float64)).to("cuda")
            order = torch.argsort(d_t, dim=1, stable=True)
            nm = torch.gather(torch.from_numpy(norm_match[off:end]).to("cuda", dtype=torch.float64), 1, order)
            prefix = torch.cumsum(nm, dim=1)
            rel = g_t[order] == qi_t[off:end, None]
            counts = rel.sum(dim=1).cpu().numpy()
            _, col = torch.nonzero(rel, as_tuple=True)
            pref = prefix[rel].cpu().numpy()
            ranks_all = col.cpu().numpy().astype(np.int64) + 1
            pos = 0
            for rc in counts:
                rc = int(rc)
                ranks = ranks_all[pos:pos + rc]
                pr = pref[pos:pos + rc]
                pos += rc
                adm.append(float(np.mean(pr / ranks)))
                ap.append(float(np.mean(np.arange(1, rc + 1) / ranks)))
                r1.append(float(ranks[0] <= 1))
    return {"adm": np.asarray(adm), "ap": np.asarray(ap), "r1": np.asarray(r1)}


class Surfaces:
    def __init__(self, want: set[str]):
        sh = np.load(INP / "shared.npz", allow_pickle=False)
        self.sh = {k: sh[k] for k in sh.files}
        self.names = [str(x) for x in self.sh["attribute_names"]]
        if self.names != list(CFG["attribute_names"]):
            raise SystemExit("attribute order differs from the package config")
        self.want = want
        if "oof_norm_match" in self.sh:  # independent train-split surface carries its own eval context
            self.oof_nm = self.sh["oof_norm_match"]
            self.oof_g_qids = self.sh["oof_g_qids"].astype(np.int64)
        else:
            self.oof_nm = np.load(S / "extracted_features_v1/graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy")
            from upar.data import load_retrieval_split
            split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
            self.oof_g_qids = split.gallery_query_ids.astype(np.int64)
        if "simpub" in want or "island" in want:
            saved = torch.load(S / "extracted_features_v1/graph_cache_v1/simpub_ctxs_s5_3462.pt", map_location="cpu", weights_only=False)
            self.synth, self.mevid = saved["synth_ctxs"], saved["mevid_ctx"]

    def run(self, mod, var: dict, K: dict | None, log=lambda *a, **k: None) -> dict:
        sh = self.sh
        res = {}
        t0 = time.monotonic()
        if "oof" in self.want or "simpub" in self.want:
            kw = {} if mod is DEP else {"K": K}
            d = mod.run_b132c_pentj_body_rerank(var["oof_p"], var.get("u_pe", sh["u_pe"]), var.get("u_eva", sh["u_eva"]), var.get("u_hp", sh["u_hp"]), var["oof_mstd"],
                                                 sh["queries_oof"], self.names, make_score_fn(sh["w_oof"]), log, **kw)
            res["oof_dist"] = d
            res["oof"] = eval_q(d, sh["qi_oof"], self.oof_nm, self.oof_g_qids)
            if "simpub" in self.want:
                m = self.mevid
                res["simmev"] = eval_q(d[m["row_idx"]], m["qi"], m["norm_match"], m["g_qids"])
                syn = []
                for c in self.synth:
                    sub = d[c["row_idx"]][:, c["col_idx"]]
                    syn.append(eval_q(sub, c["qi"], c["norm_match"], c["g_qids"]))
                res["simsyn"] = syn
        if "island" in self.want:
            # Synth-like island: each 577-query x 1,153-image slice reranked as its own gallery (the hidden Synth set is
            # a small separate domain inside the combined gallery); bk pinned to the public run's value (29 rows)
            K_is = {**(K or {}), "bk": (29, 0.0)}
            isl = []
            for c in self.synth:
                cols = np.asarray(c["col_idx"])
                d = PAR.run_b132c_pentj_body_rerank(var["oof_p"][cols], sh["u_pe"][cols], sh["u_eva"][cols], sh["u_hp"][cols],
                                                    var["oof_mstd"][cols], sh["queries_oof"][c["row_idx"]], self.names,
                                                    make_score_fn(sh["w_oof"]), log, K=K_is)
                isl.append(eval_q(d, c["qi"], c["norm_match"], c["g_qids"]))
            res["island"] = isl
        if "mixed" in self.want:
            for f in FOLDS:
                kw = {} if mod is DEP else {"K": K}
                d = mod.run_b132c_pentj_body_rerank(var[f"{f}_p"], sh[f"{f}_u_pe"], sh[f"{f}_u_eva"], sh[f"{f}_u_hp"],
                                                     var[f"{f}_mstd"], sh[f"{f}_queries"], self.names,
                                                     make_score_fn(sh[f"{f}_w"]), log, **kw)
                d = d[:, sh[f"{f}_hp"]]
                res[f"mix_{f}"] = eval_q(d, sh[f"{f}_qi"], sh[f"{f}_norm_match"], sh[f"{f}_g_qids"])
        res["seconds"] = time.monotonic() - t0
        return res


def summary(res: dict) -> dict:
    out = {"seconds": round(res["seconds"], 1)}
    if "oof" in res:
        out.update(oof=res["oof"]["adm"].mean(), oof_ap=res["oof"]["ap"].mean(), oof_r1=res["oof"]["r1"].mean())
    if "simmev" in res:
        mev = res["simmev"]["adm"].mean()
        syn = float(np.mean([s["adm"].mean() for s in res["simsyn"]]))
        out.update(simmev=mev, simsyn=syn, simpub=0.5 * mev + 0.5 * syn)
    if "island" in res:
        out["island"] = float(np.mean([s["adm"].mean() for s in res["island"]]))
    if "mix_PETA" in res:
        for f in FOLDS:
            out[f"mix_{f}"] = res[f"mix_{f}"]["adm"].mean()
        out["mixed"] = float(np.mean([out[f"mix_{f}"] for f in FOLDS]))
    return {k: (float(v) if not isinstance(v, str) else v) for k, v in out.items()}


def load_var(name: str) -> dict:
    z = np.load(INP / f"{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def equiv() -> None:
    surf = Surfaces({"oof", "simpub"})
    var = load_var("b139c")
    r_dep = surf.run(DEP, var, None)
    print("deployed", summary(r_dep), flush=True)
    r_par = surf.run(PAR, var, None)
    print("param   ", summary(r_par), flush=True)
    same = np.array_equal(r_dep["oof_dist"], r_par["oof_dist"])
    print("BIT_EXACT_PARAM_VS_DEPLOYED", same, "maxabs", float(np.max(np.abs(r_dep["oof_dist"] - r_par["oof_dist"]))), flush=True)
    rep = {"deployed": summary(r_dep), "param": summary(r_par), "bit_exact": bool(same)}
    (HERE / "equiv_v1.json").write_text(json.dumps(rep, indent=1))


def sweep(cfg_path: str, out_path: str, surfaces: str, shard: str) -> None:
    cfgs = json.loads(Path(cfg_path).read_text())
    i, n = (int(x) for x in shard.split("/"))
    cfgs = [c for j, c in enumerate(cfgs) if j % n == i]
    surf = Surfaces(set(surfaces.split(",")))
    pq_dir = Path(out_path).with_suffix("")
    pq_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(out_path).exists():
        done = {json.loads(l)["name"] for l in Path(out_path).read_text().splitlines() if l.strip()}
    cache = {}
    for c in cfgs:
        if c["name"] in done:
            continue
        v = c.get("variant", "b139c")
        if v not in cache:
            cache[v] = load_var(v)
        K = c.get("K") or {}
        for key in ("ks", "anchor_w"):
            if key in K:
                K[key] = {**PAR.DEFAULT_K[key], **K[key]}
        if "branch_specs" in K:
            K["branch_specs"] = [tuple(b) for b in K["branch_specs"]]
        res = surf.run(PAR, cache[v], K)
        row = {"name": c["name"], "variant": v, "K": c.get("K"), **summary(res)}
        arrs = {}
        for key in ("oof", "simmev", "mix_Market1501", "mix_PA100k", "mix_PETA"):
            if key in res:
                arrs[key] = res[key]["adm"]
        if "simsyn" in res:
            arrs["simsyn"] = np.concatenate([s["adm"] for s in res["simsyn"]])
        if "island" in res:
            arrs["island"] = np.concatenate([s["adm"] for s in res["island"]])
        np.savez(pq_dir / f"{c['name']}.npz", **arrs)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "equiv":
        equiv()
    else:
        args = sys.argv[2:]
        surfaces = "oof"
        shard = "0/1"
        if "--surfaces" in args:
            surfaces = args[args.index("--surfaces") + 1]
        if "--shard" in args:
            shard = args[args.index("--shard") + 1]
        sweep(args[0], args[1], surfaces, shard)
