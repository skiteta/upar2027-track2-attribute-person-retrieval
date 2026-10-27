"""Final-round sweep (2026-10-26): combo_sweep.py semantics (same inputs, member pooling, attr_wmul, K overrides, eval) plus
per-query distance-level fusion of several rerank outputs.
usage: fr_sweep.py SURFACE(val|train) CONFIGS.json OUT.jsonl [--shard i/n]
  env KNOB_PARAM=<abs path to rerank_param5_fr.py>  PYTHONDONTWRITEBYTECODE=1
config: as combo_sweep.py ({"name", "K", "attr_wmul", "dw", "post"}; optional "eva": npz path with eva_t12 + u_eva,
  "{surface}" substituted, replacing the EVA member), or a fusion
  {"name": ..., "fuse": {"parts": [part, part, ...], "w": [w0, w1, ...], "mode": "z"}}
  where each part is a combo_sweep-style dict (K / attr_wmul / dw / post) and the fused distance row of a query is
  sum_i w_i * (d_i - mean_row(d_i)) / std_row(d_i)  (per-query row statistics only; z of a single part keeps its ranking).
Part distances are cached in RAM within one process (keyed by their JSON), so put fusions that share parts in one shard.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
KD = Path("/home/syouya/ml/lab/attribute_based_person_retrieval/artifacts/runs/20261021T2250+0900_takeover_knob_search")
surface = sys.argv[1]
cfg_path, out_path = Path(sys.argv[2]), Path(sys.argv[3])
shard = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
sys.argv = [sys.argv[0], surface, "/dev/null"]
if surface == "train":
    os.environ["KNOB_INPUTS"] = str(KD / "train_surface/inputs")
sys.path.insert(0, str(KD))
import mw_sweep as mw  # noqa: E402

ke = mw.ke
GROUPS = {"all": list(range(40)), "colors": list(range(8, 20)) + list(range(21, 33)), "head": [0, 1, 2, 3, 4, 5, 6],
          "type": [7, 20, 33, 34], "acc": [35, 36, 37, 38, 39], "ub_colors": list(range(8, 20)), "lb_colors": list(range(21, 33))}
BASE_K = {"ham_tau": 0.17, "margin": 1.85, "nd": {"q": 0.85, "s": 0.01, "steps": 8}}
AGROUPS = {"age": [0, 1, 2], "gender": [3], "hair": [4, 5, 6], "ub_len": [7], "ub_color": list(range(8, 20)), "lb_len": [20],
           "lb_color": list(range(21, 33)), "lb_type": [33, 34], "acc": [35, 36, 37, 38, 39],
           "acc_backpack": [35], "acc_bag": [36], "acc_glasses": [37, 38], "acc_hat": [39]}
CACHE_MAX = int(os.environ.get("FR_CACHE_MAX", "8"))


def main() -> None:
    cfgs = json.loads(cfg_path.read_text())
    i, n = (int(x) for x in shard.split("/"))
    cfgs = [c for j, c in enumerate(cfgs) if j % n == i]
    sh = {k: v for k, v in np.load(ke.INP / "shared.npz", allow_pickle=False).items()}
    names = [str(x) for x in sh["attribute_names"]]
    allm = mw.members_all()
    raw = [allm["pe_b137c"], allm["hp_ltype2"], allm["eva_t12"]]
    if surface == "train":
        nm, gq = sh["oof_norm_match"], sh["oof_g_qids"].astype(np.int64)
    else:
        ctx = mw.S / "extracted_features_v1/graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy"
        if not ctx.exists():  # moved to the regenerable-cache archive on 10/24 (read-only use)
            ctx = Path("/mnt/ARCHIVE/attribute_based_person_retrieval/20261024_regenerable_cache_backup") / ctx.relative_to(mw.ROOT)
        nm = np.load(ctx)
        from upar.data import load_retrieval_split
        gq = load_retrieval_split(mw.ROOT / "starter_kit/data/annotations/task2/val").gallery_query_ids.astype(np.int64)
    cnt = np.bincount(gq, minlength=int(max(gq.max(), sh["qi_oof"].max())) + 1)[sh["qi_oof"]]
    done = {json.loads(l)["name"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    pq = out_path.with_suffix("")
    pq.mkdir(parents=True, exist_ok=True)
    POST0 = mw.POST
    base_w = mw.anat10()
    base_w[0, :] -= 0.05
    base_w[2, :] -= 0.05
    base_w[1, :] += 0.10
    cache: dict[str, np.ndarray] = {}
    eqc: dict[str, np.ndarray] = {}  # per-query log E_q of parts run with K["log_eq"]

    def dist_for(c: dict) -> np.ndarray | None:
        key = json.dumps({k: c.get(k) for k in ("K", "attr_wmul", "dw", "post", "eva")}, sort_keys=True)
        if key in cache:
            return cache[key]
        w = base_w.copy()
        for g, (dpe, dhp, deva) in (c.get("dw") or {}).items():
            cols = GROUPS[g]
            w[0, cols] += dpe
            w[1, cols] += dhp
            w[2, cols] += deva
        if (w <= 0).any():
            return None
        K = {**BASE_K, **json.loads(json.dumps(c.get("K") or {}))}
        for kk in ("ks", "anchor_w"):
            if kk in K:
                K[kk] = {**ke.PAR.DEFAULT_K[kk], **K[kk]}
        mw.POST = c.get("post") or POST0
        raw_c, u_eva = raw, sh["u_eva"]
        if c.get("eva"):  # 10/27: EVA member from another npz (eva_t12 probabilities + u_eva features, surface row order)
            ze = np.load(c["eva"].replace("{surface}", surface))
            raw_c, u_eva = [raw[0], raw[1], ze["eva_t12"]], ze["u_eva"]
        p, mstd = mw.pool(raw_c, w)
        wv = sh["w_oof"].astype(np.float64).copy()
        for g, mult in (c.get("attr_wmul") or {}).items():
            wv[AGROUPS[g] if g in AGROUPS else [int(g[1:])]] *= mult
        d = ke.PAR.run_b132c_pentj_body_rerank(p, sh["u_pe"], u_eva, sh["u_hp"], mstd, sh["queries_oof"], names,
                                               ke.make_score_fn(wv.astype(np.float32)), lambda *a, **k: None, K=K)
        if len(cache) >= CACHE_MAX:
            k0 = next(iter(cache))
            cache.pop(k0)
            eqc.pop(k0, None)
        cache[key] = d
        if K.get("log_eq"):
            eqc[key] = ke.PAR._FR_LOG_EQ.copy()
        return d

    for c in cfgs:
        if c["name"] in done:
            continue
        if c.get("fuse"):
            fz = c["fuse"]
            d = None
            wts = [np.full((len(sh["qi_oof"]), 1), float(x)) for x in fz["w"]]
            if fz.get("gate"):  # two parts: per-query weight g_q on part 1 and 1 - g_q on part 0, g_q from part 0's log E_q
                gt = fz["gate"]
                dist_for(fz["parts"][0])
                k0 = json.dumps({k: fz["parts"][0].get(k) for k in ("K", "attr_wmul", "dw", "post")}, sort_keys=True)
                gq_ = gt["lo"] + (gt["hi"] - gt["lo"]) / (1.0 + np.exp(-(eqc[k0] - np.log(gt["c"])) / gt["s"]))
                wts = [(1.0 - gq_)[:, None], gq_[:, None]]
            for part, wt in zip(fz["parts"], wts):
                dp = dist_for(part)
                if dp is None:
                    break
                dp64 = dp.astype(np.float64)
                z = (dp64 - dp64.mean(axis=1, keepdims=True)) / np.maximum(dp64.std(axis=1, keepdims=True), 1e-9)
                d = wt * z if d is None else d + wt * z
            if d is None:
                print("skip", c["name"], flush=True)
                continue
        else:
            d = dist_for(c)
            if d is None:
                print("skip non-positive weights", c["name"], flush=True)
                continue
        if (c.get("K") or {}).get("log_eq") and getattr(ke.PAR, "_FR_LOG_EQ", None) is not None:
            np.save(pq / f"{c['name']}_logeq.npy", ke.PAR._FR_LOG_EQ)
        adm = ke.eval_q(d, sh["qi_oof"], nm, gq)["adm"]
        np.savez(pq / f"{c['name']}.npz", oof=adm)
        row = {"name": c["name"], "oof": float(adm.mean()), "hi15": float(adm[cnt >= 15].mean()),
               "hi40": float(adm[cnt >= 40].mean()), "lo1_4": float(adm[cnt <= 4].mean())}
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
