"""B145C-based exploration: member-weight group shifts and rerank knobs, evaluated on one surface per process.
Base = B139C head set (pe_b137c, hp_ltype2, eva_t12), member weights w_c36_hp41 + HP +0.10 (PE -0.05, EVA -0.05),
K = B136C rerank (.17 / 1.85) + nd (.85 / .01 / 8).
usage: combo_sweep.py SURFACE(val|train) CONFIGS.json OUT.jsonl [--shard i/n]
config: {"name": ..., "dw": {"group": [dPE, dHP, dEVA], ...}, "K": {...overrides...}}
groups: all, colors (8-19, 21-32), head (0-6), type (7, 20, 33, 34), acc (35-39), ub_colors (8-19), lb_colors (21-32)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
surface = sys.argv[1]
cfg_path, out_path = Path(sys.argv[2]), Path(sys.argv[3])
shard = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
sys.argv = [sys.argv[0], surface, "/dev/null"]
if surface == "train":
    os.environ["KNOB_INPUTS"] = str(HERE / "train_surface/inputs")
sys.path.insert(0, str(HERE))
import mw_sweep as mw  # noqa: E402

ke = mw.ke
GROUPS = {"all": list(range(40)), "colors": list(range(8, 20)) + list(range(21, 33)), "head": [0, 1, 2, 3, 4, 5, 6],
          "type": [7, 20, 33, 34], "acc": [35, 36, 37, 38, 39], "ub_colors": list(range(8, 20)), "lb_colors": list(range(21, 33))}
BASE_K = {"ham_tau": 0.17, "margin": 1.85, "nd": {"q": 0.85, "s": 0.01, "steps": 8}}
AGROUPS = {"age": [0, 1, 2], "gender": [3], "hair": [4, 5, 6], "ub_len": [7], "ub_color": list(range(8, 20)), "lb_len": [20],
           "lb_color": list(range(21, 33)), "lb_type": [33, 34], "acc": [35, 36, 37, 38, 39],
           "acc_backpack": [35], "acc_bag": [36], "acc_glasses": [37, 38], "acc_hat": [39]}


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
        nm = np.load(mw.S / "extracted_features_v1/graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy")
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
    for c in cfgs:
        if c["name"] in done:
            continue
        w = base_w.copy()
        for g, (dpe, dhp, deva) in (c.get("dw") or {}).items():
            cols = GROUPS[g]
            w[0, cols] += dpe
            w[1, cols] += dhp
            w[2, cols] += deva
        if (w <= 0).any():
            print("skip non-positive weights", c["name"], flush=True)
            continue
        K = {**BASE_K, **(c.get("K") or {})}
        for key in ("ks", "anchor_w"):
            if key in K:
                K[key] = {**ke.PAR.DEFAULT_K[key], **K[key]}
        mw.POST = c.get("post") or POST0  # optional post-processing override (renorm groups / clip)
        p, mstd = mw.pool(raw, w)
        if c.get("pool"):  # alternative ensemble pooling: logit (geometric-odds) mix and/or temperature, then the same post-processing
            pc = c["pool"]
            st = np.stack([r.astype(np.float64) for r in raw])
            ww = w[:, None, :] / w.sum(axis=0)[None, None, :]
            pa = (ww * st).sum(axis=0)
            T = float(pc.get("T", 1.0))
            if pc["mode"] == "logit":
                sc = np.clip(st, 1e-6, 1 - 1e-6)
                zl = (ww * (np.log(sc) - np.log1p(-sc))).sum(axis=0)
                pm = (1.0 - float(pc["alpha"])) * pa + float(pc["alpha"]) / (1.0 + np.exp(-zl / T))
            else:  # 'temp': temperature on the arithmetic pool
                ac = np.clip(pa, 1e-6, 1 - 1e-6)
                pm = 1.0 / (1.0 + np.exp(-(np.log(ac) - np.log1p(-ac)) / T))
            p = mw.apply_post(pm.astype(np.float32), mw.POST)
        wv = sh["w_oof"].astype(np.float64).copy()
        for g, mult in (c.get("attr_wmul") or {}).items():  # per-group attribute distance weight multipliers
            wv[AGROUPS[g] if g in AGROUPS else [int(g[1:])]] *= mult  # 'a<idx>' = one attribute
        d = ke.PAR.run_b132c_pentj_body_rerank(p, sh["u_pe"], sh["u_eva"], sh["u_hp"], mstd, sh["queries_oof"], names,
                                               ke.make_score_fn(wv.astype(np.float32)), lambda *a, **k: None, K=K)
        adm = ke.eval_q(d, sh["qi_oof"], nm, gq)["adm"]
        np.savez(pq / f"{c['name']}.npz", oof=adm)
        row = {"name": c["name"], "oof": float(adm.mean()), "hi15": float(adm[cnt >= 15].mean()),
               "hi40": float(adm[cnt >= 40].mean()), "lo1_4": float(adm[cnt <= 4].mean())}
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
