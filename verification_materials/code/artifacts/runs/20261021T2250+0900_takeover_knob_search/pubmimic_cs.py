"""Public-structure mimic (pubmimic.py contexts: Market+PA100k big part + 1,153-image PETA island, one graph, per-dataset
scoring, 3 island seeds) for combo_sweep-style configs: members pooled with the B145C-family member weights (+ optional
dw), attribute_distance_weights multiplied by attr_wmul, K overrides as in combo_sweep.py.
usage: pubmimic_cs.py CONFIGS.json OUT.jsonl"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
cfg_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [sys.argv[0], "train", "/dev/null", "/dev/null"]  # combo_sweep / mw_sweep read their argv at import
os.environ["KNOB_INPUTS"] = str(HERE / "train_surface/inputs")
sys.path.insert(0, str(HERE))
import mw_sweep as mw  # noqa: E402
import pubmimic as pm  # noqa: E402
import combo_sweep as cs  # noqa: E402  (GROUPS / AGROUPS / BASE_K only; its main() is not run)

ke = mw.ke


def main() -> None:
    cfgs = json.loads(cfg_path.read_text())
    pq = out_path.with_suffix("")
    pq.mkdir(parents=True, exist_ok=True)
    sh = {k: v for k, v in np.load(ke.INP / "shared.npz", allow_pickle=False).items()}
    names = [str(x) for x in sh["attribute_names"]]
    ctxs = pm.build_contexts(sh)
    allm = mw.members_all()
    raw = [allm["pe_b137c"], allm["hp_ltype2"], allm["eva_t12"]]
    base_w = mw.anat10()
    base_w[0, :] -= 0.05
    base_w[2, :] -= 0.05
    base_w[1, :] += 0.10
    done = {json.loads(l)["name"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    for c in cfgs:
        if c["name"] in done:
            continue
        w = base_w.copy()
        for g, (dpe, dhp, deva) in (c.get("dw") or {}).items():
            w[0, cs.GROUPS[g]] += dpe
            w[1, cs.GROUPS[g]] += dhp
            w[2, cs.GROUPS[g]] += deva
        K = {**cs.BASE_K, **(c.get("K") or {})}
        p, mstd = mw.pool(raw, w)
        wv = sh["w_oof"].astype(np.float64).copy()
        for g, mult in (c.get("attr_wmul") or {}).items():
            wv[cs.AGROUPS[g] if g in cs.AGROUPS else [int(g[1:])]] *= mult
        sf = ke.make_score_fn(wv.astype(np.float32))
        mev, syn, cnt = [], [], []
        for cx in ctxs:
            g = cx["gal"]
            q = np.concatenate([cx["q_big"], cx["q_isl"]])
            d = ke.PAR.run_b132c_pentj_body_rerank(p[g], sh["u_pe"][g], sh["u_eva"][g], sh["u_hp"][g], mstd[g], q, names, sf,
                                                   lambda *a, **k: None, K=K)
            nb, qb = cx["nb"], len(cx["q_big"])
            mev.append(ke.eval_q(np.ascontiguousarray(d[:qb, :nb]), np.arange(qb), cx["nm_big"], cx["g_big"])["adm"])
            syn.append(ke.eval_q(np.ascontiguousarray(d[qb:, nb:]), np.arange(len(cx["q_isl"])), cx["nm_isl"], cx["g_isl"])["adm"])
            cnt.append(cx["cnt_big"])
        mev, syn, cnt = np.concatenate(mev), np.concatenate(syn), np.concatenate(cnt)
        row = {"name": c["name"], "MEV": float(mev.mean()), "SYN": float(syn.mean()), "PUB": float(0.5 * mev.mean() + 0.5 * syn.mean()),
               "MEV_hi15": float(mev[cnt >= 15].mean()), "MEV_hi40": float(mev[cnt >= 40].mean())}
        np.savez(pq / f"{c['name']}.npz", mev=mev, syn=syn, cnt=cnt)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
