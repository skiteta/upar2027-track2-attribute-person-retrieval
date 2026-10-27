"""Public-structure mimic on the independent train-split surface (fold models never saw these images or sources).

Hidden Public = MEVID (real video, 367 q x 28,095 imgs, ~77 matches/query) + Synth (synthetic, 577 q x 1,153 imgs,
~2/query), reranked as ONE combined gallery and scored per dataset (queries vs own-dataset columns, degree-of-match
normalised over that dataset's gallery), Public = mean of the two.
Mimic: big part = Market1501 train (identity-structured: many images per identity) + 10k PA100k train (video frames);
small foreign-domain island = 1,153 PETA train images (150 profiles x1, 278 x2, 149 x3; PETA is a different domain from
Market/PA100k), all reranked together as one ~21k gallery; then 'MEV' = big-part queries on big-part columns and
'SYN' = island queries on island columns; 'PUB' = mean. 3 island seeds. Per-query arrays saved for paired comparisons.
usage: pubmimic.py CONFIGS.json OUT.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
os.environ.setdefault("KNOB_INPUTS", str(HERE / "train_surface/inputs"))
sys.path.insert(0, str(HERE))
import knob_eval as ke  # noqa: E402
from upar.metrics import _normalized_degree_of_match  # noqa: E402

SEEDS = (0, 1, 2)


def profile_ids(attrs: np.ndarray):
    uniq, inv = np.unique(attrs, axis=0, return_inverse=True)
    return uniq.astype(np.float32), inv.reshape(-1).astype(np.int64)


def build_contexts(sh):
    src = np.asarray([str(s) for s in sh["sources"]])
    attrs = sh["gallery_attributes"].astype(np.uint8)
    big = np.flatnonzero((src == "Market1501") | (src == "PA100k"))
    peta = np.flatnonzero(src == "PETA")
    ctxs = []
    for seed in SEEDS:
        rng = np.random.RandomState(20261022 + seed)
        _, pinv = profile_ids(attrs[peta])
        cnt = np.bincount(pinv)
        pick = []
        for need, n_prof in ((1, 150), (2, 278), (3, 149)):
            cand = np.flatnonzero(cnt >= need)
            cand = np.setdiff1d(cand, np.asarray([p for p, _ in pick], dtype=np.int64))
            for p in rng.choice(cand, size=n_prof, replace=False):
                pick.append((int(p), need))
        cols = []
        for p, need in pick:
            rows = peta[pinv == p]
            cols.extend(rng.choice(rows, size=need, replace=False).tolist())
        isl = np.sort(np.asarray(cols, dtype=np.int64))
        assert len(isl) == 1153
        gal = np.concatenate([big, isl])
        q_big, g_big = profile_ids(attrs[big])
        q_isl, g_isl = profile_ids(attrs[isl])
        ctxs.append({"gal": gal, "nb": len(big), "q_big": q_big, "g_big": g_big, "q_isl": q_isl, "g_isl": g_isl,
                     "nm_big": _normalized_degree_of_match(q_big.astype(np.uint8), attrs[big]).astype(np.float32),
                     "nm_isl": _normalized_degree_of_match(q_isl.astype(np.uint8), attrs[isl]).astype(np.float32),
                     "cnt_big": np.bincount(g_big, minlength=len(q_big))})
    return ctxs


def run(var, K, sh, names, ctxs):
    out = {"mev": [], "syn": [], "mev_hi15": [], "mev_hi40": []}
    sf = ke.make_score_fn(sh["w_oof"])
    for c in ctxs:
        g = c["gal"]
        q = np.concatenate([c["q_big"], c["q_isl"]])
        d = ke.PAR.run_b132c_pentj_body_rerank(var["oof_p"][g], var.get("u_pe", sh["u_pe"])[g], var.get("u_eva", sh["u_eva"])[g],
                                               var.get("u_hp", sh["u_hp"])[g], var["oof_mstd"][g], q, names, sf,
                                               lambda *a, **k: None, K=K)
        nb, qb = c["nb"], len(c["q_big"])
        rb = ke.eval_q(np.ascontiguousarray(d[:qb, :nb]), np.arange(qb), c["nm_big"], c["g_big"])["adm"]
        ri = ke.eval_q(np.ascontiguousarray(d[qb:, nb:]), np.arange(len(c["q_isl"])), c["nm_isl"], c["g_isl"])["adm"]
        out["mev"].append(rb)
        out["syn"].append(ri)
    return out


def main() -> None:
    cfgs = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    pq = out_path.with_suffix("")
    pq.mkdir(parents=True, exist_ok=True)
    sh = {k: v for k, v in np.load(ke.INP / "shared.npz", allow_pickle=False).items()}
    names = [str(x) for x in sh["attribute_names"]]
    ctxs = build_contexts(sh)
    done = {json.loads(l)["name"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    cache = {}
    for c in cfgs:
        if c["name"] in done:
            continue
        v = c.get("variant", "b136c")
        if v not in cache:
            z = np.load(ke.INP / f"{v}.npz", allow_pickle=False)
            cache[v] = {k: z[k] for k in z.files}
        K = dict(c.get("K") or {})
        for key in ("ks", "anchor_w"):
            if key in K:
                K[key] = {**ke.PAR.DEFAULT_K[key], **K[key]}
        r = run(cache[v], K, sh, names, ctxs)
        mev = np.concatenate(r["mev"])
        syn = np.concatenate(r["syn"])
        cnt = np.concatenate([cx["cnt_big"] for cx in ctxs])
        row = {"name": c["name"], "variant": v, "K": c.get("K"), "MEV": float(mev.mean()), "SYN": float(syn.mean()),
               "PUB": float(0.5 * mev.mean() + 0.5 * syn.mean()), "MEV_hi15": float(mev[cnt >= 15].mean()),
               "MEV_hi40": float(mev[cnt >= 40].mean()), "n_mev_q": int(len(r["mev"][0])), "n_hi40": int((ctxs[0]["cnt_big"] >= 40).sum())}
        np.savez(pq / f"{c['name']}.npz", mev=mev, syn=syn, cnt=cnt)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
