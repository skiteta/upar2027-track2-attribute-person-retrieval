"""Adoption-bar check for final-round candidates vs B153C (bps).
Bar: positive on BOTH surfaces at hi15 (>=15 matches) and overall; no train source x multiplicity cell below -3 bps;
train paired query bootstrap 95% CI for overall or hi15 excluding zero. Val cells are shown as a second check.
usage: fr_gate.py VAL_BASE.npz TRAIN_BASE.npz VAL_CAND.npz TRAIN_CAND.npz [label]"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
KD = ROOT / "artifacts/runs/20261021T2250+0900_takeover_knob_search"
sys.path.insert(0, str(ROOT / "src"))
from upar.data import load_retrieval_split  # noqa: E402

BINS = ((1, 4), (5, 14), (15, 10**9))


def qsrc_of(qi, gq, src):
    out = []
    for q in qi:
        vals, c = np.unique(src[gq == q], return_counts=True)
        out.append(vals[np.argmax(c)])
    return np.asarray(out)


sh = np.load(KD / "train_surface/inputs/shared.npz")
t_qi, t_gq = sh["qi_oof"], sh["oof_g_qids"].astype(np.int64)
t_src = np.asarray([str(s) for s in sh["sources"]])
t_cnt = np.bincount(t_gq, minlength=int(max(t_gq.max(), t_qi.max())) + 1)[t_qi]
t_qs = qsrc_of(t_qi, t_gq, t_src)
sp = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
v_gq = sp.gallery_query_ids.astype(np.int64)
v_src = np.asarray([str(s) for s in sp.sources])
v_qi = np.unique(v_gq)
v_cnt = np.bincount(v_gq)[v_qi]
v_qs = qsrc_of(v_qi, v_gq, v_src)


def cells(d, qs, cnt):
    out = {}
    for s in np.unique(qs):
        for lo, hi in BINS:
            m = (qs == s) & (cnt >= lo) & (cnt <= hi)
            if m.sum():
                out[f"{s[:6]}[{lo}-{hi if hi < 10**9 else 'inf'}]"] = (float(d[m].mean()), int(m.sum()))
    return out


def boot(x, rng, n=5000):
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


def main():
    vb, tb, vc, tc = (np.load(p)["oof"] for p in sys.argv[1:5])
    label = sys.argv[5] if len(sys.argv) > 5 else Path(sys.argv[4]).stem
    dv, dt = (vc - vb) * 1e4, (tc - tb) * 1e4
    rng = np.random.default_rng(0)
    res = {"v_oof": dv.mean(), "v_hi15": dv[v_cnt >= 15].mean(), "t_oof": dt.mean(), "t_hi15": dt[t_cnt >= 15].mean(),
           "v_lo14": dv[v_cnt <= 4].mean(), "t_lo14": dt[t_cnt <= 4].mean()}
    tcell = cells(dt, t_qs, t_cnt)
    vcell = cells(dv, v_qs, v_cnt)
    ci_all = boot(dt, rng)
    ci_hi = boot(dt[t_cnt >= 15], rng)
    pos = all(res[k] > 0 for k in ("v_oof", "v_hi15", "t_oof", "t_hi15"))
    min_cell = min(v for v, _ in tcell.values())
    ci_ok = ci_all[0] > 0 or ci_hi[0] > 0
    print(f"== {label}")
    print("  " + " ".join(f"{k} {v:+.1f}" for k, v in res.items()))
    print("  train cells: " + " ".join(f"{k} {v:+.1f}(n{n})" for k, (v, n) in tcell.items()))
    print("  val cells:   " + " ".join(f"{k} {v:+.1f}(n{n})" for k, (v, n) in vcell.items()))
    print(f"  train boot all [{ci_all[0]:+.1f},{ci_all[1]:+.1f}] P>0 {ci_all[2]:.3f} | >=15 [{ci_hi[0]:+.1f},{ci_hi[1]:+.1f}] P>0 {ci_hi[2]:.3f}")
    print(f"  BAR: both-surface positive {pos} | min train cell {min_cell:+.1f} (>= -3: {min_cell >= -3}) | CI excl 0 {ci_ok}"
          f" => {'PASS' if pos and min_cell >= -3 and ci_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
