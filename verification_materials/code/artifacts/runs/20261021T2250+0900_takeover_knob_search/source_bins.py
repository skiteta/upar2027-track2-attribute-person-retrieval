"""Train-surface deltas (bps) of candidate npz files vs a base npz, by majority source of the query's matches x multiplicity.
usage: source_bins.py BASE_NPZ CAND_NPZ [...]"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sh = np.load(HERE / "train_surface/inputs/shared.npz")
qi, gq = sh["qi_oof"], sh["oof_g_qids"].astype(np.int64)
src = np.asarray([str(s) for s in sh["sources"]])
cnt = np.bincount(gq, minlength=int(max(gq.max(), qi.max())) + 1)[qi]
qsrc = []
for q in qi:
    vals, c = np.unique(src[gq == q], return_counts=True)
    qsrc.append(vals[np.argmax(c)])
qsrc = np.asarray(qsrc)
base = np.load(sys.argv[1])["oof"]
BINS = ((1, 4), (5, 14), (15, 10**9))
print("bins per source: " + " ".join(f"[{a}-{b if b < 10**9 else 'inf'}]" for a, b in BINS))
for p in sys.argv[2:]:
    d = (np.load(p)["oof"] - base) * 1e4
    parts = []
    for s in ("Market1501", "PA100k", "PETA"):
        cells = []
        for lo, hi in BINS:
            m = (qsrc == s) & (cnt >= lo) & (cnt <= hi)
            cells.append(f"{d[m].mean():+5.1f}")
        parts.append(f"{s[:6]} " + "/".join(cells))
    lo4 = d[cnt <= 4].mean()
    print(f"{Path(p).stem:30s} " + " | ".join(parts) + f" | all<=4 {lo4:+.1f} >=15 {d[cnt >= 15].mean():+.1f}")
