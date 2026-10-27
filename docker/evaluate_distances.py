"""Score a Q x G distance matrix on a public UPAR Track 2 split (the organiser starter-kit format).

Metrics follow the published Codabench definitions (mADM, mAP, Rank-1/5/10, mINP); the per-query computation is
docker/official_adm.py, an independent re-implementation that matched the official scorer to 2.4e-8 on our own
submissions. Ties are ranked in gallery order. The gallery is the whole split (one ranking per query); per-source
means are reported for information.

usage: evaluate_distances.py --split-dir DIR --distances distances.npy --out metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import official_adm as oa
from upar.data import load_retrieval_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--distances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    split = load_retrieval_split(a.split_dir)
    d = np.load(a.distances, allow_pickle=False)
    queries = np.asarray(split.queries, dtype=np.uint8)
    gallery = np.asarray(split.gallery_attributes, dtype=np.uint8)
    if d.shape != (queries.shape[0], gallery.shape[0]):
        raise ValueError(f"distances {d.shape} != queries x gallery {(queries.shape[0], gallery.shape[0])}")
    source = np.asarray([str(x) for x in split.sources])

    rows = {k: [] for k in ("ADM", "AP", "R1", "R5", "R10", "INP")}
    q_source = []
    for start in range(0, queries.shape[0], 128):
        ranking = oa.rank_gallery(d[start:start + 128], "distance")
        for j in range(ranking.shape[0]):
            q = queries[start + j]
            m = oa.query_metrics_from_ranking(ranking[j], q, gallery)
            relevant = np.all(gallery[ranking[j]] == q[None, :], axis=1)
            first = int(np.flatnonzero(relevant)[0]) + 1
            for k in ("ADM", "AP", "R1", "INP"):
                rows[k].append(m[k])
            rows["R5"].append(float(first <= 5))
            rows["R10"].append(float(first <= 10))
            vals, counts = np.unique(source[np.all(gallery == q[None, :], axis=1)], return_counts=True)
            q_source.append(str(vals[np.argmax(counts)]))
    q_source = np.asarray(q_source)
    arr = {k: np.asarray(v) for k, v in rows.items()}
    out = {
        "split_dir": str(a.split_dir), "distances": str(a.distances), "queries": int(queries.shape[0]), "gallery": int(gallery.shape[0]),
        "pooled_gallery": {("m" + k if k in ("ADM", "AP", "INP") else k.replace("R", "Rank-")): float(v.mean()) for k, v in arr.items()},
        "per_majority_source_of_matches": {s: {"queries": int((q_source == s).sum()), "mADM": float(arr["ADM"][q_source == s].mean())}
                                           for s in np.unique(q_source)},
        "note": "Evaluated with the whole split as one gallery. Final B156C models were trained on train+validation, so "
                "validation scores are in-sample execution checks, not accuracy estimates.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["pooled_gallery"]))


if __name__ == "__main__":
    main()
