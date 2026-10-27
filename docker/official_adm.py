"""UPAR Track 2 (attribute-based person retrieval) mADM, written from the organisers' published definition.

No official scoring CODE is public (checked 2026-10-21):
  * github.com/speckean/UPAR-Challenge-2027 @ a19ab2fb6470140606d3c1982c303937b58fbd14 (starter kit): data, download
    script and sample submissions only. The sample run.py says "Smaller = better match. Ties keep the gallery order."
  * github.com/speckean/upar_challenge @ d79c1916a12b362433ef880900af525ecce479c1 (2023/2024 kit, full history of 15
    commits): README, download script, PETA mapping only. No evaluation script ever committed.
  * github.com/speckean/upar_dataset @ ef8e5a3a64ab770f400c71a0cd50b85069c3d5d3: dataset loader only.
  * Codabench 18196 task dataset "scoring_program @ '09-21-2026 10:16'" (10,275 B) has is_public=False.
  * The cited metric paper (Specker & Beyerer, "Balanced Pedestrian Attribute Recognition for Improved Attribute-based
    Person Retrieval", ICPRS 2023, doi:10.1109/icprs58416.2023.10178990) is closed access; the UPAR-2024 report
    (Cormier et al., WACVW 2024, openaccess.thecvf.com) cites it for mADM without a formula.

The definition implemented here is therefore the one on Codabench competition 18196, page "Evaluation" (public, read
unauthenticated via https://www.codabench.org/api/competitions/18196/ ; page text sha256
2c40ebaf562a7cb6b6c0a7cddd4471253ffb9d7d795192c453bb6bb512c733e1, identical in the project's 2026-10-12 and
2026-10-17 snapshots), which says "The scoring program reproduces the original evaluation script", plus the per-dataset
aggregation printed by the scoring program in every submission's detailed_results.html / scores.json:

    dom(g)  = (number of attributes where image g equals the query) / 40
    ndom(g) = max( 0, (dom(g) - mean dom) / (max dom - mean dom) )      mean, max over the gallery
    ADM     = (1/R) * sum_k ( ndom(g_1) + ... + ndom(g_{r_k}) ) / r_k   r_1<...<r_R: ranks of the exact matches
    mADM    = mean over the queries
    Ties: equal distances / similarities are ranked in gallery order.
    "Every dataset is evaluated on its own: its queries are ranked against its gallery images only."
    "Mean and standard deviation over the datasets of the gallery."  (scores.json: mADM, mADM_std)

Deliberately independent of src/upar/metrics*.py and knob_eval.eval_q: dom by explicit equality counting, ranking by
lexsort on (value, gallery index), relevance by full 40-bit equality, prefix sums recomputed per rank.
"""
from __future__ import annotations

import numpy as np

N_ATTR = 40


def rank_gallery(values: np.ndarray, direction: str = "distance") -> np.ndarray:
    """Return the ranking (best first) of every row; equal values keep gallery order (key = (value, column index))."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    if direction == "similarity":
        v = -v
    elif direction != "distance":
        raise ValueError(direction)
    if not np.isfinite(v).all():
        raise ValueError("non-finite values")
    col = np.broadcast_to(np.arange(v.shape[1], dtype=np.int64), v.shape)
    return np.lexsort((col, v), axis=-1)


def degree_of_match(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """dom(g): share of the attributes on which image g equals the query."""
    return (np.asarray(gallery) == np.asarray(query)[None, :]).sum(axis=1) / gallery.shape[1]


def normalized_degree_of_match(query: np.ndarray, gallery: np.ndarray, stats_gallery: np.ndarray | None = None) -> np.ndarray:
    """ndom over `gallery`; mean/max are taken over `stats_gallery` (default: the same gallery, as published)."""
    dom = degree_of_match(query, gallery)
    ref = dom if stats_gallery is None else degree_of_match(query, stats_gallery)
    mean, mx = float(ref.mean()), float(ref.max())
    if mx - mean <= 0.0:
        # 0/0 in the published formula; only possible when every gallery image is an exact match (then ADM = 1).
        return np.ones_like(dom, dtype=np.float64)
    return np.maximum(0.0, (dom - mean) / (mx - mean))


def query_metrics_from_ranking(ranking: np.ndarray, query: np.ndarray, gallery: np.ndarray,
                               stats_gallery: np.ndarray | None = None) -> dict[str, float]:
    """ADM / AP / Rank-1 / INP for one query given its full ranking of `gallery` (indices into gallery)."""
    ranking = np.asarray(ranking, dtype=np.int64)
    if ranking.shape != (gallery.shape[0],) or not np.array_equal(np.sort(ranking), np.arange(gallery.shape[0])):
        raise ValueError("ranking is not a permutation of the gallery")
    relevant = np.all(gallery == query[None, :], axis=1)  # correct match: all 40 attributes equal
    ndom = normalized_degree_of_match(query, gallery, stats_gallery)
    nd_ranked = ndom[ranking]
    ranks = np.flatnonzero(relevant[ranking]) + 1  # 1-based r_1 < ... < r_R
    if ranks.size == 0:
        raise ValueError("query has no correct match in this gallery")
    adm_terms = [float(nd_ranked[:r].sum()) / int(r) for r in ranks]  # prefix recomputed at every r_k
    ap_terms = [(k + 1) / int(r) for k, r in enumerate(ranks)]
    return {"ADM": float(np.mean(adm_terms)), "AP": float(np.mean(ap_terms)), "R1": float(ranks[0] <= 1),
            "INP": float(ranks.size / ranks[-1]), "R": int(ranks.size)}


def evaluate_rows(values: np.ndarray, queries: np.ndarray, gallery: np.ndarray, direction: str = "distance",
                  stats_gallery: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-query metrics; row i of `values` scores queries[i] against every gallery column."""
    queries = np.asarray(queries)
    gallery = np.asarray(gallery)
    if values.shape != (queries.shape[0], gallery.shape[0]):
        raise ValueError("values must be queries x gallery")
    out = {k: [] for k in ("ADM", "AP", "R1", "INP", "R")}
    for start in range(0, queries.shape[0], 128):
        rk = rank_gallery(values[start:start + 128], direction)
        for j in range(rk.shape[0]):
            m = query_metrics_from_ranking(rk[j], queries[start + j], gallery, stats_gallery)
            for k in out:
                out[k].append(m[k])
    return {k: np.asarray(v) for k, v in out.items()}


def public_score(values: np.ndarray, queries: np.ndarray, gallery: np.ndarray, dataset_of_image: np.ndarray,
                 direction: str = "distance", stats: str = "dataset") -> dict:
    """Leaderboard-style score of one Q x G output over a multi-dataset gallery.

    The full ranking is built once over the whole gallery (the scoring program loads rankings.npy, Q x G), then for each
    dataset the ranking is filtered to that dataset's images (order preserved); the dataset's queries are those with at
    least one exact match among its images. mADM = mean over datasets of the per-dataset query mean; *_std = population
    std over datasets. stats='dataset' normalises dom over the dataset's gallery (published reading); stats='pooled'
    uses the whole combined gallery (sensitivity check only).
    """
    queries = np.asarray(queries)
    gallery = np.asarray(gallery)
    ds = np.asarray(dataset_of_image)
    names = list(dict.fromkeys(ds.tolist()))
    per: dict[str, dict] = {}
    rows_by_ds: dict[str, list[dict]] = {n: [] for n in names}
    cols = {n: np.flatnonzero(ds == n) for n in names}
    local = {n: np.full(gallery.shape[0], -1, np.int64) for n in names}
    for n in names:
        local[n][cols[n]] = np.arange(cols[n].size)
    for start in range(0, queries.shape[0], 128):
        rk_full = rank_gallery(values[start:start + 128], direction)
        for j in range(rk_full.shape[0]):
            q = queries[start + j]
            for n in names:
                g = gallery[cols[n]]
                if not np.all(g == q[None, :], axis=1).any():
                    continue
                filt = local[n][rk_full[j]]
                filt = filt[filt >= 0]
                m = query_metrics_from_ranking(filt, q, g, None if stats == "dataset" else gallery)
                m["query_row"] = start + j
                rows_by_ds[n].append(m)
    for n in names:
        rows = rows_by_ds[n]
        per[n] = {"queries": len(rows), "gallery": int(cols[n].size),
                  **{f"m{k}" if k in ("ADM", "AP", "INP") else k: float(np.mean([r[k] for r in rows]))
                     for k in ("ADM", "AP", "R1", "INP")},
                  "per_query_ADM": np.asarray([r["ADM"] for r in rows]),
                  "query_rows": np.asarray([r["query_row"] for r in rows], np.int64)}
    summary = {}
    for key in ("mADM", "mAP", "R1", "mINP"):
        vals = np.asarray([per[n][key] for n in names])
        summary[key] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std(ddof=0))
    return {"summary": summary, "per_dataset": per}
