"""Compare knob_eval.eval_q (+ the cached / per-dataset contexts) with official_adm.py. CPU only.

eval_q and make_score_fn are taken verbatim from ../knob_eval.py via ast (only the device string "cuda" -> "cpu"
is changed), so the package / GPU are never touched.

usage: compare_adm.py OUT_PREFIX     (writes OUT_PREFIX_results.json)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("CMP_THREADS", "3")))
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import official_adm as off  # noqa: E402
from upar.metrics_strict import _normalized_degree_of_match as nm_strict, evaluate_scores  # noqa: E402
from upar.metrics import _normalized_degree_of_match as nm_v1  # noqa: E402

PKG = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/package_v14/package"
SRC = (RUN / "knob_eval.py").read_text()
tree = ast.parse(SRC)
seg = {n.name: ast.get_source_segment(SRC, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
ns = {"np": np, "torch": torch, "CFG": json.loads((PKG / "assets/config.json").read_text())}
exec(seg["eval_q"].replace('"cuda"', '"cpu"'), ns)
exec(seg["make_score_fn"], ns)
eval_q, make_score_fn = ns["eval_q"], ns["make_score_fn"]
PROV = {"knob_eval_sha256": hashlib.sha256(SRC.encode()).hexdigest(),
        "eval_q_source_sha256": hashlib.sha256(seg["eval_q"].encode()).hexdigest(),
        "device_patch": '"cuda" -> "cpu" in eval_q only'}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------- (a) synthetic cases
def make_case(rng, n_q, counts, n_distr, tie_mode, a=40):
    base = rng.integers(0, 2, a)
    prof = []
    seen = set()
    while len(prof) < n_q + max(n_distr, 1):
        p = base ^ (rng.random(a) < rng.uniform(0.03, 0.5)).astype(np.int64)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            prof.append(p)
    prof = np.asarray(prof, np.uint8)
    queries = prof[:n_q]
    g_attr, g_q = [], []
    for k, c in enumerate(counts):
        g_attr += [queries[k]] * c
        g_q += [k] * c
    for i in range(n_distr):
        g_attr.append(prof[n_q + rng.integers(0, max(n_distr, 1))])
        g_q.append(-1)  # distractor: matches no query
    g_attr = np.asarray(g_attr, np.uint8)
    g_q = np.asarray(g_q, np.int64)
    perm = rng.permutation(len(g_q))
    g_attr, g_q = g_attr[perm], g_q[perm]
    # duplicate query rows (same query id evaluated twice), as in repeated qi
    qi = np.concatenate([np.arange(n_q), rng.choice(n_q, size=max(1, n_q // 4), replace=True)])
    rng.shuffle(qi)
    G = len(g_q)
    ham = (queries[qi][:, None, :] != g_attr[None, :, :]).sum(-1).astype(np.float64)
    if tie_mode == "int4":
        d = rng.integers(0, 4, (len(qi), G)).astype(np.float32)
    elif tie_mode == "const":
        d = np.zeros((len(qi), G), np.float32)
    elif tie_mode == "hamming":  # many ties among equal Hamming distance; exact matches tied at 0
        d = ham.astype(np.float32)
    elif tie_mode == "noisy_hamming_snap":
        d = np.round(ham + rng.normal(0, 2.0, ham.shape)).astype(np.float32)
    elif tie_mode == "float":
        d = rng.random((len(qi), G)).astype(np.float32)
    elif tie_mode == "float_snap":
        d = rng.random((len(qi), G)).astype(np.float32)
        m = rng.random(d.shape) < 0.4
        d[m] = np.round(d[m] * 3) / 3
    else:
        raise ValueError(tie_mode)
    return queries, qi, g_attr, g_q, d


def run_synthetic():
    rng = np.random.default_rng(20261021)
    modes = ("int4", "const", "hamming", "noisy_hamming_snap", "float", "float_snap")
    worst = {"eval_q_vs_official": 0.0, "strict_vs_official": 0.0, "nm_strict_vs_official": 0.0, "nm_v1_vs_official": 0.0,
             "ap_eval_q_vs_official": 0.0, "r1_mismatch": 0}
    n_cases = n_rows = 0
    singles = manys = 0
    for case in range(240):
        n_q = int(rng.integers(1, 25))
        counts = [int(rng.choice([1, 1, 2, 3, 5, 12, 40])) for _ in range(n_q)]
        n_distr = int(rng.choice([0, 0, 5, 60, 300]))
        mode = modes[case % len(modes)]
        queries, qi, g_attr, g_q, d = make_case(rng, n_q, counts, n_distr, mode)
        nm = nm_strict(queries[qi], g_attr)
        ours = eval_q(d, qi, nm, g_q)
        ref = off.evaluate_rows(d, queries[qi], g_attr, "distance")
        worst["eval_q_vs_official"] = max(worst["eval_q_vs_official"], float(np.max(np.abs(ours["adm"] - ref["ADM"]))))
        worst["ap_eval_q_vs_official"] = max(worst["ap_eval_q_vs_official"], float(np.max(np.abs(ours["ap"] - ref["AP"]))))
        worst["r1_mismatch"] += int(np.sum(ours["r1"] != ref["R1"]))
        nm_ref = np.stack([off.normalized_degree_of_match(queries[q], g_attr) for q in qi])
        worst["nm_strict_vs_official"] = max(worst["nm_strict_vs_official"], float(np.max(np.abs(nm - nm_ref))))
        worst["nm_v1_vs_official"] = max(worst["nm_v1_vs_official"],
                                         float(np.max(np.abs(nm_v1(queries[qi], g_attr) - nm_ref))))
        if n_distr == 0:  # metrics_strict requires every gallery image to belong to a query
            uq = np.unique(qi)
            st = evaluate_scores(d[[int(np.flatnonzero(qi == u)[0]) for u in uq]], direction="distance",
                                 queries=queries, gallery_attributes=g_attr, gallery_query_ids=g_q, query_indices=uq)
            st_adm = np.asarray([r["ADM"] for r in st.per_query])
            ref_u = off.evaluate_rows(d[[int(np.flatnonzero(qi == u)[0]) for u in uq]], queries[uq], g_attr)
            worst["strict_vs_official"] = max(worst["strict_vs_official"], float(np.max(np.abs(st_adm - ref_u["ADM"]))))
        n_cases += 1
        n_rows += len(qi)
        singles += int(np.sum(ref["R"] == 1))
        manys += int(np.sum(ref["R"] >= 12))
    # similarity direction + ranking filter equivalence on a two-dataset gallery
    queries, qi, g_attr, g_q, d = make_case(rng, 30, [int(rng.choice([1, 2, 3, 8])) for _ in range(30)], 50, "int4")
    sim = -d
    a = off.evaluate_rows(d, queries[qi], g_attr, "distance")["ADM"]
    b = off.evaluate_rows(sim, queries[qi], g_attr, "similarity")["ADM"]
    worst["official_distance_vs_similarity"] = float(np.max(np.abs(a - b)))
    log("synthetic", n_cases, "cases", n_rows, "query rows;", singles, "rows with R=1,", manys, "rows with R>=12;", worst)
    return {"cases": n_cases, "query_rows": n_rows, "rows_R1": singles, "rows_R_ge_12": manys, "max_abs": worst}


def run_public_structure_synthetic():
    """Two 'datasets' in one gallery, overlapping query profiles, integer ties; official public_score vs eval_q per dataset."""
    rng = np.random.default_rng(7)
    worst = 0.0
    worst_ds_order = 0.0
    for case in range(30):
        queries, _, g_attr, g_q, d_full = make_case(rng, 40, [int(rng.choice([1, 2, 3, 9, 30])) for _ in range(40)], 0, "int4")
        G = len(g_q)
        ds = np.where(rng.random(G) < 0.3, "Synth", "MEVID")
        d = rng.integers(0, 5, (40, G)).astype(np.float32)  # 40 unique queries x whole two-dataset gallery
        pub = off.public_score(d, queries, g_attr, ds)
        for name in ("MEVID", "Synth"):
            cols = np.flatnonzero(ds == name)
            qi = np.unique(g_q[cols])
            sub = np.ascontiguousarray(d[qi][:, cols])
            nm = nm_strict(queries[qi], g_attr[cols])
            ours = eval_q(sub, qi, nm, g_q[cols])
            ref = pub["per_dataset"][name]
            assert np.array_equal(ref["query_rows"], qi), "dataset query sets differ"
            worst = max(worst, float(np.max(np.abs(ours["adm"] - ref["per_query_ADM"]))))
            direct = off.evaluate_rows(sub, queries[qi], g_attr[cols])["ADM"]
            worst_ds_order = max(worst_ds_order, float(np.max(np.abs(direct - ref["per_query_ADM"]))))
    log("public-structure synthetic: eval_q(own columns, own-gallery ndom) vs official public_score per query max abs",
        worst, "; full-ranking filter vs direct per-dataset sort", worst_ds_order)
    return {"cases": 30, "eval_q_vs_official_public_per_query": worst, "filter_vs_direct_sort": worst_ds_order}


def run_sensitivity_controls():
    """Negative controls: the synthetic tests must be able to detect a wrong tie order / missing clip / unnormalised dom."""
    rng = np.random.default_rng(99)
    diffs = {"reverse_tie_order": 0.0, "no_max0_clip": 0.0, "raw_dom_instead_of_ndom": 0.0}
    for case in range(60):
        queries, qi, g_attr, g_q, d = make_case(rng, 15, [int(rng.choice([1, 2, 5, 12])) for _ in range(15)], 40, "int4")
        nm = nm_strict(queries[qi], g_attr)
        ours = eval_q(d, qi, nm, g_q)["adm"]
        col = np.broadcast_to(np.arange(d.shape[1]), d.shape)
        rk_rev = np.lexsort((-col, d.astype(np.float64)), axis=-1)
        for j, q in enumerate(qi):
            rel = g_q == q
            dom = off.degree_of_match(queries[q], g_attr)
            mean, mx = dom.mean(), dom.max()
            variants = {"reverse_tie_order": (rk_rev[j], np.maximum(0, (dom - mean) / (mx - mean))),
                        "no_max0_clip": (off.rank_gallery(d[j])[0], (dom - mean) / (mx - mean)),
                        "raw_dom_instead_of_ndom": (off.rank_gallery(d[j])[0], dom)}
            for k, (rk, nd) in variants.items():
                r = np.flatnonzero(rel[rk]) + 1
                v = np.mean([nd[rk][:x].sum() / x for x in r])
                diffs[k] = max(diffs[k], abs(float(v) - float(ours[j])))
    log("sensitivity controls (should be >> 1e-12)", diffs)
    return diffs


# ---------------------------------------------------------------- (b) real val surfaces
def run_real():
    from upar.data import load_retrieval_split
    t0 = time.monotonic()
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    attrs = split.gallery_attributes.astype(np.uint8)
    src = split.sources
    g_qids = split.gallery_query_ids.astype(np.int64)
    sh = np.load(RUN / "inputs_v1/shared.npz", allow_pickle=False)
    names = [str(x) for x in sh["attribute_names"]]
    qi_oof = sh["qi_oof"].astype(np.int64)
    queries_oof = sh["queries_oof"]
    assert np.array_equal(queries_oof.astype(np.uint8), split.queries[qi_oof])
    p = np.load(RUN / "inputs_v1/b136c.npz", allow_pickle=False)["oof_p"]
    score = make_score_fn(sh["w_oof"])
    out = {}
    # relevance by id == relevance by 40-bit equality
    out["ids_equal_attribute_equality"] = bool(np.array_equal(split.queries[g_qids], attrs))
    out["queries_unique"] = bool(np.unique(split.queries, axis=0).shape[0] == split.queries.shape[0])

    # (b1) pooled OOF surface, first 300 queries, cached eval ctx (float32) as used by Surfaces
    nq = 300
    d = score(p, queries_oof[:nq], names)
    nm_cache = np.load(ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle/extracted_features_v1/"
                       "graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy", mmap_mode="r")
    nm_c = np.asarray(nm_cache[:nq])
    ours = eval_q(d, qi_oof[:nq], nm_c, g_qids)
    ref = off.evaluate_rows(d, split.queries[qi_oof[:nq]], attrs)
    nm_ref = np.stack([off.normalized_degree_of_match(split.queries[q], attrs) for q in qi_oof[:nq]])
    ours64 = eval_q(d, qi_oof[:nq], nm_strict(split.queries[qi_oof[:nq]], attrs), g_qids)
    ties = int(sum(int(len(r) - len(np.unique(r))) for r in d))
    out["pooled_oof_first300"] = {
        "max_abs_ADM_eval_q_cached_f32_ctx": float(np.max(np.abs(ours["adm"] - ref["ADM"]))),
        "max_abs_ADM_eval_q_f64_ctx": float(np.max(np.abs(ours64["adm"] - ref["ADM"]))),
        "max_abs_ndom_cached_vs_official": float(np.max(np.abs(nm_c.astype(np.float64) - nm_ref))),
        "max_abs_AP": float(np.max(np.abs(ours["ap"] - ref["AP"]))),
        "r1_mismatch": int(np.sum(ours["r1"] != ref["R1"])),
        "mean_ADM_eval_q": float(ours["adm"].mean()), "mean_ADM_official": float(ref["ADM"].mean()),
        "tied_distance_entries_in_rows": ties,
        "R_range": [int(ref["R"].min()), int(ref["R"].max())]}
    log("pooled OOF first300", out["pooled_oof_first300"])
    del d

    # (b2) per-dataset structure on the val gallery (Market1501 / PA100k / PETA as the 'datasets'): official public_score
    #      on the full 3,462 x 33,407 output vs eval_q with the cached fold contexts of shared.npz ('mixed' surface ctx)
    d_all = score(p, queries_oof, names)
    pub = off.public_score(d_all, split.queries[qi_oof], attrs, src)
    pub_pooled_stats = off.public_score(d_all, split.queries[qi_oof], attrs, src, stats="pooled")
    per = {}
    our_means = []
    for f in ("Market1501", "PA100k", "PETA"):
        hp = sh[f"{f}_hp"]
        fq = sh[f"{f}_qi"].astype(np.int64)
        rows = np.searchsorted(qi_oof, fq)
        assert np.array_equal(qi_oof[rows], fq)
        sub = np.ascontiguousarray(d_all[rows][:, hp])
        nm_f = sh[f"{f}_norm_match"]
        ours_f = eval_q(sub, fq, nm_f, sh[f"{f}_g_qids"].astype(np.int64))
        ref_f = pub["per_dataset"][f]
        same_q = bool(np.array_equal(qi_oof[ref_f["query_rows"]], fq))
        hp_is_source = bool(np.array_equal(np.sort(hp), np.flatnonzero(src == f)))
        nm_ref_f = np.stack([off.normalized_degree_of_match(split.queries[q], attrs[hp]) for q in fq])
        per[f] = {"queries": int(len(fq)), "gallery": int(len(hp)), "same_query_set_as_official": same_q,
                  "hp_equals_source_columns": hp_is_source,
                  "max_abs_ADM_eval_q_vs_official": float(np.max(np.abs(ours_f["adm"] - ref_f["per_query_ADM"]))),
                  "max_abs_ndom_cached_vs_official": float(np.max(np.abs(nm_f.astype(np.float64) - nm_ref_f))),
                  "mADM_eval_q": float(ours_f["adm"].mean()), "mADM_official": ref_f["mADM"],
                  "mADM_if_ndom_stats_over_pooled_gallery": pub_pooled_stats["per_dataset"][f]["mADM"]}
        our_means.append(float(ours_f["adm"].mean()))
    out["per_dataset_val"] = {
        "per_dataset": per,
        "official_summary": pub["summary"],
        "ours_mean_over_datasets": float(np.mean(our_means)),
        "abs_diff_mean_over_datasets": abs(float(np.mean(our_means)) - pub["summary"]["mADM"]),
        "official_summary_if_pooled_ndom_stats": pub_pooled_stats["summary"],
        "query_weighted_mean_instead": float(np.average([per[f]["mADM_official"] for f in per],
                                                        weights=[per[f]["queries"] for f in per])),
        "n_unique_queries": int(len(qi_oof)), "sum_dataset_queries": int(sum(per[f]["queries"] for f in per))}
    # pooled-gallery (OOF surface) value for the same output, for contrast with the per-dataset protocol
    ours_pooled_all = eval_q(d_all, qi_oof, np.asarray(nm_cache), g_qids)
    out["per_dataset_val"]["pooled_oof_surface_mADM_same_output"] = float(ours_pooled_all["adm"].mean())
    log("per-dataset val", json.dumps(out["per_dataset_val"], indent=1))
    out["seconds"] = time.monotonic() - t0
    return out


def public_aggregation_check():
    """Reconstruct the B137C Public numbers from its scoring_result.zip (scores.json + per-dataset table in HTML)."""
    import re
    import zipfile
    zp = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/b137c_watch_v3/scoring_result.zip"
    with zipfile.ZipFile(zp) as z:
        s = json.loads(z.read("scores.json"))
        html = z.read("detailed_results.html").decode("utf-8")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    ds = {}
    for name in ("MEVID", "Synth"):
        m = re.search(name + r" (\d+) (\d+) ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+)", text)
        q, g, adm, ap, r1, r5, r10, inp = m.groups()
        ds[name] = {"queries": int(q), "gallery": int(g), "mADM": float(adm), "mAP": float(ap), "Rank-1": float(r1),
                    "mINP": float(inp)}
    r = {"source": str(zp.relative_to(ROOT)), "scores_json": s, "per_dataset_html": ds,
         "html_says": [t for t in ("Every dataset is evaluated on its own", "Mean and standard deviation over the datasets of the gallery")
                       if t in text]}
    w = [ds["MEVID"]["queries"], ds["Synth"]["queries"]]
    for k in ("mADM", "mAP", "Rank-1", "mINP"):
        v = np.asarray([ds["MEVID"][k], ds["Synth"][k]])
        r[k] = {"reported": s[k], "reported_std": s[f"{k}_std"], "equal_weight_mean_of_rounded": float(v.mean()),
                "pop_std_of_rounded": float(v.std(ddof=0)), "sample_std_of_rounded": float(v.std(ddof=1)),
                "query_weighted_mean_of_rounded": float(np.average(v, weights=w))}
    m = re.search(r"queries (\d+) attributes", text)
    r["total_unique_queries_reported"] = int(m.group(1)) if m else None
    r["sum_of_dataset_queries"] = int(sum(w))
    log("public aggregation", json.dumps(r))
    return r


def main():
    prefix = Path(sys.argv[1])
    res = {"provenance": PROV, "synthetic": run_synthetic(), "public_structure_synthetic": run_public_structure_synthetic(),
           "public_aggregation_from_b137c_scores": public_aggregation_check(),
           "sensitivity_controls": run_sensitivity_controls()}
    res["real"] = run_real()
    res["log"] = LOG
    Path(f"{prefix}_results.json").write_text(json.dumps(res, indent=1, default=float))
    print("WROTE", f"{prefix}_results.json")


if __name__ == "__main__":
    main()
