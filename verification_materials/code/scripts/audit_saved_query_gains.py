#!/usr/bin/env python3
"""Audit fixed saved mADM vectors without fitting, inference, tuning or CUDA.

Query bootstrap intervals are conditional on this already reused gallery and
selection history. They are NOT probabilities of Public/Private improvement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from upar.data import load_retrieval_split


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def vector(value, count):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (count,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("mADM vector shape, finiteness or [0,1] range invalid")
    return value


def source_membership(qi, gallery_qids, sources):
    qi, gallery_qids, sources = np.asarray(qi), np.asarray(gallery_qids), np.asarray(sources)
    if len(np.unique(qi)) != len(qi) or np.any(qi < 0) or np.any(gallery_qids < 0):
        raise ValueError("nonnegative unique query IDs required")
    if gallery_qids.shape != sources.shape:
        raise ValueError("gallery identity/source mismatch")
    names = np.unique(sources)
    size = int(max(qi.max(), gallery_qids.max())) + 1
    counts = np.stack([np.bincount(gallery_qids[sources == s], minlength=size)[qi] for s in names], axis=1)
    total = counts.sum(axis=1)
    if np.any(total == 0):
        raise ValueError("query has no gallery match")
    return names, counts, total, names[counts.argmax(axis=1)]


def summary(baseline, candidate, *, seed, repetitions):
    baseline, candidate = np.asarray(baseline), np.asarray(candidate)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("nonempty paired vectors required")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for off in range(0, repetitions, 128):
        count = min(128, repetitions - off)
        idx = rng.integers(0, len(delta), size=(count, len(delta)))
        boot[off:off + count] = delta[idx].mean(axis=1)
    remove = max(1, int(np.ceil(len(delta) * .1)))
    keep = np.sort(delta)[:-remove]
    return {"n_queries": len(delta), "baseline_mADM": float(baseline.mean()), "candidate_mADM": float(candidate.mean()),
            "delta_mADM": float(delta.mean()), "delta_bps": float(delta.mean() * 1e4),
            "conditional_query_ci95_mADM": np.quantile(boot, [.025, .975]).tolist(),
            "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())],
            "delta_quantiles": dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                                        np.quantile(delta, [0, .05, .25, .5, .75, .95, 1]).tolist())),
            "without_largest_10pct_gains_delta": float(keep.mean()) if len(keep) else None,
            "interval_scope": "Conditional query resampling only; selection, shared-gallery dependence and domain shift are not covered."}


def audit_surface(root, spec, repetitions, seed):
    refs = []
    shared_path = root / spec["shared"]
    refs.append(pin(shared_path))
    with np.load(shared_path, allow_pickle=False) as sh:
        qi = sh["qi_oof"].astype(np.int64)
        queries = sh["queries_oof"]
        names = sh["attribute_names"].astype(str).tolist()
        if "oof_g_qids" in sh:
            gq, sources = sh["oof_g_qids"], sh["sources"].astype(str)
        else:
            ann = root / spec["annotations"]
            split = load_retrieval_split(ann)
            if not np.array_equal(queries, split.queries[qi]) or names != list(split.attribute_names):
                raise ValueError("annotation/shared query identities differ")
            gq, sources = split.gallery_query_ids, split.sources
            refs.extend(pin(ann / n) for n in ("gt.csv", "ids.csv", "queries.csv"))
    source_names, source_counts, counts, majority = source_membership(qi, gq, sources)
    if queries.shape != (len(qi), 40) or np.any((queries != 0) & (queries != 1)):
        raise ValueError("40-bit binary query matrix required")
    values, alignment = {}, {}
    for name, row in spec["candidates"].items():
        p = root / row["path"]
        refs.append(pin(p))
        with np.load(p, allow_pickle=False) as z:
            values[name] = vector(z[row["key"]], len(qi))
            if "qi" in z:
                if not np.array_equal(z["qi"], qi) or not np.array_equal(z["n_match"], counts):
                    raise ValueError("embedded query identities/counts differ")
                alignment[name] = "embedded qi and n_match checked"
            else:
                # Legacy sweep outputs omitted IDs. Cross-check independent saved
                # copies and producer summary, but do not invent embedded identity.
                if not row.get("summary") or not row.get("crosscheck"):
                    raise ValueError("legacy vector needs producer summary and repeat-output evidence")
                sp = root / row["summary"]
                cp = root / row["crosscheck"]
                refs.extend((pin(sp), pin(cp)))
                records = [json.loads(line) for line in sp.read_text().splitlines() if line.strip()]
                match = [r for r in records if r["name"] == row["record_name"]]
                if len(match) != 1 or not np.isclose(match[0]["oof"], values[name].mean(), rtol=0, atol=1e-10):
                    raise ValueError("saved vector and producing aggregate differ")
                with np.load(cp, allow_pickle=False) as z2:
                    if not np.array_equal(values[name], z2[row["key"]]):
                        raise ValueError("repeat reference vector differs")
                alignment[name] = "producer order contract + aggregate + repeated vector checked; original vector has no embedded IDs"
    masks = {"all": np.ones(len(qi), bool), "matches_1_4": counts <= 4,
             "matches_5_14": (counts >= 5) & (counts <= 14), "matches_ge15": counts >= 15}
    comparisons = {}
    for baseline in ("B136C", "B137C", "B152C"):
        delta = values["B153C"] - values[baseline]
        result = {label: summary(values[baseline][mask], values["B153C"][mask], seed=seed, repetitions=repetitions)
                  for label, mask in masks.items() if mask.any()}
        cells = {}
        for source in source_names:
            cells[source] = {}
            for label, mask in masks.items():
                selected = mask & (majority == source)
                if selected.any():
                    cells[source][label] = {"n_queries": int(selected.sum()), "delta_bps": float(delta[selected].mean() * 1e4)}
        result["majority_source_descriptive_slices"] = cells
        source_deltas = {s: float(delta[majority == s].mean()) for s in source_names if np.any(majority == s)}
        result["source_equal_weight_delta"] = float(np.mean(list(source_deltas.values())))
        result["leave_one_majority_source_out_delta"] = {s: float(delta[majority != s].mean()) for s in source_names}
        result["source_count"] = len(source_names)
        result["source_inference_warning"] = "Only three observed source domains. Majority assignment is descriptive; mixed-source queries and the shared gallery prevent treating these as independent new domains."
        order = np.argsort(delta, kind="stable")
        def example(i):
            return {"query_index": int(qi[i]), "matching_images": int(counts[i]), "majority_source": str(majority[i]),
                    "baseline_ADM": float(values[baseline][i]), "candidate_ADM": float(values["B153C"][i]),
                    "delta_ADM": float(delta[i]), "positive_attributes": [n for n, bit in zip(names, queries[i]) if bit]}
        result["worst_five"] = [example(i) for i in order[:5]]
        result["best_five"] = [example(i) for i in order[::-1][:5]]
        result["attribute_slices_exploratory"] = [
            {"attribute": name, "value": value, "n_queries": int((queries[:, j] == value).sum()),
             "delta_bps": float(delta[queries[:, j] == value].mean() * 1e4)}
            for j, name in enumerate(names) for value in (0, 1) if np.any(queries[:, j] == value)]
        comparisons["B153C_minus_" + baseline] = result
    return {"gallery_rows": len(gq), "query_rows": len(qi), "multiple_source_queries": int(((source_counts > 0).sum(axis=1) > 1).sum()),
            "alignment": alignment, "comparisons": comparisons, "evidence": refs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text())
    report = {"status": "COMPLETE_SAVED_DEVELOPMENT_AUDIT", "at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "spec": pin(args.spec), "script": pin(Path(__file__)), "new_fits": 0, "new_inference": 0,
              "gpu_seconds": 0, "evaluation_role": "development", "selection_adjusted": False,
              "public_probability": None, "private_probability": None,
              "limitations": ["Both surfaces were reused for selection among about 190 configurations.",
                              "Saved fold-model results are a proxy, not the final all-data model or hidden Synth evaluation.",
                              "Intervals are conditional query sensitivity, not calibrated generalization coverage.",
                              "Attribute slices are post-hoc diagnostics, not independent hypotheses or tuning instructions."],
              "surfaces": {name: audit_surface(args.workspace.resolve(), row, spec["repetitions"], spec["seed"])
                           for name, row in spec["surfaces"].items()}}
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({s: {k: {"all_bps": v["all"]["delta_bps"], "ge15_bps": v["matches_ge15"]["delta_bps"],
                                 "ge15_conditional_ci": v["matches_ge15"]["conditional_query_ci95_mADM"]}
                              for k, v in row["comparisons"].items()} for s, row in report["surfaces"].items()}))


if __name__ == "__main__":
    main()
