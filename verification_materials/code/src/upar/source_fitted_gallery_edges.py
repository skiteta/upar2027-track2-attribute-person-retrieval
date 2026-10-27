"""Prepare/extract/fit the fixed source-validation one-bit edge rescue.

No implicit GPU execution, target-label fitting, submission or deployment. Each
outer fold uses its own source-held PE4 checkpoint for BOTH meta-fit sources.
Never substitute another fold's saved features. All outputs are exclusive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .fixed_gallery_rerank import build_mutual_graph
from .learned_gallery_edges import FEATURE_NAMES, RIDGE, fit_pair_model, one_bit_pairs, pair_features, unit_features
from .postproc import apply_post
from .rank_sibling_pipeline import ROOT, exclusive_json, fingerprint, read_json

RUN = ROOT / "artifacts/runs/20261017T1105+0900_cycle_continuation/frontier/sourcefit_edge_v2"
PE = ROOT / "artifacts/runs/20261017T0448+0900_pe_gallery_features"
SPEC = ROOT / "artifacts/runs/20261016T1710+0900_b119g/candidate_b119g.json"
AUDIT = ROOT / "artifacts/runs/20261017T0446+0900_gallery_reciprocal_rerank/comparison_v1/report_v1.json"
RULES = ROOT / "artifacts/runs/20261017T0310+0900_candidate_recovery/official_current_rules_20261017_v1.json"
FOLDS = ("Market1501", "PA100k", "PETA")
META_SHA = "e3d5d942049f0c69ac055d9d1539b79397d6a7fa695a24bbc6f32266256d477b"


def verify(pins):
    for expected in pins:
        if fingerprint(ROOT / expected["path"]) != expected:
            raise ValueError(f"frozen input changed: {expected['path']}")


def source_val_rows(meta, fold):
    if fold not in FOLDS:
        raise ValueError("unknown outer held source")
    sources = np.asarray(meta["sources"])
    split = np.asarray(meta["split"])
    if sources.shape != split.shape or sources.ndim != 1 or set(sources) != set(FOLDS):
        raise ValueError("unaligned or unexpected cache sources")
    selected = np.flatnonzero((split == "val") & (sources != fold))
    if set(sources[selected]) != set(FOLDS) - {fold} or not len(selected):
        raise ValueError("both non-target validation sources required")
    val = np.flatnonzero(split == "val")
    positions = np.flatnonzero(sources[val] != fold)
    if not np.array_equal(selected, val[positions]):
        raise ValueError("source validation order mismatch")
    return selected, positions


def _image_stat(path):
    s = path.stat()
    return {"path": str(path.relative_to(ROOT)), "size": s.st_size, "mtime_ns": s.st_mtime_ns,
            "device": s.st_dev, "inode": s.st_ino, "content_hash_rechecked": False}


def _validate_audit(leaves, fold):
    if not leaves:
        raise ValueError("empty upstream fit audit")
    for leaf in leaves:
        if leaf.get("fold") != fold or leaf.get("include_val") is not False or not leaf.get("held_out_absent_from_fit"):
            raise ValueError("encoder validation fit or held-source exposure")
        if fold in leaf["fit_rows_by_source"] or set(leaf["fit_rows_by_source"]) != set(FOLDS) - {fold}:
            raise ValueError("wrong upstream fit sources")


def prediction_lineage(run, fold, *, root=ROOT, ancestors=()):
    """Resolve actual selected producer config graph, not an unbound leaf list.

    Soup arithmetic has precedence over copied xres metadata. Prior-alignment
    references are audited but do not become additional encoder soup members.
    Unknown teacher/initializer fit dependencies fail closed for this recipe.
    """
    path = (root / run).resolve()
    if path in ancestors or not path.is_relative_to(root.resolve()):
        raise ValueError("cyclic/outside-root prediction lineage")
    config_path = path / "config.json"
    cfg = read_json(config_path)
    if cfg.get("fold") != fold or cfg.get("include_val") is not False:
        raise ValueError("probability producer fit boundary differs")
    if cfg.get("init_from") or cfg.get("online_teachers") or any(value for key, value in cfg.items() if "teacher" in key):
        raise ValueError("unresolved teacher/initializer dependency in probability producer")
    next_ancestors = (*ancestors, path)
    paths = [config_path]
    leaves = []
    children = cfg.get("soup_of")
    if children:
        weights = cfg.get("soup_weights")
        if not isinstance(children, list) or not weights or len(weights) != len(children) or any(w <= 0 for w in weights):
            raise ValueError("incomplete probability soup manifest")
    else:
        children = [cfg[k] for k in ("xres_of", "pw_aligned_of") if cfg.get(k)]
        if len(children) > 1:
            raise ValueError("ambiguous derived probability lineage")
    if children:
        for child in children:
            child_leaves, child_paths = prediction_lineage(child, fold, root=root, ancestors=next_ancestors)
            leaves += child_leaves
            paths += child_paths
    else:
        report_path = path / "train_report.json"
        if not report_path.exists() and cfg.get("ema_of"):
            report_path = root / cfg["ema_of"] / "train_report.json"
        report = read_json(report_path)
        if report.get("status") != "TRAIN_DONE" or report.get("fold") != fold or report.get("include_val") is not False:
            raise ValueError("probability leaf training not complete/source-clean")
        paths.append(report_path)
        leaves = [{"run": str(path.relative_to(root.resolve())), "train_report": str(report_path.resolve().relative_to(root.resolve())),
                   "fold": cfg["fold"], "include_val": cfg["include_val"], "fit_rows_by_source": report["fit_rows_by_source"],
                   "held_out_absent_from_fit": fold not in report["fit_rows_by_source"]}]
        _validate_audit(leaves, fold)
    if cfg.get("pw_ref"):
        _, prior_paths = prediction_lineage(cfg["pw_ref"], fold, root=root, ancestors=next_ancestors)
        paths += prior_paths
    return leaves, list(dict.fromkeys(paths))


def match_selected_lineage(spec, fold, audit, *, root=ROOT):
    all_leaves, all_paths, producers = [], [], []
    for member in spec["lodo"]["members"]:
        for run in member["runs"][fold]:
            leaves, paths = prediction_lineage(run, fold, root=root)
            all_leaves += leaves
            all_paths += paths
            producers.append({"selected_run": run, "member_weight": member["weight"],
                              "resolved_ordered_leaf_runs": [x["run"] for x in leaves]})
    if Counter(x["run"] for x in all_leaves) != Counter(x["run"] for x in audit):
        raise ValueError("historical audit does not cover exact selected probability producers")
    audited = {x["run"]: x for x in audit}
    for leaf in all_leaves:
        if any(audited[leaf["run"]].get(key) != value for key, value in leaf.items()):
            raise ValueError("actual probability producer differs from historical audit")
    return producers, list(dict.fromkeys(all_paths))


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    if (run / "protocol.json").exists():
        raise FileExistsError("protocol exists; resume exact protocol, do not overwrite")
    spec, old, audit = read_json(SPEC), read_json(PE / "protocol.json"), read_json(AUDIT)
    rows, cache_paths = [], set()
    for fold in FOLDS:
        source = next(r for r in old["rows"] if r["fold"] == fold)
        model_dir = PE / f"pe4_{fold}"
        ready, cfg = read_json(model_dir / "weights_ready.json"), read_json(model_dir / "config.json")
        model_pin, config_pin = fingerprint(model_dir / "model.safetensors"), fingerprint(model_dir / "config.json")
        if model_pin["sha256"] != ready["model"]["sha256"] or config_pin["sha256"] != ready["config_sha256"]:
            raise ValueError("existing PE4 checkpoint/config differs from its receipt")
        if cfg["fold"] != fold or cfg["include_val"] is not False or cfg["head"] != "attrq" or cfg["backbone"] != "pe_core_l14_280":
            raise ValueError("not the exact source-held PE4 attrq checkpoint")
        recipe = {k: v for k, v in cfg.items() if k != "output_dir"}
        old_recipe = {k: v for k, v in source["config"].items() if k != "output_dir"}
        if cfg.get("soup_weights") != [.25] * 4 or old_recipe != recipe or Path(cfg["output_dir"]).resolve() != model_dir.resolve():
            raise ValueError("source-held PE4 reconstruction arithmetic changed")
        leaves = audit["folds"][fold]["upstream_fit"]
        _validate_audit(leaves, fold)
        linked_producers, lineage_paths = match_selected_lineage(spec, fold, leaves)
        cache = ROOT / cfg["cache_dir"]
        cache_paths.add(cache)
        meta_pin = fingerprint(cache / "meta.npz")
        if meta_pin["sha256"] != META_SHA:
            raise ValueError("canonical cache metadata changed")
        with np.load(cache / "meta.npz", allow_pickle=False) as meta:
            selected, positions = source_val_rows(meta, fold)
            counts = {s: int(np.sum(meta["sources"][selected] == s)) for s in FOLDS if s != fold}
            expected_train = {s: int(np.sum((meta["sources"] == s) & (meta["split"] == "train"))) for s in FOLDS if s != fold}
            if any(leaf["fit_rows_by_source"] != expected_train for leaf in leaves):
                raise ValueError("producer fit rows differ from canonical source training partition")
            prediction_pins = []
            for member in spec["lodo"]["members"]:
                for name in member["runs"][fold]:
                    path = ROOT / name / "val_logits.npz"
                    with np.load(path, allow_pickle=False) as z:
                        val = np.flatnonzero(meta["split"] == "val")
                        if not np.array_equal(z["image_paths"], meta["image_paths"][val]) or not np.array_equal(z["attribute_names"], meta["attribute_names"]):
                            raise ValueError("source probability row/attribute mismatch")
                    prediction_pins.append(fingerprint(path))
        upstream_pins = [fingerprint(path) for path in lineage_paths]
        rows.append({"fold": fold, "sourceval_counts": counts, "rows": len(selected), "model": model_pin,
                     "config": config_pin, "weights_ready": fingerprint(model_dir / "weights_ready.json"),
                     "meta": meta_pin, "image_cache_stat": _image_stat(cache / cfg.get("cache_file", "images_u8_256x128.npy")),
                     "predictions": prediction_pins, "encoder_fit_audit": leaves, "upstream_pins": upstream_pins,
                     "selected_probability_producers": linked_producers, "historical_audit_exact_leaf_multiset_matches": True})
    if len(cache_paths) != 1:
        raise ValueError("unexpected distinct image caches")
    paths = [Path(__file__), ROOT / "src/upar/learned_gallery_edges.py", ROOT / "src/upar/fixed_gallery_rerank.py",
             ROOT / "src/upar/fixed_gallery_agreement.py", ROOT / "src/upar/pe_gallery_features.py", ROOT / "src/upar/h1_finetune.py",
             ROOT / "src/upar/postproc.py", ROOT / "src/upar/rank_sibling_pipeline.py", ROOT / "tests/test_learned_gallery_edges.py"]
    protocol = {"schema": 1, "candidate": "SOURCEVAL_FITTED_ONEBIT_EDGE_RESCUE_V1", "status": "FROZEN_BEFORE_EXTRACTION_OR_FIT",
                "baseline": fingerprint(SPEC), "prior_feature_protocol": fingerprint(PE / "protocol.json"),
                "prior_fullensemble_fit_audit": fingerprint(AUDIT), "official_rules": fingerprint(RULES),
                "sources": [fingerprint(p) for p in paths], "folds": rows,
                "method": {"k": 20, "alpha": .1, "steps": 1, "predicted_threshold": .5,
                           "restore_only_hamming": 1, "keep_existing_hamming0_weight": 1., "restore_hamming2plus": False,
                           "pair_features": list(FEATURE_NAMES), "ridge_mean_loss": RIDGE, "class_rebalance": False,
                           "source_mass": .5, "fit_edges": "unique undirected reciprocal pairs within each source; target is exact40 GT equality",
                           "graph_mass": "original reciprocal degree; never renormalize retained/rescued mass",
                           "fit_optimizer": "NumPy damped Newton maxiter200 max_abs_gradient1e-8 Armijo1e-4 step_halving maxls50; fixed, no sweep"},
                "fit_boundary": "For outer target S, each encoder and its probability producer excludes ALL S train+val. The SAME PE4(S) extracts the two other sources' official val images. Only these rows/labels fit the pair model. Source IDs only form training groups; not classifier inputs or inference branches.",
                "sourceval_justification": "Official organizer val is allowed for training; these exact val image rows were not encoder-fit. A source-meta-fit on val is training, not independent validation. Prior development exposure and unresolved PETA/PA person overlaps remain. Outer held-source labels are excluded from fit and later used once for fixed mADM evaluation.",
                "no_hidden_fit": True, "no_target_label_HPO": True, "no_other_fold_features": True,
                "evaluation_plan": {
                    "primary": "Fixed learned-rescue minus plain B119, equal-source macro mADM and all three source deltas",
                    "secondary": "Same fixed learned-rescue minus fixed B121 identical-pred40 gate; never select comparator post hoc",
                    "held_labels": "One fixed-recipe full evaluation after all three source-fitted classifiers are complete; held labels only enter downstream metrics and error diagnosis, not pair fit/edge selection/HPO",
                    "uncertainty": "Paired semantic-query bootstrap within each fixed gallery/model, seed20261014 resamples2000, conditional CI plus exactly three source directions; not unseen-domain/Private probability",
                    "outputs": "Saved predictions/graphs/per-query ADM/AP/relevant_count, macro and all source results, singleton/relevant-count slices, worst and best paired cases with unaltered original labels",
                    "acceptance": "A positive exposed-development comparison does not establish Gold. Preserve all adverse sources/slices, source-meta-fit provenance, actual classifier artifacts, and require separate INT8/runtime/official-failure deployment checks",
                    "evaluator": "Separate new module, never modify these frozen extraction/fit modules after extraction starts"
                },
                "input": "canonical uint8 256x128 cache -> existing deterministic 280x140 PE preprocessing; not native-size new detail",
                "extraction": {"batch": 16, "torch_threads": 2, "pooled_width": 1024, "parameter_dtype": "float32", "autocast": "bfloat16",
                               "training_updates": 0, "extra_model": False, "total_rows": sum(r["rows"] for r in rows)},
                "resource": {"gpu_min_free_gib": 3., "host_min_available_gib": 16., "automatic_gpu_launch": False,
                             "new_feature_storage_estimate_mib": 310, "historical_estimated_aggregate_gpu_wall_minutes": [15, 20],
                             "timing_not_guaranteed": True, "image_bytes_hash_rechecked": False},
                "limits": ["0.68 and positive transfer are unproved; this is a new fixed development comparison, not confirmation.",
                           "Edge counts are not independent statistical samples; only three exposed source domains exist.",
                           "Loss/classifier fit may transfer source-specific annotation and repeated-person biases.",
                           "FP32/BF16 features differ from deployed INT8 features; complete-graph numerical and runtime validation remains.",
                           "Submission is blocked pending independent runtime/failure resolution; this protocol authorizes no submission."]}
    exclusive_json(run / "protocol.json", protocol)
    return protocol


def _load(run, fold):
    protocol = read_json(run / "protocol.json")
    row = next(r for r in protocol["folds"] if r["fold"] == fold)
    pins = protocol["sources"] + [protocol[k] for k in ("baseline", "prior_feature_protocol", "prior_fullensemble_fit_audit", "official_rules")]
    pins += [row[k] for k in ("model", "config", "weights_ready", "meta")] + row["predictions"] + row["upstream_pins"]
    verify(pins)
    _validate_audit(row["encoder_fit_audit"], fold)
    if _image_stat(ROOT / row["image_cache_stat"]["path"]) != row["image_cache_stat"]:
        raise ValueError("image cache stat changed")
    return protocol, row, pins


def _source_probabilities(spec, fold, meta, selected):
    mix, total = None, 0.
    val = np.flatnonzero(meta["split"] == "val")
    positions = np.searchsorted(val, selected)
    if not np.array_equal(val[positions], selected):
        raise ValueError("selected rows not all validation")
    for member in spec["lodo"]["members"]:
        arrays = []
        for name in member["runs"][fold]:
            with np.load(ROOT / name / "val_logits.npz", allow_pickle=False) as z:
                if not np.array_equal(z["image_paths"], meta["image_paths"][val]) or not np.array_equal(z["attribute_names"], meta["attribute_names"]):
                    raise ValueError("probability alignment changed")
                correction = np.log(z["pos_weight"]).astype(np.float32)
                arrays.append((1. / (1. + np.exp(-(z["logits_original"] - correction)))).astype(np.float32)[positions])
        p = np.mean(arrays, axis=0)
        mix = p * member["weight"] if mix is None else mix + p * member["weight"]
        total += member["weight"]
    return apply_post((mix / total).astype(np.float32), spec["lodo"]["post"])


def extract(run, fold, *, allow_gpu=False):
    if not allow_gpu:
        raise RuntimeError("explicit root --allow-gpu admission required")
    protocol, row, pins = _load(run, fold)
    import torch
    from safetensors.torch import load_file
    from . import h1_finetune as h
    from .pe_gallery_features import PooledCapture

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    host = next(int(s.split()[1])*1024 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
    if free < 3 * 2**30 or host < 16 * 2**30:
        raise RuntimeError("need actual GPU free3GiB and host available16GiB; guard is not a reservation")
    stage = run / f"source_features_{fold}"
    stage.mkdir()
    start = time.monotonic()
    exclusive_json(stage / "started.json", {"protocol": fingerprint(run / "protocol.json"), "fold": fold,
                                           "GPU_free_bytes": free, "host_available_bytes": host, "training": False})
    try:
        cfg = read_json(ROOT / row["config"]["path"])
        meta = dict(np.load(ROOT / row["meta"]["path"], allow_pickle=False))
        selected, positions = source_val_rows(meta, fold)
        if len(selected) != row["rows"]:
            raise ValueError("source-val row count differs")
        images = np.load(ROOT / row["image_cache_stat"]["path"], mmap_mode="r")
        probabilities = _source_probabilities(read_json(SPEC), fold, meta, selected)
        model, mean, std = h._build_model(cfg["backbone"], pretrained=False, head=cfg["head"])
        model.load_state_dict(load_file(str(ROOT / row["model"]["path"])), strict=True)
        model.cuda().eval()
        capture = PooledCapture(model.head.base)
        torch.cuda.reset_peak_memory_stats()
        features, logits = np.empty((len(selected), 1024), np.float32), np.empty((len(selected), 40), np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for offset in range(0, len(selected), 16):
                end = min(offset + 16, len(selected))
                x = torch.from_numpy(np.ascontiguousarray(images[selected[offset:end]])).cuda()
                x = h._gpu_augment(x, train=False, out_hw=h.BACKBONES[cfg["backbone"]]["img_hw"], mean=mean, std=std)
                value = model(x)
                if capture.value.shape != (len(x), 1024):
                    raise ValueError("pooled-feature shape changed")
                features[offset:end] = capture.value.float().cpu().numpy()
                logits[offset:end] = value.float().cpu().numpy()
                if offset % 1600 == 0:
                    print(json.dumps({"fold": fold, "rows": end, "total": len(selected), "seconds": time.monotonic()-start}), flush=True)
        capture.close()
        if not all(np.isfinite(x).all() for x in (features, logits, probabilities)):
            raise ValueError("nonfinite inference")
        with (stage / "features.npz").open("xb") as stream:
            np.savez(stream, features=features, pe_logits=logits, probabilities=probabilities,
                     cache_rows=selected, val_positions=positions, sources=meta["sources"][selected],
                     image_paths=meta["image_paths"][selected], attribute_names=meta["attribute_names"])
        verify(pins)
        if _image_stat(ROOT / row["image_cache_stat"]["path"]) != row["image_cache_stat"]:
            raise ValueError("image cache stat changed during extraction")
        report = {"status": "SOURCEVAL_FEATURES_COMPLETE", "fold": fold, "rows": len(selected),
                  "sourceval_counts": row["sourceval_counts"], "features": fingerprint(stage / "features.npz"),
                  "model": row["model"], "protocol": fingerprint(run / "protocol.json"), "training_updates": 0,
                  "elapsed_seconds": time.monotonic()-start, "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_total_bytes": total,
                  "actual_process_exit_confirmed": False, "target_source_rows": 0}
        exclusive_json(stage / "completed.json", report)
        return report
    except BaseException as exc:
        exclusive_json(stage / "failure.json", {"type": type(exc).__name__, "error": str(exc), "partial_preserved": True, "automatic_retry": False})
        raise


def fit(run, fold):
    protocol, row, pins = _load(run, fold)
    directory = run / f"source_features_{fold}"
    completed = read_json(directory / "completed.json")
    if completed["status"] != "SOURCEVAL_FEATURES_COMPLETE" or completed["fold"] != fold or completed["model"] != row["model"] or completed["protocol"] != fingerprint(run / "protocol.json"):
        raise ValueError("feature producer/outer-fold provenance mismatch")
    feature_pin = fingerprint(directory / "features.npz")
    if feature_pin != completed["features"]:
        raise ValueError("feature output changed")
    stage = run / f"pair_fit_{fold}"
    stage.mkdir()
    start = time.monotonic()
    exclusive_json(stage / "started.json", {"protocol": fingerprint(run / "protocol.json"), "features": feature_pin,
                                           "fit_role": "sourceval supervised meta-fit, NOT held-source validation"})
    try:
        z = dict(np.load(directory / "features.npz", allow_pickle=False))
        meta = dict(np.load(ROOT / row["meta"]["path"], allow_pickle=False))
        selected, positions = source_val_rows(meta, fold)
        if not np.array_equal(z["cache_rows"], selected) or not np.array_equal(z["val_positions"], positions) or not np.array_equal(z["image_paths"], meta["image_paths"][selected]) or not np.array_equal(z["sources"], meta["sources"][selected]) or not np.array_equal(z["attribute_names"], meta["attribute_names"]):
            raise ValueError("exact source-val alignment failed")
        xrows, yrows, groups, summaries = [], [], [], {}
        for source in FOLDS:
            if source == fold:
                continue
            take = z["sources"] == source
            f, p = z["features"][take], z["probabilities"][take]
            gt = meta["labels"][selected[take]]
            if gt.shape != (len(f), 40) or np.any((gt != 0) & (gt != 1)):
                raise ValueError("official binary40 source labels required")
            unit = unit_features(f)
            graph, counts = build_mutual_graph((np.clip(unit, -1, 1)+1)/2, np.ones(unit.shape[1], np.float64), k=20, chunk_size=256)
            src, dst = one_bit_pairs(p, graph, counts, undirected=True)
            if not len(src):
                raise ValueError(f"no source-fit one-bit pairs for {source}; no target-dependent fallback")
            x = pair_features(f, p, src, dst)
            y = np.all(gt[src] == gt[dst], axis=1).astype(np.float64)
            xrows.append(x)
            yrows.append(y)
            groups.append(np.full(len(y), source))
            summaries[source] = {"rows": len(f), "original_directed_edges": int(counts.sum()),
                                 "onebit_undirected_pairs": len(y), "GT_same_pairs": int(y.sum()), "GT_same_fraction": float(y.mean())}
            with (stage / f"pairs_{source}.npz").open("xb") as stream:
                np.savez_compressed(stream, src_cache_rows=selected[take][src], dst_cache_rows=selected[take][dst], features=x, labels=y)
        model = fit_pair_model(np.concatenate(xrows), np.concatenate(yrows), np.concatenate(groups))
        model.update({"outer_held_source": fold, "training_sources": sorted(summaries),
                      "training_partition": "official organizer val of the two non-target sources; supervised meta-fit",
                      "sourceval_training_rows": row["sourceval_counts"], "target_source_rows_or_labels_fit": 0,
                      "feature_producer": row["model"], "source_features": feature_pin,
                      "protocol": fingerprint(run / "protocol.json"), "hidden_images_or_labels_fit": False})
        exclusive_json(stage / "model.json", model)
        verify(pins + [feature_pin])
        report = {"status": "SOURCEVAL_PAIR_FIT_COMPLETE", "fold": fold, "model": fingerprint(stage / "model.json"),
                  "protocol": fingerprint(run / "protocol.json"), "input_features": feature_pin, "source_summary": summaries,
                  "elapsed_seconds": time.monotonic()-start, "GPU_used": False, "target_rows_or_labels_fit": 0,
                  "held_mADM": "NOT_EVALUATED", "target_source_HPO": False, "training_metrics_not_validation": True,
                  "actual_process_exit_confirmed": False}
        exclusive_json(stage / "completed.json", report)
        return report
    except BaseException as exc:
        exclusive_json(stage / "failure.json", {"type": type(exc).__name__, "error": str(exc), "partial_preserved": True, "automatic_retry": False})
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "extract", "fit"))
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--fold", choices=FOLDS)
    ap.add_argument("--allow-gpu", action="store_true")
    args = ap.parse_args()
    if args.action != "prepare" and args.fold is None:
        ap.error("extract/fit requires --fold")
    if args.action == "prepare":
        result = prepare(args.run)
        print(json.dumps({"protocol": fingerprint(args.run / "protocol.json"), "total_rows": result["extraction"]["total_rows"]}), flush=True)
    else:
        result = extract(args.run, args.fold, allow_gpu=args.allow_gpu) if args.action == "extract" else fit(args.run, args.fold)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
