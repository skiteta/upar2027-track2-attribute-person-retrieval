"""Local public-only exact B121 runtime capture and independent CPU reference.

This file is NOT included in the submitted ZIP. Runtime inference is one normal
rank_gallery call; captured arrays stay local. Reference CPU time is reported
separately and never presented as submission runtime. No tolerant old cached
mADM gate is reused to establish correctness of the new reranking mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

from .gallery_feature_package import NAME, pin, sha, write_new
from .numpy_blas_scope import numpy_blas_threads

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SPLIT = ROOT / "starter_kit/data/annotations/task2/val"
IMAGE_ROOT = ROOT / "starter_kit/data"


def validate_extracted(artifact, extracted):
    """Verify exact archive member set/bytes; reject links and extra files."""
    artifact, extracted = Path(artifact).resolve(), Path(extracted).resolve()
    expected = {}
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP members")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe/non-file ZIP member")
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 * 2**20), b""):
                    digest.update(block)
            expected[name] = digest.hexdigest()
    actual = {p.relative_to(extracted).as_posix(): p for p in extracted.rglob("*") if p.is_file()}
    if set(expected) != set(actual) or any(p.is_symlink() for p in extracted.rglob("*")):
        raise ValueError("extracted member set or symlinks differ")
    if any(sha(actual[name]) != digest for name, digest in expected.items()):
        raise ValueError("extracted bytes differ from exact ZIP")
    config = json.loads((extracted / "assets/config.json").read_text())
    if config["candidate"] != NAME or config["time_budget_s"] != 292 or config["rank_reserve_s"] != 40:
        raise ValueError("not fixed B121 production config")
    return expected, config


def execute(artifact, extracted, output, expected_sha, *, force_plan=None, time_scale=1):
    """Root-admitted GPU worker only. Full public val, no hidden-data option."""
    from .data import load_retrieval_split
    from .submission import load_submission_module, make_fixture, validate_output

    artifact, extracted, output = map(lambda p: Path(p).resolve(), (artifact, extracted, output))
    if sha(artifact) != expected_sha:
        raise ValueError("artifact differs from admitted ZIP")
    if not np.isfinite(time_scale) or not 1 <= time_scale <= 4:
        raise ValueError("finite local time scale in [1,4] required")
    members, config = validate_extracted(artifact, extracted)
    sources = [pin(Path(__file__)), pin(artifact)]
    annotations = [pin(p) for p in sorted(PUBLIC_SPLIT.rglob("*")) if p.is_file()]
    split = load_retrieval_split(PUBLIC_SPLIT)
    if (split.gallery_size, split.query_count) != (33407, 3462):
        raise ValueError("expected full public33407 gallery /3462 queries")
    fixture = make_fixture(split, query_count=split.query_count, gallery_size=split.gallery_size, image_root=IMAGE_ROOT)
    if any(not Path(row["image_path"]).is_file() for row in fixture["gallery"]):
        raise ValueError("missing public input image")
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "started.json",
        {
            "status": "RUNNING_NOT_ACCEPTED",
            "sources": sources,
            "annotations": annotations,
            "archive_members": members,
            "time_scale": time_scale,
            "force_plan": force_plan,
            "public_only": True,
            "runtime_sources_unmodified": True,
            "local_predictor_capture": True,
        },
    )
    os.environ["UPAR_TIME_SCALE"] = str(time_scale)
    if force_plan is None:
        os.environ.pop("UPAR_FORCE_PLAN", None)
    else:
        if force_plan != "A":
            raise ValueError("only existing full three-member A plan")
        os.environ["UPAR_FORCE_PLAN"] = force_plan
    # Submitted helpers live next to run.py. No source-tree helper substitutes.
    if any(name in sys.modules for name in ("b121_graph", "b121_agreement", "b121_blas", "b121_runtime")):
        raise RuntimeError("fresh worker required; submitted helper already imported")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(extracted))
    captured = []
    try:
        started = time.monotonic()
        module = load_submission_module(extracted / "run.py")
        module.load_model()
        loaded = time.monotonic()
        if str(module._DEVICE) != "cuda":
            raise RuntimeError("full acceptance requires actual CUDA, no CPU/fallback")
        predictor = module.predict_attributes

        def capture(*args, **kwargs):
            value = predictor(*args, **kwargs)
            captured.append(np.asarray(value).copy())
            return value

        module.predict_attributes = capture
        result = module.rank_gallery(fixture)
        finished = time.monotonic()
        module.predict_attributes = predictor
        kind, distances = validate_output(result, query_count=split.query_count, gallery_size=split.gallery_size)
        if kind != "distances" or distances.dtype != np.float32 or len(captured) != 1:
            raise ValueError("expected one FP32 probability producer and distance output")
        probabilities = captured[0]
        features = np.asarray(module._GALLERY_FEATURES)
        if (
            probabilities.shape != (33407, 40)
            or probabilities.dtype != np.float32
            or features.shape != (33407, 1024)
            or features.dtype != np.float32
            or not np.isfinite(probabilities).all()
            or not np.isfinite(features).all()
        ):
            raise ValueError("incomplete actual probabilities/features")
        np.save(output / "distances.npy", distances, allow_pickle=False)
        np.save(output / "features.npy", features, allow_pickle=False)
        np.savez(
            output / "capture.npz",
            probabilities=probabilities,
            queries=np.asarray(fixture["queries"], np.float32),
            image_paths=np.asarray(split.image_paths),
            attribute_names=np.asarray(fixture["attribute_names"]),
        )
        if validate_extracted(artifact, extracted)[0] != members:
            raise ValueError("runtime package changed")
        for ref in sources + annotations:
            if pin(ref["path"]) != ref:
                raise ValueError("source or input changed during replay")
        report = {
            "schema": 1,
            "status": "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING",
            "artifact": pin(artifact),
            "sources": sources,
            "annotations": annotations,
            "archive_members": members,
            "outputs": [pin(output / name) for name in ("distances.npy", "features.npy", "capture.npz")],
            "load_seconds": loaded - started,
            "rank_seconds": finished - loaded,
            "runtime_seconds_before_local_saves": finished - started,
            "time_scale": time_scale,
            "forced_plan": force_plan,
            "device": str(module._DEVICE),
            "rows": 33407,
            "queries": 3462,
            "new_reference_pass": False,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "Exact ZIP full public val, capture is local harness only; not official hidden runtime or independent heldout quality.",
            "config": config,
        }
        write_new(output / "execute_report.json", report)
        return report
    except BaseException as exc:
        write_new(
            output / "failure.json",
            {"type": type(exc).__name__, "error": str(exc), "partial_preserved": True, "retry": False},
        )
        raise


def independent_graph(features):
    """Independent implementation of fixed cosine-order mutual20, bounded RAM.

    Mirrors specified FP64 arithmetic to make exact floating-tie behavior
    comparable, but does not import/call the deployed graph/gating functions.
    """
    x = np.asarray(features, np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty matrix required")
    maximum = np.abs(x).max(axis=1, keepdims=True)
    if np.any(maximum == 0):
        raise ValueError("zero feature")
    x = x / maximum
    x /= np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    x = (np.clip(x, -1, 1) + 1) / 2
    n, k = len(x), min(20, len(x) - 1)
    if k == 0:
        return np.empty((n, 0), np.int64), np.zeros(n, np.int64)
    norms = np.einsum("ij,ij->i", x, x)
    directed = np.empty((n, k), np.int64)
    for start in range(0, n, min(256, n - 1)):
        end = min(n, start + min(256, n - 1))
        distance = norms[start:end, None] + norms[None, :] - 2 * (x[start:end] @ x.T)
        np.maximum(distance, 0, out=distance)
        distance[np.arange(end - start), np.arange(start, end)] = np.inf
        partition = np.argpartition(distance, k - 1, axis=1)[:, :k]
        for local, subset in enumerate(partition):
            threshold = distance[local, subset].max()
            strictly = np.where(distance[local] < threshold)[0]
            ties = np.where(distance[local] == threshold)[0]
            chosen = np.r_[strictly, ties[: k - len(strictly)]]
            directed[start + local] = chosen[np.lexsort((chosen, distance[local, chosen]))]
    neighbors = np.full_like(directed, -1)
    counts = np.zeros(n, np.int64)
    for row in range(n):
        keep = np.array([row in directed[other] for other in directed[row]])
        counts[row] = keep.sum()
        neighbors[row, : counts[row]] = directed[row, keep]
    return neighbors, counts


def independent_base(probabilities, queries, attribute_names, config):
    p, q = np.asarray(probabilities, np.float32), np.asarray(queries, np.float32)
    order = [config["attribute_names"].index(str(name)) for name in attribute_names]
    weights = np.asarray(config["attribute_distance_weights"], np.float32)[order]
    ent = (config.get("post") or {}).get("entropy_delta")
    if ent:
        f64 = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
        entropy = -(f64 * np.log(f64) + (1 - f64) * np.log1p(-f64)) / np.log(2.0)
        factor = np.exp((1.0 - entropy) * float(ent))
        weights = (weights.astype(np.float64)[None, :] * (factor / factor.mean(axis=1, keepdims=True))).astype(
            np.float32
        )
    eps, lam = np.float32(config["bernoulli_nll_epsilon"]), np.float32(config["bernoulli_nll_lambda"])
    clipped = np.clip(p, eps, np.float32(1) - eps)
    neg, pos = -np.log1p(-clipped).astype(np.float32), -np.log(clipped).astype(np.float32)
    constant = (np.float32(1) - lam) * (p * weights).sum(axis=1, dtype=np.float32) + lam * (neg * weights).sum(
        axis=1, dtype=np.float32
    )
    change = (np.float32(1) - lam) * weights * (np.float32(1) - np.float32(2) * p) + lam * weights * (pos - neg)
    return constant[None, :] + q @ change.T


def independent_smooth(base, probabilities, neighbors, counts):
    """Independent row-wise graph gate + simultaneous fixed mass propagation."""
    bits = np.asarray(probabilities) >= 0.5
    out = base.copy()
    k = neighbors.shape[1]
    for row, original_degree in enumerate(counts):
        if not original_degree:
            continue
        ids = neighbors[row, :original_degree]
        ids = ids[np.all(bits[ids] == bits[row], axis=1)]
        if not len(ids):
            continue
        # Padded original width preserves the declared FP32 reduction order.
        # Deployed NumPy advanced indexing makes the query axis contiguous.
        # Preserve that strided reduction (otherwise NumPy pairwise summation
        # creates ~1e-6 differences despite the same mathematical formula).
        values = np.zeros((k, len(base)), dtype=base.dtype).T
        values[:, : len(ids)] = base[:, ids]
        mean = values.sum(axis=1, dtype=base.dtype) / np.float32(len(ids))
        mass = np.float32(0.1) * (np.float32(len(ids)) / np.float32(original_degree))
        out[:, row] = base[:, row] + mass * (mean - base[:, row])
    return out


def verify(capture_dir, output, process_receipt):
    capture_dir, output = Path(capture_dir).resolve(), Path(output).resolve()
    receipt = json.loads((capture_dir / "execute_report.json").read_text())
    if receipt["status"] != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING":
        raise ValueError("actual captured runtime required")
    actual_process = json.loads(Path(process_receipt).read_text())
    if (
        type(actual_process.get("actual_exit_code")) is not int
        or actual_process["actual_exit_code"] != 0
        or not actual_process.get("exec_session")
        or actual_process.get("execute_report_sha256") != sha(capture_dir / "execute_report.json")
        or actual_process.get("artifact_sha256") != receipt["artifact"]["sha256"]
    ):
        raise ValueError("actual successful process exit receipt bound to exact report/ZIP required")
    inputs = (
        [pin(capture_dir / "execute_report.json"), pin(Path(__file__)), pin(process_receipt)]
        + receipt["outputs"]
        + [receipt["artifact"]]
    )
    for ref in inputs:
        if pin(ref["path"]) != ref:
            raise ValueError("captured input changed")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {"inputs": inputs, "role": "independent CPU reference, not submission runtime"})
    started, cpu = time.monotonic(), time.process_time()
    with np.load(capture_dir / "capture.npz", allow_pickle=False) as value:
        p, q, names = value["probabilities"], value["queries"], value["attribute_names"]
    features = np.load(capture_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    actual = np.load(capture_dir / "distances.npy", mmap_mode="r", allow_pickle=False)
    with numpy_blas_threads(2) as blas:
        graph, counts = independent_graph(features)
    reference = np.lib.format.open_memmap(
        output / "independent_distances.npy", mode="w+", dtype=np.float32, shape=actual.shape
    )
    unequal, ranks_differ, max_abs = 0, 0, 0.0
    for start in range(0, len(q), 64):
        end = min(len(q), start + 64)
        base = independent_base(p, q[start:end], names, receipt["config"])
        expected = independent_smooth(base, p, graph, counts)
        reference[start:end] = expected
        observed = np.asarray(actual[start:end])
        unequal += int(np.count_nonzero(observed != expected))
        max_abs = max(max_abs, float(np.abs(observed - expected).max()))
        ranks_differ += sum(
            not np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable"))
            for a, b in zip(observed, expected)
        )
    reference.flush()
    for ref in inputs:
        if pin(ref["path"]) != ref:
            raise ValueError("source/capture changed during reference")
    report = {
        "schema": 1,
        "status": "PASS" if unequal == 0 and ranks_differ == 0 else "FAIL",
        "kind": "exact_reranked_reference_not_old_cached_madm_tolerance",
        "inputs": inputs,
        "reference": pin(output / "independent_distances.npy"),
        "shape": list(actual.shape),
        "unequal_distance_values": unequal,
        "distance_max_abs": max_abs,
        "queries_with_different_ranks": ranks_differ,
        "runtime_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time() - cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "GPU_seconds": 0,
        "graph_numpy_blas": blas,
        "runtime_cost_excludes_this_reference": True,
        "quality_or_gold_claim": False,
        "scope": "Independent implementation from actual same-ZIP probabilities/features; exact distance/rank parity, not official hidden score.",
    }
    write_new(output / "report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("exact new-reranker reference mismatch; no tolerance loosening")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("execute")
    for name in ("artifact", "extracted", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--expected-sha", required=True)
    run.add_argument("--force-plan", choices=["A"])
    run.add_argument("--time-scale", type=float, default=1)
    check = sub.add_parser("verify")
    check.add_argument("--capture-dir", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--process-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "execute":
        result = execute(
            args.artifact,
            args.extracted,
            args.output,
            args.expected_sha,
            force_plan=args.force_plan,
            time_scale=args.time_scale,
        )
    else:
        result = verify(args.capture_dir, args.output, args.process_receipt)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
