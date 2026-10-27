"""Reference check for a replayed candidate: recompute the final distances from the captured inputs with the run-dir
rerank_param.py (the module the knob search evaluated; its default path is bit-exact with the deployed package_v14
code) and the chosen K, plus a copy of the package's own scoring function, and require bit equality with the
distances the packaged run.py returned. Also reports the val mADM of the replayed distances.
usage: reference_candidate.py BUILD_DIR REPLAY_DIR
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")


def main() -> None:
    build = Path(sys.argv[1]).resolve()
    rep = Path(sys.argv[2]).resolve()
    report = json.loads((build / "complete.json").read_text())
    cfg = report["config"]
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(build / "package"))  # b121_* helpers of the exact package (pinned by the ZIP)
    spec = importlib.util.spec_from_file_location("rp_ref", Path(report["module"]["source"]))
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    sys.path.insert(0, str(HERE))
    import build_candidate as bc  # noqa: E402
    import knob_eval as ke  # noqa: E402
    K = bc.merged_k(report["K_overrides"], rp.DEFAULT_K)
    cap = np.load(rep / "capture.npz", allow_pickle=False)
    names = [str(x) for x in cap["attribute_names"]]
    assert names == list(cfg["attribute_names"])
    w = np.asarray(cfg["attribute_distance_weights"], dtype=np.float32)
    assert cfg["bernoulli_nll_epsilon"] == ke.CFG["bernoulli_nll_epsilon"] and cfg["bernoulli_nll_lambda"] == ke.CFG["bernoulli_nll_lambda"]
    assert cfg["post"]["entropy_delta"] == ke.CFG["post"]["entropy_delta"]
    t0 = time.monotonic()
    args = (cap["probabilities"], np.load(rep / "features.npy"), np.load(rep / "eva_features.npy"),
            np.load(rep / "hp_features.npy"), np.load(rep / "member_std.npy"), cap["queries"], names, ke.make_score_fn(w),
            lambda *a, **k: None)
    d_ref = rp.run_b132c_pentj_body_rerank(*args, K=K)
    blend = report.get("blend")
    if blend:  # independent re-implementation of the per-query expected-match gate (same chunking as the package)
        import torch
        d_strong = rp.run_b132c_pentj_body_rerank(*args, K={**K, **blend["strong"]})
        pr = np.clip(np.asarray(cap["probabilities"], dtype=np.float64), 1e-5, 1 - 1e-5)
        lp, ln = torch.from_numpy(np.log(pr)).cuda(), torch.from_numpy(np.log1p(-pr)).cuda()
        qq = np.asarray(cap["queries"], dtype=np.float64)
        le = np.empty(len(qq))
        with torch.inference_mode():
            for s in range(0, len(qq), 256):
                qt = torch.from_numpy(qq[s:s + 256]).cuda()
                le[s:s + 256] = torch.logsumexp(qt @ lp.T + (1.0 - qt) @ ln.T, dim=1).cpu().numpy()
        g = np.clip((le - np.log(blend["a"])) / blend["b"], 0.0, 1.0).astype(np.float32)[:, None]
        d_ref = ((np.float32(1.0) - g) * d_ref + g * d_strong).astype(np.float32)
    d_pkg = np.load(rep / "distances.npy")
    unequal = int(np.count_nonzero(d_ref != d_pkg))
    rank_diff = int(np.count_nonzero(np.any(np.argsort(d_ref, axis=1, kind="stable") != np.argsort(d_pkg, axis=1, kind="stable"), axis=1)))
    from upar.data import load_retrieval_split
    split = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    nm = np.load(ke.S / "extracted_features_v1/graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy")
    qi = np.unique(split.gallery_query_ids)
    assert np.array_equal(split.queries[qi].astype(np.float32), cap["queries"]), "capture query order differs from eval ctx"
    r = ke.eval_q(d_pkg, qi, nm, split.gallery_query_ids.astype(np.int64))
    out = {"status": "PASS" if unequal == 0 and rank_diff == 0 else "FAIL", "candidate": report["candidate"],
           "unequal_distance_values": unequal, "queries_with_different_ranks": rank_diff,
           "distance_max_abs": float(np.max(np.abs(d_ref - d_pkg))), "val_mADM_submission_models": float(r["adm"].mean()),
           "zip_sha256": report["zip"]["sha256"], "wall_seconds": time.monotonic() - t0}
    (rep / "reference_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
