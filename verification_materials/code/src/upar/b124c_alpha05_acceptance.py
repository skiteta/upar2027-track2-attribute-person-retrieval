"""Runtime release gate for B124C_CONSTANT_RESCUE_ALPHA05 (session 71, 10/17). Stdlib only (loaded by path by
scripts/codabench_submit.py). It proves, from files on disk, that the exact ZIP to be sent:
  - is the built B124C ZIP (SHA-256 of package_v1/complete.json) whose parent is the official B123C ZIP (995164, Finished);
  - differs from that parent only in assets/config.json and b121_runtime.py, every other member byte-identical;
  - carries alpha .5 in both gallery_rerank.alpha and constant_rescue_diffusion_alpha and in both runtime literals, the
    production one-bit constant unchanged, and no other config change than candidate identity;
  - was executed on the local CUDA GPU over the full validation set (33,407 images x 3,462 queries) from this exact ZIP;
  - produced FP32 distances that equal, bit for bit, an independent recomputation from the same inputs (0 unequal values,
    0 queries with different ranks), with every pinned capture file still unchanged.
It does not certify A10 speed (the graph and models equal the parent's, whose official run took 252.3 s) or quality."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05"
PARENT_RUN = ROOT / "artifacts/runs/20261017T1422+0900_b123_alpha03"
KIND = "b124c_constant_alpha05_runtime_acceptance_v1"
CANDIDATE = "B124C_CONSTANT_RESCUE_ALPHA05"
PARENT_CANDIDATE = "B123C_CONSTANT_RESCUE_ALPHA03"
PARENT_SHA = "4d85d05f147472354ce21f30009e9f5a7b7c2a79ff0c9ab5d12b4c3a307afb40"
PARENT_SUBMISSION = 995164
ALPHA, PARENT_ALPHA = 0.5, 0.3
CONSTANT = 0.5599229900622915
REFERENCE_KIND = "exact_new_alpha05_same_input_CUDA32_graph_independent_row_smoothing"
CHANGED = {"assets/config.json", "b121_runtime.py"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned(ref):
    p = Path(ref["path"])
    if not p.is_file() or sha(p) != ref["sha256"]:
        raise ValueError("pinned file changed or missing: " + str(p))
    return p


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}


def member_digests(zf):
    names = sorted(i.filename for i in zf.infolist() if not i.is_dir())
    if len(set(names)) != len(names) or any(n.startswith("/") or ".." in n.split("/") for n in names):
        raise ValueError("unsafe or duplicate ZIP members")
    return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in names}


def verify_evidence(artifact, acc):
    if (acc.get("kind") != KIND or acc.get("status") != "ACCEPTED_RUNTIME_PREFLIGHT" or acc.get("candidate") != CANDIDATE
            or acc.get("alpha") != ALPHA or acc.get("grants_submission_authority") is not False):
        raise ValueError("dedicated B124C acceptance with explicit limits required")
    artifact = Path(artifact).resolve()
    if pinned(acc["artifact"]).resolve() != artifact:
        raise ValueError("acceptance names another artifact")
    package = json.loads(pinned(acc["package_report"]).read_text())
    if (package.get("status") != "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED" or package.get("candidate") != CANDIDATE
            or package.get("alpha") != ALPHA or package.get("constant") != CONSTANT
            or package["zip"]["sha256"] != acc["artifact"]["sha256"] or package["parent"]["sha256"] != PARENT_SHA):
        raise ValueError("built B124C package report required")
    official = json.loads(pinned(acc["parent_official_receipt"]).read_text())
    if (official["submission"]["id"] != PARENT_SUBMISSION or official["submission"]["status"] != "Finished"
            or official["artifact"]["sha256"] != PARENT_SHA):
        raise ValueError("official Finished B123C parent required")
    parent_zip = pinned(acc["parent_artifact"])
    with zipfile.ZipFile(parent_zip) as old, zipfile.ZipFile(artifact) as new:
        a, b = member_digests(old), member_digests(new)
        if set(a) != set(b):
            raise ValueError("member set differs from the B123C parent")
        changed = {n for n in a if a[n] != b[n]}
        if changed != CHANGED:
            raise ValueError("only config and runtime may change: " + str(sorted(changed)))
        old_cfg, new_cfg = json.loads(old.read("assets/config.json")), json.loads(new.read("assets/config.json"))
        runtime_old, runtime_new = old.read("b121_runtime.py").decode(), new.read("b121_runtime.py").decode()
    expected = json.loads(json.dumps(old_cfg))
    if (old_cfg.get("candidate") != PARENT_CANDIDATE or old_cfg["gallery_rerank"]["alpha"] != PARENT_ALPHA
            or old_cfg["constant_rescue_diffusion_alpha"] != PARENT_ALPHA or old_cfg["onebit_rescue_constant"] != CONSTANT):
        raise ValueError("unexpected parent config")
    expected["candidate"] = CANDIDATE
    expected["gallery_rerank"]["alpha"] = ALPHA
    expected["constant_rescue_diffusion_alpha"] = ALPHA
    if new_cfg != expected or package.get("config") != new_cfg:
        raise ValueError("config may change only in candidate identity and the two alpha fields")
    if (runtime_new.count("rescue, alpha=.5)") != 1 or runtime_new.count("diffusion_alpha=.5)") != 1
            or "alpha=.3)" in runtime_new
            or runtime_new.replace("alpha=.5)", "alpha=.3)") != runtime_old):
        raise ValueError("runtime may change only the two alpha literals .3 -> .5")
    runtime = json.loads(pinned(acc["runtime_report"]).read_text())
    if (runtime.get("candidate") != CANDIDATE or runtime.get("artifact", {}).get("sha256") != acc["artifact"]["sha256"]
            or runtime.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING" or runtime.get("device") != "cuda"
            or runtime.get("forced_plan") is not None or (runtime.get("rows"), runtime.get("queries")) != (33407, 3462)
            or runtime.get("config") != new_cfg):
        raise ValueError("full CUDA replay of this exact ZIP required")
    reference = json.loads(pinned(acc["reference_report"]).read_text())
    if (reference.get("status") != "PASS" or reference.get("kind") != REFERENCE_KIND or reference.get("candidate") != CANDIDATE
            or reference.get("alpha") != ALPHA or reference.get("constant") != CONSTANT
            or reference.get("unequal_distance_values") != 0 or reference.get("queries_with_different_ranks") != 0):
        raise ValueError("bit-exact independent reference PASS required")
    for name in ("capture.npz", "features.npy", "distances.npy", "execute_report.json"):
        pinned(reference["capture"][name])
    if reference["capture"]["execute_report.json"]["sha256"] != acc["runtime_report"]["sha256"]:
        raise ValueError("reference was computed on another capture")
    return {"kind": KIND, "candidate": CANDIDATE, "artifact_sha256": acc["artifact"]["sha256"], "alpha": ALPHA,
            "constant": CONSTANT, "parent": {"candidate": PARENT_CANDIDATE, "submission": PARENT_SUBMISSION, "sha256": PARENT_SHA},
            "changed_members": sorted(CHANGED), "full_cuda_replay": True, "bit_exact_reference": True,
            "local_time_scale": runtime.get("time_scale"),
            "official_runtime": "unknown; same models and graph as B123C (official 252.3 s), not an A10 guarantee",
            "quality_or_gold_claim": False, "submission_authority": False}


def verify_release(artifact_path, acceptance_path, **_ignored):
    return verify_evidence(Path(artifact_path).resolve(), json.loads(Path(acceptance_path).read_text()))


def build_acceptance(output):
    output = Path(output).resolve()
    if output.parent != RUN or output.exists():
        raise ValueError("new B124C acceptance inside its run dir only; never overwrite")
    acc = {"kind": KIND, "status": "ACCEPTED_RUNTIME_PREFLIGHT", "candidate": CANDIDATE, "alpha": ALPHA,
           "artifact": pin(RUN / "package_v1" / (CANDIDATE + ".zip")), "verifier_source": pin(Path(__file__)),
           "package_report": pin(RUN / "package_v1/complete.json"),
           "parent_artifact": pin(PARENT_RUN / "package_v1" / (PARENT_CANDIDATE + ".zip")),
           "parent_official_receipt": pin(PARENT_RUN / "watch_v1/final_receipt.json"),
           "runtime_report": pin(RUN / "full_replay_v1/execute_report.json"),
           "reference_report": pin(RUN / "reference_v1/report.json"),
           "official_hardware_runtime_guaranteed": False, "grants_submission_authority": False, "quality_or_gold_claim": False}
    result = verify_evidence(Path(acc["artifact"]["path"]), acc)
    with output.open("x") as stream:
        json.dump(acc, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"acceptance": pin(output), "verification": result}


if __name__ == "__main__":
    import sys
    print(json.dumps(build_acceptance(sys.argv[1]), indent=1))
