"""Runtime release gate (B153C family = B151C family plus per-attribute keys "a<idx>" in the multipliers and an optional
recorded replacement of config post.renorm, the exclusive-group renormalisation that run.py applies to the ensemble and
member probabilities; B151C family: per-group attribute_distance_weights multipliers + module base rerank_param4_frozen.py = experimental module with attribute-space PRF)
for rerank-only candidates built on the B139C ZIP (takeover session, 2026-10-22). B139C was never
scored; instead of an official receipt this gate re-verifies B139C's own release evidence (runtime_acceptance_v14.json with
its stdlib verifier: exact build chained to the official B123C/B124C line, full CUDA replay, bit-exact reference). Otherwise
the same proofs as b140c_rerank_only_acceptance.py; config may additionally change models[i].member_weights to the
build report's recorded 3 x 40 weights (pooling only; models/heads untouched).
Stdlib only (loaded by path by scripts/codabench_submit.py). It proves, from files on disk, that the exact ZIP to be sent:
  - is the ZIP recorded in its build report (artifacts/runs/20261021T2250+0900_takeover_knob_search/<build>/complete.json)
    whose parent is the official B136C ZIP (submission 1005359, Finished, Public 0.701494);
  - differs from that parent only in assets/config.json and b127_jaccard_prob_diffusion.py, every other member
    byte-identical (same models, heads, member weights, run.py and b121_runtime.py);
  - config differs only in candidate identity and the descriptive "rerank_knobs" block;
  - the rerank module equals the run dir's rerank_param.py (generated from the deployed package_v14 module, default path
    bit-exact) plus exactly one DEFAULT_K.update(...) line carrying the build report's merged knobs;
  - was executed on the local CUDA GPU over the full validation set (33,407 images x 3,462 queries) from this exact ZIP;
  - produced FP32 distances equal, bit for bit, to a recomputation from the captured inputs by the run dir's
    rerank_param.py with the same knobs (0 unequal values, 0 queries with different ranks), capture files unchanged.
It does not certify A10 speed or quality."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/runs/20261021T2250+0900_takeover_knob_search"
PARENT_RUN = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"
KIND = "b153c_b139c_rerank_aqe_attrweights_post_runtime_acceptance_v1"
PARENT_CANDIDATE = "B139C_ANAT_MULTIRES_SPHERICAL_HEAD_RERANK"
PARENT_SHA = "93507697d74aeba9f27df7b8fb2b411eaccc4f8ac8d1b1990dace1fbe2f4334d"
PARENT_SUBMISSION = None  # never submitted; released by its own acceptance
CHANGED = {"assets/config.json", "b127_jaccard_prob_diffusion.py"}
MODULE = "b127_jaccard_prob_diffusion.py"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned(ref):
    p = Path(ref["path"])
    if not p.is_absolute():
        p = ROOT / p
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
    cand = acc.get("candidate")
    if (acc.get("kind") != KIND or acc.get("status") != "ACCEPTED_RUNTIME_PREFLIGHT" or not isinstance(cand, str)
            or acc.get("grants_submission_authority") is not False):
        raise ValueError("dedicated B153C-family acceptance with explicit limits required")
    artifact = Path(artifact).resolve()
    if pinned(acc["artifact"]).resolve() != artifact:
        raise ValueError("acceptance names another artifact")
    build_report = pinned(acc["package_report"])
    if build_report.parent.parent != RUN:
        raise ValueError("build report must live in the takeover run dir")
    package = json.loads(build_report.read_text())
    if (package.get("status") != "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED" or package.get("candidate") != cand
            or package["zip"]["sha256"] != acc["artifact"]["sha256"] or package["parent"]["sha256"] != PARENT_SHA):
        raise ValueError("build report of this exact ZIP with the B139C parent required")
    import importlib.util
    pv = pinned(acc["parent_verifier"])
    spec = importlib.util.spec_from_file_location("parent_release_gate", pv)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parent_proof = mod.verify_release(pinned(acc["parent_artifact"]), pinned(acc["parent_acceptance"]))
    if not isinstance(parent_proof, dict) or parent_proof.get("artifact_sha256") != PARENT_SHA or parent_proof.get("candidate") != PARENT_CANDIDATE:
        raise ValueError("B139C parent must pass its own release verifier")
    parent_zip = pinned(acc["parent_artifact"])
    with zipfile.ZipFile(parent_zip) as old, zipfile.ZipFile(artifact) as new:
        a, b = member_digests(old), member_digests(new)
        if set(a) != set(b):
            raise ValueError("member set differs from the B139C parent")
        changed = {n for n in a if a[n] != b[n]}
        if changed != CHANGED:
            raise ValueError("only config and the rerank module may change: " + str(sorted(changed)))
        old_cfg, new_cfg = json.loads(old.read("assets/config.json")), json.loads(new.read("assets/config.json"))
        module_new = new.read(MODULE).decode()
    if old_cfg.get("candidate") != PARENT_CANDIDATE:
        raise ValueError("unexpected parent config")
    expected = json.loads(json.dumps(old_cfg))
    expected["candidate"] = cand
    expected["rerank_knobs"] = new_cfg.get("rerank_knobs")
    am = package.get("attr_wmul")
    if not am:
        raise ValueError("recorded attribute-weight multipliers required")
    agroups = {"age": [0, 1, 2], "gender": [3], "hair": [4, 5, 6], "ub_len": [7], "ub_color": list(range(8, 20)),
               "lb_len": [20], "lb_color": list(range(21, 33)), "lb_type": [33, 34], "acc": [35, 36, 37, 38, 39],
                               "acc_backpack": [35], "acc_bag": [36], "acc_glasses": [37, 38], "acc_hat": [39]}
    aw = [float(x) for x in old_cfg["attribute_distance_weights"]]
    for g, mult in am.items():
        for j in (agroups[g] if g in agroups else [int(g[1:])]):  # "a<idx>" = one attribute
            aw[j] = aw[j] * float(mult)
    expected["attribute_distance_weights"] = aw
    pr = package.get("post_renorm")
    if pr is not None:  # recorded replacement of post.renorm; clip and entropy_delta stay the parent's
        groups = [[int(a), int(b), float(al)] for a, b, al in pr]
        if any(not (0 <= a < b <= 40) or not (0.0 <= al <= 1.0) for a, b, al in groups):
            raise ValueError("renorm groups must be column ranges in [0, 40) with alpha in [0, 1]")
        expected["post"]["renorm"] = groups
    mw = package.get("member_weights")
    if not mw or len(mw) != 3 or any(len(r) != 40 for r in mw):
        raise ValueError("recorded 3 x 40 member weights required")
    for m_idx in range(3):
        expected["models"][m_idx]["member_weights"] = [round(float(x), 6) for x in mw[m_idx]]
    if new_cfg != expected or package.get("config") != new_cfg or not isinstance(new_cfg.get("rerank_knobs"), dict):
        raise ValueError("config may change only in candidate identity and the descriptive rerank_knobs block")
    base_src = pinned(acc["rerank_param_source"]).read_text()
    if pinned(acc["rerank_param_source"]).resolve() != (RUN / "rerank_param4_frozen.py").resolve():
        raise ValueError("rerank_param4_frozen.py of the takeover run dir required")
    tail = module_new[len(base_src):]
    if (not module_new.startswith(base_src) or tail.count("DEFAULT_K.update(") != 1
            or ("DEFAULT_K.update(" + package["K_merged_repr"] + ")\n") not in tail
            or hashlib.sha256(module_new.encode()).hexdigest() != package["module"]["sha256"]):
        raise ValueError("module must be rerank_param.py plus the single recorded DEFAULT_K.update line")
    runtime = json.loads(pinned(acc["runtime_report"]).read_text())
    if (runtime.get("candidate") != cand or runtime.get("artifact", {}).get("sha256") != acc["artifact"]["sha256"]
            or runtime.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING" or runtime.get("device") != "cuda"
            or runtime.get("forced_plan") is not None or (runtime.get("rows"), runtime.get("queries")) != (33407, 3462)
            or runtime.get("config") != new_cfg):
        raise ValueError("full CUDA replay of this exact ZIP required")
    for ref in runtime.get("outputs", []):
        pinned(ref)
    reference = json.loads(pinned(acc["reference_report"]).read_text())
    if (reference.get("status") != "PASS" or reference.get("candidate") != cand
            or reference.get("zip_sha256") != acc["artifact"]["sha256"]
            or reference.get("unequal_distance_values") != 0 or reference.get("queries_with_different_ranks") != 0):
        raise ValueError("bit-exact reference PASS required")
    if Path(acc["reference_report"]["path"]).resolve().parent != Path(acc["runtime_report"]["path"]).resolve().parent:
        raise ValueError("reference must be computed on this replay's capture")
    return {"kind": KIND, "candidate": cand, "artifact_sha256": acc["artifact"]["sha256"],
            "parent": {"candidate": PARENT_CANDIDATE, "submission": PARENT_SUBMISSION, "sha256": PARENT_SHA},
            "changed_members": sorted(CHANGED), "rerank_knobs": package.get("K_overrides"), "full_cuda_replay": True,
            "bit_exact_reference": True, "local_time_scale": runtime.get("time_scale"),
            "val_mADM_submission_models": reference.get("val_mADM_submission_models"),
            "official_runtime": "unknown; same models/run.py as B139C (graph and encoder workload of B136C/B137C, official 252-258 s), rerank knobs may add seconds",
            "quality_or_gold_claim": False, "submission_authority": False}


def verify_release(artifact_path, acceptance_path, **_ignored):
    return verify_evidence(Path(artifact_path).resolve(), json.loads(Path(acceptance_path).read_text()))


def build_acceptance(build_dir, replay_dir, output):
    build_dir, replay_dir, output = Path(build_dir).resolve(), Path(replay_dir).resolve(), Path(output).resolve()
    if output.parent != RUN or output.exists():
        raise ValueError("new acceptance inside the takeover run dir only; never overwrite")
    package = json.loads((build_dir / "complete.json").read_text())
    acc = {"kind": KIND, "status": "ACCEPTED_RUNTIME_PREFLIGHT", "candidate": package["candidate"],
           "artifact": pin(package["zip"]["path"]), "verifier_source": pin(Path(__file__)),
           "package_report": pin(build_dir / "complete.json"), "rerank_param_source": pin(RUN / "rerank_param4_frozen.py"),
           "parent_artifact": pin(PARENT_RUN / "package_v14" / (PARENT_CANDIDATE + ".zip")),
           "parent_acceptance": pin(PARENT_RUN / "runtime_acceptance_v14.json"),
           "parent_verifier": pin(ROOT / "src/upar/b127c_jaccard_prob_diffusion_acceptance.py"),
           "runtime_report": pin(replay_dir / "execute_report.json"),
           "reference_report": pin(replay_dir / "reference_report.json"),
           "official_hardware_runtime_guaranteed": False, "grants_submission_authority": False, "quality_or_gold_claim": False}
    result = verify_evidence(Path(acc["artifact"]["path"]), acc)
    with output.open("x") as stream:
        json.dump(acc, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"acceptance": pin(output), "verification": result}


if __name__ == "__main__":
    import sys
    print(json.dumps(build_acceptance(sys.argv[1], sys.argv[2], sys.argv[3]), indent=1))
