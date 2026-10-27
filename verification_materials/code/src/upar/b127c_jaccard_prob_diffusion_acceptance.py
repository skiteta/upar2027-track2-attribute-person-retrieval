"""Runtime release gate for B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR (and B127C_JACCARD_PROB_DIFFUSION).

Stdlib only (loaded by path by scripts/codabench_submit.py). It proves, from files on disk,
that the exact ZIP to be sent:
  - is the built B128C ZIP (SHA-256 of package_v2/complete.json) whose parent is the accepted
    B124C ZIP (whose own stdlib gate chains to the official B123C ZIP 995164, Finished);
  - differs from B124C only in assets/config.json, b121_runtime.py, and run.py, plus the added
    module b127_jaccard_prob_diffusion.py, with every other member byte-identical;
  - was executed on the local CUDA GPU over the full validation set (33,407 images x 3,462 queries)
    from this exact ZIP;
  - produced FP32 distances that equal, bit for bit, an independent recomputation from the same
    inputs (0 unequal values, 0 queries with different ranks), with every pinned capture file unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"
B124 = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05"
KIND = "b139c_anat_multires_spherical_head_runtime_acceptance_v1"
B124_KIND = "b124c_constant_alpha05_runtime_acceptance_v1"
CANDIDATE = "B139C_ANAT_MULTIRES_SPHERICAL_HEAD_RERANK"
B124_CANDIDATE = "B124C_CONSTANT_RESCUE_ALPHA05"
B124_SHA = "5fae8b2b0d28ce4e313249e5cfb47c6627d0164aaac5a9a54c07288e88c20da7"
PARENT_CANDIDATE = "B123C_CONSTANT_RESCUE_ALPHA03"
PARENT_SUBMISSION = 995164
REFERENCE_KIND = "exact_new_b139c_anat_multires_spherical_head_same_input_CUDA32_graph_independent_reference"
CHANGED = {
    "assets/config.json",
    "assets/model_0.weights.pt",
    "assets/model_1.weights.pt",
    "assets/model_2.weights.pt",
    "b121_runtime.py",
    "run.py",
}
ADDED = "b127_jaccard_prob_diffusion.py"

EXPECTED_B134_CFG = {
    "branches": [
        {"name": "pe", "k": 20, "w_pe": 1.0, "w_eva": 0.0, "w_hp": 0.0, "weight": 0.275},
        {"name": "eva", "k": 20, "w_pe": 0.0, "w_eva": 1.0, "w_hp": 0.0, "weight": 0.230},
        {"name": "j25_k28", "k": 28, "w_pe": 0.75, "w_eva": 0.25, "w_hp": 0.0, "weight": 0.235},
        {"name": "j50_k28", "k": 28, "w_pe": 0.50, "w_eva": 0.50, "w_hp": 0.0, "weight": 0.190},
        {"name": "tri25_k28", "k": 28, "w_pe": 0.5625, "w_eva": 0.1875, "w_hp": 0.25, "weight": 0.070},
    ],
    "cross_cos_alpha": 0.35,
    "cos_c_hp": 0.20,
    "max_hamming": 5,
    "ham_tau": 0.172,
    "margin_gamma": 1.83,
    "jaccard_power": 0.80,
    "prob_diffusion_alpha": 0.95,
    "stage1_steps": 3,
    "stage2_steps": 6,
    "deg_beta": 0.35,
    "tau_w": 5.0,
    "trust_gamma": 0.25,
    "std_trust_gamma": 0.40,
    "coevolve_jac_low": 0.24,
    "coevolve_jac_span": 0.30,
    "coevolve_blend": 0.55,
    "post_tau": 1.05,
    "prob_clip_eps": 1e-5,
    "entropy_delta": 0.35,
    "nll_lam": 0.275,
    "nll_lam_neg": 0.215,
    "err_gamma": 1.0,
    "err_gamma_neg": 0.97,
    "base_dich_sym_loss": 1.28,
    "dich_pos_nll_add": 0.08,
    "dich_attrs": ["Gender-Female", "UpperBody-Length-Short", "LowerBody-Length-Short"],
    "score_ens_blend": 0.45,
    "dist_diffusion_steps": 2,
    "dist_diffusion_alpha": 0.205,
    "dist_kappa": 0.20,
    "exact_ent_delta": 0.45,
    "exact_nll_lam": 0.43,
    "exact_nll_lam_neg": 0.30,
    "exact_err_gamma": 1.25,
    "exact_err_gamma_neg": 1.25,
    "exact_dich_sym_loss": 1.25,
    "exact_branch_blend": 0.22,
    "ex_diff_alpha": 0.07,
    "ex_kappa": 0.24,
    "ex_diff_head_blend": 0.50,
    "exact_seed_blend": 0.72,
    "tail_mul_kap": 0.04,
    "seed_disagree_kap": 0.030,
    "seed_kap_weight": False,
    "seed_coh_beta": 0.25,
    "ms_k1": 6,
    "ms_temp1": 0.035,
    "ms_blend": 0.22,
    "sim_diff_alpha": 0.34,
    "sim_diff_steps": 1,
    "sim_diff_alpha2": 0.04,
    "head_dq_blend": 0.15,
    "ex_vis_alpha": 0.120,
    "head_local_lam": 0.12,
    "head_local_k": 12,
    "head_beta": 0.044,
    "head_tau": 0.225,
    "body_alpha": 0.33,
    "body_k0": 10,
    "body_tau": 0.355,
    "body_pow": 1.0,
    "body_w_uni": True,
    "body_ent_delta": 0.15,
    "body_nll_lam": 0.05,
    "body_nll_lam_neg": 0.00,
    "body_err_gamma": 0.90,
    "body_err_gamma_neg": 0.75,
    "body_pos_q_mul": 0.70,
    "body_dich_sym_loss": 0.0,
    "body_d_alpha": 0.235,
    "body_kap": 0.20,
    "body_vis_scale": 1.35,
    "body_head_keep": 3.5,
    "body_head_tau": 0.55,
    "anchor_spectra": [
        {"name": "pe", "w_pe": 1.0, "w_eva": 0.0, "w_hp": 0.0, "weight": 0.10},
        {"name": "eva", "w_pe": 0.0, "w_eva": 1.0, "w_hp": 0.0, "weight": 0.10},
        {"name": "j25", "w_pe": 0.75, "w_eva": 0.25, "w_hp": 0.0, "weight": 0.18},
        {"name": "j50", "w_pe": 0.50, "w_eva": 0.50, "w_hp": 0.0, "weight": 0.18},
        {"name": "tri25", "w_pe": 0.5625, "w_eva": 0.1875, "w_hp": 0.25, "weight": 0.16},
        {"name": "tri40", "w_pe": 0.40, "w_eva": 0.30, "w_hp": 0.30, "weight": 0.14},
        {"name": "pe_hp", "w_pe": 0.65, "w_eva": 0.0, "w_hp": 0.35, "weight": 0.07},
        {"name": "eva_hp", "w_pe": 0.0, "w_eva": 0.65, "w_hp": 0.35, "weight": 0.07},
    ],
    "anchor_top_m": 11,
    "anchor_alpha_q": 0.126,
    "anchor_temp_q": 0.052,
    "anchor_gamma_q": 2.6,
    "anchor_hub_norm": 0.35,
    "symmetric_zero_fit": True,
}


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


def build_anat10_member_weights():
    w_pe, w_hp, w_eva = [], [], []
    for j in range(40):
        if j in (4, 5, 6):
            pe, hp, eva = 0.425, 0.340, 0.235
        elif j in (0, 1, 2, 3, 37, 38, 39):
            pe, hp, eva = 0.420, 0.345, 0.235
        elif j in (7, 20, 36):
            pe, hp, eva = 0.355, 0.410, 0.235
        else:
            pe, hp, eva = 0.370, 0.395, 0.235
        w_pe.append(round(pe, 6))
        w_hp.append(round(hp, 6))
        w_eva.append(round(eva, 6))
    return [w_pe, w_hp, w_eva]


def verify_evidence(artifact, acc):
    if (
        acc.get("kind") != KIND
        or acc.get("status") != "ACCEPTED_RUNTIME_PREFLIGHT"
        or acc.get("candidate") != CANDIDATE
        or acc.get("grants_submission_authority") is not False
    ):
        raise ValueError("dedicated B139C acceptance with explicit limits required")
    artifact = Path(artifact).resolve()
    if pinned(acc["artifact"]).resolve() != artifact:
        raise ValueError("acceptance names another artifact")
    package = json.loads(pinned(acc["package_report"]).read_text())
    if (
        package.get("status") != "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED"
        or package.get("candidate") != CANDIDATE
        or package.get("b139_anat_multires_spherical_head_rerank") != EXPECTED_B134_CFG
        or package["zip"]["sha256"] != acc["artifact"]["sha256"]
        or package["parent"]["sha256"] != B124_SHA
    ):
        raise ValueError("built B139C package report required")
    parent_complete = json.loads(pinned(acc["parent_complete"]).read_text())
    parent_acc = json.loads(pinned(acc["parent_acceptance"]).read_text())
    if (
        parent_complete.get("candidate") != B124_CANDIDATE
        or parent_complete.get("zip", {}).get("sha256") != B124_SHA
        or parent_acc.get("kind") != B124_KIND
        or parent_acc.get("artifact", {}).get("sha256") != B124_SHA
    ):
        raise ValueError("accepted B124C parent complete/acceptance required")
    a = {m["member"]: m["new_sha256"] for m in parent_complete["member_proof"]}
    parent_pkg_dir = B124 / "package_v1/package"
    for m_name, m_sha in a.items():
        if sha(parent_pkg_dir / m_name) != m_sha:
            raise ValueError("parent extracted member modified: " + m_name)
    old_cfg = json.loads((parent_pkg_dir / "assets/config.json").read_text())
    with zipfile.ZipFile(artifact) as new:
        b = member_digests(new)
        if set(b) != set(a) | {ADDED}:
            raise ValueError("member set must be the B124C set plus " + ADDED)
        changed = {n for n in a if a[n] != b[n]}
        if changed != CHANGED:
            raise ValueError("unexpected changed members: " + str(sorted(changed)))
        new_cfg = json.loads(new.read("assets/config.json"))
        module = new.read(ADDED)
    if hashlib.sha256(module).hexdigest() != package["module"]["sha256"]:
        raise ValueError("packaged B136C module differs from the built module")
    traced_dir = RUN / "traced_b138c_dich3_ltype2_r00"
    if b["assets/model_0.weights.pt"] != sha(traced_dir / "assets/model_0.weights.pt"):
        raise ValueError("model_0.weights.pt hash mismatch")
    if b["assets/model_1.weights.pt"] != sha(traced_dir / "assets/model_1.weights.pt"):
        raise ValueError("model_1.weights.pt hash mismatch")
    if b["assets/model_2.weights.pt"] != sha(traced_dir / "assets/model_2.weights.pt"):
        raise ValueError("model_2.weights.pt hash mismatch")
    canary_ref_0 = json.loads((traced_dir / "canary_ref_0.json").read_text())
    canary_ref_1 = json.loads((traced_dir / "canary_ref_1.json").read_text())
    canary_ref_2 = json.loads((traced_dir / "canary_ref_2.json").read_text())
    w_anat10 = build_anat10_member_weights()
    expected_cfg = json.loads(json.dumps(old_cfg))
    expected_cfg["candidate"] = CANDIDATE
    expected_cfg["post"]["entropy_delta"] = 0.35
    expected_cfg["models"][0]["training_run"] = "artifacts/runs/20261015T0410+0900_b116g_pe/soup_pe4q_all_s7_spherical_qk120_b060"
    expected_cfg["models"][1]["training_run"] = "artifacts/runs/20261017T0345+0900_hplus6/hp_dich3_ltype2_r00_body12_m15_r10_all_spherical_o103"
    expected_cfg["models"][2]["training_run"] = "artifacts/runs/20261014T0100+0900_b113g/soup_eva224s7_eva252s7_spherical_t12"
    for m_idx in range(3):
        expected_cfg["models"][m_idx]["member_weights"] = w_anat10[m_idx]
    expected_cfg["canary"]["ref"][0] = canary_ref_0
    expected_cfg["canary"]["ref"][1] = canary_ref_1
    expected_cfg["canary"]["ref"][2] = canary_ref_2
    expected_cfg["b139_anat_multires_spherical_head_rerank"] = EXPECTED_B134_CFG
    if old_cfg.get("candidate") != B124_CANDIDATE or new_cfg != expected_cfg or package.get("config") != new_cfg:
        raise ValueError("config mismatch vs expected B139C config")
    runtime = json.loads(pinned(acc["runtime_report"]).read_text())
    if (
        runtime.get("candidate") != CANDIDATE
        or runtime.get("artifact", {}).get("sha256") != acc["artifact"]["sha256"]
        or runtime.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING"
        or runtime.get("device") != "cuda"
        or runtime.get("forced_plan") is not None
        or (runtime.get("rows"), runtime.get("queries")) != (33407, 3462)
        or runtime.get("config") != new_cfg
    ):
        raise ValueError("full CUDA replay of this exact ZIP required")
    reference = json.loads(pinned(acc["reference_report"]).read_text())
    if (
        reference.get("status") != "PASS"
        or reference.get("kind") != REFERENCE_KIND
        or reference.get("candidate") != CANDIDATE
        or reference.get("b139_anat_multires_spherical_head_rerank") != EXPECTED_B134_CFG
        or reference.get("unequal_distance_values") != 0
        or reference.get("queries_with_different_ranks") != 0
    ):
        raise ValueError("bit-exact independent reference PASS required")
    for name in ("execute_report.json", "extra_capture.json"):
        pinned(reference["capture"][name])
    for name in ("capture.npz", "features.npy", "eva_features.npy", "hp_features.npy", "member_std.npy", "distances.npy"):
        if Path(reference["capture"][name]["path"]).exists():
            pinned(reference["capture"][name])
    if reference["capture"]["execute_report.json"]["sha256"] != acc["runtime_report"]["sha256"]:
        raise ValueError("reference was computed on another capture")
    return {
        "kind": KIND,
        "candidate": CANDIDATE,
        "artifact_sha256": acc["artifact"]["sha256"],
        "b139_anat_multires_spherical_head_rerank": EXPECTED_B134_CFG,
        "parent": {"candidate": B124_CANDIDATE, "sha256": B124_SHA},
        "grandparent_official": {"candidate": PARENT_CANDIDATE, "submission": PARENT_SUBMISSION},
        "changed_members": sorted(CHANGED),
        "added_members": [ADDED],
        "full_cuda_replay": True,
        "bit_exact_reference": True,
        "local_time_scale": runtime.get("time_scale"),
        "official_runtime": "unknown; PE4(qk120_b060)+HP_Dich3_LType2_r00_Body12_m15_r10(o103)+EVA2(t12) Column-Selective Spherical Heads + Anat11 MoE (c36->[7,20,36]) + PentJ rerank (~252s A10, <5.1GB VRAM), not an A10 guarantee",
        "quality_or_gold_claim": False,
        "submission_authority": False,
    }


def verify_release(artifact_path, acceptance_path, **_ignored):
    return verify_evidence(Path(artifact_path).resolve(), json.loads(Path(acceptance_path).read_text()))


def build_acceptance(output):
    output = Path(output).resolve()
    if output.parent != RUN or output.exists():
        raise ValueError("new B139C acceptance inside its run dir only; never overwrite")
    acc = {
        "kind": KIND,
        "status": "ACCEPTED_RUNTIME_PREFLIGHT",
        "candidate": CANDIDATE,
        "b139_anat_multires_spherical_head_rerank": EXPECTED_B134_CFG,
        "artifact": pin(RUN / "package_v14" / (CANDIDATE + ".zip")),
        "verifier_source": pin(Path(__file__)),
        "package_report": pin(RUN / "package_v14/complete.json"),
        "parent_complete": pin(B124 / "package_v1/complete.json"),
        "parent_acceptance": pin(B124 / "runtime_acceptance_v1.json"),
        "parent_verifier": pin(ROOT / "src/upar/b124c_alpha05_acceptance.py"),
        "runtime_report": pin(RUN / "b139c_cuda_capture_v1/execute_report.json"),
        "reference_report": pin(RUN / "b139c_reference_v1/report.json"),
        "official_hardware_runtime_guaranteed": False,
        "grants_submission_authority": False,
        "quality_or_gold_claim": False,
    }
    result = verify_evidence(Path(acc["artifact"]["path"]), acc)
    with output.open("x") as stream:
        json.dump(acc, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"acceptance": pin(output), "verification": result}


if __name__ == "__main__":
    import sys

    print(json.dumps(build_acceptance(sys.argv[1]), indent=1))


