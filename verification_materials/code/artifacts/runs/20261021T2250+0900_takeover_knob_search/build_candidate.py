"""Build a rerank-only candidate from an exact, Public-scored parent ZIP (default: B136C, submission 1005359).

The new ZIP equals the parent byte-for-byte except
  - b127_jaccard_prob_diffusion.py : rerank_param.py (generated from the deployed package_v14 module; its default path
                                     is bit-exact with the deployed code) + one DEFAULT_K.update({...}) line holding the
                                     chosen knobs (merged exactly like knob_eval.py sweep merges them)
  - assets/config.json             : candidate name + descriptive "rerank_knobs" block (runtime ignores it)
usage: build_candidate.py SPEC.json OUT_DIR
SPEC = {"candidate": "...", "parent_zip": "...", "parent_sha256": "...", "K": {...}}
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merged_k(K: dict, defaults: dict) -> dict:
    K = json.loads(json.dumps(K))
    for key in ("ks", "anchor_w"):
        if key in K:
            K[key] = {**defaults[key], **K[key]}
    if "branch_specs" in K:
        K["branch_specs"] = [tuple(b) for b in K["branch_specs"]]
    if "extra_graphs" in K:
        K["extra_graphs"] = [tuple(b) for b in K["extra_graphs"]]
    return K


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]).resolve()
    assert out.parent == HERE and not out.exists(), "fresh output inside the run dir only"
    parent = Path(spec["parent_zip"]).resolve()
    assert sha(parent) == spec["parent_sha256"], "exact parent ZIP required"
    t0 = time.monotonic()
    base_name = spec.get("module_base", "rerank_param.py")
    par_spec = importlib.util.spec_from_file_location("rp_defaults", HERE / base_name)
    rp = importlib.util.module_from_spec(par_spec)
    par_spec.loader.exec_module(rp)
    K = merged_k(spec["K"], rp.DEFAULT_K)
    module_src = (HERE / base_name).read_text()
    module_src += ("\n\n# Candidate " + spec["candidate"] + ": rerank knobs chosen by the takeover knob search "
                   "(knob_eval.py / compare.py evidence in this run dir)\nDEFAULT_K.update(" + repr(K) + ")\n")
    if spec.get("blend"):  # optional per-query gated blend wrapper (fixed text, pinned by sha in the report)
        module_src += "\nBLEND = " + repr(spec["blend"]) + "\n" + (HERE / "blend_wrapper.py.txt").read_text()
    package = out / "package"
    package.mkdir(parents=True)
    with zipfile.ZipFile(parent) as z:
        names = sorted(i.filename for i in z.infolist() if not i.is_dir())
        parent_hashes = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names}
        for n in names:
            dest = package / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(n)
            if n == "b127_jaccard_prob_diffusion.py":
                data = module_src.encode()
            elif n == "assets/config.json":
                cfg = json.loads(data)
                cfg["candidate"] = spec["candidate"]
                cfg["rerank_knobs"] = {"source": "takeover knob search 2026-10-21", "K_overrides": spec["K"],
                                       "parent_zip_sha256": spec["parent_sha256"]}
                if spec.get("blend"):
                    cfg["rerank_knobs"]["blend"] = spec["blend"]
                if spec.get("attr_wmul"):  # optional per-group multipliers of attribute_distance_weights (recorded in the report)
                    agroups = {"age": [0, 1, 2], "gender": [3], "hair": [4, 5, 6], "ub_len": [7], "ub_color": list(range(8, 20)),
                               "lb_len": [20], "lb_color": list(range(21, 33)), "lb_type": [33, 34], "acc": [35, 36, 37, 38, 39],
                               "acc_backpack": [35], "acc_bag": [36], "acc_glasses": [37, 38], "acc_hat": [39]}
                    aw = [float(x) for x in cfg["attribute_distance_weights"]]
                    for g, mult in spec["attr_wmul"].items():
                        for j in (agroups[g] if g in agroups else [int(g[1:])]):  # 'a<idx>' = one attribute
                            aw[j] = aw[j] * float(mult)
                    cfg["attribute_distance_weights"] = aw
                    cfg["rerank_knobs"]["attr_wmul"] = spec["attr_wmul"]
                if spec.get("post_renorm") is not None:  # optional replacement of post.renorm (exclusive-group renormalisation)
                    cfg["post"]["renorm"] = [[int(a), int(b), float(al)] for a, b, al in spec["post_renorm"]]
                    cfg["rerank_knobs"]["post_renorm"] = spec["post_renorm"]
                if spec.get("member_weights"):  # optional per-attribute member weights (3 x 40), recorded in the report
                    assert len(spec["member_weights"]) == len(cfg["models"]) == 3
                    for m_idx, mw in enumerate(spec["member_weights"]):
                        assert len(mw) == 40
                        cfg["models"][m_idx]["member_weights"] = [round(float(x), 6) for x in mw]
                    cfg["rerank_knobs"]["member_weights_changed"] = True
                data = (json.dumps(cfg, indent=2) + "\n").encode()
            dest.write_bytes(data)
    artifact = out / (spec["candidate"] + ".zip")
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for p in sorted(package.rglob("*")):
            if p.is_file():
                new.write(p, p.relative_to(package).as_posix())
    proof = []
    with zipfile.ZipFile(artifact) as z:
        new_names = sorted(i.filename for i in z.infolist() if not i.is_dir())
        assert new_names == names, "member set must equal the parent's"
        for n in new_names:
            after = hashlib.sha256(z.read(n)).hexdigest()
            changed = after != parent_hashes[n]
            assert changed == (n in ("b127_jaccard_prob_diffusion.py", "assets/config.json")), "unexpected delta " + n
            proof.append({"member": n, "parent_sha256": parent_hashes[n], "new_sha256": after, "changed": changed})
    cfg = json.loads((package / "assets/config.json").read_text())
    report = {"status": "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED", "candidate": spec["candidate"],
              "zip": {"path": str(artifact), "sha256": sha(artifact), "size": artifact.stat().st_size},
              "parent": {"path": str(parent), "sha256": spec["parent_sha256"]}, "K_overrides": spec["K"],
              "K_merged_repr": repr(K), "blend": spec.get("blend"), "member_weights": spec.get("member_weights"), "attr_wmul": spec.get("attr_wmul"), "post_renorm": spec.get("post_renorm"), "member_proof": proof, "config": cfg,
              "module": {"source": str(HERE / base_name), "sha256": hashlib.sha256(module_src.encode()).hexdigest()},
              "wall_seconds": time.monotonic() - t0}
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "candidate": spec["candidate"], "zip_sha256": report["zip"]["sha256"],
                      "changed": [m["member"] for m in proof if m["changed"]]}), flush=True)


if __name__ == "__main__":
    main()
