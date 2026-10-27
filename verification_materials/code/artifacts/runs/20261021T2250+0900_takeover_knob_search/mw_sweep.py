"""Member-weight re-selection for the B144C family (B139C heads + B136C rerank + near-dup diffusion), chosen on the val OOF
surface's >=15-match queries and confirmed on the independent train surface (the calibrated MEVID proxy).
usage: mw_sweep.py SURFACE(val|train) OUT_JSONL
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
S = ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle"
surface = sys.argv[1]
if surface == "train":
    os.environ["KNOB_INPUTS"] = str(HERE / "train_surface/inputs")
sys.path.insert(0, str(HERE))
import knob_eval as ke  # noqa: E402
from upar.postproc import apply_post  # noqa: E402
from upar.rank_sibling_pipeline import read_json  # noqa: E402
from upar.source_fitted_gallery_edges import SPEC  # noqa: E402

POST = read_json(SPEC)["lodo"]["post"]
BODY_12 = [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
DICH_3 = [3, 7, 20]
LTYPE_2 = [33, 34]
K = json.loads(os.environ["MW_K"]) if os.environ.get("MW_K") else {"ham_tau": 0.17, "margin": 1.85, "nd": {"q": 0.85, "s": 0.01, "steps": 8}}


def blend_cols(base, pairs):
    out = base.copy()
    for cols, src in pairs:
        out[:, cols] = src[:, cols]
    return out


def members_all():
    if surface == "train":
        z = np.load(HERE / "train_surface/inputs/members_raw.npz")
        d = {k: z[k] for k in z.files}
        d["hp_ltype2"] = blend_cols(blend_cols(z["hp_b137c"], [(BODY_12, z["hp_mask15_m15_r10_o103"])]),
                                    [(DICH_3, z["hp_b136c"]), (LTYPE_2, z["hp_mask15_m15_r00_o103"])])
        ze = np.load(HERE / "train_surface/inputs/members_raw_extra.npz")
        d.update({k: ze[k] for k in ze.files})
        return d
    z = np.load(HERE / "inputs_v1/members_all_val.npz")
    d = {k: z[k] for k in z.files}
    ze = np.load(HERE / "inputs_v1/members_extra_val.npz")
    d.update({k: ze[k] for k in ze.files})
    return d


def members():
    if surface == "train":
        z = np.load(HERE / "train_surface/inputs/members_raw.npz")
        hp = blend_cols(blend_cols(z["hp_b137c"], [(BODY_12, z["hp_mask15_m15_r10_o103"])]),
                        [(DICH_3, z["hp_b136c"]), (LTYPE_2, z["hp_mask15_m15_r00_o103"])])
        return [z["pe_b137c"], hp, z["eva_t12"]]
    z = np.load(HERE / "inputs_v1/b139c_members_val.npz")
    return [z["pe"], z["hp"], z["eva"]]


def anat10():
    w = np.zeros((3, 40))
    for j in range(40):
        if j in (4, 5, 6):
            w[:, j] = (0.425, 0.340, 0.235)
        elif j in (0, 1, 2, 3, 36, 37, 38, 39):
            w[:, j] = (0.420, 0.345, 0.235)
        elif j in (7, 20):
            w[:, j] = (0.355, 0.410, 0.235)
        else:
            w[:, j] = (0.370, 0.395, 0.235)
    w[:, 36] = (0.355, 0.410, 0.235)
    return w


def pool(raw, w):
    st = np.stack([r.astype(np.float64) for r in raw])
    ww = w[:, None, :] / w.sum(axis=0)[None, None, :]
    p = apply_post((ww * st).sum(axis=0).astype(np.float32), POST)
    mstd = np.std(np.stack([apply_post(r, POST) for r in raw]), axis=0).astype(np.float32)
    return p, mstd


def variants():
    base = anat10()
    out = {"base_w_c36_hp41": base}
    groups = {"all": list(range(40)), "colors": list(range(8, 20)) + list(range(21, 33)), "head": [0, 1, 2, 3, 4, 5, 6],
              "type": [7, 20, 33, 34], "acc": [35, 36, 37, 38, 39]}
    for gname, cols in groups.items():
        for src, dst in ((0, 1), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2)):
            w = base.copy()
            w[dst, cols] += 0.05
            w[src, cols] -= 0.05
            out[f"{gname}_m{src}to{dst}"] = w
    if os.environ.get("MW_SET") == "v3":
        w = base.copy(); w[0, :] -= 0.05; w[2, :] -= 0.05; w[1, :] += 0.10
        return {"base_w_c36_hp41": base, "hp_p10_pe05_eva05": w}
    if os.environ.get("MW_SET") == "v2":  # follow-up: combine the two confirmed HP-ward shifts and scale them
        keep = {"base_w_c36_hp41": base}
        for name, dpe, deva in (("hp_p10_pe05_eva05", -0.05, -0.05), ("hp_p10_pe10", -0.10, 0.0), ("hp_p075_pe05_eva025", -0.05, -0.025),
                                ("hp_p15_pe10_eva05", -0.10, -0.05)):
            w = base.copy()
            w[0, :] += dpe
            w[2, :] += deva
            w[1, :] -= dpe + deva
            keep[name] = w
        return keep
    return out


def main() -> None:
    out_path = Path(sys.argv[2])
    sh = {k: v for k, v in np.load(ke.INP / "shared.npz", allow_pickle=False).items()}
    names = [str(x) for x in sh["attribute_names"]]
    raw = members()
    if os.environ.get("MW_SET") == "heads2":  # extra head variants (B145C weights), PE x HP grid with eva_t12
        allm = members_all()
        hp10 = anat10(); hp10[0, :] -= 0.05; hp10[2, :] -= 0.05; hp10[1, :] += 0.10
        combos = {}
        for pe in ("pe_b137c", "pe_b136c", "pe_fix19_qk120_b060", "pe_fix_pos5h_qk120_b060", "pe_fix_19_lb_qk120_b060",
                   "pe_unmask24_m20_qk120_b060", "pe_qk115_o100_b050"):
            for hp in ("hp_ltype2", "hp_mask15_m12_r10_o103", "hp7_fix19_m10_r10_o103", "hp7_fix19_m12_r10_o103",
                       "hp5_fix19_m12_o103", "hp6_15_qk110_o103_b075"):
                combos[f"{pe}|{hp}|eva_t12|hp10"] = ([allm[pe], allm[hp], allm["eva_t12"]], hp10)
        shard = os.environ.get("MW_SHARD", "0/1").split("/")
        combos = {k: v for j, (k, v) in enumerate(combos.items()) if j % int(shard[1]) == int(shard[0])}
    if os.environ.get("MW_SET") == "heads":  # head-set x member-weight combinations
        allm = members_all()
        base = anat10(); hp10 = base.copy(); hp10[0, :] -= 0.05; hp10[2, :] -= 0.05; hp10[1, :] += 0.10
        combos = {}
        for pe in ("pe_b136c", "pe_b137c"):
            for hp in ("hp_b136c", "hp_b137c", "hp_ltype2"):
                for ev in ("eva_soup_raw", "eva_t12"):
                    for wn, ww in (("w0", base), ("hp10", hp10)):
                        combos[f"{pe}|{hp}|{ev}|{wn}"] = ([allm[pe], allm[hp], allm[ev]], ww)
    if surface == "train":
        nm, gq = sh["oof_norm_match"], sh["oof_g_qids"].astype(np.int64)
    else:
        nm = np.load(S / "extracted_features_v1/graph_cache_v1/eval_ctx_3462_33407_489e9e2af5d709bd.npy")
        from upar.data import load_retrieval_split
        gq = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val").gallery_query_ids.astype(np.int64)
    cnt = np.bincount(gq, minlength=int(max(gq.max(), sh["qi_oof"].max())) + 1)[sh["qi_oof"]]
    done = {json.loads(l)["name"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    pq = out_path.with_suffix("")
    pq.mkdir(parents=True, exist_ok=True)
    items = combos.items() if os.environ.get("MW_SET") in ("heads", "heads2") else ((n, (raw, w)) for n, w in variants().items())
    for name, (rr, w) in items:
        if name in done:
            continue
        p, mstd = pool(rr, w)
        d = ke.PAR.run_b132c_pentj_body_rerank(p, sh["u_pe"], sh["u_eva"], sh["u_hp"], mstd, sh["queries_oof"], names,
                                               ke.make_score_fn(sh["w_oof"]), lambda *a, **k: None, K=dict(K))
        adm = ke.eval_q(d, sh["qi_oof"], nm, gq)["adm"]
        np.savez(pq / f"{name}.npz", oof=adm)
        row = {"name": name, "oof": float(adm.mean()), "hi15": float(adm[cnt >= 15].mean()), "hi40": float(adm[cnt >= 40].mean()),
               "lo1_4": float(adm[cnt <= 4].mean())}
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
