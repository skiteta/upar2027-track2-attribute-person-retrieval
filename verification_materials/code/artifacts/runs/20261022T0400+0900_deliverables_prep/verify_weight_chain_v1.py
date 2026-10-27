"""Read-only CPU check: can the three int8 member weight files of B150C be re-derived from the fp32 inputs that are
still on disk, using the recipes of the soup / pack scripts that produced them?

Writes one JSON report (argv[1]); never modifies any input. Re-implements, verbatim in arithmetic:
  - upar.h1_finetune.soup              (member 0 and member 2 backbone soups, fp32 weighted sum)
  - scripts/make_xres_member.py        (EVA02 224 -> 252 pos_embed resample, member 2)
  - scripts/make_pw_aligned.py         (head-bias shift of the label-mask-v2 PE run; head only)
  - build_and_extract_hp4_20_30_30_20.build_weighted_soup   (member 1 backbone, 0.2/0.3/0.3/0.2)
  - pack_hp4_2030_192_cpu.py / h1_package._int8_store       (per-row int8 + fp16)
  - pack_b137c_tri_spherical_heads_cpu.make_spherical_head_dict   (member 0 and member 2 heads)
  - pack_b138c_colselective_heads_cpu.make_colselective_head_dict (member 1 head, mode dich3_ltype2_r00)
usage: .venv-train/bin/python -I verify_weight_chain_v1.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(4)
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
RUNS = ROOT / "artifacts/runs"
KS = RUNS / "20261008T231500+0900_knob_sweep"
PKG = RUNS / "20261021T2250+0900_takeover_knob_search/build_b150c_v1/package/assets"
B124 = RUNS / "20261017T1615+0900_b124_alpha05/package_v1/package/assets"
B127 = RUNS / "20261017T2145+0900_b127_jaccard_prob_diffusion"
REPORT: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def ema(name: str) -> Path:
    return KS / f"{name}_all_trainval_seed20261007_ema0.999/model.safetensors"


def pack(val: torch.Tensor):
    val = val.detach().float()
    if val.ndim == 2 and val.numel() >= 4096:
        scale = val.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        return {"q": torch.round(val / scale[:, None]).clamp(-127, 127).to(torch.int8), "s": scale.half()}
    return val.half()


def deq(v) -> torch.Tensor:
    return v["q"].float() * v["s"].float()[:, None] if isinstance(v, dict) else v.float()


def cmp(a, b) -> dict:
    """exact = identical stored bytes (int8 codes and fp16 scales, or fp16 values); else the int8-code / value deltas."""
    if isinstance(a, dict) != isinstance(b, dict):
        return {"exact": False, "note": "storage kind differs"}
    if isinstance(a, dict):
        if a["q"].shape != b["q"].shape:
            return {"exact": False, "note": f"shape {tuple(a['q'].shape)} vs {tuple(b['q'].shape)}"}
        dq = (a["q"].int() - b["q"].int()).abs()
        ex = bool(torch.equal(a["q"], b["q"]) and torch.equal(a["s"], b["s"]))
        return {"exact": ex, "codes_differ": int((dq > 0).sum()), "codes_total": int(dq.numel()), "max_code_delta": int(dq.max()),
                "scales_differ": int((a["s"] != b["s"]).sum()), "max_abs_dequant": float((deq(a) - deq(b)).abs().max())}
    if a.shape != b.shape:
        return {"exact": False, "note": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    return {"exact": bool(torch.equal(a, b)), "values_differ": int((a != b).sum()), "max_abs": float((a.float() - b.float()).abs().max())}


def summarize(rows: dict) -> dict:
    exact = [k for k, r in rows.items() if r.get("exact")]
    bad = {k: r for k, r in rows.items() if not r.get("exact")}
    out = {"compared": len(rows), "exact": len(exact), "not_exact": len(bad)}
    if bad:
        codes = sum(r.get("codes_differ", 0) for r in bad.values())
        total = sum(r.get("codes_total", 0) for r in bad.values())
        out["not_exact_keys"] = {k: r for k, r in list(bad.items())[:12]}
        out["codes_differ_total"] = codes
        out["codes_total_in_not_exact"] = total
        out["max_code_delta"] = max((r.get("max_code_delta", 0) for r in bad.values()), default=0)
        out["max_abs_dequant"] = max((r.get("max_abs_dequant", r.get("max_abs", 0.0)) for r in bad.values()), default=0.0)
    return out


def load_pt(p: Path) -> dict:
    return torch.load(str(p), map_location="cpu", mmap=True)


def heads(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith("head.")}


def diff_keys(a: dict, b: dict) -> list[str]:
    assert set(a) == set(b), (sorted(set(a) ^ set(b)))[:10]
    return sorted(k for k in a if not (cmp(a[k], b[k]).get("exact")))


def spherical(mems: dict, weights: dict, tau_qk=1.0, tau_out=1.0, tau_base=1.0, q_scale=1.0, out_scale=1.0, base_scale=1.0):
    """pack_b137c_tri_spherical_heads_cpu.make_spherical_head_dict (verbatim arithmetic)."""
    head_keys = [k for k in next(iter(mems.values())).keys() if k.startswith("head.")]
    out = {}
    for k in head_keys:
        soup_t = sum(w * mems[m][k].float() for m, w in weights.items())
        if soup_t.ndim == 2:
            t_exp = 0.0
            if k in ("head.q", "head.k.weight"):
                t_exp = tau_qk
            elif k == "head.out":
                t_exp = tau_out
            elif k in ("head.base.weight", "head.weight"):
                t_exp = tau_base
            if t_exp != 0.0:
                mean_norm = sum(w * mems[m][k].float().norm(dim=-1, keepdim=True) for m, w in weights.items())
                soup_norm = soup_t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                soup_t = soup_t * ((mean_norm / soup_norm) ** t_exp)
        if k == "head.q" and q_scale != 1.0:
            soup_t = soup_t * q_scale
        elif k == "head.out" and out_scale != 1.0:
            soup_t = soup_t * out_scale
        elif k in ("head.base.weight", "head.weight") and base_scale != 1.0:
            soup_t = soup_t * base_scale
        out[k] = soup_t
    return out


def colselective(member_sds, head_weights, col_overrides, out_scale):
    """pack_b138c_colselective_heads_cpu.make_colselective_head_dict with tau_* = 1, q/base scale 1 (verbatim)."""
    out = {}
    for k, tau, sc in [("head.q", 1.0, 1.0), ("head.k.weight", 1.0, 1.0), ("head.out", 1.0, out_scale),
                       ("head.base.weight", 1.0, 1.0), ("head.base.bias", 0.0, 1.0)]:
        base_t = sum(float(w) * member_sds[m][k].float() for m, w in head_weights.items() if w > 0).clone()
        if tau != 0.0 and base_t.ndim == 2:
            soup_norm = base_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mem_norm = sum(float(w) * member_sds[m][k].float().norm(dim=-1, keepdim=True) for m, w in head_weights.items() if w > 0)
            base_t = base_t * ((mem_norm / soup_norm) ** float(tau))
        if col_overrides and k != "head.k.weight":
            for j, w_col in col_overrides.items():
                row_t = sum(float(w) * member_sds[m][k][j].float() for m, w in w_col.items() if w > 0)
                if tau != 0.0 and base_t.ndim == 2:
                    s_norm = row_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    m_norm = sum(float(w) * member_sds[m][k][j].float().norm(dim=-1, keepdim=True) for m, w in w_col.items() if w > 0)
                    row_t = row_t * ((m_norm / s_norm) ** float(tau))
                base_t[j] = row_t
        if sc != 1.0:
            base_t = base_t * float(sc)
        out[k] = base_t
    return out


def check_heads(stored: dict, head: dict) -> dict:
    rows = {}
    for k, v in head.items():
        if k in stored:
            rows[k] = cmp(stored[k], pack(v))
    rows["head.weight<-head.base.weight" if "head.base.weight" in head else "head.weight"] = cmp(
        stored["head.weight"], pack(head.get("head.base.weight", head.get("head.weight"))))
    rows["head.bias<-head.base.bias" if "head.base.bias" in head else "head.bias"] = cmp(
        stored["head.bias"], pack(head.get("head.base.bias", head.get("head.bias"))))
    return rows


def soup_check(stored: dict, sources: list[tuple[Path, float]], keys: list[str], transform=None, mode="h1") -> dict:
    """Recompute selected backbone keys of a soup from its fp32 EMA inputs and compare after int8/fp16 packing."""
    handles = [safe_open(str(p), "pt") for p, _ in sources]
    avail = set(handles[0].keys())
    rows, skipped = {}, []
    for k in keys:
        if k not in avail:
            skipped.append(k)
            continue
        ts = [h.get_tensor(k) for h in handles]
        if transform:
            ts = [transform(k, i, t) for i, t in enumerate(ts)]
        if mode == "h1":  # upar.h1_finetune.soup: Python sum of float(w) * fp32 tensor
            val = sum(float(w) * t.float() for (_, w), t in zip(sources, ts))
        else:  # build_weighted_soup: fp32 accumulator, in place, cast back to the source dtype
            val = torch.zeros_like(ts[0], dtype=torch.float32)
            for (_, w), t in zip(sources, ts):
                val += w * t.float()
            val = val.to(ts[0].dtype)
        rows[k] = cmp(stored[k], pack(val))
    return {"summary": summarize(rows), "skipped_not_in_safetensors": skipped}


def main() -> None:
    out_path = Path(sys.argv[1])
    t0 = time.monotonic()
    w = {i: load_pt(PKG / f"model_{i}.weights.pt") for i in range(3)}
    b124 = {i: load_pt(B124 / f"model_{i}.weights.pt") for i in (0, 2)}

    # ---------------- member 0: PE-Core-L/14 280
    m0: dict = {}
    m0["keys_changed_vs_B124C_model_0"] = diff_keys(w[0], b124[0])
    pw_ref = np.load(KS / "peL280_attrq_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    pw_mask = np.load(KS / "pLq_maskv2_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    bias_delta = torch.from_numpy((np.log(pw_ref) - np.log(pw_mask)).astype(np.float32))
    h_mask = heads(ema("pLq_maskv2"))
    for bk in ("head.base.bias", "head.bias"):
        if bk in h_mask:
            h_mask[bk] = (h_mask[bk].float() + bias_delta).to(h_mask[bk].dtype).contiguous()
    pe_mems = {"attrq": heads(ema("peL280_attrq")), "rankq": heads(ema("pLq_rankq")), "maskv2": h_mask, "pswap": heads(ema("pLq_pswap03"))}
    pe_head = spherical(pe_mems, {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}, tau_qk=1.20, tau_out=1.00, tau_base=0.60)
    m0["head_recompute_pack_b137c"] = check_heads(w[0], pe_head)
    bb0 = [k for k in b124[0] if not k.startswith("head.") and k != "backbone.rope.pos_embed"]
    pe_src = [(ema("peL280_attrq"), 0.25), (ema("pLq_rankq"), 0.25), (ema("pLq_maskv2"), 0.25), (ema("pLq_pswap03"), 0.25)]
    m0["backbone_soup_recompute_vs_B124C_all_keys"] = soup_check(b124[0], pe_src, bb0, mode="h1")
    m0["note"] = ("pLq_maskv2 (knob_sweep EMA) is used for the backbone in place of the deleted pw-aligned copy "
                  "b115g/tmp_maskv2_all_s7_pw: make_pw_aligned.py only shifts head.base.bias/head.bias")
    REPORT["member_0"] = m0
    print("member 0 done", round(time.monotonic() - t0, 1), flush=True)

    # ---------------- member 2: EVA02-L/14 252
    m2: dict = {}
    m2["keys_changed_vs_B124C_model_2"] = diff_keys(w[2], b124[2])
    eva_head = spherical({"eva224": heads(ema("eva02L224")), "eva252": heads(ema("eva02L252"))}, {"eva224": 0.50, "eva252": 0.50},
                         tau_qk=0.0, tau_out=0.0, tau_base=1.20)
    m2["head_recompute_pack_b137c"] = check_heads(w[2], eva_head)
    from timm.layers import resample_abs_pos_embed

    def xres(k, i, t):
        if i == 0 and k == "backbone.pos_embed":  # make_xres_member.py: 224x112 (16x8) -> 252x126 (18x9)
            return resample_abs_pos_embed(t.float(), new_size=[18, 9], old_size=[16, 8], num_prefix_tokens=1,
                                          interpolation="bicubic", antialias=True).to(t.dtype).contiguous()
        return t
    bb2 = [k for k in b124[2] if not k.startswith("head.") and k != "backbone.rope.pos_embed" and ".mlp.fc1." not in k]
    eva_src = [(ema("eva02L224"), 0.5), (ema("eva02L252"), 0.5)]
    m2["backbone_soup_recompute_vs_B124C_all_unfused_keys"] = soup_check(b124[2], eva_src, bb2, transform=xres, mode="h1")
    # fused SwiGLU fc1 = [fc1_x; fc1_g] or [fc1_g; fc1_x] (h1_package fast path); compare 3 blocks both ways
    fused = {}
    hs = [safe_open(str(p), "pt") for p, _ in eva_src]
    for blk in (0, 11, 23):
        for suffix in ("weight", "bias"):
            key = f"backbone.blocks.{blk}.mlp.fc1.{suffix}"
            parts = {}
            for part in ("fc1_x", "fc1_g"):
                pk = f"backbone.blocks.{blk}.mlp.{part}.{suffix}"
                parts[part] = sum(0.5 * h.get_tensor(pk).float() for h in hs)
            for order in (("fc1_x", "fc1_g"), ("fc1_g", "fc1_x")):
                cat = torch.cat([parts[o] for o in order], dim=0)
                if tuple(cat.shape) == tuple(deq(b124[2][key]).shape):
                    fused[f"{key} as cat{order}"] = cmp(b124[2][key], pack(cat))
                else:
                    fused[f"{key} as cat{order}"] = {"exact": False, "note": f"packaged shape {tuple(deq(b124[2][key]).shape)} vs {tuple(cat.shape)}"}
    m2["fused_fc1_sample"] = fused
    REPORT["member_2"] = m2
    print("member 2 done", round(time.monotonic() - t0, 1), flush=True)

    # ---------------- member 1: DINOv3-H+/16 192
    m1: dict = {}
    t136 = load_pt(B127 / "traced_b136c_spherical/assets/model_1.weights.pt")
    t137 = load_pt(B127 / "traced_b137c_tri_spherical/assets/model_1.weights.pt")
    m1["keys_changed_vs_traced_b136c_model_1"] = diff_keys(w[1], t136)
    m1["keys_changed_vs_traced_b137c_model_1"] = diff_keys(w[1], t137)
    hp = {n: heads(ema(n)) for n in ("d3hpq", "d3hpqr", "d3hpqrq", "d3hpqs")}
    pw_q = np.load(KS / "d3hpq_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    pw_qm = np.load(KS / "d3hpqm_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    bkp = torch.load(str(RUNS / "20261018T0300+0900_gpu_representation_and_soup_cycle/extracted_features_v1/extra_heads_backup_v1.pt"),
                     map_location="cpu")
    sd_qm = dict(bkp["d3hpqm_all_trainval_seed20261007_ema0.999"])
    dqm = torch.from_numpy((np.log(pw_q) - np.log(pw_qm)).astype(np.float32))
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_qm:
            sd_qm[bk] = (sd_qm[bk].float() + dqm).to(sd_qm[bk].dtype).contiguous()
    h6 = heads(RUNS / "20261017T0345+0900_hplus6/hp6_all/model.safetensors")
    r224 = {}
    for k in hp["d3hpq"]:
        if hp["d3hpq"][k].is_floating_point():
            sum4 = hp["d3hpq"][k].float() + hp["d3hpqr"][k].float() + hp["d3hpqrq"][k].float() + hp["d3hpqs"][k].float()
            r224[k] = 3.0 * h6[k].float() - 0.5 * sum4
        else:
            r224[k] = h6[k]
    mems = {"q": hp["d3hpq"], "qr": hp["d3hpqr"], "qrq": hp["d3hpqrq"], "qs": hp["d3hpqs"], "qm": sd_qm, "r224": r224}
    w4 = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20, "qm": 0.0, "r224": 0.0}
    w610 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "qm": 0.0, "r224": 0.10}
    w5m15 = {"q": 0.17, "qr": 0.255, "qrq": 0.255, "qs": 0.17, "qm": 0.15, "r224": 0.0}
    w7m15 = {"q": 0.15, "qr": 0.225, "qrq": 0.225, "qs": 0.15, "qm": 0.15, "r224": 0.10}
    col = {c: w7m15 for c in [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]}
    col.update({c: w4 for c in [3, 7, 20]})
    col[33] = w5m15
    col[34] = w5m15
    hp_head = colselective(mems, w610, col, out_scale=1.03)
    m1["head_recompute_pack_b138c_dich3_ltype2_r00"] = check_heads(w[1], hp_head)
    # r224 extrapolation = mean of the two deleted 224-px recipe heads (algebra: 3 * mean6 - sum4 / 2 = (a + b) / 2)
    m1["r224_is_mean_of_d3hp224q_d3hp224qr"] = "exact in exact arithmetic (hp6_all = uniform 1/6 soup of q, qr, qrq, qs, 224q, 224qr)"
    blocks = (0, 10, 20, 31)
    bb1 = [k for k in w[1] if not k.startswith("head.") and k != "backbone.rope.periods"
           and (not k.startswith("backbone.blocks.") or int(k.split(".")[2]) in blocks)]
    hp_src = [(ema("d3hpq"), 0.20), (ema("d3hpqr"), 0.30), (ema("d3hpqrq"), 0.30), (ema("d3hpqs"), 0.20)]
    m1["backbone_soup_recompute_sampled_blocks"] = {"blocks": list(blocks),
                                                    **soup_check(w[1], hp_src, bb1, mode="build_weighted_soup")}
    REPORT["member_1"] = m1
    REPORT["wall_seconds"] = round(time.monotonic() - t0, 1)
    out_path.write_text(json.dumps(REPORT, indent=1) + "\n")
    print("done", REPORT["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
