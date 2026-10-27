from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from safetensors.torch import load_file
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from upar.execute_submission import load_submission_module


def make_colselective_head_dict(
    member_sds: dict[str, dict[str, torch.Tensor]],
    head_weights: dict[str, float],
    col_overrides: dict[int, dict[str, float]] | None = None,
    tau_qk: float = 1.0,
    tau_out: float = 1.0,
    tau_base: float = 1.0,
    q_scale: float = 1.0,
    out_scale: float = 1.0,
    base_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, tau, sc in [
        ("head.q", tau_qk, q_scale),
        ("head.k.weight", tau_qk, 1.0),
        ("head.out", tau_out, out_scale),
        ("head.base.weight", tau_base, base_scale),
        ("head.base.bias", 0.0, 1.0),
    ]:
        base_t = sum(float(w) * member_sds[m][k].float() for m, w in head_weights.items() if w > 0).clone()
        if tau != 0.0 and base_t.ndim == 2:
            soup_norm = base_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mem_norm = sum(
                float(w) * member_sds[m][k].float().norm(dim=-1, keepdim=True)
                for m, w in head_weights.items()
                if w > 0
            )
            base_t = base_t * ((mem_norm / soup_norm) ** float(tau))

        if col_overrides and k != "head.k.weight":
            for j, w_col in col_overrides.items():
                row_t = sum(float(w) * member_sds[m][k][j].float() for m, w in w_col.items() if w > 0)
                if tau != 0.0 and base_t.ndim == 2:
                    s_norm = row_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    m_norm = sum(
                        float(w) * member_sds[m][k][j].float().norm(dim=-1, keepdim=True)
                        for m, w in w_col.items()
                        if w > 0
                    )
                    row_t = row_t * ((m_norm / s_norm) ** float(tau))
                base_t[j] = row_t

        if sc != 1.0:
            base_t = base_t * float(sc)
        out[k] = base_t
    return out


def pack_tensor(val: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
    val = val.detach().float()
    if val.ndim == 2 and val.numel() >= 4096:
        scale = val.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        return {
            "q": torch.round(val / scale[:, None]).clamp(-127, 127).to(torch.int8),
            "s": scale.half(),
        }
    return val.half()


def main() -> None:
    t0 = time.monotonic()
    mode = sys.argv[1] if len(sys.argv) > 1 else "body12_m15"
    parent_pkg = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/package_v1/package"
    prev_traced = HERE / "traced_b137c_tri_spherical"
    ref_w1_pt = torch.load(prev_traced / "assets/model_1.weights.pt", map_location="cpu")
    ref_w2_pt = torch.load(prev_traced / "assets/model_2.weights.pt", map_location="cpu")

    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    feat_dir = ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle/extracted_features_v1"

    sd_q = load_file(str(ks / "d3hpq_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qr = load_file(str(ks / "d3hpqr_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qrq = load_file(str(ks / "d3hpqrq_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qs = load_file(str(ks / "d3hpqs_all_trainval_seed20261007_ema0.999/model.safetensors"))

    pw_ref = np.load(ks / "d3hpq_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    pw_qm = np.load(ks / "d3hpqm_all_trainval_seed20261007_ema0.999/val_logits.npz")["pos_weight"]
    bias_delta_qm = torch.from_numpy((np.log(pw_ref) - np.log(pw_qm)).astype(np.float32))
    bkp = torch.load(feat_dir / "extra_heads_backup_v1.pt", map_location="cpu")
    sd_qm = dict(bkp["d3hpqm_all_trainval_seed20261007_ema0.999"])
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_qm:
            sd_qm[bk] = (sd_qm[bk].float() + bias_delta_qm).to(sd_qm[bk].dtype).contiguous()

    sd_hp6 = load_file(str(ROOT / "artifacts/runs/20261017T0345+0900_hplus6/hp6_all/model.safetensors"))
    sd_r224 = {}
    for k in sd_q:
        if k.startswith("head.") and sd_q[k].is_floating_point():
            sum4 = sd_q[k].float() + sd_qr[k].float() + sd_qrq[k].float() + sd_qs[k].float()
            sd_r224[k] = 3.0 * sd_hp6[k].float() - 0.5 * sum4
        else:
            sd_r224[k] = sd_hp6[k]
    del sd_hp6

    hp_mems = {"q": sd_q, "qr": sd_qr, "qrq": sd_qrq, "qs": sd_qs, "qm": sd_qm, "r224": sd_r224}
    hp_w_4_r00 = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20, "qm": 0.0, "r224": 0.0}
    hp_w_6_10 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "qm": 0.0, "r224": 0.10}
    hp_w_6_15 = {"q": 0.17, "qr": 0.255, "qrq": 0.255, "qs": 0.17, "qm": 0.0, "r224": 0.15}
    hp_w_5_m15_r00 = {"q": 0.17, "qr": 0.255, "qrq": 0.255, "qs": 0.17, "qm": 0.15, "r224": 0.0}
    hp_w_7_m10_r10 = {"q": 0.16, "qr": 0.24, "qrq": 0.24, "qs": 0.16, "qm": 0.10, "r224": 0.10}
    hp_w_7_m12_r10 = {"q": 0.156, "qr": 0.234, "qrq": 0.234, "qs": 0.156, "qm": 0.12, "r224": 0.10}
    hp_w_7_m15_r10 = {"q": 0.15, "qr": 0.225, "qrq": 0.225, "qs": 0.15, "qm": 0.15, "r224": 0.10}

    CLEAN_BODY_8 = [10, 13, 14, 26, 29, 32, 33, 34]
    AMBIG_4 = [16, 18, 27, 31]
    BODY_12 = [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
    DICH_3 = [3, 7, 20]
    FG_UNMASK_4 = [4, 23, 35, 36]
    BAG_2 = [35, 36]

    if mode == "body12_m15":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
    elif mode == "dich3_r00":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3})
    elif mode == "dich3_c34_r00":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3})
        col_ov[34] = hp_w_5_m15_r00
    elif mode in ("dich3_ltype2_r00", "dich3_ltype2_r00_eva_c36_40", "dich3_ltype2_r00_eva_bag42"):
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3})
        col_ov[33] = hp_w_5_m15_r00
        col_ov[34] = hp_w_5_m15_r00
    elif mode == "dich3_c34_c22_r00":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3 + [22]})
        col_ov[34] = hp_w_5_m15_r00
    elif mode == "all3of3_r00":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in [3, 7, 20, 22, 28, 38]})
        for c in [10, 14, 18, 29, 31, 34]:
            col_ov[c] = hp_w_5_m15_r00
    elif mode == "dich3_r00_fg4_r15":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3})
        col_ov.update({c: hp_w_6_15 for c in FG_UNMASK_4})
    elif mode == "dich3_c34_r00_fg4_r15":
        col_ov = {c: hp_w_7_m15_r10 for c in BODY_12}
        col_ov.update({c: hp_w_4_r00 for c in DICH_3})
        col_ov[34] = hp_w_5_m15_r00
        col_ov.update({c: hp_w_6_15 for c in FG_UNMASK_4})
    elif mode == "c8m15_a4m12":
        col_ov = {c: hp_w_7_m15_r10 for c in CLEAN_BODY_8}
        col_ov.update({c: hp_w_7_m12_r10 for c in AMBIG_4})
    elif mode == "c8m15_a4m10":
        col_ov = {c: hp_w_7_m15_r10 for c in CLEAN_BODY_8}
        col_ov.update({c: hp_w_7_m10_r10 for c in AMBIG_4})
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hp_head = make_colselective_head_dict(
        hp_mems,
        hp_w_6_10,
        col_overrides=col_ov,
        tau_qk=1.0,
        tau_out=1.0,
        tau_base=1.0,
        q_scale=1.0,
        out_scale=1.03,
        base_scale=1.0,
    )

    out_dir = HERE / f"traced_b138c_{mode}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Hardlink unchanged model_0 and model_2 weights + canary refs from traced_b137c_tri_spherical
    os.link(prev_traced / "assets/model_0.weights.pt", out_dir / "assets/model_0.weights.pt")
    shutil.copyfile(prev_traced / "canary_ref_0.json", out_dir / "canary_ref_0.json")

    need_canary2 = False
    if mode in ("dich3_ltype2_r00_eva_c36_40", "dich3_ltype2_r00_eva_bag42"):
        sd224 = load_file(str(ks / "eva02L224_all_trainval_seed20261007_ema0.999/model.safetensors"))
        sd252 = load_file(str(ks / "eva02L252_all_trainval_seed20261007_ema0.999/model.safetensors"))
        W1, B1 = sd224["head.weight"].float(), sd224["head.bias"].float()
        W2, B2 = sd252["head.weight"].float(), sd252["head.bias"].float()
        w252_vec = torch.full((40, 1), 0.50, dtype=torch.float32)
        if mode == "dich3_ltype2_r00_eva_c36_40":
            w252_vec[36, 0] = 0.40
        else:
            w252_vec[35, 0] = 0.42
            w252_vec[36, 0] = 0.42
        w224_vec = 1.0 - w252_vec
        W = w224_vec * W1 + w252_vec * W2
        B = w224_vec[:, 0] * B1 + w252_vec[:, 0] * B2
        mn = w224_vec * W1.norm(dim=-1, keepdim=True) + w252_vec * W2.norm(dim=-1, keepdim=True)
        sn = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W = W * ((mn / sn) ** 1.20)
        stored2 = dict(ref_w2_pt)
        stored2["head.weight"] = pack_tensor(W)
        stored2["head.bias"] = pack_tensor(B)
        out_w2 = out_dir / "assets/model_2.weights.pt"
        torch.save(stored2, str(out_w2))
        need_canary2 = True
    else:
        os.link(prev_traced / "assets/model_2.weights.pt", out_dir / "assets/model_2.weights.pt")
        shutil.copyfile(prev_traced / "canary_ref_2.json", out_dir / "canary_ref_2.json")

    stored1 = dict(ref_w1_pt)
    for k, v in hp_head.items():
        if k in stored1:
            stored1[k] = pack_tensor(v)
    stored1["head.weight"] = pack_tensor(hp_head["head.base.weight"])
    stored1["head.bias"] = pack_tensor(hp_head["head.base.bias"])
    assert set(stored1.keys()) == set(ref_w1_pt.keys()), "model_1 key mismatch"

    out_w1 = out_dir / "assets/model_1.weights.pt"
    torch.save(stored1, str(out_w1))
    print(f"Saved {out_w1} (mode={mode}) in {time.monotonic() - t0:.1f}s", flush=True)

    sys.path.insert(0, str(parent_pkg))
    mod = load_submission_module(parent_pkg / "run.py")
    cfg = json.loads((parent_pkg / "assets/config.json").read_text())
    mod._CONFIG = cfg
    mod._DEVICE = "cpu"
    h, w = (int(v) for v in cfg["cache_hw"])
    x = mod._canary_input(4, h, w)

    spec1 = dict(cfg["models"][1])
    spec1["weights_file"] = str(out_w1.resolve())
    m1 = mod._load_ts(spec1, "cpu")
    with torch.inference_mode():
        ref1 = mod._run(m1, spec1, x).tolist()
    (out_dir / "canary_ref_1.json").write_text(json.dumps(ref1))
    print(f"Computed canary_ref_1.json in {time.monotonic() - t0:.1f}s", flush=True)

    if need_canary2:
        spec2 = dict(cfg["models"][2])
        spec2["weights_file"] = str((out_dir / "assets/model_2.weights.pt").resolve())
        m2 = mod._load_ts(spec2, "cpu")
        with torch.inference_mode():
            ref2 = mod._run(m2, spec2, x).tolist()
        (out_dir / "canary_ref_2.json").write_text(json.dumps(ref2))
        print(f"Computed canary_ref_2.json in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
