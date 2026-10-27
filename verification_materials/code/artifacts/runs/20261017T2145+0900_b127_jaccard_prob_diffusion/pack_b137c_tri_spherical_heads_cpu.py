from __future__ import annotations

import json
import os
from pathlib import Path
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


def make_spherical_head_dict(
    mems: dict[str, dict[str, torch.Tensor]],
    weights: dict[str, float],
    tau_qk: float = 1.0,
    tau_out: float = 1.0,
    tau_base: float = 1.0,
    q_scale: float = 1.0,
    out_scale: float = 1.0,
    base_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    head_keys = [k for k in next(iter(mems.values())).keys() if k.startswith("head.")]
    out: dict[str, torch.Tensor] = {}
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
                ratio = (mean_norm / soup_norm) ** t_exp
                soup_t = soup_t * ratio
        if k == "head.q" and q_scale != 1.0:
            soup_t = soup_t * q_scale
        elif k == "head.out" and out_scale != 1.0:
            soup_t = soup_t * out_scale
        elif k in ("head.base.weight", "head.weight") and base_scale != 1.0:
            soup_t = soup_t * base_scale
        out[k] = soup_t
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
    parent_pkg = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/package_v1/package"
    ref_w0_pt = torch.load(parent_pkg / "assets/model_0.weights.pt", map_location="cpu")
    ref_w1_pt = torch.load(HERE / "traced_b136c_spherical/assets/model_1.weights.pt", map_location="cpu")
    ref_w2_pt = torch.load(parent_pkg / "assets/model_2.weights.pt", map_location="cpu")

    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"

    # 1. Member 0: PE-Core-L/14 (280x140) Spatial-SNR Decoupled Spherical Head (tau_qk=1.20, tau_out=1.00, tau_base=0.60)
    p_attrq = ks / "peL280_attrq_all_trainval_seed20261007_ema0.999"
    p_maskv2 = ks / "pLq_maskv2_all_trainval_seed20261007_ema0.999"
    pw_ref = np.load(p_attrq / "val_logits.npz")["pos_weight"]
    pw_mask = np.load(p_maskv2 / "val_logits.npz")["pos_weight"]
    bias_delta = torch.from_numpy((np.log(pw_ref) - np.log(pw_mask)).astype(np.float32))
    sd_mask = load_file(str(p_maskv2 / "model.safetensors"))
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_mask:
            sd_mask[bk] = (sd_mask[bk].float() + bias_delta).to(sd_mask[bk].dtype).contiguous()
    pe_mems = {
        "attrq": load_file(str(p_attrq / "model.safetensors")),
        "rankq": load_file(str(ks / "pLq_rankq_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "maskv2": sd_mask,
        "pswap": load_file(str(ks / "pLq_pswap03_all_trainval_seed20261007_ema0.999/model.safetensors")),
    }
    pe_w = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
    pe_head = make_spherical_head_dict(
        pe_mems, pe_w, tau_qk=1.20, tau_out=1.00, tau_base=0.60, q_scale=1.0, out_scale=1.0, base_scale=1.0
    )

    # 2. Member 1: DINOv3-H+/16 (192x96) 6-Recipe Multi-Resolution Head (hp6_10: 18/27/27/18 + 10% r224, tau=1.0, out_scale=1.03)
    sd_q = load_file(str(ks / "d3hpq_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qr = load_file(str(ks / "d3hpqr_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qrq = load_file(str(ks / "d3hpqrq_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_qs = load_file(str(ks / "d3hpqs_all_trainval_seed20261007_ema0.999/model.safetensors"))
    sd_hp6 = load_file(str(ROOT / "artifacts/runs/20261017T0345+0900_hplus6/hp6_all/model.safetensors"))
    sd_r224 = {}
    for k in sd_q:
        if k.startswith("head.") and sd_q[k].is_floating_point():
            sum4 = sd_q[k].float() + sd_qr[k].float() + sd_qrq[k].float() + sd_qs[k].float()
            sd_r224[k] = 3.0 * sd_hp6[k].float() - 0.5 * sum4
        else:
            sd_r224[k] = sd_hp6[k]
    del sd_hp6

    hp_mems = {"q": sd_q, "qr": sd_qr, "qrq": sd_qrq, "qs": sd_qs, "r224": sd_r224}
    hp_w = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "r224": 0.10}
    hp_head = make_spherical_head_dict(
        hp_mems, hp_w, tau_qk=1.0, tau_out=1.0, tau_base=1.0, q_scale=1.0, out_scale=1.03, base_scale=1.0
    )

    # 3. Member 2: EVA02-L/14 (252x126) 2-Member Multi-Resolution Spherical Head (0.50 eva224 + 0.50 eva252, tau_base=1.20)
    eva_mems = {
        "eva224": load_file(str(ks / "eva02L224_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "eva252": load_file(str(ks / "eva02L252_all_trainval_seed20261007_ema0.999/model.safetensors")),
    }
    eva_w = {"eva224": 0.50, "eva252": 0.50}
    eva_head = make_spherical_head_dict(
        eva_mems, eva_w, tau_qk=0.0, tau_out=0.0, tau_base=1.20, q_scale=1.0, out_scale=1.0, base_scale=1.0
    )

    out_dir = HERE / "traced_b137c_tri_spherical"
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Update only head.* keys in ref_w0_pt, ref_w1_pt, ref_w2_pt (keeping backbone.* 100% bit-identical)
    stored0 = dict(ref_w0_pt)
    for k, v in pe_head.items():
        if k in stored0:
            stored0[k] = pack_tensor(v)
    stored0["head.weight"] = pack_tensor(pe_head["head.base.weight"])
    stored0["head.bias"] = pack_tensor(pe_head["head.base.bias"])
    assert set(stored0.keys()) == set(ref_w0_pt.keys()), "model_0 key mismatch"

    stored1 = dict(ref_w1_pt)
    for k, v in hp_head.items():
        if k in stored1:
            stored1[k] = pack_tensor(v)
    stored1["head.weight"] = pack_tensor(hp_head["head.base.weight"])
    stored1["head.bias"] = pack_tensor(hp_head["head.base.bias"])
    assert set(stored1.keys()) == set(ref_w1_pt.keys()), "model_1 key mismatch"

    stored2 = dict(ref_w2_pt)
    stored2["head.weight"] = pack_tensor(eva_head["head.weight"])
    stored2["head.bias"] = pack_tensor(eva_head["head.bias"])
    assert set(stored2.keys()) == set(ref_w2_pt.keys()), "model_2 key mismatch"

    out_w0 = out_dir / "assets/model_0.weights.pt"
    out_w1 = out_dir / "assets/model_1.weights.pt"
    out_w2 = out_dir / "assets/model_2.weights.pt"
    torch.save(stored0, str(out_w0))
    torch.save(stored1, str(out_w1))
    torch.save(stored2, str(out_w2))
    print(f"Saved {out_w0}, {out_w1}, {out_w2} in {time.monotonic() - t0:.1f}s", flush=True)

    sys.path.insert(0, str(parent_pkg))
    mod = load_submission_module(parent_pkg / "run.py")
    cfg = json.loads((parent_pkg / "assets/config.json").read_text())
    mod._CONFIG = cfg
    mod._DEVICE = "cpu"
    h, w = (int(v) for v in cfg["cache_hw"])
    x = mod._canary_input(4, h, w)

    for idx, out_w in ((0, out_w0), (1, out_w1), (2, out_w2)):
        spec = dict(cfg["models"][idx])
        spec["weights_file"] = str(out_w.resolve())
        m = mod._load_ts(spec, "cpu")
        with torch.inference_mode():
            ref = mod._run(m, spec, x).tolist()
        (out_dir / f"canary_ref_{idx}.json").write_text(json.dumps(ref))
        print(f"Computed canary_ref_{idx}.json in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
