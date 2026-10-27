from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

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
    ref_w1_pt = torch.load(HERE / "traced_hp4_2030_192/assets/model_1.weights.pt", map_location="cpu")

    ks = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
    pe_mems = {
        "attrq": load_file(str(ks / "peL280_attrq_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "rankq": load_file(str(ks / "pLq_rankq_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "maskv2": load_file(str(ROOT / "artifacts/runs/20261014T1400+0900_b115g/tmp_maskv2_all_s7_pw/model.safetensors")),
        "pswap": load_file(str(ks / "pLq_pswap03_all_trainval_seed20261007_ema0.999/model.safetensors")),
    }
    pe_w = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
    pe_head = make_spherical_head_dict(pe_mems, pe_w, tau_qk=1.0, tau_out=1.0, tau_base=1.0, q_scale=1.0, out_scale=1.0, base_scale=1.0)

    hp_mems = {
        "q": load_file(str(ks / "d3hpq_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "qr": load_file(str(ks / "d3hpqr_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "qrq": load_file(str(ks / "d3hpqrq_all_trainval_seed20261007_ema0.999/model.safetensors")),
        "qs": load_file(str(ks / "d3hpqs_all_trainval_seed20261007_ema0.999/model.safetensors")),
    }
    hp_w = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20}
    hp_head = make_spherical_head_dict(hp_mems, hp_w, tau_qk=1.0, tau_out=1.0, tau_base=1.0, q_scale=1.0, out_scale=1.03, base_scale=1.0)

    out_dir = HERE / "traced_b136c_spherical"
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Update only head.* keys in ref_w0_pt and ref_w1_pt (keeping backbone.* 100% bit-identical)
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

    out_w0 = out_dir / "assets/model_0.weights.pt"
    out_w1 = out_dir / "assets/model_1.weights.pt"
    torch.save(stored0, str(out_w0))
    torch.save(stored1, str(out_w1))
    print(f"Saved {out_w0} and {out_w1} in {time.monotonic() - t0:.1f}s", flush=True)

    sys.path.insert(0, str(parent_pkg))
    mod = load_submission_module(parent_pkg / "run.py")
    cfg = json.loads((parent_pkg / "assets/config.json").read_text())
    mod._CONFIG = cfg
    mod._DEVICE = "cpu"
    h, w = (int(v) for v in cfg["cache_hw"])
    x = mod._canary_input(4, h, w)

    spec0 = dict(cfg["models"][0])
    spec0["weights_file"] = str(out_w0.resolve())
    m0 = mod._load_ts(spec0, "cpu")
    with torch.inference_mode():
        ref0 = mod._run(m0, spec0, x).tolist()
    (out_dir / "canary_ref_0.json").write_text(json.dumps(ref0))
    print(f"Computed canary_ref_0.json in {time.monotonic() - t0:.1f}s", flush=True)

    spec1 = dict(cfg["models"][1])
    spec1["weights_file"] = str(out_w1.resolve())
    m1 = mod._load_ts(spec1, "cpu")
    with torch.inference_mode():
        ref1 = mod._run(m1, spec1, x).tolist()
    (out_dir / "canary_ref_1.json").write_text(json.dumps(ref1))
    print(f"Computed canary_ref_1.json in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

