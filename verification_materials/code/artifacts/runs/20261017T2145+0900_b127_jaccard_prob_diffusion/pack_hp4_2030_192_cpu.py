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


def main() -> None:
    t0 = time.monotonic()
    parent_pkg = ROOT / "artifacts/runs/20261017T1615+0900_b124_alpha05/package_v1/package"
    ref_w_pt = torch.load(parent_pkg / "assets/model_1.weights.pt", map_location="cpu")

    hp4_2030_dir = ROOT / "artifacts/runs/20261017T0345+0900_hplus6/hp4_20_30_30_20_all"
    sd = load_file(str(hp4_2030_dir / "model.safetensors"))

    out_dir = HERE / "traced_hp4_2030_192"
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    remap = {
        "head.base.weight": "head.weight",
        "head.base.bias": "head.bias",
    }
    stored = {}
    for k, v in sd.items():
        k_pt = k if k in ref_w_pt else remap.get(k, k)
        val = v.detach().float()
        if val.ndim == 2 and val.numel() >= 4096:
            scale = val.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            stored[k_pt] = {
                "q": torch.round(val / scale[:, None]).clamp(-127, 127).to(torch.int8),
                "s": scale.half(),
            }
        else:
            stored[k_pt] = val.half()
    stored["backbone.rope.periods"] = ref_w_pt["backbone.rope.periods"]
    assert set(stored.keys()) == set(ref_w_pt.keys()), "key set mismatch vs parent model_1.weights.pt"

    out_w_file = out_dir / "assets/model_1.weights.pt"
    torch.save(stored, str(out_w_file))
    print(f"Saved {out_w_file} in {time.monotonic() - t0:.1f}s", flush=True)

    # Compute CPU fp32 canary_ref[1] using parent run.py and model_1 graphs
    sys.path.insert(0, str(parent_pkg))
    mod = load_submission_module(parent_pkg / "run.py")
    cfg = json.loads((parent_pkg / "assets/config.json").read_text())
    mod._CONFIG = cfg
    mod._DEVICE = "cpu"
    spec1 = dict(cfg["models"][1])
    # Point weights_file to our new weights
    spec1["weights_file"] = str(out_w_file.resolve())
    m1 = mod._load_ts(spec1, "cpu")
    h, w = (int(v) for v in cfg["cache_hw"])
    x = mod._canary_input(4, h, w)
    with torch.inference_mode():
        ref1 = mod._run(m1, spec1, x).tolist()
    (out_dir / "canary_ref_1.json").write_text(json.dumps(ref1))
    print(f"Computed canary_ref_1.json in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
