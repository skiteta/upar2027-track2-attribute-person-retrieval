"""Follow-up to verify_weight_chain_v1.py: re-check the backbone keys that h1_package's --fast rewrites change
(src/upar/h1_package.py::_fast_rewrite): rope_half permutes the q/k output rows of every head (evens, then odds) for
PE-Core and EVA02, and mlp_pad stores EVA02's SwiGLU fc1_g / fc1_x as one zero-padded fc1 (2730 -> 2736 rows each).
Applies the same transforms to the recomputed fp32 soups and compares after int8/fp16 packing. Read-only.
usage: .venv-train/bin/python -I verify_weight_chain_fastpath_v1.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_weight_chain_v2 import B124, PKG, cmp, ema, load_pt, pack, summarize  # noqa: E402

torch.set_num_threads(4)


def rope_perm(num_heads: int, hd: int) -> torch.Tensor:
    ph = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])
    return (torch.arange(num_heads)[:, None] * hd + ph[None]).reshape(-1)


def soup(handles, weights, key):
    return sum(float(w) * h.get_tensor(key).float() for h, w in zip(handles, weights))


def main() -> None:
    out = {}
    w150 = {i: load_pt(PKG / f"model_{i}.weights.pt") for i in (0, 2)}
    b124 = {i: load_pt(B124 / f"model_{i}.weights.pt") for i in (0, 2)}

    # member 0: PE-Core-L/14, fused qkv, 16 heads x 64
    hs = [safe_open(str(ema(n)), "pt") for n in ("peL280_attrq", "pLq_rankq", "pLq_maskv2", "pLq_pswap03")]
    ws = [0.25] * 4
    perm = rope_perm(16, 64)
    c = 1024
    rows = torch.cat([perm, c + perm, torch.arange(2 * c, 3 * c)])
    r0 = {}
    for b in range(24):
        for suf in ("weight", "bias"):
            k = f"backbone.blocks.{b}.attn.qkv.{suf}"
            r0[k] = cmp(b124[0][k], pack(soup(hs, ws, k)[rows]))
            r0[k + " (B150C = B124C)"] = cmp(w150[0][k], b124[0][k])
    out["member_0_qkv_rope_half"] = summarize(r0)

    # member 2: EVA02-L/14, separate q/k/v projections, 16 heads x 64; SwiGLU hidden 2730 padded to 2736
    hs = [safe_open(str(ema(n)), "pt") for n in ("eva02L224", "eva02L252")]
    ws = [0.5, 0.5]
    keys = set(hs[0].keys())
    r2 = {}
    for b in range(24):
        for proj in ("q_proj", "k_proj"):
            for suf in ("weight", "bias"):
                k = f"backbone.blocks.{b}.attn.{proj}.{suf}"
                if k in keys and k in b124[2]:
                    r2[k] = cmp(b124[2][k], pack(soup(hs, ws, k)[perm]))
        for n in ("q_bias", "k_bias"):
            k = f"backbone.blocks.{b}.attn.{n}"
            if k in keys and k in b124[2]:
                r2[k] = cmp(b124[2][k], pack(soup(hs, ws, k)[perm]))
        for suf in ("weight", "bias"):
            g = soup(hs, ws, f"backbone.blocks.{b}.mlp.fc1_g.{suf}")
            x = soup(hs, ws, f"backbone.blocks.{b}.mlp.fc1_x.{suf}")
            h = g.shape[0]
            p = -(-h // 8) * 8
            fused = torch.zeros((2 * p,) + tuple(g.shape[1:]), dtype=torch.float32)
            fused[:h], fused[p:p + h] = g, x
            k = f"backbone.blocks.{b}.mlp.fc1.{suf}"
            r2[k] = cmp(b124[2][k], pack(fused))
    for k in [k for k in b124[2] if not k.startswith("head.")]:
        r2.setdefault(k + " (B150C = B124C)", cmp(w150[2][k], b124[2][k]))
    out["member_2_qk_rope_half_and_mlp_pad"] = summarize(r2)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("compared", "exact", "not_exact")} for k, v in out.items()}))


if __name__ == "__main__":
    main()
