"""Make a soup-able copy of an EVA02-L run at another training resolution: only backbone.pos_embed differs (224x112 -> 16x8
grid, 252x126 -> 18x9 grid; RoPE is rebuilt from the grid at load), so resample it (timm resample_abs_pos_embed, bicubic,
antialias, 1 prefix token) and relabel the backbone. val_logits.npz is copied ONLY for pos_weight (h1_finetune soup reads it);
the copy's logits are the source model's, so never evaluate this directory itself.
usage: make_xres_member.py SRC_RUN_DIR TARGET_BACKBONE OUT_DIR"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from timm.layers import resample_abs_pos_embed
sys.path.insert(0, "src")
from upar.h1_finetune import BACKBONES
src, tgt, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = json.loads((src / "config.json").read_text())
def grid(bb):
    h, w = BACKBONES[bb]["img_hw"]; return h // 14, w // 14
sd = load_file(str(src / "model.safetensors"))
pe = sd["backbone.pos_embed"]; g0, g1 = grid(cfg["backbone"]), grid(tgt)
assert pe.shape[1] == 1 + g0[0] * g0[1], (pe.shape, g0)
sd["backbone.pos_embed"] = resample_abs_pos_embed(pe.float(), new_size=list(g1), old_size=list(g0), num_prefix_tokens=1,
                                                  interpolation="bicubic", antialias=True).to(pe.dtype).contiguous()
out.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out / "model.safetensors"))
cfg.update(backbone=tgt, output_dir=str(out), xres_of=str(src), note="pos_embed resampled copy for soups only; val_logits copied for pos_weight only")
(out / "config.json").write_text(json.dumps(cfg, indent=1))
shutil.copy(src / "val_logits.npz", out / "val_logits.npz")
print("ok", cfg["backbone"], tuple(pe.shape), "->", tuple(sd["backbone.pos_embed"].shape))
