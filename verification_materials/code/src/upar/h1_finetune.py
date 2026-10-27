"""H1: full fine-tuning of generic ViT-B foundation encoders for the 40 UPAR attributes.

Subcommands
  cache     decode every train/val image once with the official runtime resize (PIL bilinear
            to 256x128) into a uint8 memmap so training is GPU-bound.
  train     LP-FT of one backbone on one fold (all sources or one held-out source), then save
            original/flip logits for every validation image and a safetensors checkpoint.
  evaluate  score saved fold predictions with the fixed B022 common ranker (B011 weights,
            gamma 0.5, NLL lambda 0.25) and the strict-v2 metric, per held source and full.

Fit boundary: a held-source fold never sees that source's images or labels (train or val).
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from multiprocessing import Pool
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "starter_kit/data"
SPLITS = {"train": ROOT / "starter_kit/data/annotations/task2/train", "val": ROOT / "starter_kit/data/annotations/task2/val"}
EXCLUDED = ("PETA/images/08572.png",)  # known cross-split exact-pixel duplicate (train side)
SOURCES = ("Market1501", "PA100k", "PETA")
CACHE_HW = (256, 128)
B011 = ROOT / "artifacts/runs/20261003T134237325056065+0900_b011_crossfit_attribute_reliability/reliability_search.json"

BACKBONES: dict[str, dict[str, Any]] = {
    "clip_b16": {
        "timm": "vit_base_patch16_clip_224.openai",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.openai/open_clip_model.safetensors",
        "img_hw": (256, 128),
    },
    "siglip2_b16": {
        "timm": "vit_base_patch16_siglip_256.v2_webli",
        "file": "pretrained/_h1/vit_base_patch16_siglip_256.v2_webli/model.safetensors",
        "img_hw": (256, 128),
    },
    "clip_b16_192": {
        "timm": "vit_base_patch16_clip_224.openai",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.openai/open_clip_model.safetensors",
        "img_hw": (192, 96),
    },
    "dinov2_b14_168": {
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "file": "pretrained/_h1/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors",
        "img_hw": (168, 84),
    },
    "eva02_b14": {
        "timm": "eva02_base_patch14_224.mim_in22k",
        "file": "pretrained/_h1/eva02_base_patch14_224.mim_in22k/model.safetensors",
        "img_hw": (224, 112),
    },
    "dinov2_s14": {
        "timm": "vit_small_patch14_reg4_dinov2.lvd142m",
        "file": "pretrained/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors",
        "img_hw": (224, 112),
    },
    "dinov2_b14": {
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "file": "pretrained/_h1/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors",
        "img_hw": (224, 112),
    },
    # same-token-budget (72 patches) alternatives to the B089 low-res pair
    "dinov3_b16_192": {
        "timm": "vit_base_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_base_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (192, 96),
    },
    "pe_core_b16_192": {
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (192, 96),
    },
    "dfn2b_clip_b16_192": {
        "timm": "vit_base_patch16_clip_224.dfn2b",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.dfn2b/model.safetensors",
        "img_hw": (192, 96),
    },
    "metaclip_b16_192": {
        "timm": "vit_base_patch16_clip_224.metaclip_2pt5b",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.metaclip_2pt5b/model.safetensors",
        "img_hw": (192, 96),
    },
    # reduced-token third members (fit the ~0.022 s/img slack on the slowest official worker)
    "dinov2_b14_126": {
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "file": "pretrained/_h1/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors",
        "img_hw": (126, 56),
    },
    "dinov3_b16_128": {
        "timm": "vit_base_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_base_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (128, 64),
    },
    "clip_b16_144": {
        "timm": "vit_base_patch16_clip_224.openai",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.openai/open_clip_model.safetensors",
        "img_hw": (144, 64),
    },
    # 98-token (14x7) pair members: the slack on the slowest worker spent on resolution instead of a third model
    "pe_core_b16_224": {
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (224, 112),
    },
    "pe_core_b16_224x96": {  # 14x6 = 84 tokens: plan A that keeps >= 8% headroom on the slowest worker
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (224, 96),
    },
    "mclip2_b16_224": {  # MobileCLIP2-B (ViT-B with MCi conv stem), 14x7 patches at 224x112
        "timm": "vit_base_mci_224.apple_mclip2_dfndr2b",
        "file": "pretrained/_h1/vit_base_mci_224.apple_mclip2_dfndr2b/model.safetensors",
        "img_hw": (224, 112),
    },
    "pe_core_l14_224": {  # PE-Core-L/14 at 224x112: 16x8 = 128 tokens (GPU runners)
        "timm": "vit_pe_core_large_patch14_336.fb",
        "file": "pretrained/_h1/vit_pe_core_large_patch14_336.fb/model.safetensors",
        "img_hw": (224, 112),
    },
    "dinov3_l16_192": {  # DINOv3-L/16 at 192x96: 12x6 = 72 tokens (GPU runners)
        "timm": "vit_large_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_large_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (192, 96),
    },
    "pe_core_b16_256": {  # PE-Core-B/16 at 256x128: 16x8 = 128 tokens (GPU runners)
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (256, 128),
    },
    "pe_core_b16_288": {  # 18x9 = 162 tokens (GPU runners)
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (288, 144),
    },
    "dinov3_b16_256": {  # 16x8 = 128 tokens (GPU runners)
        "timm": "vit_base_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_base_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (256, 128),
    },
    "pe_core_b16_320": {  # 20x10 = 200 tokens (GPU runners; training resolution keeps helping PE: 224<256<288)
        "timm": "vit_pe_core_base_patch16_224.fb",
        "file": "pretrained/_h1/vit_pe_core_base_patch16_224.fb/model.safetensors",
        "img_hw": (320, 160),
    },
    "dinov3_l16_224": {  # DINOv3-L/16 at 224x112: 98 tokens
        "timm": "vit_large_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_large_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (224, 112),
    },
    "pe_core_l14_280": {  # PE-Core-L/14 at 280x140: 20x10 = 200 tokens (training resolution helped PE-B up to 288)
        "timm": "vit_pe_core_large_patch14_336.fb",
        "file": "pretrained/_h1/vit_pe_core_large_patch14_336.fb/model.safetensors",
        "img_hw": (280, 140),
    },
    "dinov3_l16_256": {  # DINOv3-L/16 at 256x128: 16x8 = 128 tokens (192 -> 224 gave +0.0055 single)
        "timm": "vit_large_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_large_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (256, 128),
    },
    "dinov3_hp16_192": {  # DINOv3-H+/16 (840M) at 192x96: 72 tokens
        "timm": "vit_huge_plus_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_huge_plus_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (192, 96),
    },
    "dinov3_hp16_224": {  # DINOv3-H+/16 at 224x112: 98 tokens (RoPE, no absolute position embedding -> weights are
        "timm": "vit_huge_plus_patch16_dinov3.lvd1689m",  # interchangeable with the 192x96 key; 2026-10-15)
        "file": "pretrained/_h1/vit_huge_plus_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (224, 112),
    },
    "clip_l14_224": {  # OpenAI CLIP ViT-L/14 at 224x112: 16x8 = 128 tokens
        "timm": "vit_large_patch14_clip_224.openai",
        "file": "pretrained/_h1/vit_large_patch14_clip_224.openai/open_clip_model.safetensors",
        "img_hw": (224, 112),
    },
    "siglip2_l16_256": {  # SigLIP2 ViT-L/16 at 256x128: 128 tokens
        "timm": "vit_large_patch16_siglip_256.v2_webli",
        "file": "pretrained/_h1/vit_large_patch16_siglip_256.v2_webli/model.safetensors",
        "img_hw": (256, 128),
    },
    "eva02_l14_224": {  # EVA02-L/14 at 224x112: 128 tokens
        "timm": "eva02_large_patch14_224.mim_m38m",
        "file": "pretrained/_h1/eva02_large_patch14_224.mim_m38m/model.safetensors",
        "img_hw": (224, 112),
    },
    "eva02_l14_252": {  # EVA02-L/14 at 252x126: 18x9 = 162 tokens (inferring the 224 model at 252 gave +0.005 single)
        "timm": "eva02_large_patch14_224.mim_m38m",
        "file": "pretrained/_h1/eva02_large_patch14_224.mim_m38m/model.safetensors",
        "img_hw": (252, 126),
    },
    "pe_spatial_b16_192": {  # PE-Spatial-B/16 (distilled from PE-Core-G spatial features), 12x6 = 72 tokens
        "timm": "vit_pe_spatial_base_patch16_512.fb",
        "file": "pretrained/_h1/vit_pe_spatial_base_patch16_512.fb/model.safetensors",
        "img_hw": (192, 96),
    },
    "pe_spatial_b16_224": {
        "timm": "vit_pe_spatial_base_patch16_512.fb",
        "file": "pretrained/_h1/vit_pe_spatial_base_patch16_512.fb/model.safetensors",
        "img_hw": (224, 112),
    },
    "d3convnext_b_192": {  # DINOv3 ConvNeXt-B (distilled from DINOv3-7B); a CNN for architectural diversity
        "timm": "convnext_base.dinov3_lvd1689m",
        "file": "pretrained/_h1/convnext_base.dinov3_lvd1689m/model.safetensors",
        "img_hw": (192, 96),
        "arch": "convnext",
    },
    "d3convnext_b_224": {
        "timm": "convnext_base.dinov3_lvd1689m",
        "file": "pretrained/_h1/convnext_base.dinov3_lvd1689m/model.safetensors",
        "img_hw": (224, 112),
        "arch": "convnext",
    },
    "mclip2_b16_192": {  # 12x6 = 72 tokens: optional third member for fast workers (DINOv3-192 cost)
        "timm": "vit_base_mci_224.apple_mclip2_dfndr2b",
        "file": "pretrained/_h1/vit_base_mci_224.apple_mclip2_dfndr2b/model.safetensors",
        "img_hw": (192, 96),
    },
    "dfn2b_clip_b16_224": {
        "timm": "vit_base_patch16_clip_224.dfn2b",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.dfn2b/model.safetensors",
        "img_hw": (224, 112),
    },
    "dinov3_b16_224": {
        "timm": "vit_base_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_base_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (224, 112),
    },
    # small DINOv3 encoders at higher resolution as a cheap third member
    "dinov3s_b16_256": {
        "timm": "vit_small_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_small_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (256, 128),
    },
    "dinov3sp_b16_256": {
        "timm": "vit_small_plus_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_small_plus_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (256, 128),
    },
    "dinov3sp_b16_224": {
        "timm": "vit_small_plus_patch16_dinov3.lvd1689m",
        "file": "pretrained/_h1/vit_small_plus_patch16_dinov3.lvd1689m/model.safetensors",
        "img_hw": (224, 112),
    },
    # head-region member: top 80 of 256 cache rows at native cache resolution (5x8 = 40 patches)
    "clip_b16_head80": {
        "timm": "vit_base_patch16_clip_224.openai",
        "file": "pretrained/_h1/vit_base_patch16_clip_224.openai/open_clip_model.safetensors",
        "img_hw": (80, 128),
        "crop_top_frac": 0.3125,
    },
}


# ----------------------------------------------------------------------------- cache
def _decode(rel: str) -> np.ndarray:
    from PIL import Image

    with Image.open(DATA_ROOT / rel) as image:
        rgb = image.convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    rgb = rgb.resize((CACHE_HW[1], CACHE_HW[0]), resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.uint8)
    if array.shape != (CACHE_HW[0], CACHE_HW[1], 3):
        raise ValueError(f"bad decoded shape {array.shape} for {rel}")
    return array


def build_cache(out_dir: Path, workers: int) -> None:
    from .data import load_retrieval_split

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    labels: list[np.ndarray] = []
    split_names: list[str] = []
    names = None
    for split_name, split_dir in SPLITS.items():
        split = load_retrieval_split(split_dir)
        names = split.attribute_names if names is None else names
        if split.attribute_names != names:
            raise ValueError("attribute order differs between splits")
        keep = np.asarray([p not in EXCLUDED for p in split.image_paths])
        paths.extend(np.asarray(split.image_paths)[keep].tolist())
        labels.append(split.gallery_attributes[keep])
        split_names.extend([split_name] * int(keep.sum()))
    label_array = np.concatenate(labels).astype(np.uint8)
    images = np.lib.format.open_memmap(out_dir / "images_u8_256x128.npy", mode="w+", dtype=np.uint8, shape=(len(paths), *CACHE_HW, 3))
    started = time.time()
    with Pool(workers) as pool:
        for index, array in enumerate(pool.imap(_decode, paths, chunksize=256)):
            images[index] = array
            if index % 20000 == 0:
                print(json.dumps({"cached": index, "elapsed_s": round(time.time() - started, 1)}), flush=True)
    images.flush()
    np.savez(
        out_dir / "meta.npz",
        image_paths=np.asarray(paths),
        labels=label_array,
        split=np.asarray(split_names),
        sources=np.asarray([p.split("/", 1)[0] for p in paths]),
        attribute_names=np.asarray(names),
    )
    print(json.dumps({"status": "CACHE_DONE", "rows": len(paths), "elapsed_s": round(time.time() - started, 1)}), flush=True)


# ----------------------------------------------------------------------------- model
def _build_model(key: str, pretrained: bool = True, drop_path_rate: float = 0.0, head: str = "linear", moe_experts: int = 4):
    import timm
    import torch

    spec = BACKBONES[key]
    kwargs: dict[str, Any] = {"num_classes": 0}
    if spec.get("arch") != "convnext":  # ConvNeXt is resolution-agnostic and takes no img_size
        kwargs["img_size"] = spec["img_hw"]
    if drop_path_rate:
        kwargs["drop_path_rate"] = drop_path_rate
    if pretrained:
        kwargs["pretrained"] = True
        kwargs["pretrained_cfg_overlay"] = {"file": str(ROOT / spec["file"])}
    backbone = timm.create_model(spec["timm"], **kwargs)

    class AttributeNet(torch.nn.Module):
        def __init__(self, backbone, dim: int):
            super().__init__()
            self.backbone = backbone
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(dim, 40)

        def forward(self, x):
            return self.head(self.dropout(self.backbone(x)))

    class TokenHeadNet(torch.nn.Module):
        """Heads that also read the patch tokens. Every variant keeps a linear head on the backbone's own pooled
        feature (head.base: same prior-bias init as AttributeNet) and adds a residual path that starts at zero:
          cls_avg - linear on [pooled, mean of patch tokens]
          attrq   - one learned query per attribute attends over the patch tokens (per-attribute spatial pooling)
          moe     - K expert linear heads on the pooled feature mixed by a softmax gate (mixture of experts)"""

        def __init__(self, backbone, dim: int, kind: str, experts: int):
            super().__init__()
            self.backbone = backbone
            self.kind = kind
            self.npt = int(getattr(backbone, "num_prefix_tokens", 1))
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Module()
            self.head.base = torch.nn.Linear(dim, 40)
            if kind == "cls_avg":
                self.head.extra = torch.nn.Linear(dim, 40)
                torch.nn.init.zeros_(self.head.extra.weight); torch.nn.init.zeros_(self.head.extra.bias)
            elif kind == "attrq":
                d = 256
                self.head.q = torch.nn.Parameter(torch.randn(40, d) * 0.02)
                self.head.k = torch.nn.Linear(dim, d, bias=False)
                self.head.out = torch.nn.Parameter(torch.zeros(40, dim))
            elif kind == "moe":
                self.head.experts = torch.nn.Linear(dim, 40 * experts)
                self.head.gate = torch.nn.Linear(dim, experts)
                torch.nn.init.zeros_(self.head.experts.weight); torch.nn.init.zeros_(self.head.experts.bias)
                self.experts = experts
            else:
                raise ValueError(f"unknown head {kind}")
            # train() initialises model.head.bias / model.head.weight (prior bias): point them at the base linear
            self.head.weight = self.head.base.weight
            self.head.bias = self.head.base.bias

        def forward(self, x):
            feats = self.backbone.forward_features(x)
            pooled = self.dropout(self.backbone.forward_head(feats, pre_logits=True))
            logits = self.head.base(pooled)
            if self.kind == "cls_avg":
                logits = logits + self.head.extra(self.dropout(feats[:, self.npt:].mean(1)))
            elif self.kind == "attrq":
                patches = feats[:, self.npt:]
                att = torch.softmax(torch.einsum("qd,bpd->bqp", self.head.q, self.head.k(patches)) / 16.0, dim=-1)
                z = torch.einsum("bqp,bpc->bqc", att, patches)
                logits = logits + (self.dropout(z) * self.head.out).sum(-1)
            else:
                e = self.head.experts(pooled).view(pooled.shape[0], self.experts, 40)
                g = torch.softmax(self.head.gate(pooled), dim=-1)
                logits = logits + torch.einsum("bk,bka->ba", g, e)
            return logits

    model = AttributeNet(backbone, backbone.num_features) if head == "linear" else TokenHeadNet(backbone, backbone.num_features, head, moe_experts)
    cfg = backbone.pretrained_cfg
    return model, tuple(cfg["mean"]), tuple(cfg["std"])


def _convnext_layer(name: str) -> int:
    """Layer-wise decay depth of a ConvNeXt parameter (0 = stem ... 12 = last stage), as in the ConvNeXt recipe:
    stages 0/1 -> 1/2, stage 2 in groups of three blocks -> 3..11, stage 3 -> 12; downsample layers join their stage."""
    parts = name.split(".")
    if parts[1] == "stem":
        return 0
    if parts[1] == "stages":
        stage = int(parts[2])
        if stage < 2:
            return stage + 1
        if stage == 2:
            return 3 + int(parts[4]) // 3 if parts[3] == "blocks" else 3
        return 12
    return 12  # backbone head norm


def _param_groups(model, lr: float, head_lr: float, llrd: float, weight_decay: float):
    convnext = hasattr(model.backbone, "stages")
    blocks = 12 if convnext else len(model.backbone.blocks)
    groups: dict[tuple[float, float], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            scale_lr = head_lr
        elif convnext:
            scale_lr = lr * llrd ** (blocks - _convnext_layer(name))
        elif name.startswith("backbone.blocks."):
            layer = int(name.split(".")[2])
            scale_lr = lr * llrd ** (blocks - 1 - layer)
        elif any(name.startswith("backbone." + p) for p in ("patch_embed", "cls_token", "pos_embed", "reg_token", "norm_pre")):
            scale_lr = lr * llrd ** blocks
        else:  # final norm, attention pooling head, fc_norm
            scale_lr = lr
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
        key = (scale_lr, 0.0 if no_decay else weight_decay)
        groups.setdefault(key, []).append(param)
    return [{"params": ps, "lr": k[0], "base_lr": k[0], "weight_decay": k[1]} for k, ps in groups.items()]


def _label_mask_v2(labels: np.ndarray, sources: np.ndarray, names: list[str]) -> np.ndarray:
    """Per-row loss mask from each source's labelling policy (data audit 2026-10-12); fixed vocabulary rules plus each row's
    own labels, no validation-label statistics:
    - Market has no upper Brown/Orange/Pink and no lower Orange/Red labels (always 0): masked on every Market row; the
      previous audit's hypothesized ambiguity pairs (not a documented official mapping) are upper Red~Pink,
      Yellow~Orange, Other~Brown and lower Pink~Red, Yellow~Orange. When the first bit is positive, both are masked;
    - PETA has no 'Other' colour (upper/lower): masked on every PETA row; PETA rows whose LowerBody type bits do not sum to
      one (UPAR conversion artefacts) have both type bits masked;
    - rows with no hair-length label and no hat (unannotated hair, mostly PA100k) have the three hair bits masked."""
    idx = {n: i for i, n in enumerate(names)}
    mask = np.ones_like(labels, dtype=np.float32)
    mk = np.asarray(sources) == "Market1501"
    pe = np.asarray(sources) == "PETA"
    for n in ("UpperBody-Color-Brown", "UpperBody-Color-Orange", "UpperBody-Color-Pink", "LowerBody-Color-Orange", "LowerBody-Color-Red"):
        mask[mk, idx[n]] = 0.0
    for a, b in (("UpperBody-Color-Red", "UpperBody-Color-Pink"), ("UpperBody-Color-Yellow", "UpperBody-Color-Orange"),
                 ("UpperBody-Color-Other", "UpperBody-Color-Brown"), ("LowerBody-Color-Pink", "LowerBody-Color-Red"),
                 ("LowerBody-Color-Yellow", "LowerBody-Color-Orange")):
        rows = mk & (labels[:, idx[a]] > 0.5)
        mask[rows, idx[a]] = 0.0
        mask[rows, idx[b]] = 0.0
    for n in ("UpperBody-Color-Other", "LowerBody-Color-Other"):
        mask[pe, idx[n]] = 0.0
    t, sk = idx["LowerBody-Type-Trousers&Shorts"], idx["LowerBody-Type-Skirt&Dress"]
    bad = pe & ((labels[:, t] + labels[:, sk]) != 1)
    mask[bad, t] = 0.0
    mask[bad, sk] = 0.0
    hair = [idx["Hair-Length-Short"], idx["Hair-Length-Long"], idx["Hair-Length-Bald"]]
    nohair = (labels[:, hair].sum(axis=1) == 0) & (labels[:, idx["Accessory-Hat"]] < 0.5)
    mask[np.ix_(np.flatnonzero(nohair), hair)] = 0.0
    return mask


def _focal_factor(logits, target, gamma: float, log_pw=None):
    """(1 - p_t)^gamma with p_t the probability of the (possibly smoothed) target. p is the DEPLOYED probability
    sigmoid(logit - log pos_weight) (review 10/12: on the raw training logit the factor biases deployed probabilities by
    an amount that depends on each attribute's pos_weight)."""
    import torch

    p = torch.sigmoid(logits.float() - (log_pw if log_pw is not None else 0.0))
    p_t = p * target + (1 - p) * (1 - target)
    return (1 - p_t).clamp(min=0).pow(gamma)


def _rank_loss(logits, labels, log_pw, tau_target: float = 1.0, tau_pred: float = 1.0, nll_lambda: float = 0.25, row_weight=None):
    """Listwise retrieval loss matched to the deployed ranker: every image's label vector is a query over the batch; the
    distance is the ranker's (1-l)*L1 + l*Bernoulli NLL on the deployed (prior-corrected) probabilities, and the target
    distribution over the batch decays with the attribute Hamming distance (mADM also rewards partial matches)."""
    import torch

    p = torch.sigmoid(logits.float() - log_pw).clamp(1e-6, 1 - 1e-6)
    q = labels.float()
    l1 = q @ (1 - p).T + (1 - q) @ p.T
    nll = -(q @ torch.log(p).T + (1 - q) @ torch.log1p(-p).T)
    d = (1 - nll_lambda) * l1 + nll_lambda * nll
    ham = q @ (1 - q).T + (1 - q) @ q.T
    target = torch.softmax(-ham / tau_target, dim=1)
    per_query = -(target * torch.log_softmax(-d / tau_pred, dim=1)).sum(1)
    if row_weight is None:
        return per_query.mean()
    # "rank_query_balance": weight each query row by count^-beta of its label vector within its source (mADM averages
    # distinct semantic queries; the frequent PA100k query is ~110x over-weighted when rows are averaged)
    return (per_query * row_weight).sum() / row_weight.sum().clamp(min=1e-12)


def _rank_term(logits, labels, sources, row_weight, log_pw, cfg):
    """The configured rank loss over one list set: all rows as one list, or (rank_group "source") one list per source,
    weighted by its rows; lists under 4 rows are skipped."""
    import torch

    tau_t, tau_p = float(cfg.get("rank_tau_target", 1.0)), float(cfg.get("rank_tau_pred", 1.0))
    kind = cfg.get("rank_loss_kind", "list_ce")
    if kind not in ("list_ce", "soft_adm"):
        raise ValueError(f"unknown rank_loss_kind {kind}")
    if kind == "soft_adm" and (cfg.get("label_mask") or cfg.get("partswap_p", 0) or cfg.get("mixup_p", 0)):
        raise ValueError("soft_adm requires complete unmixed labels; masked/partswap/mixup recipes need a separate design")

    def term(z, y, w):
        if kind == "soft_adm":
            from .soft_adm import soft_adm_loss
            return soft_adm_loss(z, y, log_pw, temperature=float(cfg.get("rank_adm_temperature", 1.0)), row_weight=w)
        return _rank_loss(z, y, log_pw, tau_t, tau_p, row_weight=w)

    if sources is None:
        return term(logits, labels, row_weight)
    rl = 0.0
    for c in torch.unique(sources).tolist():
        sel = sources == c
        k = int(sel.sum())
        if k >= 4:
            rl = rl + (k / sources.numel()) * term(logits[sel], labels[sel], None if row_weight is None else row_weight[sel])
    return rl


def _gpu_augment(batch_u8, *, train: bool, out_hw, mean, std, generator=None, scale_min: float = 0.85, erasing_p: float = 0.0,
                 crop_top_frac: float | None = None, lowres_p: float = 0.0, lowres_min: float = 0.35, sat_jitter: float = 0.0):
    import torch
    import torch.nn.functional as F

    x = batch_u8.permute(0, 3, 1, 2).float().div_(255.0)
    if crop_top_frac:  # fixed top-of-image region (head member), cut before any resize
        x = x[:, :, : int(round(x.shape[-2] * crop_top_frac)), :]
    n = x.shape[0]
    if tuple(x.shape[-2:]) != tuple(out_hw):  # same antialiased resize for train and inference
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False, antialias=True)
    if train:
        flip = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        scale = torch.empty(n, device=x.device).uniform_(scale_min, 1.0)
        aspect = torch.empty(n, device=x.device).uniform_(0.92, 1.08)
        sx = scale * aspect.sqrt()
        sy = scale / aspect.sqrt()
        tx = (torch.rand(n, device=x.device) * 2 - 1) * (1 - sx).clamp(min=0)
        ty = (torch.rand(n, device=x.device) * 2 - 1) * (1 - sy).clamp(min=0)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (n, 3, out_hw[0], out_hw[1]), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        brightness = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        contrast = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        mean_lum = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean_lum) * contrast + mean_lum) * brightness
        if sat_jitter > 0:  # saturation only (hue, hence colour attributes, unchanged)
            gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
            sat = torch.empty(n, 1, 1, 1, device=x.device).uniform_(1 - sat_jitter, 1 + sat_jitter)
            x = gray + (x - gray) * sat
        x = x.clamp_(0, 1)
        if erasing_p > 0:  # random erasing of one rectangle (area 2-20%), filled with per-image mean
            h, w = x.shape[-2:]
            area = torch.empty(n, device=x.device).uniform_(0.02, 0.2)
            ratio = torch.exp(torch.empty(n, device=x.device).uniform_(-1.2, 1.2))
            eh = (torch.sqrt(area * ratio) * h).clamp(1, h).long()
            ew = (torch.sqrt(area / ratio) * w).clamp(1, w).long()
            top = (torch.rand(n, device=x.device) * (h - eh + 1)).long()
            left = (torch.rand(n, device=x.device) * (w - ew + 1)).long()
            rows = torch.arange(h, device=x.device).view(1, h, 1)
            cols = torch.arange(w, device=x.device).view(1, 1, w)
            box = (rows >= top.view(n, 1, 1)) & (rows < (top + eh).view(n, 1, 1)) & (cols >= left.view(n, 1, 1)) & (cols < (left + ew).view(n, 1, 1))
            box = box & (torch.rand(n, device=x.device) < erasing_p).view(n, 1, 1)
            x = torch.where(box.unsqueeze(1), x.mean(dim=(2, 3), keepdim=True), x)
        if lowres_p > 0:  # resolution degradation: antialiased downscale by s ~ U(lowres_min, 1), bilinear upscale back
            h, w = x.shape[-2:]
            pick = torch.rand(n, device=x.device) < lowres_p
            if bool(pick.any()):
                s_ = float(torch.empty(1).uniform_(lowres_min, 1.0))
                small = F.interpolate(x[pick], size=(max(8, int(h * s_)), max(4, int(w * s_))), mode="bilinear", align_corners=False, antialias=True)
                x = x.clone()
                x[pick] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ----------------------------------------------------------------------------- train
def _fit_rows(meta, fold: str, include_val: bool) -> np.ndarray:
    split = meta["split"]
    sources = meta["sources"]
    use = split == "train"
    if include_val:
        use = use | (split == "val")
    if fold != "all":
        use = use & (sources != fold)
    return np.flatnonzero(use)


# ----------------------------------------------------------------------------- opt-in checkpoint / resume
# "ckpt_every_steps": N > 0 writes <output_dir>/resume_ckpt.pt after every N-th optimizer step and after the last step
# (one file per run, replaced atomically: tmp + fsync + os.replace), and a restarted run whose config matches (up to the
# keys below) continues from it: model, AdamW state, every EMA state, step/history and the torch CPU/CUDA RNG states;
# the data order is the seed's sample_index, so the producer simply starts at the saved step. The file is deleted once
# the final outputs are saved. "ckpt_min_free_gb" (default 10): a checkpoint is skipped (training goes on, the previous
# checkpoint is kept) when writing it would leave less free disk. Absent / 0 = the unchanged code path: nothing is read,
# written or deleted.
_CKPT_NAME = "resume_ckpt.pt"
_CKPT_FREE_KEYS = ("output_dir", "ckpt_every_steps", "ckpt_min_free_gb", "save_weights", "save_raw_weights")  # output-only


def _ckpt_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in json.loads(json.dumps(cfg)).items() if k not in _CKPT_FREE_KEYS}


def _ckpt_digest(rows: np.ndarray, sample_index: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(sample_index, dtype=np.int64).tobytes())
    return h.hexdigest()


def _ckpt_nbytes(obj) -> int:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_ckpt_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_ckpt_nbytes(v) for v in obj)
    return 0


def _ckpt_save(path: Path, payload: dict[str, Any], min_free_gb: float) -> None:
    """Atomic single-file checkpoint; a failure (disk full, IO error) is logged and training continues."""
    try:
        _ckpt_write(path, payload, min_free_gb)
    finally:
        # torch.save leaves cyclic garbage (its local pickler class/closures) that holds every saved CUDA storage until
        # the next full collection; without this the AdamW state of the last checkpoint survives `del opt` and the
        # end-of-run predictions would need that memory on top (measured: +0.12 GiB peak on the PE-B test)
        import gc

        gc.collect()


def _ckpt_write(path: Path, payload: dict[str, Any], min_free_gb: float) -> None:
    import os
    import shutil

    import torch

    tmp = path.with_name(path.name + ".tmp")
    need = _ckpt_nbytes(payload)
    free = shutil.disk_usage(path.parent).free
    if free - need < min_free_gb * 2**30:
        print(json.dumps({"event": "ckpt_skipped_low_disk", "step": payload["step"], "need_gib": round(need / 2**30, 2),
                          "free_gib": round(free / 2**30, 2), "min_free_gb": min_free_gb}), flush=True)
        return
    t0 = time.time()
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception as exc:  # never lose a healthy run to a checkpoint failure
        tmp.unlink(missing_ok=True)
        print(json.dumps({"event": "ckpt_failed", "step": payload["step"], "error": repr(exc)[:300]}), flush=True)
        return
    print(json.dumps({"event": "ckpt_saved", "step": payload["step"], "gib": round(path.stat().st_size / 2**30, 3),
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def _ckpt_load(path: Path, cfg: dict[str, Any], steps: int, lp_steps: int, digest: str) -> dict[str, Any] | None:
    """The saved state of this exact run, or None when there is no checkpoint; any mismatch raises (fail closed: a
    checkpoint of a different config is never resumed and never silently overwritten)."""
    import torch

    path.with_name(path.name + ".tmp").unlink(missing_ok=True)  # an interrupted write; the previous file is intact
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if ckpt.get("format") != 1:
        raise ValueError(f"{path}: unknown checkpoint format {ckpt.get('format')}")
    saved, now = _ckpt_cfg(ckpt["cfg"]), _ckpt_cfg(cfg)
    if saved != now:
        diff = sorted(k for k in set(saved) | set(now) if saved.get(k, "<absent>") != now.get(k, "<absent>"))
        raise ValueError(f"{path} was written by a different config (keys {diff}); delete it to start this run from step 0")
    if (ckpt["steps"], ckpt["lp_steps"], ckpt["digest"]) != (steps, lp_steps, digest):
        raise ValueError(f"{path}: schedule or data order differs (steps {ckpt['steps']} vs {steps}, lp {ckpt['lp_steps']} vs "
                         f"{lp_steps}, digest match {ckpt['digest'] == digest}); delete it to start from step 0")
    if not 0 < int(ckpt["step"]) <= steps:
        raise ValueError(f"{path}: bad step {ckpt['step']}")
    return ckpt


def train(cfg: dict[str, Any]) -> None:
    import torch
    from safetensors.torch import save_file

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda")
    cache = Path(cfg["cache_dir"])
    images = np.load(cache / cfg.get("cache_file", "images_u8_256x128.npy"), mmap_mode="r")  # cache variants: other sizes / letterbox
    meta = dict(np.load(cache / "meta.npz"))
    labels = meta["labels"].astype(np.float32)
    rows = _fit_rows(meta, cfg["fold"], cfg["include_val"])
    row_sources = meta["sources"][rows]
    present = [s for s in SOURCES if np.any(row_sources == s)]
    # domain-balanced sampling and positive rate (each present source has equal mass)
    weights = np.zeros(len(rows), dtype=np.float64)
    for s in present:
        mask = row_sources == s
        weights[mask] = 1.0 / mask.sum() / len(present)
    # "label_mask": "unannotated" ignores (source, attribute) pairs that are never positive anywhere in that source
    # (e.g. Market-1501 has no brown/orange/pink upper-body labels; UPAR fills them with 0 = false negatives)
    label_mask_all = np.ones_like(labels)
    if cfg.get("label_mask") == "unannotated":
        for s in SOURCES:
            in_s = meta["sources"] == s
            never = labels[in_s].sum(axis=0) == 0
            label_mask_all[np.ix_(np.flatnonzero(in_s), np.flatnonzero(never))] = 0.0
    elif cfg.get("label_mask") == "v2":
        label_mask_all = _label_mask_v2(labels, meta["sources"], [str(n) for n in meta["attribute_names"]])
    elif cfg.get("label_mask") == "v2_restore_market_upper_other_v1":
        from .label_mask_restore import restore_market_upper_other, validate_config

        validate_config(cfg)
        names = [str(n) for n in meta["attribute_names"]]
        label_mask_all = _label_mask_v2(labels, meta["sources"], names)
        label_mask_all[rows] = restore_market_upper_other(
            label_mask_all[rows], labels[rows], row_sources, names
        )
    elif cfg.get("label_mask"):
        raise ValueError(f"unknown label_mask {cfg['label_mask']}")
    use_mask = bool(cfg.get("label_mask"))
    partswap_p = float(cfg.get("partswap_p", 0.0))
    _names = [str(n) for n in meta["attribute_names"]]
    lower_idx = torch.tensor([i for i, n in enumerate(_names) if n.startswith("LowerBody-")], device=device)
    ambiguous_idx = torch.tensor([i for i, n in enumerate(_names) if n.startswith(("Age-", "Gender-")) or n in ("Accessory-Backpack", "Accessory-Bag")], device=device)
    m_rows = label_mask_all[rows]
    pos_rate = (labels[rows] * m_rows * weights[:, None]).sum(axis=0) / np.maximum((m_rows * weights[:, None]).sum(axis=0), 1e-12)
    pos_weight = np.clip((1.0 - pos_rate) / np.maximum(pos_rate, 1e-6), 1.0, cfg["pos_weight_cap"]).astype(np.float32)

    source_prior_offsets_all = None
    if cfg.get("source_prior_offset_bce"):
        from .source_prior_offset_bce import (
            fit_source_prior_offsets, source_prior_bce_part,
            validate_config as validate_source_prior_config,
        )

        validate_source_prior_config(cfg)
        source_prior_fit, source_prior_report = fit_source_prior_offsets(
            labels[rows], row_sources, meta["split"][rows], _names,
            fold=cfg["fold"], include_val=cfg["include_val"],
            reference_prior=pos_rate, pos_weight=pos_weight,
        )
        source_prior_offsets_all = np.full_like(labels, np.nan, dtype=np.float32)
        source_prior_offsets_all[rows] = source_prior_fit
        (out / "source_prior_offset_fit_report.json").write_text(json.dumps(source_prior_report, indent=2))
        print(json.dumps({"event": "source_prior_offset_fit_only", "recipe": cfg["source_prior_offset_bce"],
                          "fit_rows": len(rows), "denominator": "unchanged_full_batch_N_times_40"}), flush=True)

    profile_weights_all = None
    if cfg.get("bce_profile_balance"):
        from .profile_balanced_bce import fit_cell_weights, validate_config as validate_profile_config, weighted_bce_part

        validate_profile_config(cfg)
        profile_fit, profile_report = fit_cell_weights(
            labels[rows], row_sources, meta["split"][rows],
            fold=cfg["fold"], include_val=cfg["include_val"],
        )
        profile_weights_all = np.zeros_like(labels, dtype=np.float32)
        profile_weights_all[rows] = profile_fit
        (out / "profile_bce_fit_report.json").write_text(json.dumps(profile_report, indent=2))
        print(json.dumps({"event": "profile_bce_fit_only", "recipe": cfg["bce_profile_balance"],
                          "fit_rows": len(rows), "denominator": "unchanged_full_batch_N_times_40"}), flush=True)

    coarse_active_all, coarse_pairs = None, ()
    if cfg.get("coarse_positive"):
        from .coarse_label_loss import fit_positive_events, positive_union_loss, validate_config

        validate_config(cfg)
        coarse_pairs, coarse_fit = fit_positive_events(labels[rows], row_sources, _names)
        # No held-source label statistics. If Market is held out, leave the training graph exactly unchanged.
        if coarse_fit.any():
            coarse_active_all = np.zeros((len(labels), len(coarse_pairs)), dtype=bool)
            coarse_active_all[rows] = coarse_fit
        print(json.dumps({"event": "coarse_positive_fit_only", "recipe": cfg["coarse_positive"],
                          "positive_counts": coarse_fit.sum(0).tolist(),
                          "active": coarse_active_all is not None, "normalization": "unchanged_mask_v2_full_batch"}), flush=True)

    model, mean, std = _build_model(cfg["backbone"], drop_path_rate=float(cfg.get("drop_path_rate", 0.0)), head=cfg.get("head", "linear"))
    if cfg.get("grad_checkpoint"):  # recompute block activations in backward: ~1/3 of the memory, same numbers
        model.backbone.set_grad_checkpointing(True)
    model.to(device)
    with torch.no_grad():
        prior_logit = np.log(np.clip(pos_rate, 1e-4, 1 - 1e-4) / np.clip(1 - pos_rate, 1e-4, 1)) + np.log(pos_weight)
        model.head.bias.copy_(torch.tensor(prior_logit, dtype=torch.float32))
        model.head.weight.mul_(0.1)
    if cfg.get("init_from"):
        # "branch" fine-tuning (2026-10-15): start from an existing run of the SAME backbone/head/fold/fit rows (so the result
        # shares its basin and can be weight-souped with it) and train a short schedule with a different recipe
        from safetensors.torch import load_file
        icfg = json.loads((Path(cfg["init_from"]) / "config.json").read_text())
        for key in ("backbone", "fold", "include_val"):
            if icfg.get(key) != cfg.get(key):
                raise ValueError(f"init_from {key} differs: {icfg.get(key)} vs {cfg.get(key)}")
        if icfg.get("head", "linear") != cfg.get("head", "linear"):
            raise ValueError("init_from head differs")
        model.load_state_dict({k: v.to(device) for k, v in load_file(str(Path(cfg["init_from"]) / "model.safetensors")).items()})
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    kd_alpha = float(cfg.get("kd_alpha", 0.0))
    teacher_prob = None
    if cfg.get("kd_teacher_logits"):
        # mean of teacher sigmoid(raw logit); teachers were trained with the same pos_weight convention
        teacher_prob = np.mean([1.0 / (1.0 + np.exp(-np.load(path)["logits"].astype(np.float32))) for path in cfg["kd_teacher_logits"]], axis=0).astype(np.float32)
        if teacher_prob.shape != labels.shape:
            raise ValueError("teacher logits must cover every cache row")
    kd_criterion = torch.nn.BCEWithLogitsLoss()
    rank_alpha = float(cfg.get("rank_alpha", 0.0))
    # "rank_scope": "micro" (default, old behaviour) ranks within each grad_accum micro-batch (~22 images per list for
    # batch 128 / accum 4 / per-source lists); "batch" keeps the micro-batch graphs (cheap under grad checkpointing) and
    # computes the rank loss once over the whole optimizer batch (~43 per list), one backward per step (10/16 review)
    rank_scope = cfg.get("rank_scope", "micro")
    if rank_scope not in ("micro", "batch"):
        raise ValueError(f"unknown rank_scope {rank_scope}")
    focal_gamma = float(cfg.get("focal_gamma", 0.0))
    log_pw = torch.log(torch.tensor(pos_weight, device=device, dtype=torch.float32))
    # online "function matching" KD: fold-clean teachers see the same augmented (and mixed) image as the student
    teachers = []
    for run_dir in cfg.get("online_teachers", []):
        tcfg = json.loads((Path(run_dir) / "config.json").read_text())
        if tcfg["fold"] != cfg["fold"] or bool(tcfg["include_val"]) != bool(cfg["include_val"]):
            raise ValueError(f"teacher {run_dir} fit boundary differs from the student")
        from safetensors.torch import load_file

        tmodel, tmean, tstd = _build_model(tcfg["backbone"], pretrained=False, head=tcfg.get("head", "linear"))
        tmodel.load_state_dict(load_file(str(Path(run_dir) / "model.safetensors")))
        tmodel.to(device).eval().requires_grad_(False)
        teachers.append((tmodel, BACKBONES[tcfg["backbone"]]["img_hw"], tmean, tstd))
    mixup_p = float(cfg.get("mixup_p", 0.0))
    mixup_beta = torch.distributions.Beta(float(cfg.get("mixup_alpha", 1.0)), float(cfg.get("mixup_alpha", 1.0)))
    batch = cfg["batch_size"]
    total_samples = int(min(cfg["sample_budget"], cfg["max_epochs"] * len(rows)))
    steps = total_samples // batch
    lp_steps = int(steps * cfg["lp_fraction"])
    warm_steps = int(steps * cfg["warmup_fraction"])
    opt = torch.optim.AdamW(_param_groups(model, cfg["lr"], cfg["head_lr"], cfg["llrd"], cfg["weight_decay"]), betas=(0.9, 0.999))
    out_hw = BACKBONES[cfg["backbone"]]["img_hw"]
    crop = BACKBONES[cfg["backbone"]].get("crop_top_frac")

    if cfg.get("sampling") == "cycle":
        # same source mix and step count, but each source is drawn without replacement and reshuffled when exhausted:
        # with replacement, ~22% of PA100k rows are never drawn in an all-trainval fit (133k draws over 89k rows)
        pick = rng.choice(len(present), size=steps * batch, p=np.full(len(present), 1.0 / len(present)))
        sample_index = np.empty(steps * batch, dtype=np.int64)
        for k, s in enumerate(present):
            idx_s, need = np.flatnonzero(row_sources == s), int((pick == k).sum())
            sample_index[pick == k] = np.concatenate([rng.permutation(idx_s) for _ in range(-(-need // len(idx_s)))])[:need]
    elif cfg.get("sampling") not in (None, "replace"):
        raise ValueError(f"unknown sampling {cfg['sampling']}")
    else:
        sample_index = rng.choice(len(rows), size=steps * batch, replace=True, p=weights / weights.sum())
    queue: Queue = Queue(maxsize=6)
    # "rank_group": "source" -> rank-loss lists hold one source each (like the single-domain gallery at evaluation); the
    # batch is ordered by source so micro-batches are mostly single-source, and each micro-batch is split by source.
    # Without it, lists follow cache order: ~75% single-source in 2-source LODO folds but ~26% in all-trainval fits.
    rank_group = cfg.get("rank_group")
    if rank_group not in (None, "source"):
        raise ValueError(f"unknown rank_group {rank_group}")
    src_codes = np.unique(meta["sources"], return_inverse=True)[1].astype(np.int64) if rank_group == "source" else None
    qbal = cfg.get("rank_query_balance")
    qw_all = None
    if qbal:  # true = query-uniform (beta 1); a number = the exponent beta in count^-beta
        beta = 1.0 if qbal is True else float(qbal)
        qw_all = np.ones(len(labels), dtype=np.float32)
        for s_name in present:
            r_s = rows[row_sources == s_name]
            _, inv, cnt = np.unique((labels[r_s] > 0.5).astype(np.uint8), axis=0, return_inverse=True, return_counts=True)
            qw_all[r_s] = cnt[inv.reshape(-1)].astype(np.float32) ** (-beta)
    ckpt_every = int(cfg.get("ckpt_every_steps", 0) or 0)
    ckpt_path = out / _CKPT_NAME
    resume = None
    start_step = 0
    if ckpt_every > 0:
        ckpt_digest = _ckpt_digest(rows, sample_index)
        resume = _ckpt_load(ckpt_path, cfg, steps, lp_steps, ckpt_digest)
        if resume is not None:
            model.load_state_dict(resume["model"])
            opt.load_state_dict(resume["opt"])
            for state in opt.state.values():  # CPU-resident entries (AdamW 'step') would otherwise pin the mmapped file
                for k, v in list(state.items()):
                    if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                        state[k] = v.clone()
            start_step = int(resume["step"])

    def producer() -> None:
        for step in range(start_step, steps):
            idx = np.sort(rows[sample_index[step * batch : (step + 1) * batch]])
            if src_codes is not None:
                idx = idx[np.argsort(src_codes[idx], kind="stable")]
            x = torch.from_numpy(np.ascontiguousarray(images[idx])).pin_memory()
            y = torch.from_numpy(labels[idx]).pin_memory()
            t = torch.from_numpy(teacher_prob[idx]).pin_memory() if teacher_prob is not None else None
            m = torch.from_numpy(label_mask_all[idx]).pin_memory() if use_mask else None
            s = torch.from_numpy(src_codes[idx]).pin_memory() if src_codes is not None else None
            w = torch.from_numpy(qw_all[idx]).pin_memory() if qw_all is not None else None
            c = torch.from_numpy(coarse_active_all[idx]).pin_memory() if coarse_active_all is not None else None
            b = torch.from_numpy(profile_weights_all[idx]).pin_memory() if profile_weights_all is not None else None
            if source_prior_offsets_all is None:
                queue.put((x, y, t, m, s, w, c, b))
            else:
                source_prior_offset = torch.from_numpy(source_prior_offsets_all[idx]).pin_memory()
                queue.put((x, y, t, m, s, w, c, b, source_prior_offset))
        queue.put(None)

    threading.Thread(target=producer, daemon=True).start()
    # EMA of all weights (float32 copies), evaluated and saved next to the raw model. "ema_decay" writes <out>_ema;
    # "ema_decays" writes one <out>_ema<decay> directory per decay from the same trajectory.
    ema_targets: list[tuple[float, Path]] = []
    if float(cfg.get("ema_decay", 0.0)) > 0:
        ema_targets.append((float(cfg["ema_decay"]), Path(str(out) + "_ema")))
    for d in cfg.get("ema_decays", []):
        ema_targets.append((float(d), Path(f"{out}_ema{d}")))
    ema_states = [{k: v.detach().float().clone() for k, v in model.state_dict().items()} for _ in ema_targets]
    history = []
    started = time.time()
    model.train()
    step = 0
    running = 0.0
    if resume is not None:
        if len(resume["ema"]) != len(ema_states):
            raise ValueError(f"{ckpt_path}: {len(resume['ema'])} EMA states saved, {len(ema_states)} configured")
        with torch.no_grad():
            for ema_state, saved_ema in zip(ema_states, resume["ema"]):
                if set(saved_ema) != set(ema_state):
                    raise ValueError(f"{ckpt_path}: EMA keys differ from the model")
                for k, v in ema_state.items():
                    v.copy_(saved_ema[k])
        step, running, history = int(resume["step"]), float(resume["running"]), list(resume["history"])
        started = time.time() - float(resume["train_seconds"])
        torch.set_rng_state(resume["rng_cpu"])
        torch.cuda.set_rng_state(resume["rng_cuda"], device)
        if step >= lp_steps and cfg.get("save_lp_state") and not (Path(f"{out}_lp") / "model.safetensors").exists():
            print(json.dumps({"event": "resume_lp_state_missing", "hint": "recreate the _lp endpoint with lp_only"}), flush=True)
        print(json.dumps({"event": "resumed", "step": step, "of": steps, "lp_steps": lp_steps, "ckpt": str(ckpt_path)}), flush=True)
        resume = saved_ema = None  # drop every reference into the mmapped file (the next save replaces it)
        import gc

        gc.collect()
    while True:
        item = queue.get()
        if item is None:
            break
        source_prior_offset = None
        if source_prior_offsets_all is None:
            x, y, t, m, s_codes, q_w, coarse_active, bce_w = item
        else:
            x, y, t, m, s_codes, q_w, coarse_active, bce_w, source_prior_offset = item
            source_prior_offset = source_prior_offset.to(device, non_blocking=True)
        if bce_w is not None:
            bce_w = bce_w.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        if s_codes is not None:
            s_codes = s_codes.to(device, non_blocking=True)
        if q_w is not None:
            q_w = q_w.to(device, non_blocking=True)
        if coarse_active is not None:
            coarse_active = coarse_active.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        backbone_on = step >= lp_steps
        for group in opt.param_groups:
            is_head = group["base_lr"] == cfg["head_lr"]
            if step < lp_steps:
                factor = 1.0 if is_head else 0.0
            else:
                ft_step = step - lp_steps
                ft_total = max(1, steps - lp_steps)
                warm = min(1.0, (ft_step + 1) / max(1, warm_steps))
                factor = warm * 0.5 * (1 + math.cos(math.pi * ft_step / ft_total))
            group["lr"] = group["base_lr"] * factor
        if teachers:
            # augment once at cache resolution, optionally mix, then resize per network
            view = _gpu_augment(x, train=True, out_hw=CACHE_HW, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                                scale_min=float(cfg.get("aug_scale_min", 0.85)), erasing_p=float(cfg.get("random_erasing_p", 0.0)))
            if mixup_p > 0:
                lam = mixup_beta.sample((view.shape[0],)).to(device)
                lam = torch.where(torch.rand(view.shape[0], device=device) < mixup_p, torch.maximum(lam, 1 - lam), torch.ones_like(lam))
                perm = torch.randperm(view.shape[0], device=device)
                view = lam.view(-1, 1, 1, 1) * view + (1 - lam.view(-1, 1, 1, 1)) * view[perm]
                y = lam.view(-1, 1) * y + (1 - lam.view(-1, 1)) * y[perm]

            def _net_input(hw, m_, s_, crop_frac=None):
                v = view[:, :, : int(round(view.shape[-2] * crop_frac)), :] if crop_frac else view
                if tuple(v.shape[-2:]) != tuple(hw):
                    v = torch.nn.functional.interpolate(v, size=hw, mode="bilinear", align_corners=False, antialias=True)
                return (v - torch.tensor(m_, device=device).view(1, 3, 1, 1)) / torch.tensor(s_, device=device).view(1, 3, 1, 1)

            inputs = _net_input(out_hw, mean, std, crop)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                t = torch.stack([torch.sigmoid(tm(_net_input(thw, tme, tsd)).float()) for tm, thw, tme, tsd in teachers]).mean(0)
            t = t.clone()
        else:
            inputs = _gpu_augment(x, train=True, out_hw=out_hw, mean=mean, std=std, crop_top_frac=crop,
                                  scale_min=float(cfg.get("aug_scale_min", 0.85)), erasing_p=float(cfg.get("random_erasing_p", 0.0)),
                                  lowres_p=float(cfg.get("lowres_p", 0.0)), lowres_min=float(cfg.get("lowres_min", 0.35)),
                                  sat_jitter=float(cfg.get("sat_jitter", 0.0)))
        if partswap_p > 0:
            # synthetic part-swap composition: lower body from another image of the batch; lower-body labels follow it,
            # identity-level attributes (age, gender, bags) become ambiguous and are masked for the swapped samples
            pick = torch.rand(inputs.shape[0], device=device) < partswap_p
            if bool(pick.any()):
                rows_ = pick.nonzero().squeeze(1)
                perm = torch.randperm(inputs.shape[0], device=device)
                cut = int(inputs.shape[-2] * float(torch.empty(1).uniform_(0.45, 0.6)))
                inputs = inputs.clone(); inputs[rows_, :, cut:, :] = inputs[perm[rows_], :, cut:, :]
                y = y.clone(); y[rows_[:, None], lower_idx] = y[perm[rows_]][:, lower_idx]
                if t is not None:  # KD targets follow the swapped lower body too (review 10/12)
                    t = t.to(device, non_blocking=True).clone(); t[rows_[:, None], lower_idx] = t[perm[rows_]][:, lower_idx]
                m = (torch.ones_like(y) if m is None else m.to(device, non_blocking=True)).clone()
                m[rows_[:, None], lower_idx] = m[perm[rows_]][:, lower_idx]  # the label mask follows the swapped labels
                m[rows_[:, None], ambiguous_idx] = 0.0
        eps_ls = float(cfg.get("label_smoothing", 0.0))
        target = y * (1.0 - eps_ls) + 0.5 * eps_ls if eps_ls > 0 else y
        if m is not None:
            m = m.to(device, non_blocking=True)
        if t is not None:
            t = t.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        # gradient accumulation (grad_accum micro-batches of one batch): the same update as one full batch, because every
        # micro-batch loss is normalised by the full batch (mean losses weighted by their share, masked sums by the full mask)
        accum = int(cfg.get("grad_accum", 1))
        parts = torch.arange(inputs.shape[0], device=device).chunk(accum) if accum > 1 else [None]
        loss = 0.0
        batch_logits, pending = [], None
        for idx in parts:
            take = (lambda v: v) if idx is None else (lambda v: v[idx])
            share = 1.0 if idx is None else idx.numel() / inputs.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(take(inputs))
            if source_prior_offset is not None:
                part = source_prior_bce_part(
                    logits, take(target), criterion.pos_weight, take(source_prior_offset),
                    target.numel(), attribute_names=_names,
                )
            elif cfg.get("bce_profile_balance"):
                part = weighted_bce_part(logits, take(target), criterion.pos_weight, take(bce_w), target.numel())
            elif m is not None:
                per = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), take(target), pos_weight=criterion.pos_weight, reduction="none")
                if focal_gamma > 0:
                    per = per * _focal_factor(logits, take(target), focal_gamma, log_pw)
                part = (per * take(m)).sum() / m.sum().clamp(min=1.0)
            elif focal_gamma > 0:  # focal BCE (Specker: gamma ~1 improves mADM via calibration); gamma 0 = the plain criterion
                per = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), take(target), pos_weight=criterion.pos_weight, reduction="none")
                part = (per * _focal_factor(logits, take(target), focal_gamma, log_pw)).mean() * share
            else:
                part = criterion(logits.float(), take(target)) * share
            if cfg.get("coarse_positive") and coarse_active is not None:
                # Positive coarse observations only; all mask-v2 negative supervision and priors stay unchanged.
                part = part + positive_union_loss(logits, take(coarse_active), coarse_pairs,
                                                 criterion.pos_weight, m.sum().clamp(min=1.0))
            if t is not None:
                part = (1.0 - kd_alpha) * part + kd_alpha * kd_criterion(logits.float(), take(t)) * share
            if rank_alpha > 0 and rank_scope == "micro":  # retrieval-aligned auxiliary loss within the micro-batch
                rl = _rank_term(logits, take(y), take(s_codes) if s_codes is not None else None,
                                take(q_w) if q_w is not None else None, log_pw, cfg)
                part = part + rank_alpha * share * rl
            if rank_alpha > 0 and rank_scope == "batch":  # keep the graph; the rank loss over the whole batch follows
                batch_logits.append(logits)
                pending = part if pending is None else pending + part
            else:
                part.backward()
            loss = loss + part.detach()
        if rank_alpha > 0 and rank_scope == "batch":
            rl = _rank_term(torch.cat(batch_logits), y, s_codes, q_w, log_pw, cfg)
            total = pending + rank_alpha * rl
            total.backward()
            loss = loss + (rank_alpha * rl).detach()
            batch_logits, pending = [], None
        if backbone_on:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema_states:
            with torch.no_grad():
                current = model.state_dict()
                for (ema_decay, _), ema_state in zip(ema_targets, ema_states):
                    for k, v in current.items():
                        if v.dtype.is_floating_point:
                            ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                        else:
                            ema_state[k].copy_(v)
        running += float(loss.detach())
        step += 1
        if step == lp_steps and cfg.get("save_lp_state"):
            # end of the linear-probe stage (pretrained backbone + fitted head): the WiSE-FT interpolation endpoint
            lp_dir = Path(f"{out}_lp")
            lp_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}, str(lp_dir / "model.safetensors"))
            (lp_dir / "config.json").write_text(json.dumps(dict(cfg, output_dir=str(lp_dir), lp_of=str(out), pos_weight=pos_weight.tolist()), indent=1))
            if ckpt_every > 0:  # durable before any later checkpoint can claim the LP boundary was passed
                import os

                fd = os.open(lp_dir / "model.safetensors", os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            if cfg.get("lp_only"):
                if ckpt_every > 0:
                    ckpt_path.unlink(missing_ok=True)
                # recreate only the WiSE endpoint of an existing fine-tuning run (same seed and data order up to here)
                report = json.dumps({"status": "LP_ONLY_DONE", "fold": cfg["fold"], "lp_steps": lp_steps,
                                     "steps_of_full_run": steps, "seconds": round(time.time() - started, 1)}, indent=1)
                (lp_dir / "train_report.json").write_text(report)
                (out / "train_report.json").write_text(report)  # lets chain_guarded.sh see the job as done
                print(json.dumps({"status": "LP_ONLY_DONE", "out": str(lp_dir)}), flush=True)
                return
        if step % 200 == 0 or step == steps:
            rate = step * batch / (time.time() - started)
            history.append({"step": step, "loss": running / (200 if step % 200 == 0 else (step % 200)), "img_per_s": round(rate, 1)})
            print(json.dumps({"fold": cfg["fold"], "backbone": cfg["backbone"], **history[-1], "of": steps}), flush=True)
            running = 0.0
        if ckpt_every > 0 and (step % ckpt_every == 0 or step == steps):
            # the state after `step` optimizer steps; the RNG states are the ones the next step will draw from
            _ckpt_save(ckpt_path, {
                "format": 1, "cfg": json.loads(json.dumps(cfg)), "step": step, "steps": steps, "lp_steps": lp_steps,
                "digest": ckpt_digest, "model": model.state_dict(), "opt": opt.state_dict(), "ema": ema_states,
                "running": running, "history": history, "train_seconds": time.time() - started,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device),
            }, float(cfg.get("ckpt_min_free_gb", 10.0)))
    train_seconds = time.time() - started
    # release the optimizer state before the final predictions: on a shared GPU the end-of-run inference must not need
    # more memory than training did (several finished runs were lost to an OOM here)
    opt.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()

    variants = [(out, None)]
    raw_dtype = {k: v.dtype for k, v in model.state_dict().items()}
    for (ema_decay, target), ema_state in zip(ema_targets, ema_states):
        variants.append((target, ema_state))
    for target, state in variants:
        if state is not None:
            model.load_state_dict({k: v.to(raw_dtype[k]) for k, v in state.items()})
            target.mkdir(parents=True, exist_ok=True)
            decay = next(d for (d, t) in ema_targets if t == target)
            (target / "config.json").write_text(json.dumps(dict(cfg, output_dir=str(target), ema_of=str(out), ema_used=decay), indent=1))
        # save_raw_weights=False keeps only the predictions of the raw (non-EMA) model: soups and packages use the EMA weights
        cfg_v = cfg if state is not None else dict(cfg, save_weights=cfg.get("save_weights", True) and cfg.get("save_raw_weights", True))
        _predict_and_save(model, cfg_v, meta, images, target, out_hw, crop, mean, std, pos_weight, rows, row_sources, present,
                          steps, lp_steps, train_seconds, history)
    if ckpt_every > 0 and not cfg.get("retain_resume_checkpoint", False):  # old default unchanged
        ckpt_path.unlink(missing_ok=True)
        print(json.dumps({"event": "ckpt_removed", "ckpt": str(ckpt_path)}), flush=True)
    elif ckpt_every > 0:
        # New opt-in: root verifies actual exit/final artifacts before retiring this exact new checkpoint.
        print(json.dumps({"event": "ckpt_retained_for_actual_exit_review", "ckpt": str(ckpt_path)}), flush=True)


def _predict_and_save(model, cfg, meta, images, out, out_hw, crop, mean, std, pos_weight, rows, row_sources, present,
                      steps, lp_steps, train_seconds, history) -> None:
    import torch
    from safetensors.torch import save_file

    # weights first: a failure in the prediction below must not lose the trained model
    if cfg.get("save_weights", True):  # screening runs may keep only predictions (disk pressure)
        state = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(out / "model.safetensors"))
        del state
    # predict every validation image (original and horizontal flip); halve the batch on a CUDA OOM (shared GPU)
    device = next(model.parameters()).device
    model.eval()
    val_rows = np.flatnonzero(meta["split"] == "val")
    batch = cfg.get("predict_batch_size", 256)  # old default unchanged
    if type(batch) is not int or batch < 8:
        raise ValueError("predict_batch_size must be an integer >= 8")
    while True:
        try:
            logits_o, logits_f = [], []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(val_rows), batch):
                    idx = val_rows[start : start + batch]
                    x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device)
                    inputs = _gpu_augment(x, train=False, out_hw=out_hw, mean=mean, std=std, crop_top_frac=crop)
                    logits_o.append(model(inputs).float().cpu().numpy())
                    logits_f.append(model(inputs.flip(-1)).float().cpu().numpy())
            break
        except torch.OutOfMemoryError:
            if batch <= 8:
                raise
            batch //= 4
            torch.cuda.empty_cache()
            print(json.dumps({"event": "predict_oom_retry", "batch": batch}), flush=True)
            time.sleep(30)
    np.savez(
        out / "val_logits.npz",
        logits_original=np.concatenate(logits_o),
        logits_flip=np.concatenate(logits_f),
        pos_weight=pos_weight,
        image_paths=meta["image_paths"][val_rows],
        attribute_names=meta["attribute_names"],
    )
    report = {
        "status": "TRAIN_DONE",
        "fold": cfg["fold"],
        "include_val": cfg["include_val"],
        "backbone": cfg["backbone"],
        "fit_rows": int(len(rows)),
        "fit_rows_by_source": {s: int((row_sources == s).sum()) for s in present},
        "steps": steps,
        "lp_steps": lp_steps,
        "train_seconds": round(train_seconds, 1),
        "pos_weight": pos_weight.tolist(),
        "history": history,
        "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "pos_weight")}), flush=True)


def predict_all(run_dir: Path) -> None:
    """Original-view raw logits of a trained run for every cache row (train and val), for KD targets."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((run_dir / "config.json").read_text())
    cache = Path(cfg["cache_dir"])
    images = np.load(cache / cfg.get("cache_file", "images_u8_256x128.npy"), mmap_mode="r")  # cache variants: other sizes / letterbox
    model, mean, std = _build_model(cfg["backbone"], pretrained=False, head=cfg.get("head", "linear"))
    model.load_state_dict(load_file(str(run_dir / "model.safetensors")))
    model = model.cuda().eval()
    out_hw = BACKBONES[cfg["backbone"]]["img_hw"]
    crop = BACKBONES[cfg["backbone"]].get("crop_top_frac")
    logits = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(images), 512):
            x = torch.from_numpy(np.ascontiguousarray(images[start : start + 512])).cuda()
            logits.append(model(_gpu_augment(x, train=False, out_hw=out_hw, mean=mean, std=std, crop_top_frac=crop)).float().cpu().numpy())
    np.savez(run_dir / "all_rows_logits.npz", logits=np.concatenate(logits).astype(np.float16))
    print(json.dumps({"status": "PREDICT_ALL_DONE", "run": str(run_dir), "rows": int(len(images))}), flush=True)


def soup(run_dirs: list[Path], out: Path, weights: list[float] | None = None) -> None:
    """Weight average (uniform unless `weights`) of fine-tuned runs that share backbone, fold and fit rows (different
    seeds, or a run and its `_lp` linear-probe endpoint for WiSE-FT), then predict every validation image like train()
    does; the soup is an ordinary run directory for evaluation/packaging."""
    import torch
    from safetensors.torch import load_file, save_file

    cfgs = [json.loads((r / "config.json").read_text()) for r in run_dirs]
    if len({(c["backbone"], c["fold"], bool(c["include_val"])) for c in cfgs}) != 1:
        raise ValueError("soup members must share backbone, fold and fit rows")
    pos = [np.load(r / "val_logits.npz")["pos_weight"] if (r / "val_logits.npz").exists() else np.asarray(c["pos_weight"], dtype=np.float32)
           for r, c in zip(run_dirs, cfgs)]
    if any(not np.allclose(p_, pos[0]) for p_ in pos):
        raise ValueError("soup members must share pos_weight")
    w = np.ones(len(run_dirs)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(run_dirs) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("one non-negative weight per run")
    w = w / w.sum()
    states = [load_file(str(r / "model.safetensors")) for r in run_dirs]
    avg = {k: sum(float(wi) * st[k].float() for wi, st in zip(w, states)) for k in states[0]}
    cfg = dict(cfgs[0], output_dir=str(out), soup_of=[str(r) for r in run_dirs], soup_weights=w.tolist())
    cfg.pop("lp_of", None)
    model, mean, std = _build_model(cfg["backbone"], pretrained=False, head=cfg.get("head", "linear"))
    model.load_state_dict(avg)
    model = model.cuda().eval()
    cache = Path(cfg["cache_dir"])
    images = np.load(cache / cfg.get("cache_file", "images_u8_256x128.npy"), mmap_mode="r")  # cache variants: other sizes / letterbox
    meta = dict(np.load(cache / "meta.npz"))
    out_hw = BACKBONES[cfg["backbone"]]["img_hw"]
    crop = BACKBONES[cfg["backbone"]].get("crop_top_frac")
    val_rows = np.flatnonzero(meta["split"] == "val")
    lo, lf = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(val_rows), 256):
            x = torch.from_numpy(np.ascontiguousarray(images[val_rows[start : start + 256]])).cuda()
            inputs = _gpu_augment(x, train=False, out_hw=out_hw, mean=mean, std=std, crop_top_frac=crop)
            lo.append(model(inputs).float().cpu().numpy())
            lf.append(model(inputs.flip(-1)).float().cpu().numpy())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    np.savez(out / "val_logits.npz", logits_original=np.concatenate(lo), logits_flip=np.concatenate(lf), pos_weight=pos[0],
             image_paths=meta["image_paths"][val_rows], attribute_names=meta["attribute_names"])
    save_file({k: v.contiguous() for k, v in avg.items()}, str(out / "model.safetensors"))
    (out / "train_report.json").write_text(json.dumps({"status": "SOUP_DONE", "fold": cfg["fold"], "include_val": cfg["include_val"],
                                                       "backbone": cfg["backbone"], "soup_of": cfg["soup_of"]}, indent=1))
    print(json.dumps({"status": "SOUP_DONE", "out": str(out)}), flush=True)


# ----------------------------------------------------------------------------- evaluate
def _probabilities(npz, tta: bool) -> np.ndarray:
    correction = np.log(npz["pos_weight"]).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(npz["logits_original"] - correction)))
    if tta:
        pf = 1.0 / (1.0 + np.exp(-(npz["logits_flip"] - correction)))
        p = 0.5 * p + 0.5 * pf
    return p.astype(np.float32)


def evaluate_probabilities(probabilities: np.ndarray, fold: str, split=None, b011=None, entropy_delta: float | None = None) -> dict[str, float]:
    from .attribute_reliability import reliability_weights
    from .bernoulli_distance import mixed_distances
    from .data import load_retrieval_split
    from .metrics_strict import evaluate_scores

    split = split or load_retrieval_split(SPLITS["val"])
    b011 = b011 or json.loads(B011.read_text())
    if fold == "all":
        mask = np.ones(split.gallery_size, dtype=bool)
        weights = np.asarray(b011["full_weight"]["weights"], dtype=np.float32)
    else:
        mask = np.asarray(split.sources) == fold
        weights = reliability_weights(np.asarray(b011["crossfit"][fold]["reliability"], dtype=np.float32), gamma=0.5)
    sub = split.subset_gallery(mask)
    p = np.ascontiguousarray(probabilities[mask], dtype=np.float32)
    qidx = np.unique(sub.gallery_query_ids)
    per = []
    for start in range(0, len(qidx), 128):
        idx = qidx[start : start + 128]
        if entropy_delta:  # per-image entropy weights (run.py post "entropy_delta"); absent = the unchanged path below
            from .postproc import entropy_weight_matrix, mixed_distances_per_image
            if start == 0:
                W = entropy_weight_matrix(p, weights, float(entropy_delta))
            d = mixed_distances_per_image(p, sub.queries[idx].astype(np.float32), W, 0.25, float(np.finfo(np.float32).eps))
        else:
            d = mixed_distances(p, sub.queries[idx].astype(np.float32), weights=weights, nll_lambda=0.25, epsilon=float(np.finfo(np.float32).eps))
        per.extend(evaluate_scores(d, direction="distance", queries=sub.queries, gallery_attributes=sub.gallery_attributes,
                                   gallery_query_ids=sub.gallery_query_ids, query_indices=idx).per_query)
    return {k: float(np.mean([q[k] for q in per])) for k in ("ADM", "AP", "Rank-1", "Rank-5", "Rank-10", "INP")} | {"queries": len(per)}


def evaluate(run_dirs: list[Path], output: Path) -> None:
    from .data import load_retrieval_split

    split = load_retrieval_split(SPLITS["val"])
    b011 = json.loads(B011.read_text())
    results = {}
    for run in run_dirs:
        cfg = json.loads((run / "config.json").read_text())
        npz = np.load(run / "val_logits.npz")
        if tuple(npz["image_paths"].tolist()) != split.image_paths:
            raise ValueError(f"{run}: validation order differs")
        fold = cfg["fold"]
        row = {"backbone": cfg["backbone"], "fold": fold, "include_val": cfg["include_val"]}
        for tta in (False, True):
            row["tta" if tta else "single"] = evaluate_probabilities(_probabilities(npz, tta), fold, split, b011)
        results[str(run)] = row
        print(json.dumps({"run": run.name, "fold": fold, "single_mADM": round(row["single"]["ADM"], 5), "tta_mADM": round(row["tta"]["ADM"], 5)}), flush=True)
    output.write_text(json.dumps(results, indent=1))


def evaluate_combos(spec: dict[str, dict[str, list[str]]], output: Path, tta: bool = False) -> None:
    """Equal-weight probability ensembles per held fold, as run.py averages them.

    spec: {combo_name: {fold: [run_dir, ...]}}; folds are held sources (LODO) and/or "all".
    Optional "member_weights": one 40-vector per listed run (same order in every fold), as run.py applies them.
    """
    from .data import load_retrieval_split

    split = load_retrieval_split(SPLITS["val"])
    b011 = json.loads(B011.read_text())
    results: dict[str, Any] = {}
    for name, folds in spec.items():
        row: dict[str, Any] = {}
        folds = dict(folds)
        member_w = folds.pop("member_weights", None)
        for fold, runs in folds.items():
            probs = []
            for run in runs:
                npz = np.load(Path(run) / "val_logits.npz")
                if tuple(npz["image_paths"].tolist()) != split.image_paths:
                    raise ValueError(f"{run}: validation order differs")
                cfg = json.loads((Path(run) / "config.json").read_text())
                if cfg["fold"] != fold:
                    raise ValueError(f"{run}: trained for fold {cfg['fold']}, listed under {fold}")
                probs.append(_probabilities(npz, tta))
            w = np.ones((len(probs), 40), dtype=np.float32) if member_w is None else np.asarray(member_w, dtype=np.float32)
            mixed = np.einsum("mga,ma->ga", np.stack(probs), w) / w.sum(axis=0)
            row[fold] = evaluate_probabilities(mixed.astype(np.float32), fold, split, b011)
        held = [row[s]["ADM"] for s in SOURCES if s in row]
        if len(held) == len(SOURCES):
            row["lodo_macro_mADM"] = float(np.mean(held))
        results[name] = row
        print(json.dumps({"combo": name, **{f: round(v["ADM"], 5) for f, v in row.items() if isinstance(v, dict)},
                          "macro": round(row.get("lodo_macro_mADM", float("nan")), 5)}), flush=True)
    output.write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--config", type=Path, required=True)
    pa = sub.add_parser("predict-all")
    pa.add_argument("--runs", type=Path, nargs="+", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", type=Path, nargs="+", required=True)
    e.add_argument("--output", type=Path, required=True)
    sp = sub.add_parser("soup")
    sp.add_argument("--runs", type=Path, nargs="+", required=True)
    sp.add_argument("--output", type=Path, required=True)
    sp.add_argument("--weights", type=float, nargs="+", help="one weight per run (normalised); default uniform")
    ec = sub.add_parser("evaluate-combos")
    ec.add_argument("--spec", type=Path, required=True, help="JSON {combo: {fold: [run_dir, ...]}}")
    ec.add_argument("--output", type=Path, required=True)
    ec.add_argument("--tta", action="store_true", help="average original and flip views (deployed runs use single view)")
    args = parser.parse_args()
    if args.cmd == "cache":
        build_cache(args.out_dir, args.workers)
    elif args.cmd == "train":
        train(json.loads(args.config.read_text()))
    elif args.cmd == "predict-all":
        for run in args.runs:
            predict_all(run)
    elif args.cmd == "soup":
        soup(args.runs, args.output, args.weights)
    elif args.cmd == "evaluate-combos":
        evaluate_combos(json.loads(args.spec.read_text()), args.output, tta=args.tta)
    else:
        evaluate(args.runs, args.output)


if __name__ == "__main__":
    main()
