"""Independent confirmation surface on the organiser TRAIN split (never used for any selection).

Each held-out source's train images are predicted by the LODO fold model that never saw that source (fold == source),
with the *identical* member computations that produced the cached val-surface inputs (build_inputs_v1.py):

  PE  (vit0): uniform 4-recipe weight soup backbone (attrq/rankq/maskv2w(bias-aligned)/pswap03) @ 280x140;
              pooled = forward_head(pre_logits) -> u_pe; spherical head soups on bf16 pooled+patch tokens:
              pe_b136c = 20/20/30/30, tau (qk,out,base) = (1,1,1);  pe_b137c = 20/20/30/30, tau (1.2,1.0,0.6)
  HP  (vit1): DINOv3-H+ 20/30/30/20 (q/qr/qrq/qs) backbone soup @ 192x96; mean of patch tokens [5:] -> u_hp
              (= hp4_2030_192 features); heads: hp_b136c (hp_w_4, tau 1, out x1.03), hp_b137c (hp6_10 incl. 10% r224
              head, tau 1, out x1.03), hp_mask15_m15_r10_o103 / hp_mask15_m15_r00_o103 (column-selective, incl. qm head
              from extra_heads_backup_v1.pt); hp_ltype2 column blend for B139C
  EVA (vit2): 0.5 * eva02L224(pos_embed resampled 16x8 -> 18x9, timm bicubic antialias) + 0.5 * eva02L252 @ 252x126
              (= recipe_soup/eva_224_252_at252, whose model file is deleted, rebuilt here); backbone pooled -> u_eva;
              B136C raw = soup head logits (autocast, as h1_finetune.soup saved val_logits); B137C/B139C raw =
              compute_eva_probs(w252=0.5, tau=1.2) on the stored float32 features

subcommands
  extract --split val|train --fold F --member pe|hp|eva   (one member, one fold -> members/{split}_{fold}_{member}.npz)
  extract_all --split train                                (all folds x members, sequential, one model on GPU)
  check --fold F                                           (val reproduction vs cached arrays -> json)
  assemble                                                 (inputs/shared.npz + b136c/b137c/b139c.npz)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
S = ROOT / "artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle"
B127 = ROOT / "artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion"
HERE = Path(__file__).resolve().parent
TK = HERE.parent
MEM = HERE / "members"
OUT = HERE / "inputs"
KS = ROOT / "artifacts/runs/20261008T231500+0900_knob_sweep"
HP6 = ROOT / "artifacts/runs/20261017T0345+0900_hplus6"
RS = ROOT / "artifacts/runs/20261013T1500+0900_recipe_soup"
CACHE = ROOT / "artifacts/runs/20261007T020500+0900_gold_cycle/cache_256x128"
FEAT = S / "extracted_features_v1"
FOLDS = ("Market1501", "PA100k", "PETA")
PA_SEED = 20261021
PA_N = 10000
BATCH = 256  # = every val extraction script (cuBLAS kernel choice depends on M: batch 64 broke bit-exactness)
VRAM_CAP_GIB = 7.0

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(S))
sys.path.insert(0, str(B127))

from extract_pe_and_spherical_suite import eval_head_on_tokens, load_pe_members, make_custom_head, make_soup_state_dict  # noqa: E402
from upar.data import load_retrieval_split  # noqa: E402
from upar.h1_finetune import BACKBONES, _build_model, _decode, _gpu_augment  # noqa: E402

# ---- head recipes (copied verbatim from the producing scripts; names are the cached array keys) -----------------
PE_W_UNI = {"attrq": 0.25, "rankq": 0.25, "maskv2": 0.25, "pswap": 0.25}
PE_W_2030 = {"attrq": 0.20, "rankq": 0.20, "maskv2": 0.30, "pswap": 0.30}
PE_CONFIGS = {  # (w, tau_qk, tau_out, tau_base, q_scale, out_scale, base_scale)
    "pe_b136c": (PE_W_2030, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),            # test_spatial_snr_heads_gpu.py (v3)
    "pe_qk120_o100_b060": (PE_W_2030, 1.20, 1.0, 0.60, 1.0, 1.0, 1.0),  # == z4["pe_b137c"] (bit-identical arrays)
}
HP_W_4 = {"q": 0.20, "qr": 0.30, "qrq": 0.30, "qs": 0.20}
HP_W_6_10_V3 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "r224": 0.10}
HP_CONFIGS = {
    "hp_b136c": (HP_W_4, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),                      # v3
    "hp6_10_qk100_o103_b100": (HP_W_6_10_V3, 1.00, 1.0, 1.00, 1.0, 1.03, 1.0),  # == z4["hp_b137c"] (bit-identical)
}
MASKED_15 = [4, 5, 6, 10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
HP_W_6_10_V5 = {"q": 0.18, "qr": 0.27, "qrq": 0.27, "qs": 0.18, "qm": 0.0, "r224": 0.10}
HP_W_7_M15_R10 = {"q": 0.15, "qr": 0.225, "qrq": 0.225, "qs": 0.15, "qm": 0.15, "r224": 0.10}
HP_W_5_M15_R00 = {"q": 0.17, "qr": 0.255, "qrq": 0.255, "qs": 0.17, "qm": 0.15, "r224": 0.0}
HP_COLSEL = {  # test_b138c_colselective_heads_gpu.py (v5)
    "hp_mask15_m15_r10_o103": (HP_W_6_10_V5, {c: HP_W_7_M15_R10 for c in MASKED_15}, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
    "hp_mask15_m15_r00_o103": (HP_W_6_10_V5, {c: HP_W_5_M15_R00 for c in MASKED_15}, 1.0, 1.0, 1.0, 1.0, 1.03, 1.0),
}
BODY_12 = [10, 13, 14, 16, 18, 26, 27, 29, 31, 32, 33, 34]
DICH_3 = [3, 7, 20]
LTYPE_2 = [33, 34]


def now_tag() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%dT%H%M")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- gallery / image access --------------------------------------------------------------------------------------
def train_gallery_mask(tr) -> np.ndarray:
    src = np.asarray(tr.sources)
    pa = np.flatnonzero(src == "PA100k")
    pa_sel = np.sort(np.random.RandomState(PA_SEED).choice(pa, PA_N, replace=False))
    mask = (src == "Market1501") | (src == "PETA")
    mask[pa_sel] = True
    return mask


class Images:
    """uint8 256x128 images of split rows: the gold_cycle cache (PIL bilinear 256x128, the val pipeline's input) and,
    for the one train image the cache excluded (PETA/images/08572.png, cross-split duplicate), the same _decode()."""

    def __init__(self, split_name: str):
        self.split_name = split_name
        self.images = np.load(CACHE / "images_u8_256x128.npy", mmap_mode="r")
        meta = np.load(CACHE / "meta.npz")
        cache_rows = np.flatnonzero(meta["split"] == split_name)
        cache_paths = meta["image_paths"][cache_rows]
        split = load_retrieval_split(ROOT / f"starter_kit/data/annotations/task2/{split_name}")
        self.paths = np.asarray(split.image_paths)
        pos = {p: i for i, p in enumerate(cache_paths)}
        self.cache_idx = np.asarray([cache_rows[pos[p]] if p in pos else -1 for p in self.paths], dtype=np.int64)
        miss = np.flatnonzero(self.cache_idx < 0)
        if split_name == "val" and len(miss):
            raise ValueError("val rows missing from cache")
        self.missing = {int(i): self.paths[i] for i in miss}

    def get(self, rows: np.ndarray) -> np.ndarray:
        idx = self.cache_idx[rows]
        ok = idx >= 0
        out = np.empty((len(rows), 256, 128, 3), dtype=np.uint8)
        if ok.all():
            out[:] = self.images[idx]
        else:
            out[ok] = self.images[idx[ok]]
            for j in np.flatnonzero(~ok):
                out[j] = _decode(self.paths[rows[j]])
        return out


def fold_rows(split_name: str, fold: str) -> np.ndarray:
    split = load_retrieval_split(ROOT / f"starter_kit/data/annotations/task2/{split_name}")
    src = np.asarray(split.sources)
    if split_name == "val":
        return np.flatnonzero(src == fold)
    mask = train_gallery_mask(split)
    return np.flatnonzero(mask & (src == fold))


def slim_cuda(model):
    """Pre-cast nn.Linear / nn.Conv2d weights+biases to bf16 (exactly the cast autocast applies to them on every call;
    LayerNorm / pos-embed / layer-scale / RoPE / q-k-v bias parameters stay fp32 as in the val run), then move to GPU."""
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            mod.to(torch.bfloat16)
    for attempt in range(200):
        try:
            return model.cuda().eval()
        except torch.OutOfMemoryError:
            model.cpu()
            torch.cuda.empty_cache()
            log(f"OOM moving model to GPU (attempt {attempt}); waiting 15 s for the shared GPU")
            time.sleep(15)
    raise RuntimeError("persistent OOM while loading model")


def sigmoid_corr(lo: np.ndarray, pw: np.ndarray) -> np.ndarray:
    corr = np.log(pw).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-(lo - corr)))).astype(np.float32)


# ---- member loaders --------------------------------------------------------------------------------------------
def load_hp_lean(fold: str):
    """Same arithmetic as extract_pe_and_spherical_suite.load_hp_members + make_soup_state_dict(hp_w_4), but streams the
    four 3.4 GB member state dicts (keeps only their head.* tensors) so host RAM peaks at ~2 state dicts, not 6."""
    p = {m: KS / f"d3hp{m}_{fold}_ema0.999" for m in ("q", "qr", "qrq", "qs", "qm")}
    pw_ref = np.load(p["q"] / "val_logits.npz")["pos_weight"]
    pw_qm = np.load(p["qm"] / "val_logits.npz")["pos_weight"]
    bias_delta_qm = torch.from_numpy((np.log(pw_ref) - np.log(pw_qm)).astype(np.float32))
    soup: dict[str, torch.Tensor] = {}
    heads: dict[str, dict[str, torch.Tensor]] = {}
    for m, w in HP_W_4.items():  # make_soup_state_dict order (q, qr, qrq, qs)
        sd = load_file(str(p[m] / "model.safetensors"))
        for k, ten in sd.items():
            if ten.is_floating_point():
                val = float(w) * ten.float()
                soup[k] = soup[k] + val if k in soup else val
            elif k not in soup:
                soup[k] = ten
        heads[m] = {k: v for k, v in sd.items() if k.startswith("head.")}
        del sd
    if (p["qm"] / "model.safetensors").exists():
        sd_qm = {k: v for k, v in load_file(str(p["qm"] / "model.safetensors")).items() if k.startswith("head.")}
    else:
        bkp = torch.load(FEAT / "extra_heads_backup_v1.pt", map_location="cpu")
        sd_qm = dict(bkp[p["qm"].name])
    for bk in ("head.base.bias", "head.bias"):
        if bk in sd_qm:
            sd_qm[bk] = (sd_qm[bk].float() + bias_delta_qm).to(sd_qm[bk].dtype).contiguous()
    heads["qm"] = sd_qm
    sd_hp6 = load_file(str(HP6 / f"hp6_{fold}/model.safetensors"))
    r224 = {}
    for k in heads["q"]:
        if heads["q"][k].is_floating_point():
            sum4 = heads["q"][k].float() + heads["qr"][k].float() + heads["qrq"][k].float() + heads["qs"][k].float()
            r224[k] = 3.0 * sd_hp6[k].float() - 0.5 * sum4
        else:
            r224[k] = sd_hp6[k]
    del sd_hp6
    heads["r224"] = r224
    return soup, heads, pw_ref


def load_eva_soup(fold: str):
    """make_xres_member.py (eva02L224 -> eva02_l14_252 pos_embed resample) + h1_finetune.soup (uniform 0.5/0.5)."""
    from timm.layers import resample_abs_pos_embed

    src = KS / f"eva02L224_{fold}_ema0.999"
    cfg = json.loads((src / "config.json").read_text())

    def grid(bb):
        h, w = BACKBONES[bb]["img_hw"]
        return h // 14, w // 14

    sd = load_file(str(src / "model.safetensors"))
    pe = sd["backbone.pos_embed"]
    g0, g1 = grid(cfg["backbone"]), grid("eva02_l14_252")
    assert pe.shape[1] == 1 + g0[0] * g0[1], (pe.shape, g0)
    sd["backbone.pos_embed"] = resample_abs_pos_embed(pe.float(), new_size=list(g1), old_size=list(g0), num_prefix_tokens=1,
                                                      interpolation="bicubic", antialias=True).to(pe.dtype).contiguous()
    sd252 = load_file(str(KS / f"eva02L252_{fold}_ema0.999/model.safetensors"))
    states = [sd, sd252]
    w = np.ones(2) / 2.0
    avg = {k: sum(float(wi) * st[k].float() for wi, st in zip(w, states)) for k in states[0]}
    pw_soup = np.load(src / "val_logits.npz")["pos_weight"]  # soup() keeps member 0's pos_weight (== eva02L252's)
    return avg, pw_soup


def eva_t12_probs(fold: str, features: np.ndarray, w252: float = 0.50, tau: float = 1.2, scale: float = 1.0) -> np.ndarray:
    """Body of test_eva_head_and_exact_b137c.compute_eva_probs for one fold, applied to given float32 features."""
    w224 = 1.0 - w252
    sd224 = load_file(str(KS / f"eva02L224_{fold}_ema0.999/model.safetensors"))
    sd252 = load_file(str(KS / f"eva02L252_{fold}_ema0.999/model.safetensors"))
    W1, B1 = sd224["head.weight"].float(), sd224["head.bias"].float()
    W2, B2 = sd252["head.weight"].float(), sd252["head.bias"].float()
    W = w224 * W1 + w252 * W2
    B = w224 * B1 + w252 * B2
    if tau != 0.0:
        mn = w224 * W1.norm(dim=-1, keepdim=True) + w252 * W2.norm(dim=-1, keepdim=True)
        sn = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W = W * ((mn / sn) ** tau)
    if scale != 1.0:
        W = W * scale
    pw = np.load(KS / f"eva02L252_{fold}_ema0.999/val_logits.npz")["pos_weight"]
    feats = torch.from_numpy(features).bfloat16()
    logits = torch.nn.functional.linear(feats, W.bfloat16(), B.bfloat16()).float().numpy() - np.log(pw).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


# ---- extraction ------------------------------------------------------------------------------------------------
def _forward_batch(model, member, batch_u8, hw, mean, std, npt, pool, heads):
    x = torch.from_numpy(batch_u8).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inp = _gpu_augment(x, train=False, out_hw=hw, mean=mean, std=std)
        if member == "eva":
            f = model.backbone(inp)
            lo = model.head(model.dropout(f))  # == model(inp) of h1_finetune.soup (val_logits of the spec member 2)
            return f.float().cpu().numpy(), lo.float().cpu().numpy()
        ff = model.backbone.forward_features(inp)
        patches = ff[:, npt:]
        pooled = model.backbone.forward_head(ff, pre_logits=True) if pool == "head" else patches.mean(dim=1)
        feat = pooled.float().cpu().numpy()
        pooled_b = pooled.to(torch.bfloat16).cpu()
        patches_b = patches.to(torch.bfloat16).cpu()
    return feat, pooled_b, patches_b


def extract(split_name: str, fold: str, member: str, imgs: Images | None = None) -> Path:
    MEM.mkdir(parents=True, exist_ok=True)
    out = MEM / f"{split_name}_{fold}_{member}.npz"
    if out.exists():
        log(f"skip existing {out.name}")
        return out
    rows = fold_rows(split_name, fold)
    imgs = imgs or Images(split_name)
    t0 = time.monotonic()
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    res: dict[str, np.ndarray] = {"rows": rows}
    if member == "pe":
        mems, pw = load_pe_members(fold)
        model, mean, std = _build_model("pe_core_l14_280", pretrained=False, head="attrq")
        model.load_state_dict(make_soup_state_dict(mems, PE_W_UNI), strict=True)
        model = slim_cuda(model)
        heads = {c: make_custom_head(mems, w, tq, to, tb, qs, os_, bs) for c, (w, tq, to, tb, qs, os_, bs) in PE_CONFIGS.items()}
        del mems
        hw, npt, pool = (280, 140), 1, "head"
    elif member == "hp":
        soup, mems, pw = load_hp_lean(fold)
        model, mean, std = _build_model("dinov3_hp16_192", pretrained=False, head="attrq")
        model.load_state_dict(soup, strict=True)
        del soup
        model = slim_cuda(model)
        from test_b138c_colselective_heads_gpu import make_colselective_head
        heads = {c: make_custom_head(mems, w, tq, to, tb, qs, os_, bs) for c, (w, tq, to, tb, qs, os_, bs) in HP_CONFIGS.items()}
        heads.update({c: make_colselective_head(mems, w, ov, tq, to, tb, qs, os_, bs)
                      for c, (w, ov, tq, to, tb, qs, os_, bs) in HP_COLSEL.items()})
        del mems
        hw, npt, pool = (192, 96), 5, "mean"
    elif member == "eva":
        avg, pw = load_eva_soup(fold)
        model, mean, std = _build_model("eva02_l14_252", pretrained=False, head="linear")
        model.load_state_dict(avg)
        del avg
        model = slim_cuda(model)
        heads = {}
        hw = (252, 126)
    else:
        raise ValueError(member)
    t_load = time.monotonic() - t0
    feats, logits, pooled_l, patches_l = [], [], [], []
    probs = {}
    t1 = time.monotonic()
    for s in range(0, len(rows), BATCH):
        batch_u8 = imgs.get(rows[s:s + BATCH])
        for attempt in range(200):
            try:
                out_b = _forward_batch(model, member, batch_u8, hw, mean, std, npt if member != "eva" else 0,
                                       pool if member != "eva" else "", heads)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"OOM at batch {s} (attempt {attempt}); waiting 15 s for the shared GPU")
                time.sleep(15)
        else:
            raise RuntimeError("persistent OOM")
        if member == "eva":
            feats.append(out_b[0])
            logits.append(out_b[1])
        else:
            feats.append(out_b[0])
            pooled_l.append(out_b[1])
            patches_l.append(out_b[2])
        if (s // BATCH) % 40 == 0:
            log(f"{split_name} {fold} {member} {s + len(batch_u8)}/{len(rows)}")
    if member != "eva":
        # heads on the whole fold's bf16 pooled + patch tokens, 1024-row chunks from the fold start (as the val scripts)
        pooled_cpu = torch.cat(pooled_l, dim=0)
        patches_cpu = torch.cat(patches_l, dim=0)
        del pooled_l, patches_l
        for c, hparams in heads.items():
            for attempt in range(200):
                try:
                    probs[c] = eval_head_on_tokens(pooled_cpu, patches_cpu, hparams)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"OOM in head {c} (attempt {attempt}); waiting 15 s for the shared GPU")
                    time.sleep(15)
            else:
                raise RuntimeError("persistent OOM")
        del pooled_cpu, patches_cpu
    torch.cuda.synchronize()
    t_fwd = time.monotonic() - t1
    res["features"] = np.concatenate(feats, axis=0)
    res["pos_weight"] = np.asarray(pw)
    for c in heads:
        res[c] = sigmoid_corr(probs[c], pw)
    if member == "eva":
        res["logits_soup"] = np.concatenate(logits, axis=0)
        res["eva_soup_raw"] = sigmoid_corr(res["logits_soup"], pw)  # tk.load_member_probs_raw on the soup's val_logits
        res["eva_t12"] = eva_t12_probs(fold, res["features"])
    peak = torch.cuda.max_memory_allocated() / 2**30
    timing = {"split": split_name, "fold": fold, "member": member, "n": int(len(rows)), "load_s": round(t_load, 1),
              "forward_s": round(t_fwd, 1), "img_per_s": round(len(rows) / max(t_fwd, 1e-9), 1), "peak_vram_gib": round(peak, 2)}
    res["timing_json"] = np.asarray(json.dumps(timing))
    del model
    torch.cuda.empty_cache()
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **res)
    tmp.replace(out)
    log("DONE " + json.dumps(timing))
    return out


# ---- assembly helpers --------------------------------------------------------------------------------------------
def member_arrays(split_name: str, fold: str) -> dict[str, np.ndarray]:
    zp = np.load(MEM / f"{split_name}_{fold}_pe.npz")
    zh = np.load(MEM / f"{split_name}_{fold}_hp.npz")
    ze = np.load(MEM / f"{split_name}_{fold}_eva.npz")
    assert np.array_equal(zp["rows"], zh["rows"]) and np.array_equal(zp["rows"], ze["rows"])
    return {
        "rows": zp["rows"], "pe_feat": zp["features"], "eva_feat": ze["features"], "hp_feat": zh["features"],
        "pe_b136c": zp["pe_b136c"], "pe_b137c": zp["pe_qk120_o100_b060"],
        "hp_b136c": zh["hp_b136c"], "hp_b137c": zh["hp6_10_qk100_o103_b100"],
        "hp_mask15_m15_r10_o103": zh["hp_mask15_m15_r10_o103"], "hp_mask15_m15_r00_o103": zh["hp_mask15_m15_r00_o103"],
        "eva_soup_raw": ze["eva_soup_raw"], "eva_t12": ze["eva_t12"], "eva_logits_soup": ze["logits_soup"],
    }


def variant_inputs(m: dict[str, np.ndarray]):
    from test_b139c_tri_colselective_multires_gpu import make_col_blend
    from test_hp6_224_final_comb import build_anat_moe

    w_anat10 = build_anat_moe([0.425, 0.340, 0.235], [0.420, 0.345, 0.235], [0.370, 0.395, 0.235], [0.355, 0.410, 0.235])
    w_c36_hp41 = w_anat10.copy()
    w_c36_hp41[:, 36] = np.asarray([0.355, 0.410, 0.235])
    hp_b138c = make_col_blend(m["hp_b137c"], [(BODY_12, m["hp_mask15_m15_r10_o103"])])
    hp_ltype2 = make_col_blend(hp_b138c, [(DICH_3, m["hp_b136c"]), (LTYPE_2, m["hp_mask15_m15_r00_o103"])])
    return {
        "b136c": ([m["pe_b136c"], m["hp_b136c"], m["eva_soup_raw"]], w_anat10),
        "b137c": ([m["pe_b137c"], m["hp_b137c"], m["eva_t12"]], w_anat10),
        "b139c": ([m["pe_b137c"], hp_ltype2, m["eva_t12"]], w_c36_hp41),
    }


def pool_variant(raw, w_m, post_cfg):
    from test_hp6_224_stack import pool_custom
    from upar.postproc import apply_post

    return (pool_custom(raw, raw, post_cfg, w_m),
            np.std(np.stack([apply_post(r, post_cfg) for r in raw]), axis=0).astype(np.float32))


def post_cfg_obj() -> dict:
    from upar.rank_sibling_pipeline import read_json
    from upar.source_fitted_gallery_edges import SPEC

    return read_json(SPEC)["lodo"]["post"]


def unit_hp(f: np.ndarray) -> np.ndarray:
    f = f.astype(np.float64)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


# ---- val reproduction check ------------------------------------------------------------------------------------
def check(fold: str) -> None:
    import test_tri_graph_combined_knobs as tk  # sets the b121 package path; tk.unit_features_fp64 as in build_variant_contexts
    unit_features_fp64 = tk.unit_features_fp64

    m = member_arrays("val", fold)
    hp = m["rows"]
    val = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/val")
    assert np.array_equal(hp, np.flatnonzero(np.asarray(val.sources) == fold))
    z3 = np.load(FEAT / "spherical_head_oof_v3.npz")
    z4 = dict(np.load(FEAT / "spherical_head_oof_v4.npz"))
    z4.update(dict(np.load(FEAT / "spherical_head_oof_v5.npz")))
    rep: dict = {"fold": fold, "rows": int(len(hp)), "prob_maxabs": {}, "prob_meanabs": {}, "feature": {}, "pooled": {}}

    def cmp(name, mine, ref):
        d = np.abs(mine.astype(np.float64) - ref.astype(np.float64))
        rep["prob_maxabs"][name] = float(d.max())
        rep["prob_meanabs"][name] = float(d.mean())

    cmp("pe_b136c (z3)", m["pe_b136c"], z3["pe_b136c"][hp])
    cmp("pe_b137c (z4)", m["pe_b137c"], z4["pe_b137c"][hp])
    cmp("hp_b136c (z3)", m["hp_b136c"], z3["hp_b136c"][hp])
    cmp("hp_b137c (z4)", m["hp_b137c"], z4["hp_b137c"][hp])
    cmp("hp_mask15_m15_r10_o103 (z5)", m["hp_mask15_m15_r10_o103"], z4["hp_mask15_m15_r10_o103"][hp])
    cmp("hp_mask15_m15_r00_o103 (z5)", m["hp_mask15_m15_r00_o103"], z4["hp_mask15_m15_r00_o103"][hp])
    zs = np.load(RS / f"eva_224_252_at252_{fold}/val_logits.npz")
    ref_soup_raw = sigmoid_corr(zs["logits_original"], zs["pos_weight"])[hp]
    cmp("eva_soup_raw (spec member-2 val_logits, B136C)", m["eva_soup_raw"], ref_soup_raw)
    rep["eva_soup_logits_maxabs"] = float(np.abs(m["eva_logits_soup"] - zs["logits_original"][hp]).max())
    rep["eva_soup_logits_meanabs"] = float(np.abs(m["eva_logits_soup"] - zs["logits_original"][hp]).mean())
    # compute_eva_probs itself on the cached features (my helper must be bit-identical to it)
    from test_eva_head_and_exact_b137c import compute_eva_probs

    fake = {f: {"hp": np.flatnonzero(np.asarray(val.sources) == f)} for f in FOLDS}
    p2_ref, _ = compute_eva_probs(KS, FEAT, fake, w252=0.50, tau=1.2, scale=1.0)
    ef = np.load(FEAT / f"eva_full_val_{fold}.npz")["features"]
    rep["eva_t12_helper_vs_compute_eva_probs_on_cached_feats_maxabs"] = float(np.abs(eva_t12_probs(fold, ef[hp]) - p2_ref[hp]).max())
    cmp("eva_t12 (compute_eva_probs w252=.5 tau=1.2, B137C/B139C)", m["eva_t12"], p2_ref[hp])

    def fcmp(name, mine, ref):
        a = mine.astype(np.float64)
        b = ref.astype(np.float64)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
        rep["feature"][name] = {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "rel_l2_max": float(rel.max()),
                                "maxabs": float(np.abs(a - b).max())}

    fcmp("pe (pe_full_val)", m["pe_feat"], np.load(FEAT / f"pe_full_val_{fold}.npz")["features"][hp])
    fcmp("eva (eva_full_val)", m["eva_feat"], ef[hp])
    fcmp("hp (hp4_2030_192_full_val)", m["hp_feat"], np.load(FEAT / f"hp4_2030_192_full_val_{fold}.npz")["features"][hp])
    # unit features exactly as inputs_v1/shared.npz rows
    sh = np.load(TK / "inputs_v1/shared.npz")
    for k, mine in (("u_pe", unit_features_fp64(m["pe_feat"])), ("u_eva", unit_features_fp64(m["eva_feat"])), ("u_hp", unit_hp(m["hp_feat"]))):
        ref = sh[k][hp]
        rep["feature"][k + " (inputs_v1 rows)"] = {"cos_min": float((mine * ref).sum(1).min()), "maxabs": float(np.abs(mine - ref).max())}
    post_cfg = post_cfg_obj()
    for name, (raw, w_m) in variant_inputs(m).items():
        p, ms = pool_variant(raw, w_m, post_cfg)
        zv = np.load(TK / f"inputs_v1/{name}.npz")
        rep["pooled"][name] = {"oof_p_maxabs": float(np.abs(p - zv["oof_p"][hp]).max()),
                               "oof_p_meanabs": float(np.abs(p - zv["oof_p"][hp]).mean()),
                               "oof_mstd_maxabs": float(np.abs(ms - zv["oof_mstd"][hp]).max())}
        if name == "b139c":
            # isolate member contributions: the HP column blend alone vs cached arrays
            hp_b138c_ref = z4["hp_b137c"].copy()
            hp_b138c_ref[:, BODY_12] = z4["hp_mask15_m15_r10_o103"][:, BODY_12]
            ref_ltype2 = hp_b138c_ref.copy()
            ref_ltype2[:, DICH_3] = z3["hp_b136c"][:, DICH_3]
            ref_ltype2[:, LTYPE_2] = z4["hp_mask15_m15_r00_o103"][:, LTYPE_2]
            cmp("hp_ltype2 blend (B139C member 1)", raw[1], ref_ltype2[hp])
    rep["timings"] = [json.loads(str(np.load(MEM / f"val_{fold}_{mm}.npz")["timing_json"])) for mm in ("pe", "hp", "eva")]
    path = HERE / f"{now_tag()}_val_check_{fold}_v1.json"
    path.write_text(json.dumps(rep, indent=1))
    log("CHECK " + json.dumps(rep, indent=1))


# ---- assembly --------------------------------------------------------------------------------------------------
def assemble() -> None:
    import test_tri_graph_combined_knobs as tk  # sets the b121 package path; tk.unit_features_fp64 as in build_variant_contexts
    unit_features_fp64 = tk.unit_features_fp64
    from upar.metrics_strict import _normalized_degree_of_match

    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_retrieval_split(ROOT / "starter_kit/data/annotations/task2/train")
    mask = train_gallery_mask(tr)
    g_rows = np.flatnonzero(mask)
    sub = tr.subset_gallery(mask)
    src = np.asarray(tr.sources)[g_rows]
    n = len(g_rows)
    pos = np.full(mask.shape[0], -1, dtype=np.int64)
    pos[g_rows] = np.arange(n)
    keys40 = ("pe_b136c", "pe_b137c", "hp_b136c", "hp_b137c", "hp_mask15_m15_r10_o103", "hp_mask15_m15_r00_o103",
              "eva_soup_raw", "eva_t12")
    full = {k: np.full((n, 40), np.nan, dtype=np.float32) for k in keys40}
    feats = {"pe_feat": np.full((n, 1024), np.nan, np.float32), "eva_feat": np.full((n, 1024), np.nan, np.float32),
             "hp_feat": np.full((n, 1280), np.nan, np.float32)}
    for fold in FOLDS:
        m = member_arrays("train", fold)
        r = pos[m["rows"]]
        assert (r >= 0).all() and set(src[r]) == {fold}
        for k in keys40:
            full[k][r] = m[k]
        for k in feats:
            feats[k][r] = m[k]
    assert all(np.isfinite(v).all() for v in full.values()) and all(np.isfinite(v).all() for v in feats.values())
    sh_v1 = np.load(TK / "inputs_v1/shared.npz")
    names = np.asarray(tr.attribute_names)
    assert list(map(str, names)) == list(map(str, sh_v1["attribute_names"]))
    qi = np.unique(sub.gallery_query_ids)
    norm_match = _normalized_degree_of_match(sub.queries[qi].astype(np.uint8), sub.gallery_attributes.astype(np.uint8)).astype(np.float32)
    shared = {
        "qi_oof": qi, "queries_oof": sub.queries[qi].astype(np.float32), "w_oof": sh_v1["w_oof"],
        "u_pe": unit_features_fp64(feats["pe_feat"]), "u_eva": unit_features_fp64(feats["eva_feat"]), "u_hp": unit_hp(feats["hp_feat"]),
        "attribute_names": sh_v1["attribute_names"], "oof_norm_match": norm_match,
        "oof_g_qids": sub.gallery_query_ids.astype(np.int64), "sources": src, "train_rows": g_rows,
        "image_paths": np.asarray(sub.image_paths), "gallery_attributes": sub.gallery_attributes.astype(np.uint8),
    }
    np.savez(OUT / "shared.npz", **shared)
    log(f"shared saved G={n} Q={len(qi)} sources={ {s: int((src == s).sum()) for s in FOLDS} } {time.monotonic() - t0:.1f}s")
    post_cfg = post_cfg_obj()
    for name, (raw, w_m) in variant_inputs(full).items():
        p, ms = pool_variant(raw, w_m, post_cfg)
        np.savez(OUT / f"{name}.npz", oof_p=p, oof_mstd=ms)
        log(f"variant saved {name}")
    np.savez(OUT / "members_raw.npz", **full)
    meta = {"created": now_tag(), "gallery": {"Market1501": "all train", "PETA": "all train (incl. PETA/images/08572.png decoded with h1_finetune._decode)",
                                              "PA100k": f"RandomState({PA_SEED}).choice(PA100k train rows, {PA_N}, replace=False), sorted"},
            "G": int(n), "Q": int(len(qi)), "w_oof": "copied from inputs_v1/shared.npz",
            "timings": [json.loads(str(np.load(MEM / f"train_{f}_{mm}.npz")["timing_json"])) for f in FOLDS for mm in ("pe", "hp", "eva")]}
    (OUT / "README.json").write_text(json.dumps(meta, indent=1))
    log("ASSEMBLE DONE " + json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--split", choices=("val", "train"), required=True)
    e.add_argument("--fold", choices=FOLDS, required=True)
    e.add_argument("--member", choices=("pe", "hp", "eva"), nargs="+", required=True)
    ea = sp.add_parser("extract_all")
    ea.add_argument("--split", choices=("val", "train"), required=True)
    c = sp.add_parser("check")
    c.add_argument("--fold", choices=FOLDS, required=True)
    sp.add_parser("assemble")
    a = ap.parse_args()
    if a.cmd == "extract":
        imgs = Images(a.split)
        for mm in a.member:
            extract(a.split, a.fold, mm, imgs)
    elif a.cmd == "extract_all":
        imgs = Images(a.split)
        for fold in FOLDS:
            for mm in ("pe", "hp", "eva"):
                extract(a.split, fold, mm, imgs)
    elif a.cmd == "check":
        check(a.fold)
    else:
        assemble()


if __name__ == "__main__":
    main()
