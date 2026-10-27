"""Package H1 fine-tuned ViT(s) as a Codabench code submission (run in the torch 2.4.1 env).

Each model (timm ViT + 40-way linear head) is traced to TorchScript and stored with FP16
weights (cast back to FP32 at load; compute is FP32) so the ZIP needs only torch/numpy/Pillow
and stays small. A B015 ResNet-18 TorchScript is bundled as a speed fallback.

run.py is time-budgeted: while processing the gallery it projects the remaining time from the
throughput measured on this worker, drops ViT tiers (all ViTs -> first ViT only) when the
projection exceeds the budget, and switches to the R18 fallback for every image if even one
ViT cannot finish. A tier is applied to every gallery image, so the ranking stays consistent.
Ranking is the fixed B022 common ranker (prior correction, B011 weights, L1 + 0.25 NLL).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from .h1_finetune import B011, BACKBONES, ROOT

B022_ASSETS = ROOT / "submissions/B022_B014_B016_SLOT_ENSEMBLE/assets"

RUN_PY = r'''"""H1 fine-tuned ViT attribute retrieval: time-budgeted tiers, FP32 compute, CPU or CUDA."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_T0 = time.time()
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG: dict[str, Any] | None = None
_MODELS: list[Any] = []
_FALLBACK = None
_DEVICE = None
_POOL = None


def _log(event: str, **values: Any) -> None:
    print(json.dumps({"upar_h1": 1, "event": event, "t": round(time.time() - _T0, 2), **values}), flush=True)


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"torch": torch.__version__, "cpu_count": os.cpu_count(), "threads": torch.get_num_threads()}
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name") and "cpu_model" not in info:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    info["avx512f"] = "avx512f" in flags
                    info["avx2"] = "avx2" in flags
                    info["amx"] = "amx_tile" in flags
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            info["mem_total_kib"] = int(handle.readline().split()[1])
    except Exception:
        pass
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_mem_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception as exc:
            info["cuda_error"] = repr(exc)[:200]
    return info


def _maybe_fuse(model, spec: dict[str, Any], device: str):
    """oneDNN Graph fusion on CPU (about 1.2x on AVX2); verified against the unfused model, else unused."""
    if device != "cpu" or not _CONFIG.get("onednn_fusion") or not spec.get("allow_onednn_fusion"):
        return model
    try:
        h, w = spec["model_hw"]
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(32, 3, h, w, generator=generator)
        with torch.inference_mode():
            reference = model(x)
            torch.jit.enable_onednn_fusion(True)
            fused = torch.jit.freeze(model)
            for _ in range(2):
                fused(x)
            diff = float((fused(x) - reference).abs().max())
        if diff > 1e-3:
            raise RuntimeError(f"fused output differs by {diff}")
        _log("onednn_fusion", model=spec["name"], max_abs_diff=diff)
        return fused
    except Exception as exc:
        torch.jit.enable_onednn_fusion(False)
        _log("onednn_fusion_disabled", model=spec["name"], error=repr(exc)[:200])
        return model


def _set_slot(module, name: str, value) -> None:
    """Install a tensor in a parameter/buffer slot of a ScriptModule (slim graphs ship 1-element stride-0 placeholders)."""
    *path, leaf = name.split(".")
    for part in path:
        module = getattr(module, part)
    module._c.setattr(leaf, value)


class _BlockStack:
    """layout "blocks": prefix graph -> ONE transformer-block graph applied with each block's weights -> head graph.
    The TorchScript profiling executor compiles each NNC fusion group of the block once per member instead of once per
    block (B108G: 146 NVRTC compiles -> ~8); kernels, and therefore outputs, are the same as the unrolled graph's."""

    def __init__(self, prefix, block, head, params):
        self.prefix, self.block, self.head, self.params = prefix, block, head, params

    def parameters(self):
        return iter(self.params[0])

    def __call__(self, x):
        h, rope = self.prefix(x)
        for p in self.params:
            h = self.block(h, rope, *p)
        return self.head(h)


def _load_ts(spec: dict[str, Any], device: str):
    if device == "cuda":
        torch.cuda.set_device(0)  # loader threads bind the primary context explicitly
    dtype = torch.float32
    if device == "cuda" and _CONFIG.get("gpu_half", True):
        # GPU runners: FP16 weights and activations for ViT-B members; ViT-L/H members overflow FP16 (DINOv3-L: every
        # output non-finite) and run in bfloat16, which has the float32 range (probability error ~5e-4)
        dtype = torch.bfloat16 if spec.get("gpu_dtype") == "bfloat16" else torch.float16
    blocks = spec.get("layout") == "blocks"
    files = [spec["file"]] + ([spec["block_file"], spec["head_file"]] if blocks else [])
    graphs = [torch.jit.load(os.path.join(_HERE, f), map_location=device).eval() for f in files]
    slim = bool(spec.get("slim_graph"))
    if not slim:
        graphs = [g.float() for g in graphs]
    params = [[None] * len(spec["block_params"]) for _ in range(int(spec["num_blocks"]))] if blocks else None
    if spec.get("weights_file"):
        stored = torch.load(os.path.join(_HERE, spec["weights_file"]), map_location=device)
        state = None if slim else graphs[0].state_dict()
        owned = set(spec.get("graph_owned", []))  # resolution variants keep their own resampled position embedding
        index = {n: i for i, n in enumerate(spec.get("block_params", []))}
        homes = [(graphs[0], set(spec.get("prefix_keys", []))), (graphs[-1], set(spec.get("head_keys", [])))]
        with torch.no_grad():
            for name, value in stored.items():
                if name in owned:
                    continue
                if isinstance(value, dict):
                    value = value["q"].float() * value["s"].float().unsqueeze(1)
                if not slim:
                    state[name].copy_(value.float())
                    continue
                # slim graphs: install the restored tensor in the compute dtype (same bits as copy_ into the FP32
                # zero graph followed by .to(dtype)); no full-size zero copy is ever read, moved or converted
                value = value.float().to(dtype)
                if not blocks:
                    _set_slot(graphs[0], name, value)
                elif name.startswith("backbone.blocks."):
                    i, sub = name[len("backbone.blocks."):].split(".", 1)
                    if sub in index:  # block buffers identical in every block stay inside the block graph
                        params[int(i)][index[sub]] = value
                else:
                    for graph, keys in homes:
                        if name in keys:
                            _set_slot(graph, name, value)
        del stored
        if blocks and any(p is None for row in params for p in row):
            raise RuntimeError(f"{spec['name']}: block weights missing from {spec['weights_file']}")
    graphs = [g.to(dtype) for g in graphs]  # slim: only exact graph-kept tensors (e.g. RoPE tables) still convert
    model = _BlockStack(graphs[0], graphs[1], graphs[2], params) if blocks else graphs[0]
    h, w = spec["model_hw"]
    with torch.inference_mode():
        if device == "cuda":
            # profiling executor (one profiled run): call 1 profiles, call 2 builds the optimized plan (NNC/NVRTC fusion)
            # and runs it, so every timed batch already uses the fused kernels; a third call only adds a forward
            x = torch.zeros(int(_CONFIG["gpu_batch_size"]), 3, h, w, device=device, dtype=dtype)
            for _ in range(int(_CONFIG.get("gpu_warmup_calls", 3))):
                model(x)
            torch.cuda.synchronize()
        else:
            model(torch.zeros(1, 3, h, w, device=device))
    return model if blocks else _maybe_fuse(model, spec, device)


def _canary_input(n, h, w):
    """Deterministic smooth synthetic images (the same at package time and on the runner)."""
    g = np.random.RandomState(20261012)
    x = torch.from_numpy(g.rand(n, 3, max(1, h // 8), max(1, w // 4)).astype(np.float32))
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def _canary_check():
    """Fail closed when a loaded member does not reproduce its package-time (CPU fp32) outputs on fixed synthetic input:
    a broken loader or graph would otherwise produce finite but wrong probabilities and replace the leaderboard entry."""
    c = _CONFIG["canary"]
    n = int(c["n"])
    h, w = (int(v) for v in _CONFIG["cache_hw"])
    bs = int(_CONFIG["gpu_batch_size"])
    raw = _canary_input(n, h, w).to(_DEVICE)
    rep = raw.repeat((bs + n - 1) // n, 1, 1, 1)[:bs]  # warm-up batch shape: no new profiling/compilation
    rows = []
    with torch.inference_mode():
        for model, spec, ref in zip(_MODELS, _CONFIG["models"], c["ref"]):
            d = np.abs(_run(model, spec, rep)[:n] - np.asarray(ref, dtype=np.float32))
            rows.append({"name": spec["name"], "mean_abs": round(float(d.mean()), 5), "max_abs": round(float(d.max()), 5)})
    _log("canary", members=rows)
    bad = [r["name"] for r in rows if r["mean_abs"] > float(c.get("mean_tol", 0.02)) or r["max_abs"] > float(c.get("max_tol", 0.3))]
    if bad:
        raise RuntimeError(f"canary mismatch {bad}: refusing to produce output (fail closed)")


def load_model() -> None:
    global _CONFIG, _MODELS, _FALLBACK, _DEVICE
    with open(os.path.join(_HERE, "assets", "config.json"), encoding="utf-8") as handle:
        _CONFIG = json.load(handle)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _log("hardware", **_hardware())
    device = "cpu"
    if torch.cuda.is_available():
        try:
            specs = list(_CONFIG["models"])
            if _CONFIG.get("parallel_load", False) and len(specs) > 1:
                # loading is dominated by CPU-bound NNC/NVRTC compilation during each member's warm-up; members are
                # independent graph executors, so they load and compile concurrently
                with ThreadPoolExecutor(max_workers=len(specs)) as loader:
                    _MODELS = list(loader.map(lambda spec: _load_ts(spec, "cuda"), specs))
            else:
                _MODELS = [_load_ts(spec, "cuda") for spec in specs]
            if not _CONFIG.get("fail_closed_on_fallback"):
                _FALLBACK = _load_ts(_CONFIG["fallback"], "cuda")  # fail-closed runners can never use the fallback
            device = "cuda"
        except Exception as exc:
            _log("cuda_fallback", error=repr(exc)[:300])
            _MODELS = []
    if device == "cpu" and _CONFIG.get("require_cuda"):
        # fail closed: a failed run leaves the previous valid leaderboard entry in place, whereas a CPU run on a
        # GPU runner could only finish with a much weaker fallback (or time out)
        raise RuntimeError("CUDA unavailable or failed; refusing the CPU path on a GPU runner (require_cuda)")
    if device == "cpu":
        _MODELS = [_load_ts(spec, "cpu") for spec in _CONFIG["models"]]
        _FALLBACK = _load_ts(_CONFIG["fallback"], "cpu")
    _DEVICE = torch.device(device)
    global _POOL
    if device == "cuda":
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(16, (os.cpu_count() or 2) - 2)))
    _log("loaded", device=device, models=[s["name"] for s in _CONFIG["models"]], decode_threads=_POOL._max_workers if _POOL else 1)
    if device == "cuda" and _CONFIG.get("canary"):
        _canary_check()


def _decode_one(item) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    size = (int(_CONFIG["cache_hw"][1]), int(_CONFIG["cache_hw"][0]))
    path = item.get("image_path") or item.get("image")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return np.asarray(rgb.resize(size, resampling.BILINEAR), dtype=np.uint8)


def _decode_batch(items) -> torch.Tensor:
    # GPU runners have many idle CPU cores: decode in parallel threads (PIL releases the GIL); CPU runners stay serial
    if _POOL is not None:
        arrays = list(_POOL.map(_decode_one, items))
    else:
        arrays = [_decode_one(item) for item in items]
    return torch.from_numpy(np.stack(arrays)).to(_DEVICE).permute(0, 3, 1, 2).float().div_(255.0)


def _run(model, spec: dict[str, Any], raw: torch.Tensor) -> np.ndarray:
    hw = tuple(int(v) for v in spec["model_hw"])
    n = raw.shape[0]
    full = int(_CONFIG["gpu_batch_size"])
    if raw.device.type == "cuda" and n < full:
        # keep the warmed-up static shape: a smaller last batch fails the fused graph's shape guards and runs the
        # unfused fallback graph (slightly different bf16 rounding, extra profiling pass); images are independent
        raw = torch.cat([raw, raw.new_zeros((full - n,) + tuple(raw.shape[1:]))])
    x = raw
    if spec.get("crop_top_frac"):  # head member: fixed top region of the 256x128 decode, cut before resize
        x = x[:, :, : int(round(x.shape[-2] * float(spec["crop_top_frac"]))), :]
    if tuple(x.shape[-2:]) != hw:
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(spec["mean"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    x = ((x - mean) / std).contiguous()
    correction = torch.log(torch.tensor(spec["pos_weight"], dtype=torch.float32, device=x.device))
    dtype = next(model.parameters()).dtype
    x = x.to(dtype)
    p = torch.sigmoid(model(x).float() - correction)
    if spec.get("flip_tta"):
        p = 0.5 * p + 0.5 * torch.sigmoid(model(x.flip(-1)).float() - correction)
    return p[:n].float().cpu().numpy()


def _predict_members(gallery: list[dict[str, Any]], members: list[int], deadline: float, can_switch: bool, optional: int = 0):
    """Run one plan (ordered member indices). Returns (probabilities, tier) or None when a later plan should run.
    The last `optional` members are a bonus for fast workers: whenever their projection exceeds optional_margin x time
    left they are dropped first, and the remaining members keep their progress (no restart)."""
    global _FALLBACK
    specs = [_CONFIG["models"][i] for i in members]
    models = [_MODELS[i] for i in members]
    total = len(gallery)
    min_images = int(_CONFIG.get("min_images_before_projection_gpu", 1024) if _DEVICE.type == "cuda" else _CONFIG["min_images_before_projection"])
    batch = int(_CONFIG["gpu_batch_size"] if _DEVICE.type == "cuda" else _CONFIG["cpu_batch_size"])
    probs = [np.empty((total, 40), dtype=np.float32) for _ in specs]
    active = len(specs)
    spent = [0.0] * (len(specs) + 1)  # per-model seconds, last slot = decode/transfer
    use_fallback = False
    done = 0
    fetch = ThreadPoolExecutor(max_workers=1) if _POOL is not None else None
    pending = fetch.submit(_decode_batch, gallery[0:batch]) if fetch else None
    for start in range(0, total, batch):
        t = time.time()
        if fetch:
            raw = pending.result()
            if start + batch < total:
                pending = fetch.submit(_decode_batch, gallery[start + batch:start + 2 * batch])
        else:
            raw = _decode_batch(gallery[start:start + batch])
        spent[-1] += time.time() - t
        for i in range(active):
            t = time.time()
            probs[i][start:start + raw.shape[0]] = _run(models[i], specs[i], raw)
            spent[i] += time.time() - t
        done = start + raw.shape[0]
        remaining = total - done
        left = deadline - (time.time() - _T0)
        def projected(k: int) -> float:
            return remaining * (spent[-1] + sum(spent[:k])) / done
        # on GPU the early shortcut waits for min_images as well (first batches carry kernel/JIT warm-up)
        extreme = done > 0 and projected(active) > 3.0 * max(left, 1.0) and (_DEVICE.type != "cuda" or done >= min_images)
        if done >= min_images or left < 0 or extreme:
            while optional > 0 and active > 1 and projected(active) > float(_CONFIG.get("optional_margin", _CONFIG.get("plan_margin", 1.0))) * left:
                active -= 1
                optional -= 1
                _log("optional_drop", active=active, done=done, projected_s=round(projected(active), 1), left_s=round(left, 1))
            if can_switch and done <= 2 * min_images and projected(active) > float(_CONFIG.get("plan_margin", 1.0)) * left:
                # early only: a cheaper full plan is preferred over dropping members of this one; restart from image 0.
                # Later overruns use tier-drop, because a restart could no longer finish.
                _log("plan_switch", done=done, projected_s=round(projected(active), 1), left_s=round(left, 1),
                     seconds_per_image={"decode": round(spent[-1] / done, 4), **{s["name"]: round(spent[i] / done, 4) for i, s in enumerate(specs)}})
                return None
            while active > 1 and projected(active) > left:
                active -= 1
                _log("tier_drop", active=active, done=done, projected_s=round(projected(active), 1), left_s=round(left, 1))
            if projected(active) > left:
                use_fallback = True
                _log("fallback", done=done, projected_s=round(projected(active), 1), left_s=round(left, 1))
            if active < int(_CONFIG.get("min_active_members", 1)) or (use_fallback and _CONFIG.get("fail_closed_on_fallback")):
                # fail closed rather than replace the leaderboard entry with a much weaker single model or fallback
                raise RuntimeError(f"refusing degraded output: active={active} fallback={use_fallback} (fail-closed config)")
            if use_fallback:
                break
        if (start // batch) % 50 == 0 or remaining == 0:
            _log("progress", done=done, total=total, active=active, img_per_s=round(done / max(sum(spent), 1e-6), 2),
                 projected_remaining_s=round(projected(active), 1), left_s=round(left, 1),
                 seconds_per_image={"decode": round(spent[-1] / done, 4), **{s["name"]: round(spent[i] / done, 4) for i, s in enumerate(specs)}})
    if use_fallback:
        if _FALLBACK is None:
            _FALLBACK = _load_ts(_CONFIG["fallback"], _DEVICE.type)
        out = np.empty((total, 40), dtype=np.float32)
        for start in range(0, total, batch):
            raw = _decode_batch(gallery[start:start + batch])
            out[start:start + raw.shape[0]] = _run(_FALLBACK, _CONFIG["fallback"], raw)
        return out, _CONFIG["fallback"]["name"]
    # per-attribute member weights (default 1 = plain mean); renormalised over the active tier
    w = np.asarray([s.get("member_weights", [1.0] * 40) for s in specs[:active]], dtype=np.float32)
    if np.any(w.sum(axis=0) <= 0):
        w = np.ones_like(w)
    out = (np.einsum("mga,ma->ga", np.stack(probs[:active]), w) / w.sum(axis=0)).astype(np.float32)
    return out, "+".join(s["name"] for s in specs[:active])


def _apply_post(p, post):
    if not post:
        return p
    q = np.array(p, dtype=np.float32, copy=True)
    for start, end, alpha in post.get("renorm", []):
        g = q[:, int(start):int(end)]
        s = g.sum(axis=1, keepdims=True)
        q[:, int(start):int(end)] = (1.0 - float(alpha)) * g + float(alpha) * g / np.maximum(s, np.float32(1e-12))
    if post.get("clip"):
        eps = float(post["clip"])
        q = np.clip(q, eps, 1.0 - eps)
    return q.astype(np.float32)


def predict_attributes(gallery: list[dict[str, Any]], attribute_names: list[str]) -> np.ndarray:
    if not _MODELS:
        load_model()
    names = list(_CONFIG["attribute_names"])
    if sorted(attribute_names) != sorted(names):
        raise ValueError("attribute vocabulary differs from the trained model")
    reorder = np.asarray([names.index(n) for n in attribute_names], dtype=np.int64)
    deadline = float(_CONFIG["time_budget_s"]) - float(_CONFIG["rank_reserve_s"])
    # local replay hook (unset on the platform): a shared local GPU can be several times slower than the runner, so local
    # replays may stretch the deadline to exercise the full default path of the exact bytes
    scale = float(os.environ.get("UPAR_TIME_SCALE", "1"))
    if scale != 1.0:
        deadline *= scale
        _log("time_scale", scale=scale, deadline=round(deadline, 1))
    plans = _CONFIG.get("plans") or [{"name": "all", "members": list(range(len(_CONFIG["models"])))}]
    forced = os.environ.get("UPAR_FORCE_PLAN")  # local replay hook to exercise a later plan; unset on the platform
    if forced is not None:
        plans = [p for p in plans if p["name"] == forced]
        if not plans:
            raise ValueError(f"unknown plan {forced}")
    result = None
    with torch.inference_mode():
        for k, plan in enumerate(plans):
            _log("plan_start", plan=plan["name"], members=[_CONFIG["models"][i]["name"] for i in plan["members"]])
            result = _predict_members(gallery, list(plan["members"]), deadline, can_switch=k < len(plans) - 1,
                                      optional=int(plan.get("optional", 0)))
            if result is not None:
                break
    out, tier = result
    out = _apply_post(out, _CONFIG.get("post"))  # ensemble post-processing (identical to upar/postproc.py); absent = identity
    _log("predicted", tier=tier, seconds=round(time.time() - _T0, 1))
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite probabilities")
    return np.ascontiguousarray(out[:, reorder])


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    attribute_names = list(sample["attribute_names"])
    probabilities = predict_attributes(list(sample["gallery"]), attribute_names)
    queries = np.asarray(sample["queries"], dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 40:
        raise ValueError("queries must be Q x 40")
    names = list(_CONFIG["attribute_names"])
    reorder = np.asarray([names.index(n) for n in attribute_names], dtype=np.int64)
    weights = np.asarray(_CONFIG["attribute_distance_weights"], dtype=np.float32)[reorder]
    eps = np.float32(_CONFIG["bernoulli_nll_epsilon"])
    lam = np.float32(_CONFIG["bernoulli_nll_lambda"])
    ent = (_CONFIG.get("post") or {}).get("entropy_delta")
    if ent:  # per-image entropy weights (identical to upar/postproc.py entropy_weight_matrix); absent = per-attribute weights
        q = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
        h = np.exp((1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))) * float(ent))
        weights = (weights.astype(np.float64)[None, :] * (h / h.mean(axis=1, keepdims=True))).astype(np.float32)
    clipped = np.clip(probabilities, eps, np.float32(1.0) - eps)
    neg = -np.log1p(-clipped).astype(np.float32, copy=False)
    pos = -np.log(clipped).astype(np.float32, copy=False)
    base = (np.float32(1.0) - lam) * (probabilities * weights).sum(axis=1, dtype=np.float32) \
        + lam * (neg * weights).sum(axis=1, dtype=np.float32)
    delta = (np.float32(1.0) - lam) * weights * (np.float32(1.0) - np.float32(2.0) * probabilities) \
        + lam * weights * (pos - neg)
    distances = np.empty((len(queries), len(probabilities)), dtype=np.float32)
    for start in range(0, len(queries), 64):
        distances[start:start + 64] = base[None, :] + queries[start:start + 64] @ delta.T
    if not np.isfinite(distances).all():
        raise RuntimeError("non-finite distances")
    _log("ranked", queries=len(queries), gallery=len(probabilities))
    return {"distances": distances}
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fp16_store(module, dest: Path, scratch: Path):
    import torch

    tmp = scratch / (".tmp_" + dest.name)
    module.save(str(tmp))
    stored = torch.jit.load(str(tmp)).half()
    stored.save(str(dest))
    tmp.unlink()
    return torch.jit.load(str(dest)).float().eval()


def _freeze_rope(backbone, h: int, w: int) -> None:
    """Register the rotary position embedding of a fixed input size as a (non-persistent) buffer. Without this,
    torch.jit.trace bakes it in as a CPU constant and the traced graph fails on CUDA ("cuda:0 and cpu")."""
    rope = getattr(backbone, "rope", None)
    if rope is None:
        return
    import torch

    ph, pw = backbone.patch_embed.patch_size
    with torch.no_grad():
        try:
            emb = rope.get_embed(shape=(h // ph, w // pw))
        except TypeError:
            emb = rope.get_embed()
    backbone.register_buffer("rope_fixed", emb.detach().clone(), persistent=False)
    rope.get_embed = lambda *args, **kwargs: backbone.rope_fixed


FAST_REWRITES = ("rope_half", "mlp_pad", "gelu_tanh", "patch_linear")


def _fast_rewrite(model, opts) -> list[str]:
    """GPU speed rewrites of a timm Eva-family backbone, in place, before tracing (exact in real arithmetic; only the
    floating-point summation order / GEMM kernel choice changes, except gelu_tanh).
    Speed study (A/B ratios, parity): artifacts/runs/20261012T0000+0900_agents/speed2/results/ab_pkg.jsonl, ab_member.jsonl.
    - rope_half: interleaved RoPE (rot(): neg + stack copies the NNC fuser leaves unfused, per q and k and block) ->
      rotate-half RoPE (one fused kernel) with the q/k channels of every head permuted (evens, then odds) and the frozen
      RoPE table permuted alike; q.k is invariant to a common channel permutation. PE-Core and EVA02 use interleaved RoPE.
    - mlp_pad: SwiGLU hidden size not a multiple of 8 (EVA02-L: 2730, which forces align2 GEMM kernels) -> gate and value
      in one GEMM, each half zero-padded to a multiple of 8 rows; the LayerNorm and fc2 still see exactly the real channels
      (padding fc2's K as well needs a pad copy and measured no gain).
    - gelu_tanh: timm Mlp fc1 -> GELU -> fc2 (PE-Core, DINOv3) with fc1 + bias + GELU in ONE cuBLASLt call
      (torch._addmm_activation, GELU_BIAS epilogue): removes the separate GELU pass over the 4x hidden. NOT exact on CUDA:
      the epilogue (and torch's non-Lt CUDA fallback) is the tanh approximation; the CPU kernel stays exact GELU, so the
      CPU fp32 checks below cannot see it - check GPU parity against the current package.
    - patch_linear: stride == kernel patch-embedding conv with K = 3*P*P not a multiple of 8 (/14 models: 588, which
      cuDNN runs with a non-tensor-core kernel) as reshape + one GEMM.
    Per-row int8 storage commutes with these row permutations/concatenations; zero rows store as exact zeros.
    Parameter names/shapes are unchanged except mlp_pad (mlp.fc1.* replaces mlp.fc1_g.* / mlp.fc1_x.*)."""
    import torch

    class PaddedSwiGLU(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            h, d = mlp.fc1_g.weight.shape
            p = -(-h // 8) * 8
            self.hidden, self.padded = h, p
            self.fc1 = torch.nn.Linear(d, 2 * p)
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc1.weight[:h], self.fc1.weight[p:p + h] = mlp.fc1_g.weight, mlp.fc1_x.weight
            self.fc1.bias[:h], self.fc1.bias[p:p + h] = mlp.fc1_g.bias, mlp.fc1_x.bias
            self.act, self.norm, self.fc2 = mlp.act, mlp.norm, mlp.fc2

        def forward(self, x):
            y = self.fc1(x)
            h, p = self.hidden, self.padded
            return self.fc2(self.norm(self.act(y[..., :h]) * y[..., p:p + h]))

    class ConvAsLinear(torch.nn.Module):
        # every size is a Python int fixed at rewrite time: traced size arithmetic would leave tensor constants in the
        # graph that load onto CUDA, and aten::Int on them is a device sync per call (measured: ~60 ms/call when shared)
        def __init__(self, conv, grid):
            super().__init__()
            if tuple(conv.kernel_size) != tuple(conv.stride) or any(conv.padding) or conv.groups != 1:
                raise ValueError("patch_linear: the patch conv is not a non-overlapping patchify")
            self.weight, self.bias = conv.weight, conv.bias
            (self.ph, self.pw), (self.gh, self.gw) = tuple(conv.kernel_size), tuple(int(g) for g in grid)
            self.c, self.dim, self.k = conv.in_channels, conv.out_channels, conv.weight[0].numel()

        def forward(self, x):
            t = x.reshape(-1, self.c, self.gh, self.ph, self.gw, self.pw).permute(0, 2, 4, 1, 3, 5)
            y = torch.nn.functional.linear(t.reshape(-1, self.gh * self.gw, self.k), self.weight.reshape(self.dim, self.k), self.bias)
            return y.reshape(-1, self.gh, self.gw, self.dim).permute(0, 3, 1, 2)  # NCHW view, as the conv returned

    class GeluEpilogueMlp(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.fc1, self.fc2, self.d, self.h = mlp.fc1, mlp.fc2, mlp.fc1.in_features, mlp.fc1.out_features

        def forward(self, x):
            y = torch._addmm_activation(self.fc1.bias, x.reshape(-1, self.d), self.fc1.weight.t(), use_gelu=True)
            return self.fc2(y.reshape(x.shape[0], -1, self.h))

    unknown = set(opts) - set(FAST_REWRITES)
    if unknown:
        raise ValueError(f"unknown fast rewrites {sorted(unknown)}")
    bb, done = model.backbone, []
    with torch.no_grad():
        for blk in bb.blocks:
            if "mlp_pad" in opts and hasattr(blk.mlp, "fc1_g") and blk.mlp.fc1_g.out_features % 8:
                blk.mlp = PaddedSwiGLU(blk.mlp)
                done.append("mlp_pad")
            m = blk.mlp
            if ("gelu_tanh" in opts and type(m).__name__ == "Mlp" and isinstance(m.act, torch.nn.GELU)
                    and isinstance(m.norm, torch.nn.Identity) and m.fc1.bias is not None):
                blk.mlp = GeluEpilogueMlp(m)
                done.append("gelu_tanh")
        proj = getattr(bb.patch_embed, "proj", None)  # only K = C*P*P not a multiple of 8 (cuDNN: non-tensor-core kernel)
        if "patch_linear" in opts and isinstance(proj, torch.nn.Conv2d) and proj.weight[0].numel() % 8:
            bb.patch_embed.proj = ConvAsLinear(proj, bb.patch_embed.grid_size)
            done.append("patch_linear")
        rope = getattr(bb, "rope_fixed", None)
        if "rope_half" in opts and rope is not None and not getattr(bb.blocks[0].attn, "rotate_half", False):
            hd = rope.shape[-1] // 2
            for blk in bb.blocks:
                a = blk.attn
                if a.num_heads * hd * (3 if a.qkv is not None else 1) != (a.qkv if a.qkv is not None else a.q_proj).out_features:
                    raise ValueError("rope_half: RoPE table width is not 2 x head_dim")
                if not (isinstance(a.q_norm, torch.nn.Identity) and isinstance(a.k_norm, torch.nn.Identity)):
                    raise ValueError("rope_half: q/k norm present")
            ph = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])
            for blk in bb.blocks:
                a = blk.attn
                perm = (torch.arange(a.num_heads)[:, None] * hd + ph[None]).reshape(-1)
                if a.qkv is not None:
                    c = a.qkv.out_features // 3
                    rows = torch.cat([perm, c + perm, torch.arange(2 * c, 3 * c)])
                    mods = [(a.qkv, rows)]
                else:
                    mods = [(a.q_proj, perm), (a.k_proj, perm)]
                for m, rows in mods:
                    m.weight.copy_(m.weight[rows])
                    if m.bias is not None:
                        m.bias.copy_(m.bias[rows])
                for n in ("q_bias", "k_bias"):
                    if getattr(a, n, None) is not None:
                        getattr(a, n).copy_(getattr(a, n)[perm])
                a.rotate_half = True
            sin, cos = rope.tensor_split(2, -1)
            rope.copy_(torch.cat([sin[..., ph], cos[..., ph]], -1))
            done.append("rope_half")
    return sorted(set(done))


def _keep_in_graph(name: str) -> bool:
    """Tensors that stay exact inside the stored graph (not quantized, not zeroed): the frozen RoPE embedding."""
    return name.endswith("rope_fixed")


def _set_slot(module, name: str, value) -> None:
    """Install a tensor in a parameter/buffer slot of a ScriptModule (same helper as run.py)."""
    *path, leaf = name.split(".")
    for part in path:
        module = getattr(module, part)
    module._c.setattr(leaf, value)


def _slim(module, keep) -> None:
    """Replace every tensor slot whose name fails keep(name) by a 1-element stride-0 placeholder of the same shape and
    dtype. The saved graph shrinks from a full-size zero copy (1.2 GB per ViT-L) to ~1 MB; run.py installs the restored
    tensors into the slots (spec["slim_graph"])."""
    import torch

    for name, value in list(module.state_dict().items()):
        if not keep(name):
            _set_slot(module, name, torch.zeros(1, dtype=value.dtype).expand(value.shape))


def _restore(module, weights: Path, skip=()) -> None:
    import torch

    with torch.no_grad():
        for name, value in torch.load(str(weights)).items():
            if name in skip:
                continue
            if isinstance(value, dict):
                value = value["q"].float() * value["s"].float().unsqueeze(1)
            _set_slot(module, name, value.float())


def _used_keys(module) -> list[str]:
    """Tensor attributes a traced graph actually reads (GetAttr chains from self in the inlined graph)."""
    graph = module.inlined_graph
    paths = {next(iter(graph.inputs())): ""}
    used = set()
    for node in graph.nodes():
        if node.kind() == "prim::GetAttr" and node.input() in paths:
            path = (paths[node.input()] + "." if paths[node.input()] else "") + node.s("name")
            paths[node.output()] = path
            if node.output().type().kind() == "TensorType":
                used.add(path)
    return sorted(used)


def _int8_store(module, dest: Path, weights: Path, scratch: Path):
    """Per-output-row symmetric int8 for 2-D weights, fp16 otherwise; graph saved slim (stride-0 placeholders)."""
    import torch

    tmp = scratch / (".tmp_" + dest.name)
    module.save(str(tmp))
    stored = {}
    for name, value in module.state_dict().items():
        if _keep_in_graph(name):
            continue
        value = value.detach().float()
        if value.ndim == 2 and value.numel() >= 4096:
            scale = value.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            stored[name] = {"q": torch.round(value / scale[:, None]).clamp(-127, 127).to(torch.int8), "s": scale.half()}
        else:
            stored[name] = value.half()
    torch.save(stored, str(weights))
    slim = torch.jit.load(str(tmp))  # the frozen RoPE embedding stays exact (fp32) inside the graph
    _slim(slim, _keep_in_graph)
    slim.save(str(dest))
    tmp.unlink()
    restored = torch.jit.load(str(dest)).eval()
    _restore(restored, weights)
    return restored.float()


def _attribute_net(cfg: dict, spec: dict, img_hw: tuple):
    """The network exactly as trained: a linear head on the pooled feature (default), or one of h1_finetune's token
    heads (cfg["head"], e.g. attrq) with its own forward. The backbone is built at img_hw (variants: another size)."""
    import timm
    import torch

    backbone = timm.create_model(spec["timm"], pretrained=False, num_classes=0, img_size=img_hw)
    head = cfg.get("head", "linear")
    if head == "linear":
        class AttributeNet(torch.nn.Module):
            def __init__(self, backbone, dim: int):
                super().__init__()
                self.backbone = backbone
                self.dropout = torch.nn.Dropout(0.1)
                self.head = torch.nn.Linear(dim, 40)

            def forward(self, x):
                return self.head(self.dropout(self.backbone(x)))

        return AttributeNet(backbone, backbone.num_features), backbone
    from .h1_finetune import _build_model

    model, _, _ = _build_model(cfg["backbone"], pretrained=False, head=head, moe_experts=int(cfg.get("moe_experts", 4)))
    model.backbone = backbone  # same architecture, built at the requested input size
    return model, backbone


def _export_blocks(model, h: int, w: int, dest: Path, weights: Path, reference) -> dict:
    """layout "blocks" (timm Eva-family backbones: PE-Core, DINOv3, EVA02): three slim graphs next to `dest`
    - prefix: patch embed + class/register tokens + position embedding + norm_pre; also returns the frozen RoPE table
    - block:  ONE transformer block whose parameters are graph inputs (run.py loops over the stored block weights)
    - head:   final norm + pooling + attribute head
    The TorchScript profiling executor then compiles each fusion group once instead of once per block. `reference` is the
    restored full graph; the restored block stack must reproduce it on CPU FP32."""
    import torch
    from timm.models.eva import Eva

    bb = model.backbone
    if not isinstance(bb, Eva) or getattr(bb, "rope_mixed", False):
        raise ValueError("block layout supports timm Eva-family backbones without mixed RoPE")
    blocks = bb.blocks
    names = [n for n, _ in blocks[0].named_parameters()]
    buffers0 = dict(blocks[0].named_buffers())
    for blk in blocks:
        if [n for n, _ in blk.named_parameters()] != names or any(not torch.equal(b, buffers0[n]) for n, b in blk.named_buffers()):
            raise ValueError("blocks differ in parameter names or per-block buffers")
    rope = getattr(bb, "rope_fixed", None)

    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb  # parameter names == stored state keys

        def forward(self, x):
            x = self.backbone.patch_embed(x)
            x, r = self.backbone._pos_embed(x)
            return self.backbone.norm_pre(x), (r if r is not None else x.new_zeros(0))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = blocks[0]
            self.slots = []
            for n in names:
                mod = self.blk
                *path, leaf = n.split(".")
                for part in path:
                    mod = getattr(mod, part)
                self.slots.append((mod, leaf))

        def forward(self, x, r, *params):
            # torch.func.functional_call refuses to be traced: swap the parameter slots by hand for this call
            saved = [mod._parameters[leaf] for mod, leaf in self.slots]
            try:
                for (mod, leaf), p in zip(self.slots, params):
                    mod._parameters[leaf] = p
                return self.blk(x, rope=r if rope is not None else None)
            finally:
                for (mod, leaf), p in zip(self.slots, saved):
                    mod._parameters[leaf] = p

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = bb, model.head  # parameter names == stored state keys
            object.__setattr__(self, "_model", model)

        def forward(self, x):
            return self._model(x)  # backbone.forward_features is patched to the final norm while tracing

    stem = dest.with_suffix("")
    files = {k: stem.with_name(f"{stem.name}.{k}.ts") for k in ("prefix", "block", "head")}
    params = [[dict(blk.named_parameters())[n].detach() for n in names] for blk in blocks]
    with torch.inference_mode():
        x = torch.randn(2, 3, h, w)
        bb.blocks = torch.nn.ModuleList()  # prefix/head graphs must not carry the block weights
        bb.forward_features = lambda t: bb.norm(t)
        try:
            prefix = torch.jit.trace(Prefix(), x, check_trace=False)
            hid, r = prefix(x)
            for blk in blocks:
                hid = blk(hid, rope=rope)
            head = torch.jit.trace(Head(), hid, check_trace=False)
        finally:
            bb.blocks = blocks
            del bb.forward_features
        hid, r = prefix(x)
        block = torch.jit.trace(Block(), (hid, r, *params[0]), check_trace=False)
        spec = {"layout": "blocks", "num_blocks": len(blocks), "block_params": names,
                "prefix_keys": [k for k in _used_keys(prefix) if not _keep_in_graph(k)],
                "head_keys": [k for k in _used_keys(head) if not _keep_in_graph(k)]}
        _slim(prefix, _keep_in_graph)
        _slim(head, _keep_in_graph)
        _slim(block, lambda k: k.split(".", 1)[-1] in buffers0)  # block weights are inputs; shared buffers stay
        for key, mod in (("prefix", prefix), ("block", block), ("head", head)):
            mod.save(str(files[key]))
        # restore exactly as run.py does and compare with the restored full graph
        graphs = {k: torch.jit.load(str(f)).eval() for k, f in files.items()}
        index = {n: i for i, n in enumerate(names)}
        rows = [[None] * len(names) for _ in blocks]
        for key, value in torch.load(str(weights)).items():
            value = (value["q"].float() * value["s"].float().unsqueeze(1)) if isinstance(value, dict) else value.float()
            if key.startswith("backbone.blocks."):
                i, sub = key[len("backbone.blocks."):].split(".", 1)
                if sub in index:
                    rows[int(i)][index[sub]] = value
            else:
                for g, keys in ((graphs["prefix"], spec["prefix_keys"]), (graphs["head"], spec["head_keys"])):
                    if key in keys:
                        _set_slot(g, key, value)
        x = torch.randn(8, 3, h, w)
        hid, r = graphs["prefix"].float()(x)
        for row in rows:
            hid = graphs["block"].float()(hid, r, *row)
        diff = float((graphs["head"].float()(hid) - reference(x)).abs().max())
    if diff > 1e-4:
        raise ValueError(f"block layout differs from the full graph by {diff}")
    spec.update(file="assets/" + files["prefix"].name, block_file="assets/" + files["block"].name,
                head_file="assets/" + files["head"].name)
    return {"spec": spec, "check": diff}


def _trace_vit(run_dir: Path, scratch: Path, dest: Path, name: str, flip_tta: bool, storage: str = "fp16",
               layout: str = "full", fast: tuple = ()) -> dict:
    import timm
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((run_dir / "config.json").read_text())
    if cfg.get("cache_file", "images_u8_256x128.npy") != "images_u8_256x128.npy":
        # run.py decodes through the 256x128 path only; a model trained on another cache would be deployed with different
        # preprocessing than it was trained and scored on (review 10/12) - apply agents/input_pipeline/*.diff first
        raise ValueError(f"{run_dir}: trained on cache_file={cfg['cache_file']}, which this run.py cannot reproduce")
    spec = BACKBONES[cfg["backbone"]]
    logits_npz = np.load(run_dir / "val_logits.npz")
    model, backbone = _attribute_net(cfg, spec, tuple(spec["img_hw"]))
    model.load_state_dict(load_file(str(run_dir / "model.safetensors")), strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    h, w = spec["img_hw"]
    _freeze_rope(model.backbone, h, w)
    checks = {}
    applied = []
    if fast:
        if layout != "blocks":
            raise ValueError("--fast rewrites are only supported with --layout blocks")
        with torch.inference_mode():
            x = torch.randn(4, 3, h, w)
            before = model(x)
        applied = _fast_rewrite(model, fast)
        with torch.inference_mode():
            checks["fast_rewrite_logit_max_abs_diff"] = float((model(x) - before).abs().max())
        if checks["fast_rewrite_logit_max_abs_diff"] > 1e-3:  # fp32 summation-order noise ~1e-5..1e-4; bugs are O(1)
            raise ValueError(f"fast rewrites {applied} change the fp32 logits by {checks['fast_rewrite_logit_max_abs_diff']}")
    with torch.inference_mode():
        traced = torch.jit.trace(model, torch.randn(2, 3, h, w), check_trace=False)
        for bs in (1, 3, 16):
            x = torch.randn(bs, 3, h, w)
            checks[f"trace_bs{bs}"] = float((traced(x) - model(x)).abs().max())
        weights_file = None
        if storage == "int8":
            weights_file = dest.with_suffix(".weights.pt")
            restored = _int8_store(traced, dest, weights_file, scratch)
        else:
            restored = _fp16_store(traced, dest, scratch)
        x = torch.randn(8, 3, h, w)
        checks[f"{storage}_storage_logit_max_abs_diff"] = float((restored(x) - model(x)).abs().max())
    extra = {"slim_graph": True} if storage == "int8" else {}
    if layout == "blocks":
        if storage != "int8":
            raise ValueError("the block layout restores weights from the int8 weights file")
        blocks = _export_blocks(model, h, w, dest, weights_file, restored)
        checks["blocks_vs_full_graph_logit_max_abs_diff"] = blocks["check"]
        extra.update(blocks["spec"])
        dest.unlink()  # the full graph is not shipped
    return {
        "name": name,
        "file": "assets/" + dest.name,
        **extra,
        **({"fast": applied} if applied else {}),
        "allow_onednn_fusion": cfg["backbone"].startswith("clip"),
        "weights_file": ("assets/" + weights_file.name) if weights_file else None,
        "weight_storage": storage,
        "gpu_dtype": "bfloat16" if n_params > 2e8 else "float16",
        "n_params": int(n_params),
        "backbone": cfg["backbone"],
        "timm_name": spec["timm"],
        "model_hw": list(spec["img_hw"]),
        "crop_top_frac": spec.get("crop_top_frac"),
        "mean": list(backbone.pretrained_cfg["mean"]),
        "std": list(backbone.pretrained_cfg["std"]),
        "pos_weight": [float(v) for v in logits_npz["pos_weight"]],
        "flip_tta": bool(flip_tta),
        "training_run": str(run_dir),
        "training_fold": cfg["fold"],
        "training_include_val": cfg["include_val"],
        "checks": checks,
        "attribute_names": [str(v) for v in logits_npz["attribute_names"]],
    }


def _trace_variant(base: dict, run_dir: Path, reference_run: Path, scratch: Path, dest: Path, name: str, hw: list[int], sub: Path) -> dict:
    """The base model's weights traced at another input size (RoPE backbones only: identical parameter shapes).
    Only the slim graph is stored; at load time run.py fills it from the base model's weights file."""
    if base.get("layout") == "blocks":
        raise ValueError("resolution variants of a block-layout member are not supported yet (use --layout full)")
    import timm
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((run_dir / "config.json").read_text())
    spec = BACKBONES[cfg["backbone"]]
    model, backbone = _attribute_net(cfg, spec, tuple(hw))
    sd = load_file(str(run_dir / "model.safetensors"))
    graph_owned = []
    if "backbone.pos_embed" in sd and tuple(sd["backbone.pos_embed"].shape) != tuple(model.backbone.pos_embed.shape):
        from timm.layers import resample_abs_pos_embed

        ph, pw = model.backbone.patch_embed.patch_size
        oh, ow = (v // ph for v in spec["img_hw"])
        sd["backbone.pos_embed"] = resample_abs_pos_embed(sd["backbone.pos_embed"], new_size=(hw[0] // ph, hw[1] // pw), old_size=(oh, ow),
                                                          num_prefix_tokens=1)
        graph_owned.append("backbone.pos_embed")
    model.load_state_dict(sd, strict=True)
    model.eval()
    h, w = hw
    _freeze_rope(model.backbone, h, w)
    checks = {}
    with torch.inference_mode():
        traced = torch.jit.trace(model, torch.randn(2, 3, h, w), check_trace=False)
        for bs in (1, 3, 16):
            x = torch.randn(bs, 3, h, w)
            checks[f"trace_bs{bs}"] = float((traced(x) - model(x)).abs().max())
        tmp = scratch / (".tmp_" + dest.name)
        traced.save(str(tmp))
        # every tensor whose shape depends on the input size (resampled absolute position embedding, RoPE caches)
        # stays exact in the variant's own graph; all other tensors are shared from the base model's weights file
        base_stored = torch.load(str(sub / base["weights_file"]))
        traced_state = traced.state_dict()
        for key, value in base_stored.items():
            shape = tuple((value["q"] if isinstance(value, dict) else value).shape)
            if key in traced_state and tuple(traced_state[key].shape) != shape and key not in graph_owned:
                graph_owned.append(key)
        slim = torch.jit.load(str(tmp))
        _slim(slim, lambda key: _keep_in_graph(key) or key in graph_owned)
        slim.save(str(dest))
        tmp.unlink()
        restored = torch.jit.load(str(dest)).eval()
        state = restored.state_dict()
        stored = torch.load(str(sub / base["weights_file"]))
        if set(stored) != {k for k in state if not _keep_in_graph(k)} or any(k not in graph_owned and tuple(state[k].shape) != tuple((v["q"] if isinstance(v, dict) else v).shape) for k, v in stored.items()):
            raise ValueError(f"{name}: parameters differ from {base['name']} (variant needs a size-independent backbone)")
        _restore(restored, sub / base["weights_file"], skip=set(graph_owned))
        restored = restored.float()
        x = torch.randn(8, 3, h, w)
        checks["shared_weights_logit_max_abs_diff"] = float((restored(x) - model(x)).abs().max())
    ref = np.load(reference_run / "val_logits.npz")
    if not np.allclose(ref["pos_weight"], base["pos_weight"]):
        raise ValueError("reference run pos_weight differs from the base model")
    return dict(base, name=name, file="assets/" + dest.name, model_hw=list(hw), training_run=str(reference_run),
                variant_of=base["name"], graph_owned=graph_owned, checks=checks, slim_graph=True)


def _canary_refs(sub: Path, n: int = 4) -> dict:
    """CPU fp32 outputs of every member of the built package on run.py's synthetic canary input (subprocess with CUDA
    hidden, fail-closed switches relaxed in a temporary copy only)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "pkg"
        shutil.copytree(sub, work)
        cfg = json.loads((work / "assets/config.json").read_text())
        cfg.update(require_cuda=False, fail_closed_on_fallback=False, min_active_members=1)
        (work / "assets/config.json").write_text(json.dumps(cfg))
        code = (
            "import json, sys\n"
            "sys.path.insert(0, %r)\n"
            "from upar.execute_submission import load_submission_module\n"
            "mod = load_submission_module(__import__('pathlib').Path(%r) / 'run.py')\n"
            "mod.load_model()\n"
            "h, w = (int(v) for v in mod._CONFIG['cache_hw'])\n"
            "x = mod._canary_input(%d, h, w)\n"
            "import torch\n"
            "with torch.inference_mode():\n"
            "    print(json.dumps([mod._run(m, s, x).tolist() for m, s in zip(mod._MODELS, mod._CONFIG['models'])]))\n"
        ) % (str(ROOT / "src"), str(work), n)
        env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
        refs = json.loads(out.strip().splitlines()[-1])
    return {"n": n, "ref": refs, "mean_tol": 0.02, "max_tol": 0.3,
            "note": "fail closed if a member's CUDA output on the synthetic canary input differs from these CPU fp32 outputs"}


def build(run_dirs: list[Path], out_dir: Path, name: str, flip_tta: bool, time_budget_s: float, storage: str = "fp16",
          onednn_fusion: bool = False, member_weights: list[list[float]] | None = None,
          plans: list[dict] | None = None, rank_reserve_s: float = 120.0, plan_margin: float = 1.0,
          optional_margin: float | None = None, variants: list[dict] | None = None, require_cuda: bool = False,
          min_active_members: int = 1, fail_closed: bool = False, post: dict | None = None, layout: str = "full",
          fast: tuple = ()) -> dict:
    import timm
    import torch

    sub = out_dir / name
    if sub.exists():
        raise FileExistsError(sub)
    (sub / "assets").mkdir(parents=True)
    (sub / "licenses").mkdir()
    models = [_trace_vit(r, out_dir, sub / f"assets/model_{i}.ts", f"vit{i}_{json.loads((r / 'config.json').read_text())['backbone']}", flip_tta, storage,
                         layout, fast)
              for i, r in enumerate(run_dirs)]
    for v in variants or []:  # {"of": base index, "hw": [h, w], "reference_run": dir with val_logits at that size}
        if storage != "int8":
            raise ValueError("resolution variants share the base model's int8 weights file")
        i = len(models)
        models.append(_trace_variant(models[v["of"]], run_dirs[v["of"]], Path(v["reference_run"]), out_dir, sub / f"assets/model_{i}.ts",
                                     f"vit{i}_{models[v['of']]['backbone']}_{v['hw'][0]}x{v['hw'][1]}", v["hw"], sub))
    names = models[0]["attribute_names"]
    if any(m["attribute_names"] != names for m in models):
        raise ValueError("attribute order differs between models")
    if member_weights is not None:
        if len(member_weights) != len(models) or any(len(w) != 40 for w in member_weights):
            raise ValueError("member weights must be one 40-vector per model")
        if np.any(np.asarray(member_weights[:1]).sum(axis=0) <= 0):
            raise ValueError("the first member alone must cover every attribute (tier-drop safety)")
        for m, w in zip(models, member_weights):
            m["member_weights"] = [float(v) for v in w]
    if plans is not None:
        if not plans or any(not p["members"] or any(i < 0 or i >= len(models) for i in p["members"]) for p in plans):
            raise ValueError("every plan needs member indices into --run-dirs")
        if len({p["name"] for p in plans}) != len(plans):
            raise ValueError("plan names must be unique")
        if any(not 0 <= int(p.get("optional", 0)) < len(p["members"]) for p in plans):
            raise ValueError("optional members must leave at least one required member")
    b022 = json.loads((B022_ASSETS / "config.json").read_text())
    if b022["attribute_names"] != names:
        raise ValueError("fallback attribute order differs")
    restored = _fp16_store(torch.jit.load(str(B022_ASSETS / "model_b015.ts")), sub / "assets/fallback_b015.ts", out_dir)
    original = torch.jit.load(str(B022_ASSETS / "model_b015.ts")).eval()
    with torch.inference_mode():
        x = torch.randn(4, 3, 256, 128)
        fallback_diff = float((restored(x) - original(x)).abs().max())
    fallback = {"name": "fallback_b015_r18", "file": "assets/fallback_b015.ts", "model_hw": [256, 128], "mean": b022["mean"],
                "std": b022["std"], "pos_weight": b022["pos_weight"], "flip_tta": False}
    b011 = json.loads(B011.read_text())
    config = {
        "schema": 2,
        "candidate": name,
        "attribute_names": names,
        "cache_hw": [256, 128],
        "models": [{k: v for k, v in m.items() if k not in ("attribute_names", "checks")} for m in models],
        "member_weights_note": "per-attribute weights over active members; absent = plain mean",
        "plans": plans,
        "plan_margin": plan_margin,
        **({"optional_margin": optional_margin} if optional_margin is not None else {}),
        **({"post": post} if post else {}),
        "fallback": fallback,
        "time_budget_s": time_budget_s,
        "onednn_fusion": onednn_fusion,
        "rank_reserve_s": rank_reserve_s,
        "min_images_before_projection": 256,
        "min_images_before_projection_gpu": 1024,
        "gpu_half": True,
        "require_cuda": require_cuda,
        "min_active_members": min_active_members,
        "fail_closed_on_fallback": fail_closed,
        "cpu_batch_size": 32,
        "gpu_batch_size": 128,
        "gpu_warmup_calls": 3,  # conservative: torch 2.4.1 on the A10 is not testable locally (2 suffices per the speed study)
        "parallel_load": False,  # conservative: thread-safety of concurrent jit load/compile not verified on torch 2.4.1
        "attribute_distance_weights": b011["full_weight"]["weights"],
        "bernoulli_nll_lambda": 0.25,
        "bernoulli_nll_epsilon": float(np.finfo(np.float32).eps),
    }
    (sub / "assets/config.json").write_text(json.dumps(config, indent=1))
    (sub / "run.py").write_text(RUN_PY)
    shutil.copyfile(ROOT / "starter_kit/examples/task2/sample_code_submission/metadata.yaml", sub / "metadata.yaml")  # verbatim
    notice = ["Fine-tuned only on the organiser-provided UPAR training data; initialisations are generic pretrained models not trained for PAR:"]
    for m in models:
        notice.append(f"- timm `{m['timm_name']}`")
        for card_name in ("README.md", "HF_README.md"):
            card = (ROOT / BACKBONES[m["backbone"]]["file"]).parent / card_name
            if card.is_file():
                shutil.copyfile(card, sub / f"licenses/MODEL_CARD_{m['backbone']}.md")
                break
        if "dinov3" in m["backbone"]:  # DINOv3 License 1.b.i: ship a copy of the licence with derivative weights
            shutil.copyfile(ROOT / "pretrained/_h1/vit_base_patch16_dinov3.lvd1689m/LICENSE.md", sub / "licenses/DINOv3_LICENSE.md")
    notice.append("- fallback: torchvision ResNet-18 ImageNet weights (BSD-3-Clause), fine-tuned layer4 (B015)")
    (sub / "licenses/THIRD_PARTY_NOTICES.txt").write_text("\n".join(notice) + "\n")
    if require_cuda:  # GPU-runner packages carry a load-time canary (reference outputs from the CPU fp32 path)
        config["canary"] = _canary_refs(sub, n=4)
        (sub / "assets/config.json").write_text(json.dumps(config, indent=1))
    archive = out_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in sub.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(str(path.relative_to(sub)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    manifest = {
        "schema": 2,
        "candidate": name,
        "artifact": {"path": str(archive), "sha256": sha256(archive), "size": archive.stat().st_size},
        "members": {str(p.relative_to(sub)): sha256(p) for p in sorted(sub.rglob("*")) if p.is_file()},
        "model_checks": {m["name"]: m["checks"] for m in models},
        "fallback_fp16_storage_logit_max_abs_diff": fallback_diff,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "flip_tta": flip_tta,
        "training_runs": [str(r) for r in run_dirs],
        "weight_storage": f"{storage} in ZIP (fallback fp16), restored to fp32 at load; FP32 compute",
    }
    (out_dir / f"{name}_package_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("candidate", "artifact", "model_checks", "fallback_fp16_storage_logit_max_abs_diff")}, indent=1))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--flip-tta", action="store_true")
    parser.add_argument("--time-budget-s", type=float, default=3300.0)
    parser.add_argument("--weight-storage", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--onednn-fusion", action="store_true")
    parser.add_argument("--member-weights", type=Path, help="JSON list: one 40-vector of per-attribute weights per run dir")
    parser.add_argument("--plans", type=Path, help='JSON list [{"name": "A", "members": [0, 2]}, ...] in preference order')
    parser.add_argument("--rank-reserve-s", type=float, default=120.0)
    parser.add_argument("--plan-margin", type=float, default=1.0,
                        help="keep an earlier plan only if its projection <= margin x time left (e.g. 0.92 keeps 8%% headroom)")
    parser.add_argument("--variants", type=Path, help='JSON list [{"of": 1, "hw": [160, 80], "reference_run": dir}, ...]: extra models that reuse a base model\'s weights at another input size')
    parser.add_argument("--gpu-runner", action="store_true",
                        help="GPU-runner safety: require CUDA, never run fewer than two members, never use the R18 fallback (fail closed)")
    parser.add_argument("--optional-margin", type=float, help="drop a plan's optional trailing members when their projection > margin x time left (default: plan margin)")
    parser.add_argument("--layout", choices=("full", "blocks"), default="full",
                        help="blocks: ship prefix/ONE block/head graphs per member (int8 only; NNC compiles once per member)")
    parser.add_argument("--fast", default="",
                        help="comma list of GPU speed rewrites (blocks layout): " + ",".join(FAST_REWRITES)
                        + "; members they do not apply to are unchanged")
    parser.add_argument("--min-active-members", type=int, help="override (no-degrade packages: the size of plan A, with plans = [A] only)")
    parser.add_argument("--post", help='ensemble post-processing JSON, e.g. \'{"renorm": [[8, 20, 0.5]], "clip": 1e-5}\' (see upar/postproc.py)')
    args = parser.parse_args()
    weights = json.loads(args.member_weights.read_text()) if args.member_weights else None
    plans = json.loads(args.plans.read_text()) if args.plans else None
    build(args.run_dirs, args.out_dir, args.name, args.flip_tta, args.time_budget_s, args.weight_storage, args.onednn_fusion, weights, plans,
          args.rank_reserve_s, args.plan_margin, args.optional_margin,
          json.loads(args.variants.read_text()) if args.variants else None,
          require_cuda=args.gpu_runner, min_active_members=args.min_active_members or (2 if args.gpu_runner else 1), fail_closed=args.gpu_runner,
          post=json.loads(args.post) if args.post else None, layout=args.layout,
          fast=tuple(o for o in args.fast.split(",") if o))


if __name__ == "__main__":
    main()
