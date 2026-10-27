"""H1 fine-tuned ViT attribute retrieval: time-budgeted tiers, FP32 compute, CPU or CUDA."""
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
from b121_blas import numpy_blas_threads
from b121_observe import runtime_snapshot
from b121_deadline import fixed_member_step
from b121_runtime import load_feature_head, apply_feature_rerank, validated_time_scale, runtime_budget_seconds

_T0 = time.time()
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG: dict[str, Any] | None = None
_MODELS: list[Any] = []
_FALLBACK = None
_DEVICE = None
_POOL = None
_GALLERY_FEATURES = None
_GALLERY_EVA_FEATURES = None
_GALLERY_HP_FEATURES = None
_GALLERY_MEMBER_STD = None
_B121_TIME_SCALE = validated_time_scale()


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
        self.feature_head = None
        self.last_features = None

    def parameters(self):
        return iter(self.params[0])

    def __call__(self, x):
        h, rope = self.prefix(x)
        for p in self.params:
            h = self.block(h, rope, *p)
        logits = self.head(h)
        if self.feature_head is not None:
            self.last_features = self.feature_head(h)
        return logits


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
    if spec.get('gallery_feature_head'):
        if not blocks or spec.get('flip_tta') or spec.get('backbone') != 'pe_core_l14_280':
            raise RuntimeError('feature head requires exact original-view PE block member')
        model.feature_head = load_feature_head(_HERE, spec, device, dtype)
    elif spec.get('backbone') == 'eva02_l14_252':
        if not blocks or spec.get('flip_tta'):
            raise RuntimeError('EVA feature head requires exact original-view EVA02 block member')
        model.feature_head = model.head.backbone
    elif spec.get('backbone') in ('dinov3_hp16_192', 'dinov3_hp16_224'):
        if not blocks or spec.get('flip_tta'):
            raise RuntimeError('DINOv3-H+ feature head requires exact original-view DINOv3-H+ block member')
        model.feature_head = lambda h_in, _norm=model.head.backbone.norm: _norm(h_in)[:, 5:].mean(dim=1)
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
    out_m = model if blocks else _maybe_fuse(model, spec, device)
    out_m._mean = torch.tensor(spec["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    out_m._std = torch.tensor(spec["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    out_m._correction = torch.log(torch.tensor(spec["pos_weight"], dtype=torch.float32, device=device))
    out_m._dtype = next(out_m.parameters()).dtype
    return out_m


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
    mean = getattr(model, "_mean", None)
    if mean is None:
        mean = torch.tensor(spec["mean"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = getattr(model, "_std", None)
    if std is None:
        std = torch.tensor(spec["std"], dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    x = ((x - mean) / std).contiguous()
    correction = getattr(model, "_correction", None)
    if correction is None:
        correction = torch.log(torch.tensor(spec["pos_weight"], dtype=torch.float32, device=x.device))
    dtype = getattr(model, "_dtype", None) or next(model.parameters()).dtype
    x = x.to(dtype)
    p = torch.sigmoid(model(x).float() - correction)
    if spec.get("flip_tta"):
        p = 0.5 * p + 0.5 * torch.sigmoid(model(x.flip(-1)).float() - correction)
    return p[:n].float().cpu().numpy()


def _predict_members(gallery: list[dict[str, Any]], members: list[int], deadline: float, can_switch: bool, optional: int = 0):
    """Run one plan (ordered member indices). Returns (probabilities, tier) or None when a later plan should run.
    The last `optional` members are a bonus for fast workers: whenever their projection exceeds optional_margin x time
    left they are dropped first, and the remaining members keep their progress (no restart)."""
    global _FALLBACK, _GALLERY_FEATURES, _GALLERY_EVA_FEATURES, _GALLERY_HP_FEATURES, _GALLERY_MEMBER_STD
    specs = [_CONFIG["models"][i] for i in members]
    models = [_MODELS[i] for i in members]
    total = len(gallery)
    min_images = int(_CONFIG.get("min_images_before_projection_gpu", 1024) if _DEVICE.type == "cuda" else _CONFIG["min_images_before_projection"])
    batch = int(_CONFIG["gpu_batch_size"] if _DEVICE.type == "cuda" else _CONFIG["cpu_batch_size"])
    probs = [np.empty((total, 40), dtype=np.float32) for _ in specs]
    if sum(bool(s.get('gallery_feature_head')) for s in specs) != 1:
        raise RuntimeError('exactly one PE pooled feature source required')
    _GALLERY_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)
    _GALLERY_EVA_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)
    _GALLERY_HP_FEATURES = np.full((total, 1280), np.nan, dtype=np.float32)
    active = len(specs)
    spent = [0.0] * (len(specs) + 1)  # per-model seconds, last slot = decode/transfer
    use_fallback = False
    done = 0
    fetch = ThreadPoolExecutor(max_workers=1) if _POOL is not None else None
    pending = fetch.submit(_decode_batch, gallery[0:batch]) if fetch else None
    if can_switch or optional or len(specs) != int(_CONFIG['min_active_members']):
        raise RuntimeError('locked ensemble requires one complete mandatory plan')
    for start in range(0, total, batch):
        fixed_member_step(elapsed=time.time()-_T0,
            total_budget=runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE),
            soft_prediction_deadline=deadline, projected_remaining=0.,
            active=active, required=int(_CONFIG['min_active_members']), remaining=total-start)
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
            if specs[i].get('gallery_feature_head'):
                feature = models[i].last_features
                if feature is None or feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < raw.shape[0]:
                    raise RuntimeError('missing or incomplete same-forward PE pooled features')
                _GALLERY_FEATURES[start:start + raw.shape[0]] = feature[:raw.shape[0]].float().cpu().numpy()
            elif specs[i].get('backbone') == 'eva02_l14_252':
                eva_feat = models[i].last_features
                if eva_feat is None or eva_feat.ndim != 2 or eva_feat.ndim != 2 or eva_feat.shape[1] != 1024 or eva_feat.shape[0] < raw.shape[0]:
                    raise RuntimeError('missing or incomplete same-forward EVA02 pooled features')
                _GALLERY_EVA_FEATURES[start:start + raw.shape[0]] = eva_feat[:raw.shape[0]].float().cpu().numpy()
            elif specs[i].get('backbone') in ('dinov3_hp16_192', 'dinov3_hp16_224'):
                hp_feat = models[i].last_features
                if hp_feat is None or hp_feat.ndim != 2 or hp_feat.shape[1] != 1280 or hp_feat.shape[0] < raw.shape[0]:
                    raise RuntimeError('missing or incomplete same-forward DINOv3-H+ pooled features')
                _GALLERY_HP_FEATURES[start:start + raw.shape[0]] = hp_feat[:raw.shape[0]].float().cpu().numpy()
            spent[i] += time.time() - t
        done = start + raw.shape[0]
        remaining = total - done
        left = deadline - (time.time() - _T0)
        def projected(k: int) -> float:
            return remaining * (spent[-1] + sum(spent[:k])) / done
        # on GPU the early shortcut waits for min_images as well (first batches carry kernel/JIT warm-up)
        extreme = done > 0 and projected(active) > 3.0 * max(left, 1.0) and (_DEVICE.type != "cuda" or done >= min_images)
        decision = fixed_member_step(
            elapsed=time.time()-_T0, total_budget=runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE),
            soft_prediction_deadline=deadline, projected_remaining=projected(active),
            active=active, required=int(_CONFIG['min_active_members']), remaining=remaining)
        if decision['projection_risk'] and (start // batch) % 10 == 0:
            _log('required_projection_warning', done=done,
                 projected_s=round(projected(active), 2), **decision)
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
    if not np.isfinite(_GALLERY_FEATURES).all() or not np.isfinite(_GALLERY_EVA_FEATURES).all() or not np.isfinite(_GALLERY_HP_FEATURES).all():
        raise RuntimeError('incomplete/nonfinite PE, EVA02, or DINOv3-H+ feature gallery')
    _GALLERY_MEMBER_STD = np.std(
        np.stack([_apply_post(p_m, _CONFIG.get('post')) for p_m in probs[:active]], axis=0),
        axis=0,
    ).astype(np.float32)
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
    global _GALLERY_MEMBER_STD
    if _GALLERY_MEMBER_STD is not None:
        _GALLERY_MEMBER_STD = np.ascontiguousarray(_GALLERY_MEMBER_STD[:, reorder])
    return np.ascontiguousarray(out[:, reorder])


def _b121_stage_log(event, **values):
    _log(event, **values, resources=runtime_snapshot())


def _b121_score_distances(probabilities, queries, attribute_names, ent_override=None, lam_override=None, err_gamma=1.0, uniform_weights=False, lam_neg_override=None, err_gamma_neg=None, q_pos_rho=0.0, dich_sym_loss=0.0, dich_pos_nll_add=0.0, pos_q_mul=1.0, _pre=None):
    names = list(_CONFIG["attribute_names"])
    reorder = np.asarray([names.index(n) for n in attribute_names], dtype=np.int64)
    if uniform_weights:
        weights = np.ones(len(attribute_names), dtype=np.float32)
    else:
        weights = np.asarray(_CONFIG["attribute_distance_weights"], dtype=np.float32)[reorder]
    eps = np.float32(_CONFIG["bernoulli_nll_epsilon"])
    lam = np.float32(_CONFIG["bernoulli_nll_lambda"] if lam_override is None else lam_override)
    lam0 = lam if lam_neg_override is None else np.float32(lam_neg_override)
    ent = (_CONFIG.get("post") or {}).get("entropy_delta") if ent_override is None else ent_override
    if _pre is not None:
        one_minus_ent, clipped, neg, pos = _pre
    else:
        one_minus_ent = None
        clipped = np.clip(probabilities, eps, np.float32(1.0) - eps)
        neg = -np.log1p(-clipped).astype(np.float32, copy=False)
        pos = -np.log(clipped).astype(np.float32, copy=False)
    if ent:  # per-image entropy weights (identical to upar/postproc.py entropy_weight_matrix); absent = per-attribute weights
        if one_minus_ent is None:
            q = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
            one_minus_ent = 1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))
        h = np.exp(one_minus_ent * float(ent))
        weights = (weights.astype(np.float64)[None, :] * (h / h.mean(axis=1, keepdims=True))).astype(np.float32)
    eg_pos = float(err_gamma)
    eg_neg = eg_pos if err_gamma_neg is None else float(err_gamma_neg)
    pmul = np.float32(pos_q_mul)
    if eg_pos != 1.0 or eg_neg != 1.0 or lam != lam0 or pmul != 1.0 or float(q_pos_rho) != 0.0 or float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:
        f0 = (clipped ** np.float32(eg_neg)).astype(np.float32, copy=False) if eg_neg != 1.0 else probabilities
        f1 = ((np.float32(1.0) - clipped) ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else (np.float32(1.0) - probabilities)
        loss0 = (np.float32(1.0) - lam0) * f0 + lam0 * neg
        loss1 = pmul * ((np.float32(1.0) - lam) * f1 + lam * pos)
        if float(dich_sym_loss) > 0.0 or float(dich_pos_nll_add) != 0.0:
            dich_idx = [i for i, n in enumerate(attribute_names) if str(n) in ("Gender-Female", "UpperBody-Length-Short", "LowerBody-Length-Short")]
            if float(dich_sym_loss) > 0.0 and dich_idx:
                s_d = np.float32(dich_sym_loss)
                lam_d0 = np.float32(float(lam) + float(dich_pos_nll_add))
                f0_pos = (clipped[:, dich_idx] ** np.float32(eg_pos)).astype(np.float32, copy=False) if eg_pos != 1.0 else probabilities[:, dich_idx]
                loss0_pos = pmul * ((np.float32(1.0) - lam_d0) * f0_pos + lam_d0 * neg[:, dich_idx])
                loss0 = loss0.copy()
                loss0[:, dich_idx] = (np.float32(1.0) - s_d) * loss0[:, dich_idx] + s_d * loss0_pos
            if float(dich_pos_nll_add) != 0.0 and dich_idx:
                dl_p = np.float32(dich_pos_nll_add)
                loss1 = loss1.copy()
                loss1[:, dich_idx] = loss1[:, dich_idx] + pmul * dl_p * (pos[:, dich_idx] - f1[:, dich_idx])
        d0 = weights * loss0
        d1 = weights * loss1
        base = d0.sum(axis=1, dtype=np.float32)
        delta = d1 - d0
    else:
        base = (np.float32(1.0) - lam) * (probabilities * weights).sum(axis=1, dtype=np.float32) \
            + lam * (neg * weights).sum(axis=1, dtype=np.float32)
        delta = (np.float32(1.0) - lam) * weights * (np.float32(1.0) - np.float32(2.0) * probabilities) \
            + lam * weights * (pos - neg)
    distances = np.empty((len(queries), len(probabilities)), dtype=np.float32)
    for start in range(0, len(queries), 64):
        distances[start:start + 64] = base[None, :] + queries[start:start + 64] @ delta.T
    if not np.isfinite(distances).all():
        raise RuntimeError("non-finite distances")
    return distances


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    attribute_names = list(sample["attribute_names"])
    probabilities = predict_attributes(list(sample["gallery"]), attribute_names)
    queries = np.asarray(sample["queries"], dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 40:
        raise ValueError("queries must be Q x 40")
    _b121_stage_log("score_enter", queries=len(queries), gallery=len(probabilities))
    with numpy_blas_threads(2) as blas:
        _b121_stage_log("score_blas", numpy_blas=blas)
        distances = _b121_score_distances(probabilities, queries, attribute_names)
    _b121_stage_log("score_done", queries=len(queries), gallery=len(probabilities))
    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE):
        raise RuntimeError('no remaining time for mandatory feature reranking')
    _b121_stage_log("graph_enter", arithmetic="CUDA_FP32_unit_dot_TF32_off", no_cpu_fallback=True)
    distances = apply_feature_rerank(
        distances, _GALLERY_FEATURES, probabilities, _b121_stage_log,
        queries, attribute_names, _b121_score_distances,
        _GALLERY_EVA_FEATURES, _GALLERY_HP_FEATURES, _GALLERY_MEMBER_STD,
    )
    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE) + 6.0:
        raise RuntimeError('feature reranking exceeded fixed total budget; fail closed')
    _log("ranked", queries=len(queries), gallery=len(probabilities))
    return {"distances": distances}
