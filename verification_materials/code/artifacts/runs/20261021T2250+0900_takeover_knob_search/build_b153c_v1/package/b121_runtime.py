"""Fixed no-fit gallery feature agreement; exact local implementation."""
import hashlib
import math
import os
import time
import numpy as np
import torch
from b121_graph import build_mutual_graph
from b121_agreement import retain_identical_predicted_profiles, smooth_with_original_degree
from b121_blas import numpy_blas_threads
from b121_fp32_cosine_graph import build_feature_graph
from b127_jaccard_prob_diffusion import run_b132c_pentj_body_rerank
from b122c_constant import constant_rescue, smooth_with_rescue, CONSTANT

def validated_time_scale():
    scale = float(os.environ.get('UPAR_TIME_SCALE', '1'))
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError('UPAR_TIME_SCALE must be finite in [1,4]; local replay only')
    return scale

def runtime_budget_seconds(config, scale):
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError('invalid frozen local time scale')
    budget = float(config['time_budget_s'])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('invalid time budget')
    return budget * scale

def load_feature_head(here, spec, device, dtype):
    path = os.path.join(here, spec['gallery_feature_head'])
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if digest != spec['gallery_feature_sha256']:
        raise RuntimeError('PE feature-head bytes differ from fixed package')
    return torch.jit.load(path, map_location=device).eval().to(dtype)

def encode_unit_features(features):
    """Rowwise stable L2, then affine to existing graph API; no cohort fit."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero feature norm")
    scaled = x / scale
    unit = scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    if not np.isfinite(unit).all() or np.any(np.abs(unit) > 1 + 1e-12):
        raise ValueError("invalid unit feature")
    # Only the mathematical [-1,1] boundary's FP64 roundoff is clipped.
    return (np.clip(unit, -1, 1) + 1) / 2

def apply_feature_rerank(distances, features, probabilities, log, queries, attribute_names, score_fn, eva_features, hp_features, member_std):
    if features is None or features.shape != (len(probabilities), 1024):
        raise RuntimeError('missing complete same-forward PE pooled gallery')
    if eva_features is None or eva_features.shape != (len(probabilities), 1024):
        raise RuntimeError('missing complete same-forward EVA02 pooled gallery')
    if hp_features is None or hp_features.shape != (len(probabilities), 1280):
        raise RuntimeError('missing complete same-forward DINOv3-H+ pooled gallery')
    if member_std is None or member_std.shape != probabilities.shape:
        raise RuntimeError('missing complete member_std gallery')
    if not torch.cuda.is_available():
        raise RuntimeError('mandatory CUDA FP32 unit-dot graph unavailable; no CPU fallback')
    started = time.time()
    log('feature_rerank_enter', device='cuda', dtype='float32', normalization='float64', tf32=False)
    with numpy_blas_threads(2) as blas:
        output = run_b132c_pentj_body_rerank(
            probabilities, features, eva_features, hp_features, member_std,
            queries, attribute_names, score_fn, log,
        )
    if not np.isfinite(output).all() or output.shape != distances.shape:
        raise RuntimeError('invalid final gated distances')
    log('feature_reranked', total_seconds=round(time.time()-started, 4),
        extra_encoder_forward=False, extra_pool_head_forward=True, numpy_blas=blas,
        graph_backend='CUDA_FP32_unit_dot_TF32_off_CPU_deterministic_topk')
    return output
