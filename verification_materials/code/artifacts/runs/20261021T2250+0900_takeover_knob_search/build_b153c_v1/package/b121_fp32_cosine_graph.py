"""One explicitly different numeric recipe: FP32 unit-dot exhaustive search.

Stable row normalization remains FP64; CUDA dot products are FP32 with TF32
disabled. No affine .5 offset or 1-dot subtraction, no approximate neighbour
index, learned parameter, query/GT access, or silent fallback. This is NOT an
exact-numeric substitute for the original FP64 graph without observed parity.
"""
from __future__ import annotations

from contextlib import contextmanager
import time

import numpy as np
import torch

from b121_graph import _deterministic_smallest, _positive_int
from b121_gpu_graph import _mutual_from_directed


def unit_features_fp64(features):
    x = np.array(features, dtype=np.float64, copy=True, order="C")
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero feature norm")
    np.divide(x, scale, out=x)
    np.divide(x, np.sqrt(np.sum(x*x, axis=1, keepdims=True)), out=x)
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1+1e-12):
        raise ValueError("invalid unit features")
    return x


def encode_unit_features_inplace(features):
    """Diagnostic only: recover original encoding from exact pre-cast units.

    The actual FP32 graph does not use this affine encoding. This lets the
    shared fixed benchmark verify that pre-cast normalization is unchanged.
    """
    x = unit_features_fp64(features)
    np.clip(x, -1, 1, out=x)
    np.add(x, 1, out=x)
    np.divide(x, 2, out=x)
    return x


@contextmanager
def strict_fp32():
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("strict FP32 setting not effective")
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def build_feature_graph(features, *, k=20, chunk_size=256, device="cuda", allow_gpu=False, log=None):
    _positive_int(k, "k")
    _positive_int(chunk_size, "chunk_size")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be exactly cpu or cuda")
    if device == "cuda" and not allow_gpu:
        raise PermissionError("explicit root GPU admission required")
    emit = log if log is not None else lambda *args, **kwargs: None
    started = time.monotonic()
    emit("feature_encode_enter", device=device, normalization="FP64_then_FP32_cast")
    unit = unit_features_fp64(features).astype(np.float32)
    n, width = len(unit), min(k, len(unit)-1)
    emit("feature_encode_done", rows=n, width=unit.shape[1], seconds=time.monotonic()-started)
    if width == 0:
        return np.empty((n, 0), np.int64), np.zeros(n, np.int64)
    emit("feature_graph_enter", device=device, arithmetic="FP32_unit_dot_TF32_off", chunk_size=chunk_size)
    x = torch.from_numpy(unit).to(device=device, dtype=torch.float32)
    directed = np.empty((n, width), np.int64)
    block = min(chunk_size, n-1)
    with strict_fp32(), torch.inference_mode():
        for start in range(0, n, block):
            end = min(n, start+block)
            negative_dot = -(x[start:end] @ x.T).cpu().numpy()
            negative_dot[np.arange(end-start), np.arange(start, end)] = np.inf
            directed[start:end] = _deterministic_smallest(negative_dot, width)
            if start == 0 or end == n or (start // block) % 16 == 0:
                emit("feature_graph_progress", done=end, total=n, seconds=time.monotonic()-started)
            del negative_dot
    del x, unit
    neighbours, counts = _mutual_from_directed(directed)
    emit("feature_graph_done", rows=n, edges=int(counts.sum()), seconds=time.monotonic()-started)
    return neighbours, counts
