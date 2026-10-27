"""Opt-in exhaustive FP64 graph GEMM; no model, labels, tuning, or fallback.

This is a proposed runtime revision, NOT the submitted implementation. CUDA
GEMM can round differently from NumPy BLAS. Exact observed graph/rank parity
must be checked on real saved features before any package adopts this path.
The deterministic CPU top-k/tie and mutual-graph definitions remain unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from b121_graph import _deterministic_smallest, _positive_int


def encode_unit_features_inplace(features):
    """Same FP64 rowwise operations as original, but reuse one owned matrix.

    Input is never mutated, including an already-FP64 input or a memmap. The
    temporary squared matrix is released after reduction; encoded/unit/scaled
    matrices no longer coexist. This is not a fitted transformation.
    """
    x = np.array(features, dtype=np.float64, copy=True, order="C")
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero feature norm")
    np.divide(x, scale, out=x)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    np.divide(x, norm, out=x)
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1 + 1e-12):
        raise ValueError("invalid unit feature")
    np.clip(x, -1, 1, out=x)
    np.add(x, 1, out=x)
    np.divide(x, 2, out=x)
    return x


def _mutual_from_directed(directed):
    n = len(directed)
    mutual = np.any(directed[directed] == np.arange(n)[:, None, None], axis=2)
    counts = mutual.sum(axis=1, dtype=np.int64)
    neighbours = np.full(directed.shape, -1, dtype=np.int64)
    for row in range(n):
        neighbours[row, :counts[row]] = directed[row, mutual[row]]
    return neighbours, counts


def build_feature_graph(features, *, k=20, chunk_size=256, device="cuda",
                        allow_gpu=False, log=None):
    """Exact exhaustive search, not approximate index or FP32 substitute.

    Device 'cpu' exists only for small contract tests. CUDA requires explicit
    admission; a CUDA error is surfaced, never silently rerun on CPU. The GPU
    holds one GxD FP64 feature matrix plus one chunkxG dot-product block. Norms,
    post-GEMM arithmetic and top-k/tie sorting are NumPy FP64 as before.
    """
    _positive_int(k, "k")
    _positive_int(chunk_size, "chunk_size")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be exactly cpu or cuda")
    if device == "cuda" and not allow_gpu:
        raise PermissionError("explicit root GPU admission required")
    emit = log if log is not None else lambda *args, **kwargs: None
    started = time.monotonic()
    emit("feature_encode_enter", device=device)
    encoded = encode_unit_features_inplace(features)
    n = len(encoded)
    width = min(k, n - 1)
    emit("feature_encode_done", rows=n, width=encoded.shape[1], seconds=time.monotonic()-started)
    if width == 0:
        return np.empty((n, 0), np.int64), np.zeros(n, np.int64)
    norms = np.einsum("ij,ij->i", encoded, encoded)
    if not np.isfinite(norms).all():
        raise ValueError("feature magnitude overflows")
    emit("feature_graph_enter", device=device, arithmetic="float64", chunk_size=chunk_size)
    x = torch.from_numpy(encoded).to(device=device, dtype=torch.float64)
    directed = np.empty((n, width), dtype=np.int64)
    block = min(chunk_size, n - 1)
    with torch.inference_mode():
        for start in range(0, n, block):
            end = min(n, start + block)
            dots = (x[start:end] @ x.T).cpu().numpy()
            # Match original CPU expression and self exclusion; GPU only GEMM.
            distances = norms[start:end, None] + norms[None, :] - 2.0 * dots
            np.maximum(distances, 0.0, out=distances)
            distances[np.arange(end-start), np.arange(start, end)] = np.inf
            directed[start:end] = _deterministic_smallest(distances, width)
            if start == 0 or end == n or (start // block) % 16 == 0:
                emit("feature_graph_progress", done=end, total=n, seconds=time.monotonic()-started)
            del distances, dots
    del x, encoded
    neighbours, counts = _mutual_from_directed(directed)
    emit("feature_graph_done", rows=n, edges=int(counts.sum()), seconds=time.monotonic()-started)
    return neighbours, counts
