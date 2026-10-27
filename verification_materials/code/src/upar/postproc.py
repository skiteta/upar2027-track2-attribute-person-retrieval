"""Ensemble post-processing shared by run.py (an identical copy lives in the RUN_PY template), h1_replay, pkg_numeric_check
and gold_gate. Config: {"renorm": [[start, end, alpha], ...], "clip": eps}; absent/empty = identity.
renorm: within columns [start, end) push each row's probabilities `alpha` of the way toward summing to 1 (mutually exclusive
attribute groups, e.g. the 12 UpperBody colours); clip: bound probabilities to [eps, 1 - eps]. No fitted parameters and no
test-set statistics (error analysis 2026-10-12: UpperBody-colour alpha 0.5 + clip 1e-5 = +0.0009 LODO, 3/3 sources up)."""
from __future__ import annotations

import numpy as np


def apply_post(p: np.ndarray, post: dict | None) -> np.ndarray:
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


def entropy_weight_matrix(p: np.ndarray, w: np.ndarray, delta: float) -> np.ndarray:
    """Per-image attribute weights W[g,a] = w[a] * h[g,a] / mean_a h[g,a] with h = exp((1 - H(p)/ln 2) * delta): uncertain
    attribute predictions of an image count slightly less (hardness study 2026-10-12: delta 0.1875 with the UpperBody-colour
    post-processing gives +0.0028 LODO, all 3 sources up). Uses only the image's own probabilities."""
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    h = np.exp((1.0 - (-(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2.0))) * float(delta))
    h = h / h.mean(axis=1, keepdims=True)
    return (np.asarray(w, dtype=np.float64)[None, :] * h).astype(np.float32)


def mixed_distances_per_image(p: np.ndarray, queries: np.ndarray, W: np.ndarray, nll_lambda: float, epsilon: float) -> np.ndarray:
    """(1-l) * sum_a W[g,a] |q_a - p_ga| + l * sum_a W[g,a] Bernoulli-NLL, with per-image weights (same algebra as run.py)."""
    p = np.asarray(p, dtype=np.float32)
    lam = np.float32(nll_lambda)
    clipped = np.clip(p, np.float32(epsilon), np.float32(1.0 - epsilon))
    neg = -np.log1p(-clipped).astype(np.float32, copy=False)
    pos = -np.log(clipped).astype(np.float32, copy=False)
    base = (np.float32(1.0) - lam) * (p * W).sum(axis=1, dtype=np.float32) + lam * (neg * W).sum(axis=1, dtype=np.float32)
    delta = (np.float32(1.0) - lam) * W * (np.float32(1.0) - np.float32(2.0) * p) + lam * W * (pos - neg)
    return base[None, :] + np.asarray(queries, dtype=np.float32) @ delta.T
