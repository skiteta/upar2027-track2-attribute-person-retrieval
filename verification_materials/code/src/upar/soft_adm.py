"""Differentiable *batch approximation* to Track-2 average degree of match.

For each exact-match position, replace the indicator that a gallery image
precedes it with a sigmoid. Normalize Hamming agreement using this training
gallery only, then average prefix agreement at exact matches as in
metrics_strict.py. This is neither an unbiased full-gallery estimator nor a
proven upper/lower bound. At vanishing temperature and distinct distances it
converges to the exact metric on the SAME gallery. No hidden data are used.
"""
from __future__ import annotations

import math

import torch


def soft_adm_per_query(distances, queries, gallery_labels, *, temperature=1.0, pair_chunk=256):
    """Return one smooth ADM per query (distance: smaller is better).

    Every query must have an exact gallery match. The positive's own prefix
    membership is exactly one, not sigmoid(0)=0.5. Other tied items have soft
    half-membership; unlike the official stable sort this is permutation
    invariant. Targets are detached, never learned or estimated from test.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if pair_chunk < 1:
        raise ValueError("pair_chunk must be positive")
    if distances.ndim != 2 or queries.ndim != 2 or gallery_labels.ndim != 2:
        raise ValueError("distances and labels must be matrices")
    if distances.shape != (len(queries), len(gallery_labels)) or queries.shape[1] != gallery_labels.shape[1]:
        raise ValueError("distance/label shapes do not align")
    if not queries.numel() or not gallery_labels.numel():
        raise ValueError("empty query or gallery")
    # FP32 for AMP training, retain FP64 for numerical gradient/metric tests.
    dtype = torch.float64 if distances.dtype == torch.float64 else torch.float32
    d = distances.to(dtype)
    with torch.no_grad():
        q, y = queries.to(dtype), gallery_labels.to(dtype)
        if not bool((((q == 0) | (q == 1)).all() & ((y == 0) | (y == 1)).all())):
            raise ValueError("soft ADM needs complete binary labels (no mixup/masked labels)")
        hamming = q @ (1 - y).T + (1 - q) @ y.T
        exact = hamming == 0
        counts = exact.sum(1)
        if bool((counts == 0).any()):
            raise ValueError("each query needs at least one exact match")
        # max agreement is 1 because exact matches exist; width cancels.
        mean_hamming = hamming.mean(1, keepdim=True)
        quality = (1 - hamming / mean_hamming.clamp_min(torch.finfo(dtype).eps)).clamp(0, 1)
        pair_q, pair_p = exact.nonzero(as_tuple=True)
    totals = d.sum(1) * 0  # keep a differentiable zero even in degenerate cases
    gallery_ids = torch.arange(d.shape[1], device=d.device)
    for start in range(0, pair_q.numel(), pair_chunk):
        qi, pi = pair_q[start:start + pair_chunk], pair_p[start:start + pair_chunk]
        before = torch.sigmoid((d[qi, pi, None] - d[qi]) / temperature)
        before = torch.where(gallery_ids[None, :] == pi[:, None], torch.ones_like(before), before)
        prefix_adm = (before * quality[qi]).sum(1) / before.sum(1)
        totals = totals.index_add(0, qi, prefix_adm)
    return totals / counts


def soft_adm_loss(logits, labels, log_pw, *, temperature=1.0, nll_lambda=0.25, row_weight=None):
    """Legacy ranker's prior-corrected distance with the smooth ADM objective.

    Post-processing/reliability remain unchanged from the legacy rank loss
    here, isolating loss replacement. BCE remains the main training objective.
    """
    p = torch.sigmoid(logits.float() - log_pw).clamp(1e-6, 1 - 1e-6)
    q = labels.float()
    l1 = q @ (1 - p).T + (1 - q) @ p.T
    nll = -(q @ torch.log(p).T + (1 - q) @ torch.log1p(-p).T)
    d = (1 - nll_lambda) * l1 + nll_lambda * nll
    losses = 1 - soft_adm_per_query(d, q, q, temperature=temperature)
    if row_weight is None:
        return losses.mean()
    return (losses * row_weight).sum() / row_weight.sum().clamp_min(1e-12)
