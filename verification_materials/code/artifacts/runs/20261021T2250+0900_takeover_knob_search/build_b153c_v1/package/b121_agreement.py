"""Fixed semantic agreement gate for a pre-existing mutual image-feature graph.

No labels, query bank, identities, paths, learned threshold or cohort statistics.
All40 predicted bits must agree at .5. Removed edges' mass is not redistributed.
"""
from __future__ import annotations

import math

import numpy as np

from b121_graph import _positive_int, _validate_graph


def retain_identical_predicted_profiles(probabilities, neighbours, counts):
    """Return a compact reciprocal subgraph; never add or re-order an edge."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 40 or not len(p) or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("floating gallery-by-40 probabilities required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite probabilities in [0,1] required")
    index, degree = _validate_graph(neighbours, counts, len(p))
    bits = p >= .5
    valid = np.arange(index.shape[1])[None, :] < degree[:, None]
    keep = valid & np.all(bits[:, None, :] == bits[np.maximum(index, 0)], axis=2)
    retained = np.full_like(index, -1)
    number = keep.sum(axis=1, dtype=np.int64)
    for row in range(len(p)):
        retained[row, :number[row]] = index[row, keep[row]]
    _validate_graph(retained, number, len(p))
    return retained, number


def smooth_with_original_degree(distances, neighbours, counts, original_counts, *, alpha=.1,
                                query_chunk_size=64, gallery_chunk_size=256):
    """One step d + alpha/original_degree * sum_kept(d_neighbor-d).

Actual arithmetic is d + (alpha * kept/original) * (mean_kept(d)-d),
which keeps the old implementation's all-retained arithmetic exactly. Isolated
or zero-kept items and alpha0 are bitwise unchanged. Effective alpha never rises.
"""
    _positive_int(query_chunk_size, "query_chunk_size")
    _positive_int(gallery_chunk_size, "gallery_chunk_size")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("finite alpha in [0,1] required")
    d = np.asarray(distances)
    if d.ndim != 2 or not all(d.shape) or not np.issubdtype(d.dtype, np.floating) or not np.isfinite(d).all():
        raise ValueError("finite nonempty floating query-by-gallery distances required")
    index, degree = _validate_graph(neighbours, counts, d.shape[1])
    original = np.asarray(original_counts)
    if original.shape != degree.shape or not np.issubdtype(original.dtype, np.integer):
        raise ValueError("aligned integer original degrees required")
    if np.any(original < degree) or np.any(original > index.shape[1]):
        raise ValueError("original degree must cover retained degree within original graph width")
    result = d.copy()
    if alpha == 0 or not np.any(degree):
        return result
    for gstart in range(0, d.shape[1], gallery_chunk_size):
        gend = min(d.shape[1], gstart + gallery_chunk_size)
        active = np.flatnonzero(degree[gstart:gend]) + gstart
        if not len(active):
            continue
        selected = index[active]
        valid = selected >= 0
        safe_index = np.maximum(selected, 0)
        denominator = degree[active].astype(d.dtype)
        effective_alpha = np.asarray(alpha, dtype=d.dtype) * (denominator / original[active].astype(d.dtype))
        for qstart in range(0, d.shape[0], query_chunk_size):
            qend = min(d.shape[0], qstart + query_chunk_size)
            values = d[qstart:qend, safe_index]
            values = np.where(valid[None, :, :], values, np.asarray(0, dtype=d.dtype))
            mean = values.sum(axis=2, dtype=d.dtype) / denominator[None, :]
            before = d[qstart:qend, active]
            result[qstart:qend, active] = before + effective_alpha[None, :] * (mean - before)
    if not np.isfinite(result).all():
        raise ValueError("gated score propagation overflowed")
    return result
