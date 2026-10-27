"""Fixed source-held constant one-bit rescue; no fit or extra feature inference.

The constant is the exact saved equal-source OOF target prevalence, not a
held-score-selected hyperparameter. Original B121 graph and arithmetic survive.
"""
import numpy as np
from b121_graph import _validate_graph
from b121_agreement import retain_identical_predicted_profiles, smooth_with_original_degree

CONSTANT = 0.5599229900622915
ORIGIN_MODEL_SHA256 = "8805a189583d5843110cd9855c531bb1fb42e72478146849ef998339ea92eaac"


def probabilities_checked(probabilities):
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 40 or not len(p) or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("floating gallery-by-40 probabilities required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite probabilities in [0,1] required")
    return p


def one_bit_pairs(probabilities, neighbours, counts, *, undirected=True):
    p = probabilities_checked(probabilities)
    n, c = _validate_graph(neighbours, counts, len(p))
    valid = np.arange(n.shape[1])[None, :] < c[:, None]
    src = np.broadcast_to(np.arange(len(p))[:, None], n.shape)[valid]
    dst = n[valid]
    one = np.count_nonzero((p[src] >= .5) != (p[dst] >= .5), axis=1) == 1
    if undirected:
        one &= src < dst
    return src[one], dst[one]


def constant_rescue(probabilities, neighbours, counts):
    n, c = _validate_graph(neighbours, counts, len(probabilities))
    src, dst = one_bit_pairs(probabilities, n, c, undirected=False)
    result = np.zeros(n.shape, np.float64)
    for offset in range(n.shape[1]):
        take = n[src, offset] == dst
        result[src[take], offset] = CONSTANT
    return result


def smooth_with_rescue(distances, probabilities, neighbours, counts, rescue_weights, *, alpha=.1,
                       query_chunk_size=64, gallery_chunk_size=256):
    """B121 exactly + alpha/original_degree * sum r_ij*(d_j-d_i)."""
    p = probabilities_checked(probabilities)
    n, c = _validate_graph(neighbours, counts, len(p))
    r = np.asarray(rescue_weights, np.float64)
    if r.shape != n.shape or not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("bounded graph-aligned rescue weights required")
    valid = np.arange(n.shape[1])[None, :] < c[:, None]
    one = np.count_nonzero((p[:, None, :] >= .5) != (p[np.maximum(n, 0)] >= .5), axis=2) == 1
    if np.any(r[~(valid & one)] != 0):
        raise ValueError("rescue mass outside valid one-bit edges")
    src = np.broadcast_to(np.arange(len(p))[:, None], n.shape)[valid]
    dst = n[valid]
    keys = src * len(p) + dst
    order, reverse = np.argsort(keys), np.argsort(dst * len(p) + src)
    if not np.array_equal(r[valid][order], r[valid][reverse]):
        raise ValueError("asymmetric rescue mass")
    base_n, base_c = retain_identical_predicted_profiles(p, n, c)
    result = smooth_with_original_degree(distances, base_n, base_c, c, alpha=alpha,
                                         query_chunk_size=query_chunk_size, gallery_chunk_size=gallery_chunk_size)
    if alpha == 0 or not np.any(r):
        return result
    d = np.asarray(distances)
    for start in range(0, len(p), gallery_chunk_size):
        active = np.flatnonzero(r[start:start+gallery_chunk_size].sum(axis=1) > 0) + start
        if not len(active):
            continue
        safe = np.maximum(n[active], 0)
        weight = r[active].astype(d.dtype)
        factor = np.asarray(alpha, dtype=d.dtype) / c[active].astype(d.dtype)
        for qstart in range(0, len(d), query_chunk_size):
            qend = min(qstart + query_chunk_size, len(d))
            before = d[qstart:qend, active]
            delta = ((d[qstart:qend, safe] - before[:, :, None]) * weight[None, :, :]).sum(axis=2, dtype=d.dtype)
            result[qstart:qend, active] += factor[None, :] * delta
    if not np.isfinite(result).all():
        raise ValueError("nonfinite rescue smoothing")
    return result
