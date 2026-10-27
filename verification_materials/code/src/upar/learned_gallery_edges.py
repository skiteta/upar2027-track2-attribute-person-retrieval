"""Source-fitted one-bit edge rescue; no labels or dataset metadata at inference.

The B121 identical-predicted-profile graph is retained exactly. Only reciprocal
edges whose 40 predicted bits differ once can receive additional mass. A small
fixed ridge logistic model estimates exact-profile agreement from image and
prediction evidence. This is not identity inference or hidden-data fitting.
"""
from __future__ import annotations

import numpy as np

from .fixed_gallery_agreement import retain_identical_predicted_profiles, smooth_with_original_degree
from .fixed_gallery_rerank import _validate_graph

FEATURE_NAMES = (
    "cosine_unit_interval", "mean_abs_probability_difference", "max_abs_probability_difference",
    "disagreeing_bit_probability_difference", "disagreeing_bit_min_margin", "disagreeing_bit_max_margin",
    "mean_endpoint_confidence",
) + tuple(f"disagreeing_attribute_{i:02d}" for i in range(40))
RIDGE = .01


def expit(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x, np.float64)))


def unit_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero feature norm")
    x = x / scale
    return x / np.sqrt(np.sum(x * x, axis=1, keepdims=True))


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


def pair_features(features, probabilities, src, dst):
    """Symmetric 47 inputs; no queries, labels, source IDs or file paths."""
    p = probabilities_checked(probabilities).astype(np.float64)
    f = unit_features(features)
    src, dst = np.asarray(src), np.asarray(dst)
    if src.ndim != 1 or dst.shape != src.shape or not np.issubdtype(src.dtype, np.integer) or not np.issubdtype(dst.dtype, np.integer):
        raise ValueError("aligned integer pair vectors required")
    if len(f) != len(p) or np.any(src < 0) or np.any(dst < 0) or np.any(src >= len(p)) or np.any(dst >= len(p)) or np.any(src == dst):
        raise ValueError("pair/feature row mismatch or self edge")
    out = np.empty((len(src), len(FEATURE_NAMES)), np.float64)
    for start in range(0, len(src), 4096):
        end = min(start + 4096, len(src))
        s, d = src[start:end], dst[start:end]
        a, b = p[s], p[d]
        mismatch = (a >= .5) != (b >= .5)
        if np.any(mismatch.sum(axis=1) != 1):
            raise ValueError("pair is not an exactly-one-bit disagreement")
        diff = np.abs(a - b)
        attr = mismatch.argmax(axis=1)
        row = np.arange(len(s))
        ma, mb = np.abs(2 * a[row, attr] - 1), np.abs(2 * b[row, attr] - 1)
        cosine = np.einsum("ij,ij->i", f[s], f[d])
        out[start:end, :7] = np.column_stack((
            (np.clip(cosine, -1, 1) + 1) / 2, diff.mean(axis=1), diff.max(axis=1),
            diff[row, attr], np.minimum(ma, mb), np.maximum(ma, mb),
            (np.abs(2 * a - 1).mean(axis=1) + np.abs(2 * b - 1).mean(axis=1)) / 2,
        ))
        out[start:end, 7:] = mismatch
    return out


def equal_source_weights(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("nonempty source groups required")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) != 2:
        raise ValueError("exactly two source-fit groups required")
    return 1.0 / (len(unique) * counts[inverse])


def logistic_objective(theta, x, y, weights, ridge=RIDGE):
    z = x @ theta[:-1] + theta[-1]
    error = weights * (expit(z) - y)
    value = np.dot(weights, np.logaddexp(0, z) - y * z) + .5 * ridge * np.dot(theta[:-1], theta[:-1])
    grad = np.r_[x.T @ error + ridge * theta[:-1], error.sum()]
    return float(value), grad


def fit_pair_model(x, labels, groups):
    """One fixed CPU fit: equal source mass, no class rebalancing or HPO."""
    x, y = np.asarray(x, np.float64), np.asarray(labels, np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x),) or np.asarray(groups).shape != y.shape:
        raise ValueError("unaligned pair training inputs")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)) or not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
        raise ValueError("bounded inputs and binary exact-GT-profile labels required")
    if len(np.unique(y)) != 2:
        raise ValueError("both target classes required; no implicit constant fallback")
    weights = equal_source_weights(groups)
    initial = np.zeros(x.shape[1] + 1)
    prevalence = float(np.dot(weights, y))
    initial[-1] = np.log(prevalence / (1 - prevalence))
    theta = initial
    converged = False
    extended = np.column_stack((x, np.ones(len(x))))
    for iteration in range(200):
        value, gradient = logistic_objective(theta, x, y, weights)
        if np.max(np.abs(gradient)) <= 1e-8:
            converged = True
            break
        p = expit(extended @ theta)
        curvature = weights * p * (1-p)
        hessian = extended.T @ (curvature[:, None] * extended)
        hessian[:-1, :-1] += RIDGE * np.eye(x.shape[1])
        direction = np.linalg.solve(hessian, -gradient)
        slope = float(np.dot(gradient, direction))
        if not np.isfinite(direction).all() or slope >= 0:
            raise RuntimeError("Newton direction is not finite/descent")
        step = 1.
        for _ in range(50):
            proposed = theta + step * direction
            updated = logistic_objective(proposed, x, y, weights)[0]
            if updated <= value + 1e-4 * step * slope:
                break
            step *= .5
        else:
            raise RuntimeError("fixed Newton line search failed")
        theta = proposed
    if not converged or not np.isfinite(theta).all():
        raise RuntimeError("fixed Newton pair fit did not converge in 200 iterations")
    p = expit(x @ theta[:-1] + theta[-1])
    return {"schema": 1, "method": "one_bit_rescue_ridge_logistic_v1", "feature_names": list(FEATURE_NAMES),
            "coefficient": theta[:-1].tolist(), "intercept": float(theta[-1]), "ridge_mean_loss": RIDGE,
            "class_balance": False, "source_balance": "each of two source domains has total mass .5",
            "fit_pairs_undirected": len(y), "fit_prevalence": prevalence, "iterations": iteration,
            "training_brier": float(np.dot(weights, (p - y) ** 2)), "training_objective": float(value),
            "training_not_confirmation": True}


def predict_pair_model(model, x):
    if model.get("method") != "one_bit_rescue_ridge_logistic_v1" or model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("unknown pair model or feature order")
    w = np.asarray(model["coefficient"], np.float64)
    intercept = float(model["intercept"])
    x = np.asarray(x, np.float64)
    if w.shape != (len(FEATURE_NAMES),) or x.ndim != 2 or x.shape[1] != len(w) or not np.isfinite(w).all() or not np.isfinite(intercept) or not np.isfinite(x).all():
        raise ValueError("invalid pair model or inputs")
    return expit(x @ w + intercept)


def rescue_edge_weights(features, probabilities, neighbours, counts, model):
    """Return weights aligned with original graph, only on one-bit edges."""
    p = probabilities_checked(probabilities)
    n, c = _validate_graph(neighbours, counts, len(p))
    src, dst = one_bit_pairs(p, n, c, undirected=True)
    r = predict_pair_model(model, pair_features(features, p, src, dst))
    result = np.zeros(n.shape, np.float64)
    # Assign each undirected result to its two orientations, avoiding roundoff
    # asymmetry and never allocating a gallery-square lookup table.
    for offset in range(n.shape[1]):
        take = n[src, offset] == dst
        result[src[take], offset] = r[take]
        take = n[dst, offset] == src
        result[dst[take], offset] = r[take]
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
