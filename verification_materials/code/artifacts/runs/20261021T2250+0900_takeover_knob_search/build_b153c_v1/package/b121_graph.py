"""One fixed, label-free mutual-neighbour score-smoothing mechanism.

The runner, not this reusable toy-testable module, must freeze k=20, alpha=.1,
the existing reliability vector and probability producer. This is not the
Jaccard re-ranking algorithm of Zhong et al.; there is one uniform averaging
step. No labels, query bank, paths, identities, fitting or centering enter the
gallery graph. Exact exhaustive nearest-neighbour search uses bounded blocks.
"""

from __future__ import annotations

import math

import numpy as np


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _squared_l2_block(x, norms, start, end):
    # FP64 exhaustive distance calculation, not an approximate-neighbour index.
    # Roundoff can put coincident vectors slightly below zero; clamp only that
    # mathematical lower bound, without fitted statistics or centering.
    result = norms[start:end, None] + norms[None, :] - 2.0 * (x[start:end] @ x.T)
    np.maximum(result, 0.0, out=result)
    result[np.arange(end - start), np.arange(start, end)] = np.inf
    return result


def _deterministic_smallest(distances, k):
    """Exact k lowest, ordered by (computed distance, original gallery index)."""
    partition = np.argpartition(distances, k - 1, axis=1)[:, :k]
    result = np.empty((len(distances), k), dtype=np.int64)
    for row, selected in enumerate(partition):
        cutoff = distances[row, selected].max()
        below = np.flatnonzero(distances[row] < cutoff)
        tied = np.flatnonzero(distances[row] == cutoff)
        chosen = np.concatenate((below, tied[: k - len(below)]))
        order = np.lexsort((chosen, distances[row, chosen]))
        result[row] = chosen[order]
    return result


def build_mutual_graph(probabilities, reliability, *, k=20, chunk_size=256):
    """Return compact int64 neighbour[G,min(k,G-1)] and counts[G].

    Unused trailing neighbour cells equal -1. Retained edges are mutual in
    the directed top-k lists, self excluded. Distances are squared L2 on
    FP64(p)*sqrt(FP64(w)); exact floating ties use original row indices.
    Peak distance workspace is O(chunk_size*G), never an allocated G*G array.
    The graph is query-independent and has no ground-truth argument.
    """
    _positive_int(k, "k")
    _positive_int(chunk_size, "chunk_size")
    p, w = np.asarray(probabilities), np.asarray(reliability)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0 or w.shape != (p.shape[1],):
        raise ValueError("nonempty gallery-by-attribute probabilities and aligned reliability required")
    if not np.issubdtype(p.dtype, np.floating) or not np.issubdtype(w.dtype, np.floating):
        raise ValueError("floating probabilities/reliability required")
    if not np.isfinite(p).all() or not np.isfinite(w).all() or np.any((p < 0) | (p > 1)) or np.any(w <= 0):
        raise ValueError("finite probabilities in [0,1] and positive reliability required")
    n = len(p)
    width = min(k, n - 1)
    if width == 0:
        return np.empty((n, 0), dtype=np.int64), np.zeros(n, dtype=np.int64)
    x = p.astype(np.float64) * np.sqrt(w.astype(np.float64))[None, :]
    norms = np.einsum("ij,ij->i", x, x)
    if not np.isfinite(norms).all():
        raise ValueError("weighted probability magnitude overflows")
    directed = np.empty((n, width), dtype=np.int64)
    # Keep even tiny toy runs off a full G*G allocation.
    block = min(chunk_size, n - 1)
    for start in range(0, n, block):
        end = min(n, start + block)
        distances = _squared_l2_block(x, norms, start, end)
        directed[start:end] = _deterministic_smallest(distances, width)
    mutual = np.any(directed[directed] == np.arange(n)[:, None, None], axis=2)
    counts = mutual.sum(axis=1, dtype=np.int64)
    neighbours = np.full(directed.shape, -1, dtype=np.int64)
    for row in range(n):
        neighbours[row, : counts[row]] = directed[row, mutual[row]]
    return neighbours, counts


def _validate_graph(neighbours, counts, size):
    indices, number = np.asarray(neighbours), np.asarray(counts)
    if indices.ndim != 2 or indices.shape[0] != size or number.shape != (size,):
        raise ValueError("graph shape differs from gallery")
    if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(number.dtype, np.integer):
        raise ValueError("integer graph indices/counts required")
    if np.any(number < 0) or np.any(number > indices.shape[1]):
        raise ValueError("graph degree out of range")
    valid = np.arange(indices.shape[1])[None, :] < number[:, None]
    if np.any(indices[~valid] != -1) or np.any(indices[valid] < 0) or np.any(indices[valid] >= size):
        raise ValueError("compact graph padding or neighbour index invalid")
    if np.any((indices == np.arange(size)[:, None]) & valid):
        raise ValueError("self edges are not allowed")
    if indices.shape[1]:
        ordered = np.sort(indices, axis=1)
        if np.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)):
            raise ValueError("duplicate neighbour")
    # Encode edges and verify reverses without a G*G adjacency matrix.
    src = np.broadcast_to(np.arange(size, dtype=np.int64)[:, None], indices.shape)[valid]
    dst = indices[valid].astype(np.int64, copy=False)
    edges = np.sort(src * size + dst)
    if len(edges) and not np.array_equal(edges, np.sort(dst * size + src)):
        raise ValueError("graph is not reciprocal")
    return indices.astype(np.int64, copy=False), number.astype(np.int64, copy=False)


def smooth_distances(distances, neighbours, counts, *, alpha=0.1, query_chunk_size=64, gallery_chunk_size=256):
    """One simultaneous uniform-neighbour step, same floating dtype as input.

    Isolated gallery items and alpha=0 are exact unchanged copies. Neighbour
    distances always read original d, never previously updated outputs.
    Workspace besides output is O(query_chunk_size*gallery_chunk_size*k).
    """
    _positive_int(query_chunk_size, "query_chunk_size")
    _positive_int(gallery_chunk_size, "gallery_chunk_size")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("finite alpha in [0,1] required")
    d = np.asarray(distances)
    if d.ndim != 2 or not all(d.shape) or not np.issubdtype(d.dtype, np.floating) or not np.isfinite(d).all():
        raise ValueError("finite nonempty floating query-by-gallery distance matrix required")
    index, degree = _validate_graph(neighbours, counts, d.shape[1])
    result = d.copy()
    if alpha == 0 or not np.any(degree):
        return result
    cast_alpha = np.asarray(alpha, dtype=d.dtype)
    for gstart in range(0, d.shape[1], gallery_chunk_size):
        gend = min(d.shape[1], gstart + gallery_chunk_size)
        active = np.flatnonzero(degree[gstart:gend]) + gstart
        if not len(active):
            continue
        selected = index[active]
        valid = selected >= 0
        safe_index = np.maximum(selected, 0)
        denominator = degree[active].astype(d.dtype)
        for qstart in range(0, d.shape[0], query_chunk_size):
            qend = min(d.shape[0], qstart + query_chunk_size)
            values = d[qstart:qend, safe_index]
            values = np.where(valid[None, :, :], values, np.asarray(0, dtype=d.dtype))
            mean = values.sum(axis=2, dtype=d.dtype) / denominator[None, :]
            original = d[qstart:qend, active]
            result[qstart:qend, active] = original + cast_alpha * (mean - original)
    if not np.isfinite(result).all():
        raise ValueError("score smoothing overflowed")
    return result
