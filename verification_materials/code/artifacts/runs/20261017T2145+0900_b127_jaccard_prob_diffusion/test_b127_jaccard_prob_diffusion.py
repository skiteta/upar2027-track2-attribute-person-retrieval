"""Unit tests for B127C Jaccard-structural continuous power-margin probability diffusion."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from b127_jaccard_prob_diffusion import (
    DIST_DIFFUSION_ALPHA,
    JACCARD_POWER,
    MARGIN_GAMMA,
    MAX_HAMMING,
    PROB_CLIP_EPS,
    PROB_DIFFUSION_ALPHA,
    PROB_DIFFUSION_STEPS,
    diffuse_probabilities,
    jaccard_cosine_margin_weights,
    smooth_distances_weighted,
)


def _make_mutual_graph(u: np.ndarray, k: int = 4):
    sim = u @ u.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argsort(-sim, axis=1)[:, :k]
    g = len(u)
    adj = np.zeros((g, g), dtype=bool)
    for i in range(g):
        adj[i, topk[i]] = True
    mutual = adj & adj.T
    counts = mutual.sum(axis=1).astype(np.int64)
    neighbours = np.full((g, k), -1, dtype=np.int64)
    for i in range(g):
        nbs = np.flatnonzero(mutual[i])
        neighbours[i, : len(nbs)] = nbs
    return neighbours, counts


def test_jaccard_cosine_margin_weights_symmetry_and_chunk_invariance():
    rng = np.random.default_rng(20261017)
    g = 48
    feats = rng.normal(size=(g, 32))
    u = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    p = rng.uniform(0.05, 0.95, size=(g, 40)).astype(np.float32)
    n, c = _make_mutual_graph(u, k=6)

    w_full = jaccard_cosine_margin_weights(p, u, n, c, gallery_chunk_size=1024)
    w_chunked = jaccard_cosine_margin_weights(p, u, n, c, gallery_chunk_size=7)
    np.testing.assert_allclose(w_full, w_chunked, rtol=0.0, atol=1e-15)
    assert np.all(w_full >= 0.0) and np.all(w_full <= 1.0)

    # Verify exact undirected edge symmetry w_ij == w_ji
    for i in range(g):
        for slot in range(c[i]):
            j = int(n[i, slot])
            rev_slots = np.flatnonzero(n[j, : c[j]] == i)
            assert len(rev_slots) == 1
            np.testing.assert_allclose(w_full[i, slot], w_full[j, int(rev_slots[0])], rtol=0.0, atol=1e-14)


def test_diffuse_probabilities_convexity_and_bounds():
    rng = np.random.default_rng(42)
    g = 32
    feats = rng.normal(size=(g, 16))
    u = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    p = rng.uniform(0.01, 0.99, size=(g, 40)).astype(np.float32)
    n, c = _make_mutual_graph(u, k=5)
    w = jaccard_cosine_margin_weights(
        p, u, n, c, max_hamming=MAX_HAMMING, margin_gamma=MARGIN_GAMMA, jaccard_power=JACCARD_POWER
    )
    out = diffuse_probabilities(
        p, n, c, w, alpha_p=PROB_DIFFUSION_ALPHA, steps=PROB_DIFFUSION_STEPS, clip_eps=PROB_CLIP_EPS
    )
    assert out.shape == (g, 40) and out.dtype == np.float32
    assert np.all(out >= np.float32(PROB_CLIP_EPS)) and np.all(out <= np.float32(1.0 - PROB_CLIP_EPS))


def test_smooth_distances_weighted_chunk_determinism():
    rng = np.random.default_rng(99)
    q, g = 150, 60
    feats = rng.normal(size=(g, 16))
    u = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    p = rng.uniform(0.1, 0.9, size=(g, 40)).astype(np.float32)
    n, c = _make_mutual_graph(u, k=5)
    w = jaccard_cosine_margin_weights(p, u, n, c)
    d = rng.uniform(0.0, 2.0, size=(q, g)).astype(np.float32)

    s1 = smooth_distances_weighted(d, n, c, w, alpha_d=DIST_DIFFUSION_ALPHA, gallery_chunk_size=256)
    s2 = smooth_distances_weighted(d, n, c, w, alpha_d=DIST_DIFFUSION_ALPHA, gallery_chunk_size=13)
    np.testing.assert_array_equal(s1, s2)

