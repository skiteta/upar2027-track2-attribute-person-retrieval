import numpy as np
import pytest

from upar import fixed_gallery_rerank as module


def brute_graph(p, w, k):
    x = p.astype(np.float64) * np.sqrt(w.astype(np.float64))
    distance = ((x[:, None, :] - x[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(distance, np.inf)
    directed = np.argsort(distance, axis=1, kind="stable")[:, : min(k, len(p) - 1)]
    neigh = np.full(directed.shape, -1, dtype=np.int64)
    counts = np.zeros(len(p), dtype=np.int64)
    for i in range(len(p)):
        selected = [int(j) for j in directed[i] if i in directed[j]]
        neigh[i, : len(selected)] = selected
        counts[i] = len(selected)
    return neigh, counts


def test_brute_force_distances_and_graph_chunking():
    rng = np.random.default_rng(2026)
    p = rng.uniform(size=(23, 5)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=5).astype(np.float32)
    x = p.astype(np.float64) * np.sqrt(w.astype(np.float64))
    norms = (x * x).sum(1)
    actual = module._squared_l2_block(x, norms, 3, 8)
    expected = ((x[3:8, None, :] - x[None, :, :]) ** 2).sum(2)
    expected[np.arange(5), np.arange(3, 8)] = np.inf
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
    reference = brute_graph(p, w, 4)
    for chunk in (1, 6, 256):
        actual = module.build_mutual_graph(p, w, k=4, chunk_size=chunk)
        assert all(np.array_equal(a, b) for a, b in zip(actual, reference))


def test_distance_index_ties_self_exclusion_and_isolation():
    p = np.zeros((5, 3), np.float32)
    neighbours, counts = module.build_mutual_graph(p, np.ones(3, np.float32), k=2, chunk_size=2)
    assert neighbours.tolist() == [[1, 2], [0, 2], [0, 1], [-1, -1], [-1, -1]]
    assert counts.tolist() == [2, 2, 2, 0, 0]
    d = np.asarray([[1, 2, 7, 4, 9], [8, 4, 1, 3, 2]], np.float32)
    result = module.smooth_distances(d, neighbours, counts)
    assert np.array_equal(result[:, 3:], d[:, 3:])
    np.testing.assert_allclose(result[:, 0], d[:, 0] + 0.1 * ((d[:, 1] + d[:, 2]) / 2 - d[:, 0]), rtol=0, atol=1e-6)
    assert np.array_equal(module.smooth_distances(d, neighbours, counts, alpha=0), d)


def test_one_step_uniform_means_and_bounds_no_mutation():
    rng = np.random.default_rng(8)
    p = rng.uniform(size=(17, 4)).astype(np.float64)
    w = np.ones(4, np.float64)
    neighbours, counts = module.build_mutual_graph(p, w, k=5)
    d = rng.uniform(-10, 30, size=(11, 17)).astype(np.float32)
    before = [x.copy() for x in (p, w, neighbours, counts, d)]
    out = module.smooth_distances(d, neighbours, counts, alpha=0.1, query_chunk_size=3, gallery_chunk_size=4)
    for g in range(len(p)):
        if counts[g]:
            expected = d[:, g] + np.float32(0.1) * (d[:, neighbours[g, : counts[g]]].mean(1) - d[:, g])
            np.testing.assert_allclose(out[:, g], expected, atol=2e-6, rtol=1e-6)
            joined = np.concatenate([d[:, g : g + 1], d[:, neighbours[g, : counts[g]]]], axis=1)
            assert np.all(out[:, g] >= joined.min(1) - 1e-6)
            assert np.all(out[:, g] <= joined.max(1) + 1e-6)
        else:
            assert np.array_equal(out[:, g], d[:, g])
    assert out.dtype == d.dtype and not np.shares_memory(out, d)
    for x, old in zip((p, w, neighbours, counts, d), before):
        assert np.array_equal(x, old)


def test_permutation_equivariance_without_ties():
    rng = np.random.default_rng(42)
    p = rng.uniform(size=(19, 4)).astype(np.float64)
    w = np.asarray([0.5, 0.8, 1.2, 1.4])
    d = rng.uniform(size=(7, 19)).astype(np.float32)
    n, c = module.build_mutual_graph(p, w, k=4)
    original = module.smooth_distances(d, n, c)
    permutation = rng.permutation(len(p))
    pn, pc = module.build_mutual_graph(p[permutation], w, k=4)
    permuted = module.smooth_distances(d[:, permutation], pn, pc)
    np.testing.assert_allclose(permuted, original[:, permutation], atol=1e-7, rtol=1e-7)


def test_no_full_gallery_distance_allocation(monkeypatch):
    seen = []
    old = module._squared_l2_block

    def observed(x, norms, start, end):
        seen.append((end - start, len(x)))
        return old(x, norms, start, end)

    monkeypatch.setattr(module, "_squared_l2_block", observed)
    p = np.random.default_rng(3).uniform(size=(31, 4)).astype(np.float32)
    module.build_mutual_graph(p, np.ones(4, np.float32), k=20, chunk_size=7)
    assert seen and all(a <= 7 and a < b for a, b in seen)


def test_single_image_graph_identity():
    n, c = module.build_mutual_graph(np.asarray([[0.2, 0.4]], np.float32), np.ones(2, np.float32))
    assert n.shape == (1, 0) and c.tolist() == [0]
    d = np.asarray([[2.0], [3.0]], np.float32)
    assert np.array_equal(module.smooth_distances(d, n, c), d)


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "inf",
        "negative",
        "aboveone",
        "shape",
        "weight_shape",
        "negative_weight",
        "integer",
        "empty",
        "zero_k",
        "zero_chunk",
    ],
)
def test_bad_graph_inputs(fault):
    p = np.full((4, 3), 0.5, np.float32)
    w = np.ones(3, np.float32)
    args = {}
    if fault == "nan":
        p[0, 0] = np.nan
    if fault == "inf":
        w[0] = np.inf
    if fault == "negative":
        p[0, 0] = -0.1
    if fault == "aboveone":
        p[0, 0] = 1.1
    if fault == "shape":
        p = p[0]
    if fault == "weight_shape":
        w = w[:2]
    if fault == "negative_weight":
        w[0] = -1
    if fault == "integer":
        p = np.ones((4, 3), int)
    if fault == "empty":
        p = p[:0]
    if fault == "zero_k":
        args["k"] = 0
    if fault == "zero_chunk":
        args["chunk_size"] = 0
    with pytest.raises(ValueError):
        module.build_mutual_graph(p, w, **args)


@pytest.mark.parametrize(
    "fault",
    ["nan", "shape", "self", "nonreciprocal", "badpadding", "duplicates", "baddegree", "alpha", "alpha_nan", "integer"],
)
def test_bad_smooth_inputs(fault):
    n = np.asarray([[1, -1], [0, -1], [-1, -1]], np.int64)
    c = np.asarray([1, 1, 0], np.int64)
    d = np.ones((2, 3), np.float32)
    alpha = 0.1
    if fault == "nan":
        d[0, 0] = np.nan
    if fault == "shape":
        d = d[:, :2]
    if fault == "self":
        n[0, 0] = 0
    if fault == "nonreciprocal":
        n[0, 0] = 2
    if fault == "badpadding":
        n[2, 1] = 0
    if fault == "duplicates":
        n[0] = [1, 1]
        c[0] = 2
    if fault == "baddegree":
        c[0] = 3
    if fault == "alpha":
        alpha = 2
    if fault == "alpha_nan":
        alpha = float("nan")
    if fault == "integer":
        d = np.ones((2, 3), int)
    with pytest.raises(ValueError):
        module.smooth_distances(d, n, c, alpha=alpha)
