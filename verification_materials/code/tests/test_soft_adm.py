import numpy as np
import pytest
import torch

from upar.metrics_strict import evaluate_scores
from upar.soft_adm import soft_adm_loss, soft_adm_per_query


def fixture():
    q = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    ids = np.array([0, 1, 1, 2, 3, 0], dtype=np.int64)
    y = q[ids]
    d = np.random.default_rng(17).permutation(24).reshape(4, 6).astype(np.float64)
    return q, ids, y, d


def test_zero_temperature_limit_matches_exact_adm_with_multiple_positives():
    q, ids, y, d = fixture()
    expected = evaluate_scores(d, direction="distance", queries=q, gallery_attributes=y, gallery_query_ids=ids)
    actual = soft_adm_per_query(torch.tensor(d), torch.tensor(q), torch.tensor(y), temperature=1e-4)
    np.testing.assert_allclose(actual.numpy(), [v["ADM"] for v in expected.per_query], atol=1e-12)


def test_singleton_and_all_exact_are_perfect():
    for n in (1, 5):
        y = torch.ones(n, 3)
        d = torch.randn(n, n, requires_grad=True)
        a = soft_adm_per_query(d, y, y)
        torch.testing.assert_close(a, torch.ones(n))
        a.sum().backward()
        torch.testing.assert_close(d.grad, torch.zeros_like(d))


def test_gradient_matches_finite_difference():
    q, _, y, d = fixture()
    value = torch.tensor(d / 10, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: soft_adm_per_query(x, torch.tensor(q), torch.tensor(y)), (value,))


def test_bad_negative_is_pushed_behind_exact_positive():
    q = torch.tensor([[0., 0.]])
    y = torch.tensor([[0., 0.], [1., 1.]])
    d = torch.tensor([[1., 0.]], requires_grad=True)
    (1 - soft_adm_per_query(d, q, y)).sum().backward()
    assert d.grad[0, 0] > 0  # gradient descent lowers true-match distance
    assert d.grad[0, 1] < 0  # and raises the wrong match's distance


def test_chunking_and_permutation_preserve_objective_and_gradient():
    q, _, y, d = fixture()
    x = torch.tensor(d / 10, requires_grad=True)
    a = soft_adm_per_query(x, torch.tensor(q), torch.tensor(y), pair_chunk=1)
    b = soft_adm_per_query(x, torch.tensor(q), torch.tensor(y), pair_chunk=256)
    torch.testing.assert_close(a, b)
    g1, = torch.autograd.grad(a.sum(), x, retain_graph=True)
    g2, = torch.autograd.grad(b.sum(), x)
    torch.testing.assert_close(g1, g2)
    order = torch.tensor([5, 2, 3, 0, 1, 4])
    c = soft_adm_per_query(x[:, order], torch.tensor(q), torch.tensor(y)[order])
    torch.testing.assert_close(a, c)


def test_missing_matches_and_incomplete_labels_rejected():
    with pytest.raises(ValueError, match="exact match"):
        soft_adm_per_query(torch.ones(1, 2), torch.zeros(1, 2), torch.ones(2, 2))
    with pytest.raises(ValueError, match="binary"):
        soft_adm_per_query(torch.ones(1, 1), torch.full((1, 2), .5), torch.ones(1, 2))
    with pytest.raises(ValueError, match="temperature"):
        soft_adm_per_query(torch.ones(1, 1), torch.ones(1, 2), torch.ones(1, 2), temperature=0)


def test_probability_loss_has_finite_gradient_and_row_weighting():
    torch.manual_seed(3)
    z = torch.randn(12, 40, requires_grad=True)
    y = torch.randint(0, 2, (12, 40)).float()
    weights = torch.arange(1, 13).float()
    value = soft_adm_loss(z, y, torch.zeros(40), row_weight=weights)
    assert 0 <= value <= 1
    value.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_production_dispatch_and_invalid_masked_recipes():
    from upar.h1_finetune import _rank_loss, _rank_term
    torch.manual_seed(7)
    z, y = torch.randn(12, 40), torch.randint(0, 2, (12, 40)).float()
    pw = torch.zeros(40)
    sources = torch.tensor([0] * 6 + [1] * 6)
    cfg = {"rank_loss_kind": "soft_adm", "rank_adm_temperature": 1.0}
    actual = _rank_term(z, y, sources, None, pw, cfg)
    expected = (soft_adm_loss(z[:6], y[:6], pw) + soft_adm_loss(z[6:], y[6:], pw)) / 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(_rank_term(z, y, None, None, pw, {}), _rank_loss(z, y, pw))
    for key, value in (("partswap_p", .3), ("mixup_p", .2), ("label_mask", "v2")):
        with pytest.raises(ValueError, match="complete unmixed"):
            _rank_term(z, y, sources, None, pw, dict(cfg, **{key: value}))
