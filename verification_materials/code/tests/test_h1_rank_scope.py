"""Tiny CPU gradient checks of the actual H1 accumulation block (no model/data IO).

The block is extracted from train's AST because the complete trainer is CUDA/data
bound. Only its autocast context is replaced; production loss/backward statements
are executed unchanged. These are mathematical checks, not candidate scores.
"""
import ast
from contextlib import nullcontext
import copy
import inspect

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from upar import h1_finetune as h1


class _CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(self, *args, **kwargs):
        return nullcontext()


def _production_block():
    """Find the precise zero_grad -> loss.backward block, fail on a refactor."""
    tree = ast.parse(inspect.getsource(h1.train))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        body = node.body
        starts = [i for i, stmt in enumerate(body)
                  if isinstance(stmt, ast.Expr)
                  and isinstance(stmt.value, ast.Call)
                  and ast.unparse(stmt.value.func) == "opt.zero_grad"]
        ends = [i for i, stmt in enumerate(body)
                if isinstance(stmt, ast.If)
                and ast.unparse(stmt.test) == "backbone_on"]
        if starts and ends and starts[0] < ends[0]:
            return compile(ast.Module(body=body[starts[0]:ends[0]], type_ignores=[]),
                           inspect.getsourcefile(h1.train), "exec")
    raise AssertionError("H1 accumulation block moved; update its test extraction")


class _TinyNet(torch.nn.Module):
    def __init__(self, dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.GELU(),
                                           torch.nn.Dropout(dropout))
        self.head = torch.nn.Linear(11, 40)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            z = checkpoint(self.encoder, x, use_reentrant=False, preserve_rng_state=True)
        else:
            z = self.encoder(x)
        return self.head(z)


def _fixture(*, alpha=0.2, scope="batch", accum=4, mask=False, focal=0.0, kd=False,
             dropout=0.0, use_checkpoint=False):
    torch.manual_seed(314159)
    model = _TinyNet(dropout, use_checkpoint)
    inputs = torch.randn(40, 7)
    y = torch.randint(0, 2, (40, 40)).float()
    pos_weight = torch.linspace(0.6, 2.0, 40)
    return dict(
        torch=_CpuTorch(), model=model,
        opt=torch.optim.SGD(model.parameters(), lr=0.01), inputs=inputs,
        target=0.98 * y + 0.01, y=y, device=torch.device("cpu"),
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        kd_criterion=torch.nn.BCEWithLogitsLoss(), kd_alpha=0.3,
        t=torch.rand_like(y) if kd else None,
        m=(torch.rand_like(y) > 0.2).float() if mask else None,
        focal_gamma=focal, log_pw=pos_weight.log(), rank_alpha=alpha,
        # train() initializes this before the extracted zero_grad block. These
        # tests exercise the original BCE/rank recipe; the opt-in prior-offset
        # branch has its own tests in test_source_prior_offset_bce.py.
        source_prior_offset=None,
        rank_scope=scope, cfg={"grad_accum": accum, "rank_tau_target": 1.0,
                               "rank_tau_pred": 1.0},
        # Last source is intentionally too small for a valid list.
        s_codes=torch.tensor([0] * 20 + [1] * 17 + [2] * 3),
        q_w=torch.linspace(0.1, 1.5, 40),
        _rank_term=h1._rank_term, _focal_factor=h1._focal_factor,
    )


def _grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def _reference(env, *, partitions):
    """Independent loss aggregation; use identical forwards for stochastic nets."""
    model = copy.deepcopy(env["model"])
    parts = env["inputs"].chunk(partitions)
    logits = torch.cat([model(x) for x in parts])
    target, m = env["target"], env["m"]
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=env["criterion"].pos_weight, reduction="none")
    if env["focal_gamma"]:
        p = torch.sigmoid(logits - env["log_pw"])
        pt = p * target + (1 - p) * (1 - target)
        per = per * (1 - pt).clamp(min=0).pow(env["focal_gamma"])
    bce = per.mean() if m is None else (per * m).sum() / m.sum().clamp(min=1)
    if env["t"] is not None:
        bce = (1 - env["kd_alpha"]) * bce + env["kd_alpha"] * \
              torch.nn.functional.binary_cross_entropy_with_logits(logits, env["t"])
    rank = 0.0
    for source in torch.unique(env["s_codes"]):
        take = env["s_codes"] == source
        if take.sum() >= 4:
            rank = rank + take.float().mean() * h1._rank_loss(
                logits[take], env["y"][take], env["log_pw"],
                row_weight=env["q_w"][take])
    loss = bce + env["rank_alpha"] * rank
    loss.backward()
    return loss.detach(), _grads(model)


@pytest.mark.parametrize("mask,focal,kd", [(False, 0, False), (True, 0, False),
                                           (False, 1.0, False), (True, 1.0, True)])
def test_batch_scope_matches_full_batch_gradient(mask, focal, kd):
    env = _fixture(mask=mask, focal=focal, kd=kd)
    expected_loss, expected_grad = _reference(env, partitions=1)
    exec(_production_block(), env)
    torch.testing.assert_close(env["loss"], expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(_grads(env["model"]), expected_grad, atol=1e-7, rtol=2e-5)


def test_checkpoint_dropout_preserves_actual_micro_forward_rng_and_gradient():
    env = _fixture(dropout=0.3, use_checkpoint=True)
    rng = torch.get_rng_state()
    expected_loss, expected_grad = _reference(env, partitions=4)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    exec(_production_block(), env)
    torch.testing.assert_close(env["loss"], expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(_grads(env["model"]), expected_grad, atol=1e-7, rtol=2e-5)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("scope", ["micro", "batch"])
def test_alpha_zero_is_full_batch_bce(scope):
    env = _fixture(alpha=0, scope=scope, mask=True, focal=1, kd=True)
    expected_loss, expected_grad = _reference(env, partitions=1)
    exec(_production_block(), env)
    torch.testing.assert_close(env["loss"], expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(_grads(env["model"]), expected_grad, atol=1e-7, rtol=2e-5)


def test_micro_scope_does_not_silently_claim_full_batch_rank():
    env = _fixture(scope="micro", alpha=1.0)
    expected_loss, expected_grad = _reference(env, partitions=1)
    exec(_production_block(), env)
    assert abs(float(env["loss"] - expected_loss)) > 0.1
    assert float((_grads(env["model"]) - expected_grad).norm()) > 1e-3
