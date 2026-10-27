from types import SimpleNamespace

import pytest
import torch

from upar.pe_feature_head import FeatureHead, restore_value


class MeanPool(torch.nn.Module):
    def forward(self, value):
        return value.mean(1)


def test_feature_head_trace_dynamic_batch_dtype_and_eval_no_rng(tmp_path):
    backbone = SimpleNamespace(norm=torch.nn.LayerNorm(7), attn_pool=MeanPool(),
                               fc_norm=torch.nn.Identity(), head_drop=torch.nn.Dropout(.1))
    native = SimpleNamespace(backbone=backbone, dropout=torch.nn.Dropout(.1))
    feature = FeatureHead(native).eval()
    inputs = [torch.randn(n, 5, 7) for n in (1, 3, 8)]
    rng = torch.get_rng_state()
    traced = torch.jit.trace(feature, inputs[1])
    path = tmp_path / "feature.ts"
    traced.save(str(path))
    loaded = torch.jit.load(str(path)).eval()
    for x in inputs:
        torch.testing.assert_close(loaded(x), feature(x), rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.to(torch.bfloat16)(inputs[0].bfloat16()).dtype == torch.bfloat16


def test_restore_exact_original_loader_rowwise_arithmetic():
    q = torch.tensor([[-128, 127], [0, 30]], dtype=torch.int8)
    s = torch.tensor([.01, .02], dtype=torch.float16)
    torch.testing.assert_close(restore_value({"q": q, "s": s}), q.float()*s.float().unsqueeze(1), rtol=0, atol=0)
    torch.testing.assert_close(restore_value(s), s.float(), rtol=0, atol=0)


@pytest.mark.parametrize("value", [{"q": torch.ones(2, 3), "s": torch.ones(3)},
                                   {"q": torch.ones(2), "s": torch.ones(2)},
                                   {"q": torch.ones(2, 3), "s": torch.ones(2), "extra": 1},
                                   torch.tensor([float("nan")])])
def test_malformed_or_nonfinite_storage_rejected(value):
    with pytest.raises(ValueError):
        restore_value(value)
