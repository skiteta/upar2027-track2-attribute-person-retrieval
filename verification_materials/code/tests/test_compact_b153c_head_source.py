import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


spec = importlib.util.spec_from_file_location("compact_heads", Path(__file__).parents[1] / "scripts/compact_b153c_head_source.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_only_heads_retained_exact(tmp_path):
    source, target = tmp_path / "full.safetensors", tmp_path / "heads.safetensors"
    save_file({"head.weight": torch.tensor([[1., -0., 3.]]), "backbone.weight": torch.zeros(5, 5)}, str(source))
    before = source.read_bytes()
    result = mod.extract_heads(source, target)
    assert result["all_tensor_bytes_equal"]
    assert list(load_file(str(target))) == ["head.weight"]
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        mod.extract_heads(source, target)


def test_rejects_no_head(tmp_path):
    source = tmp_path / "full.safetensors"
    save_file({"backbone.weight": torch.zeros(1)}, str(source))
    with pytest.raises(ValueError, match="no head"):
        mod.extract_heads(source, tmp_path / "heads.safetensors")


def test_bitwise_not_just_numerical_equality():
    assert torch.equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not mod.byte_equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not mod.byte_equal(torch.ones(1), torch.ones(1, dtype=torch.float64))
    assert not mod.byte_equal({"q": torch.ones(1)}, {"s": torch.ones(1)})
