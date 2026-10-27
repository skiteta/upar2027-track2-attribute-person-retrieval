import torch

from upar.pe_gallery_features import PooledCapture


def test_pre_hook_captures_input_without_forward_or_rng_change():
    torch.manual_seed(413)
    model = torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.ReLU(), torch.nn.Linear(11, 4)).eval()
    x = torch.randn(13, 7)
    state = torch.random.get_rng_state().clone()
    original = model(x)
    capture = PooledCapture(model[-1])
    actual = model(x)
    assert torch.equal(actual, original)
    assert torch.equal(capture.value, model[:-1](x))
    assert capture.calls == 1
    assert not capture.value.requires_grad
    assert torch.equal(state, torch.random.get_rng_state())
    capture.close()
    model(x)
    assert capture.calls == 1


def test_capture_distinct_batches_do_not_alias_new_values():
    head = torch.nn.Linear(3, 2).eval()
    capture = PooledCapture(head)
    x, y = torch.zeros(2, 3), torch.ones(5, 3)
    head(x)
    saved = capture.value
    head(y)
    assert torch.equal(saved, x)
    assert torch.equal(capture.value, y)
    assert capture.calls == 2
    capture.close()
