import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file, load_file

from upar.streaming_weight_soup import ordered_average, sha256_file, build_atomic_soup


def parents(tmp_path, count=6):
    torch.manual_seed(73)
    paths = []
    states = []
    for index in range(count):
        state = {"a": torch.randn(7, 11), "b": torch.randn(5).half(),
                 "c": torch.tensor([(-1.)**index * 1e8, index / 7])}
        path = tmp_path / f"parent{index}.safetensors"
        save_file(state, str(path))
        states.append(state)
        paths.append(path)
    return paths, states


@pytest.mark.parametrize("weights", [None, [1., 2., 3., 4., 5., 6.]])
def test_bitwise_matches_existing_h1_soup_expression(tmp_path, weights):
    paths, states = parents(tmp_path)
    w = np.ones(6) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    expected = {key: sum(float(wi) * state[key].float() for wi, state in zip(w, states)) for key in states[0]}
    actual, receipt = ordered_average(paths, weights)
    for key in expected:
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))
    assert receipt["parent_order"] == list(map(str, paths))


def test_bad_hash_schema_and_nonfinite_fail(tmp_path):
    paths, states = parents(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        ordered_average(paths, expected_sha256=["0" * 64] * 6)
    states[-1]["a"] = torch.zeros(2)
    save_file(states[-1], str(paths[-1]))
    with pytest.raises(ValueError, match="schemas"):
        ordered_average(paths)
    paths, states = parents(tmp_path)
    states[-1]["a"][0, 0] = float("nan")
    save_file(states[-1], str(paths[-1]))
    with pytest.raises(ValueError, match="non-finite"):
        ordered_average(paths)


def test_atomic_weights_only_publish_and_no_overwrite(tmp_path):
    paths, _ = parents(tmp_path)
    manifest = [{"checkpoint": {"path": str(path), "sha256": sha256_file(path)}, "weight": 1/6} for path in paths]
    output = tmp_path / "output"
    receipt = build_atomic_soup(parents=manifest, output=output, config={"fold": "PA100k"},
                                protocol_ref="fixed.json", minimum_available_gib=0)
    assert receipt["status"] == "WEIGHTS_READY"
    assert not (output / "train_report.json").exists()
    assert not (output / "val_logits.npz").exists()
    assert sha256_file(output / "model.safetensors") == receipt["model"]["sha256"]
    assert not receipt["predictions_generated"]
    with pytest.raises(FileExistsError):
        build_atomic_soup(parents=manifest, output=output, config={}, protocol_ref="fixed.json", minimum_available_gib=0)


def test_partial_failure_not_published_and_is_preserved(tmp_path):
    paths, _ = parents(tmp_path)
    manifest = [{"checkpoint": {"path": str(path), "sha256": "bad"}, "weight": 1/6} for path in paths]
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="SHA256"):
        build_atomic_soup(parents=manifest, output=output, config={}, protocol_ref="fixed.json", minimum_available_gib=0)
    assert not output.exists()
    partial = list(tmp_path.glob(".failed.partial-*"))
    assert len(partial) == 1
    assert json.loads((partial[0] / "failure.json").read_text())["status"] == "PARTIAL_FAILURE_PRESERVED"


def test_invalid_weights_and_repeated_parent_rejected(tmp_path):
    paths, _ = parents(tmp_path)
    for weights in ([0.] * 6, [float("nan")] * 6, [-1.] * 6, [1.]):
        with pytest.raises(ValueError):
            ordered_average(paths, weights)
    with pytest.raises(ValueError, match="distinct"):
        ordered_average([paths[0], paths[0]])
