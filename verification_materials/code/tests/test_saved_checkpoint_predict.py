import fcntl
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from upar import h1_finetune as h1
from upar.saved_checkpoint_predict import _move_no_replace, _publish, predict_saved_checkpoint, sha256


class ToyNet(torch.nn.Module):
    def __init__(self, fail_at=None, callback=None):
        super().__init__()
        self.linear = torch.nn.Linear(3, 40)
        self.calls = 0
        self.fail_at = fail_at
        self.callback = callback

    def forward(self, x):
        self.calls += 1
        assert not self.training
        assert not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in self.parameters())
        if self.callback:
            self.callback()
        if self.calls == self.fail_at:
            raise torch.OutOfMemoryError("deliberate CPU fixture OOM; must not retry")
        return self.linear(x.mean(dim=(2, 3)))


def fixture(tmp_path, monkeypatch, fail_at=None, callback=None):
    torch.set_num_threads(1)
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    run.mkdir()
    cache.mkdir()
    images = np.arange(7 * 4 * 2 * 3, dtype=np.uint8).reshape(7, 4, 2, 3)
    np.save(cache / "images_u8_256x128.npy", images)
    np.savez(cache / "meta.npz", image_paths=np.array([f"image{i}.png" for i in range(7)]),
             attribute_names=np.array([f"attribute{i}" for i in range(40)]), labels=np.zeros((7, 40), np.uint8),
             split=np.array(["train", "val", "train", "val", "val", "train", "val"]),
             sources=np.array(["Market1501", "PA100k", "PA100k", "PETA", "Market1501", "PETA", "PA100k"]))
    cfg = dict(backbone="toy", head="attrq", fold="PA100k", include_val=False, cache_dir=str(cache),
               pos_weight=np.linspace(1, 10, 40, dtype=np.float32).tolist())
    (run / "config.json").write_text(json.dumps(cfg))
    original = ToyNet()
    save_file(original.state_dict(), str(run / "model.safetensors"))
    models = []

    def build(key, pretrained, head):
        assert key == "toy" and pretrained is False and head == "attrq"
        model = ToyNet(fail_at=fail_at, callback=callback)
        models.append(model)
        return model, (0.5,) * 3, (0.5,) * 3

    monkeypatch.setattr(h1, "_build_model", build)
    monkeypatch.setitem(h1.BACKBONES, "toy", {"img_hw": (4, 2)})
    pins = dict(config_sha256=sha256(run / "config.json"), state_sha256=sha256(run / "model.safetensors"),
                meta_sha256=sha256(cache / "meta.npz"), batch_size=2, device="cpu", threads=1, progress_batches=1)
    return run, cache, images, models, pins


def test_original_only_order_parity_sha_and_no_weight_mutation(tmp_path, monkeypatch):
    run, cache, images, models, pins = fixture(tmp_path, monkeypatch)
    report = predict_saved_checkpoint(run, **pins)
    assert report["status"] == "PREDICT_SAVED_DONE"
    assert report["flip_generated"] is False and report["training_performed"] is False
    assert report["fit_rows"] == 2
    assert report["fit_rows_by_source"] == {"Market1501": 1, "PETA": 1}
    assert report["completed_rows"] == report["total_rows"] == 4
    assert models[0].calls == 2
    with np.load(run / "val_logits.npz") as data:
        assert "logits_flip" not in data.files and not bool(data["flip_generated"])
        np.testing.assert_array_equal(data["cache_rows"], [1, 3, 4, 6])
        np.testing.assert_array_equal(data["image_paths"], ["image1.png", "image3.png", "image4.png", "image6.png"])
        with torch.inference_mode():
            x = h1._gpu_augment(torch.from_numpy(images[[1, 3, 4, 6]]), train=False, out_hw=(4, 2),
                                mean=(0.5,) * 3, std=(0.5,) * 3)
            expected = models[0](x).numpy()
        np.testing.assert_allclose(data["logits_original"], expected, atol=1e-6)
        assert h1._probabilities(data, False).shape == (4, 40)
        with pytest.raises(KeyError):
            h1._probabilities(data, True)
    assert sha256(run / "model.safetensors") == pins["state_sha256"]
    assert sha256(run / "config.json") == pins["config_sha256"]
    assert sha256(cache / "meta.npz") == pins["meta_sha256"]
    assert report["logits_sha256"] == sha256(run / "val_logits.npz")
    stage = Path(report["stage_dir"])
    assert not (stage / "val_logits.npz").exists() and not (stage / "train_report.json").exists()
    assert (run / "val_logits.npz").stat().st_nlink == 1
    assert (run / "train_report.json").stat().st_nlink == 1
    receipt = json.loads((stage / "receipt.json").read_text())
    assert receipt["status"] == "PUBLISHED" and receipt["pid"] > 0
    assert receipt["process_exit_confirmed"] is False


def test_hash_mismatch_never_constructs_model_and_preserves_failure(tmp_path, monkeypatch):
    run, _, _, models, pins = fixture(tmp_path, monkeypatch)
    pins["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        predict_saved_checkpoint(run, **pins)
    assert not models and not (run / "val_logits.npz").exists()
    receipt = json.loads(next(run.glob(".predict_original_*/receipt.json")).read_text())
    assert receipt["status"] == "FAILED" and receipt["completed_rows"] == 0


def test_oom_retains_partial_predictions_and_never_retries_or_publishes(tmp_path, monkeypatch):
    run, _, _, models, pins = fixture(tmp_path, monkeypatch, fail_at=2)
    with pytest.raises(torch.OutOfMemoryError):
        predict_saved_checkpoint(run, **pins)
    assert len(models) == 1 and models[0].calls == 2
    assert not (run / "val_logits.npz").exists() and not (run / "train_report.json").exists()
    stage = next(run.glob(".predict_original_*/receipt.json")).parent
    receipt = json.loads((stage / "receipt.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["completed_rows"] == 2
    assert receipt["error_type"] == "OutOfMemoryError" and receipt["automatic_retry"] is False
    assert (stage / "logits_partial.npy").exists()


def test_existing_output_refuses_replay_before_model_load(tmp_path, monkeypatch):
    run, _, _, models, pins = fixture(tmp_path, monkeypatch)
    (run / "val_logits.npz").write_bytes(b"preserved prior result")
    with pytest.raises(FileExistsError):
        predict_saved_checkpoint(run, **pins)
    assert not models and (run / "val_logits.npz").read_bytes() == b"preserved prior result"


def test_changed_config_during_inference_fails_after_computation_without_publish(tmp_path, monkeypatch):
    def mutate():
        path = tmp_path / "run/config.json"
        path.write_text(path.read_text() + " ")

    run, _, _, models, pins = fixture(tmp_path, monkeypatch, callback=mutate)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        predict_saved_checkpoint(run, **pins)
    assert models[0].calls == 2
    assert not (run / "train_report.json").exists()
    receipt = json.loads(next(run.glob(".predict_original_*/receipt.json")).read_text())
    assert receipt["status"] == "FAILED" and receipt["completed_rows"] == 4


def test_publish_fails_closed_without_overwrite(tmp_path):
    stage, run = tmp_path / "stage", tmp_path / "run"
    stage.mkdir()
    run.mkdir()
    (stage / "val_logits.npz").write_bytes(b"candidate")
    (stage / "train_report.json").write_text("{}")
    (run / "train_report.json").write_text("old")
    with pytest.raises(FileExistsError):
        _publish(stage, run)
    assert (run / "train_report.json").read_text() == "old"
    assert not (run / "val_logits.npz").exists()


def test_atomic_move_never_replaces_existing_target(tmp_path):
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"candidate")
    target.write_bytes(b"protected")
    with pytest.raises(FileExistsError):
        _move_no_replace(source, target)
    assert source.read_bytes() == b"candidate" and target.read_bytes() == b"protected"


def test_live_lock_prevents_second_writer_without_new_stage(tmp_path, monkeypatch):
    run, _, _, models, pins = fixture(tmp_path, monkeypatch)
    with (run / ".predict_original.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            predict_saved_checkpoint(run, **pins)
    assert not models and not list(run.glob(".predict_original_*/receipt.json"))
