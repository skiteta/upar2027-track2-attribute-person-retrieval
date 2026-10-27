"""Pinned, original-only validation inference from an existing H1 checkpoint.

No training, checkpoint rewriting, flip inference, automatic OOM retry or implicit
resume. Outputs are prepared in a unique stage; no-overwrite atomic moves publish
the complete NPZ first and the completion report last (one hardlink per file). The report is the commit
marker, not a claim that the supervisor has observed this process exit.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path: Path, expected: str) -> dict:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"a lowercase SHA256 is required for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: expected {expected}, got {actual}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path: Path, value: dict) -> None:
    # Each stage has one writer. Existing receipts are atomically updated, while
    # the published completion report is never overwritten.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _sync_dir(path.parent)


def _move_no_replace(source: Path, target: Path) -> None:
    """Linux atomic no-overwrite rename, retaining nlink=1 for Work OS evidence."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("renameat2 unavailable; refusing non-atomic or overwriting publication")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:  # AT_FDCWD, RENAME_NOREPLACE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _publish(stage: Path, run: Path) -> None:
    """Atomic per-file, fail-if-exists publication; report last is the commit."""
    for name in ("val_logits.npz", "train_report.json"):
        if (run / name).exists():
            raise FileExistsError(f"refusing to replace existing output: {run / name}")
    _move_no_replace(stage / "val_logits.npz", run / "val_logits.npz")
    _sync_dir(stage)
    _sync_dir(run)
    _move_no_replace(stage / "train_report.json", run / "train_report.json")
    _sync_dir(stage)
    _sync_dir(run)


def predict_saved_checkpoint(
    run_dir: Path, *, config_sha256: str, state_sha256: str, meta_sha256: str,
    batch_size: int = 64, threads: int = 2, device: str = "cuda", progress_batches: int = 20,
) -> dict:
    """Evaluate every validation row in cache order, using H1's actual model/preprocessing."""
    from safetensors.torch import load_file

    from . import h1_finetune as h1

    run = Path(run_dir).resolve()
    if not run.is_dir() or run == ROOT or run == Path("/"):
        raise ValueError("an existing dedicated saved-model run directory is required")
    if batch_size < 1 or threads not in (1, 2, 3, 4) or progress_batches < 1:
        raise ValueError("positive batch/progress interval and 1..4 CPU threads required")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError("only explicit CPU or CUDA device is supported")
    started = time.monotonic()
    torch.set_num_threads(threads)
    # An advisory lock prevents overlapping writers. It is not a GPU lease.
    with (run / ".predict_original.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("val_logits.npz", "train_report.json"):
            if (run / name).exists():
                raise FileExistsError(f"existing output requires inspection, not inference replay: {run / name}")
        stage = Path(tempfile.mkdtemp(prefix=".predict_original_", dir=run))
        receipt = {
            "status": "VALIDATING", "pid": os.getpid(), "ppid": os.getppid(), "stage_dir": str(stage),
            "run_dir": str(run), "started_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(dev), "batch_size": batch_size, "threads": threads,
            "autocast": "bfloat16" if dev.type == "cuda" else "disabled_cpu_test_path",
            "flip_generated": False, "training_performed": False, "completed_rows": 0,
            "automatic_retry": False, "process_exit_confirmed": False,
        }
        _json(stage / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        try:
            cfg_path, state_path = run / "config.json", run / "model.safetensors"
            config_ref = _pin(cfg_path, config_sha256)
            state_ref = _pin(state_path, state_sha256)
            cfg = json.loads(cfg_path.read_text())
            cache = Path(cfg["cache_dir"])
            cache = cache if cache.is_absolute() else ROOT / cache
            meta_path = cache / "meta.npz"
            meta_ref = _pin(meta_path, meta_sha256)
            with np.load(meta_path, allow_pickle=False) as loaded:
                meta = {k: loaded[k] for k in loaded.files}
            pos_weight = np.asarray(cfg["pos_weight"], dtype=np.float32)
            if pos_weight.shape != (40,) or not np.isfinite(pos_weight).all() or np.any(pos_weight <= 0):
                raise ValueError("saved config must contain finite positive pos_weight[40]")
            if len(meta["attribute_names"]) != 40:
                raise ValueError("expected the exact 40-attribute vocabulary")
            if cfg["fold"] not in (*h1.SOURCES, "all"):
                raise ValueError("unknown fit fold")
            fit_rows = h1._fit_rows(meta, cfg["fold"], bool(cfg["include_val"]))
            val_rows = np.flatnonzero(meta["split"] == "val")
            if not len(val_rows) or not len(fit_rows):
                raise ValueError("empty validation or declared fit rows")
            if cfg["fold"] != "all" and (meta["sources"][fit_rows] == cfg["fold"]).any():
                raise ValueError("held-source image in declared fit rows")
            image_path = cache / cfg.get("cache_file", "images_u8_256x128.npy")
            images = np.load(image_path, mmap_mode="r", allow_pickle=False)
            if (images.shape[0] != len(meta["split"]) or images.dtype != np.uint8
                    or images.ndim != 4 or images.shape[-1] != 3):
                raise ValueError("image cache must be row-aligned uint8 NHWC")
            image_stat = image_path.stat()
            cache_stamp = (image_stat.st_ino, image_stat.st_size, image_stat.st_mtime_ns)
            source_hashes = {str(Path(__file__).resolve()): sha256(Path(__file__)),
                             str(Path(h1.__file__).resolve()): sha256(Path(h1.__file__))}
            receipt.update(inputs={"config": config_ref, "state": state_ref, "meta": meta_ref},
                           source_hashes=source_hashes, total_rows=int(len(val_rows)),
                           pos_weight_sha256=hashlib.sha256(pos_weight.tobytes()).hexdigest(),
                           image_cache={"path": str(image_path.resolve()), "bytes": image_stat.st_size,
                                        "mtime_ns": image_stat.st_mtime_ns, "content_sha256": None,
                                        "note": "existing cache; content not rehashed by this predictor"})
            if dev.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
                with torch.cuda.device(dev):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("BF16 unsupported; no precision substitution")
                    torch.cuda.reset_peak_memory_stats(dev)
            receipt.update(status="LOADING", seconds=time.monotonic() - started)
            _json(stage / "receipt.json", receipt)
            model, mean, std = h1._build_model(cfg["backbone"], pretrained=False, head=cfg.get("head", "linear"))
            state = load_file(str(state_path), device="cpu")
            model.load_state_dict(state, strict=True)
            del state
            model = model.requires_grad_(False).to(dev).eval()
            spec = h1.BACKBONES[cfg["backbone"]]
            logits = np.lib.format.open_memmap(stage / "logits_partial.npy", mode="w+", dtype=np.float32,
                                              shape=(len(val_rows), 40))
            inference_started = time.monotonic()
            receipt.update(status="PREDICTING")
            _json(stage / "receipt.json", receipt)
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                for batch_no, start in enumerate(range(0, len(val_rows), batch_size), 1):
                    idx = val_rows[start:start + batch_size]
                    x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(dev)
                    inputs = h1._gpu_augment(x, train=False, out_hw=spec["img_hw"], mean=mean, std=std,
                                             crop_top_frac=spec.get("crop_top_frac"))
                    prediction = model(inputs).float().cpu().numpy()
                    if prediction.shape != (len(idx), 40) or not np.isfinite(prediction).all():
                        raise ValueError("saved model returned nonfinite or incorrectly shaped logits")
                    logits[start:start + len(idx)] = prediction
                    receipt["completed_rows"] = start + len(idx)
                    if batch_no % progress_batches == 0 or start + len(idx) == len(val_rows):
                        logits.flush()
                        receipt.update(seconds=time.monotonic() - started,
                                       predict_seconds=time.monotonic() - inference_started)
                        if dev.type == "cuda":
                            receipt["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
                            receipt["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved(dev) / 2**30
                        _json(stage / "receipt.json", receipt)
                        print(json.dumps({k: receipt[k] for k in ("status", "pid", "completed_rows", "total_rows",
                                                                 "seconds", "predict_seconds")}), flush=True)
            logits.flush()
            receipt.update(status="VERIFYING_INPUTS", predict_seconds=time.monotonic() - inference_started)
            _json(stage / "receipt.json", receipt)
            # No checkpoint/config rewriting; reject changed inputs after the completed calculation.
            _pin(cfg_path, config_sha256)
            _pin(state_path, state_sha256)
            _pin(meta_path, meta_sha256)
            final_stat = image_path.stat()
            if cache_stamp != (final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns):
                raise ValueError("image cache identity/size/mtime changed during prediction")
            if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
                raise ValueError("predictor or H1 source changed during prediction")
            npz = stage / "val_logits.npz"
            with npz.open("xb") as stream:
                np.savez(stream, logits_original=np.asarray(logits), pos_weight=pos_weight,
                         image_paths=meta["image_paths"][val_rows], attribute_names=meta["attribute_names"],
                         cache_rows=val_rows, flip_generated=np.asarray(False))
                stream.flush()
                os.fsync(stream.fileno())
            report = dict(receipt, status="PREDICT_SAVED_DONE", fold=cfg["fold"], include_val=cfg["include_val"],
                          backbone=cfg["backbone"], head=cfg.get("head", "linear"),
                          fit_rows=int(len(fit_rows)),
                          fit_rows_by_source={s: int((meta["sources"][fit_rows] == s).sum())
                                              for s in np.unique(meta["sources"][fit_rows])},
                          fit_boundary_evidence="Declared config/rows only; upstream parent fit audit remains separate",
                          evaluation_role="seen_fit" if cfg["include_val"] else "all_validation_predictions",
                          pos_weight=pos_weight.tolist(), seconds=time.monotonic() - started,
                          weights_unchanged=True, logits_sha256=sha256(npz),
                          publication="renameat2 no-overwrite NPZ then report-last commit, nlink=1; stage receipt retained")
            _json(stage / "train_report.json", report)
            _publish(stage, run)
            receipt.update(status="PUBLISHED", seconds=time.monotonic() - started,
                           outputs={"val_logits.npz": report["logits_sha256"],
                                    "train_report.json": sha256(run / "train_report.json")})
            _json(stage / "receipt.json", receipt)
            print(json.dumps({"status": "PUBLISHED", "pid": os.getpid(), "run_dir": str(run),
                              "stage_dir": str(stage), "rows": len(val_rows), "seconds": receipt["seconds"]}), flush=True)
            return report
        except BaseException as error:
            receipt.update(status="FAILED", seconds=time.monotonic() - started,
                           error_type=type(error).__name__, error=str(error),
                           published_npz=(run / "val_logits.npz").exists(),
                           published_report=(run / "train_report.json").exists(),
                           recovery="Inspect actual receipt and outputs; preserve stage. No automatic retry or overwrite.")
            try:
                _json(stage / "receipt.json", receipt)
            except OSError:
                pass  # The original failure must survive even when the volume is full.
            print(json.dumps(receipt), flush=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--meta-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-batches", type=int, default=20)
    args = vars(parser.parse_args())
    predict_saved_checkpoint(**args)


if __name__ == "__main__":
    main()
