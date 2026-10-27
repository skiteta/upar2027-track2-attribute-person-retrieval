"""CPU-only ordered weight averaging with one source checkpoint open at a time.

The arithmetic matches h1_finetune.soup: convert each tensor to float32,
multiply by the normalized Python-float weight, then add in source order.
Do not fuse multiplication into add(alpha=...), which can change rounding.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import tempfile
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_schema(path: Path) -> dict:
    with Path(path).open("rb") as stream:
        count_bytes = stream.read(8)
        if len(count_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        count = struct.unpack("<Q", count_bytes)[0]
        if count > 64 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header: {path}")
        header = json.loads(stream.read(count))
    return {key: {"dtype": value["dtype"], "shape": value["shape"]}
            for key, value in header.items() if key != "__metadata__"}


def file_identity(path: Path) -> dict:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def ordered_average(paths: list[Path], weights: list[float] | None = None,
                    expected_sha256: list[str] | None = None,
                    progress=None) -> tuple[dict[str, torch.Tensor], dict]:
    """Return a CPU float32 state; never simultaneously load all source states.

    Hash each checkpoint before using it, reject changed identity after reading,
    and reject mismatched schemas/non-finite values before publishing anything.
    The caller owns staging and publication.
    """
    paths = [Path(path) for path in paths]
    if not paths or len(set(map(str, paths))) != len(paths):
        raise ValueError("nonempty distinct ordered checkpoint paths required")
    w = np.ones(len(paths), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(paths),) or not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("one finite nonnegative weight per checkpoint required")
    w = w / w.sum()
    if expected_sha256 is not None and len(expected_sha256) != len(paths):
        raise ValueError("one expected SHA256 per checkpoint required")
    schemas = [tensor_schema(path) for path in paths]
    if not schemas[0] or any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("checkpoint key/dtype/shape schemas differ")
    if any(item["dtype"] not in {"F16", "BF16", "F32", "F64"} for item in schemas[0].values()):
        raise ValueError("this implementation accepts floating-point state tensors only")
    keys = sorted(schemas[0])
    result = {}
    receipts = []
    for index, (path, weight) in enumerate(zip(paths, w)):
        before = file_identity(path)
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256[index]:
            raise ValueError(f"checkpoint SHA256 differs: {path}")
        if file_identity(path) != before:
            raise ValueError(f"checkpoint changed while hashing: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for key in keys:
                value = source.get_tensor(key)
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"non-finite parent tensor: {path}:{key}")
                term = value.float().mul(float(weight))
                if index == 0:
                    result[key] = 0 + term  # Preserve even Python sum's signed-zero behavior.
                else:
                    result[key].add_(term)
                del value, term
        if file_identity(path) != before:
            raise ValueError(f"checkpoint changed during averaging: {path}")
        receipts.append({"path": str(path), "sha256": digest, "weight": float(weight), **before})
        if progress:
            progress({"event": "SOUP_PARENT_ACCUMULATED", "index": index + 1,
                      "parents": len(paths), "path": str(path)})
    if any(not torch.isfinite(value).all().item() for value in result.values()):
        raise ValueError("non-finite averaged tensor")
    return result, {"parents": receipts, "keys": len(keys), "dtype": "float32",
                    "parent_order": [str(path) for path in paths],
                    "arithmetic": "float32(parent) * float64_normalized_python_weight, then float32 add in parent order"}


def build_atomic_soup(*, parents: list[dict], output: Path, config: dict,
                      protocol_ref: str, minimum_available_gib: float = 16,
                      progress=None) -> dict:
    """Publish weights/config/receipt together; preserve all partial failures.

    No predictions or train_report are produced. WEIGHTS_READY is deliberately
    not SOUP_DONE: inference and prediction verification must happen separately.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    memory = available_memory_bytes()
    if memory < minimum_available_gib * 2**30:
        raise RuntimeError(f"CPU memory headroom insufficient: {memory / 2**30:.2f} GiB available")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    started = time.time()
    cpu_before = time.process_time()
    state = None
    try:
        state, arithmetic = ordered_average(
            [Path(parent["checkpoint"]["path"]) for parent in parents],
            [parent["weight"] for parent in parents],
            [parent["checkpoint"]["sha256"] for parent in parents], progress=progress)
        checkpoint = stage / "model.safetensors"
        save_file(state, str(checkpoint))
        del state
        state = None
        with checkpoint.open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoint_sha = sha256_file(checkpoint)
        published_config = dict(config, output_dir=str(output))
        (stage / "config.json").write_text(json.dumps(published_config, indent=2) + "\n")
        receipt = {
            "schema": 1, "status": "WEIGHTS_READY", "protocol_ref": protocol_ref,
            "output": str(output), "model": {"path": str(output / "model.safetensors"),
            "sha256": checkpoint_sha, "size": checkpoint.stat().st_size},
            "config_sha256": sha256_file(stage / "config.json"),
            "predictions_generated": False, "flip_generated": False,
            "views": [], "new_training": False, "gpu_seconds": 0,
            "available_memory_gib_before": memory / 2**30,
            "elapsed_seconds": time.time() - started,
            "cpu_seconds": time.process_time() - cpu_before,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "arithmetic": arithmetic,
        }
        (stage / "weights_ready.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Files are complete before the directory name becomes externally visible.
        for name in ("config.json", "weights_ready.json"):
            with (stage / name).open("rb") as stream:
                os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(f"output appeared during construction: {output}")
        os.rename(stage, output)
        if progress:
            progress({"event": "WEIGHTS_READY", "output": str(output),
                      "sha256": checkpoint_sha, "elapsed_seconds": receipt["elapsed_seconds"]})
        return receipt
    except BaseException as exc:
        if stage.exists():
            (stage / "failure.json").write_text(json.dumps({
                "status": "PARTIAL_FAILURE_PRESERVED", "exception": type(exc).__name__,
                "message": str(exc), "elapsed_seconds": time.time() - started,
                "final_output_published": False}, indent=2) + "\n")
        raise
