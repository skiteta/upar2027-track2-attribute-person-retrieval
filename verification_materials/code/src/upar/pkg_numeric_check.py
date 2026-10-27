"""Fast numeric check of a packaged run.py: decode a fixed subset of validation images through the package's own
loader and preprocessing, run every model (including resolution variants) and every plan mixture, and compare the
probabilities with the cached reference predictions of each model's training_run. Complements the full replay
(timing, end-to-end ranking) when the host is too busy for a clean timing run.
Run with the official-version interpreter on CPU, as h1_replay does:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m upar.pkg_numeric_check --artifact ZIP --output JSON [--stride 16]"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    from .data import load_retrieval_split
    from .execute_submission import load_submission_module
    from .h1_finetune import SPLITS, _probabilities
    from .postproc import apply_post

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--image-root", type=Path, default=Path("starter_kit/data"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    split = load_retrieval_split(SPLITS["val"])
    idx = np.arange(0, split.gallery_size, args.stride)
    paths = [str(split.image_paths[i]) for i in idx]
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.artifact) as zf:
            zf.extractall(tmp)
        # numeric check only: allow the CPU path in the extracted copy (GPU-runner packages fail closed without CUDA);
        # the artifact itself is not modified
        cfg_path = Path(tmp) / "assets/config.json"
        cfg_json = json.loads(cfg_path.read_text())
        cfg_json.update(require_cuda=False, fail_closed_on_fallback=False, min_active_members=1)
        cfg_path.write_text(json.dumps(cfg_json))
        mod = load_submission_module(Path(tmp) / "run.py")
        mod.load_model()
        cfg = mod._CONFIG
        items = [{"image_path": str((root / args.image_root / p).resolve())} for p in paths]
        import torch

        probs = []
        with torch.inference_mode():
            for i, spec in enumerate(cfg["models"]):
                out = []
                for s in range(0, len(items), 32):
                    out.append(mod._run(mod._MODELS[i], spec, mod._decode_batch(items[s:s + 32])))
                probs.append(np.concatenate(out))
    refs = []
    for spec in cfg["models"]:
        z = np.load(root / spec["training_run"] / "val_logits.npz")
        row = {str(p): k for k, p in enumerate(z["image_paths"])}
        refs.append(_probabilities(z, bool(spec.get("flip_tta")))[[row[p] for p in paths]])
    report = {"artifact": str(args.artifact), "images": len(paths), "models": {}, "plans": {}}
    for spec, p, r in zip(cfg["models"], probs, refs):
        d = np.abs(p - r)
        report["models"][spec["name"]] = {"mean_abs": float(d.mean()), "p99_abs": float(np.quantile(d, 0.99)), "max_abs": float(d.max())}
    for plan in cfg.get("plans") or [{"name": "all", "members": list(range(len(cfg["models"])))}]:
        w = np.asarray([cfg["models"][i].get("member_weights", [1.0] * 40) for i in plan["members"]], dtype=np.float32)
        mix = lambda arr: np.einsum("mga,ma->ga", np.stack([arr[i] for i in plan["members"]]), w) / w.sum(axis=0)
        d = np.abs(mod._apply_post(mix(probs), cfg.get("post")) - apply_post(mix(refs), cfg.get("post"))) if hasattr(mod, "_apply_post") \
            else np.abs(mix(probs) - mix(refs))
        report["plans"][plan["name"]] = {"mean_abs": float(d.mean()), "p99_abs": float(np.quantile(d, 0.99)), "max_abs": float(d.max())}
    report["pass"] = all(v["mean_abs"] < 0.01 and v["p99_abs"] < 0.05 for v in report["plans"].values())
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
