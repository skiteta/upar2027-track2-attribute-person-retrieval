"""Portable PE pooled head from the exact archived B119 quantized state.

Input is the already-computed final transformer hidden tensor, not images.
No backbone block is executed or added to the submitted representation path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch
import zipfile

import torch

from . import h1_finetune as h1
from .h1_package import _set_slot
from .rank_sibling_pipeline import exclusive_json, fingerprint

ZIP_SHA = "ace7e1778e70abe55cc3fbfa09304769ca6a8a615ea827a250b2c6a9bf5908ca"


class FeatureHead(torch.nn.Module):
    """The native model's actual head.base input, including learned MAP pool."""

    def __init__(self, native):
        super().__init__()
        self.norm = native.backbone.norm
        self.pool = native.backbone.attn_pool
        self.fc_norm = native.backbone.fc_norm
        self.head_drop = native.backbone.head_drop
        self.dropout = native.dropout

    def forward(self, hidden):
        return self.dropout(self.head_drop(self.fc_norm(self.pool(self.norm(hidden)))))


def restore_value(value):
    if isinstance(value, dict):
        if set(value) != {"q", "s"} or value["q"].ndim != 2 or value["s"].shape != (value["q"].shape[0],):
            raise ValueError("unexpected rowwise INT8 storage")
        result = value["q"].float() * value["s"].float().unsqueeze(1)
    else:
        result = value.float()
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite restored tensor")
    return result


def native_reference(spec, restored):
    """Same H1 class/PE pool; depth0 omits only unused pre-head blocks here."""
    import timm

    original_create = timm.create_model

    def pool_only(*args, **kwargs):
        if kwargs.get("pretrained"):
            raise ValueError("pretrained downloads forbidden")
        return original_create(*args, **dict(kwargs, depth=0))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261017)
        with patch.object(timm, "create_model", side_effect=pool_only):
            native, _, _ = h1._build_model(spec["backbone"], pretrained=False, head="attrq")
    native.eval().requires_grad_(False)
    if len(native.backbone.blocks) != 0 or native.backbone.num_features != 1024:
        raise ValueError("unexpected native PE reference")
    state = native.state_dict()
    with torch.no_grad():
        for name, value in restored.items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"native head state differs: {name}")
            state[name].copy_(value)
    # Same final-hidden boundary used by h1_package._export_blocks. No image
    # preprocessing, transformer weight, source training, or dataset is used.
    native.backbone.forward_features = lambda hidden: native.backbone.norm(hidden)
    return native


def build(package, output):
    started, cpu = time.monotonic(), time.process_time()
    package, output = Path(package), Path(output)
    package_pin = fingerprint(package)
    if package_pin["sha256"] != ZIP_SHA:
        raise ValueError("not exact immutable B119 ZIP")
    output.mkdir(parents=False, exist_ok=False)
    pins = [fingerprint(Path(__file__)), fingerprint(Path(h1.__file__))]
    exclusive_json(output / "started.json", {"source_zip": package_pin, "sources": pins, "gpu": False, "training": False})
    try:
        with zipfile.ZipFile(package) as archive:
            config_raw = archive.read("assets/config.json")
            cfg = json.loads(config_raw)
            spec = cfg["models"][0]
            if (spec["backbone"] != "pe_core_l14_280" or spec["layout"] != "blocks"
                    or spec["weight_storage"] != "int8" or spec["model_hw"] != [280, 140]
                    or spec["head_file"] != "assets/model_0.head.ts"):
                raise ValueError("B119 PE architecture/boundary differs")
            head_raw = archive.read(spec["head_file"])
            weight_raw = archive.read(spec["weights_file"])
        stored = torch.load(io.BytesIO(weight_raw), map_location="cpu", weights_only=True)
        restored = {name: restore_value(stored[name]) for name in spec["head_keys"]}
        stored_sha = hashlib.sha256(weight_raw).hexdigest()
        del stored, weight_raw
        archived = torch.jit.load(io.BytesIO(head_raw), map_location="cpu").eval()
        for name, value in restored.items():
            _set_slot(archived, name, value)
        archived.float()
        native = native_reference(spec, restored)
        feature = FeatureHead(native).eval().requires_grad_(False)
        captured = []
        hook = native.head.base.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        generator = torch.Generator().manual_seed(20261017)
        canaries = [torch.randn(n, 201, 1024, generator=generator) * scale
                    for n, scale in ((1, .1), (3, 1.), (8, 10.))]
        checks = []
        with torch.inference_mode():
            traced = torch.jit.trace(feature, torch.zeros(2, 201, 1024), check_trace=False)
            temporary = output / "feature_head.stage.ts"
            traced.save(str(temporary))
            reloaded = torch.jit.load(str(temporary), map_location="cpu").eval()
            for hidden in canaries:
                captured.clear()
                original_logits = archived(hidden)
                native_logits = native(hidden)
                expected = captured[0]
                actual = reloaded(hidden)
                logits_diff = float((original_logits - native_logits).abs().max())
                feature_diff = float((expected - actual).abs().max())
                if len(captured) != 1 or actual.shape != (len(hidden), 1024) or not torch.isfinite(actual).all():
                    raise ValueError("incomplete/nonfinite pooled feature")
                if logits_diff > 1e-4 or feature_diff > 1e-6:
                    raise ValueError(f"archived/native/feature CPU parity failed: {logits_diff}/{feature_diff}")
                checks.append({"batch": len(hidden), "hidden_shape": list(hidden.shape),
                               "native_full_head_vs_original_archived_logit_max_abs": logits_diff,
                               "reloaded_feature_vs_native_head_base_prehook_max_abs": feature_diff})
        hook.remove()
        for pin in pins + [package_pin]:
            if fingerprint(h1.ROOT / pin["path"]) != pin:
                raise ValueError("source changed during export")
        target = output / "feature_head.ts"
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        report = {"status": "CPU_FEATURE_HEAD_READY_NOT_GPU_VERIFIED", "source_zip": package_pin,
                  "source_members": {"config_sha256": hashlib.sha256(config_raw).hexdigest(),
                                     "head_graph_sha256": hashlib.sha256(head_raw).hexdigest(),
                                     "weights_sha256": stored_sha},
                  "feature_head": fingerprint(target), "sources": pins,
                  "input": "final transformer hidden [B,201,1024], same dtype/device as original PE head",
                  "output": "existing head.base input [B,1024], unnormalized", "weights": "exact INT8 q.float()*s.float() row dequantization from original B119 ZIP",
                  "pool_parameter_count": sum(p.numel() for p in feature.parameters()), "cpu_checks": checks,
                  "torch": torch.__version__, "training": False, "gpu": False, "extra_encoder_forward": False,
                  "original_zip_and_logit_graph_unchanged": True,
                  "runtime_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu,
                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "limitations": "CPU FP32 canary-hidden parity only; quantized model geometry differs from unquantized LODO research, GPU BF16/actual image/full-gallery runtime remain root acceptance gates."}
        exclusive_json(output / "report.json", report)
        return report
    except BaseException as exc:
        exclusive_json(output / "failure.json", {"error": str(exc), "type": type(exc).__name__, "partial_preserved": True, "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(build(args.package, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
