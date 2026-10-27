"""Build one B121 inference-only variant from the immutable successful B119 ZIP.

All original model/weight/license bytes are retained. Only runner/config change;
one independently verified restored PE pooling head and pure NumPy helpers are
added. This does not train, submit, run GPU inference or certify runtime quality.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from .saved_checkpoint_predict import _move_no_replace, _sync_dir

ROOT = Path(__file__).resolve().parents[2]
BASE_ZIP = ROOT / "artifacts/runs/20261016T1710+0900_b119g/package/B119G_PE4Q_HP4SIB_EVAS_FASTG.zip"
BASE_SHA = "ace7e1778e70abe55cc3fbfa09304769ca6a8a615ea827a250b2c6a9bf5908ca"
BASE_RUN_SHA = "a64ef0e42e196a902c4c2dd854322e367751f08101ba406653c94c27d1e6df24"
BASE_CONFIG_SHA = "f6d71f2e343423de1de753b4bd01cb588842238c837f63de5f30e99af20815f7"
NAME = "B121G_PE4_FEATURE_AGREEMENT"
FEATURE_NAME = "assets/pe_feature_head.ts"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}


def write_new(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"baseline patch boundary differs: {old[:90]!r}")
    return text.replace(old, new, 1)


def patch_runner(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_RUN_SHA:
        raise ValueError("not the immutable successful B119 runner")
    text = source
    for old, new in (
        (
            "from PIL import Image\n",
            "from PIL import Image\nfrom b121_runtime import load_feature_head, apply_feature_rerank, validated_time_scale, runtime_budget_seconds\n",
        ),
        ("_POOL = None\n", "_POOL = None\n_GALLERY_FEATURES = None\n_B121_TIME_SCALE = validated_time_scale()\n"),
        (
            "        self.prefix, self.block, self.head, self.params = prefix, block, head, params\n",
            "        self.prefix, self.block, self.head, self.params = prefix, block, head, params\n        self.feature_head = None\n        self.last_features = None\n",
        ),
        (
            "        return self.head(h)\n",
            "        logits = self.head(h)\n        if self.feature_head is not None:\n            self.last_features = self.feature_head(h)\n        return logits\n",
        ),
        (
            "    model = _BlockStack(graphs[0], graphs[1], graphs[2], params) if blocks else graphs[0]\n",
            "    model = _BlockStack(graphs[0], graphs[1], graphs[2], params) if blocks else graphs[0]\n    if spec.get('gallery_feature_head'):\n        if not blocks or spec.get('flip_tta') or spec.get('backbone') != 'pe_core_l14_280':\n            raise RuntimeError('feature head requires exact original-view PE block member')\n        model.feature_head = load_feature_head(_HERE, spec, device, dtype)\n",
        ),
        ("    global _FALLBACK\n", "    global _FALLBACK, _GALLERY_FEATURES\n"),
        (
            "    probs = [np.empty((total, 40), dtype=np.float32) for _ in specs]\n",
            "    probs = [np.empty((total, 40), dtype=np.float32) for _ in specs]\n    if sum(bool(s.get('gallery_feature_head')) for s in specs) != 1:\n        raise RuntimeError('exactly one PE pooled feature source required')\n    _GALLERY_FEATURES = np.full((total, 1024), np.nan, dtype=np.float32)\n",
        ),
        (
            "            probs[i][start:start + raw.shape[0]] = _run(models[i], specs[i], raw)\n",
            "            probs[i][start:start + raw.shape[0]] = _run(models[i], specs[i], raw)\n            if specs[i].get('gallery_feature_head'):\n                feature = models[i].last_features\n                if feature is None or feature.ndim != 2 or feature.shape[1] != 1024 or feature.shape[0] < raw.shape[0]:\n                    raise RuntimeError('missing or incomplete same-forward PE pooled features')\n                _GALLERY_FEATURES[start:start + raw.shape[0]] = feature[:raw.shape[0]].float().cpu().numpy()\n",
        ),
        (
            '    return out, "+".join(s["name"] for s in specs[:active])\n',
            '    if not np.isfinite(_GALLERY_FEATURES).all():\n        raise RuntimeError(\'incomplete/nonfinite PE feature gallery\')\n    return out, "+".join(s["name"] for s in specs[:active])\n',
        ),
        (
            '    _log("ranked", queries=len(queries), gallery=len(probabilities))\n',
            "    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE):\n        raise RuntimeError('no remaining time for mandatory feature reranking')\n    distances = apply_feature_rerank(distances, _GALLERY_FEATURES, probabilities, _log)\n    if time.time() - _T0 >= runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE):\n        raise RuntimeError('feature reranking exceeded fixed total budget; fail closed')\n    _log(\"ranked\", queries=len(queries), gallery=len(probabilities))\n",
        ),
    ):
        text = replace_once(text, old, new)
    old_ast, new_ast = ast.parse(source), ast.parse(text)
    unchanged = (
        "_run",
        "_canary_check",
        "load_model",
        "_decode_one",
        "_decode_batch",
        "_apply_post",
        "predict_attributes",
    )
    for name in unchanged:
        old_node = next(n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new_node = next(n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old_node, include_attributes=False) != ast.dump(new_node, include_attributes=False):
            raise ValueError(f"unchanged B119 function unexpectedly changed: {name}")
    return text


def patch_config(config, feature_sha, reserve_seconds):
    if not 6 < float(reserve_seconds) < float(config["time_budget_s"]):
        raise ValueError("measured additional-head/graph reserve required, below unchanged total budget")
    if (
        config["candidate"] != "B119G_PE4Q_HP4SIB_EVAS_FASTG"
        or len(config["models"]) != 3
        or config["plans"] != [{"name": "A", "members": [0, 1, 2]}]
        or config.get("min_active_members") != 3
        or not config.get("fail_closed_on_fallback")
        or config["models"][0]["backbone"] != "pe_core_l14_280"
        or any(s.get("flip_tta") for s in config["models"])
    ):
        raise ValueError("not the expected full three-member original B119 plan")
    result = copy.deepcopy(config)
    result["candidate"] = NAME
    result["rank_reserve_s"] = float(reserve_seconds)
    result["models"][0]["gallery_feature_head"] = FEATURE_NAME
    result["models"][0]["gallery_feature_sha256"] = feature_sha
    result["gallery_rerank"] = {
        "recipe": "pe_mutual20_pred40_same_original_mass_v1",
        "k": 20,
        "alpha": 0.1,
        "threshold": 0.5,
        "steps": 1,
        "feature_width": 1024,
        "graph_chunk": 256,
        "extra_encoder_forward": False,
        "extra_pool_head_forward": True,
        "original_neighbor_mass": True,
    }
    return result


def portable_sources():
    graph = (ROOT / "src/upar/fixed_gallery_rerank.py").read_text()
    gate = (ROOT / "src/upar/fixed_gallery_agreement.py").read_text()
    gate = replace_once(
        gate,
        "from .fixed_gallery_rerank import _positive_int, _validate_graph",
        "from b121_graph import _positive_int, _validate_graph",
    )
    original = (
        ROOT / "artifacts/runs/20261017T0446+0900_gallery_reciprocal_rerank/run_feature_fixed_v1.py"
    ).read_text()
    node = next(
        n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "encode_unit_features"
    )
    normalization = ast.get_source_segment(original, node)
    runtime = (
        '''"""Fixed no-fit gallery feature agreement; exact local implementation."""
import hashlib
import math
import os
import time
import numpy as np
import torch
from b121_graph import build_mutual_graph
from b121_agreement import retain_identical_predicted_profiles, smooth_with_original_degree
from b121_blas import numpy_blas_threads

def validated_time_scale():
    scale = float(os.environ.get('UPAR_TIME_SCALE', '1'))
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError('UPAR_TIME_SCALE must be finite in [1,4]; local replay only')
    return scale

def runtime_budget_seconds(config, scale):
    if not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError('invalid frozen local time scale')
    budget = float(config['time_budget_s'])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('invalid time budget')
    return budget * scale

def load_feature_head(here, spec, device, dtype):
    path = os.path.join(here, spec['gallery_feature_head'])
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if digest != spec['gallery_feature_sha256']:
        raise RuntimeError('PE feature-head bytes differ from fixed package')
    return torch.jit.load(path, map_location=device).eval().to(dtype)

'''
        + normalization
        + """

def apply_feature_rerank(distances, features, probabilities, log):
    if features is None or features.shape != (len(probabilities), 1024):
        raise RuntimeError('missing complete same-forward PE pooled gallery')
    started = time.time()
    encoded = encode_unit_features(features)
    with numpy_blas_threads(2) as blas:
        graph, original = build_mutual_graph(encoded, np.ones(1024, np.float64), k=20, chunk_size=256)
        del encoded
        retained, count = retain_identical_predicted_profiles(probabilities, graph, original)
        graph_seconds = time.time()-started
        output = smooth_with_original_degree(distances, retained, count, original, alpha=.1)
    if not np.isfinite(output).all() or output.shape != distances.shape:
        raise RuntimeError('invalid final gated distances')
    log('feature_reranked', graph_seconds=round(graph_seconds, 4), total_seconds=round(time.time()-started, 4),
        original_edges=int(original.sum()), kept_edges=int(count.sum()), isolated=int((count==0).sum()),
        extra_encoder_forward=False, extra_pool_head_forward=True, numpy_blas=blas)
    return output
"""
    )
    blas = (ROOT / "src/upar/numpy_blas_scope.py").read_text()
    return {"b121_graph.py": graph, "b121_agreement.py": gate, "b121_blas.py": blas, "b121_runtime.py": runtime}


def build(feature_head, feature_receipt, reserve_receipt, output, reserve_seconds):
    """CPU-only immutable parent reuse; requires caller-verified feature receipt."""
    feature_head, feature_receipt, reserve_receipt, output = map(
        lambda p: Path(p).resolve(), (feature_head, feature_receipt, reserve_receipt, output)
    )
    if output.exists():
        raise FileExistsError("new dedicated output required; partial evidence must be inspected")
    if sha(BASE_ZIP) != BASE_SHA:
        raise ValueError("B119 ZIP changed")
    if (
        shutil.disk_usage(output.parent).free
        < BASE_ZIP.stat().st_size * 4 + feature_head.stat().st_size * 3 + 20 * 2**30
    ):
        raise RuntimeError("insufficient shared disk for preserved package and reserve")
    input_pins = [pin(BASE_ZIP), pin(feature_head), pin(feature_receipt), pin(reserve_receipt), pin(Path(__file__))]
    input_pins += [
        pin(ROOT / "src/upar" / n)
        for n in ("fixed_gallery_rerank.py", "fixed_gallery_agreement.py", "numpy_blas_scope.py")
    ]
    input_pins.append(pin(ROOT / "artifacts/runs/20261017T0446+0900_gallery_reciprocal_rerank/run_feature_fixed_v1.py"))
    # Receipt schema is finalized with the independent feature-head exporter;
    # no build is allowed merely because a nonempty .ts file exists.
    receipt = json.loads(feature_receipt.read_text())
    if (
        receipt.get("status") != "CPU_FEATURE_HEAD_READY_NOT_GPU_VERIFIED"
        or receipt.get("source_zip", {}).get("sha256") != BASE_SHA
        or receipt.get("feature_head", {}).get("sha256") != sha(feature_head)
        or receipt.get("original_zip_and_logit_graph_unchanged") is not True
        or receipt.get("extra_encoder_forward") is not False
        or {row.get("batch") for row in receipt.get("cpu_checks", [])} != {1, 3, 8}
        or any(
            row.get("native_full_head_vs_original_archived_logit_max_abs") != 0
            or row.get("reloaded_feature_vs_native_head_base_prehook_max_abs") != 0
            for row in receipt.get("cpu_checks", [])
        )
    ):
        raise ValueError("actual feature-head verification receipt required")
    process_receipt = feature_receipt.with_name("process_exit_v1.json")
    actual = json.loads(process_receipt.read_text())
    if (
        actual.get("actual_exit_code") != 0
        or not actual.get("exec_session")
        or actual.get("report_sha256") != sha(feature_receipt)
        or actual.get("feature_head_sha256") != sha(feature_head)
    ):
        raise ValueError("feature-head actual successful process receipt required")
    input_pins.append(pin(process_receipt))
    reserve = json.loads(reserve_receipt.read_text())
    if reserve.get("chosen_rank_reserve_s") != float(reserve_seconds) or reserve.get("total_time_budget_s") != 292:
        raise ValueError("fixed measured runtime reserve receipt differs")
    started = time.monotonic()
    stage = Path(tempfile.mkdtemp(prefix=".b121_package_", dir=output.parent))
    write_new(
        stage / "started.json",
        {"inputs": input_pins, "reserve_s": reserve_seconds, "new_training": False, "GPU": False},
    )
    try:
        package = stage / "package"
        package.mkdir()
        kept = {}
        with zipfile.ZipFile(BASE_ZIP) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
                raise ValueError("invalid baseline ZIP paths")
            source = archive.read("run.py")
            config = archive.read("assets/config.json")
            if (
                hashlib.sha256(source).hexdigest() != BASE_RUN_SHA
                or hashlib.sha256(config).hexdigest() != BASE_CONFIG_SHA
            ):
                raise ValueError("B119 runner/config exact bytes differ")
            for name in names:
                if name in ("run.py", "assets/config.json"):
                    continue
                dest = package / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024**2)
                kept[name] = pin(dest)
            (package / "run.py").write_text(patch_runner(source.decode()))
            write_new(
                package / "assets/config.json", patch_config(json.loads(config), sha(feature_head), reserve_seconds)
            )
        for name, source in portable_sources().items():
            with (package / name).open("x") as stream:
                stream.write(source)
        with feature_head.open("rb") as src, (package / FEATURE_NAME).open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        zip_path = stage / f"{NAME}.zip"
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(package.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(package))
        # Compare every unchanged member against the original archive, not just
        # model filename/config declarations. No old weight is re-quantized.
        preserved = []
        with zipfile.ZipFile(BASE_ZIP) as before, zipfile.ZipFile(zip_path) as after:
            for name in kept:
                hashes = []
                for archive in (before, after):
                    h = hashlib.sha256()
                    with archive.open(name) as stream:
                        for block in iter(lambda: stream.read(8 * 1024**2), b""):
                            h.update(block)
                    hashes.append(h.hexdigest())
                if hashes[0] != hashes[1]:
                    raise ValueError(f"unchanged asset differs: {name}")
                preserved.append({"name": name, "sha256": hashes[0]})
        for ref in input_pins:
            if pin(ref["path"]) != ref:
                raise ValueError("packaging input changed")
        report = {
            "schema": 1,
            "status": "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED",
            "candidate": NAME,
            "inputs": input_pins,
            "zip": {"name": zip_path.name, "sha256": sha(zip_path), "size": zip_path.stat().st_size},
            "unchanged_original_members": preserved,
            "rank_reserve_s": reserve_seconds,
            "total_time_budget_s": 292,
            "runtime_seconds": time.monotonic() - started,
            "GPU_seconds": 0,
            "new_training": False,
            "submission": False,
            "GPU_replay": "NOT_RUN",
            "full_runtime_acceptance": False,
        }
        write_new(stage / "complete.json", report)
        _move_no_replace(stage, output)
        _sync_dir(output.parent)
        return report
    except BaseException as exc:
        write_new(
            stage / "failure.json",
            {"type": type(exc).__name__, "error": str(exc), "partial_preserved": True, "retry": False},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-head", "feature-receipt", "reserve-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reserve-seconds", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.feature_head, args.feature_receipt, args.reserve_receipt, args.output, args.reserve_seconds)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
