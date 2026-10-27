"""Full public CUDA capture with an explicit immutable recipe identity.

Unlike the historical B121 harness, no module-global candidate is replaced.
This local-only helper neither fits nor submits, and never certifies A10 time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

from .gallery_feature_package import pin, sha, write_new

ROOT = Path(__file__).resolve().parents[2]


def validate_package(artifact, extracted, expected_config):
    artifact, extracted = Path(artifact).resolve(), Path(extracted).resolve()
    if (not isinstance(expected_config.get("candidate"), str)
            or expected_config.get("plans") != [{"name": "A", "members": [0, 1, 2]}]
            or expected_config.get("min_active_members") != 3
            or expected_config.get("time_budget_s") != 292
            or expected_config.get("rank_reserve_s") != 40
            or expected_config.get("require_cuda") is not True
            or expected_config.get("fail_closed_on_fallback") is not True):
        raise ValueError("explicit mandatory3/292s/40s/CUDA recipe required")
    members = {}
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP members")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
                raise ValueError("unsafe non-file ZIP entry")
            with archive.open(name) as stream:
                members[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    files = {p.relative_to(extracted).as_posix(): p for p in extracted.rglob("*") if p.is_file()}
    if set(files) != set(members) or any(p.is_symlink() for p in extracted.rglob("*")):
        raise ValueError("extracted files differ or contain links")
    if any(sha(files[name]) != value for name, value in members.items()):
        raise ValueError("extracted bytes differ from archive")
    if json.loads((extracted / "assets/config.json").read_text()) != expected_config:
        raise ValueError("actual config differs from explicit recipe")
    return members


def execute(artifact, extracted, output, expected_sha, expected_config, *, time_scale=1.):
    from .data import load_retrieval_split
    from .submission import load_submission_module, make_fixture, validate_output

    artifact, extracted, output = (Path(x).resolve() for x in (artifact, extracted, output))
    if sha(artifact) != expected_sha or not np.isfinite(time_scale) or not 1 <= time_scale <= 4:
        raise ValueError("exact artifact and finite local scale1..4 required")
    members = validate_package(artifact, extracted, expected_config)
    split_dir = ROOT / "starter_kit/data/annotations/task2/val"
    sources = [pin(Path(__file__)), pin(artifact)]
    sources += [pin(ROOT / "src/upar" / name) for name in ("data.py", "submission.py", "gallery_feature_package.py")]
    annotations = [pin(p) for p in sorted(split_dir.glob("*.csv"))]
    split = load_retrieval_split(split_dir)
    if (split.gallery_size, split.query_count) != (33407, 3462):
        raise ValueError("complete public33407/3462 only")
    fixture = make_fixture(split, query_count=3462, gallery_size=33407, image_root=ROOT / "starter_kit/data")
    if any(not Path(x["image_path"]).is_file() for x in fixture["gallery"]):
        raise ValueError("missing actual image")
    if any(name in sys.modules for name in ("b121_graph", "b121_agreement", "b121_blas", "b121_runtime", "b122c_constant")):
        raise ValueError("fresh isolated worker required")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {"status": "RUNNING_NOT_ACCEPTED", "candidate": expected_config["candidate"],
        "sources": sources, "annotations": annotations, "archive_members": members,
        "time_scale": time_scale, "public_only": True, "runtime_sources_unmodified": True})
    os.environ["UPAR_TIME_SCALE"] = str(time_scale)
    os.environ.pop("UPAR_FORCE_PLAN", None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(extracted))
    try:
        started = time.monotonic()
        module = load_submission_module(extracted / "run.py")
        module.load_model()
        loaded = time.monotonic()
        if str(module._DEVICE) != "cuda":
            raise ValueError("actual CUDA required; no fallback")
        original, captured = module.predict_attributes, []
        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            captured.append(np.asarray(value).copy())
            return value
        module.predict_attributes = capture
        try:
            value = module.rank_gallery(fixture)
        finally:
            module.predict_attributes = original
        finished = time.monotonic()
        kind, distances = validate_output(value, query_count=3462, gallery_size=33407)
        if kind != "distances" or distances.dtype != np.float32 or len(captured) != 1:
            raise ValueError("one FP32 complete predictor/distance result required")
        p, features = captured[0], np.asarray(module._GALLERY_FEATURES)
        if any(x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all()
               for x, shape in ((p, (33407, 40)), (features, (33407, 1024)), (distances, (3462, 33407)))):
            raise ValueError("complete finite FP32 outputs required")
        np.save(output / "distances.npy", distances, allow_pickle=False)
        np.save(output / "features.npy", features, allow_pickle=False)
        np.savez(output / "capture.npz", probabilities=p, queries=np.asarray(fixture["queries"], np.float32),
                 image_paths=np.asarray(split.image_paths), attribute_names=np.asarray(fixture["attribute_names"]))
        if validate_package(artifact, extracted, expected_config) != members:
            raise ValueError("package changed during runtime")
        for ref in sources + annotations:
            if pin(ref["path"]) != ref:
                raise ValueError("input/source changed")
        report = {"status": "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING", "candidate": expected_config["candidate"],
            "artifact": pin(artifact), "sources": sources, "annotations": annotations, "archive_members": members,
            "outputs": [pin(output / n) for n in ("distances.npy", "features.npy", "capture.npz")],
            "load_seconds": loaded-started, "rank_seconds": finished-loaded,
            "runtime_seconds_before_local_saves": finished-started, "time_scale": time_scale,
            "forced_plan": None, "device": str(module._DEVICE), "rows": 33407, "queries": 3462,
            "new_reference_pass": False, "config": expected_config,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "Exact new ZIP public capture; not official A10 guarantee or new quality evaluation."}
        write_new(output / "execute_report.json", report)
        return report
    except BaseException as error:
        write_new(output / "failure.json", {"type": type(error).__name__, "error": str(error), "partial_preserved": True, "retry": False})
        raise
