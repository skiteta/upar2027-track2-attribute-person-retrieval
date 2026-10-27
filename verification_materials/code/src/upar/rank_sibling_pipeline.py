"""Collect a fixed EMA rank-sibling replacement, weight-soup it, and compare.

No training, queue changes, submission, probability-average substitution for
weight soups, or implicit GPU admission. Existing h1.soup and the registered
paired comparator are reused. New outputs are staged and atomically published.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/runs/20261017T0300+0900_gold_execution"
BASELINE = ROOT / "artifacts/runs/20261015T1200+0900_b116g/candidate_b116g.json"
CURRENT_BASELINE = ROOT / "artifacts/runs/20261016T1710+0900_b119g/candidate_b119g.json"
COMPARATOR = ROOT / "artifacts/runs/20261014T2310+0900_os_restore/local_validation/lodo_foundation_v1.py"
FOLDS = ("PA100k", "PETA", "Market1501")
CANDIDATE = "PE_SOFTADM_RANK_REPLACEMENT_V1"
CURRENT_CANDIDATE = "PE_SOFTADM_RANK_REPLACEMENT_B119_SECONDARY_V1"
POST = {"renorm": [[8, 20, 0.5]], "clip": 1e-5, "entropy_delta": 0.1875}


class CollectorBusy(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def path_of(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ref(path):
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    p = Path(path)
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed during hashing: {p}")
    return {"path": ref(p), "sha256": h.hexdigest(), "size": after.st_size}


def exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_start_ticks(pid):
    try:
        # Field 22, after the parenthesized process name (which can contain spaces).
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def atomic_json(path, value):
    """Only the single collector owner may update its live status, via rename."""
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    exclusive_json(temporary, value)
    os.replace(temporary, path)


class CollectorOwner:
    """Nonblocking flock, shared with any live subprocess through pass_fds.

    Closing (not explicit LOCK_UN) preserves the lock if a child survives its
    parent. No process is stopped, and this lock is NOT a GPU resource lease.
    """
    def __init__(self, run):
        self.directory = Path(run) / "softadm_collection_v1"
        self.lock_fd = None
        self.value = {}

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_fd = os.open(self.directory / "collector.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise CollectorBusy("COLLECTOR_BUSY: another collector or its live child owns the lock; nothing dispatched") from exc
        self.value = {"schema": 1, "owner_token": uuid.uuid4().hex, "pid": os.getpid(),
                      "process_start_ticks": process_start_ticks(os.getpid()), "process_started_at": now(),
                      "process_completed_at": None, "completed_folds": [],
                      "resource_probe_is_reservation": False,
                      "coordination": "Root coordinates GPU admission with the existing queue; no automatic ADMISSION_HOLD modification and no training stops."}
        try:
            self.update(state="STARTED", activity="collector initialization")
        except BaseException:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise
        return self

    def update(self, **values):
        self.value.update(values)
        self.value["updated_at"] = now()
        atomic_json(self.directory / "collector_status.json", self.value)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self.update(state="FAILED_OR_INTERRUPTED", error_type=exc_type.__name__, error=str(exc),
                            process_completed_at=now(),
                            note="No child or training process was stopped; a surviving child retains the ownership lock.")
        finally:
            # Do not flock(LOCK_UN): it would unlock the shared open-file
            # description even if a subprocess is still using this descriptor.
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None


def emit(owner, **values):
    if owner is not None:
        owner.update(**values)


def run_child(command, log, owner, *, kind, fold):
    kwargs = {"cwd": ROOT, "stdout": log, "stderr": subprocess.STDOUT}
    if owner is not None:
        kwargs["pass_fds"] = (owner.lock_fd,)
    process = subprocess.Popen(command, **kwargs)
    operation = {"kind": kind, "fold": fold, "pid": process.pid,
                 "process_start_ticks": process_start_ticks(process.pid),
                 "started_at": now(), "completed_at": None, "exit_code": None}
    emit(owner, state="RUNNING", activity=kind, fold=fold, operation=operation)
    code = process.wait()
    operation.update(completed_at=now(), exit_code=code)
    emit(owner, state="OPERATION_COMPLETED" if code == 0 else "OPERATION_FAILED", operation=operation)
    if code:
        raise subprocess.CalledProcessError(code, command)


def completed(run, *, require_weights=True):
    """A last-written parseable report plus closed output files, not mere NPZ existence."""
    p = Path(run)
    files = [p / "config.json", p / "val_logits.npz", p / "train_report.json"]
    if require_weights:
        files.append(p / "model.safetensors")
    if not all(x.is_file() and x.stat().st_size > 0 for x in files):
        return False
    try:
        report = read_json(p / "train_report.json")
    except (OSError, ValueError):
        return False
    # The current trainer writes its report last. Two seconds excludes a writer
    # completing just as the observer sees the directory; deep validation follows.
    outputs = [p / "val_logits.npz"] + ([p / "model.safetensors"] if require_weights else [])
    return (report.get("status") in ("TRAIN_DONE", "SOUP_DONE")
            and all(time.time() - x.stat().st_mtime >= 2 for x in files)
            and (p / "train_report.json").stat().st_mtime_ns >= max(x.stat().st_mtime_ns for x in outputs))


def expected_validation(cfg):
    with np.load(path_of(cfg["cache_dir"]) / "meta.npz", allow_pickle=False) as z:
        meta = {k: z[k] for k in ("split", "sources", "image_paths", "attribute_names")}
    val = meta["split"] == "val"
    fit = (meta["split"] == "train") & (meta["sources"] != cfg["fold"])
    sources, counts = np.unique(meta["sources"][fit], return_counts=True)
    return meta["image_paths"][val], meta["attribute_names"], dict(zip(sources.tolist(), counts.astype(int).tolist()))


def validate_run(run, fold, *, require_weights=True, deep_weights=False, ema=False, seen=None):
    """Validate saved rows, complete fit lineage, and optionally the exact weight bytes."""
    p = Path(run)
    seen = set() if seen is None else set(seen)
    if str(p.resolve()) in seen:
        raise ValueError(f"cyclic provenance: {p}")
    seen.add(str(p.resolve()))
    tracked = [p / n for n in ("config.json", "train_report.json", "val_logits.npz")]
    if require_weights:
        tracked.append(p / "model.safetensors")
    snapshot = {str(x): (x.stat().st_size, x.stat().st_mtime_ns) for x in tracked}
    cfg, report = read_json(p / "config.json"), read_json(p / "train_report.json")
    if cfg.get("fold") != fold or cfg.get("include_val") is not False:
        raise ValueError(f"held-out fit boundary mismatch: {p}")
    if cfg.get("seed") != 20261007 or cfg.get("head", "linear") != "attrq" or cfg.get("backbone") != "pe_core_l14_280":
        raise ValueError(f"PE same-seed architecture mismatch: {p}")
    if ema and (cfg.get("ema_used") != .999 or not cfg.get("ema_of")):
        raise ValueError(f"EMA0.999 required, not raw weights: {p}")
    if report.get("status") not in ("TRAIN_DONE", "SOUP_DONE"):
        raise ValueError(f"incomplete report: {p}")
    if report.get("fold") != fold or report.get("include_val") is not False:
        raise ValueError(f"report boundary mismatch: {p}")
    paths, names, counts = expected_validation(cfg)
    zpath = p / "val_logits.npz"
    before = zpath.stat()
    with zipfile.ZipFile(zpath) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt NPZ: {p}")
    with np.load(zpath, allow_pickle=False) as z:
        for key in ("logits_original", "logits_flip"):
            if z[key].shape != (len(paths), len(names)) or not np.isfinite(z[key]).all():
                raise ValueError(f"invalid logits shape/finiteness: {p}/{key}")
        if not np.array_equal(z["image_paths"], paths) or not np.array_equal(z["attribute_names"], names):
            raise ValueError(f"prediction row/attribute order mismatch: {p}")
        if z["pos_weight"].shape != (len(names),) or not (np.isfinite(z["pos_weight"]).all() and (z["pos_weight"] > 0).all()):
            raise ValueError(f"invalid positive weights: {p}")
    after = zpath.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"prediction changed during validation: {p}")
    lineage = cfg.get("soup_of") or ([cfg["pw_aligned_of"]] if cfg.get("pw_aligned_of") else [])
    if not lineage and report.get("fit_rows_by_source") != counts:
        raise ValueError(f"source fit counts differ from cache split: {p}")
    extra = ([cfg["init_from"]] if cfg.get("init_from") else []) + cfg.get("online_teachers", [])
    parents = [validate_run(path_of(r), fold, require_weights=False, seen=seen) for r in lineage + extra]
    result = {"run": ref(p), "fold": fold, "include_val": False, "fit_rows_by_source": counts,
              "files": [fingerprint(p / n) for n in ("config.json", "train_report.json", "val_logits.npz")],
              "ema_used": cfg.get("ema_used"), "parents": parents}
    if require_weights:
        from safetensors import safe_open
        with safe_open(p / "model.safetensors", framework="np") as weights:
            if not list(weights.keys()):
                raise ValueError(f"empty weights: {p}")
        result["weights"] = (fingerprint(p / "model.safetensors") if deep_weights
                              else {"path": ref(p / "model.safetensors"), "size": (p / "model.safetensors").stat().st_size,
                                    "sha256": "not_read_in_status_check"})
    if cfg.get("ema_of"):
        raw = path_of(cfg["ema_of"])
        requests = sorted(raw.glob("attempt_*.request.json"))
        result["training_launch_evidence"] = [fingerprint(x) for x in requests] or "legacy_not_recorded"
    if snapshot != {str(x): (x.stat().st_size, x.stat().st_mtime_ns) for x in tracked}:
        raise RuntimeError(f"run files changed during validation: {p}")
    return result


def fold_plan(run, fold):
    base = read_json(BASELINE)
    if [m["weight"] for m in base["lodo"]["members"]] != [.4, .35, .25] or base["lodo"]["post"] != POST:
        raise ValueError("B116 reference weights/post changed")
    old = path_of(base["lodo"]["members"][0]["runs"][fold][0])
    cfg = read_json(old / "config.json")
    parents = [path_of(x) for x in cfg["soup_of"]]
    if len(parents) != 4 or cfg.get("soup_weights") != [.25] * 4:
        raise ValueError("Expected B116 equal-weight four-recipe PE soup")
    if "pLq_rankq_" not in parents[1].name:
        raise ValueError("Second soup member is not the registered micro rank sibling")
    mask_cfg = read_json(parents[2] / "config.json")
    if not mask_cfg.get("pw_aligned_of") or path_of(mask_cfg["pw_ref"]) != parents[0]:
        raise ValueError("Unexpected mask prior-alignment lineage")
    return {"fold": fold, "old_soup": old, "new_ema": Path(str(run / f"pe_softadm_{fold}") + "_ema0.999"),
            "parents": parents, "mask_source": path_of(mask_cfg["pw_aligned_of"]),
            "mask_output": run / "softadm_collection_v1" / f"maskv2_pw_{fold}",
            "output": run / "softadm_collection_v1" / f"pe4adm_{fold}", "weights": [.25] * 4}


def source_evidence():
    return [fingerprint(p) for p in (Path(__file__), ROOT / "src/upar/h1_finetune.py", ROOT / "scripts/make_pw_aligned.py", COMPARATOR)]


def publish(stage, output):
    if output.exists():
        raise FileExistsError(f"Never overwrite completed or partial output: {output}")
    os.rename(stage, output)  # same parent/filesystem; readers see the complete directory


def relocated(value, stage, output):
    """Preserve hashes while recording the eventual atomic publication path."""
    if isinstance(value, dict):
        return {k: relocated(v, stage, output) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated(v, stage, output) for v in value]
    if isinstance(value, str):
        old, new = ref(stage), ref(output)
        if value == old or value.startswith(old + "/"):
            return new + value[len(old):]
    return value


def check_resources(gpu_need_mib=8000, host_need_gib=24):
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    free = int(result.stdout.split()[0])
    available = next(int(line.split()[1]) / 1024**2 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if free < gpu_need_mib + 2048 or available < host_need_gib:
        raise RuntimeError(f"RESOURCE_WAIT: GPU free {free} MiB, RAM available {available:.2f} GiB")
    return {"free_gpu_mib": free, "min_gpu_mib_including_reserve": gpu_need_mib + 2048,
            "host_available_gib": available, "min_host_available_gib": host_need_gib}


def build_fold(run, fold, *, allow_gpu=False, owner=None):
    if not allow_gpu:
        raise PermissionError("Explicit --allow-gpu required after root resource admission")
    plan = fold_plan(run, fold)
    if (plan["output"] / "collection_receipt.json").exists():
        if read_json(plan["output"] / "collection_receipt.json").get("status") != "COMPLETE":
            raise ValueError("Existing collection receipt is not complete")
        validate_run(plan["output"], fold)
        return {"status": "ALREADY_COLLECTED", "output": ref(plan["output"])}
    if plan["output"].exists():
        raise RuntimeError(f"Existing output lacks completion receipt; preserve and inspect it: {plan['output']}")
    if not completed(plan["new_ema"]):
        emit(owner, state="WAITING_FOR_EMA", activity="waiting for closed EMA outputs", fold=fold)
        return {"status": "WAITING_FOR_COMPLETE_EMA", "fold": fold}
    resources = check_resources()
    emit(owner, state="RUNNING", activity="CPU input validation and weight hashing", fold=fold, resource_probe=resources)
    inputs = [plan["parents"][0], plan["new_ema"], plan["mask_source"], plan["parents"][3]]
    evidence = [validate_run(p, fold, deep_weights=True, ema=True) for p in inputs]
    new_cfg = read_json(plan["new_ema"] / "config.json")
    if new_cfg.get("rank_loss_kind") != "soft_adm" or new_cfg.get("rank_scope") != "batch":
        raise ValueError("New sibling is not the committed softADM batch recipe")
    if new_cfg.get("rank_alpha") != .5 or new_cfg.get("rank_adm_temperature") != 1.0:
        raise ValueError("New sibling coefficient/temperature differs from committed recipe")
    parent = plan["output"].parent
    parent.mkdir(parents=True, exist_ok=True)
    if not plan["mask_output"].exists():
        stage_mask = Path(tempfile.mkdtemp(prefix=f"stage_mask_{fold}_", dir=parent))
        command = [sys.executable, str(ROOT / "scripts/make_pw_aligned.py"), str(plan["mask_source"]),
                   str(plan["parents"][0]), str(stage_mask)]
        with (stage_mask / "build.log").open("x") as log:
            run_child(command, log, owner, kind="CPU prior alignment", fold=fold)
        cfg = read_json(stage_mask / "config.json")
        cfg["output_dir"] = ref(plan["mask_output"])
        (stage_mask / "config.json").write_text(json.dumps(cfg, indent=2))
        validate_run(stage_mask, fold)
        publish(stage_mask, plan["mask_output"])
    else:
        validate_run(plan["mask_output"], fold)
    parents = [plan["parents"][0], plan["new_ema"], plan["mask_output"], plan["parents"][3]]
    stage = Path(tempfile.mkdtemp(prefix=f"stage_soup_{fold}_", dir=parent))
    command = [sys.executable, "-m", "upar.h1_finetune", "soup", "--runs", *[ref(p) for p in parents], "--output", str(stage)]
    resources = check_resources()  # hashing/CPU prior alignment may have taken time
    emit(owner, resource_probe=resources)
    request = {"created_at": now(), "operation": "state_dict_weight_soup_then_new_forward", "argv": command,
               "weights": plan["weights"], "inputs": evidence, "source": source_evidence(), "resources": resources}
    exclusive_json(stage / "collection_request.json", request)
    with (stage / "build.log").open("x") as log:
        run_child(command, log, owner, kind="GPU state-dict soup prediction", fold=fold)
    cfg = read_json(stage / "config.json")
    if cfg.get("soup_of") != [ref(p) for p in parents] or cfg.get("soup_weights") != plan["weights"]:
        raise ValueError("Generated soup parent order/weights differ")
    cfg["output_dir"] = ref(plan["output"])
    (stage / "config.json").write_text(json.dumps(cfg, indent=2))
    result = relocated(validate_run(stage, fold, deep_weights=True), stage, plan["output"])
    # All outputs are validated before the final receipt and atomic publication.
    exclusive_json(stage / "collection_receipt.json", {"status": "COMPLETE", "completed_at": now(),
                   "output": ref(plan["output"]), "validation": result,
                   "semantics": "Uniform arithmetic state_dict soup, EMA0.999 leaves; predictions from the resulting model"})
    publish(stage, plan["output"])
    done_folds = sorted(set((owner.value.get("completed_folds", []) if owner else []) + [fold]))
    emit(owner, state="FOLD_COMPLETED", activity="verified outputs atomically published", fold=fold,
         completed_folds=done_folds, last_fold_completed_at=now())
    return {"status": "COLLECTED", "output": ref(plan["output"])}


def candidate_spec(run):
    spec = copy.deepcopy(read_json(BASELINE))
    # A local candidate is not an already-packaged / runtime-validated submission.
    spec = {"candidate": CANDIDATE, "lodo": spec["lodo"], "evaluation_role": "development"}
    for fold in FOLDS:
        spec["lodo"]["members"][0]["runs"][fold] = [ref(fold_plan(run, fold)["output"])]
    return spec


def current_candidate_spec(run):
    """Additional current-incumbent contrast; never revise the B116 primary."""
    original, current = read_json(BASELINE), read_json(CURRENT_BASELINE)
    for value, name in ((original, "B116G"), (current, "B119G")):
        members = value["lodo"]["members"]
        if value["candidate"] != name or [m["weight"] for m in members] != [.4, .35, .25] or value["lodo"]["post"] != POST:
            raise ValueError("Fixed B116/B119 member weights or postprocessor changed")
        if any(set(m["runs"]) != set(FOLDS) or any(len(m["runs"][f]) != 1 for f in FOLDS) for m in members):
            raise ValueError("Expected one fixed run per member and fold")
    a, b = original["lodo"]["members"], current["lodo"]["members"]
    if a[0] != b[0] or a[2] != b[2]:
        raise ValueError("B116 and B119 must retain the identical PE4 and EVA controls")
    for fold in FOLDS:
        if path_of(b[1]["runs"][fold][0]).name != f"hp_q_qr_qrq_s_{fold}":
            raise ValueError("B119 H+4 sibling control changed")
    spec = {"candidate": CURRENT_CANDIDATE, "lodo": copy.deepcopy(current["lodo"]),
            "evaluation_role": "development"}
    for fold in FOLDS:
        spec["lodo"]["members"][0]["runs"][fold] = [ref(fold_plan(run, fold)["output"])]
    return spec


def freeze_current_comparison(run):
    """Freeze this additive contrast before observing its results, with no GPU."""
    parent = Path(run) / "softadm_collection_v1"
    parent.mkdir(parents=True, exist_ok=True)
    spec_path = parent / "candidate_current_b119_fixed_v1.json"
    protocol_path = parent / "current_b119_protocol_v1.json"
    expected = current_candidate_spec(run)
    if spec_path.exists():
        if read_json(spec_path) != expected:
            raise ValueError("Fixed B119 candidate specification changed")
    else:
        exclusive_json(spec_path, expected)
    protocol = {"schema": 1, "role": "additional_secondary_not_primary_replacement",
                "candidate": fingerprint(spec_path), "baseline": fingerprint(CURRENT_BASELINE),
                "original_primary_baseline": fingerprint(BASELINE),
                "contrast": "Same actual PE4 rank-sibling replacement, fixed B119 H+4/EVA, outer .4/.35/.25 and original post.",
                "original_comparisons": "B116 primary and single softADM-vs-batch-CE remain unchanged and are not reclassified.",
                "selection": "No signs, thresholds or coefficient search gate this comparison; exposed development only.",
                "output": ref(parent / "current_b119")}
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Fixed B119 comparison protocol/input specification changed")
    else:
        exclusive_json(protocol_path, protocol)
    return spec_path, protocol_path


def compare_primary(run, *, owner=None):
    parent = run / "softadm_collection_v1"
    final = parent / "comparison"
    if (final / "comparison_receipt.json").exists():
        return read_json(final / "comparison_receipt.json")
    if final.exists():
        raise RuntimeError(f"Existing comparison lacks receipt; preserve and inspect it: {final}")
    evidence = []
    single_new, single_old = {}, {}
    for fold in FOLDS:
        output = fold_plan(run, fold)["output"]
        if not (output / "collection_receipt.json").exists():
            emit(owner, state="WAITING_FOR_THREE_SOUPS", activity="waiting for complete comparison inputs", fold=fold)
            return {"status": "WAITING_FOR_THREE_SOUPS", "missing_fold": fold}
        evidence.append(validate_run(output, fold))
        new = fold_plan(run, fold)["new_ema"]
        old = ROOT / f"artifacts/runs/20261008T231500+0900_knob_sweep/pLq_rankqb_{fold}_ema0.999"
        if not completed(old):
            emit(owner, state="WAITING_FOR_BATCH_CE_CONTROL", activity="waiting for closed CE control outputs", fold=fold)
            return {"status": "WAITING_FOR_BATCH_CE_CONTROL", "missing_fold": fold}
        validate_run(new, fold, ema=True)
        validate_run(old, fold, ema=True)
        single_new[fold], single_old[fold] = [ref(new)], [ref(old)]
    stage = Path(tempfile.mkdtemp(prefix="stage_comparison_", dir=parent))
    spec_path = parent / "candidate_fixed_v1.json"
    expected = candidate_spec(run)
    if spec_path.exists():
        if read_json(spec_path) != expected:
            raise ValueError("Fixed candidate specification changed")
    else:
        exclusive_json(spec_path, expected)
    single_paths = []
    for name, runs in (("SOFTADM_SINGLE", single_new), ("BATCH_CE_SINGLE", single_old)):
        path = parent / f"{name.lower()}_fixed_v1.json"
        spec = {"candidate": name, "lodo": {"members": [{"weight": 1.0, "runs": runs}], "post": POST}}
        if path.exists():
            if read_json(path) != spec:
                raise ValueError("Fixed single-model specification changed")
        else:
            exclusive_json(path, spec)
        single_paths.append((name, ref(path), None))
    module_spec = importlib.util.spec_from_file_location("rank_sibling_registered_comparator", COMPARATOR)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # Keep its exact scoring / paired-bootstrap implementation. The fixed reference
    # is B116 for this task, so do not re-score or re-audit unrelated B115 controls.
    baseline = ("B116G", ref(BASELINE), None)
    module.FOUNDATION = baseline
    split = module.load_retrieval_split(module.H.SPLITS["val"])
    emit(owner, state="RUNNING", activity="CPU fixed three-fold paired comparison", comparison_started_at=now())
    try:
        result, _ = module.replay_and_compare(split, stage, (CANDIDATE, ref(spec_path), None), baseline)
        single_dir = stage / "single_softadm_vs_batch_ce"
        single_dir.mkdir()
        module.CAP.clear()
        module.FOUNDATION = single_paths[1]
        single_result, _ = module.replay_and_compare(split, single_dir, single_paths[0], single_paths[1])
        exclusive_json(single_dir / "lodo_foundation_results_v1.json", single_result)
    finally:
        module.H.evaluate_probabilities = module._EP
        module.MS.evaluate_scores = module._ES
    exclusive_json(stage / "lodo_foundation_results_v1.json", result)
    receipt = {"status": "COMPLETE", "completed_at": now(), "candidate": CANDIDATE,
               "baseline": "B116G", "paired": result["paired"], "sources": source_evidence(),
               "single_vs_batch_ce": single_result["paired"],
               "comparison_interpretation": {
                   "primary": "Only the rank sibling inside PE4 is replaced; that sibling jointly changes micro->batch scope, list CE->softADM, and alpha0.1->0.5. Fixed outer weights and other parents.",
                   "secondary": "Single models at fixed batch128/source lists; objective and alpha change together. Not isolated causality of alpha or objective alone."},
               "candidate_spec": fingerprint(spec_path), "baseline_spec": fingerprint(BASELINE), "inputs": evidence,
               "limitations": "Reused 3-source development surface; query bootstrap conditional on gallery/model. Not Public/Private probability or package/runtime acceptance."}
    exclusive_json(stage / "comparison_receipt.json", receipt)
    publish(stage, final)
    emit(owner, state="COMPARISON_COMPLETED", activity="verified comparison atomically published", comparison_completed_at=now())
    return receipt


def compare_current(run, *, owner=None):
    """CPU-only additive B119 secondary; complete or partial outputs are immutable."""
    parent = Path(run) / "softadm_collection_v1"
    spec_path, protocol_path = freeze_current_comparison(run)
    final = parent / "current_b119"
    receipt_path = final / "comparison_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("status") != "COMPLETE" or receipt.get("protocol") != fingerprint(protocol_path):
            raise ValueError("Existing B119 secondary receipt is not the fixed complete comparison")
        return receipt
    if final.exists() or list(parent.glob("stage_current_b119_*")):
        raise RuntimeError("Existing partial B119 comparison; preserve and inspect, do not retry")
    primary_path = parent / "comparison/comparison_receipt.json"
    if not primary_path.is_file() or read_json(primary_path).get("status") != "COMPLETE":
        return {"status": "WAITING_FOR_ORIGINAL_COMPARISONS"}
    evidence = []
    for fold in FOLDS:
        output = fold_plan(run, fold)["output"]
        if not (output / "collection_receipt.json").is_file() or read_json(output / "collection_receipt.json").get("status") != "COMPLETE":
            return {"status": "WAITING_FOR_THREE_SOUPS", "missing_fold": fold}
        evidence.append(validate_run(output, fold))
    # Bind every saved prediction used in this CPU replay. Encoder fit boundaries
    # remain those of the same registered held-source parents, not a new fit.
    input_pins = []
    runs = {path_of(value) for spec in (read_json(spec_path), read_json(CURRENT_BASELINE))
            for member in spec["lodo"]["members"] for values in member["runs"].values() for value in values}
    for directory in sorted(runs):
        input_pins.extend(fingerprint(directory / name) for name in ("config.json", "val_logits.npz", "train_report.json"))
    pins = [fingerprint(protocol_path), fingerprint(spec_path), fingerprint(CURRENT_BASELINE),
            fingerprint(BASELINE), fingerprint(primary_path), *source_evidence(), *input_pins]
    stage = Path(tempfile.mkdtemp(prefix="stage_current_b119_", dir=parent))
    exclusive_json(stage / "comparison_request.json", {"started_at": now(), "protocol": fingerprint(protocol_path), "inputs": pins})
    module_spec = importlib.util.spec_from_file_location("rank_sibling_current_registered_comparator", COMPARATOR)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    baseline = ("B119G", ref(CURRENT_BASELINE), None)
    module.FOUNDATION = baseline
    emit(owner, state="RUNNING", activity="CPU additional fixed B119 secondary comparison")
    try:
        split = module.load_retrieval_split(module.H.SPLITS["val"])
        result, _ = module.replay_and_compare(split, stage, (CURRENT_CANDIDATE, ref(spec_path), None), baseline)
    finally:
        module.H.evaluate_probabilities = module._EP
        module.MS.evaluate_scores = module._ES
    for pin in pins:
        if fingerprint(path_of(pin["path"])) != pin:
            raise RuntimeError(f"Comparison input changed: {pin['path']}")
    exclusive_json(stage / "lodo_foundation_results_v1.json", result)
    receipt = {"status": "COMPLETE", "completed_at": now(), "candidate": CURRENT_CANDIDATE,
               "baseline": "B119G", "role": "additional_secondary_not_primary_replacement",
               "paired": result["paired"], "protocol": fingerprint(protocol_path),
               "original_primary_receipt": fingerprint(primary_path), "inputs": pins, "soup_validation": evidence,
               "limitations": "Same exposed three-source development surface; no independent confirmation, isolated objective causality, Gold probability or package acceptance."}
    exclusive_json(stage / "comparison_receipt.json", receipt)
    publish(stage, final)
    emit(owner, state="COMPARISON_COMPLETED", activity="additional B119 secondary atomically published")
    return receipt


def compare(run, *, owner=None):
    freeze_current_comparison(run)
    primary = compare_primary(run, owner=owner)
    if primary["status"] != "COMPLETE":
        return primary
    secondary = compare_current(run, owner=owner)
    # Do not rewrite the immutable primary receipt to add a later contrast.
    return {**primary, "status": secondary["status"], "current_b119_secondary": secondary}


def status(run):
    result = {"checked_at": now(), "folds": {f: {"ema_complete": completed(fold_plan(run, f)["new_ema"]),
                "soup_collected": (fold_plan(run, f)["output"] / "collection_receipt.json").exists()} for f in FOLDS},
            "gpu_admission_requirement": "Explicit root admission, >=10048 MiB free GPU and >=24 GiB MemAvailable; no new training"}
    status_path = Path(run) / "softadm_collection_v1/collector_status.json"
    if status_path.exists():
        result["collector"] = read_json(status_path)
        current = process_start_ticks(result["collector"]["pid"])
        result["owner_observation"] = ("matching_process_visible" if current is not None and current == result["collector"].get("process_start_ticks")
                                       else "not_matching_or_not_visible_in_this_process_namespace")
        operation = result["collector"].get("operation", {})
        if operation.get("pid") and operation.get("completed_at") is None:
            child = process_start_ticks(operation["pid"])
            result["child_observation"] = ("matching_child_process_visible" if child is not None and child == operation.get("process_start_ticks")
                                           else "not_matching_or_not_visible_in_this_process_namespace")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "build", "compare", "watch", "freeze-current"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "status":
        result = status(args.run)
    else:
        if args.mode == "build" and args.fold is None:
            parser.error("build requires --fold")
        if args.mode == "watch" and not args.allow_gpu:
            parser.error("watch requires --allow-gpu; status is always read-only")
        try:
            with CollectorOwner(args.run) as owner:
                if args.mode == "freeze-current":
                    spec, protocol = freeze_current_comparison(args.run)
                    result = {"status": "COMPLETE", "candidate_spec": fingerprint(spec), "protocol": fingerprint(protocol), "gpu": False}
                elif args.mode == "build":
                    try:
                        result = build_fold(args.run, args.fold, allow_gpu=args.allow_gpu, owner=owner)
                    except RuntimeError as exc:
                        if not str(exc).startswith("RESOURCE_WAIT"):
                            raise
                        result = {"status": "WAITING_FOR_RESOURCES", "fold": args.fold, "reason": str(exc)}
                        emit(owner, state="WAITING_FOR_RESOURCES", resource_waits=[result])
                elif args.mode == "compare":
                    result = compare(args.run, owner=owner)
                else:
                    while True:
                        resource_waits = []
                        for fold in FOLDS:
                            try:
                                print(json.dumps(build_fold(args.run, fold, allow_gpu=True, owner=owner)), flush=True)
                            except RuntimeError as exc:
                                if not str(exc).startswith("RESOURCE_WAIT"):
                                    raise
                                resource_waits.append({"fold": fold, "reason": str(exc), "observed_at": now()})
                                emit(owner, state="WAITING_FOR_RESOURCES", activity=str(exc), fold=fold)
                                print(str(exc), flush=True)
                        result = compare(args.run, owner=owner)
                        if result["status"] == "COMPLETE":
                            break
                        if resource_waits:
                            emit(owner, state="WAITING_FOR_RESOURCES", resource_waits=resource_waits,
                                 activity="Resource probes are not reservations; root coordinates queue admission")
                        time.sleep(60)
                emit(owner, state="COMPLETED" if result["status"] in ("COMPLETE", "COLLECTED", "ALREADY_COLLECTED") else "EXITED_WAITING",
                     process_completed_at=now(), result=result["status"])
        except CollectorBusy as exc:
            print(str(exc), file=sys.stderr, flush=True)
            raise SystemExit(2) from exc
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
