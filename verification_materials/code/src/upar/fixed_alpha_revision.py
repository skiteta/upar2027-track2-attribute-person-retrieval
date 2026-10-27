"""Explicit fixed-alpha package revisions; no old verifier/global overrides.

Historical parent acceptance is evidence about its actual past execution, not
permission to re-certify changed live trainer sources. Every new package needs
its own complete capture, actual child exit and independent numeric reference.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
import zipfile

import numpy as np

from . import fixed_revision_replay, gallery_feature_replay
from .gallery_feature_package import pin, sha, write_new, replace_once
from .numpy_blas_scope import numpy_blas_threads


def verify_refs(refs):
    for ref in refs:
        if pin(ref["path"]) != ref:
            raise ValueError("changed pinned input: " + ref["path"])


def archive_members(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("duplicate ZIP member")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts or name.endswith("/"):
            raise ValueError("unsafe ZIP member")
    return names


def independent_smooth(base, probabilities, neighbours, counts, *, alpha, constant):
    """Independent FP32 row arithmetic, never importing the runtime smoother.

    Preserve original full degree, edge positions and FP32 reduction strides.
    Same-bit and one-bit corrections both read the original base, not each other.
    """
    if (type(alpha) is not float or not math.isfinite(alpha) or not 0 <= alpha <= 1
            or type(constant) is not float or not math.isfinite(constant) or not 0 <= constant <= 1):
        raise ValueError("explicit bounded finite float alpha and constant required")
    if base.dtype != np.float32 or base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("finite FP32 base required")
    if (probabilities.shape != (base.shape[1], 40) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or neighbours.ndim != 2 or neighbours.shape[0] != base.shape[1]
            or counts.shape != (base.shape[1],) or not np.issubdtype(counts.dtype, np.integer)
            or not np.issubdtype(neighbours.dtype, np.integer)
            or np.any((counts < 0) | (counts > neighbours.shape[1]))):
        raise ValueError("aligned probabilities and integer graph required")
    if alpha == 0:
        return base.copy()
    if len(base) > 64:
        return np.concatenate([independent_smooth(base[s:s+64], probabilities, neighbours, counts,
                               alpha=alpha, constant=constant) for s in range(0, len(base), 64)])
    out = base.copy()
    bits, width = probabilities >= .5, neighbours.shape[1]
    for row, degree in enumerate(counts):
        if not degree:
            continue
        selected = neighbours[row, :degree]
        hamming = np.count_nonzero(bits[selected] != bits[row], axis=1)
        same = selected[hamming == 0]
        if len(same):
            values = np.zeros((width, len(base)), dtype=base.dtype).T
            values[:, :len(same)] = base[:, same]
            mean = values.sum(axis=1, dtype=base.dtype) / np.asarray(len(same), base.dtype)
            mass = np.asarray(alpha, base.dtype) * (np.asarray(len(same), base.dtype) / np.asarray(degree, base.dtype))
            out[:, row] = base[:, row] + mass * (mean-base[:, row])
        if np.any(hamming == 1):
            delta = np.zeros((width, len(base)), dtype=base.dtype).T
            weights = (hamming == 1).astype(base.dtype) * np.asarray(constant, base.dtype)
            delta[:, :degree] = (base[:, selected]-base[:, row, None]) * weights[None, :]
            out[:, row] += (np.asarray(alpha, base.dtype) / np.asarray(degree, base.dtype)) * delta.sum(axis=1, dtype=base.dtype)
    return out


def validate_exit(receipt, status, request, *, candidate, stage, report, adapter_path):
    """Root supplies actual tool/supervisor and direct child evidence separately."""
    if receipt.get("schema") != "fixed_alpha_revision_direct_exit_v1":
        raise ValueError("new revision actual-exit schema required")
    for key in ("actual_exit_code", "supervisor_actual_tool_exit_code", "child_actual_returncode"):
        if type(receipt.get(key)) is not int or receipt[key] != 0:
            raise ValueError("actual zero integer exits required")
    for key in ("exec_session", "supervisor_pid", "child_pid"):
        if type(receipt.get(key)) is not int or receipt[key] <= 0:
            raise ValueError("actual positive integer session and PIDs required")
    argv = request.get("argv", [])
    root = Path(request.get("root", "/__missing_root__"))
    script_positions = [i for i, x in enumerate(argv) if isinstance(x, str) and (root / x).resolve() == adapter_path]
    script_present = len(script_positions) == 1 and script_positions[0]+1 < len(argv) and argv[script_positions[0]+1] == stage
    if argv.count("--output") != 1 or argv.index("--output")+1 >= len(argv):
        raise ValueError("exact output argument required")
    output = (root / argv[argv.index("--output")+1]).resolve()
    if (receipt.get("candidate") != candidate or receipt.get("stage") != stage
            or not isinstance(receipt.get("actual_tool_chunk"), str) or not receipt["actual_tool_chunk"]
            or receipt["supervisor_pid"] == receipt["child_pid"]
            or status.get("state") != "finished" or status.get("child_pid") != receipt["child_pid"]
            or type(status.get("child_pid")) is not int or type(status.get("supervisor_pid")) is not int
            or type(status.get("returncode")) is not int or status["returncode"] != 0
            or status.get("argv") != argv or not script_present or stage not in argv
            or "--root-reviewed" not in argv or (stage == "replay" and "--allow-gpu" not in argv)
            or output != Path(receipt["report"]["path"]).resolve().parent
            or status.get("candidate") != candidate or request.get("candidate") != candidate
            or status.get("supervisor_pid") != receipt["supervisor_pid"]
            or status.get("request_sha256") != receipt["request"]["sha256"]
            or (root / request.get("status_path", "")).resolve() != Path(receipt["status"]["path"]).resolve()
            or report.get("candidate") != candidate or report.get("pid") != receipt["child_pid"]
            or type(report.get("pid")) is not int
            or not request.get("log_path") or status.get("log_path") != request["log_path"]):
        raise ValueError("new candidate/command/report/PID binding differs")


class FixedAlphaRevision:
    def __init__(self, spec_path, adapter_path):
        self.spec_path, self.adapter_path = Path(spec_path).resolve(), Path(adapter_path).resolve()
        self.here = self.spec_path.parent
        self.root = self.here.parents[2]
        self.s = json.loads(self.spec_path.read_text())
        if self.s.get("schema") != "fixed_alpha_revision_spec_v1":
            raise ValueError("explicit versioned recipe required")
        for key in ("alpha", "parent_alpha", "constant"):
            value = self.s[key]
            if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid fixed recipe value")
        if self.s["candidate"] == self.s["parent_candidate"] or self.s["alpha"] == self.s["parent_alpha"]:
            raise ValueError("distinct fixed revision required")

    def path(self, value):
        return (self.root / value).resolve()

    def fixed_refs(self, key):
        refs = []
        for relative, expected in self.s[key].items():
            p = self.path(relative)
            if sha(p) != expected:
                raise ValueError("changed frozen dependency: " + relative)
            refs.append(pin(p))
        return refs

    def private_output(self, output):
        output = Path(output).resolve()
        if not output.is_relative_to(self.here) or output == self.here or output.exists():
            raise ValueError("fresh private output only; never retry or overwrite")
        return output

    def parent_config(self):
        path = self.path(self.s["parent_config"])
        if sha(path) != self.s["fixed_inputs"][self.s["parent_config"]]:
            raise ValueError("changed parent config")
        return json.loads(path.read_text())

    def patch_config(self, config):
        if (config != self.parent_config() or config.get("candidate") != self.s["parent_candidate"]
                or config["gallery_rerank"]["alpha"] != self.s["parent_alpha"]
                or config["constant_rescue_diffusion_alpha"] != self.s["parent_alpha"]
                or config["onebit_rescue_constant"] != self.s["constant"]):
            raise ValueError("exact parent recipe and production constant required")
        result = copy.deepcopy(config)
        result["candidate"] = self.s["candidate"]
        result["gallery_rerank"]["alpha"] = self.s["alpha"]
        result["constant_rescue_diffusion_alpha"] = self.s["alpha"]
        return result

    def patch_runtime(self, source):
        if hashlib.sha256(source.encode()).hexdigest() != self.s["parent_runtime_sha256"]:
            raise ValueError("exact parent runtime bytes required")
        def literal(value):
            text = str(value)
            return text[1:] if text.startswith("0.") else text
        old, new = literal(self.s["parent_alpha"]), literal(self.s["alpha"])
        result = replace_once(source, "rescue, alpha="+old+")", "rescue, alpha="+new+")")
        result = replace_once(result, "diffusion_alpha="+old+")", "diffusion_alpha="+new+")")
        # Only the designated function body may change; imports and other AST nodes are equal.
        a, b = ast.parse(source), ast.parse(result)
        if len(a.body) != len(b.body):
            raise ValueError("changed runtime structure")
        changed = [getattr(x, "name", None) for x, y in zip(a.body, b.body) if ast.dump(x) != ast.dump(y)]
        if changed != ["apply_feature_rerank"]:
            raise ValueError("unrelated runtime change")
        return result

    def inputs(self):
        refs = self.fixed_refs("fixed_inputs") + self.fixed_refs("source_inputs")
        refs += [pin(self.spec_path), pin(self.adapter_path), pin(Path(__file__))]
        # Historical acceptance is checked as immutable actual past evidence.
        # Do not call its old recursive live-source verifier with changed H1.
        history = json.loads(self.path(self.s["historical_release"]).read_text())
        accepted = json.loads(self.path(self.s["historical_acceptance"]).read_text())
        official = json.loads(self.path(self.s["official_receipt"]).read_text())
        if (history.get("candidate") != self.s["parent_candidate"]
                or type(history["release_gate"].get("actual_exit_code")) is not int
                or history["release_gate"]["actual_exit_code"] != 0
                or history["release_gate"]["acceptance"] != pin(self.path(self.s["historical_acceptance"]))
                or accepted.get("status") != "ACCEPTED_RUNTIME_PREFLIGHT"
                or accepted.get("candidate") != self.s["parent_candidate"]
                or accepted["artifact"]["sha256"] != self.s["parent_zip_sha256"]
                or official["artifact"]["sha256"] != self.s["parent_zip_sha256"]
                or official["submission"]["id"] != self.s["parent_submission_id"]
                or official["submission"]["status"] != "Finished"):
            raise ValueError("exact historical parent acceptance/official identity required")
        return refs

    def build(self, output):
        refs, output = self.inputs(), self.private_output(output)
        parent = self.path(self.s["parent_zip"])
        if sha(parent) != self.s["parent_zip_sha256"]:
            raise ValueError("exact parent ZIP required")
        refs.append(pin(parent))
        if shutil.disk_usage(self.here).free < 3 * parent.stat().st_size + 20 * 2**30:
            raise RuntimeError("insufficient preserved-build disk reserve")
        output.mkdir(parents=True)
        package = output / "package"
        package.mkdir()
        started = time.monotonic()
        try:
            with zipfile.ZipFile(parent) as old:
                names = archive_members(old)
                for name in names:
                    dest = package / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open("xb") as target:
                        if name == "b121_runtime.py":
                            target.write(self.patch_runtime(old.read(name).decode()).encode())
                        elif name == "assets/config.json":
                            target.write((json.dumps(self.patch_config(json.loads(old.read(name))), indent=2)+"\n").encode())
                        else:
                            with old.open(name) as source:
                                shutil.copyfileobj(source, target, 8*2**20)
                artifact = output / (self.s["candidate"]+".zip")
                with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as new:
                    for p in sorted(package.rglob("*")):
                        if p.is_file():
                            new.write(p, p.relative_to(package).as_posix())
                proof = []
                with zipfile.ZipFile(artifact) as new:
                    if set(archive_members(new)) != set(names):
                        raise ValueError("member set changed")
                    for name in names:
                        with old.open(name) as a, new.open(name) as b:
                            before = hashlib.file_digest(a, "sha256").hexdigest()
                            after = hashlib.file_digest(b, "sha256").hexdigest()
                        changed = name in ("b121_runtime.py", "assets/config.json")
                        if (before != after) != changed:
                            raise ValueError("unexpected member delta: " + name)
                        proof.append({"member": name, "parent_sha256": before, "new_sha256": after, "changed": changed})
            verify_refs(refs)
            report = {"status": "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED", "candidate": self.s["candidate"],
                "pid": os.getpid(), "alpha": self.s["alpha"], "constant": self.s["constant"],
                "zip": pin(artifact), "parent": pin(parent), "inputs": refs, "member_proof": proof,
                "config": self.patch_config(self.parent_config()), "new_fit_updates": 0,
                "new_encoder_forward": 0, "wall_seconds": time.monotonic()-started,
                "parent_acceptance_scope": "historical actual release only; old live trainer not re-certified",
                "requires_new_full_capture_and_reference_actual_exits": True}
            write_new(output / "complete.json", report)
            return report
        except BaseException as error:
            self.failure(output, error)
            raise

    @staticmethod
    def failure(output, error):
        write_new(output / "failure.json", {"type": type(error).__name__, "error": str(error),
                  "partial_preserved": True, "automatic_retry": False})

    def checked_package(self, report_path):
        path = Path(report_path).resolve()
        if not path.is_relative_to(self.here):
            raise ValueError("new revision package only")
        report = json.loads(path.read_text())
        if (report.get("status") != "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED"
                or report.get("candidate") != self.s["candidate"]
                or report.get("config") != self.patch_config(self.parent_config())
                or report.get("alpha") != self.s["alpha"] or report.get("constant") != self.s["constant"]
                or Path(report["zip"]["path"]).resolve() != path.parent / (self.s["candidate"]+".zip")
                or report["parent"]["sha256"] != self.s["parent_zip_sha256"]
                or sorted(r["member"] for r in report["member_proof"] if r["changed"]) != ["assets/config.json", "b121_runtime.py"]):
            raise ValueError("exact new build required")
        verify_refs(report["inputs"]+[report["zip"], report["parent"]])
        return report

    def actual_exit(self, receipt_path, stage, report_path):
        receipt_path, report_path = Path(receipt_path).resolve(), Path(report_path).resolve()
        if not receipt_path.is_relative_to(self.here) or not report_path.is_relative_to(self.here):
            raise ValueError("new revision actual exit only")
        receipt = json.loads(receipt_path.read_text())
        verify_refs([receipt[k] for k in ("request", "status", "report")])
        if receipt["report"] != pin(report_path):
            raise ValueError("actual exit report differs")
        status, request, report = [json.loads(Path(receipt[k]["path"]).read_text()) for k in ("status", "request", "report")]
        validate_exit(receipt, status, request, candidate=self.s["candidate"], stage=stage,
                      report=report, adapter_path=self.adapter_path)
        return [pin(receipt_path)]+[receipt[k] for k in ("request", "status", "report")]

    def replay(self, package_report, build_exit, output, time_scale):
        output = self.private_output(output)
        package = self.checked_package(package_report)
        refs = self.actual_exit(build_exit, "build", package_report)
        result = fixed_revision_replay.execute(package["zip"]["path"], Path(package_report).parent / "package",
            output, package["zip"]["sha256"], package["config"], time_scale=time_scale)
        verify_refs(refs)
        write_new(output / "revision_adapter_receipt.json", {"status": "NEW_ZIP_CAPTURED_NOT_ACCEPTED",
            "candidate": self.s["candidate"], "pid": os.getpid(), "adapter": pin(self.adapter_path),
            "spec": pin(self.spec_path), "package_report": pin(package_report), "build_exit_inputs": refs,
            "capture_report": pin(output / "execute_report.json"), "artifact": package["zip"]})
        return result

    def reference(self, capture_dir, replay_exit, output):
        capture_dir, output = Path(capture_dir).resolve(), self.private_output(output)
        if not capture_dir.is_relative_to(self.here):
            raise ValueError("new capture only")
        adapter_path = capture_dir / "revision_adapter_receipt.json"
        refs = self.actual_exit(replay_exit, "replay", adapter_path)
        adapter = json.loads(adapter_path.read_text())
        report_path = capture_dir / "execute_report.json"
        report = json.loads(report_path.read_text())
        package = self.checked_package(adapter["package_report"]["path"])
        if (adapter.get("status") != "NEW_ZIP_CAPTURED_NOT_ACCEPTED" or adapter.get("spec") != pin(self.spec_path)
                or adapter.get("adapter") != pin(self.adapter_path) or adapter["capture_report"] != pin(report_path)
                or adapter["artifact"] != package["zip"] or report.get("artifact") != package["zip"]
                or report.get("candidate") != self.s["candidate"] or report.get("config") != package["config"]
                or report.get("device") != "cuda" or report.get("forced_plan") is not None
                or report.get("status") != "RUNTIME_OUTPUT_CAPTURED_REFERENCE_PENDING"
                or not math.isfinite(report.get("time_scale", 0)) or not 1 <= report["time_scale"] <= 4
                or (report.get("rows"), report.get("queries")) != (33407, 3462)):
            raise ValueError("exact complete new CUDA capture required")
        if len(report["outputs"]) != 3 or {r["path"] for r in report["outputs"]} != {str(capture_dir / n) for n in ("capture.npz", "features.npy", "distances.npy")}:
            raise ValueError("exact three output arrays required")
        refs += self.inputs()+self.fixed_refs("reference_inputs")+package["inputs"]+adapter["build_exit_inputs"]
        refs += report["sources"]+report["annotations"]+report["outputs"]+[package["zip"], pin(report_path), adapter["package_report"]]
        verify_refs(refs)
        if sha(capture_dir / "features.npy") != self.s["reference_inputs"][self.s["reference_features"]]:
            raise ValueError("graph reuse requires bit-identical feature file, no tolerance")
        with np.load(capture_dir / "capture.npz", allow_pickle=False) as new, np.load(self.path(self.s["reference_capture"]), allow_pickle=False) as old:
            fields = {"probabilities", "queries", "image_paths", "attribute_names"}
            if set(new.files) != fields or set(old.files) != fields:
                raise ValueError("capture schema differs")
            for field in fields:
                if new[field].dtype != old[field].dtype or new[field].shape != old[field].shape or new[field].tobytes() != old[field].tobytes():
                    raise ValueError("exact input identity differs: " + field)
            p, q, names = new["probabilities"], new["queries"], new["attribute_names"]
        with np.load(self.path(self.s["reference_graph"]), allow_pickle=False) as graph:
            n, c = graph["neighbours"], graph["counts"]
        actual = np.load(capture_dir / "distances.npy", mmap_mode="r", allow_pickle=False)
        if actual.dtype != np.float32 or actual.shape != (3462, 33407):
            raise ValueError("complete FP32 distances required")
        output.mkdir(parents=True)
        write_new(output / "started.json", {"candidate": self.s["candidate"], "inputs": refs})
        started, cpu = time.monotonic(), time.process_time()
        try:
            saved = np.lib.format.open_memmap(output / "independent_distances.npy", mode="w+", dtype=np.float32, shape=actual.shape)
            unequal, rank_diff, maximum = 0, 0, 0.
            with numpy_blas_threads(2) as blas:
                for start in range(0, len(q), 64):
                    end = min(start+64, len(q))
                    base = gallery_feature_replay.independent_base(p, q[start:end], names, package["config"])
                    expected = independent_smooth(base, p, n, c, alpha=self.s["alpha"], constant=self.s["constant"])
                    observed = np.asarray(actual[start:end])
                    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
                        raise ValueError("nonfinite distance")
                    saved[start:end] = expected
                    unequal += int(np.count_nonzero(expected.view(np.uint32) != observed.view(np.uint32)))
                    rank_diff += sum(not np.array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable")) for a, b in zip(expected, observed))
                    maximum = max(maximum, float(np.abs(observed-expected).max()))
                    print(json.dumps({"queries": end, "total": len(q), "unequal": unequal, "rank_diff": rank_diff}), flush=True)
            saved.flush()
            verify_refs(refs)
            result = {"status": "PASS" if unequal == rank_diff == 0 else "FAIL", "candidate": self.s["candidate"],
                "kind": "fixed_alpha_new_zip_same_input_independent_reference_v1", "pid": os.getpid(),
                "alpha": self.s["alpha"], "constant": self.s["constant"], "inputs": refs,
                "reference": pin(output / "independent_distances.npy"), "unequal_distance_values": unequal,
                "queries_with_different_ranks": rank_diff, "distance_max_abs": maximum,
                "wall_seconds": time.monotonic()-started, "CPU_seconds": time.process_time()-cpu,
                "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "numpy_blas": blas,
                "GPU_seconds": 0, "quality_or_gold_claim": False, "official_A10_runtime_guaranteed": False,
                "graph_algorithm_independently_reimplemented": False, "actual_process_exit_confirmed": False}
            write_new(output / "report.json", result)
            if result["status"] != "PASS":
                raise ValueError("bit-exact independent reference failed; no tolerance waiver")
            return result
        except BaseException as error:
            self.failure(output, error)
            raise
