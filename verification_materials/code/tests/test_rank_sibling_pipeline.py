"""Tiny on-disk fixtures; no training, full models, or GPU inference."""
import copy
import json
import os
from pathlib import Path
import time
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file
from safetensors.numpy import load_file

from upar import rank_sibling_pipeline as p


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    sources = np.array(["PA100k", "PETA", "Market1501"] * 2)
    paths = np.array([f"{s}/{i}.jpg" for i, s in enumerate(sources)])
    names = np.array(["a", "b", "c"])
    np.savez(cache / "meta.npz", split=np.array(["train"] * 3 + ["val"] * 3),
             sources=sources, image_paths=paths, attribute_names=names)
    base_cfg = dict(fold="PA100k", include_val=False, seed=20261007, head="attrq",
                    backbone="pe_core_l14_280", cache_dir=str(cache), ema_used=.999,
                    ema_of="original_raw", rank_alpha=.1)

    def make(name, *, cfg=None, value=1):
        d = tmp_path / name
        d.mkdir(parents=True)
        config = dict(base_cfg, output_dir=str(d))
        config.update(cfg or {})
        counts = {s: 1 for s in ("PA100k", "PETA", "Market1501") if s != config["fold"]}
        write_json(d / "config.json", config)
        save_file({"head.base.bias": np.full(3, value, np.float32)}, d / "model.safetensors")
        np.savez(d / "val_logits.npz", logits_original=np.full((3, 3), value, np.float32),
                 logits_flip=np.full((3, 3), value, np.float32), pos_weight=np.ones(3, np.float32),
                 image_paths=paths[3:], attribute_names=names)
        write_json(d / "train_report.json", dict(status="TRAIN_DONE", fold=config["fold"],
                   include_val=config["include_val"], fit_rows_by_source=counts))
        before = time.time() - 10
        for path in d.iterdir():
            os.utime(path, (before, before))
        return d

    return tmp_path, make


def test_complete_requires_terminal_report_and_weights(fixture):
    _, make = fixture
    d = make("done")
    assert p.completed(d)
    (d / "train_report.json").write_text('{"status":')
    assert not p.completed(d)


def test_report_older_than_rewritten_weights_is_not_completion(fixture):
    _, make = fixture
    d = make("rewriting")
    before = time.time() - 5
    os.utime(d / "model.safetensors", (before, before))
    assert not p.completed(d)


def test_valid_run_proves_rows_fit_counts_ema_and_hashes(fixture):
    _, make = fixture
    d = make("ema")
    result = p.validate_run(d, "PA100k", deep_weights=True, ema=True)
    assert result["fit_rows_by_source"] == {"PETA": 1, "Market1501": 1}
    assert len(result["weights"]["sha256"]) == 64
    assert result["training_launch_evidence"] == "legacy_not_recorded"


@pytest.mark.parametrize("change", [{"include_val": True}, {"fold": "all"}, {"seed": 9}, {"ema_used": None}])
def test_invalid_fit_seed_or_raw_weights_rejected(fixture, change):
    _, make = fixture
    d = make("bad", cfg=change)
    with pytest.raises(ValueError):
        p.validate_run(d, "PA100k", ema=True)


def test_npz_id_order_corruption_detected(fixture):
    _, make = fixture
    d = make("wrongrows")
    with np.load(d / "val_logits.npz") as z:
        values = {k: z[k] for k in z.files}
    values["image_paths"] = values["image_paths"][::-1]
    np.savez(d / "val_logits.npz", **values)
    with pytest.raises(ValueError, match="row/attribute"):
        p.validate_run(d, "PA100k")


def test_fit_report_cannot_omit_actual_training_sources(fixture):
    _, make = fixture
    d = make("badcounts")
    r = json.loads((d / "train_report.json").read_text())
    r["fit_rows_by_source"] = {}
    write_json(d / "train_report.json", r)
    with pytest.raises(ValueError, match="fit counts"):
        p.validate_run(d, "PA100k")


def test_atomic_publish_keeps_partial_out_invisible_and_refuses_overwrite(tmp_path):
    stage, final = tmp_path / "stage", tmp_path / "final"
    stage.mkdir()
    (stage / "complete.json").write_text("{}")
    assert not final.exists()
    p.publish(stage, final)
    assert (final / "complete.json").exists() and not stage.exists()
    stage.mkdir()
    with pytest.raises(FileExistsError):
        p.publish(stage, final)


def install_baseline(fixture, monkeypatch):
    root, make = fixture
    attr = make("attr_PA100k")
    old_rank = make("pLq_rankq_PA100k_ema0.999")
    raw_mask = make("mask_raw_PA100k", value=3)
    mask = make("mask_aligned_PA100k", cfg={"pw_aligned_of": str(raw_mask), "pw_ref": str(attr)})
    part = make("part_PA100k", value=4)
    old_soup = make("old_soup", cfg={"soup_of": list(map(str, (attr, old_rank, mask, part))), "soup_weights": [.25] * 4})
    base = {"candidate": "B116G", "package": "not_a_new_package.zip", "lodo": {
        "members": [{"weight": .4, "runs": {f: [str(old_soup)] for f in p.FOLDS}},
                    {"weight": .35, "runs": {f: ["h_" + f] for f in p.FOLDS}},
                    {"weight": .25, "runs": {f: ["e_" + f] for f in p.FOLDS}}], "post": p.POST}}
    baseline = root / "baseline.json"
    write_json(baseline, base)
    monkeypatch.setattr(p, "BASELINE", baseline)
    run = root / "run"
    return run, (attr, old_rank, mask, part), base


def test_fixed_plan_replaces_only_rank_sibling_and_uses_weight_soup(fixture, monkeypatch):
    run, parents, base = install_baseline(fixture, monkeypatch)
    plan = p.fold_plan(run, "PA100k")
    assert plan["parents"] == list(parents)
    assert plan["weights"] == [.25] * 4
    assert plan["new_ema"].name == "pe_softadm_PA100k_ema0.999"
    spec = p.candidate_spec(run)
    assert "package" not in spec
    assert spec["lodo"]["members"][1:] == base["lodo"]["members"][1:]
    assert [m["weight"] for m in spec["lodo"]["members"]] == [.4, .35, .25]
    assert all(len(v) == 1 for v in spec["lodo"]["members"][0]["runs"].values())
    with pytest.raises(PermissionError):
        p.build_fold(run, "PA100k")


def test_cpu_mock_executes_real_soup_cli_and_publishes_complete_receipt(fixture, monkeypatch):
    root, make = fixture
    run, _, _ = install_baseline(fixture, monkeypatch)
    make("run/pe_softadm_PA100k_ema0.999", value=10, cfg={"rank_loss_kind": "soft_adm",
         "rank_scope": "batch", "rank_alpha": .5, "rank_adm_temperature": 1.0})
    monkeypatch.setattr(p, "check_resources", lambda: {"synthetic_cpu_fixture": True})
    monkeypatch.setattr(p, "source_evidence", lambda: [{"test": "mock, not GPU quality evidence"}])
    commands = []

    def fake_subprocess(command, **kwargs):
        commands.append(command)
        if command[1].endswith("make_pw_aligned.py"):
            src, reference, out = map(Path, command[2:])
            cfg = json.loads((src / "config.json").read_text())
            cfg.update(pw_aligned_of=str(src), pw_ref=str(reference), output_dir=str(out))
            state = load_file(src / "model.safetensors")
            with np.load(src / "val_logits.npz") as z:
                values = {k: z[k] for k in z.files}
            report = json.loads((src / "train_report.json").read_text())
        else:
            assert command[1:4] == ["-m", "upar.h1_finetune", "soup"]
            runs = [p.path_of(x) for x in command[command.index("--runs") + 1:command.index("--output")]]
            out = Path(command[-1])
            states = [load_file(d / "model.safetensors") for d in runs]
            state = {k: sum(s[k] * .25 for s in states) for k in states[0]}
            cfg = json.loads((runs[0] / "config.json").read_text())
            cfg.update(soup_of=[p.ref(d) for d in runs], soup_weights=[.25] * 4)
            with np.load(runs[0] / "val_logits.npz") as z:
                values = {k: z[k] for k in z.files}
            # A toy bias-only forward of the averaged STATE, not a probability blend.
            for key in ("logits_original", "logits_flip"):
                values[key] = np.repeat(state["head.base.bias"][None, :], 3, axis=0)
            report = dict(status="SOUP_DONE", fold="PA100k", include_val=False)
        write_json(out / "config.json", cfg)
        save_file(state, out / "model.safetensors")
        np.savez(out / "val_logits.npz", **values)
        write_json(out / "train_report.json", report)
        class Finished:
            pid = 123456789

            def wait(self):
                return 0
        return Finished()

    monkeypatch.setattr(p.subprocess, "Popen", fake_subprocess)
    result = p.build_fold(run, "PA100k", allow_gpu=True)
    assert result["status"] == "COLLECTED" and len(commands) == 2
    output = p.fold_plan(run, "PA100k")["output"]
    assert np.array_equal(load_file(output / "model.safetensors")["head.base.bias"], np.full(3, 4.5))
    receipt = json.loads((output / "collection_receipt.json").read_text())
    assert receipt["status"] == "COMPLETE"
    assert "stage_soup" not in receipt["validation"]["weights"]["path"]
    assert p.build_fold(run, "PA100k", allow_gpu=True)["status"] == "ALREADY_COLLECTED"
    assert len(commands) == 2


def test_relocated_manifest_keeps_hashes_but_uses_published_path(tmp_path):
    stage, final = tmp_path / "stage", tmp_path / "final"
    value = {"run": str(stage), "files": [{"path": str(stage / "x"), "sha256": "abc"}]}
    actual = p.relocated(value, stage, final)
    assert actual == {"run": str(final), "files": [{"path": str(final / "x"), "sha256": "abc"}]}


def test_incomplete_npz_is_rejected_not_marked_success(fixture):
    _, make = fixture
    d = make("incomplete")
    (d / "val_logits.npz").write_bytes(b"PKbroken")
    with pytest.raises(Exception):
        p.validate_run(d, "PA100k")


def test_single_owner_rejects_second_collector_without_overwriting_status(tmp_path):
    with p.CollectorOwner(tmp_path) as first:
        path = tmp_path / "softadm_collection_v1/collector_status.json"
        saved = path.read_bytes()
        with pytest.raises(p.CollectorBusy, match="nothing dispatched"):
            with p.CollectorOwner(tmp_path):
                pytest.fail("Second owner was admitted")
        assert path.read_bytes() == saved
        first.update(state="WAITING_FOR_RESOURCES", resource_probe={"free_gpu_mib": 1})
        value = json.loads(path.read_text())
        assert value["pid"] == os.getpid() and value["resource_probe_is_reservation"] is False
        assert value["state"] == "WAITING_FOR_RESOURCES"
        assert not list(path.parent.glob("collector_status.json.tmp.*"))
    with p.CollectorOwner(tmp_path) as second:
        assert second.value["owner_token"] != first.value["owner_token"]


def test_live_child_retains_lock_after_parent_owner_closes(tmp_path):
    read_fd, write_fd = os.pipe()
    child = None
    try:
        with p.CollectorOwner(tmp_path) as owner:
            child = subprocess.Popen([sys.executable, "-c", f"import os; os.read({read_fd}, 1)"],
                                     pass_fds=(owner.lock_fd, read_fd))
        with pytest.raises(p.CollectorBusy):
            with p.CollectorOwner(tmp_path):
                pytest.fail("Live orphan's lock was released")
        os.write(write_fd, b"x")
        child.wait(timeout=5)
        with p.CollectorOwner(tmp_path):
            pass
    finally:
        os.close(read_fd)
        os.close(write_fd)
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_child_status_records_real_pid_start_exit_and_completion(tmp_path):
    with p.CollectorOwner(tmp_path) as owner:
        with (tmp_path / "cpu.log").open("x") as log:
            p.run_child([sys.executable, "-c", "pass"], log, owner, kind="tiny CPU test", fold="PA100k")
        operation = owner.value["operation"]
        assert operation["pid"] > 0
        assert operation["started_at"] <= operation["completed_at"]
        assert operation["exit_code"] == 0 and owner.value["state"] == "OPERATION_COMPLETED"
        assert owner.value["resource_probe_is_reservation"] is False


def install_current_baseline(fixture, monkeypatch):
    run, parents, original = install_baseline(fixture, monkeypatch)
    current = copy.deepcopy(original)
    current["candidate"] = "B119G"
    current["lodo"]["members"][1]["runs"] = {f: [f"hp_q_qr_qrq_s_{f}"] for f in p.FOLDS}
    path = fixture[0] / "current.json"
    write_json(path, current)
    monkeypatch.setattr(p, "CURRENT_BASELINE", path)
    return run, parents, original, current


def test_current_spec_is_fixed_secondary_only_and_preserves_original(fixture, monkeypatch):
    run, _, original, current = install_current_baseline(fixture, monkeypatch)
    before = p.BASELINE.read_bytes()
    candidate = p.current_candidate_spec(run)
    assert candidate["candidate"] == p.CURRENT_CANDIDATE and "package" not in candidate
    assert candidate["lodo"]["members"][1:] == current["lodo"]["members"][1:]
    assert candidate["lodo"]["members"][0] == p.candidate_spec(run)["lodo"]["members"][0]
    assert current["lodo"]["members"][0] == original["lodo"]["members"][0]
    assert p.BASELINE.read_bytes() == before
    spec, protocol = p.freeze_current_comparison(run)
    frozen = (spec.read_bytes(), protocol.read_bytes())
    assert p.freeze_current_comparison(run) == (spec, protocol)
    assert (spec.read_bytes(), protocol.read_bytes()) == frozen
    value = p.read_json(protocol)
    assert value["role"] == "additional_secondary_not_primary_replacement"
    assert not (run / "softadm_collection_v1/comparison").exists()


@pytest.mark.parametrize("change", ["PE", "EVA", "H4", "weight", "post", "missingfold"])
def test_current_spec_rejects_changed_controls(fixture, monkeypatch, change):
    run, _, _, current = install_current_baseline(fixture, monkeypatch)
    members = current["lodo"]["members"]
    if change in ("PE", "EVA", "H4"):
        members[{"PE": 0, "H4": 1, "EVA": 2}[change]]["runs"]["PA100k"] = ["changed"]
    elif change == "weight":
        members[0]["weight"] = .41
    elif change == "post":
        current["lodo"]["post"]["clip"] = 2e-5
    else:
        members[0]["runs"].pop("PETA")
    write_json(p.CURRENT_BASELINE, current)
    with pytest.raises(ValueError):
        p.current_candidate_spec(run)


def test_current_protocol_refuses_changed_original_spec_bytes(fixture, monkeypatch):
    run, _, _, _ = install_current_baseline(fixture, monkeypatch)
    _, protocol = p.freeze_current_comparison(run)
    frozen = protocol.read_bytes()
    p.CURRENT_BASELINE.write_text(p.CURRENT_BASELINE.read_text() + "\n")
    with pytest.raises(ValueError, match="protocol"):
        p.freeze_current_comparison(run)
    assert protocol.read_bytes() == frozen


def test_primary_complete_dispatches_only_additional_comparison(fixture, monkeypatch):
    run, _, _, _ = install_current_baseline(fixture, monkeypatch)
    parent = run / "softadm_collection_v1/comparison"
    parent.mkdir(parents=True)
    path = parent / "comparison_receipt.json"
    write_json(path, {"status": "COMPLETE", "baseline": "B116G", "single_vs_batch_ce": {"fixed": True}})
    original = path.read_bytes()
    calls = []
    monkeypatch.setattr(p, "compare_current", lambda *a, **k: calls.append("current") or {"status": "COMPLETE", "baseline": "B119G"})
    result = p.compare(run)
    assert calls == ["current"] and result["baseline"] == "B116G"
    assert result["single_vs_batch_ce"] == {"fixed": True}
    assert result["current_b119_secondary"]["baseline"] == "B119G"
    assert path.read_bytes() == original


def test_secondary_does_not_blind_retry_partial(fixture, monkeypatch):
    run, _, _, _ = install_current_baseline(fixture, monkeypatch)
    p.freeze_current_comparison(run)
    stage = run / "softadm_collection_v1/stage_current_b119_interrupted"
    stage.mkdir()
    with pytest.raises(RuntimeError, match="do not retry"):
        p.compare_current(run)
    assert stage.is_dir()


def test_secondary_mock_scores_once_keeps_primary_and_restores_wrappers(fixture, monkeypatch):
    root, make = fixture
    run, _, _, current = install_current_baseline(fixture, monkeypatch)
    parent = run / "softadm_collection_v1"
    primary = parent / "comparison/comparison_receipt.json"
    primary.parent.mkdir(parents=True)
    write_json(primary, {"status": "COMPLETE", "baseline": "B116G", "paired": {"old": True}})
    original = primary.read_bytes()
    for fold in p.FOLDS:
        directory = p.fold_plan(run, fold)["output"]
        make(str(directory.relative_to(root)), cfg={"fold": fold})
        write_json(directory / "collection_receipt.json", {"status": "COMPLETE"})
        for member in current["lodo"]["members"][1:]:
            make(member["runs"][fold][0], cfg={"fold": fold})
    monkeypatch.setattr(p, "validate_run", lambda run, fold: {"run": str(run), "fold": fold, "toy": True})
    monkeypatch.setattr(p, "source_evidence", lambda: [])
    calls = []
    module = SimpleNamespace(H=SimpleNamespace(SPLITS={"val": "toy"}, evaluate_probabilities="wrapped"),
                             MS=SimpleNamespace(evaluate_scores="wrapped"), _EP="original_ep", _ES="original_es",
                             load_retrieval_split=lambda path: path)

    def replay(split, out, candidate, baseline):
        assert module.FOUNDATION == baseline and baseline[0] == "B119G"
        assert candidate[0] == p.CURRENT_CANDIDATE
        calls.append((candidate, baseline))
        np.savez(out / "lodo_per_query_v1.npz", mock=np.array([1]))
        return {"paired": {"toy": True}}, None

    module.replay_and_compare = replay
    monkeypatch.setattr(p.importlib.util, "spec_from_file_location", lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(p.importlib.util, "module_from_spec", lambda s: module)
    result = p.compare_current(run)
    assert result["role"] == "additional_secondary_not_primary_replacement"
    assert len(calls) == 1 and primary.read_bytes() == original
    assert module.H.evaluate_probabilities == "original_ep" and module.MS.evaluate_scores == "original_es"
    assert (parent / "current_b119/lodo_per_query_v1.npz").is_file()
    assert not list(parent.glob("stage_current_b119_*"))
    assert p.compare_current(run) == result and len(calls) == 1
