"""Full public CUDA replay of B128C_QUADJ_HYBRID_COEVOLVE_ANCHOR.zip (package_v2).

Uses upar.fixed_revision_replay.execute on the exact new ZIP, its extracted package,
and the built config with time_scale=4.0 on the local GPU, then saves the captured
same-forward EVA02-L/14 features and 3-member std for independent reference verification.
usage: replay_b127c.py OUTPUT_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from upar import fixed_revision_replay
from upar.gallery_feature_package import pin, write_new


def main() -> None:
    report = json.loads((HERE / "package_v14/complete.json").read_text())
    assert (
        report["status"] == "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED"
        and report["candidate"] == "B139C_ANAT_MULTIRES_SPHERICAL_HEAD_RERANK"
    )
    out = Path(sys.argv[1]).resolve()
    assert out.parent == HERE and not out.exists(), "fresh output inside the run dir only"
    result = fixed_revision_replay.execute(
        report["zip"]["path"],
        HERE / "package_v14/package",
        out,
        report["zip"]["sha256"],
        report["config"],
        time_scale=4.0,
    )
    b127_mod = sys.modules["b127_jaccard_prob_diffusion"]
    eva_features = np.asarray(b127_mod._LAST_EVA_FEATURES)
    hp_features = np.asarray(b127_mod._LAST_HP_FEATURES)
    member_std = np.asarray(b127_mod._LAST_MEMBER_STD)
    if (
        eva_features.shape != (33407, 1024)
        or eva_features.dtype != np.float32
        or not np.isfinite(eva_features).all()
        or hp_features.shape != (33407, 1280)
        or hp_features.dtype != np.float32
        or not np.isfinite(hp_features).all()
        or member_std.shape != (33407, 40)
        or member_std.dtype != np.float32
        or not np.isfinite(member_std).all()
    ):
        raise ValueError("complete finite FP32 EVA02, DINOv3-H+ features and member_std required")
    np.save(out / "eva_features.npy", eva_features, allow_pickle=False)
    np.save(out / "hp_features.npy", hp_features, allow_pickle=False)
    np.save(out / "member_std.npy", member_std, allow_pickle=False)
    write_new(
        out / "extra_capture.json",
        {
            "candidate": report["candidate"],
            "eva_features": pin(out / "eva_features.npy"),
            "hp_features": pin(out / "hp_features.npy"),
            "member_std": pin(out / "member_std.npy"),
        },
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "candidate",
                    "load_seconds",
                    "rank_seconds",
                    "runtime_seconds_before_local_saves",
                    "time_scale",
                    "device",
                    "rows",
                    "queries",
                )
            }
            if isinstance(result, dict)
            else str(result)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

