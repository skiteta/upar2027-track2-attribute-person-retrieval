"""Full local CUDA replay of a built candidate ZIP over the public val set (33,407 x 3,462), time_scale 4.

Runs the exact ZIP's extracted package through upar.fixed_revision_replay.execute (same harness as every earlier
package) and saves the same-forward EVA/DINOv3-H+ features and member_std for reference_candidate.py.
usage: replay_candidate.py BUILD_DIR OUT_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")
sys.path.insert(0, str(ROOT / "src"))

from upar import fixed_revision_replay  # noqa: E402
from upar.gallery_feature_package import pin, write_new  # noqa: E402


def main() -> None:
    build = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    report = json.loads((build / "complete.json").read_text())
    assert report["status"] == "PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED"
    result = fixed_revision_replay.execute(report["zip"]["path"], build / "package", out, report["zip"]["sha256"],
                                           report["config"], time_scale=4.0)
    mod = sys.modules["b127_jaccard_prob_diffusion"]
    arrs = {"eva_features": np.asarray(mod._LAST_EVA_FEATURES), "hp_features": np.asarray(mod._LAST_HP_FEATURES),
            "member_std": np.asarray(mod._LAST_MEMBER_STD)}
    shapes = {"eva_features": (33407, 1024), "hp_features": (33407, 1280), "member_std": (33407, 40)}
    for k, a in arrs.items():
        if a.shape != shapes[k] or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("complete finite FP32 " + k + " required")
        np.save(out / f"{k}.npy", a, allow_pickle=False)
    write_new(out / "extra_capture.json", {"candidate": report["candidate"], **{k: pin(out / f"{k}.npy") for k in arrs}})
    print(json.dumps({k: result.get(k) for k in ("status", "candidate", "load_seconds", "rank_seconds",
                                                   "runtime_seconds_before_local_saves", "time_scale", "device")}), flush=True)


if __name__ == "__main__":
    main()
