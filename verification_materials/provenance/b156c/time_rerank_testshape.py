"""Time the actual package rank_gallery (scoring + feature rerank) on the official test shape (893 queries x 29,248
gallery), with the encoder stage replaced by the saved B153C val capture (random fixed subset). Same input for every
package, so only the rerank code/config differs. Shared local GPU: use ratios between packages, not absolute seconds.

usage: time_rerank_testshape.py PKG_DIR OUT_JSON [seed]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "artifacts/runs/20261021T2250+0900_takeover_knob_search/replay_b153c_v1"
sys.path.insert(0, str(ROOT / "src"))
from upar.submission import load_submission_module  # noqa: E402

pkg, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
cap = np.load(REPLAY / "capture.npz")
rng = np.random.default_rng(seed)
gi = np.sort(rng.choice(33407, 29248, replace=False))
qi = np.sort(rng.choice(3462, 893, replace=False))
p, queries = cap["probabilities"][gi], cap["queries"][qi]
names, paths = cap["attribute_names"].tolist(), [cap["image_paths"].tolist()[i] for i in gi]
feats = [np.ascontiguousarray(np.load(REPLAY / n)[gi]) for n in ("features.npy", "eva_features.npy", "hp_features.npy", "member_std.npy")]
os.environ.pop("UPAR_FORCE_PLAN", None)
os.environ["UPAR_TIME_SCALE"] = "1"
sys.path.insert(0, str(pkg))
mod = load_submission_module(pkg / "run.py")
mod._CONFIG = json.loads((pkg / "assets/config.json").read_text())
mod._DEVICE = torch.device("cuda")
mod._GALLERY_FEATURES, mod._GALLERY_EVA_FEATURES, mod._GALLERY_HP_FEATURES, mod._GALLERY_MEMBER_STD = feats
mod.predict_attributes = lambda gallery, attribute_names: p
logs = []
orig = mod._log


def log(event, **values):
    logs.append({"event": event, "wall": time.perf_counter(), **{k: v for k, v in values.items() if isinstance(v, (int, float, str))}})


mod._log = log
sample = {"attribute_names": names, "queries": queries, "gallery": [{"index": i, "image": x} for i, x in enumerate(paths)]}
torch.cuda.synchronize()
t0 = time.perf_counter()
res = mod.rank_gallery(sample)
torch.cuda.synchronize()
el = time.perf_counter() - t0
d = np.asarray(res["distances"], dtype=np.float32)
stages = [{"event": x["event"], "t": round(x["wall"] - t0, 3), **{k: v for k, v in x.items() if k.endswith("seconds")}} for x in logs]
rep = {"pkg": pkg.name, "seed": seed, "queries": len(queries), "gallery": len(p), "entry_seconds": el, "stages": stages,
       "distances_sha_raw": __import__("hashlib").sha256(d.tobytes()).hexdigest(), "device": torch.cuda.get_device_name(0),
       "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "time": time.strftime("%F %T")}
out.write_text(json.dumps(rep, indent=1) + "\n")
print(pkg.name, round(el, 2), flush=True)
