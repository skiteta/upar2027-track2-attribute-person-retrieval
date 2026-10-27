"""Copy of a run whose output space is moved to a reference pos_weight: deployed p = sigmoid(z - log pw), so add
(log pw_ref - log pw_src) to the final bias (head.base.bias and its alias head.bias) and to the stored val logits.
Needed to weight-soup a label_mask v2 run (its pos_weight differs on 1 attribute) with attrq/rank runs.
usage: make_pw_aligned.py SRC_RUN REF_RUN OUT_DIR"""
import json, shutil, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file, save_file
src, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ps, pr = np.load(src / "val_logits.npz"), np.load(ref / "val_logits.npz")
d = (np.log(pr["pos_weight"]) - np.log(ps["pos_weight"])).astype(np.float32)
sd = load_file(str(src / "model.safetensors"))
keys = [k for k in ("head.base.bias", "head.bias") if k in sd]; assert keys, "no head bias"
for k in keys:
    sd[k] = (sd[k].float() + torch.from_numpy(d)).to(sd[k].dtype).contiguous()
out.mkdir(parents=True, exist_ok=True); save_file(sd, str(out / "model.safetensors"))
z = {k: ps[k] for k in ps.files}; z["logits_original"] = ps["logits_original"] + d; z["logits_flip"] = ps["logits_flip"] + d; z["pos_weight"] = pr["pos_weight"]
np.savez(out / "val_logits.npz", **z)
cfg = json.loads((src / "config.json").read_text()); cfg.update(output_dir=str(out), pw_aligned_of=str(src), pw_ref=str(ref))
(out / "config.json").write_text(json.dumps(cfg, indent=1)); shutil.copy(src / "train_report.json", out / "train_report.json")
print("ok", keys, "max |delta|", float(np.abs(d).max()))
