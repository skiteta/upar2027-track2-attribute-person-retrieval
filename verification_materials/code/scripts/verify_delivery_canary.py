#!/usr/bin/env python3
"""Exact-ZIP CPU synthetic canaries only; never a GPU runtime/accuracy approval."""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from verify_delivery_artifact import sha256, verify_zip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads(a.manifest.read_text())
    proof = verify_zip(a.model, manifest['model'])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = []
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='upar_delivery_canary_') as scratch:
        # ZIP was fully authenticated and its member paths/symlinks checked above.
        with zipfile.ZipFile(a.model) as archive:
            archive.extractall(scratch)
        sys.path.insert(0, scratch)
        spec = importlib.util.spec_from_file_location('verified_entry_cpu_canary', Path(scratch) / 'run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._CONFIG = proof['config']
        module._DEVICE = 'cpu'
        config = proof['config']
        raw = module._canary_input(config['canary']['n'], *config['cache_hw'])
        with torch.inference_mode():
            for model_spec, reference in zip(config['models'], config['canary']['ref']):
                t = time.monotonic()
                model = module._load_ts(model_spec, 'cpu')
                got = module._run(model, model_spec, raw)
                ref = np.asarray(reference, dtype=np.float32)
                diff = np.abs(got - ref)
                row = {'name': model_spec['name'], 'shape': list(got.shape),
                       'finite': bool(np.isfinite(got).all()),
                       'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                       'seconds': time.monotonic() - t}
                row['pass'] = (row['finite'] and row['shape'] == [4, 40]
                               and row['mean_abs'] <= config['canary'].get('mean_tol', .02)
                               and row['max_abs'] <= config['canary'].get('max_tol', .3))
                rows.append(row)
                print(json.dumps(row), flush=True)
                del model
                gc.collect()
    result = {'status': 'PASS_CPU_SYNTHETIC_CANARIES' if all(r['pass'] for r in rows) else 'FAIL',
              'candidate': manifest['candidate'], 'model_sha256': manifest['model']['sha256'],
              'verifier_sha256': sha256(Path(__file__)), 'torch': torch.__version__,
              'threads': 2, 'members': rows, 'seconds': time.monotonic() - start,
              'scope': 'Four fixed synthetic inputs per member; CPU loader + forward compatibility only. No full ranking, GPU timing, clean install or generalization claim.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    if result['status'] != 'PASS_CPU_SYNTHETIC_CANARIES':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
