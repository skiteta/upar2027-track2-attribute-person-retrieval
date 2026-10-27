#!/usr/bin/env python3
"""Prepare portable copies of the B156C training recipes; never start training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

KNOB = 'artifacts/runs/20261008T231500+0900_knob_sweep'
# recipe name -> original all_trainval config (relative to the code root)
RECIPES = {name: f'{KNOB}/{name}_all_trainval_seed20261007/config.json'
           for name in ('peL280_attrq', 'pLq_rankq', 'pLq_maskv2', 'pLq_pswap03',
                        'd3hpq', 'd3hpqr', 'd3hpqrq', 'd3hpqs', 'd3hpqm', 'd3hp224q', 'd3hp224qr',
                        'eva02L224', 'eva02L252')}


def prepare(code: Path, cache: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite training artifacts')
    rows, configs = [], []
    for name, rel in RECIPES.items():
        src = code / rel
        cfg = json.loads(src.read_text())
        if cfg['fold'] != 'all' or not cfg['include_val'] or cfg['seed'] != 20261007:
            raise ValueError('Unexpected original recipe')
        cfg['cache_dir'] = str(cache.resolve())
        cfg['output_dir'] = str(output.resolve() / 'runs' / name)
        configs.append((name, cfg))
        rows.append({'recipe': name, 'original': rel,
                     'original_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                     'config': str(output.resolve() / 'configs' / (name + '.json'))})
    (output / 'configs').mkdir(parents=True)
    for name, cfg in configs:
        (output / 'configs' / (name + '.json')).write_text(json.dumps(cfg, indent=2) + '\n')
    report = {'status': 'CONFIGS_PREPARED_NOT_TRAINED', 'changed_fields': ['cache_dir', 'output_dir'],
              'recipes': rows, 'bit_exact_retraining_claimed': False}
    (output / 'recipe_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.code_root.resolve(), a.cache, a.output)))


if __name__ == '__main__':
    main()
