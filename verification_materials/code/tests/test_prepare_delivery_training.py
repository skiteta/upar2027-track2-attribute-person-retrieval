import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_training', Path(__file__).resolve().parents[1] / 'scripts/prepare_delivery_training.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
CODE = Path(__file__).resolve().parents[1]


def test_only_paths_change_and_original_is_immutable(tmp_path):
    code = tmp_path / 'code'
    original = {'fold': 'all', 'include_val': True, 'seed': 20261007,
                'lr': 0.00004, 'cache_dir': 'old/cache', 'output_dir': 'old/run'}
    originals = []
    for rel in mod.RECIPES.values():
        src = code / rel
        src.parent.mkdir(parents=True)
        src.write_text(json.dumps(original))
        originals.append(src)
    output = tmp_path / 'new_run'
    result = mod.prepare(code, tmp_path / 'cache', output)
    assert len(result['recipes']) == 13
    for src in originals:
        assert json.loads(src.read_text()) == original
    for path in (output / 'configs').glob('*.json'):
        cfg = json.loads(path.read_text())
        changed = {key for key in cfg if cfg[key] != original[key]}
        assert changed == {'cache_dir', 'output_dir'}
    with pytest.raises(FileExistsError):
        mod.prepare(code, tmp_path / 'cache', output)


def test_delivered_recipes_are_final_all_trainval_configs():
    for name, rel in mod.RECIPES.items():
        cfg = json.loads((CODE / rel).read_text())
        assert (cfg['fold'], cfg['include_val'], cfg['seed']) == ('all', True, 20261007), name
