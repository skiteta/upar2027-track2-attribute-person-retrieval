import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('delivery_check', Path(__file__).resolve().parents[1] / 'scripts/verify_delivery_artifact.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def synthetic_zip(tmp_path, *, missing=False, duplicate=False, bad_sum=False):
    """Structural fixture only: placeholder tensors are not execution evidence."""
    path = tmp_path / 'fixture.zip'
    config = {'attribute_names': ['a' + str(i) for i in range(40)],
              'models': [{'file': 'm.ts', 'block_file': 'm.ts', 'head_file': 'm.ts',
                          'weights_file': 'm.pt', 'member_weights': [0.1 if bad_sum else 1 / 3] * 40,
                          'flip_tta': False} for _ in range(3)],
              'min_active_members': 3, 'require_cuda': True, 'fail_closed_on_fallback': True}
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in [('run.py', 'pass'), ('metadata.yaml', ''),
                              ('assets/config.json', json.dumps(config)),
                              ('b127_jaccard_prob_diffusion.py', 'pass'),
                              ('licenses/DINOv3_LICENSE.md', 'fixture'), ('m.ts', 'fixture'), ('m.pt', 'fixture')]:
            if not (missing and name == 'run.py'):
                z.writestr(name, content)
        if duplicate:
            with pytest.warns(UserWarning):
                z.writestr('run.py', 'pass')
    return path, {'bytes': path.stat().st_size, 'sha256': mod.sha256(path)}


def test_structural_fixture_only(tmp_path):
    path, expected = synthetic_zip(tmp_path)
    proof = mod.verify_zip(path, expected)
    assert proof['status'] == 'PASS_BYTES_CRC_SYNTAX_CONTRACT'
    assert len(proof['members']) == 7
    assert 'Not GPU' in proof['scope']
    mod.verify_zip(path, {**expected, 'members': proof['members']})


@pytest.mark.parametrize('flag,match', [('missing', 'Missing'), ('duplicate', 'duplicate'), ('bad_sum', 'mixture')])
def test_invalid_archive(tmp_path, flag, match):
    path, expected = synthetic_zip(tmp_path, **{flag: True})
    with pytest.raises(ValueError, match=match):
        mod.verify_zip(path, expected)


def test_tampered_bytes_and_inventory(tmp_path):
    path, expected = synthetic_zip(tmp_path)
    with pytest.raises(ValueError, match='inventory'):
        mod.verify_zip(path, {**expected, 'members': []})
    with pytest.raises(ValueError, match='SHA'):
        mod.verify_zip(path, {**expected, 'sha256': 'a' * 64})


@pytest.mark.parametrize('name', ['../escape', '/etc/passwd', 'foo/../../bar', 'C:/x', 'a\\b', './foo', 'foo//bar'])
def test_unsafe_member_names(name):
    assert not mod.safe_name(name)


def test_complete_delivery_inventory(tmp_path):
    payload = tmp_path / 'code.py'
    payload.write_text('pass\n')
    row = {'path': 'code.py', 'bytes': payload.stat().st_size, 'sha256': mod.sha256(payload)}
    assert mod.verify_files(tmp_path, {'files': [row]}) == 1
    with pytest.raises(ValueError, match='duplicate'):
        mod.verify_files(tmp_path, {'files': [row, row]})
    payload.write_text('changed')
    with pytest.raises(ValueError, match='Missing/changed'):
        mod.verify_files(tmp_path, {'files': [row]})
