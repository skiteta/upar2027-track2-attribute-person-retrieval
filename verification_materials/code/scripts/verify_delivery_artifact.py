#!/usr/bin/env python3
"""Offline, read-only verification of a delivery manifest and its exact model ZIP."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name and str(p) == name.rstrip('/'))


def verify_zip(path: Path, expected: dict) -> dict:
    if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError('Model ZIP size/SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
            raise ValueError('Unsafe/duplicate ZIP member')
        if any(stat.S_ISLNK(i.external_attr >> 16) or i.flag_bits & 1 for i in infos):
            raise ValueError('Symlink/encrypted member is not supported')
        if sum(i.file_size for i in infos) > 8 * 1024**3:
            raise ValueError('Unexpected expanded size')
        required = {'run.py', 'metadata.yaml', 'assets/config.json',
                    'b127_jaccard_prob_diffusion.py', 'licenses/DINOv3_LICENSE.md'}
        if not required.issubset(names):
            raise ValueError('Missing inference/runtime/license member')
        rows = []
        for info in infos:
            with archive.open(info) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            rows.append({'path': info.filename, 'bytes': info.file_size, 'sha256': digest})
            if info.filename.endswith('.py'):
                ast.parse(archive.read(info).decode(), filename=info.filename)
        if expected.get('members') is not None and rows != expected['members']:
            raise ValueError('Member inventory mismatch')
        config = json.loads(archive.read('assets/config.json'))
        if len(config['attribute_names']) != 40 or len(set(config['attribute_names'])) != 40:
            raise ValueError('Invalid attribute contract')
        models = config['models']
        if len(models) != 3 or config['min_active_members'] != 3 or not config['require_cuda']:
            raise ValueError('Required three-member CUDA plan changed')
        if not config['fail_closed_on_fallback']:
            raise ValueError('Unexpected fallback enabled')
        for model in models:
            for key in ('file', 'block_file', 'head_file', 'weights_file'):
                if model[key] not in names:
                    raise ValueError('Missing model component: ' + model[key])
            if len(model['member_weights']) != 40 or model['flip_tta']:
                raise ValueError('Member weights/TTA contract changed')
        if any(abs(sum(m['member_weights'][i] for m in models) - 1) > 1e-6 for i in range(40)):
            raise ValueError('Invalid probability mixture weights')
    return {'status': 'PASS_BYTES_CRC_SYNTAX_CONTRACT', 'members': rows,
            'config': config, 'scope': 'Not GPU execution, a clean install, scientific improvement or organizer approval.'}


def verify_files(root: Path, manifest: dict) -> int:
    names = [row['path'] for row in manifest['files']]
    if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
        raise ValueError('Unsafe/duplicate manifest path')
    for row in manifest['files']:
        path = root / row['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in path.parents if p.is_relative_to(root))):
            raise ValueError('Symlink/escaping manifest input')
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Missing/changed delivery file: ' + row['path'])
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--files', type=Path, help='Optional complete code/document inventory to verify too')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = verify_zip(args.model, manifest['model'])
    file_count = verify_files(args.files.resolve().parent, json.loads(args.files.read_text())) if args.files else None
    print(json.dumps({'status': result['status'], 'candidate': manifest['candidate'],
                      'member_count': len(result['members']), 'verified_delivery_files': file_count,
                      'scope': result['scope']}))


if __name__ == '__main__':
    main()
