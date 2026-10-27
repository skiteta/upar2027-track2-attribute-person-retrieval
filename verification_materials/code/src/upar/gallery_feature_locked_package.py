"""Immutable scheduler-only repair of failed B121; no weight/metric change."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import time
import zipfile

from .gallery_feature_package import ROOT, BASE_ZIP, pin, sha, write_new, replace_once

PARENT = ROOT / 'artifacts/runs/20261017T0510+0900_b121g/package_v1/B121G_PE4_FEATURE_AGREEMENT.zip'
PARENT_SHA = 'ed4a1c7597431a4042ef2bfbad83b1782290534014b5ef0e4b60909c28239f94'
NAME = 'B121G_PE4_FEATURE_AGREEMENT_LOCKED3'


def patch_runner(source):
    text = replace_once(source, 'from PIL import Image\n',
                        'from PIL import Image\nfrom b121_deadline import fixed_member_step\n')
    tree = ast.parse(text)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_predict_members')
    target = next(x for x in ast.walk(node) if isinstance(x, ast.If)
                  and ast.unparse(x.test) == 'done >= min_images or left < 0 or extreme')
    lines = text.splitlines(keepends=True)
    replacement = '''        decision = fixed_member_step(
            elapsed=time.time()-_T0, total_budget=runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE),
            soft_prediction_deadline=deadline, projected_remaining=projected(active),
            active=active, required=int(_CONFIG['min_active_members']), remaining=remaining)
        if decision['projection_risk'] and (start // batch) % 10 == 0:
            _log('required_projection_warning', done=done, active=active,
                 projected_s=round(projected(active), 2), **decision)
'''
    # Do not supply active twice through **decision in the runtime logger.
    replacement = replacement.replace("done=done, active=active,", "done=done,")
    lines[target.lineno-1:target.end_lineno] = [replacement]
    text = ''.join(lines)
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == '_predict_members')
    segment = ast.get_source_segment(text, node)
    old = '    for start in range(0, total, batch):\n        t = time.time()\n'
    new = '''    if can_switch or optional or len(specs) != int(_CONFIG['min_active_members']):
        raise RuntimeError('locked ensemble requires one complete mandatory plan')
    for start in range(0, total, batch):
        fixed_member_step(elapsed=time.time()-_T0,
            total_budget=runtime_budget_seconds(_CONFIG, _B121_TIME_SCALE),
            soft_prediction_deadline=deadline, projected_remaining=0.,
            active=active, required=int(_CONFIG['min_active_members']), remaining=total-start)
        t = time.time()
'''
    patched = replace_once(segment, old, new)
    text = replace_once(text, segment, patched)
    ast.parse(text)
    before = {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(source).body if isinstance(x, ast.FunctionDef)}
    after = {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(text).body if isinstance(x, ast.FunctionDef)}
    if [k for k in before if before[k] != after[k]] != ['_predict_members']:
        raise ValueError('only the intended scheduler function may change')
    if any(x in patched for x in ('active -= 1', 'use_fallback = True', '_log("tier_drop"', '_log("plan_switch"')):
        raise ValueError('mutable plan remains reachable')
    return text


def build(output):
    output = Path(output).resolve()
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('failed parent bytes changed')
    output.mkdir(parents=True, exist_ok=False)
    package = output / 'package'
    package.mkdir()
    policy = ROOT / 'src/upar/fixed_member_deadline.py'
    inputs = [pin(PARENT), pin(BASE_ZIP), pin(Path(__file__)), pin(policy)]
    started = time.monotonic()
    with zipfile.ZipFile(PARENT) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe archive path')
            dest = package / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, dest.open('xb') as target:
                shutil.copyfileobj(source, target, 8*2**20)
    run = package / 'run.py'
    run.write_text(patch_runner(run.read_text()))
    cfgpath = package / 'assets/config.json'
    cfg = json.loads(cfgpath.read_text())
    if cfg['candidate'] != 'B121G_PE4_FEATURE_AGREEMENT' or cfg['min_active_members'] != 3 or cfg['plans'] != [{'name':'A','members':[0,1,2]}]:
        raise ValueError('unexpected parent plan')
    cfg['candidate'] = NAME
    cfg['fixed_member_deadline_policy'] = 'mandatory_all3_absolute292_v1'
    cfgpath.write_text(json.dumps(cfg, indent=2)+'\n')
    (package / 'b121_deadline.py').write_bytes(policy.read_bytes())
    artifact = output / (NAME+'.zip')
    with zipfile.ZipFile(artifact,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(package).as_posix())
    preserved=[]
    with zipfile.ZipFile(PARENT) as old, zipfile.ZipFile(artifact) as new:
        for name in old.namelist():
            if name in ('run.py','assets/config.json'): continue
            import hashlib
            if hashlib.file_digest(old.open(name),'sha256').digest()!=hashlib.file_digest(new.open(name),'sha256').digest():
                raise ValueError('model/metric/helper bytes changed: '+name)
            preserved.append(name)
    report={'status':'PACKAGE_BUILT_NOT_RUNTIME_ACCEPTED','zip':pin(artifact),'inputs':inputs,
            'unchanged_members':preserved,'recipe':'same B121 model/feature/graph, mandatory all3 scheduling only',
            'elapsed_seconds':time.monotonic()-started,'official_runtime_guaranteed':False}
    write_new(output/'complete.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
