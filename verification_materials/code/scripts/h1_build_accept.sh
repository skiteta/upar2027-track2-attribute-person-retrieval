#!/usr/bin/env bash
# Build + verify one no-degrade GPU-runner package (plan A only, 3 members, canary, blocks layout, post) and write
# <out>/runtime_acceptance_cascade.json. Never submits. Shared by the B110R restore and later candidates.
# usage: scripts/h1_build_accept.sh OUT_DIR NAME RUN_DIR_PEL RUN_DIR_D3 RUN_DIR_EVA [TIME_BUDGET_S=286] [RANK_RESERVE_S=6] [FAST=] [MEMBER_WEIGHTS_JSON=E7 0.4/0.35/0.25]
# FAST: comma list for h1_package --fast (speed2 study; exact ones: rope_half,mlp_pad,patch_linear); empty = unchanged graphs
# Replays pause (SIGSTOP, never kill) this project's training jobs and stop the knob queue runner; both resume on exit.
set -u
cd /home/syouya/ml/lab/attribute_based_person_retrieval
O=$1; N=$2; R0=$3; R1=$4; R2=$5; TB=${6:-286}; RR=${7:-6}; FAST=${8:-}
B4=artifacts/runs/20261012T0200+0900_b110g  # plans.json / member_weights.json of the E7 design (0.4/0.35/0.25)
MW=${9:-$B4/member_weights.json}  # per-member x per-attribute weights (B118G: 0.35/0.45/0.20)
POST='{"renorm": [[8, 20, 0.5]], "clip": 1e-05, "entropy_delta": 0.1875}'
mkdir -p $O; Z=$O/package/$N.zip
for r in $R0 $R1 $R2; do [ -f $r/model.safetensors ] || { echo "ABORT missing $r"; exit 1; }; done
[ "$(python3 -c "import shutil;print(int(shutil.disk_usage('.').free>30e9))")" = "1" ] || { echo "ABORT disk below 30 GB"; exit 1; }
rm -rf $O/package $O/replay_gpu_default $O/replay_gpu_forcedA $O/verify_planA
taskset -c 12-17 scripts/train_python -m upar.h1_package --run-dirs $R0 $R1 $R2 \
  --out-dir $O/package --name $N --weight-storage int8 --time-budget-s $TB --rank-reserve-s $RR \
  --plans $B4/plans.json --plan-margin 0.95 --member-weights $MW --gpu-runner --min-active-members 3 --layout blocks --post "$POST" ${FAST:+--fast $FAST} 2>&1 | grep -v Warn | tail -1
[ -f $Z ] || { echo "ABORT package failed"; exit 1; }
rm -rf $O/package/$N
echo "$(date) packaged $(sha256sum $Z | cut -c1-16)"
taskset -c 12-17 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 PYTHONPATH=src .venv/bin/python -m upar.pkg_numeric_check --artifact $Z --output $O/numeric_check_cpu_torch241.json --stride 64 > /dev/null 2>&1
python3 -c "import json,sys;d=json.load(open('$O/numeric_check_cpu_torch241.json'));print('numeric', d['pass'], {k:round(v['mean_abs'],5) for k,v in d['plans'].items()});sys.exit(0 if d['pass'] else 1)" || { echo "ABORT numeric"; exit 1; }
V=$O/verify_planA; mkdir -p $V/src
(cd $V/src && unzip -q ../../package/$N.zip && python3 -c "
import json;c=json.load(open('assets/config.json'));c['time_budget_s']=100000.0;json.dump(c,open('assets/config.json','w'),indent=1)" && zip -q -r -X ../VERIFY_PLAN_A_ONLY.zip .) && rm -rf $V/src
python3 - <<PY
import json, hashlib
p="$V/VERIFY_PLAN_A_ONLY.zip"; b=open(p,"rb").read(); m=json.load(open("$O/package/${N}_package_manifest.json"))
m["artifact"]={"path": p, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}; m["note"]="VERIFICATION COPY ONLY - time_budget_s=100000"
json.dump(m, open("$V/verify_manifest.json","w"), indent=1)
PY
# replays: stop the queue runner (no backfill), wait for memory without killing healthy training, then pause own training
pkill -f '^[^ ]*/python[0-9.]* [^ ]*knob_sweep/knob_queue[.]py'
trap 'scripts/knob_queue_ensure.sh' EXIT
T1=$(date +%s)
# PEER_HOLD (2D-pose session, same owner) also binds the replays: leave >= 19 GiB free on top of the replay memory
need() { [ -f artifacts/runs/20261008T231500+0900_knob_sweep/PEER_HOLD ] && echo $((9000 + 19456)) || echo 9000; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge "$(need)" ]; do
  [ $(( ( $(date +%s) - T1 ) % 600 )) -lt 20 ] && echo "$(date) waiting for $(need) MiB free for the replays ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader))"
  sleep 20
done
PIDS=$(pgrep -u "$(id -u)" -f "attribute_based_person_retrieval/.venv-train/bin/python -m upar.h1_finetune train" | tr '\n' ' ')
echo "$(date) GPU replays (training paused: $PIDS)"
for p in $PIDS; do kill -STOP $p; done
trap 'for p in $PIDS; do kill -CONT $p 2>/dev/null; done; scripts/knob_queue_ensure.sh' EXIT
scripts/train_python -m upar.h1_replay --artifact $Z --manifest $O/package/${N}_package_manifest.json --out-dir $O/replay_gpu_default --cores 6-11,18-23 --gpu-python $PWD/.venv-train/bin/python --time-scale 3 > /dev/null 2>&1
scripts/train_python -m upar.h1_replay --artifact $V/VERIFY_PLAN_A_ONLY.zip --manifest $V/verify_manifest.json --out-dir $O/replay_gpu_forcedA --cores 6-11,18-23 --gpu-python $PWD/.venv-train/bin/python --force-plan A > /dev/null 2>&1
for p in $PIDS; do kill -CONT $p; done
echo "$(date) replays done, training resumed"
rm -rf $O/replay_gpu_default/extracted $O/replay_gpu_forcedA/extracted
for d in replay_gpu_default replay_gpu_forcedA; do python3 -c "
import json;d=json.load(open('$O/$d/replay_report.json'));print('$d', {k:d.get(k) for k in ('returncode','wall_seconds','device','realized_plan','abs_mADM_diff')})"; done
scripts/train_python - <<PY
import hashlib, json, importlib.util
from pathlib import Path
O=Path("$O"); sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=Path("$Z"); v=O/"verify_planA/VERIFY_PLAN_A_ONLY.zip"; d=O/"replay_gpu_default/replay_report.json"; f=O/"replay_gpu_forcedA/replay_report.json"
acc={"kind":"h1_vit_gpu_cascade_acceptance_v1","artifact":{"path":str(z),"sha256":sha(z)},"default_replay_report":{"path":str(d),"sha256":sha(d)},
     "forced_first_plan_replay_report":{"path":str(f),"sha256":sha(f)},"verification_copy":{"path":str(v),"sha256":sha(v)}}
tmp=O/"acceptance.candidate.json"; tmp.write_text(json.dumps(acc,indent=1))
spec=importlib.util.spec_from_file_location("g","src/upar/h1_vit_gpu_cascade_acceptance.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("acceptance", json.dumps(m.verify_release(z,tmp))); tmp.rename(O/"runtime_acceptance_cascade.json")
PY
[ -f $O/runtime_acceptance_cascade.json ] || { echo "ABORT acceptance"; exit 1; }
echo "$(date) accepted $Z"
