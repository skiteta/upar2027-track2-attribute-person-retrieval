#!/usr/bin/env bash
# replay -> reference -> acceptance for one built candidate (usage: verify_chain.sh TAG VERIFIER_PY)
set -u
cd /home/syouya/ml/lab/attribute_based_person_retrieval
D=artifacts/runs/20261021T2250+0900_takeover_knob_search; TAG=$1; V=$2
PYTHONDONTWRITEBYTECODE=1 scripts/train_python $D/replay_candidate.py $D/build_${TAG}_v1 $D/replay_${TAG}_v1 > $D/$(date +%Y%m%dT%H%M)_replay_${TAG}_v1.log 2>&1 < /dev/null || { echo "REPLAY_FAIL $TAG"; exit 1; }
PYTHONDONTWRITEBYTECODE=1 scripts/train_python $D/reference_candidate.py $D/build_${TAG}_v1 $D/replay_${TAG}_v1 < /dev/null 2>&1 | tail -1
python3 $V $D/build_${TAG}_v1 $D/replay_${TAG}_v1 $D/runtime_acceptance_${TAG}_v1.json < /dev/null > $D/acceptance_${TAG}_v1.out 2>&1 && echo "ACCEPTED $TAG" || { echo "ACCEPTANCE_FAIL $TAG"; tail -3 $D/acceptance_${TAG}_v1.out; }
