#!/usr/bin/env bash
# B119G = B116G structure (PE pe4q .4 / DINOv3-H+ .35 / EVA soup .25, H+ at 192) with the H+ member replaced by a same-seed soup that adds
# the 192-trained siblings d3hpqrq (rank_query_balance) and/or d3hpqs (partswap 0.3). After the B118G miss (0.672392 vs exp 0.67479;
# user 10/16: no restore of B116G) the current entry is B118G 0.672392 while the B116G anchor is 0.673449. Rules: every fold >= 0 and
# macro >= +0.0002 vs the B116G proxy (eval_b117.py, then the registered comparator with query bootstrap, macro CI lower bound > -0.0005),
# healthy soup, P(top3) >= 0.8, differential expectation 0.673449 + 0.6876*dLODO >= current entry 0.672392 + 0.001,
# P(drop > 0.002 below B116G) <= 0.15; read-only quota wait before the upload. Upload name <= 35 chars. Never two in flight.
set -u
cd /home/syouya/ml/lab/attribute_based_person_retrieval
B=artifacts/runs/20261016T1710+0900_b119g; B3=artifacts/runs/20261014T0100+0900_b113g; B6=artifacts/runs/20261015T1200+0900_b116g
K=artifacts/runs/20261008T231500+0900_knob_sweep; S=artifacts/runs/20261013T1500+0900_recipe_soup; GT=artifacts/runs/20261008T190000+0900_gold_gate
PE=artifacts/runs/20261015T0410+0900_b116g_pe/soup_pe4q_all_s7; EVS=$B3/soup_eva224s7_eva252s7
exec >> $B/pipeline.log 2>&1
echo "$(date) start"
chk() { taskset -c 0-5 env OMP_NUM_THREADS=6 PYTHONPATH=src scripts/train_python $B3/checks.py "$@" 2>&1 | grep -v Warn | tail -1; }
# 1. fold evidence: wait until every H+ sibling fold soup is built and evaluated
until ls $S/*_hplus_sibling_eval.log > /dev/null 2>&1 && grep -q "all H+ sibling soups built" $S/*_hplus_sibling_eval.log; do sleep 300; done
SEL=$(python3 - <<'PY'
import json
r = json.load(open("artifacts/runs/20261013T1500+0900_recipe_soup/eval_b117_v1.json"))
ok = {k: v for k, v in r["macro"].items() if not k.endswith("_single") and v >= 0.0002 and all(r[f][k] >= 0 for f in ("PA100k", "PETA", "Market1501"))}
print(max(ok, key=ok.get) if ok else "NONE", json.dumps(r["macro"]))
PY
)
echo "fold selection: $SEL"
set -- $SEL; COMBO=$1
[ "$COMBO" = "NONE" ] && { echo "HOLD: not submitted - no H+ sibling soup keeps every fold >= 0 with macro >= 0.0002"; exit 0; }
case $COMBO in
  hp_q_qrq) C="d3hpq d3hpqrq"; TAG=HPQRQ;;   hp_q_qr_qrq) C="d3hpq d3hpqr d3hpqrq"; TAG=HPQRRQ;;
  hp_q_s) C="d3hpq d3hpqs"; TAG=HPQS;;       hp_q_qr_s) C="d3hpq d3hpqr d3hpqs"; TAG=HPQRS;;
  hp_q_qr_qrq_s) C="d3hpq d3hpqr d3hpqrq d3hpqs"; TAG=HP4SIB;;
  *) echo "ABORT unknown combo $COMBO"; exit 1;;
esac
N=B119G_PE4Q_${TAG}_EVAS_FASTG; Z=$B/package/$N.zip; HS=$B/soup_${COMBO}_all_s7
# 2. candidate spec (fold proxies = the same components as the deployed soup) and the registered comparator vs B116G
scripts/train_python - <<PY
import json
S="$S"; F=("Market1501","PA100k","PETA"); POST={"renorm": [[8, 20, 0.5]], "clip": 1e-05, "entropy_delta": 0.1875}
m=[{"weight":0.4,"runs":{f:[f"{S}/pe4q_{f}"] for f in F}},{"weight":0.35,"runs":{f:[f"{S}/${COMBO}_{f}"] for f in F}},
   {"weight":0.25,"runs":{f:[f"{S}/eva_224_252_at252_{f}"] for f in F}}]
c=json.load(open("$B6/candidate_b116g.json"))
c.update(candidate="B119G", package="$Z", lodo={"members":m,"post":POST})
c["runtime_override"]["basis"]=["Same inference graph class as B116G (H+ member weights change only): B116G on the A10 10/15: load 6.7 s, predict 238.1 s, total 245.5 s of the 286 s deadline. Late host slowdowns took up to ~40 s (B114G)."]
c["runtime_override"]["p_plan"]={"A":0.9}
json.dump(c, open("$B/candidate_b119g.json","w"), indent=1)
PY
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 scripts/train_python artifacts/runs/20261014T2310+0900_os_restore/local_validation/lodo_foundation_v1.py \
  --out $B/compare_vs_b116g --candidate B119G=$B/candidate_b119g.json --baseline B116G=$B6/candidate_b116g.json > $B/compare_vs_b116g.log 2>&1
python3 - <<PY || { echo "HOLD: not submitted - comparator does not confirm"; exit 0; }
import json; p=json.load(open("$B/compare_vs_b116g/lodo_foundation_results_v1.json"))["paired"]
v=[p["folds"][f]["mean_diff"] for f in ("PA100k","PETA","Market1501")]; m=p["macro"]
print("comparator folds", [round(x,5) for x in v], "macro", round(m["mean_diff"],5), "ci95", [round(x,5) for x in m["ci95"]])
raise SystemExit(0 if min(v) >= 0 and m["mean_diff"] >= 0.0002 and m["ci95"][0] > -0.0005 else 1)
PY
# 3. all-trainval components and the deployed soup
# 10/17 01:43 fix: the EMA dir writes val_logits.npz (read by soup for pos_weight) ~9 min after model.safetensors -> wait for its
# val_logits.npz AND train_report.json, then settle 60 s (the first B119G attempt read a half-written val_logits.npz: KeyError pos_weight)
for c in $C; do E=$K/${c}_all_trainval_seed20261007_ema0.999; until [ -f $E/model.safetensors ] && [ -f $E/val_logits.npz ] && [ -f $E/train_report.json ] && [ -f $K/${c}_all_trainval_seed20261007/train_report.json ]; do sleep 300; done; done; sleep 60
echo "$(date) all-trainval components done: $C"
RUNS=""; for c in $C; do RUNS="$RUNS $K/${c}_all_trainval_seed20261007_ema0.999"; done
for i in 1 2 3 4; do [ -f $HS/model.safetensors ] && break
  while [ -f $K/PEER_HOLD ]; do sleep 60; done
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 9000 ]; do sleep 30; done
  taskset -c 12-17 scripts/train_python -m upar.h1_finetune soup --runs $RUNS --output $HS 2>&1 | grep -E "SOUP_DONE|Error" | tail -1; done
[ -f $HS/model.safetensors ] || { echo "ABORT H+ soup"; exit 1; }
H=$(chk health $HS $RUNS); echo "health H+: $H"; echo "$H" | grep -q '"ok": true' || { echo "HOLD: not submitted - H+ soup unhealthy"; exit 0; }
# 4. replays need >= 9 GB free: stop the queue runner, then stop our newest knob trainings (they resume from checkpoints when
#    the build restarts the runner) until 12 GB are free; never touches other users' processes
pkill -f '^[^ ]*/python[0-9.]* [^ ]*knob_sweep/knob_queue[.]py'
for p in $(pgrep -u "$(id -u)" -f "attribute_based_person_retrieval/.venv-train/bin/python -m upar.h1_finetune train" | sort -rn); do
  [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12000 ] && break
  echo "$(date) freeing VRAM for the replays: stop $(ps -o args= -p $p | grep -o 'knob_sweep/[^/]*' | head -1) (pid $p)"; kill -TERM $p; sleep 60
done
scripts/h1_build_accept.sh $B $N $PE $HS $EVS 292 6 rope_half,mlp_pad,patch_linear,gelu_tanh || exit 1
[ -f $B/runtime_acceptance_cascade.json ] || { echo "ABORT no acceptance"; exit 1; }
# 5. gate: anchored on the real B116G score
taskset -c 12-17 env OMP_NUM_THREADS=6 scripts/train_python -m upar.gold_gate gate --ledger $GT/ledger_v1.json --candidate $B/candidate_b119g.json --output $B/gate_b119g.json --min-prob 0.8 > /dev/null 2>&1
DEC=$(taskset -c 0-5 env OMP_NUM_THREADS=6 scripts/train_python - <<PY
import json
from upar.gold_gate import t_sf, lodo_of
g=json.load(open("$B/gate_b119g.json")); b116=0.673449; slope=0.6876
new=lodo_of(json.load(open("$B/candidate_b119g.json"))["lodo"])["macro"]; old=lodo_of(json.load(open("$B6/candidate_b116g.json"))["lodo"])["macro"]
d=new-old; exp=b116+slope*d; sd=g["prediction"]["sd"]
p_drop=1-t_sf((-0.002-slope*d)/sd, g["prediction"]["dof"])
cur=0.672392; ok=g["p_top3"]>=0.8 and d>=0.0002 and exp>=cur+0.001 and p_drop<=0.15
print(("GO" if ok else "HOLD"), round(g["p_top3"],4), "dLODO", round(d,5), "exp_diff", round(exp,5), "b116", b116, "current", cur, "p_drop_gt_0.002", round(p_drop,4))
PY
)
echo "gate: $DEC"
set -- $DEC; [ "$1" = "GO" ] || { echo "HOLD: not submitted"; exit 0; }
inflight() { python3 - <<'PY'
import json, urllib.request
tok=open(".secrets/codabench.token").read().strip()
h=json.load(urllib.request.urlopen(urllib.request.Request("https://www.codabench.org/api/submissions/?phase__competition=18196&page_size=100", headers={"Authorization": f"Token {tok}"}), timeout=60))
done=lambda r: str(r.get("status")).lower() in ("finished","failed","cancelled","canceled") or (str(r.get("status")).lower()=="running" and bool(r.get("scores")) and r.get("on_leaderboard") is True)
print(sum(1 for r in h.get("results",[]) if not done(r)))
PY
}
until [ "$(inflight)" = "0" ]; do echo "$(date) waiting: a submission is in flight"; sleep 300; done
CHECK=$(python3 - <<'PY'
import json, urllib.request
tok=open(".secrets/codabench.token").read().strip()
get=lambda p: json.load(urllib.request.urlopen(urllib.request.Request("https://www.codabench.org/api/"+p, headers={"Authorization": f"Token {tok}"}), timeout=60))
ph=get("phases/30704/"); can=get("can_make_submission/30704/"); h=get("submissions/?phase__competition=18196&page_size=100")
print(("OK" if can.get("can") is True and ph.get("execution_time_limit")==300 else "NO"), h.get("count"))
PY
)
set -- $CHECK; [ "$1" = "OK" ] || { echo "ABORT: no slot"; exit 1; }
# shared 15 GB Codabench quota (Track 3 uploads too): read-only check; if the upload does not fit, hold and ask the owner to delete
quota() { python3 - "$Z" <<'PY'
import json, os, sys, urllib.request
tok=open(".secrets/codabench.token").read().strip()
q=json.load(urllib.request.urlopen(urllib.request.Request("https://www.codabench.org/api/user_quota/", headers={"Authorization": f"Token {tok}"}), timeout=60))
used, cap, need = float(q["storage_used"]), float(q["quota"]) * 1e9, os.path.getsize(sys.argv[1])
print(("OK" if used + need <= cap - 0.3e9 else "NO"), round(used / 1e9, 2), round(need / 1e9, 2), round(cap / 1e9, 1))
PY
}
# shared 15 GB Codabench quota (Track 1/3 upload too; the owner deletes superseded submissions by hand): wait, never delete
T0=$(date +%s)
while true; do QS=($(quota)); [ "${QS[0]}" = "OK" ] && break
  [ $(( ( $(date +%s) - T0 ) % 1800 )) -lt 60 ] && echo "$(date) waiting for Codabench quota: used ${QS[1]} GB + upload ${QS[2]} GB > ${QS[3]} GB - 0.3 GB (owner: please delete a superseded submission)"
  sleep 60; done
echo "$(date) Codabench quota ok: used ${QS[1]} GB + upload ${QS[2]} GB"
# the owner may have deleted submissions while we waited: re-check in-flight, the slot and the history count
until [ "$(inflight)" = "0" ]; do echo "$(date) waiting: a submission is in flight"; sleep 300; done
CHECK=$(python3 - <<'PY'
import json, urllib.request
tok=open(".secrets/codabench.token").read().strip()
get=lambda p: json.load(urllib.request.urlopen(urllib.request.Request("https://www.codabench.org/api/"+p, headers={"Authorization": f"Token {tok}"}), timeout=60))
ph=get("phases/30704/"); can=get("can_make_submission/30704/"); h=get("submissions/?phase__competition=18196&page_size=100")
print(("OK" if can.get("can") is True and ph.get("execution_time_limit")==300 else "NO"), h.get("count"))
PY
)
set -- $CHECK; [ "$1" = "OK" ] || { echo "ABORT: no slot"; exit 1; }
SHA=$(sha256sum $Z | cut -d' ' -f1)
scripts/train_python scripts/codabench_submit.py --token-file .secrets/codabench.token --artifact $Z --expected-sha256 $SHA --competition 18196 --phase 30704 --task 36385 \
  --receipt $B/b119g_submission_receipt_v1.json --expected-prior-submission-count $2 --runtime-acceptance $B/runtime_acceptance_cascade.json --submit 2>&1 | tail -1 | cut -c1-300
SID=$(python3 -c "import json;print(json.load(open('$B/b119g_submission_receipt_v1.json')).get('submission',{}).get('id') or '')" 2>/dev/null)
echo "submission id: $SID"; [ -n "$SID" ] || { echo "ABORT: submission state unknown -> check Codabench history by hand (never resubmit blindly)"; exit 1; }
scripts/train_python scripts/watch_codabench_submission.py --token-file .secrets/codabench.token --submission $SID --competition 18196 --phase 30704 \
  --account syouya_tobita --artifact $Z --expected-sha256 $SHA --output-dir $B/b119g_watch_v1 --checks 360 --interval-seconds 60 2>&1 | tail -1
echo "$(date) pipeline done"
