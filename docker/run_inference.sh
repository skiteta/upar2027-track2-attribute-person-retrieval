#!/usr/bin/env bash
# Run the exact B156C Codabench submission on a public UPAR Track 2 split and score it.
# usage: run_inference.sh MODEL_ZIP DATA_ROOT OUT_DIR [SPLIT]
#   MODEL_ZIP  B156C_B155C_K2_RERANK.zip (the submitted file, SHA-256 checked below)
#   DATA_ROOT  starter-kit data root containing annotations/task2/<SPLIT>/ and the image folders
#   OUT_DIR    receives distances.npy, report.json (run.py execution) and metrics.json
#   SPLIT      val (default) or any folder with gt.csv / ids.csv / queries.csv in the starter-kit format
# env: PYTHON (default python3), UPAR_TIME_SCALE (leave unset for the official 300-second policy)
set -euo pipefail
MODEL_ZIP=$(realpath "$1"); DATA_ROOT=$(realpath "$2"); mkdir -p "$3"; OUT_DIR=$(realpath "$3"); SPLIT=${4:-val}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
EXPECTED=1279da344c73aa8c67ee55703ba7e2c6e7c1b9607e3a182e9c9e70499fb30844
ACTUAL=$($PY -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');[h.update(b) for b in iter(lambda:f.read(1<<24),b'')];print(h.hexdigest())" "$MODEL_ZIP")
[ "$ACTUAL" = "$EXPECTED" ] || { echo "model ZIP SHA-256 mismatch: $ACTUAL" >&2; exit 2; }
mkdir -p "$OUT_DIR"
ENTRY=$(mktemp -d)
$PY -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$MODEL_ZIP" "$ENTRY"
echo "model verified and extracted to $ENTRY"
# Codabench runs the submission with its folder as working directory (run.py imports its package-local modules)
(cd "$ENTRY" && $PY -m upar.execute_submission --submission-dir "$ENTRY" --split-dir "$DATA_ROOT/annotations/task2/$SPLIT" \
    --image-root "$DATA_ROOT" --output "$OUT_DIR/distances.npy" --report "$OUT_DIR/report.json")
$PY "$HERE/evaluate_distances.py" --split-dir "$DATA_ROOT/annotations/task2/$SPLIT" \
    --distances "$OUT_DIR/distances.npy" --out "$OUT_DIR/metrics.json"
