#!/usr/bin/env bash
# Retrain the 13 B156C component recipes (verification_materials/README.md, "Training data, pretraining and recipes").
# usage (inside the training image, working directory = code/): train_all.sh DATA_ROOT PRETRAINED_DIR WORK_DIR
#   DATA_ROOT       starter-kit data root (annotations/ + image folders); linked to code/starter_kit/data
#   PRETRAINED_DIR  holds _h1/<timm identifier>/model.safetensors for the 3 generic checkpoints (see README table)
#   WORK_DIR        new directory for the 256x128 image cache, rewritten configs and training outputs
set -euo pipefail
DATA_ROOT=$1; PRETRAINED=$2; WORK=$3
PY=${PYTHON:-python3}
mkdir -p "$WORK" starter_kit
if [ ! -L starter_kit/data ]; then
  [ -e starter_kit/data ] && mv starter_kit/data starter_kit/data.fixtures
  ln -s "$DATA_ROOT" starter_kit/data
fi
[ -e pretrained ] || ln -s "$PRETRAINED" pretrained
$PY -m upar.h1_finetune cache --out-dir "$WORK/cache" --workers 4
$PY scripts/prepare_delivery_training.py --code-root . --cache "$WORK/cache" --output "$WORK/training"
for cfg in "$WORK"/training/configs/*.json; do
  echo "=== $(date -u +%FT%TZ) training $(basename "$cfg")"
  $PY -m upar.h1_finetune train --config "$cfg"
done
echo "done: EMA 0.999 outputs are under $WORK/training; soups/heads/packaging follow verification_materials/README.md"
