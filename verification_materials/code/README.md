> **Historical document.** This README describes the earlier entry B137C (submission 1007281). The final submission is **B156C**; start from [`../README.md`](../README.md) and the repository [`README.md`](../../README.md).

# B137C — official Track 2 verification materials (no workshop paper)

This delivery describes **submission 1007281**, owner `syouya_tobita`, B137C.
It is separate from the unscored B153C possible-final candidate. The organizer
snapshot records rank 1/49 and mADM **0.701395**; this is not yet a declaration
that winner/code verification has completed. This local directory has not been
uploaded or sent. Please arrange private transfer of the original model ZIP
alongside `B137C_verification_materials.zip`.

## Files and authoritative sources

- `factsheet/main.pdf` and its complete LaTeX project: fact sheet, not a paper.
- `B137C_TRI_SPHERICAL_MULTIRES_HEAD_RERANK.zip`: original trained model and
  deployed inference code, SHA-256
  `bae4b4c90846bf243d79bf65b5e0929dba5b0725903731c50fea264a05a83077`.
- `delivery_manifest.json`: exact model/member hashes and snapshot/receipt match.
- `deployed_source/`: exact text members of that ZIP, for review. Runtime comes
  from the ZIP, not historical builder scripts.
- `code/`: historical training source/import closure, original configs and
  reports, test fixtures, plus explicitly newer portable preparation helpers.
  This source closure is a **superset** containing later B139/B153 exploration.
  Those later files are not part of the B137C inference entry.
- `supplement/hp6_all_heads.safetensors`: compact retained six-recipe head mixture;
  no images, unrelated weights, credentials or workflow state are included.
- `provenance/`: source assembly, original B137C build/logs, historical checks.
  Absolute original paths in historical records are provenance, not a portable
  runtime prerequisite. The original snapshot README is labelled historical.

The ZIP contains 3 encoders at 280×140 / 192×96 / 252×126, fixed heads and
per-attribute mixture weights. B137C has **12** component training recipes.
It does not use B153C's column-selective head, altered mixture weights,
near-duplicate extension or top-100 attribute pseudo-relevance feedback.

## Verify before loading any model

From the extracted materials directory with the model ZIP alongside it:

```bash
python3 verify_delivery_artifact.py --manifest delivery_manifest.json \
  --model B137C_TRI_SPHERICAL_MULTIRES_HEAD_RERANK.zip --files FILE_MANIFEST.json
```

This reads all bytes, checks SHA-256, all ZIP members/CRC, Python syntax and the
three-member inference contract. It does not prove accuracy or GPU runtime.
Only load this trusted model after its exact hash passes; PyTorch/TorchScript
files are executable artifacts, not safe arbitrary user data.

## Inference environment and execution

The official B137C log records PyTorch 2.4.1+cu121, a CUDA A10, 20 CPU threads,
40 GiB memory allocation and a 300-second limit. `code/pyproject.toml` and
`code/uv.lock` describe the submission-parity environment (Python 3.11).
Use an installed `uv`, not the historical workspace-specific `scripts/uv`:

```bash
cd code
uv sync --frozen --extra submission
```

Network/dependency installation is done only on the preparation machine, not
in the offline organizer runner. A fresh install has not been performed as part
of this delivery; tests used the existing interpreter. The training environment
is different, as described below. Blackwell needs a compatible newer CUDA/torch
stack for execution; a Blackwell timing is not an A10 timing.

Extract the **original model ZIP** into a new `entry/` directory. Ingestion
imports `entry/run.py`, calls `load_model()`, then `rank_gallery(sample)`.
The sample and returned distances follow the organizer starter-kit contract.
For a complete organizer validation split, from `code/`:

```bash
PYTHONPATH=src .venv/bin/python -m upar.execute_submission \
  --submission-dir ../entry \
  --split-dir starter_kit/data/annotations/task2/val \
  --image-root starter_kit/data \
  --output ../replay_new/distances.npy --report ../replay_new/report.json
```

Use a new replay directory. Evaluation of final train+validation-trained models
on validation images is **in-sample execution diagnosis**, not independent
accuracy estimation. The original official run completed in 258.33 seconds;
that measurement is not a worst-case guarantee. `UPAR_TIME_SCALE` is left unset
for the official 300-second policy. Previous shared-GPU engineering replays
with relaxed time limits are not proof of 300-second A10 acceptance.

## Training data, pretraining and recipes

Obtain only the organizers' Track 2 release using the included starter kit;
do not scrape or use hidden test imagery. Place its data under
`code/starter_kit/data`. The six small annotation/sample fixtures included for
unit tests are not a complete dataset and must not be mistaken for one.
Training uses all 3 sources and validation as permitted in the final phase;
one exact-pixel duplicate is excluded, leaving 131,075 images.

Recorded training environment: Python 3.11.17, torch 2.8.0+cu128,
torchvision 0.23.0+cu128, timm 1.0.20, safetensors 0.8.0, numpy 1.26.4,
Pillow 10.4.0. A full training-environment lock was not retained. An install
recipe is illustrative, **not a verified clean-room reproduction**:

```bash
uv venv --python 3.11.17 .venv-train
uv pip install --python .venv-train/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-train/bin/python timm==1.0.20 safetensors==0.8.0 numpy==1.26.4 pillow==10.4.0 huggingface_hub==2.1.1 pyyaml tqdm
```

Generic pretrained files go under `code/pretrained/_h1/<identifier>/model.safetensors`.
Identifiers and historical reference SHA-256 values:

| Identifier | SHA-256 |
| --- | --- |
| `vit_pe_core_large_patch14_336.fb` | `35edcdb68eb43329192fc1530c246191e8a72f9a3dae01e2c7cccf05338d4cac` |
| `vit_huge_plus_patch16_dinov3.lvd1689m` | `0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c` |
| `eva02_large_patch14_224.mim_m38m` | `b326a873bed26c5299647dfec380cfaa16b347ef26cb23051d949269f47b6460` |

These initial pretrained checkpoints are not redistributed here. Acquire them
from the original model providers under their licences, including DINOv3's
separate licence. The model ZIP includes original notices/model cards; no
blanket licence is asserted over third-party components.

From `code/`, build a fresh cache and prepare **new** output paths:

```bash
PYTHONPATH=src .venv-train/bin/python -m upar.h1_finetune cache --out-dir ../training_cache_new --workers 4
python3 scripts/prepare_delivery_training.py --code-root . --cache ../training_cache_new --output ../training_new
PYTHONPATH=src .venv-train/bin/python -m upar.h1_finetune train --config ../training_new/configs/peL280_attrq.json
```

Repeat the last command for the 12 configs in `training_new/configs/`, one
resource-appropriate job at a time. The preparation helper changes only
`cache_dir` and `output_dir`; original recipes/reports are never overwritten.
The twelve recipe names and original hashes are generated in
`training_new/recipe_manifest.json`. Use EMA 0.999 outputs for soups, not raw
weights. The original configs retain the effective batch, sample budget,
optimiser, source masks, augmentation, losses and accumulation settings.

## Assembly and provenance limits

The fact sheet describes exact B137C backbone/head coefficients. Original
head arithmetic is in
`code/artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/pack_b137c_tri_spherical_heads_cpu.py`.
That historical script references old intermediate directories; **do not run
it in place** or assume the whole adapter chain is reconstructable. The exact
deployed source and trained weights are the supplied model ZIP.

- The 10 surviving source EMA checkpoints remain in the project, but are not
  needed for inference and are not in this 1.33 GB model delivery.
- The two 224-pixel DINOv3 head parents were deleted historically. Their mixture
  survives in the compact head source. Original deleted parent bytes cannot
  be proven from the mixture. Recipe retraining is possible, not bit-exact.
- Original per-training source hashes were not recorded for some runs. The
  delivered later snapshot supports all listed configs but is not falsely
  labelled the exact source that trained every run.
- BF16/GPU training is not bit-deterministic. Intermediate runtime-adapter ZIPs
  were deleted; deployed runtime is preserved verbatim instead of inventing a
  replayable chain.
- Historical CPU/reconstruction tests have their original scope. New relocated
  tests are recorded separately and do not certify hidden-domain accuracy.

## Compliance disclosure requiring organizer interpretation

Our own official scoring reports exposed dataset names, gallery/query counts
and per-dataset metrics. Project-level offline proxy/calibration and synthetic
composition checks used this feedback, with only organizer training images.
No test images/labels were accessed, and inference is uniform across domains.
The terms restrict explicit knowledge about hidden-test structure. This
feedback use is disclosed in the fact sheet, and an organizer interpretation
is requested. No automated audit can decide that policy question.

## Submission logistics

Destination in the saved rules: `upar.challenge@gmail.com`. Local target:
**October 26, 2026 AoE = October 27, 2026 20:59 JST**. The workshop site also
lists October 30 for code; the earlier date is used until organizers clarify.
There is no paper in this delivery. Private file hosting/transfer and actual
email sending need the participant's chosen method and authorization.
Technical Codabench recovery is not an authorized final-round reopening.
