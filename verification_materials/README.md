# B156C — Track 2 final submission: verification materials (no workshop paper)

This delivery describes our **final submission B156C** (Codabench submission **1011976**), owner `syouya_tobita`,
the single submission of the reopened final round. Our entry in the organisers' earlier leaderboard snapshot was
B137C (submission 1007281, mADM 0.701395, rank 1 of 49). B156C uses B137C's three encoder architectures and
inference code (`run.py`, `b121_*`, `b127_jaccard_prob_diffusion.py`); it changes the DINOv3 head mixture (column-selective, one more head source), the per-attribute member weights, the attribute weights of the scorer and the re-ranking (near-duplicate diffusion, attribute-space query expansion, attribute-weighted body stream).

## Files and authoritative sources

- `factsheet/main.pdf` and its complete LaTeX project: fact sheet, not a paper.
- `B156C_B155C_K2_RERANK.zip` (supplied separately, see `../weights/README.md`): original trained model and deployed inference
  code, 1,333,500,417 bytes, SHA-256 `1279da344c73aa8c67ee55703ba7e2c6e7c1b9607e3a182e9c9e70499fb30844`.
- `delivery_manifest.json`: exact model/member hashes.
- `model_config.json`: the ZIP's `assets/config.json` (member weights, attribute weights, re-ranking knobs).
- `deployed_source/`: exact text members of that ZIP, for review. Runtime comes from the ZIP, not from the
  historical builder scripts.
- `code/`: training source/import closure (`src/upar`), original training configs and reports, the soup / head /
  packaging / candidate-building scripts, test fixtures and portable preparation helpers. It is a **superset**
  that also contains earlier exploration.
- `supplement/hp6_all_heads.safetensors`: compact retained six-recipe DINOv3 head mixture (the r224 head source,
  see "Assembly"); no images, unrelated weights or credentials are included.
- `provenance/`: build records, runtime acceptance and replay reports of B156C, the official B137C logs, and
  historical checks. Absolute paths in historical records are provenance, not a runtime prerequisite.

## Verify before loading any model

From this directory with the model ZIP copied next to it:

```bash
python3 verify_delivery_artifact.py --manifest delivery_manifest.json \
  --model B156C_B155C_K2_RERANK.zip --files FILE_MANIFEST.json
```

This reads all bytes, checks SHA-256, all ZIP members/CRC, Python syntax and the three-member inference contract
for the model, and the SHA-256 of every delivered code/document file. It does not prove accuracy or GPU runtime.
PyTorch/TorchScript files are executable artifacts; load the model only after its hash passes.

## Inference environment and execution

The official runs record PyTorch 2.4.1+cu121, a CUDA A10, 20 CPU threads and a 300-second limit.
`code/pyproject.toml` and `code/uv.lock` describe the submission-parity environment (Python 3.11):

```bash
cd code
uv sync --frozen --extra submission
```

Extract the model ZIP into a new `entry/` directory. The ingestion program imports `entry/run.py`, calls
`load_model()`, then `rank_gallery(sample)`; the sample and the returned distances follow the organiser
starter-kit contract. For a complete organiser validation split, from `code/`:

```bash
PYTHONPATH=src:../entry .venv/bin/python -m upar.execute_submission \
  --submission-dir ../entry \
  --split-dir starter_kit/data/annotations/task2/val \
  --image-root starter_kit/data \
  --output ../replay_new/distances.npy --report ../replay_new/report.json
```

`../docker/run_inference.sh` wraps exactly this and also scores the output. Evaluation of the final
train+validation models on validation images is an in-sample execution check, not an accuracy estimate.
`UPAR_TIME_SCALE` stays unset for the official 300-second policy; on a shared or slower GPU set it to 4 (it relaxes
only the internal deadline checks). B137C's official run took 258.3 s of 300 s (18.4 s re-ranking). B156C has B153C's networks, `run.py` and re-ranking code (only four member-weight columns, three attribute weights and the diffusion step differ); B153C's re-ranking on the official test shape (893 queries × 29,248 images, identical saved inputs, same machine) took 16.3 s versus 15.4–17.9 s for B137C (`provenance/b156c/*.r*.json`). B156C's own official run (submission 1011976) finished its ingestion in 279.0 s of 300 s, with 21.3 s of re-ranking (`provenance/b156c/official_1011976_prediction_ingestion_stdout.txt`).

## Training data, pretraining and recipes

Use only the organisers' Track 2 release (starter kit); place its data under `code/starter_kit/data`. That directory
already holds the small validation fixtures used by the unit tests, which are not a dataset: move it aside first
(`mv starter_kit/data starter_kit/data.fixtures && ln -s /path/to/starter_kit/data starter_kit/data`;
`../docker/train_all.sh` does this). The final models are trained on all three
sources' train + validation splits, as permitted (131,075 images after dropping one exact-pixel duplicate).

Recorded training environment: Python 3.11.17, torch 2.8.0+cu128, torchvision 0.23.0+cu128, timm 1.0.20,
safetensors 0.8.0, numpy 1.26.4, Pillow 10.4.0 (`../docker/Dockerfile.train`, `../docker/requirements-train.txt`).
Generic pretrained files go under `code/pretrained/_h1/<identifier>/model.safetensors`:

| Identifier | SHA-256 |
| --- | --- |
| `vit_pe_core_large_patch14_336.fb` | `35edcdb68eb43329192fc1530c246191e8a72f9a3dae01e2c7cccf05338d4cac` |
| `vit_huge_plus_patch16_dinov3.lvd1689m` | `0d54131c846c91a78ec7641482f32e45b27cb4895f83653c038f305f18c7e61c` |
| `eva02_large_patch14_224.mim_m38m` | `b326a873bed26c5299647dfec380cfaa16b347ef26cb23051d949269f47b6460` |

They are not redistributed; obtain them from the original providers under their licences (DINOv3 has its own
licence; notices and model cards are inside the model ZIP under `licenses/`).

The 13 training recipes (all seed 20261007, fold `all`, `include_val` true, 3,125 steps at batch 128,
EMA 0.999; the EMA weights are used). Original configs and reports:
`code/artifacts/runs/20261008T231500+0900_knob_sweep/<recipe>_all_trainval_seed20261007/`.

| Recipe | Member | Differs from the base recipe by | Used for |
|---|---|---|---|
| `peL280_attrq` | PE | base PE recipe (attribute-query head) | backbone 0.25, head 0.20 |
| `pLq_rankq` | PE | listwise rank loss 0.1 (per source) + query balancing | backbone 0.25, head 0.20 |
| `pLq_maskv2` | PE | per-source label mask v2 | backbone 0.25, head 0.30 |
| `pLq_pswap03` | PE | lower-body part-swap p = 0.3 | backbone 0.25, head 0.30 |
| `d3hpq` | DINOv3 | base DINOv3-H+ recipe | backbone 0.20, head (column-selective) |
| `d3hpqr` | DINOv3 | rank loss 0.1 (per source) | backbone 0.30, head |
| `d3hpqrq` | DINOv3 | rank loss + query balancing | backbone 0.30, head |
| `d3hpqs` | DINOv3 | part-swap p = 0.3 | backbone 0.20, head |
| `d3hpqm` | DINOv3 | label mask v2 | head only |
| `d3hp224q` | DINOv3 | input 224×112 | head only (r224) |
| `d3hp224qr` | DINOv3 | input 224×112 + rank loss | head only (r224) |
| `eva02L224` | EVA | input 224×112 (pos-embed resampled to 18×9 before souping) | backbone 0.5, head 0.5 |
| `eva02L252` | EVA | input 252×126 | backbone 0.5, head 0.5 |

From `code/`, build a fresh cache and prepare **new** output paths (only `cache_dir` and `output_dir` change):

```bash
PYTHONPATH=src .venv-train/bin/python -m upar.h1_finetune cache --out-dir ../training_cache_new --workers 4
python3 scripts/prepare_delivery_training.py --code-root . --cache ../training_cache_new --output ../training_new
PYTHONPATH=src .venv-train/bin/python -m upar.h1_finetune train --config ../training_new/configs/peL280_attrq.json
```

Repeat the last command for every config in `training_new/configs/` (`../docker/train_all.sh` does this). Each run
writes `training_new/runs/<recipe>` and its EMA copy `training_new/runs/<recipe>_ema0.999`; the assembly below uses
the original layout, so link each EMA copy as `$R/<recipe>_all_trainval_seed20261007_ema0.999` first.

## Assembly

Run from `code/` with `scripts/train_python` (= the training interpreter with `PYTHONPATH=src`). `OUT` is a new
directory; `R=artifacts/runs/20261008T231500+0900_knob_sweep`, `S=seed20261007`.

1. **PE backbone soup** (uniform 0.25):
   `scripts/make_pw_aligned.py $R/pLq_maskv2_all_trainval_${S}_ema0.999 $R/peL280_attrq_all_trainval_${S}_ema0.999 OUT/tmp_maskv2_all_s7_pw`, then
   `-m upar.h1_finetune soup --runs $R/peL280_attrq_all_trainval_${S}_ema0.999 $R/pLq_rankq_all_trainval_${S}_ema0.999 OUT/tmp_maskv2_all_s7_pw $R/pLq_pswap03_all_trainval_${S}_ema0.999 --output OUT/soup_pe4q_all_s7`.
2. **DINOv3 backbone soup** 0.20/0.30/0.30/0.20 of `d3hpq`, `d3hpqr`, `d3hpqrq`, `d3hpqs` with in-place fp32
   accumulation in that order (`build_weighted_soup` in
   `artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle/build_and_extract_hp4_20_30_30_20.py`).
   The r224 head source is the uniform soup of the six DINOv3 EMA runs (`hplus6/hp6_all`); its head tensors are
   shipped in `supplement/hp6_all_heads.safetensors` because the 224-pixel parents no longer exist.
3. **EVA member**: `scripts/make_xres_member.py $R/eva02L224_all_trainval_${S}_ema0.999 eva02_l14_252 OUT/xres_eva224s7_as252` (pos-embed 16×8 → 18×9, bicubic, antialias), then `-m upar.h1_finetune soup --runs OUT/xres_eva224s7_as252 $R/eva02L252_all_trainval_${S}_ema0.999 --output OUT/soup_eva224s7_eva252s7`; spherical head soup 0.5/0.5, τ = 1.2.
4. **Heads and int8 member weights** (CPU scripts in `artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/`):
   `pack_hp4_2030_192_cpu.py`, `pack_b136c_spherical_heads_cpu.py`, `pack_b137c_tri_spherical_heads_cpu.py`,
   `pack_b138c_colselective_heads_cpu.py dich3_ltype2_r00` (column-selective DINOv3 head). These scripts also read the
   earlier B124 package (`model_{0,1,2}.weights.pt`, `run.py`, `config.json`) as key templates and a saved head
   backup (`extracted_features_v1/extra_heads_backup_v1.pt`), which are not shipped, so this step is documented
   rather than re-runnable from this repository alone.
5. **Graphs and runtime code**: the weight-free TorchScript graphs come from the B119G build
   (`artifacts/runs/20261016T1710+0900_b119g/pipeline_b119g.sh` → `upar.h1_package`), and `run.py` / `b121_*` /
   `b122c_constant.py` from the documented adapter chain followed by `build_b127c.py` (B139C ZIP). The
   intermediate adapter ZIPs were deleted, so this chain is documented rather than re-runnable; the deployed
   files are shipped verbatim in the model ZIP and in `deployed_source/`.
6. **Re-ranking constants, member and attribute weights**: the final submission B156C is built by
   `scripts/train_python artifacts/runs/20261021T2250+0900_takeover_knob_search/build_candidate.py ../provenance/b156c/spec_b156c.json artifacts/runs/20261021T2250+0900_takeover_knob_search/build_b156c_new`
   (the output directory must be new and inside that run directory). `spec_b156c.json` holds the per-attribute member
   weights (including the four final moves for upper-body orange and pink, lower-body green and sunglasses), the
   attribute-weight multipliers (including Age-Young $\div$1.2, Accessory-Glasses-Sun $\times$1.2 and
   UpperBody-Color-Red $\div$1.1) and the knob overrides on top of `rerank_param4_frozen.py` (diffusion step 1.0).
   It starts from the B139C ZIP (SHA-256 `93507697d74aeba9f27df7b8fb2b411eaccc4f8ac8d1b1990dace1fbe2f4334d`;
   set `parent_zip` in the spec to its location), which is not shipped, so this step is documented rather than
   re-runnable; its result is the shipped B156C ZIP, whose differences from B153C are listed in
   `provenance/b156c/diff_vs_b153c.json`.

Limits: BF16/GPU training is not bit-deterministic, so retraining reproduces the method, not identical bytes. The
shipped int8 weights were re-derived bit-exactly from the surviving fp32 EMA checkpoints for all three members
(heads and backbones). Some per-run source hashes were not recorded at training time; the
delivered snapshot supports all listed configs but is not claimed to be the exact source of every run.

## Evidence for this submission

- The exact model ZIP (1,333,500,417 bytes, all 32 members) passes `verify_delivery_artifact.py`: SHA-256, ZIP CRC, Python syntax and the three-member CUDA contract. It differs from B153C only in `assets/config.json` (candidate name, four member-weight columns, three attribute-distance weights, descriptive knob block) and the single `DEFAULT_K.update` line of the re-ranking module (diffusion step 1.0 instead of 0.95); all network weights and code paths are byte-identical (`provenance/b156c/diff_vs_b153c.json`).
- Full local CUDA replay of the exact ZIP on the public validation split (3,462 queries × 33,407 images, 2026-10-27): every distance and ranking equals an independent reference implementation (`provenance/b156c/runtime_acceptance_b156c_v1.json`, `execute_report.json`, `reference_report.json`). Scoring that output with `docker/evaluate_distances.py` gives the in-sample numbers in `../weights/README.md`.
- Re-ranking time on the official test shape (893 queries × 29,248 images) with identical saved encoder outputs, run through each ZIP's real `rank_gallery`: B153C (same code as B156C) 16.3 s (two runs) versus B137C 15.4–17.9 s on the same machine (`provenance/b156c/*.r0.json`, `*.r1.json`, harness `time_rerank_testshape.py`). B137C's official A10 run took 258.3 s of 300 s.
- The shipped int8 weights of all three members were re-derived bit-exactly from the surviving fp32 EMA checkpoints (2026-10-22/23).
- CPU canary of the exact ZIP under torch 2.4.1+cu121 (`code/scripts/verify_delivery_canary.py`, record `provenance/b156c/canary_b156c.json`): each member loaded from the ZIP reproduces its packaged reference on four fixed synthetic inputs (max abs difference 2.98e-07, all finite).
- The delivered code tree passes its 324 CPU unit tests (no skips; GPU hidden).

## Compliance disclosure requiring organiser interpretation

Our own official scoring reports exposed dataset names, gallery/query counts and per-dataset metrics. Offline proxy
and calibration choices and a composition check built only from organiser training images used this feedback.
No test images or labels were accessed, the code contains no value read from the test set, and inference is uniform
across domains. This is disclosed in the fact sheet so that the organisers can review it against the rule on
hidden-test structure.
