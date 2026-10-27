# UPAR Challenge 2027 — Track 2 (Attribute-Based Person Retrieval) — `syouya_tobita`

Code, training recipes, Docker files and verification for our **final submission, B156C**
(Codabench submission **1011976**), the single submission of the reopened final round.
No workshop paper was submitted. The fact sheet is in [`factsheet/`](factsheet/) (PDF + LaTeX).

| Submission | mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
|---|---|---|---|---|---|---|
| B156C, final (1011976) | pending | pending | pending | pending | pending | pending |
| B137C, earlier snapshot entry (1007281, rank 1 of 49) | 0.701395 | 0.418136 | 0.394486 | 0.567449 | 0.642728 | 0.293015 |

Method in one paragraph: three generic foundation models (PE-Core-L/14 at 280×140, DINOv3-H+/16 at 192×96,
EVA02-L/14 at 252×126) are fully fine-tuned to predict the 40 attributes; same-seed recipe soups and spherical
head soups give one network per family; per-attribute member weights give gallery probabilities; a weighted Bernoulli
scorer plus gallery-only graph re-ranking (probability diffusion on mutual-kNN graphs with extra near-duplicate
diffusion, visual anchors, distance smoothing, and an attribute-space expansion of each query from its own top-100
results) produce the query × gallery distances. B156C keeps B137C's three encoders and inference code; it changes
the DINOv3 head mixture (column-selective, one more head source), the per-attribute member weights, the attribute weights of the scorer and the re-ranking (near-duplicate diffusion, attribute-space query expansion, attribute-weighted body stream). Details: [`factsheet/main.pdf`](factsheet/main.pdf).

## Repository layout

| Path | Content |
|---|---|
| `factsheet/` | Fact sheet (organisers' 2027 template): `main.pdf`, `main.tex`, `references.bib`, class files |
| `docker/` | `Dockerfile.inference` (Codabench runner image), `run_inference.sh`, `evaluate_distances.py`, `official_adm.py`; `Dockerfile.train`, `requirements-train.txt`, `train_all.sh` |
| `weights/` | Where to put the trained weights (download link and SHA-256 in [`weights/README.md`](weights/README.md)) |
| `verification_materials/` | Full source (`code/src/upar`), the 13 original training configs and reports (`code/artifacts/runs/...`), the soup/head/packaging scripts, the exact text members of the submitted ZIP (`deployed_source/`), provenance, `verify_delivery_artifact.py`, and a detailed guide (`verification_materials/README.md`) |

## 1. Trained weights

The trained model is the exact file we submitted to Codabench, `B156C_B155C_K2_RERANK.zip`
(1,333,500,417 bytes, SHA-256 `1279da344c73aa8c67ee55703ba7e2c6e7c1b9607e3a182e9c9e70499fb30844`). It contains
`run.py`, the three int8/fp16 TorchScript networks, heads and the configuration. Download it from the link in
[`weights/README.md`](weights/README.md) into `weights/` and check it:

```bash
sha256sum weights/B156C_B155C_K2_RERANK.zip   # must print 1279da34...b30844
```

## 2. Inference with Docker (reproduces the Codabench run)

The inference image is the Codabench Track 2 runner image (`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`,
Python 3.11, torch 2.4.1, numpy, Pillow); nothing is installed on top. `run_inference.sh` checks the SHA-256,
extracts the ZIP, imports its `run.py` from the extracted folder exactly like the ingestion program (`load_model()`,
then `rank_gallery(sample)`), saves the Q×G distances and scores them with the published metric definitions.

```bash
docker build -f docker/Dockerfile.inference -t upar2027-track2-infer .
docker run --rm --gpus all \
  -v "$PWD/weights:/weights:ro" -v /path/to/starter_kit/data:/data:ro -v "$PWD/out:/out" \
  upar2027-track2-infer /weights/B156C_B155C_K2_RERANK.zip /data /out val
```

`/path/to/starter_kit/data` is the organiser data root (`annotations/task2/val/{gt,ids,queries}.csv` and the
image folders `Market1501/`, `PA100k/`, `PETA/`) downloaded with the official starter kit. Outputs:
`out/distances.npy` (Q×G float32), `out/report.json` (load/inference seconds, output hash) and `out/metrics.json`.
Any other folder in the same starter-kit format can be passed as the 4th argument (relative to `/data/annotations/task2/`).

Expected result on the public validation split (3,462 queries × 33,407 images), from the same execution path:
see [`weights/README.md`](weights/README.md#reference-run). The final models were trained on train+validation, so
these validation numbers are an execution check, not an accuracy estimate.

GPU note: torch 2.4.1+cu121 supports Ampere to Hopper GPUs (the official runs used an A10). For Blackwell GPUs use
an environment with torch 2.8 + CUDA 12.8 (for example the training image with this repository mounted and
`--entrypoint bash`) and the command under "Without Docker" below.

Without Docker (Python 3.11 + torch 2.4.1 + numpy + Pillow):

```bash
PYTHONPATH=$PWD/verification_materials/code/src bash docker/run_inference.sh \
  weights/B156C_B155C_K2_RERANK.zip /path/to/starter_kit/data out val
```

The Codabench `run.py` enforces the official 300-second policy for 29,248 images; on a slower or shared GPU set
`UPAR_TIME_SCALE=4` (relaxes only the internal deadline checks, not the computation).

## 3. Training with Docker

```bash
docker build -f docker/Dockerfile.train -t upar2027-track2-train .
docker run --rm --gpus all --shm-size 16g \
  -v /path/to/starter_kit/data:/data:ro -v /path/to/pretrained:/pretrained:ro -v "$PWD/work:/work" \
  upar2027-track2-train /data /pretrained /work
```

`train_all.sh` builds the 256×128 image cache, rewrites only the cache/output paths of the 13 original recipe
configs (`verification_materials/code/scripts/prepare_delivery_training.py`) and trains them one after another with
`python -m upar.h1_finetune train --config <recipe>.json` (our runs took 3–7 GPU-hours per recipe on one large GPU, bf16).
`/pretrained` must contain the three generic checkpoints as `_h1/<timm identifier>/model.safetensors`
(identifiers and reference SHA-256 in `verification_materials/README.md`, section "Training data, pretraining and
recipes"); they are not redistributed here because of their licences. Weight soups, head soups and packaging into
the Codabench ZIP are documented in `verification_materials/README.md` ("Assembly"), including
which historical intermediates no longer exist. GPU bf16 training is not bit-deterministic.

## 4. Verify the delivery offline

```bash
cp weights/B156C_B155C_K2_RERANK.zip verification_materials/
cd verification_materials
python3 verify_delivery_artifact.py --manifest delivery_manifest.json \
  --model B156C_B155C_K2_RERANK.zip --files FILE_MANIFEST.json
# -> {"status": "PASS_BYTES_CRC_SYNTAX_CONTRACT", ..., "verified_delivery_files": 450}
```

## Notes

- Only the organiser Track 2 train/validation data were used for task-specific training; no test images.
- Inference uses one uniform pipeline for every gallery image; re-ranking uses only the unlabelled gallery of the
  current run and processes each query without statistics from other queries.
- The fact sheet discloses how the per-dataset scoring feedback of our own submissions was used during development.
- Third-party model licences (including the DINOv3 licence) are bundled inside the model ZIP (`licenses/`).
