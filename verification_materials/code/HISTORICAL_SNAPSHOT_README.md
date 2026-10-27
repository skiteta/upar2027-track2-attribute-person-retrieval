# Code snapshot for UPAR 2027 Track 2 (taken for candidate B153C; the final submission B156C is built from it with `provenance/b156c/spec_b156c.json`)

Snapshot taken 2026-10-22 22:00 JST from the working repository. Every file was **copied** (content and mtime preserved), not
moved. Each copy was compared byte for byte with its source after copying. Sizes and SHA-256 values are in
`../manifest.json` and `../MANIFEST.md`.

## Layout
This directory is a subset of the repository root. Every file keeps its repository-relative path, because the scripts
resolve their inputs from the repository root. Some use `Path(__file__).parents[2]`; others use the absolute constant
`ROOT = /home/syouya/ml/lab/attribute_based_person_retrieval` (see "Absolute paths" below).

**To use it:** copy this directory to a new location and treat that location as the repository root. Then restore the
large external files (`../MANIFEST.md` §C) at the paths listed there.

| Path | What it is |
|---|---|
| `src/upar/` (40 modules) | Python package. These modules are the static import closure, computed with an `ast` walk over `import upar.*`, `from upar import ...`, run-dir sibling imports and `src/upar/*.py` paths loaded by file name. The entry points are the training, soup, packaging, replay, acceptance, local-evaluation and verification scripts listed below. All 40 are byte-identical to the live `src/upar/` of 2026-10-22. |
| `scripts/` | `train_python` (runs `.venv-train/bin/python` with `PYTHONPATH=src`), `uv` (workspace uv launcher), `make_pw_aligned.py`, `make_xres_member.py`, `prepare_hplus6.py` (the 6-run H+ soup `hp6_all`) and `h1_build_accept.sh` (the B119G graph build). |
| `tests/` (20 files) | Unit tests of closure modules that need no run artifacts. On 2026-10-22 they passed against this snapshot's `src/` (CPU only): 292 tests together with `test_b127_jaccard_prob_diffusion.py`. |
| `artifacts/runs/20261008T231500+0900_knob_sweep/` | `config.json` and `train_report.json` for the 13 all-trainval training runs and their `_ema0.999` directories. It also holds the 38 LODO fold runs that still have a config (`<recipe>_<fold>/`). |
| `artifacts/runs/20261017T2145+0900_b127_jaccard_prob_diffusion/` | Head and int8 pack scripts (`pack_hp4_2030_192_cpu.py`, `pack_b136c_*`, `pack_b137c_*`, `pack_b138c_*`), `build_b127c.py` (builds the B139C ZIP), the B139C re-ranking module `b127_jaccard_prob_diffusion.py` and `b121_fp32_cosine_graph.py`. Also B139C's replay and reference (`replay_b127c.py`, `reference_b127c.py`), its unit test, and the B139C build and acceptance records (`package_v14/complete.json`, `runtime_acceptance_v14.json`). `b137c_watch_v3/` holds the official ingestion and scoring logs of B137C. |
| `artifacts/runs/20261018T0300+0900_gpu_representation_and_soup_cycle/` | `build_and_extract_hp4_20_30_30_20.py` (member-1 backbone soup 0.2/0.3/0.3/0.2), plus the feature-extraction and evaluation helpers imported by the local evaluation surfaces. |
| `artifacts/runs/20261021T2250+0900_takeover_knob_search/` | The B153C step. `build_candidate.py` and `spec_b153c.json` build B153C from the B139C ZIP. The module base is `rerank_param4_frozen.py`. `make_rerank_param.py` and `rerank_param.py` are the parametrised copy of the deployed module (`rerank_param.py` is imported by `knob_eval.py`). `rerank_param2.py` is the later working copy. Local CUDA replay, bit-exact reference and acceptance: `replay_candidate.py`, `reference_candidate.py`, `verify_chain.sh`. Local evaluation surfaces: `knob_eval.py`, `mw_sweep.py`, `combo_sweep.py`, `pubmimic.py`, `pubmimic_cs.py`, `source_bins.py`, `build_inputs_v1.py`, `train_surface/build_train_surface.py`. Metric check: `metric_check/`. The directory also holds the selection configs `cfg_combo_p{a,c,d,f,g,h}_v1.json` and `cfg_pubmimic_cs_v1.json` with their result lines `res_*.jsonl`. |
| `artifacts/runs/20261021T2250+0900_takeover_knob_search/build_b153c_v1/` | `complete.json`, the B153C build record with the per-member SHA-256 proof. `package/` holds **only the 13 text members** of the B153C ZIP (`run.py`, `b121_*.py`, `b122c_constant.py`, `b127_jaccard_prob_diffusion.py`, `metadata.yaml`, `assets/config.json`). Each one equals its ZIP member, checked on 2026-10-22. To get the full package, unzip the B153C ZIP over it; the bytes are identical. |
| `artifacts/runs/20261021T2250+0900_takeover_knob_search/replay_b153c_v1/`, `runtime_acceptance_b153c_v1.json`, `acceptance_b153c_v1.out`, `equiv_v1_rerank_param4_frozen.json` | Records of the local full CUDA replay of the exact B153C ZIP, the bit-exact reference and the acceptance. The large `.npy`/`.npz` captures are not included. |
| `artifacts/runs/20261022T0400+0900_deliverables_prep/` | `verify_weight_chain_v1.py` and `verify_weight_chain_fastpath_v1.py`, which re-derive the shipped int8 weights on CPU from the fp32 EMA checkpoints, with their JSON results. Also the SHA-256 list of the uncached inputs. |
| `artifacts/runs/20261017T1129…`, `…1216…`, `…1422…`, `…1615+0900_b124_alpha05/` | Runtime lineage B121G → CUDA32 → B122C → B123C → B124C: the adapter scripts and package-local module sources, `recipe_v1.json`, and B124C's `package_v1/complete.json`. **Documentation only.** Their parent ZIPs were deleted, so this chain cannot be re-run. The resulting files ship verbatim in the B153C ZIP. `build_b127c.py` needs B124C's `complete.json` and its extracted `package/` directory, which is shipped separately. |
| `artifacts/runs/20261003T…_b011_…/reliability_search.json`, `submissions/B022_…/assets/config.json`, `artifacts/runs/20261012T0200+0900_b110g/{plans,member_weights}.json` | Fixed inputs read by `upar.h1_package` (attribute distance weights, fallback metadata, plan and member weights of the graph build). |
| `artifacts/runs/…b116g_pe/…`, `…b115g/…`, `…b113g/…`, `…b119g/…`, `…hplus6/…` | `config.json` and `train_report.json` of the fp32 soup directories, whose weights were deleted and are re-derivable. Also the `hp6_all` soup record (`weights_ready.json`, `protocol_v1.json`, `parent_manifest_v1.json`) and the B119G package manifest. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Environment of the submission-parity `.venv` (torch 2.4.1+cu121). The training environment `.venv-train` has no lock file; see `../REPRODUCE.md` §2. |
| `starter_kit/{download_datasets.py,README.md,environment.yml}` | Organiser starter-kit files: the data download script. |

## Absolute paths (review before sharing)
Several run-dir scripts hard-code `ROOT = Path("/home/syouya/ml/lab/attribute_based_person_retrieval")`:
`knob_eval.py`, `mw_sweep.py`, `replay_candidate.py`, `reference_candidate.py`, `build_inputs_v1.py`,
`make_rerank_param.py`, `metric_check/compare_adm.py`, `train_surface/build_train_surface.py`,
`verify_weight_chain_v1.py`, `verify_chain.sh` and `scripts/h1_build_accept.sh` (`cd …`). `spec_b153c.json` also pins
the parent ZIP by absolute path (plus its SHA-256). To run these elsewhere, change that one constant or path. `../manifest.json`
flags every file that contains the home-directory path (`home_path: true`). The files are shipped unchanged so that their
hashes stay equal to the files that ran.

## What is not here
- Data, images, checkpoints, `.npz` and `.npy` arrays, ZIPs.
- `.secrets/` and the Codabench submission tools.
- Agent and workflow state (`.os_state`, `work_os/`).
- The earlier experiments in `src/upar` that this chain does not import (214 of the 254 modules).

The large artifacts needed to re-run the chain are listed with their sizes and SHA-256 in `../MANIFEST.md` §C.
