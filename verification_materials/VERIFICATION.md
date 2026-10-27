# Verification scope

This is the delivery for our final submission **B156C** (Codabench submission **1011976**).

- The exact model ZIP (1,333,500,417 bytes, all 32 members) passes `verify_delivery_artifact.py`: SHA-256, ZIP CRC, Python syntax and the three-member CUDA contract. It differs from B153C only in `assets/config.json` (candidate name, four member-weight columns, three attribute-distance weights, descriptive knob block) and the single `DEFAULT_K.update` line of the re-ranking module (diffusion step 1.0 instead of 0.95); all network weights and code paths are byte-identical (`provenance/b156c/diff_vs_b153c.json`).
- Full local CUDA replay of the exact ZIP on the public validation split (3,462 queries × 33,407 images, 2026-10-27): every distance and ranking equals an independent reference implementation (`provenance/b156c/runtime_acceptance_b156c_v1.json`, `execute_report.json`, `reference_report.json`). Scoring that output with `docker/evaluate_distances.py` gives the in-sample numbers in `../weights/README.md`.
- Re-ranking time on the official test shape (893 queries × 29,248 images) with identical saved encoder outputs, run through each ZIP's real `rank_gallery`: B153C (same code as B156C) 16.3 s (two runs) versus B137C 15.4–17.9 s on the same machine (`provenance/b156c/*.r0.json`, `*.r1.json`, harness `time_rerank_testshape.py`). B137C's official A10 run took 258.3 s of 300 s.
- The shipped int8 weights of all three members were re-derived bit-exactly from the surviving fp32 EMA checkpoints (2026-10-22/23).
- CPU canary of the exact ZIP under torch 2.4.1+cu121 (`code/scripts/verify_delivery_canary.py`, record `provenance/b156c/canary_b156c.json`): each member loaded from the ZIP reproduces its packaged reference on four fixed synthetic inputs (max abs difference 2.98e-07, all finite).
- The delivered code tree passes its 324 CPU unit tests (no skips; GPU hidden).

These are packaging, equivalence and compatibility checks. They do not establish bit-exact retraining,
fresh-environment reproducibility, hidden-test performance or a future runtime guarantee. For checksums of the
complete code/document tree, use `FILE_MANIFEST.json`; the model ZIP is supplied separately (`../weights/README.md`).
