# Final-round rerank search vs B153C (2026-10-26 20:41–22:40 JST)

Saved by the main worker from the search worker's hand-back. Full per-config table: `results_all.tsv` (246 rows).

## Decision

B153C stays the final-round candidate. One configuration, `h_m30_c7_b06_balpha36`, passes the pre-registered bar
(`prereg_r7.txt`), but its expected Public effect is about 0–1 bp. It needs a module change (`rerank_param6_cand.py`)
and adds about 1.5–3 s of rerank time, measured on a shared GPU, not on an A10.

The candidate raises `body_alpha` from 0.33 to 0.36 and adds an extra soft-query term. The term is gated per query by
the expected exact-match count E_q, computed from the gallery probabilities without ground truth.

| Check | val | train |
|---|---|---|
| hi15 (bps) | +0.42 | +1.93 |
| overall (bps) | +1.65 | +1.32 |
| lo1_4 (bps) | +2.4 | +1.4 |
| paired bootstrap, overall | [+0.83, +2.58] | [+0.5, +2.2] |
| paired bootstrap, hi15 | [−0.53, +1.44] | [+0.9, +3.1] |
| worst source × multiplicity cell (bps) | −1.9 | −0.4 |

Public-structure mimic: PUB +0.1 bps (MEV +1.1, SYN −0.9).

Train is not fully independent: round 7 was designed after train had shown that both parts help.

## The three suggested ideas were all negative

- **Distance fusion of local optima.** Results fall on the line between the fused parts, with no complementarity bonus. Fusion also needs 2–3 rerank passes, which the runtime budget does not allow.
- **Attribute weights ×1.1 or ÷1.1, selected on val only.** The pre-registered pick `awn_top5_a20d12` scored val +18.5/+4.9 but failed on train: PETA cells −8.7, −10.4 and −6.0, and the bootstrap interval includes zero. These weight changes trade Market/PA100k against PETA.
- **APRF variants.** They are within ±1 bp or trade hi15 against overall. Prototypes built from the visual anchors lose 3–7 bps.

## Knob combinations

Combinations reached +7 to +9 bps overall on both surfaces (`k_a00d_a38u_a16d11`, `k_step10_a00d_a38u`). All of them fail the PETA 5–14-match cell: −12 to −18 bps.

## Notes

- `eval_ctx_3462_33407_489e9e2af5d709bd.npy` was moved to the archive on 10/24. The main worker restored it on 10/26 at 22:45 (byte-identical), so `combo_sweep.py val` works again.
- Pre-registered selection rules: `prereg_idea2.txt`, `prereg_r6.txt`, `prereg_r7.txt`.
- Adoption-bar check: `fr_gate.py`. Candidate reference module: `rerank_param6_cand.py`. Its outputs match the research module exactly (max |Δ| = 0), and its B153C path is identical.
