# UPAR Track 2 — inventory of tried and untried levers (2026-10-27 05:00 JST)

Built from three read-only surveys (member level, scoring/re-ranking level, literature) plus today's checks.
Baseline: B153C (val OOF 0.67794 / train source-held-out 0.67356; estimate Public ~0.7043).
Deltas are in bps of mADM, given as val / train. hi15 = queries with at least 15 matches (MEVID proxy, about 0.9:1 to
official MEVID). lo1_4 = queries with 1–4 matches (Synth-like; Synth itself is not predictable locally).

## Already tried (saturated or negative)

| Level | Family | Outcome |
|---|---|---|
| Members | grouped member-weight shifts (±0.05) | −11…+1.5; HP +0.10 adopted in B145C |
| Members | head soups / column-selective heads (46 configs) | none passed (10/26) |
| Members | logit pooling; global temperature T 0.9 / 1.1 | −5…−55; −3…−14 |
| Members | flip TTA, crop TTA, 4th member, H+ at 224 | no gain; runtime fails |
| Members | label-shift EM (Saerens) | −15…−268 |
| Scoring | attribute weights (groups, then ×1.2 / ÷1.2 singles) | adopted (B151C–B153C); ×1.1 round failed PETA cells |
| Scoring | λ / entropy / err_gamma per stream; global pos_q_mul | ±2 mixed; l1-weighted body stream adopted |
| Re-rank | diffusion strength, density-adaptive, extra graphs | MEVID/Synth trade-off, kept at balance |
| Re-rank | near-duplicate diffusion | adopted, saturated |
| Re-rank | APRF and ~10 variants; E_q gating | adopted m100 β.08; variants ±2; gate worth 0–1 bp |
| Re-rank | distance fusion; knob combinations | no complementarity; PETA 5–14 cell −12…−18 |
| Re-rank | categorical NLL, coherent member ranker, renorm of age/hair/LB colour | negative |

## Untried before today, ranked by expected value

1. **Per-attribute, per-member logit calibration** (temperature / bias / Platt, fitted on OOF).
   - The organisers' own study (Specker thesis §7.2.1) reports it as the largest inference-time gain.
   - Not tried on the current ensemble. Only global versions were tested; the old ResNet B013 version gave −19.
2. **Per-attribute member weights.** Only grouped weights have been tuned. Config-only.
3. **DBA (database-side feature augmentation) for the graph features.** Implemented in the deployed module but never evaluated. Config-only (K).
4. **LowerBody-Type renormalisation 0.5.** Train +2.7 on B152C; val never run. Config-only.
5. **Per-attribute asymmetric query weights** (w1 for q=1, w0 for q=0). Needs a scorer change.
6. **member_std as a distance weight; k-reciprocal overlap of final rankings; metric-aware stream.** Need code changes; low prior.
7. **PE / DINOv3 backbone-soup weights.**
   - Never re-tuned under the B153C scorer.
   - Needs GPU extraction plus a model repack: PE also needs its feature head, so it is costly; the DINOv3 prior is weak.
8. **EMA decay or checkpoint averaging.** Impossible: no raw or intermediate weights survive.

## Today's checks (rules fixed before results: prereg_cal1.txt, prereg_b2_mw1.txt)

- **C1 calibration**, cross-fitted val↔train:
  - Temperature is negative: train hi15 −96 at full strength, about −15 at half.
  - Bias raises hi15 strongly on both surfaces (+26…+41), but lowers lo1_4 (−12…−94) and the overall score.
  - Result: none passes the pre-registered rule, so C1 is negative. The hi15 gain is the largest single effect seen in days, but it trades away the Synth-like queries.
- **C3**, leave-one-source-out cross-fit (the real domain-transfer test): every source and multiplicity cell drops by −100…−320.
  - Calibration does not transfer to an unseen source.
  - So the C1 hi15 gain came from the sources the two surfaces share. The family is rejected.
- **C2**, weaker or single-member bias:
  - The val top-1 is platt_s025 (val +4.0).
  - It fails on train: −6.2 overall, PETA cells −35 (gate_c2_platt_s025.txt).
- **B2**:
  - DBA n3/5/10: −11…−64.
  - LB-type renorm: −0.5 / −0.7.
  - Both rejected.
- **EVA-3**, rank-loss EVA as a third soup ingredient:
  - All variants give val −2.6…−5.4; only train hi15 rises (+2…+4).
  - They fail the pre-registered val filter, so B154C is dropped and EW1 is not run.
- **MW1**, 80 per-attribute PE↔H+ moves:
  - 4 singles pass the both-surface rule: a13, a14, a24, a38.
  - Their combination (gate_mw1.txt):

    | Surface | overall | hi15 |
    |---|---|---|
    | val | +1.6 | +1.0 |
    | train | +2.5 | +1.1 |

  - Min train cell −0.6, train CI [+1.3, +3.8], PUB +0.07, so the gate passes.
  - Built as **B155C** (config-only), under verification.
  - Public contribution after 0.6 shrinkage is about +0.3 bps.
- **MW2**, 80 per-attribute EVA↔H+ moves on top of MW1 (prereg_mw2.txt): running.
