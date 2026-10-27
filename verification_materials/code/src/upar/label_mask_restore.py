"""Restore existing Market UpperOther positives discarded by mask-v2.

This changes supervision availability, not labels or their meaning. Call with
actual source-fit rows only. It adds no inference graph or new observations.
The baseline mask-v2 remains unchanged for every other row/attribute cell.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

RECIPE = "v2_restore_market_upper_other_v1"
ATTRIBUTE = "UpperBody-Color-Other"


def validate_config(cfg: Mapping) -> None:
    """Keep the first trial isolated from other objective/initialization changes."""
    if cfg.get("label_mask") != RECIPE:
        raise ValueError(f"unknown restoration recipe: {cfg.get('label_mask')}")
    for name in ("rank_alpha", "partswap_p", "mixup_p", "label_smoothing", "focal_gamma", "kd_alpha"):
        if float(cfg.get(name, 0.0)) != 0.0:
            raise ValueError(f"{RECIPE} does not support {name}")
    for name in ("coarse_positive", "online_teachers", "kd_teacher_logits", "init_from"):
        if cfg.get(name):
            raise ValueError(f"{RECIPE} does not support {name}")


def restore_market_upper_other(
    base_v2_mask: np.ndarray,
    fit_labels: np.ndarray,
    fit_sources: np.ndarray,
    names: Sequence[str],
) -> np.ndarray:
    """Copy mask-v2 and restore its one bit for observed Market-positive rows.

    No statistics, sampling, fitting, label rewriting or RNG use occurs here.
    The caller subsequently recomputes its ordinary pos_weight, prior and BCE
    denominator from this restored mask. Non-fit rows must not be passed in.
    """
    mask, labels, sources = np.asarray(base_v2_mask), np.asarray(fit_labels), np.asarray(fit_sources)
    if (labels.ndim != 2 or sources.ndim != 1 or labels.shape != mask.shape
            or labels.shape != (len(sources), len(names)) or len(set(names)) != len(names)):
        raise ValueError("mask, source-fit labels, sources and unique attribute names must align")
    if ATTRIBUTE not in names:
        raise ValueError(f"missing required attribute: {ATTRIBUTE}")
    if not np.isin(labels, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("source-fit labels and mask-v2 must be binary and finite")
    index = list(names).index(ATTRIBUTE)
    active = (sources == "Market1501") & (labels[:, index] == 1)
    restored = mask.copy()
    restored[active, index] = 1
    return restored
