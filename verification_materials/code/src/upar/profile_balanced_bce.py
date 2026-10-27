"""Bounded semantic-profile emphasis while retaining source/attribute/bit mass.

This is NOT an exact mADM surrogate or conditional-posterior correction. Fit
only on the explicit training rows of the corresponding encoder/head fold.
No sampling, labels, pos_weight, prior, denominator, or RNG is changed here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

RECIPE = "source_attr_bit_mass_half_v1"
SOURCES = ("Market1501", "PA100k", "PETA")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("bce_profile_balance") != RECIPE:
        raise ValueError("unknown profile-BCE recipe")
    if config.get("label_mask") or config.get("coarse_positive"):
        raise ValueError("first profile-BCE recipe requires ordinary unmasked labels")
    for key in ("rank_alpha", "partswap_p", "mixup_p", "label_smoothing", "focal_gamma", "kd_alpha"):
        if float(config.get(key, 0.0)) != 0:
            raise ValueError(f"profile-BCE contrast cannot mix {key}")
    for key in ("online_teachers", "kd_teacher_logits", "coarse_clip_teacher"):
        if config.get(key):
            raise ValueError(f"profile-BCE contrast cannot use {key}")
    if config.get("sampling") not in (None, "replace"):
        raise ValueError("preserve ordinary replacement sampler")
    if config.get("rank_group") or config.get("rank_query_balance"):
        raise ValueError("first contrast has no rank grouping or rank weights")


def _weight_stats(weights: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(weights, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "min": None, "p99": None, "max": None, "ess": None, "ess_fraction": None}
    ess = float(values.sum() ** 2 / np.square(values).sum())
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "ess": ess,
        "ess_fraction": ess / len(values),
    }


def fit_cell_weights(
    labels: np.ndarray,
    sources: np.ndarray,
    split_roles: np.ndarray,
    *,
    fold: str,
    include_val: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return aligned FP32 N x A weights from *already selected fit rows only*.

    For source s, profile q and attribute a with observed bit b:
      r[i,a] = N[s,a,b] / (K[s,a,b] * count[s,q_i]); w = (1+r)/2.
    N counts images; K counts distinct complete binary profiles. Therefore
    sum(w[i,a] over source s and bit b) = N[s,a,b] in exact arithmetic.
    Missing classes are never divided by zero and produce an empty report.
    The output does not include held-source or validation rows for a LODO fold.
    """
    y, src, role = np.asarray(labels), np.asarray(sources), np.asarray(split_roles)
    if y.ndim != 2 or y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("nonempty fit label matrix required")
    if src.shape != (len(y),) or role.shape != (len(y),):
        raise ValueError("aligned fit source and split vectors required")
    if not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError("complete binary fit labels required")
    if not isinstance(include_val, bool) or fold not in (*SOURCES, "all"):
        raise ValueError("explicit valid fold/include_val boundary required")
    expected = set(SOURCES) if fold == "all" else set(SOURCES) - {fold}
    if set(src.tolist()) != expected:
        raise ValueError("fit sources differ or contain the held source")
    allowed_roles = {"train", "val"} if fold == "all" and include_val else {"train"}
    if fold != "all" and include_val:
        raise ValueError("LODO cannot include validation rows")
    if not set(role.tolist()) <= allowed_roles or "train" not in role:
        raise ValueError("fit split boundary differs")
    binary = y.astype(np.uint8, copy=True)
    weights = np.ones(y.shape, dtype=np.float32)
    report: dict[str, Any] = {
        "recipe": RECIPE,
        "fold": fold,
        "include_val": include_val,
        "fit_sources": sorted(expected),
        "fit_rows": len(y),
        "attributes": y.shape[1],
        "mixture": 0.5,
        "source": {},
        "normalization": "unchanged full-batch N*A; not realized weight sum",
        "prior_pos_weight": "original source-balanced image prior and pos_weight retained",
        "conditional_probability_claim": False,
    }
    for source in SOURCES:
        selected = np.flatnonzero(src == source)
        if not len(selected):
            continue
        ys = binary[selected]
        profiles, inverse, counts = np.unique(ys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        local = np.ones(ys.shape, dtype=np.float64)
        cells = []
        max_error64, max_error32 = 0.0, 0.0
        for attr in range(ys.shape[1]):
            by_bit = {}
            for bit in (0, 1):
                observed = ys[:, attr] == bit
                n = int(observed.sum())
                k = int((profiles[:, attr] == bit).sum())
                if n:
                    if k == 0:
                        raise RuntimeError("nonempty class has zero profiles")
                    local[observed, attr] = 0.5 + 0.5 * n / (k * counts[inverse[observed]])
                elif k:
                    raise RuntimeError("empty class has profiles")
                values = local[observed, attr]
                total64 = float(values.sum())
                total32 = float(values.astype(np.float32).sum(dtype=np.float64))
                error64 = abs(total64 - n) / max(n, 1)
                error32 = abs(total32 - n) / max(n, 1)
                max_error64 = max(max_error64, error64)
                max_error32 = max(max_error32, error32)
                by_bit[str(bit)] = {
                    "images": n,
                    "distinct_profiles": k,
                    "weights": _weight_stats(values.astype(np.float32)),
                    "sum_float64": total64,
                    "sum_stored_float32": total32,
                    "relative_mass_error_float64": error64,
                    "relative_mass_error_float32": error32,
                }
            cells.append(by_bit)
        if max_error64 > 1e-12 or max_error32 > 1e-6:
            raise RuntimeError("source/attribute/bit mass preservation failed")
        weights[selected] = local.astype(np.float32)
        report["source"][source] = {
            "images": len(selected),
            "distinct_profiles": len(profiles),
            "singleton_profiles": int((counts == 1).sum()),
            "max_profile_count": int(counts.max()),
            "cells_by_attribute_index": cells,
            "all_cells": _weight_stats(weights[selected].reshape(-1)),
            "max_relative_mass_error_float64": max_error64,
            "max_relative_mass_error_float32": max_error32,
        }
    if not np.isfinite(weights).all() or np.any(weights < 0.5):
        raise RuntimeError("invalid bounded weights")
    return weights, report


def weighted_bce_part(logits, target, pos_weight, cell_weights, full_batch_denominator: int):
    """Microbatch numerator / unchanged full optimizer-batch N*A denominator.

    Inputs must already be aligned and on the same device. The fit helper
    validates finite nonnegative weights once, avoiding per-step GPU syncs.
    BF16 model output is converted to FP32 as in the existing H1 BCE path.
    """
    import torch.nn.functional as functional

    if logits.ndim != 2 or logits.numel() == 0 or target.shape != logits.shape or cell_weights.shape != logits.shape:
        raise ValueError("aligned logits, targets and cell weights required")
    if pos_weight.shape != (logits.shape[1],):
        raise ValueError("aligned attribute positive weights required")
    if (
        not isinstance(full_batch_denominator, int)
        or isinstance(full_batch_denominator, bool)
        or not math.isfinite(full_batch_denominator)
        or full_batch_denominator <= 0
        or full_batch_denominator < logits.numel()
    ):
        raise ValueError("positive full-batch N*A denominator required")
    per = functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), pos_weight=pos_weight.float(), reduction="none"
    )
    return (per * cell_weights.float()).sum() / full_batch_denominator
