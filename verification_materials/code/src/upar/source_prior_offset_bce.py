"""One train-only source-prior BCE contrast; not integrated into a trainer.

Only Bag and NormalGlasses have source offsets. Canonical inference stays
sigmoid(raw_logit - log(original_global_pos_weight)). No target/source inputs
are added to the model. Conditional-shift/annotation differences are not fixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RECIPE = "source_prior_offset_bag_normal_v1"
SOURCES = ("Market1501", "PA100k", "PETA")
ATTRIBUTES = ("Accessory-Bag", "Accessory-Glasses-Normal")
WIDTH = 40


def validate_config(config: dict[str, Any]) -> None:
    if config.get("source_prior_offset_bce") != RECIPE:
        raise ValueError("Exact fixed source-prior-offset recipe required")
    for key in (
        "label_mask",
        "coarse_positive",
        "bce_profile_balance",
        "rank_group",
        "rank_query_balance",
        "online_teachers",
        "kd_teacher_logits",
        "coarse_clip_teacher",
    ):
        if config.get(key):
            raise ValueError(f"Single source-prior intervention cannot mix {key}")
    for key in ("rank_alpha", "partswap_p", "mixup_p", "label_smoothing", "focal_gamma", "kd_alpha"):
        if float(config.get(key, 0.0)) != 0:
            raise ValueError(f"Single source-prior intervention cannot mix {key}")
    if config.get("sampling") not in (None, "replace"):
        raise ValueError("Preserve original equal-source replacement sampler")
    if config.get("pos_weight_cap") != 10.0:
        raise ValueError("Original cap10 global pos_weight required")


def _names(attribute_names):
    names = tuple(attribute_names)
    if len(names) != WIDTH or len(set(names)) != WIDTH or any(not isinstance(n, str) for n in names):
        raise ValueError("Exact unique40 attribute names required")
    if any(n not in names for n in ATTRIBUTES):
        raise ValueError("Missing fixed selected attribute")
    return names, tuple(names.index(n) for n in ATTRIBUTES)


def _boundary(sources, split_roles, *, fold, include_val):
    src, role = np.asarray(sources), np.asarray(split_roles)
    if src.ndim != 1 or not len(src) or role.shape != src.shape:
        raise ValueError("Aligned nonempty source and role vectors required")
    if fold not in (*SOURCES, "all") or not isinstance(include_val, bool):
        raise ValueError("Explicit valid fold/include_val boundary required")
    expected = tuple(s for s in SOURCES if fold == "all" or s != fold)
    if set(src.tolist()) != set(expected):
        raise ValueError("Fit sources missing, unknown, or include held source")
    if fold != "all" and include_val:
        raise ValueError("LODO cannot fit any validation rows")
    allowed = {"train", "val"} if fold == "all" and include_val else {"train"}
    if not set(role.tolist()) <= allowed:
        raise ValueError("Unexpected split role or target/test data")
    if any("train" not in role[src == s] for s in expected):
        raise ValueError("Every fit source must have explicit training rows")
    return src, role, expected


def _logit(p):
    return np.log(p) - np.log1p(-p)


def fit_source_prior_offsets(
    labels,
    sources,
    split_roles,
    attribute_names,
    *,
    fold: str,
    include_val: bool,
    reference_prior,
    pos_weight,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute from already-selected fit rows only; no implicit row filtering.

    The supplied original canonical prior and global weight are checked, not
    recalibrated. Only the two selected source priors must be strictly interior;
    absent classes in the other38 attributes remain unmodified, with offset0.
    This function is pure: it reads no files, touches no RNG, and writes nothing.
    """
    y = np.asarray(labels)
    names, selected = _names(attribute_names)
    src, role, expected = _boundary(sources, split_roles, fold=fold, include_val=include_val)
    if y.shape != (len(src), WIDTH) or not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError("Complete finite binary fit labels Nx40 required")
    reference = np.asarray(reference_prior, dtype=np.float64)
    weight = np.asarray(pos_weight)
    if reference.shape != (WIDTH,) or not np.isfinite(reference).all() or np.any((reference < 0) | (reference > 1)):
        raise ValueError("Original canonical40 prior must be finite in[0,1]")
    if weight.shape != (WIDTH,) or weight.dtype != np.float32 or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("Original global positive float32 weight vector40 required")
    priors = np.stack([y[src == s].mean(0, dtype=np.float64) for s in expected])
    reference_from_fit = priors.mean(0)
    if not np.allclose(reference, reference_from_fit, rtol=0, atol=1e-12):
        raise ValueError("Supplied canonical prior is not equal-source fit-row prior")
    expected_weight = np.clip((1 - reference) / np.maximum(reference, 1e-6), 1.0, 10.0).astype(np.float32)
    if not np.array_equal(weight, expected_weight):
        raise ValueError("Global pos_weight differs from original cap10 prior convention")
    target_priors = priors[:, selected]
    if np.any((target_priors <= 0) | (target_priors >= 1)):
        raise ValueError("Selected source prior is0/1; no silent clipping or smoothing")
    selected_reference = reference[list(selected)]
    if np.any((selected_reference <= 0) | (selected_reference >= 1)):
        raise ValueError("Selected canonical prior is0/1")
    table = np.zeros((len(expected), WIDTH), dtype=np.float32)
    table[:, selected] = (_logit(target_priors) - _logit(selected_reference)).astype(np.float32)
    report: dict[str, Any] = {
        "recipe": RECIPE,
        "fold": fold,
        "include_val": include_val,
        "fit_rows": len(y),
        "attribute_names": list(names),
        "selected_attributes": list(ATTRIBUTES),
        "selected_indices": list(selected),
        "fit_sources": list(expected),
        "source_code_vocabulary": list(SOURCES),
        "canonical_prior": reference.tolist(),
        "global_pos_weight": weight.tolist(),
        "pos_weight_cap": 10.0,
        "normalization": "Original full optimizer-batch N*40; no realized source/weight-mass renormalization",
        "inference": "Unchanged sigmoid(raw_logit-log(global_pos_weight)); no source offset at inference",
        "source": {
            s: {
                "rows": int((src == s).sum()),
                "rows_by_split": {
                    r: int(((src == s) & (role == r)).sum()) for r in sorted(set(role[src == s].tolist()))
                },
                "prior": priors[i].tolist(),
                "offset_float32": table[i].tolist(),
            }
            for i, s in enumerate(expected)
        },
        "scope": "Explicit fit rows only. Labels/selection provenance remains caller responsibility; names or split claims cannot certify upstream encoder lineage.",
        "population_assumption": "Shared class-conditional likelihood ratio per selected attribute; unproved in UPAR",
        "new_inference_parameters": 0,
    }
    codes = np.asarray([SOURCES.index(s) for s in src], dtype=np.int64)
    return offsets_for_source_codes(codes, report), report


def offsets_for_source_codes(source_codes, metadata: dict[str, Any]) -> np.ndarray:
    """Map canonical source codes to immutable-fit offsets; never infer a source.

    Unknown or held-out source codes fail even if all requested offsets would
    happen to be0. This is a training-only mapping, not a deployment operation.
    """
    codes = np.asarray(source_codes)
    if codes.ndim != 1 or not len(codes) or not np.issubdtype(codes.dtype, np.integer) or codes.dtype == np.bool_:
        raise ValueError("Nonempty integer source code vector required")
    if np.any((codes < 0) | (codes >= len(SOURCES))):
        raise ValueError("Unknown source code")
    _, selected = _names(metadata.get("attribute_names", ()))
    expected = tuple(s for s in SOURCES if metadata.get("fold") == "all" or s != metadata.get("fold"))
    if (
        metadata.get("recipe") != RECIPE
        or metadata.get("fold") not in (*SOURCES, "all")
        or metadata.get("source_code_vocabulary") != list(SOURCES)
        or metadata.get("fit_sources") != list(expected)
        or metadata.get("selected_attributes") != list(ATTRIBUTES)
        or metadata.get("selected_indices") != list(selected)
        or type(metadata.get("include_val")) is not bool
        or (metadata["fold"] != "all" and metadata["include_val"])
        or set(metadata.get("source", {})) != set(expected)
    ):
        raise ValueError("Fit metadata recipe, boundary or attribute identity differs")
    requested = {SOURCES[int(c)] for c in codes}
    if not requested <= set(expected):
        raise ValueError("Held source cannot receive a fitted offset")
    table = np.zeros((len(SOURCES), WIDTH), dtype=np.float32)
    reference = np.asarray(metadata.get("canonical_prior"), dtype=np.float64)
    if reference.shape != (WIDTH,) or not np.isfinite(reference).all() or np.any((reference < 0) | (reference > 1)):
        raise ValueError("Invalid fit reference prior metadata")
    original_weight = np.asarray(metadata.get("global_pos_weight"), dtype=np.float32)
    expected_weight = np.clip((1 - reference) / np.maximum(reference, 1e-6), 1.0, 10.0).astype(np.float32)
    if metadata.get("pos_weight_cap") != 10.0 or not np.array_equal(original_weight, expected_weight):
        raise ValueError("Original global weight metadata differs")
    source_priors, total_rows = [], 0
    for source in expected:
        record = metadata["source"][source]
        roles = record.get("rows_by_split", {})
        allowed = {"train", "val"} if metadata["fold"] == "all" and metadata["include_val"] else {"train"}
        if (
            type(record.get("rows")) is not int
            or record["rows"] <= 0
            or "train" not in roles
            or not set(roles) <= allowed
            or any(type(n) is not int or n <= 0 for n in roles.values())
            or sum(roles.values()) != record["rows"]
        ):
            raise ValueError("Invalid source fit row/split metadata")
        total_rows += record["rows"]
        prior = np.asarray(record["prior"], dtype=np.float64)
        values = np.asarray(record["offset_float32"], dtype=np.float32)
        if (
            prior.shape != (WIDTH,)
            or values.shape != (WIDTH,)
            or not np.isfinite(prior).all()
            or not np.isfinite(values).all()
            or np.any((prior < 0) | (prior > 1))
            or np.any((prior[list(selected)] <= 0) | (prior[list(selected)] >= 1))
            or np.any((reference[list(selected)] <= 0) | (reference[list(selected)] >= 1))
        ):
            raise ValueError("Invalid fixed source prior/offset metadata")
        expected_values = np.zeros(WIDTH, dtype=np.float32)
        expected_values[list(selected)] = (_logit(prior[list(selected)]) - _logit(reference[list(selected)])).astype(
            np.float32
        )
        if not np.array_equal(values, expected_values):
            raise ValueError("Offsets differ from fixed source-prior formula")
        table[SOURCES.index(source)] = values
        source_priors.append(prior)
    if total_rows != metadata.get("fit_rows") or not np.allclose(
        np.mean(source_priors, axis=0), reference, rtol=0, atol=1e-12
    ):
        raise ValueError("Fit row count or equal-source canonical prior differs")
    return table[codes].copy()


def source_prior_bce_part(
    logits,
    target,
    pos_weight,
    offsets,
    full_batch_denominator: int,
    *,
    attribute_names,
    enabled: bool = True,
):
    """Original BCE normalization with one fixed per-row loss-logit offset.

    Strict value validation is intentional for this pure reference module; its
    sync/runtime cost must be reviewed before eventual trainer integration.
    No training entrypoint, dataset access, optional HPO knobs or hidden state.
    """
    import torch
    import torch.nn.functional as functional

    _, selected = _names(attribute_names)
    if (
        logits.ndim != 2
        or logits.shape[1] != WIDTH
        or logits.numel() == 0
        or target.shape != logits.shape
        or offsets.shape != logits.shape
        or pos_weight.shape != (WIDTH,)
    ):
        raise ValueError("Aligned nonempty logits/targets/offsets Nx40 and global weight40 required")
    if (
        type(full_batch_denominator) is not int
        or full_batch_denominator < logits.numel()
        or full_batch_denominator % WIDTH
        or type(enabled) is not bool
    ):
        raise ValueError("Original full-batch N*40 and boolean enabled required")
    if any(t.device != logits.device for t in (target, pos_weight, offsets)):
        raise ValueError("All tensors must already be on the same device")
    if not torch.is_floating_point(logits) or pos_weight.dtype != torch.float32 or offsets.dtype != torch.float32:
        raise ValueError("Floating logits and original float32 global weights/offsets required")
    if any(t.requires_grad for t in (target, pos_weight, offsets)):
        raise ValueError("Targets, global weights and source offsets must be fixed")
    if any(not bool(torch.isfinite(t).all()) for t in (logits, target, pos_weight, offsets)):
        raise ValueError("Nonfinite loss input")
    if not bool(((target == 0) | (target == 1)).all()) or not bool(((pos_weight >= 1) & (pos_weight <= 10)).all()):
        raise ValueError("Binary targets and original cap10 global weights required")
    unselected = [i for i in range(WIDTH) if i not in selected]
    if bool((offsets[:, unselected] != 0).any()):
        raise ValueError("Other38 attributes must have exact zero offset")
    z = logits.float() + offsets.float() if enabled else logits.float()
    # Same mean()*microbatch_share expression as existing unmasked H1 control.
    return functional.binary_cross_entropy_with_logits(
        z, target.float(), pos_weight=pos_weight.float(), reduction="mean"
    ) * (logits.numel() / full_batch_denominator)
