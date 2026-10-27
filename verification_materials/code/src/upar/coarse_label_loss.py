"""Conservative, positive-only Market colour supervision for the mask-v2 sibling.

The three candidate sets are an observation-model hypothesis, not an officially
guaranteed annotation mapping. An observed coarse positive supports the event
``A OR B``; an observed zero does NOT imply either fine attribute is negative.
Existing fine-label negatives, masks, pos_weight, priors and inference stay intact.
No Other/mixture or lower Yellow mapping is assumed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

RECIPE = "market_three_pairs_v1"
PAIR_NAMES = (
    ("UpperBody-Color-Red", "UpperBody-Color-Pink"),
    ("UpperBody-Color-Yellow", "UpperBody-Color-Orange"),
    ("LowerBody-Color-Pink", "LowerBody-Color-Red"),
)


def validate_config(cfg: Mapping) -> None:
    """Keep the first experiment one-factor and reject unsupported mixed labels."""
    if cfg.get("coarse_positive") != RECIPE:
        raise ValueError(f"unknown coarse_positive recipe: {cfg.get('coarse_positive')}")
    if cfg.get("label_mask") != "v2":
        raise ValueError("coarse_positive requires unchanged label_mask=v2")
    for name in ("rank_alpha", "partswap_p", "mixup_p", "label_smoothing", "focal_gamma", "kd_alpha"):
        if float(cfg.get(name, 0.0)) != 0.0:
            raise ValueError(f"coarse_positive v1 does not support {name}")
    for name in ("online_teachers", "kd_teacher_logits", "init_from"):
        if cfg.get(name):
            raise ValueError(f"coarse_positive v1 does not support {name}")


def fit_positive_events(labels: np.ndarray, sources: np.ndarray, names: Sequence[str]):
    """Only pass actual fit rows: no held-source labels/statistics are consumed.

    Returns vocabulary indices and a [fit_rows, 3] boolean activation mask.
    This function is deterministic, has no fitted parameters and consumes no RNG.
    """
    labels, sources = np.asarray(labels), np.asarray(sources)
    if labels.ndim != 2 or labels.shape != (len(sources), len(names)) or len(set(names)) != len(names):
        raise ValueError("fit labels, sources and unique attribute names must align")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("coarse-positive fit labels must be binary")
    index = {name: i for i, name in enumerate(names)}
    pairs = tuple((index[a], index[b]) for a, b in PAIR_NAMES)
    market = sources == "Market1501"
    if np.any(labels[market][:, [b for _, b in pairs]] != 0):
        raise ValueError("Market fine-B vocabulary changed; re-audit the observation model")
    active = market[:, None] & (labels[:, [a for a, _ in pairs]] == 1)
    return pairs, active


def positive_union_loss(logits, active, pairs, pos_weight, denominator):
    """Sum extra observed-positive losses over the UNCHANGED full-batch D0.

    eta = z - log(w) follows the existing deployed/prior-corrected convention.
    Under independent Bernoulli outputs, logit(P(A OR B)) equals
    logsumexp(eta_a, eta_b, eta_a+eta_b). Applying the same weighted-BCE link
    g = logit(P(A OR B)) + log(w_a) avoids applying the prior correction twice.
    As P(B)->0 the added term tends to the ordinary weighted positive BCE for A
    (which mask-v2 had removed); the TOTAL objective is not mask-v2-equivalent.

    This positive-only partial-label objective is not a calibrated full coarse
    Bernoulli risk. It deliberately adds no coarse negative supervision. Call
    each microbatch with the same original full-batch mask sum, never its own.
    """
    if logits.ndim != 2 or active.shape != (len(logits), len(pairs)) or active.dtype != torch.bool:
        raise ValueError("logits and boolean coarse-positive events must align")
    if pos_weight.shape != (logits.shape[1],):
        raise ValueError("pos_weight must retain the original attribute ordering")
    z = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    w = pos_weight.to(device=z.device, dtype=z.dtype)
    eta = z - w.log()
    a, b = zip(*pairs)
    ea, eb = eta[:, a], eta[:, b]
    union_logit = torch.logsumexp(torch.stack((ea, eb, ea + eb)), dim=0)
    coarse_weight = w[list(a)]
    linked_logit = union_logit + coarse_weight.log()
    per_event = F.binary_cross_entropy_with_logits(
        linked_logit, torch.ones_like(linked_logit), pos_weight=coarse_weight, reduction="none"
    )
    return (per_event * active).sum() / denominator
