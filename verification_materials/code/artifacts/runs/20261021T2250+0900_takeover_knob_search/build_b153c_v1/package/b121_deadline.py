"""Deadline guard for an immutable, mandatory model ensemble.

Projection and rank-reserve are scheduling warnings, never permission to alter
the submitted model. The absolute total budget remains a hard stop. Remaining
rank time is not guaranteed by this policy; full runtime testing is separate.
"""
from __future__ import annotations

import math


def fixed_member_step(*, elapsed, total_budget, soft_prediction_deadline,
                      projected_remaining, active, required, remaining):
    values = (elapsed, total_budget, soft_prediction_deadline, projected_remaining)
    if any(not math.isfinite(float(x)) for x in values):
        raise ValueError("finite timing observations required")
    if elapsed < 0 or total_budget <= 0 or projected_remaining < 0:
        raise ValueError("invalid timing values")
    if not 0 < soft_prediction_deadline < total_budget:
        raise ValueError("positive rank reserve inside total budget required")
    if any(type(x) is not int for x in (active, required, remaining)):
        raise ValueError("integer ensemble counts required")
    if required < 1 or active != required or remaining < 0:
        raise RuntimeError("mandatory ensemble membership changed")
    if elapsed >= total_budget:
        raise TimeoutError("fixed ensemble reached absolute total runtime budget")
    soft_left = soft_prediction_deadline - elapsed
    return {"action": "KEEP_ALL_REQUIRED", "active": active,
            "projection_risk": remaining > 0 and projected_remaining > max(0., soft_left),
            "soft_left_s": soft_left, "total_left_s": total_budget-elapsed,
            "rank_runtime_guaranteed": False}
