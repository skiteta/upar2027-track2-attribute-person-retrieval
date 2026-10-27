from __future__ import annotations

import numpy as np
import pytest

from upar.blend_predictions import blend_probabilities, select_alpha


def test_blend_endpoints_are_exact() -> None:
    baseline = np.asarray([[0.1, 0.3]], dtype=np.float32)
    candidate = np.asarray([[0.8, 0.4]], dtype=np.float32)
    assert np.array_equal(blend_probabilities(baseline, candidate, 0.0), baseline)
    assert np.array_equal(blend_probabilities(baseline, candidate, 1.0), candidate)


def test_blend_middle_is_float32_weighted_probability() -> None:
    baseline = np.asarray([[0.0, 0.4]], dtype=np.float32)
    candidate = np.asarray([[1.0, 0.8]], dtype=np.float32)
    blended = blend_probabilities(baseline, candidate, 0.25)
    assert blended.dtype == np.float32
    assert np.allclose(blended, np.asarray([[0.25, 0.5]], dtype=np.float32), atol=0, rtol=0)


def test_blend_rejects_invalid_alpha_and_shape() -> None:
    values = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="alpha"):
        blend_probabilities(values, values, 1.1)
    with pytest.raises(ValueError, match="aligned"):
        blend_probabilities(values, values[:1], 0.5)


def test_alpha_selection_uses_macro_then_larger_alpha_tie_break() -> None:
    selected = select_alpha(
        [
            {"alpha": 0.5, "macro_mADM": 0.4},
            {"alpha": 0.75, "macro_mADM": 0.4},
            {"alpha": 1.0, "macro_mADM": 0.39},
        ]
    )
    assert selected["alpha"] == 0.75
