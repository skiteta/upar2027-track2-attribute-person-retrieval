import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("audit_gains", Path(__file__).parents[1] / "scripts/audit_saved_query_gains.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_fixed_delta_bootstrap_and_wins():
    baseline = np.full(20, .5)
    result = mod.summary(baseline, baseline + .125, seed=1, repetitions=500)
    assert result["conditional_query_ci95_mADM"] == [.125, .125]
    assert result["wins_ties_losses"] == [20, 0, 0]
    assert result["delta_bps"] == 1250


def test_source_counts_and_noncontiguous_query_ids():
    names, matrix, counts, majority = mod.source_membership([5, 1], [1, 5, 5, 5], ["A", "A", "B", "B"])
    assert names.tolist() == ["A", "B"]
    assert matrix.tolist() == [[1, 2], [1, 0]]
    assert counts.tolist() == [3, 1]
    assert majority.tolist() == ["B", "A"]


@pytest.mark.parametrize("values", [[np.nan], [1.1], [-.1], [[.5]], [.2, .3]])
def test_bad_saved_vector_rejected(values):
    with pytest.raises(ValueError):
        mod.vector(values, 1)


def test_unknown_query_rejected():
    with pytest.raises(ValueError, match="no gallery match"):
        mod.source_membership([1, 2], [1], ["A"])


def test_sensitivity_not_hide_outlier():
    baseline = np.full(10, .5)
    candidate = baseline + np.asarray([.4] + [-.01] * 9)
    result = mod.summary(baseline, candidate, seed=1, repetitions=500)
    assert result["delta_mADM"] > 0
    assert result["without_largest_10pct_gains_delta"] < 0
