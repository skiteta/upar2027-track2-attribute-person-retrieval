"""CPU-only mechanism and fit-boundary tests; no real model/data forward."""
import copy
import json

import numpy as np
import pytest

from upar.fixed_gallery_agreement import retain_identical_predicted_profiles, smooth_with_original_degree
from upar.learned_gallery_edges import (
    FEATURE_NAMES, equal_source_weights, fit_pair_model, logistic_objective,
    one_bit_pairs, pair_features, predict_pair_model, rescue_edge_weights,
    smooth_with_rescue, unit_features,
)
from upar.source_fitted_gallery_edges import _validate_audit, extract, match_selected_lineage, prediction_lineage, source_val_rows


def fixture():
    p = np.full((4, 40), .2, np.float64)
    p[1, 0] = .8
    p[2, :2] = .8
    p[3] = .3
    f = np.asarray([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])
    n = np.asarray([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    return f, p, n, np.full(4, 3, np.int64)


def constant_model(value=.5):
    return {"method": "one_bit_rescue_ridge_logistic_v1", "feature_names": list(FEATURE_NAMES),
            "coefficient": [0.] * len(FEATURE_NAMES), "intercept": float(np.log(value/(1-value)))}


def test_onebit_pairs_are_unique_and_semantic():
    _, p, n, c = fixture()
    src, dst = one_bit_pairs(p, n, c)
    assert list(zip(src, dst)) == [(0, 1), (1, 2), (1, 3)]


def test_pair_features_are_symmetric_bounded_and_fixed_width():
    f, p, n, c = fixture()
    s, d = one_bit_pairs(p, n, c)
    a, b = pair_features(f, p, s, d), pair_features(f, p, d, s)
    assert a.shape == (3, 47)
    np.testing.assert_array_equal(a, b)
    assert np.all((a >= 0) & (a <= 1))
    np.testing.assert_array_equal(a[:, 7:].sum(1), 1)


def test_rejects_non_onebit_or_self_pair():
    f, p, _, _ = fixture()
    for dst in (0, 2, 3):
        with pytest.raises(ValueError):
            pair_features(f, p, np.asarray([0]), np.asarray([dst]))


def test_rescue_symmetric_and_outside_onebit_exactly_zero():
    f, p, n, c = fixture()
    r = rescue_edge_weights(f, p, n, c, constant_model())
    np.testing.assert_array_equal(r, [[.5, 0, 0], [.5, .5, .5], [0, .5, 0], [0, .5, 0]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_rescue_is_bitwise_existing_b121(dtype):
    _, p, n, c = fixture()
    d = np.arange(12, dtype=dtype).reshape(3, 4) / 7
    gn, gc = retain_identical_predicted_profiles(p, n, c)
    expected = smooth_with_original_degree(d, gn, gc, c)
    actual = smooth_with_rescue(d, p, n, c, np.zeros(n.shape))
    np.testing.assert_array_equal(actual, expected)


def test_rescue_formula_and_convex_mass():
    f, p, n, c = fixture()
    d = np.asarray([[0., 10., 20., 30.], [11., 4., -3., 8.]])
    r = rescue_edge_weights(f, p, n, c, constant_model(.75))
    new = smooth_with_rescue(d, p, n, c, r)
    direct = d.copy()
    for i in range(4):
        for slot, j in enumerate(n[i]):
            same = np.array_equal(p[i] >= .5, p[j] >= .5)
            w = 1. if same else r[i, slot]
            direct[:, i] += .1 / c[i] * w * (d[:, j] - d[:, i])
    np.testing.assert_allclose(new, direct, atol=1e-14)
    assert np.all(new >= d.min(1)[:, None]) and np.all(new <= d.max(1)[:, None])
    np.testing.assert_array_equal(smooth_with_rescue(d, p, n, c, r, alpha=0), d)


@pytest.mark.parametrize("kind", ["asymmetric", "samebit", "twobit", "nan", "overone"])
def test_invalid_rescue_rejected(kind):
    f, p, n, c = fixture()
    r = rescue_edge_weights(f, p, n, c, constant_model())
    if kind == "asymmetric":
        r[0, 0] = .4
    elif kind == "samebit":
        r[0, 2] = .4
    elif kind == "twobit":
        r[0, 1] = .4
    elif kind == "nan":
        r[0, 0] = np.nan
    else:
        r[0, 0] = 1.1
    with pytest.raises(ValueError):
        smooth_with_rescue(np.ones((1, 4)), p, n, c, r)


def test_source_balance_not_class_balance():
    groups = np.asarray(["A"] * 2 + ["B"] * 7)
    w = equal_source_weights(groups)
    assert w[:2].sum() == pytest.approx(.5)
    assert w[2:].sum() == pytest.approx(.5)
    with pytest.raises(ValueError):
        equal_source_weights(np.asarray(["A", "B", "C"]))


def test_logistic_gradient_and_repeatability():
    rng = np.random.default_rng(17)
    x = rng.uniform(size=(80, len(FEATURE_NAMES)))
    y = (x[:, 0] > .5).astype(float)
    groups = np.asarray(["A"] * 30 + ["B"] * 50)
    w = equal_source_weights(groups)
    theta = rng.normal(scale=.1, size=x.shape[1]+1)
    analytic = logistic_objective(theta, x, y, w)[1]
    numeric = np.empty_like(theta)
    for i in range(len(theta)):
        perturbation = np.zeros_like(theta)
        perturbation[i] = 1e-6
        numeric[i] = (logistic_objective(theta+perturbation, x, y, w)[0] - logistic_objective(theta-perturbation, x, y, w)[0]) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)
    a, b = fit_pair_model(x, y, groups), fit_pair_model(x, y, groups)
    assert a == b
    p = predict_pair_model(a, x)
    assert p[y == 1].mean() > p[y == 0].mean()
    with pytest.raises(ValueError):
        fit_pair_model(x, np.zeros(len(x)), groups)


def test_target_source_and_training_rows_excluded():
    meta = {"sources": np.asarray(["Market1501", "PA100k", "PETA"] * 2),
            "split": np.asarray(["train"] * 3 + ["val"] * 3)}
    selected, positions = source_val_rows(meta, "PA100k")
    np.testing.assert_array_equal(selected, [3, 5])
    np.testing.assert_array_equal(positions, [0, 2])


def test_upstream_include_val_or_wrong_target_rejected():
    good = {"fold": "PA100k", "include_val": False, "held_out_absent_from_fit": True,
            "fit_rows_by_source": {"Market1501": 10, "PETA": 8}}
    _validate_audit([good], "PA100k")
    for key, value in [("include_val", True), ("fold", "PETA"), ("held_out_absent_from_fit", False),
                       ("fit_rows_by_source", {"PA100k": 10, "PETA": 8})]:
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(ValueError):
            _validate_audit([bad], "PA100k")


def test_no_implicit_gpu_launch(tmp_path):
    with pytest.raises(RuntimeError, match="root"):
        extract(tmp_path, "PA100k")


def test_empty_pairs_and_unit_feature_boundary():
    f, p, _, _ = fixture()
    empty = np.empty(0, np.int64)
    assert pair_features(f, p, empty, empty).shape == (0, 47)
    np.testing.assert_allclose(np.linalg.norm(unit_features(f), axis=1), 1.)
    with pytest.raises(ValueError):
        unit_features(np.zeros((2, 3)))


def test_selected_producer_lineage_must_cover_exact_runs(tmp_path):
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"fold": "PA100k", "include_val": False}))
        (d / "train_report.json").write_text(json.dumps({"status": "TRAIN_DONE", "fold": "PA100k", "include_val": False,
                                                       "fit_rows_by_source": {"Market1501": 10, "PETA": 8}}))
    (tmp_path / "soup").mkdir()
    (tmp_path / "soup/config.json").write_text(json.dumps({"fold": "PA100k", "include_val": False,
                                                          "soup_of": ["a", "b"], "soup_weights": [.5, .5]}))
    leaves, _ = prediction_lineage("soup", "PA100k", root=tmp_path)
    spec = {"lodo": {"members": [{"weight": 1., "runs": {"PA100k": ["soup"]}}]}}
    producers, paths = match_selected_lineage(spec, "PA100k", leaves, root=tmp_path)
    assert producers[0]["resolved_ordered_leaf_runs"] == ["a", "b"]
    assert len(paths) == 5
    with pytest.raises(ValueError, match="exact selected"):
        match_selected_lineage(spec, "PA100k", leaves[:1], root=tmp_path)
    with pytest.raises(ValueError, match="exact selected"):
        match_selected_lineage(spec, "PA100k", leaves + leaves[:1], root=tmp_path)
    # A teacher dependency may not disappear behind an otherwise correct leaf.
    (tmp_path / "a/config.json").write_text(json.dumps({"fold": "PA100k", "include_val": False, "online_teachers": ["b"]}))
    with pytest.raises(ValueError, match="teacher"):
        prediction_lineage("soup", "PA100k", root=tmp_path)
