"""Synthetic-only tests: no project-data read, fitting real labels, or GPU."""

import copy
import json

import numpy as np
import pytest
import torch

from upar.source_prior_offset_bce import (
    ATTRIBUTES,
    RECIPE,
    SOURCES,
    WIDTH,
    fit_source_prior_offsets,
    offsets_for_source_codes,
    source_prior_bce_part,
    validate_config,
)

NAMES = tuple([f"synthetic-{i}" for i in range(38)] + list(ATTRIBUTES))
SELECTED = (38, 39)


def fixture():
    # Unequal domain sizes deliberately separate equal-source and pooled priors.
    y = np.zeros((24, WIDTH), np.float32)
    y[:2, 38], y[:6, 39] = 1, 1
    y[8:20, 38], y[8:12, 39] = 1, 1
    y[:, 0] = 1  # An unmodified attribute may validly have prior1.
    src = np.asarray(["Market1501"] * 8 + ["PETA"] * 16)
    role = np.asarray(["train"] * 24)
    reference = np.stack([y[src == s].mean(0, dtype=np.float64) for s in ("Market1501", "PETA")]).mean(0)
    pw = np.clip((1 - reference) / np.maximum(reference, 1e-6), 1, 10).astype(np.float32)
    return y, src, role, reference, pw


def fit(y=None, src=None, role=None, reference=None, pw=None, **kwargs):
    defaults = fixture()
    args = [defaults[i] if value is None else value for i, value in enumerate((y, src, role, reference, pw))]
    return fit_source_prior_offsets(
        *args[:3],
        kwargs.pop("names", NAMES),
        fold=kwargs.pop("fold", "PA100k"),
        include_val=kwargs.pop("include_val", False),
        reference_prior=args[3],
        pos_weight=args[4],
        **kwargs,
    )


def test_fit_formula_metadata_no_mutation_rng_and_serialization():
    y, src, role, ref, pw = fixture()
    before = [v.copy() for v in (y, src, role, ref, pw)]
    nrng, trng = np.random.get_state(), torch.get_rng_state().clone()
    offsets, meta = fit(y, src, role, ref, pw)
    assert offsets.dtype == np.float32 and offsets.shape == (24, 40)
    assert np.count_nonzero(offsets[:, :38]) == 0
    np.testing.assert_allclose(offsets[0, 38:], [np.log(0.25 / 0.75), np.log(0.75 / 0.25)], atol=1e-7, rtol=0)
    np.testing.assert_allclose(offsets[8, 38:], [np.log(0.75 / 0.25), np.log(0.25 / 0.75)], atol=1e-7, rtol=0)
    assert meta["canonical_prior"][38:] == [0.5, 0.5]
    assert meta["source"]["Market1501"]["rows"] == 8
    assert meta["source"]["PETA"]["rows"] == 16
    assert meta["global_pos_weight"] == pw.tolist()
    assert meta["source_code_vocabulary"] == list(SOURCES)
    codes = np.asarray([0] * 8 + [2] * 16)
    np.testing.assert_array_equal(offsets_for_source_codes(codes, json.loads(json.dumps(meta))), offsets)
    for old, actual in zip(before, (y, src, role, ref, pw)):
        np.testing.assert_array_equal(old, actual)
    later = np.random.get_state()
    assert nrng[0] == later[0] and np.array_equal(nrng[1], later[1]) and nrng[2:] == later[2:]
    assert torch.equal(trng, torch.get_rng_state())


def test_row_permutation_and_source_subset_mapping():
    y, src, role, ref, pw = fixture()
    permutation = np.random.default_rng(42).permutation(len(y))
    old, meta = fit(y, src, role, ref, pw)
    new, _ = fit(y[permutation], src[permutation], role[permutation], ref, pw)
    np.testing.assert_array_equal(new, old[permutation])
    np.testing.assert_array_equal(offsets_for_source_codes(np.asarray([2, 0, 2]), meta), old[[8, 0, 8]])


@pytest.mark.parametrize(
    "fault",
    [
        "held",
        "unknownsource",
        "missing_source",
        "val",
        "test",
        "include_val",
        "fold",
        "nonbinary",
        "nan",
        "empty",
        "shape",
        "pooled_prior",
        "wrong_pw",
        "pw_float64",
        "zero_prior",
        "one_prior",
        "missing_name",
        "duplicate_name",
    ],
)
def test_fit_rejects_boundary_prior_or_shape_fault(fault):
    y, src, role, ref, pw = fixture()
    kwargs = {}
    if fault == "held":
        src = np.asarray(["PA100k"] * 8 + ["PETA"] * 16)
    if fault == "unknownsource":
        src = np.asarray(["alien"] * 8 + ["PETA"] * 16)
    if fault == "missing_source":
        src = np.asarray(["Market1501"] * 24)
    if fault == "val":
        role[0] = "val"
    if fault == "test":
        role = np.asarray(["test"] * 24)
    if fault == "include_val":
        kwargs["include_val"] = True
    if fault == "fold":
        kwargs["fold"] = "alien"
    if fault == "nonbinary":
        y[0, 2] = 0.5
    if fault == "nan":
        y[0, 2] = np.nan
    if fault == "empty":
        y, src, role = y[:0], src[:0], role[:0]
    if fault == "shape":
        y = y[:, :-1]
    if fault == "pooled_prior":
        ref = y.mean(0, dtype=np.float64)
    if fault == "wrong_pw":
        pw[38] = 2
    if fault == "pw_float64":
        pw = pw.astype(np.float64)
    if fault in ("zero_prior", "one_prior"):
        y[:8, 38] = float(fault == "one_prior")
        ref = np.stack([y[src == s].mean(0, dtype=np.float64) for s in ("Market1501", "PETA")]).mean(0)
        pw = np.clip((1 - ref) / np.maximum(ref, 1e-6), 1, 10).astype(np.float32)
    if fault == "missing_name":
        kwargs["names"] = tuple([f"unknown{i}" for i in range(40)])
    if fault == "duplicate_name":
        kwargs["names"] = NAMES[:-1] + (NAMES[-2],)
    with pytest.raises(ValueError):
        fit(y, src, role, ref, pw, **kwargs)


def test_explicit_all_trainval_boundary_is_not_lodo():
    y, src, role, _, _ = fixture()
    y = np.concatenate((y, y[:8]))
    src = np.concatenate((src, np.asarray(["PA100k"] * 8)))
    role = np.asarray(["train"] * len(y))
    role[1::3] = "val"
    ref = np.stack([y[src == s].mean(0, dtype=np.float64) for s in SOURCES]).mean(0)
    pw = np.clip((1 - ref) / np.maximum(ref, 1e-6), 1, 10).astype(np.float32)
    with pytest.raises(ValueError):
        fit(y, src, role, ref, pw, fold="all", include_val=False)
    offsets, meta = fit(y, src, role, ref, pw, fold="all", include_val=True)
    assert offsets.shape == y.shape and meta["fit_sources"] == list(SOURCES)
    assert meta["include_val"] is True


@pytest.mark.parametrize("codes", [[1], [-1], [3], [0.0], [True], [], [[0]], [np.nan]])
def test_held_unknown_noninteger_source_codes_rejected(codes):
    _, meta = fit()
    with pytest.raises(ValueError):
        offsets_for_source_codes(np.asarray(codes), meta)


@pytest.mark.parametrize(
    "fault", ["offset", "nonselected", "ref", "pw", "fitrows", "valrole", "sourcecode", "selected", "boundary"]
)
def test_mutated_metadata_rejected(fault):
    _, meta = fit()
    if fault == "offset":
        meta["source"]["PETA"]["offset_float32"][38] += 0.1
    if fault == "nonselected":
        meta["source"]["PETA"]["offset_float32"][1] = 0.1
    if fault == "ref":
        meta["canonical_prior"][2] = 0.1
    if fault == "pw":
        meta["global_pos_weight"][2] = 1
    if fault == "fitrows":
        meta["fit_rows"] += 1
    if fault == "valrole":
        meta["source"]["PETA"]["rows_by_split"] = {"val": 16}
    if fault == "sourcecode":
        meta["source_code_vocabulary"].reverse()
    if fault == "selected":
        meta["selected_indices"].reverse()
    if fault == "boundary":
        meta["include_val"] = True
    with pytest.raises(ValueError):
        offsets_for_source_codes(np.asarray([0, 2]), meta)


def loss(z, y, pw, offset, denominator, **kwargs):
    return source_prior_bce_part(z, y, pw, offset, denominator, attribute_names=NAMES, **kwargs)


def test_mixed_source128_micro32_by4_loss_and_parameter_gradient_parity():
    _, meta = fit()
    generator = torch.Generator().manual_seed(451)
    x = torch.randn(128, 7, generator=generator)
    y = torch.randint(0, 2, (128, 40), generator=generator).float()
    # Four microbatches have distinct, nonrepresentative source proportions.
    codes = np.asarray([0] * 32 + [2] * 32 + [0, 2] * 16 + [2] * 24 + [0] * 8)
    offsets = torch.from_numpy(offsets_for_source_codes(codes, meta))
    pw = torch.tensor(meta["global_pos_weight"], dtype=torch.float32)
    model = torch.nn.Linear(7, 40)
    peer = copy.deepcopy(model)
    full = loss(model(x), y, pw, offsets, 128 * 40)
    full.backward()
    total = torch.tensor(0.0)
    for start in range(0, 128, 32):
        part = loss(peer(x[start : start + 32]), y[start : start + 32], pw, offsets[start : start + 32], 128 * 40)
        total += part.detach()
        part.backward()
    torch.testing.assert_close(full, total, rtol=1e-6, atol=1e-7)
    for a, b in zip(model.parameters(), peer.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=2e-8)


def test_disabled_exact_h1_control_loss_and_gradient_even_with_nonzero_offsets():
    generator = torch.Generator().manual_seed(31)
    z = torch.randn(32, 40, generator=generator, requires_grad=True)
    other = z.detach().clone().requires_grad_(True)
    target = torch.randint(0, 2, z.shape, generator=generator).float()
    _, meta = fit()
    pw = torch.tensor(meta["global_pos_weight"], dtype=torch.float32)
    offsets = torch.from_numpy(offsets_for_source_codes(np.asarray([0, 2] * 16), meta))
    actual = loss(z, target, pw, offsets, 128 * 40, enabled=False)
    expected = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(other.float(), target) * 0.25
    assert torch.equal(actual, expected)
    actual.backward()
    expected.backward()
    assert torch.equal(z.grad, other.grad)


def test_zero_offset_exact_control():
    z = torch.tensor([[0.2] * 40], requires_grad=True)
    target = torch.ones_like(z)
    actual = loss(z, target, torch.ones(40), torch.zeros_like(z), 40)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z, target, pos_weight=torch.ones(40))
    assert torch.equal(actual, expected)


def test_analytic_source_bayes_optimum_canonical_link_and_gradient():
    # True posterior eta_s under a fixed likelihoodratio3; no fit/optimization.
    for prior in (0.08, 0.35):
        canonical, weight, likelihood = (0.08 + 0.35) / 2, 3.7, 3.0
        eta = prior * likelihood / (prior * likelihood + 1 - prior)
        delta = np.log(prior / (1 - prior)) - np.log(canonical / (1 - canonical))
        f = np.log(likelihood) + np.log(canonical / (1 - canonical)) + np.log(weight)
        z = torch.zeros(1, 40, requires_grad=True)
        with torch.no_grad():
            z[0, 38] = f
        offsets, pw = torch.zeros(1, 40), torch.ones(40)
        offsets[0, 38], pw[38] = delta, weight
        positive, negative = torch.zeros_like(z), torch.zeros_like(z)
        positive[0, 38] = 1
        value = eta * loss(z, positive, pw, offsets, 40) + (1 - eta) * loss(z, negative, pw, offsets, 40)
        value.backward()
        assert abs(float(z.grad[0, 38])) < 1e-8
        deployed = torch.sigmoid(z.detach()[0, 38] - torch.log(pw[38]))
        assert float(deployed) == pytest.approx(
            canonical * likelihood / (canonical * likelihood + 1 - canonical), abs=1e-7
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_finite_logits_use_fp32_loss(dtype):
    z = torch.full((2, 40), 65504.0, dtype=dtype, requires_grad=True)
    target = torch.zeros(2, 40)
    offset = torch.zeros(2, 40)
    offset[:, 38], offset[:, 39] = -0.6, 0.4
    value = loss(z, target, torch.full((40,), 10.0), offset, 80)
    value.backward()
    assert value.dtype == torch.float32 and torch.isfinite(value) and torch.isfinite(z.grad).all()


@pytest.mark.parametrize(
    "fault",
    [
        "logit_nan",
        "logit_inf",
        "target_nan",
        "target_soft",
        "pw_nan",
        "pw_zero",
        "pw11",
        "offset_inf",
        "otherattribute",
        "denominator_zero",
        "denominator_bool",
        "denominator_small",
        "denominator_notN40",
        "wrongshape",
        "row_pw",
        "empty",
        "offset_grad",
        "target_grad",
        "pw_grad",
        "pw_dtype",
        "offset_dtype",
    ],
)
def test_invalid_loss_inputs_rejected(fault):
    z, target, pw, offset, denominator = torch.zeros(2, 40), torch.zeros(2, 40), torch.ones(40), torch.zeros(2, 40), 80
    if fault == "logit_nan":
        z[0, 0] = float("nan")
    if fault == "logit_inf":
        z[0, 0] = float("inf")
    if fault == "target_nan":
        target[0, 0] = float("nan")
    if fault == "target_soft":
        target[0, 0] = 0.5
    if fault == "pw_nan":
        pw[0] = float("nan")
    if fault == "pw_zero":
        pw[0] = 0
    if fault == "pw11":
        pw[0] = 11
    if fault == "offset_inf":
        offset[0, 38] = float("inf")
    if fault == "otherattribute":
        offset[0, 2] = 0.1
    if fault == "denominator_zero":
        denominator = 0
    if fault == "denominator_bool":
        denominator = True
    if fault == "denominator_small":
        denominator = 40
    if fault == "denominator_notN40":
        denominator = 81
    if fault == "wrongshape":
        target = target[:, :-1]
    if fault == "row_pw":
        pw = pw.expand(2, -1)
    if fault == "empty":
        z, target, offset = z[:0], target[:0], offset[:0]
    if fault == "offset_grad":
        offset.requires_grad_(True)
    if fault == "target_grad":
        target.requires_grad_(True)
    if fault == "pw_grad":
        pw.requires_grad_(True)
    if fault == "pw_dtype":
        pw = pw.double()
    if fault == "offset_dtype":
        offset = offset.double()
    with pytest.raises(ValueError):
        loss(z, target, pw, offset, denominator)


@pytest.mark.parametrize(
    "key,value",
    [
        ("bce_profile_balance", "x"),
        ("label_mask", "v2"),
        ("rank_alpha", 0.1),
        ("focal_gamma", 1),
        ("partswap_p", 0.1),
        ("mixup_p", 0.1),
        ("label_smoothing", 0.1),
        ("kd_alpha", 0.1),
        ("online_teachers", ["x"]),
        ("sampling", "cycle"),
        ("pos_weight_cap", 20),
    ],
)
def test_joint_changes_rejected(key, value):
    with pytest.raises(ValueError):
        validate_config({"source_prior_offset_bce": RECIPE, "pos_weight_cap": 10.0, key: value})


def test_valid_config_no_mutation():
    cfg = {"source_prior_offset_bce": RECIPE, "pos_weight_cap": 10.0, "sampling": "replace"}
    before = copy.deepcopy(cfg)
    validate_config(cfg)
    assert cfg == before


@pytest.mark.parametrize(
    "fault", ["ref_nan", "ref_inf", "ref_shape", "ref_negative", "role_shape", "src_shape", "pw_nan", "pw_inf"]
)
def test_fit_nonfinite_reference_and_shape_guards(fault):
    y, src, role, ref, pw = fixture()
    if fault == "ref_nan":
        ref[38] = np.nan
    if fault == "ref_inf":
        ref[38] = np.inf
    if fault == "ref_shape":
        ref = ref[:-1]
    if fault == "ref_negative":
        ref[38] = -0.1
    if fault == "role_shape":
        role = role[:-1]
    if fault == "src_shape":
        src = src[:, None]
    if fault == "pw_nan":
        pw[38] = np.nan
    if fault == "pw_inf":
        pw[38] = np.inf
    with pytest.raises(ValueError):
        fit(y, src, role, ref, pw)


def test_attribute_permutation_moves_only_the_two_named_offsets():
    y, src, role, ref, pw = fixture()
    order = np.random.default_rng(19).permutation(WIDTH)
    original, _ = fit(y, src, role, ref, pw)
    names = tuple(NAMES[i] for i in order)
    changed, _ = fit(y[:, order], src, role, ref[order], pw[order], names=names)
    np.testing.assert_array_equal(changed, original[:, order])
    z = torch.zeros_like(torch.from_numpy(y[:, order]))
    result = source_prior_bce_part(
        z,
        torch.from_numpy(y[:, order]),
        torch.from_numpy(pw[order]),
        torch.from_numpy(changed),
        y.size,
        attribute_names=names,
    )
    assert torch.isfinite(result)
