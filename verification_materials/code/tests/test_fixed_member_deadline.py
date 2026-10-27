import pytest
from upar.fixed_member_deadline import fixed_member_step


def decide(**kw):
    args = dict(elapsed=209.98, total_budget=292., soft_prediction_deadline=252.,
                projected_remaining=43., active=3, required=3, remaining=5056)
    args.update(kw)
    return fixed_member_step(**args)


def test_official_failure_region_preserves_all_three():
    # Official tier-drop printed 28.1 AFTER dropping: that is not the 3-model
    # projection. Test values just above the observed 42s soft remaining time.
    for projected in (42.03, 43., 70., 100.):
        result = decide(projected_remaining=projected)
        assert result['active'] == 3 and result['action'] == 'KEEP_ALL_REQUIRED'
        assert result['projection_risk']
        assert result['rank_runtime_guaranteed'] is False


def test_soft_reserve_is_not_hard_deadline():
    assert decide(elapsed=258.)['action'] == 'KEEP_ALL_REQUIRED'


@pytest.mark.parametrize('elapsed', [292., 300., 400.])
def test_absolute_budget_not_disabled(elapsed):
    with pytest.raises(TimeoutError):
        decide(elapsed=elapsed)


@pytest.mark.parametrize('active', [0, 1, 2, 4])
def test_no_degraded_or_extra_member(active):
    with pytest.raises(RuntimeError):
        decide(active=active)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_invalid_projection_rejected(value):
    with pytest.raises(ValueError):
        decide(projected_remaining=value)


def test_complete_gallery_keeps_plan():
    result = decide(elapsed=260., remaining=0, projected_remaining=0.)
    assert result['action'] == 'KEEP_ALL_REQUIRED' and not result['projection_risk']
