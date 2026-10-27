import numpy as np
import pytest

from upar.fixed_gallery_agreement import retain_identical_predicted_profiles, smooth_with_original_degree
from upar.fixed_gallery_rerank import build_mutual_graph, smooth_distances


def fixture():
    graph = np.asarray([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], np.int64)
    count = np.full(4, 3, np.int64)
    p = np.full((4,40), .25, np.float32)
    p[2:, 0] = .75
    return p, graph, count


def test_symmetric_exact40bit_gate_preserves_order_and_inputs():
    p,n,c=fixture()
    before=[x.copy() for x in (p,n,c)]
    out,count=retain_identical_predicted_profiles(p,n,c)
    assert out.tolist()==[[1,-1,-1],[0,-1,-1],[3,-1,-1],[2,-1,-1]]
    assert count.tolist()==[1,1,1,1]
    assert all(np.array_equal(a,b) for a,b in zip((p,n,c),before))
    # Equality at the natural .5 boundary belongs to positive, no fitted cut.
    p[0,0]=.5
    changed,num=retain_identical_predicted_profiles(p,n,c)
    assert num.tolist()==[2,0,2,2]
    assert changed[1].tolist()==[-1,-1,-1]


def test_original_degree_no_redistribution_closed_form():
    p,n,c=fixture()
    gated,count=retain_identical_predicted_profiles(p,n,c)
    d=np.asarray([[1,7,2,14],[2,11,3,20]],np.float32)
    old=d.copy()
    actual=smooth_with_original_degree(d,gated,count,c,alpha=.1,query_chunk_size=1,gallery_chunk_size=2)
    expected=d.copy()
    for i in range(4):
        expected[:,i]=d[:,i]+np.float32(.1/3)*(d[:,gated[i,0]]-d[:,i])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6)
    # A retained edge never gets the full original .1 after 2/3 were removed.
    assert np.max(np.abs(actual-d)) < np.max(np.abs(smooth_distances(d,gated,count,alpha=.1)-d))
    np.testing.assert_array_equal(d,old)


def test_all_retained_exact_old_arithmetic_and_alpha0():
    rng=np.random.default_rng(71)
    p=rng.uniform(.01,.49,size=(23,40)).astype(np.float32)
    n,c=build_mutual_graph(p,np.ones(40,np.float32),k=6)
    gated,count=retain_identical_predicted_profiles(p,n,c)
    assert np.array_equal(gated,n) and np.array_equal(count,c)
    d=rng.uniform(0,20,size=(8,23)).astype(np.float32)
    assert np.array_equal(smooth_with_original_degree(d,gated,count,c),smooth_distances(d,n,c))
    assert np.array_equal(smooth_with_original_degree(d,gated,count,c,alpha=0),d)


def test_zero_retained_or_original_isolated_bitwise_noop():
    p,n,c=fixture()
    for i in range(4):
        p[i,i]=.9
    gated,count=retain_identical_predicted_profiles(p,n,c)
    assert not count.any()
    d=np.arange(8,dtype=np.float32).reshape(2,4)
    assert np.array_equal(smooth_with_original_degree(d,gated,count,c),d)
    empty=np.empty((1,0),np.int64)
    assert np.array_equal(smooth_with_original_degree(d[:,:1],empty,np.zeros(1,np.int64),np.zeros(1,np.int64)),d[:,:1])


def test_permutation_equivariance_fixed_given_graph():
    p,n,c=fixture()
    d=np.random.default_rng(6).uniform(size=(3,4)).astype(np.float32)
    first,count=retain_identical_predicted_profiles(p,n,c)
    y=smooth_with_original_degree(d,first,count,c)
    perm=np.asarray([2,0,3,1]); inverse=np.argsort(perm)
    pn=inverse[n[perm]]
    second,sc=retain_identical_predicted_profiles(p[perm],pn,c[perm])
    py=smooth_with_original_degree(d[:,perm],second,sc,c[perm])
    np.testing.assert_array_equal(py,y[:,perm])


@pytest.mark.parametrize("fault",["nan","inf","negative","width","integer","shape"])
def test_bad_probability_inputs(fault):
    p,n,c=fixture()
    if fault=="nan": p[0,0]=np.nan
    if fault=="inf": p[0,0]=np.inf
    if fault=="negative": p[0,0]=-1
    if fault=="width": p=p[:,:39]
    if fault=="integer": p=p.astype(int)
    if fault=="shape": p=p[0]
    with pytest.raises(ValueError): retain_identical_predicted_profiles(p,n,c)


@pytest.mark.parametrize("original",[np.asarray([0,3,3,3]),np.asarray([4,3,3,3]),np.ones(3,dtype=int),np.ones(4,dtype=float)])
def test_invalid_original_degree_rejected(original):
    p,n,c=fixture(); g,rc=retain_identical_predicted_profiles(p,n,c)
    with pytest.raises(ValueError): smooth_with_original_degree(np.ones((2,4),np.float32),g,rc,original)


@pytest.mark.parametrize("alpha",[-1,2,np.nan,np.inf,True])
def test_invalid_alpha_rejected(alpha):
    p,n,c=fixture(); g,rc=retain_identical_predicted_profiles(p,n,c)
    with pytest.raises(ValueError): smooth_with_original_degree(np.ones((2,4),np.float32),g,rc,c,alpha=alpha)
