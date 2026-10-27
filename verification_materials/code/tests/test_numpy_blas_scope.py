import pytest

from upar import numpy_blas_scope as mod


def mocked(monkeypatch, initial=6, broken=False):
    current = [initial]
    calls = []

    def set_threads(n):
        calls.append(n)
        if not broken:
            current[0] = n

    monkeypatch.setattr(mod, "_openblas_api", lambda: (lambda: current[0], set_threads, {"backend": "mock"}))
    return current, calls


def test_scoped_control_restores(monkeypatch):
    current, calls = mocked(monkeypatch)
    with mod.numpy_blas_threads(2) as info:
        assert current[0] == 2 and info["actual_threads"] == 2
    assert current[0] == 6 and calls == [2, 6]


def test_exception_restores(monkeypatch):
    current, calls = mocked(monkeypatch)
    with pytest.raises(ValueError):
        with mod.numpy_blas_threads(2):
            raise ValueError("graph failed")
    assert current[0] == 6 and calls == [2, 6]


def test_unknown_backend_supported_without_fictitious_control(monkeypatch):
    monkeypatch.setattr(mod, "_openblas_api", lambda: None)
    with mod.numpy_blas_threads(2) as info:
        assert not info["controlled"] and info["actual_threads"] is None


def test_failed_setter_is_not_silent(monkeypatch):
    mocked(monkeypatch, broken=True)
    with pytest.raises(RuntimeError, match="did not take effect"):
        with mod.numpy_blas_threads(2):
            pass


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_bad_count_rejected(bad):
    with pytest.raises(ValueError):
        with mod.numpy_blas_threads(bad):
            pass


def test_actual_numpy_backend_restore():
    api = mod._openblas_api()
    if api is None:
        pytest.skip("not an OpenBLAS host")
    prior = api[0]()
    with mod.numpy_blas_threads(2) as info:
        assert info["actual_threads"] == 2 and api[0]() == 2
    assert api[0]() == prior
