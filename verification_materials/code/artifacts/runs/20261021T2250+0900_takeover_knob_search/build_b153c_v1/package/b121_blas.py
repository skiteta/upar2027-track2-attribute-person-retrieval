"""Scoped control of NumPy's already loaded OpenBLAS without new dependencies.

Only documented C get/set APIs on the actual NumPy extension dependency handle
are used. Unknown backends are reported, not guessed or rejected. Environment
variables are not assumed effective after NumPy was imported by ingestion.
"""

from contextlib import contextmanager
import ctypes
import importlib


def _openblas_api():
    for module_name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            extension = importlib.import_module(module_name)
            if not getattr(extension, "__file__", "").endswith((".so", ".pyd")):
                continue
            handle = ctypes.CDLL(extension.__file__)
        except (ImportError, OSError):
            continue
        for prefix, suffix in (("", "64_"), ("", ""), ("scipy_", "64_"), ("scipy_", "")):
            stem = prefix + "openblas_"
            try:
                getter = getattr(handle, stem + "get_num_threads" + suffix)
                setter = getattr(handle, stem + "set_num_threads" + suffix)
            except AttributeError:
                continue
            getter.argtypes, getter.restype = [], ctypes.c_int
            setter.argtypes, setter.restype = [ctypes.c_int], None
            return (
                getter,
                setter,
                {"backend": "OpenBLAS", "extension": extension.__file__, "getter": stem + "get_num_threads" + suffix},
            )
    return None


@contextmanager
def numpy_blas_threads(threads=2):
    if type(threads) is not int or threads <= 0:
        raise ValueError("positive integer NumPy thread count required")
    api = _openblas_api()
    if api is None:
        yield {"controlled": False, "backend": "unsupported_or_unidentified", "actual_threads": None}
        return
    getter, setter, info = api
    previous = getter()
    if previous <= 0:
        raise RuntimeError("invalid OpenBLAS prior thread count")
    try:
        setter(threads)
        actual = getter()
        if actual != threads:
            raise RuntimeError("OpenBLAS runtime control did not take effect")
        yield {**info, "controlled": True, "previous_threads": previous, "actual_threads": actual}
    finally:
        setter(previous)
        if getter() != previous:
            raise RuntimeError("OpenBLAS original thread count was not restored")
