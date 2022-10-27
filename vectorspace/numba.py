import numba


def _fake_njit(*args, **kwargs):
    # force compile cache
    global _real_njit
    if 'cache' not in kwargs:
        kwargs = dict(**kwargs, cache=True)
    return _real_njit(*args, **kwargs)


def patch_numba():
    """
    Speed up numba by monkey patching numba.njit to always use compile cache
    """
    global _real_njit
    if numba.njit.__name__ != '_fake_njit':
        _real_njit = numba.njit
        numba.njit = _fake_njit
