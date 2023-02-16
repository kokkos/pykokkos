import pykokkos as pk

def zeros(shape, *, dtype=None, device=None):
    if dtype is None:
        dtype = pk.float64

    if isinstance(shape, int):
        return pk.View([shape], dtype=dtype)
    else:
        return pk.View([*shape], dtype=dtype)


def ones(shape, *, dtype=None, device=None):
    if dtype is None:
        # NumPy also defaults to a double for ones()
        dtype = pk.float64
    view: pk.View = pk.View([*shape], dtype=dtype)
    view[:] = 1
    return view


def ones_like(x, /, *, dtype=None, device=None):
    if dtype is None:
        dtype = x.dtype
    view: pk.View = pk.View([*x.shape], dtype=dtype)
    view[:] = 1
    return view


def zeros_like(x, /, *, dtype=None, device=None):
    if dtype is None:
        dtype = x.dtype
    # NOTE: at the moment PyKokkos automatically
    # zeros out allocated memory, but this may not
    # always be the case if we want to support an
    # efficient empty() implementation
    view: pk.View = pk.View([*x.shape], dtype=dtype)
    return view


def full(shape, fill_value, *, dtype=None, device=None):
    if dtype is None:
        dtype = fill_value.dtype
    try:
        view: pk.View = pk.View([*shape], dtype=dtype)
    except TypeError:
        view: pk.View = pk.View([shape], dtype=dtype)
    view[:] = fill_value
    return view


def full_like(x, /, fill_value, *, dtype=None, device=None):
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    view: pk.View = pk.View([*shape], dtype=dtype)
    view[:] = fill_value
    return view
