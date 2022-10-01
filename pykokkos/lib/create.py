import pykokkos as pk

def zeros(shape, *, dtype=None, device=None):
    if dtype is None:
        dtype = pk.double

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


def full(shape, fill_value, *, dtype=None, device=None):
    if dtype is None:
        dtype = fill_value.dtype
    try:
        view: pk.View = pk.View([*shape], dtype=dtype)
    except TypeError:
        view: pk.View = pk.View([shape], dtype=dtype)
    view[:] = fill_value
    return view
