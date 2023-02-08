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
        if isinstance(fill_value, pk.View):
            dtype = fill_value.dtype
        elif isinstance(fill_value, bool):
            dtype = pk.uint8
        elif isinstance(fill_value, int):
            dtype = pk.int64
        elif isinstance(fill_value, float):
            dtype = pk.float64
    if not isinstance(shape, pk.View):
        try:
            view = pk.View(shape, dtype=dtype)
        except TypeError:
            view = pk.View([shape], dtype=dtype)
    else:
        view = pk.View([*shape], dtype=dtype)
    view[:] = fill_value
    return view
