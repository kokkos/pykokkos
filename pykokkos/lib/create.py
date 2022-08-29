import pykokkos as pk

def zeros(shape, *, dtype=None, device=None):
    return pk.View([*shape], dtype=dtype)


def ones(shape, *, dtype=None, device=None):
    if dtype is None:
        # NumPy also defaults to a double for ones()
        dtype = pk.float64
    view: pk.View = pk.View([*shape], dtype=dtype)
    view[:] = 1
    if shape == (0,):
        view.shape = (0,)
    view.shape = tuple(view.shape)
    return view
