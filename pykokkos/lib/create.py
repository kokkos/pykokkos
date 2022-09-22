import pykokkos as pk

def zeros(shape, *, dtype=None, device=None):
    if dtype is None:
        dtype = pk.double

    if isinstance(shape, int):
        return pk.View([shape], dtype=dtype)
    else:
        return pk.View([*shape], dtype=dtype)
