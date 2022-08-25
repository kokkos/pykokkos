import pykokkos as pk

def zeros(shape, *, dtype=None, device=None):
    return pk.View([*shape], dtype=dtype)
