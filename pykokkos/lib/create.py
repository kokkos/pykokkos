import pykokkos as pk

def zeros(shape, *, dtype=pk.double, device=None):
    if (hasattr(shape, '__iter__')):
        return pk.View([*shape], dtype=dtype)
    else:
        return pk.View([shape], dtype=dtype)
