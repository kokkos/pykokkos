import pykokkos as pk

import numpy as np


def reshape(x, /, shape, *, copy=None):
    view: pk.View = pk.View([*shape], dtype=x.dtype)
    # TODO: write in a kernel/workunit and lean
    # less on NumPy?
    view[:] = np.reshape(x, shape)
    return view
