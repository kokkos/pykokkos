import pykokkos as pk

import numpy as np

# TODO: add proper implementations rather
# than wrapping NumPy
# These are required for the array API:
# https://data-apis.org/array-api/2021.12/API_specification/utility_functions.html

def all(x, /, *, axis=None, keepdims=False):
    if x == True:
        return True
    elif x == False:
        return False
    np_result = np.all(x)
    ret_val = pk.from_numpy(np_result)
    return ret_val


def any(x, /, *, axis=None, keepdims=False):
    return pk.View(pk.from_numpy(np.any(x)))
