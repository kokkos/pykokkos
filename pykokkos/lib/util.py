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


@pk.workunit
def sum_impl_1d_double(tid: int, acc: pk.Acc[pk.double], viewA: pk.View1D[pk.double]):
    acc += viewA[tid]


@pk.workunit
def sum_axis0_impl_1d_double(tid: int, viewA: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(viewA.extent(0)):
        out[tid] += viewA[i][tid]


@pk.workunit
def sum_axis1_impl_1d_double(tid: int, viewA: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(viewA.extent(1)):
        out[tid] += viewA[tid][i]


def sum(viewA, axis=None):
    if axis is not None:
        if axis == 0:
            out = pk.View([viewA.shape[1]], pk.double)
            pk.parallel_for(viewA.shape[1], sum_axis0_impl_1d_double, viewA=viewA, out=out)
            return out
        else:
            out = pk.View([viewA.shape[0]], pk.double)
            pk.parallel_for(viewA.shape[0], sum_axis1_impl_1d_double, viewA=viewA, out=out)

            return out


    if str(viewA.dtype) == "DataType.double":
        return pk.parallel_reduce(
            viewA.shape[0],
            sum_impl_1d_double,
            viewA=viewA)


def find_max(viewA):
    return max(viewA)


def searchsorted(view, ele):
    return np.searchsorted(view, ele)
