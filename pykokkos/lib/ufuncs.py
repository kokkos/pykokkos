import pykokkos as pk

import numpy as np


@pk.workunit
def reciprocal_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_2d_double(tid: int, view: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore


@pk.workunit
def reciprocal_impl_2d_float(tid: int, view: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        view[tid][i] = 1 / view[tid][i] # type: ignore


def reciprocal(view):
    """
    Return the reciprocal of the argument, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    """
    # see gh-29 for some discussion of the dispatching
    # awkwardness used here
    if str(view.dtype) == "DataType.double" and len(view.shape) == 1:
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 1:
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_float, view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 2:
        pk.parallel_for(view.shape[0], reciprocal_impl_2d_float, view=view)
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 2:
        pk.parallel_for(view.shape[0], reciprocal_impl_2d_double, view=view)
    # NOTE: pretty awkward to both return the view
    # and operate on it in place; the former is closer
    # to NumPy semantics
    return view


@pk.workunit
def log_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log(view[tid]) # type: ignore


@pk.workunit
def log_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log(view[tid]) # type: ignore


def log(view):
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log_impl_1d_float, view=view)
    return view


@pk.workunit
def sqrt_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = sqrt(view[tid]) # type: ignore


@pk.workunit
def sqrt_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = sqrt(view[tid]) # type: ignore


def sqrt(view):
    """
    Return the non-negative square root of the argument, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    Notes
    -----
    .. note::
        This function should exhibit the same branch cut behavior
        as the equivalent NumPy ufunc.
    """
    # TODO: support complex types when they
    # are available in pykokkos?
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], sqrt_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], sqrt_impl_1d_float, view=view)
    return view


@pk.workunit
def log2_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log2(view[tid]) # type: ignore


@pk.workunit
def log2_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log2(view[tid]) # type: ignore


def log2(view):
    """
    Base-2 logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log2_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log2_impl_1d_float, view=view)
    return view


@pk.workunit
def log10_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log10(view[tid]) # type: ignore


@pk.workunit
def log10_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log10(view[tid]) # type: ignore


def log10(view):
    """
    Base-10 logarithm, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log10_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log10_impl_1d_float, view=view)
    return view


@pk.workunit
def log1p_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = log1p(view[tid]) # type: ignore


@pk.workunit
def log1p_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = log1p(view[tid]) # type: ignore


def log1p(view):
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    y : pykokkos view
        Output view.

    """
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], log1p_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], log1p_impl_1d_float, view=view)
    return view


@pk.workunit
def sign_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    if view[tid] > 0:
        view[tid] = 1
    elif view[tid] == 0:
        view[tid] = 0
    elif view[tid] < 0:
        view[tid] = -1
    else:
        view[tid] = nan("")


@pk.workunit
def sign_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    if view[tid] > 0:
        view[tid] = 1
    elif view[tid] == 0:
        view[tid] = 0
    elif view[tid] < 0:
        view[tid] = -1
    else:
        view[tid] = nan("")


def sign(view):
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], sign_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], sign_impl_1d_float, view=view)
    return view


# TODO: why can't we use concise (but not lambda) functions
# for reductions instead of this boilerplate-heavy infra
# for reducing? see gh-51


@pk.workload
class SumImpl1dDouble:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.double = 0
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        acc += self.view[i]


@pk.workload
class SumImpl1dFloat:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.float = 0
        self.view: pk.View1D[pk.float] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        acc += self.view[i]


@pk.workload
class SumImpl2dDouble:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.double = 0
        self.view: pk.View2D[pk.double] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        for j in range(self.view.extent(1)):
            acc += self.view[i][j]


@pk.workload
class SumImpl2dFloat:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.float = 0
        self.view: pk.View2D[pk.float] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        for j in range(self.view.extent(1)):
            acc += self.view[i][j]


@pk.workload
class SumImpl3dDouble:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.double = 0
        self.view: pk.View3D[pk.double] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        for j in range(self.view.extent(1)):
            for k in range(self.view.extent(2)):
                acc += self.view[i][j][k]


@pk.workload
class SumImpl3dFloat:
    def __init__(self, n, view):
        self.N: int = n
        self.total: pk.float = 0
        self.view: pk.View3D[pk.float] = view

    @pk.main
    def run(self):
        self.total = pk.parallel_reduce(self.N, self.sum)

    @pk.workunit
    def sum(self, i: int, acc: pk.Acc[pk.double]):
        for j in range(self.view.extent(1)):
            for k in range(self.view.extent(2)):
                acc += self.view[i][j][k]


def sum(view):
    """
    Sum of elements.

    Parameters
    ----------
    view : pykokkos view or NumPy array

    Returns
    -------
    y : pykokkos view or NumPy array or scalar

    """
    # TODO: support axis-aligned sums as NumPy does
    # TODO: support `where` argument like NumPy does
    if isinstance(view, (np.ndarray, np.generic)):
        if np.issubdtype(view.dtype, np.float64):
            view_loc = pk.View(view.shape, pk.double)
        elif np.issubdtype(view.dtype, np.float32):
            view_loc = pk.View(view.shape, pk.float)
        view_loc[:] = view
        view = view_loc
    if str(view.dtype) == "DataType.double" and len(view.shape) == 1:
        sum_inst = SumImpl1dDouble(n=view.shape[0],
                                   view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 1:
        sum_inst = SumImpl1dFloat(n=view.shape[0],
                                  view=view)
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 2:
        sum_inst = SumImpl2dDouble(n=view.shape[0],
                                   view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 2:
        sum_inst = SumImpl2dFloat(n=view.shape[0],
                                  view=view)
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 3:
        sum_inst = SumImpl3dDouble(n=view.shape[0],
                                   view=view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 3:
        # NOTE: I believe the Kokkos C++ docs suggest
        # going parallel on the left-most dimension
        # but I'm not sure if that is guaranteed to always
        # be optimal?
        sum_inst = SumImpl3dFloat(n=view.shape[0],
                                  view=view)
    pk.execute(pk.ExecutionSpace.Default, sum_inst)
    return sum_inst.total
