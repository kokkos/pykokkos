import pykokkos as pk

import numpy as np


@pk.workunit
def reciprocal_impl_1d_double(tid: int, view: pk.View1D[pk.double]):
    view[tid] = 1 / view[tid] # type: ignore


@pk.workunit
def reciprocal_impl_1d_float(tid: int, view: pk.View1D[pk.float]):
    view[tid] = 1 / view[tid] # type: ignore


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
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_double, view=view)
    elif str(view.dtype) == "DataType.float":
        pk.parallel_for(view.shape[0], reciprocal_impl_1d_float, view=view)
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
def arctan2_impl_1d_double(tid: int,
                           view_x1: pk.View1D[pk.double],
                           view_x2: pk.View1D[pk.double],
                           view_result: pk.View1D[pk.double],
                           ):
    view_result[tid] = atan2(view_x1[tid], view_x2[tid]) # type: ignore


@pk.workunit
def arctan2_impl_1d_float(tid: int,
                           view_x1: pk.View1D[pk.float],
                           view_x2: pk.View1D[pk.float],
                           view_result: pk.View1D[pk.float],
                           ):
    view_result[tid] = atan2(view_x1[tid], view_x2[tid]) # type: ignore




def arctan2(view_x1, view_x2):
    """
    Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Parameters
    ----------
    view_x1 : pykokkos view or NumPy array
           Input view.

    view_x2 : pykokkos view or NumPy array
           Input view.

    Returns
    -------
    y : pykokkos view or NumPy array

    """
    # TODO: what to do about the case where one argument is a view
    # and the other is a NumPy array?
    # TODO: add error handle for shape mismatches (and for broadcasting
    # at some point)?
    # TODO: what to do if the argument views/arrays have different types
    # (i.e., one is float, the other double)?
    if (isinstance(view_x1, (np.ndarray, np.generic)) and
       (isinstance(view_x2, (np.ndarray, np.generic)))):
        if np.issubdtype(view_x1.dtype, np.float64):
            view_x1_loc: pk.View1d = pk.View([view_x1.size], pk.double)
            view_x2_loc: pk.View1d = pk.View([view_x2.size], pk.double)
        elif np.issubdtype(view_x1.dtype, np.float32):
            view_x1_loc: pk.View1d = pk.View([view_x1.size], pk.float)
            view_x2_loc: pk.View1d = pk.View([view_x2.size], pk.float)
        view_x1_loc[:] = view_x1
        view_x2_loc[:] = view_x2
        view_x1 = view_x1_loc
        view_x2 = view_x2_loc
        numpy_arr = True
    else:
        numpy_arr = False
    if (str(view_x1.dtype) == "DataType.double"
        and str(view_x2.dtype) == "DataType.double"):
        view_result: pk.View1d = pk.View([view_x1.size], pk.double)
        pk.parallel_for(view_x1.shape[0],
                        arctan2_impl_1d_double,
                        view_x1=view_x1,
                        view_x2=view_x2,
                        view_result=view_result)
    elif (str(view_x1.dtype) == "DataType.float"
        and str(view_x2.dtype) == "DataType.float"):
        view_result: pk.View1d = pk.View([view_x1.size], pk.float)
        pk.parallel_for(view_x1.shape[0],
                        arctan2_impl_1d_float,
                        view_x1=view_x1,
                        view_x2=view_x2,
                        view_result=view_result)

    if numpy_arr:
        view_result = np.asarray(view_result)
    return view_result
