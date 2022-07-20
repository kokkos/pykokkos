import numpy as np

import pykokkos as pk


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


# NOTE: these workunits/kernels really make little sense
# to a Python developer who hasn't read i.e., the C++ parallel_scan
# docs for Kokkos--could we not make this more Pythonic?
# How does this behave for OpenMP vs. CUDA? The latter is quite
# a complex parallel algorithm I think, and the amount of work
# with multiple scans/passes under the hood is pretty hidden here


@pk.workunit
def cumsum_impl_1d_double(tid: int,
                          acc: pk.Acc[pk.double],
                          last_pass: bool,
                          view: pk.View1D[pk.double],
                          new_view: pk.View1D[pk.double]):
    acc += view[tid]
    new_view[tid] = acc
    if last_pass:
        view[tid] = acc


@pk.workunit
def cumsum_impl_1d_float(tid: int,
                          acc: pk.Acc[pk.float],
                          last_pass: bool,
                          view: pk.View1D[pk.float],
                          new_view: pk.View1D[pk.float]):
    acc += view[tid]
    new_view[tid] = acc
    if last_pass:
        view[tid] = acc


@pk.workunit
def cumsum_impl_1d_int32(tid: int,
                          acc: pk.Acc[pk.int32],
                          last_pass: bool,
                          view: pk.View1D[pk.int32],
                          new_view: pk.View1D[pk.int32]):
    acc += view[tid]
    new_view[tid] = acc
    if last_pass:
        view[tid] = acc


@pk.workunit
def cumsum_impl_1d_int64(tid: int,
                          acc: pk.Acc[pk.int64],
                          last_pass: bool,
                          view: pk.View1D[pk.int64],
                          new_view: pk.View1D[pk.int64]):
    acc += view[tid]
    new_view[tid] = acc
    if last_pass:
        view[tid] = acc

@pk.workunit
def cumsum_impl_2d_double(tid: int,
                          acc: pk.Acc[pk.double],
                          last_pass: bool,
                          view: pk.View2D[pk.double],
                          new_view: pk.View2D[pk.double]):
    # NOTE: by default, NumPy assigns the result
    # to a new flattened array, but it is not clear
    # to me how we'd do that here; while we can make
    # new_view 1D, the iteration behavior is fairly opaque,
    # and possibly not even guaranteed depending on the backend
    # if I understood the feedback from ctrott?
    for j in range(view.extent(1)):
        acc += view[tid][j]
        new_view[tid][j] = acc
        if last_pass:
            view[tid][j] = acc


@pk.workunit
def cumsum_impl_2d_float(tid: int,
                          acc: pk.Acc[pk.float],
                          last_pass: bool,
                          view: pk.View2D[pk.float],
                          new_view: pk.View2D[pk.float]):
    for j in range(view.extent(1)):
        acc += view[tid][j]
        new_view[tid][j] = acc
        if last_pass:
            view[tid][j] = acc


@pk.workunit
def cumsum_impl_2d_int32(tid: int,
                          acc: pk.Acc[pk.int32],
                          last_pass: bool,
                          view: pk.View2D[pk.int32],
                          new_view: pk.View2D[pk.int32]):
    for j in range(view.extent(1)):
        acc += view[tid][j]
        new_view[tid][j] = acc
        if last_pass:
            view[tid][j] = acc


@pk.workunit
def cumsum_impl_2d_int64(tid: int,
                          acc: pk.Acc[pk.int64],
                          last_pass: bool,
                          view: pk.View2D[pk.int64],
                          new_view: pk.View2D[pk.int64]):
    for j in range(view.extent(1)):
        acc += view[tid][j]
        new_view[tid][j] = acc
        if last_pass:
            view[tid][j] = acc

@pk.workunit
def cumsum_impl_3d_double(tid: int,
                          acc: pk.Acc[pk.double],
                          last_pass: bool,
                          view: pk.View3D[pk.double],
                          new_view: pk.View3D[pk.double]):
    for j in range(view.extent(1)):
        for k in range(view.extent(2)):
            acc += view[tid][j][k]
            new_view[tid][j][k] = acc
            if last_pass:
                view[tid][j][k] = acc


@pk.workunit
def cumsum_impl_3d_float(tid: int,
                          acc: pk.Acc[pk.float],
                          last_pass: bool,
                          view: pk.View3D[pk.float],
                          new_view: pk.View3D[pk.float]):
    for j in range(view.extent(1)):
        for k in range(view.extent(2)):
            acc += view[tid][j][k]
            new_view[tid][j][k] = acc
            if last_pass:
                view[tid][j][k] = acc


@pk.workunit
def cumsum_impl_3d_int32(tid: int,
                         acc: pk.Acc[pk.int32],
                         last_pass: bool,
                         view: pk.View3D[pk.int32],
                         new_view: pk.View3D[pk.int32]):
    for j in range(view.extent(1)):
        for k in range(view.extent(2)):
            acc += view[tid][j][k]
            new_view[tid][j][k] = acc
            if last_pass:
                view[tid][j][k] = acc


@pk.workunit
def cumsum_impl_3d_int64(tid: int,
                         acc: pk.Acc[pk.int64],
                         last_pass: bool,
                         view: pk.View3D[pk.int64],
                         new_view: pk.View3D[pk.int64]):
    for j in range(view.extent(1)):
        for k in range(view.extent(2)):
            acc += view[tid][j][k]
            new_view[tid][j][k] = acc
            if last_pass:
                view[tid][j][k] = acc


def cumsum(view):
    """
    Return the cumulative sum of the elements.

    Parameters
    ----------
    view : pykokkos view or NumPy array

    Returns
    -------
    y : pykokkos view or NumPy array

    """
    # TODO: support axis-aligned operation like the NumPy version
    # TODO: support the accumulator and output dtype specification
    # like NumPy
    # TODO: support an output array argument for placing the result
    # at another memory location, as NumPy allows

    # NOTE: parallel over the left-most dimension, but is this really
    # guaranteed to produce optimal parallelism in all cases/for all
    # backends?
    if isinstance(view, (np.ndarray, np.generic)):
        if np.issubdtype(view.dtype, np.float64):
            view_loc = pk.View(view.shape, pk.double)
        elif np.issubdtype(view.dtype, np.float32):
            view_loc = pk.View(view.shape, pk.float)
        elif np.issubdtype(view.dtype, np.int32):
            view_loc = pk.View(view.shape, pk.int32)
        elif np.issubdtype(view.dtype, np.int64):
            view_loc = pk.View(view.shape, pk.int64)
        view_loc[:] = view
        view = view_loc
        arr_type = "numpy"
    else:
        # NOTE: this arr_type stuff will probably need a better
        # design than just these strings eventually..
        arr_type = "kokkos"
    range_policy = pk.RangePolicy(pk.ExecutionSpace.Default, 0, view.shape[0])
    if str(view.dtype) == "DataType.double" and len(view.shape) == 1:
        new_view = pk.View(view.shape, pk.double)
        pk.parallel_scan(range_policy, cumsum_impl_1d_double, view=view, new_view=new_view)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 1:
        new_view = pk.View(view.shape, pk.float)
        pk.parallel_scan(range_policy, cumsum_impl_1d_float, view=view, new_view=new_view)
    elif str(view.dtype) == "DataType.int32" and len(view.shape) == 1:
        new_view = pk.View(view.shape, pk.int32)
        pk.parallel_scan(range_policy, cumsum_impl_1d_int32, view=view, new_view=new_view)
    elif str(view.dtype) == "DataType.int64" and len(view.shape) == 1:
        new_view = pk.View(view.shape, pk.int64)
        pk.parallel_scan(range_policy, cumsum_impl_1d_int64, view=view, new_view=new_view)
    # NOTE: careful here--the default NumPy behavior is to calculate
    # cumsum over the *flattened* array, ignoring shape of the input
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 2:
        new_view = pk.View(view.shape, pk.double)
        pk.parallel_scan(range_policy, cumsum_impl_2d_double, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 2:
        new_view = pk.View(view.shape, pk.float)
        pk.parallel_scan(range_policy, cumsum_impl_2d_float, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.int32" and len(view.shape) == 2:
        new_view = pk.View(view.shape, pk.int32)
        pk.parallel_scan(range_policy, cumsum_impl_2d_int32, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.int64" and len(view.shape) == 2:
        new_view = pk.View(view.shape, pk.int64)
        pk.parallel_scan(range_policy, cumsum_impl_2d_int64, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.double" and len(view.shape) == 3:
        new_view = pk.View(view.shape, pk.double)
        pk.parallel_scan(range_policy, cumsum_impl_3d_double, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.float" and len(view.shape) == 3:
        new_view = pk.View(view.shape, pk.float)
        pk.parallel_scan(range_policy, cumsum_impl_3d_float, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.int32" and len(view.shape) == 3:
        new_view = pk.View(view.shape, pk.int32)
        pk.parallel_scan(range_policy, cumsum_impl_3d_int32, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    elif str(view.dtype) == "DataType.int64" and len(view.shape) == 3:
        new_view = pk.View(view.shape, pk.int64)
        pk.parallel_scan(range_policy, cumsum_impl_3d_int64, view=view, new_view=new_view)
        new_view = np.reshape(new_view, view.size)
    # try to return the same type you receive
    if arr_type == "kokkos":
        if str(view.dtype) == "DataType.float":
            temp_view = pk.View([new_view.size], pk.float)
        elif str(view.dtype) == "DataType.double":
            temp_view = pk.View([new_view.size], pk.double)
        elif str(view.dtype) == "DataType.int32":
            temp_view = pk.View([new_view.size], pk.int32)
        elif str(view.dtype) == "DataType.int64":
            temp_view = pk.View([new_view.size], pk.int64)
        temp_view[:] = new_view
        new_view = temp_view
    else:
        new_view = np.asarray(new_view)
    return new_view
