import pykokkos as pk


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
