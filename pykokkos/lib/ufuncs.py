from functools import singledispatch

import pykokkos as pk

# NOTE: it might be slick if we could encapsulate
# the workunit code inside of the ufunc definitions,
# but we currently require global scoping


@pk.workunit
def reciprocal_impl_1d_double(view: pk.View1D[pk.double], tid: int):
    view[tid] = 1 / view[tid] # type: ignore

@pk.workunit
def reciprocal_impl_1d_single(view: pk.View1D[pk.single], tid: int):
    view[tid] = 1 / view[tid] # type: ignore



# TODO: how are we going to "ufunc dispatch" when the view
# has a different type/precision? i.e., something other than
# pk.double?

@singledispatch
def reciprocal(view):
    """
    Return the reciprocal of the argument, element-wise.

    Parameters
    ----------
    view : pk.View1D
           Input view.

    Returns
    -------
    y : pk.View1D
        Output view.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    """
    pass


@reciprocal.register
def _(view: pk.View1D[pk.double]):
    pk.parallel_for(view.shape[0], reciprocal_impl_1d_double, view=view)
    return view

@reciprocal.register
def _(view: pk.View1D[pk.single]):
    pk.parallel_for(view.shape[0], reciprocal_impl_1d_single, view=view)
    return view
    
