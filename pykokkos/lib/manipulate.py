import pykokkos as pk

import numpy as np


def reshape(x, /, shape, *, order="C", copy=None):
    reshaped_view = np.reshape(x, shape, order=order)
    view: pk.View = pk.View(reshaped_view.shape, dtype=x.dtype)
    # TODO: write in a kernel/workunit and lean
    # less on NumPy?
    view[:] = reshaped_view
    return view


@pk.workunit
def ravel_F_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    for i in range(view.extent(0)):
        out[tid*view.extent(0) + i] = view[i][tid]

@pk.workunit
def ravel_C_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    for i in range(view.extent(1)):
        out[tid*view.extent(1) + i] = view[tid][i]


def ravel(view, order="C"):
    if view.rank() == 2:
        if view.dtype.__name__ == "float64":
            out = pk.View([view.shape[0] * view.shape[1]], pk.double)
            if order == "F":
                pk.parallel_for(view.shape[1], ravel_F_impl_2d_double, view=view, out=out)
            elif order == "C":
                pass
                pk.parallel_for(view.shape[0], ravel_C_impl_2d_double, view=view, out=out)
            return out
    
    raise RuntimeError("Ravel supports 2D views only")

@pk.workunit
def expand_dims_0_impl_double(tid: int, view: pk.View1D[pk.double], out: pk.View2D[pk.double]):
    out[0][tid] = view[tid]


@pk.workunit
def expand_dims_1_impl_double(tid: int, view: pk.View1D[pk.double], out: pk.View2D[pk.double]):
    out[tid][0] = view[tid]


@pk.workunit
def expand_dims_0_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        out[0][tid][i] = view[tid][i]


@pk.workunit
def expand_dims_1_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View3D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][0][i] = view[tid][i]


def expand_dims(view, axis=0):
    if view.dtype.__name__ == "float64":
        raise RuntimeError("expand_dims supports views of type double only")

    if view.rank() == 1:
        if axis == 0:
            out = pk.View([1, *view.shape], pk.double)
            pk.parallel_for(view.shape[0], expand_dims_0_impl_double, view=view, out=out)
        else:
            out = pk.View([*view.shape, 1], pk.double)
            pk.parallel_for(view.shape[0], expand_dims_1_impl_double, view=view, out=out)

    if view.rank() == 2:
        if axis == 0:
            out = pk.View([1, *view.shape], pk.double)
            pk.parallel_for(view.shape[0], expand_dims_0_impl_2d_double, view=view, out=out)
        else:
            out = pk.View([view.shape[0], 1, view.shape[1]], pk.double)
            pk.parallel_for(view.shape[0], expand_dims_1_impl_2d_double, view=view, out=out)

    return out