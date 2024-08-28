import re
import math
from inspect import getmembers, isfunction
from typing import Optional

import numpy as np
import pykokkos as pk
from pykokkos.lib import ufunc_workunits
from pykokkos.interface import ViewType

kernel_dict = dict(getmembers(ufunc_workunits, isfunction))


def _supported_types_check(dtype_str, supported_type_strings):
    options = ""
    for type_str in supported_type_strings:
        options += f".*{type_str}.*|"
    options = options[:-1]
    prog = re.compile(f"({options})" )
    result = prog.match(dtype_str)
    if result is None:
        raise NotImplementedError


def _ufunc_kernel_dispatcher(profiler_name: Optional[str],
                             tid,
                             dtype,
                             ndims,
                             op,
                             sub_dispatcher,
                             **kwargs):
    dtype_extractor = re.compile(r".*(?:dtype|data_types|DataType)\.(\w+)")
    if ndims == 0:
        ndims = 1
    res = dtype_extractor.match(str(dtype))
    dtype_str = res.group(1)
    if dtype_str == "float32":
        dtype_str = "float"
    elif dtype_str == "float64":
        dtype_str = "double"
    function_name_str = f"{op}_impl_{ndims}d_{dtype_str}"
    desired_workunit = kernel_dict[function_name_str]
    # call the kernel
    ret = sub_dispatcher(profiler_name, tid, desired_workunit, **kwargs)
    return ret


def _broadcast_views(view1, view2):
    # support broadcasting by using the same
    # shape matching rules as NumPy
    # TODO: determine if this can be done with
    # more memory efficiency?
    if view1.shape != view2.shape:
        new_shape = np.broadcast_shapes(view1.shape, view2.shape)
        view1_new = pk.View([*new_shape], dtype=view1.dtype)
        view1_new[:] = view1
        view1 = view1_new
        view2_new = pk.View([*new_shape], dtype=view2.dtype)
        view2_new[:] = view2
        view2 = view2_new
    return view1, view2


def _typematch_views(view1, view2):
    # very crude casting implementation
    # for binary ufuncs
    dtype1 = view1.dtype
    dtype2 = view2.dtype
    dtype_extractor = re.compile(r".*(?:data_types|DataType)\.(\w+)")
    res1 = dtype_extractor.match(str(dtype1))
    res2 = dtype_extractor.match(str(dtype2))
    effective_dtype = dtype1
    if res1 is not None and res2 is not None:
        res1_dtype_str = res1.group(1)
        res2_dtype_str = res2.group(1)
        if res1_dtype_str == "double":
            res1_dtype_str = "float64"
        elif res1_dtype_str == "float":
            res1_dtype_str = "float32"
        if res2_dtype_str == "double":
            res2_dtype_str = "float64"
        elif res2_dtype_str == "float":
            res2_dtype_str = "float32"
        if res1_dtype_str == "bool" or res2_dtype_str == "bool":
            res1_dtype_str = "uint8"
            dtype1 = pk.uint8
            res2_dtype_str = "uint8"
            dtype2 = pk.uint8
        if (("int" in res1_dtype_str and "int" in res2_dtype_str) or
            ("float" in res1_dtype_str and "float" in res2_dtype_str)):
            dtype_1_width = int(res1_dtype_str.split("t")[1])
            dtype_2_width = int(res2_dtype_str.split("t")[1])
            if dtype_1_width >= dtype_2_width:
                effective_dtype = dtype1
                view2_new = pk.View([*view2.shape], dtype=effective_dtype)
                view2_new[:] = view2.data
                view2 = view2_new
            else:
                effective_dtype = dtype2
                view1_new = pk.View([*view1.shape], dtype=effective_dtype)
                view1_new[:] = view1.data
                view1 = view1_new
    return view1, view2, effective_dtype


def reciprocal(view, profiler_name: Optional[str] = None):
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
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=view.shape[0],
                             dtype=view.dtype.value,
                             ndims=len(view.shape),
                             op="reciprocal",
                             sub_dispatcher=pk.parallel_for,
                             view=view)
    # NOTE: pretty awkward to both return the view
    # and operate on it in place; the former is closer
    # to NumPy semantics
    return view


@pk.workunit
def log_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = log(view[tid]) # type: ignore


@pk.workunit
def log_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log(view[tid][i]) # type: ignore


@pk.workunit
def log_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = log(view[tid]) # type: ignore


@pk.workunit
def log_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log(view[tid][i]) # type: ignore


def log(view, profiler_name: Optional[str] = None):
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
    if not isinstance(view, pk.ViewType):
        return math.log(view)

    if len(view.shape) > 2:
        raise NotImplementedError("log() ufunc only supports up to 2D views")

    out = pk.View(view.shape, view.dtype)
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(profiler_name, 1, log_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(profiler_name, view.shape[0], log_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(profiler_name, view.shape[0], log_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(profiler_name, 1, log_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(profiler_name, view.shape[0], log_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(profiler_name, view.shape[0], log_impl_2d_float, view=view, out=out)
    return out


@pk.workunit
def sqrt_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = sqrt(view[tid]) # type: ignore


@pk.workunit
def sqrt_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = sqrt(view[tid][i]) # type: ignore


@pk.workunit
def sqrt_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = sqrt(view[tid]) # type: ignore


@pk.workunit
def sqrt_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = sqrt(view[tid][i]) # type: ignore


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
    if isinstance(view, (np.integer, np.floating)):
        return math.sqrt(view)
    # TODO: support complex types when they
    # are available in pykokkos?
    if len(view.shape) > 2:
        raise NotImplementedError("only up to 2D views currently supported for sqrt() ufunc.")
    out = pk.View(view.shape, view.dtype)
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, sqrt_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sqrt_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sqrt_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, sqrt_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sqrt_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sqrt_impl_2d_float, view=view, out=out)
    return out


@pk.workunit
def log2_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = log2(view[tid]) # type: ignore


@pk.workunit
def log2_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log2(view[tid][i]) # type: ignore


@pk.workunit
def log2_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = log2(view[tid]) # type: ignore


@pk.workunit
def log2_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log2(view[tid][i]) # type: ignore


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
    if len(view.shape) > 2:
        raise NotImplementedError("log2() ufunc only supports up to 2D views")
    out = pk.View(view.shape, view.dtype)
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log2_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log2_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log2_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log2_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log2_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log2_impl_2d_float, view=view, out=out)
    return out


@pk.workunit
def log10_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = log10(view[tid]) # type: ignore


@pk.workunit
def log10_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log10(view[tid][i]) # type: ignore


@pk.workunit
def log10_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = log10(view[tid]) # type: ignore


@pk.workunit
def log10_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log10(view[tid][i]) # type: ignore


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
    if view.size == 0:
        return view
    out = pk.View(view.shape, view.dtype)
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log10_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log10_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log10_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log10_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log10_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log10_impl_2d_float, view=view, out=out)
    return out


@pk.workunit
def log1p_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = log1p(view[tid]) # type: ignore


@pk.workunit
def log1p_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = log1p(view[tid]) # type: ignore


@pk.workunit
def log1p_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log1p(view[tid][i]) # type: ignore


@pk.workunit
def log1p_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        out[tid][i] = log1p(view[tid][i]) # type: ignore


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
    if view.size == 0:
        return view
    out = pk.View(view.shape, view.dtype)
    if len(view.shape) > 2:
        raise NotImplementedError("log1p() ufunc only supports up to 2D views")
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, log1p_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log1p_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log1p_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, log1p_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log1p_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log1p_impl_2d_float, view=view, out=out)
    return out


@pk.workunit
def sign_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")


@pk.workunit
def sign_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")


@pk.workunit
def sign_impl_1d_uint8(tid: int, view: pk.View1D[pk.uint8], out: pk.View1D[pk.uint8]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")


@pk.workunit
def sign_impl_1d_int8(tid: int, view: pk.View1D[pk.int8], out: pk.View1D[pk.int8]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")


@pk.workunit
def sign_impl_2d_int8(tid: int, view: pk.View2D[pk.int8], out: pk.View2D[pk.int8]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_2d_uint8(tid: int, view: pk.View2D[pk.uint8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_uint16(tid: int, view: pk.View1D[pk.uint16], out: pk.View1D[pk.uint16]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_uint16(tid: int, view: pk.View2D[pk.uint16], out: pk.View2D[pk.uint16]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")

@pk.workunit
def sign_impl_1d_uint32(tid: int, view: pk.View1D[pk.uint32], out: pk.View1D[pk.uint32]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_uint32(tid: int, view: pk.View2D[pk.uint32], out: pk.View2D[pk.uint32]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_uint64(tid: int, view: pk.View1D[pk.uint64], out: pk.View1D[pk.uint64]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_uint64(tid: int, view: pk.View2D[pk.uint64], out: pk.View2D[pk.uint64]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_int16(tid: int, view: pk.View1D[pk.int16], out: pk.View1D[pk.int16]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_int16(tid: int, view: pk.View2D[pk.int16], out: pk.View2D[pk.int16]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_int32(tid: int, view: pk.View1D[pk.int32], out: pk.View1D[pk.int32]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_int32(tid: int, view: pk.View2D[pk.int32], out: pk.View2D[pk.int32]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_int64(tid: int, view: pk.View1D[pk.int64], out: pk.View1D[pk.int64]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_int64(tid: int, view: pk.View2D[pk.int64], out: pk.View2D[pk.int64]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_1d_uint64(tid: int, view: pk.View1D[pk.uint64], out: pk.View1D[pk.uint64]):
    if view[tid] > 0:
        out[tid] = 1
    elif view[tid] == 0:
        out[tid] = 0
    elif view[tid] < 0:
        out[tid] = -1
    else:
        out[tid] = nan("")

@pk.workunit
def sign_impl_2d_uint64(tid: int, view: pk.View2D[pk.uint64], out: pk.View2D[pk.uint64]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")

@pk.workunit
def sign_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


@pk.workunit
def sign_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)): # type: ignore
        if view[tid][i] > 0:
            out[tid][i] = 1
        elif view[tid][i] == 0:
            out[tid][i] = 0
        elif view[tid][i] < 0:
            out[tid][i] = -1
        else:
            out[tid][i] = nan("")


def sign(view):
    out = pk.View(view.shape, view.dtype)
    if len(view.shape) > 2:
        raise NotImplementedError("only up to 2D views currently supported for sign() ufunc.")
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.double)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_double,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.float)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_float,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_float, view=view, out=out)
    elif "uint32" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.uint32)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_uint32,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_uint32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_uint32, view=view, out=out)
    elif "uint16" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.uint16)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_uint16,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_uint16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_uint16, view=view, out=out)
    elif "int16" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.int16)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_int16,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_int16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_int16, view=view, out=out)
    elif "int32" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.int32)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_int32,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_int32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_int32, view=view, out=out)
    elif "uint64" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.uint64)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_uint64,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_uint64, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_uint64, view=view, out=out)
    elif "int64" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.int64)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_int64,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_int64, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_int64, view=view, out=out)
    elif "uint8" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.uint8)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_uint8,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_uint8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_uint8, view=view, out=out)
    elif "int8" in view.dtype.__name__:
        if view.shape == ():
            new_view = pk.View([1], dtype=pk.int8)
            new_view[:] = view
            pk.parallel_for(1,
                            sign_impl_1d_int8,
                            view=new_view,
                            out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sign_impl_1d_int8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sign_impl_2d_int8, view=view, out=out)
    return out


@pk.workunit
def add_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double], ):
    out[tid] = viewA[tid] + viewB[tid % viewB.extent(0)]


@pk.workunit
def add_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] + viewB[tid]

@pk.workunit
def add_impl_2d_1d(tid, viewA, viewB, out):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[i % viewB.extent(0)]

@pk.workunit
def add_impl_2d_2d(tid, viewA, viewB, out):
    r_idx : int = tid / viewA.extent(1)
    c_idx : int = tid - r_idx * viewA.extent(1)
    out[r_idx][c_idx] = viewA[r_idx][c_idx] + viewB[r_idx][c_idx]


def add(viewA, viewB, profiler_name: Optional[str] = None):
    """
    Sums positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view or scalar
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if not isinstance(viewB, pk.ViewType):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if len(viewA.shape) > 2 or len(viewB.shape) > 2:
        raise NotImplementedError("only 2D views currently supported for add() ufunc.")
    
    if viewA.rank() == 2 and viewB.rank() == 2 and viewA.shape != viewB.shape:
        raise RuntimeError("2D views must have the same shape for add ufunc. Mismatch: {} and {}".format(viewA.shape, viewB.shape))

    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        if viewA.rank() == 1 and viewB.rank() == 1:
            out = pk.View([viewA.shape[0]], pk.double)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0],
                add_impl_1d_double,
                viewA=viewA,
                viewB=viewB,
                out=out)
        elif viewA.rank() == 2 and viewB.rank() == 2:
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.double)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0] * viewA.shape[1],
                add_impl_2d_2d,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            larger = viewA if len(viewA.shape) > len(viewB.shape) else viewB
            smaller = viewB if len(viewA.shape) == len(larger.shape) else viewA
            out = pk.View([larger.shape[0], larger.shape[1]], pk.double)
            pk.parallel_for(
                profiler_name,
                larger.shape[0],
                add_impl_2d_1d,
                viewA=larger,
                viewB=smaller,
                out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        if viewA.rank() == 1 and viewB.rank() == 1:
            out = pk.View([viewA.shape[0]], pk.float)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0],
                add_impl_1d_float,
                viewA=viewA,
                viewB=viewB,
                out=out)
        elif viewB.rank() == 2 and viewB.rank() == 2:
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.float)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0] * viewA.shape[1],
                add_impl_2d_2d,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            larger = viewA if len(viewA.shape) > len(viewB.shape) else viewB
            smaller = viewB if len(viewA.shape) == len(larger.shape) else viewA
            out = pk.View([larger.shape[0], larger.shape[1]], pk.float)
            pk.parallel_for(
                profiler_name,
                larger.shape[0],
                add_impl_2d_1d,
                viewA=larger,
                viewB=smaller,
                out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def multiply_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] * viewB[tid % viewB.extent(0)]


@pk.workunit
def multiply_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] * viewB[tid]


@pk.workunit
def multiply_impl_2d_with_1d(tid, viewA, viewB, out):
    r_idx : int = tid / viewA.extent(1)
    c_idx : int = tid - r_idx * viewA.extent(1)
    out[r_idx][c_idx] = viewA[r_idx][c_idx] * viewB[r_idx % viewB.extent(0)]

@pk.workunit
def multiply_impl_2d_with_2d(tid, viewA, viewB, out):
    r_idx : int = tid / viewA.extent(1)
    c_idx : int = tid - r_idx * viewA.extent(1)
    out[r_idx][c_idx] = viewA[r_idx][c_idx] * viewB[r_idx][c_idx]

def multiply(viewA, viewB, profiler_name: Optional[str] = None):
    """
    Multiplies positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view or scalar
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    if not isinstance(viewB, pk.ViewType):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if len(viewA.shape) > 2 or len(viewB.shape) > 2:
        raise NotImplementedError("only 2D views currently supported for mulitply() ufunc.")

    if viewA.rank() == 2 and viewB.rank() == 2 and viewA.shape != viewB.shape:
        raise RuntimeError("2D views must have the same shape for add ufunc. Mismatch: {} and {}".format(viewA.shape, viewB.shape))

    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        if len(viewA.shape) == 1 and len(viewB.shape) == 1:
            out = pk.View([viewA.shape[0]], pk.double)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0],
                multiply_impl_1d_double,
                viewA=viewA,
                viewB=viewB,
                out=out)
        elif len(viewA.shape) == 2 and len(viewB.shape) == 2:
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.double)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0] * viewA.shape[1],
                multiply_impl_2d_with_2d,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            larger = viewA if len(viewA.shape) > len(viewB.shape) else viewB
            smaller = viewB if len(viewA.shape) == len(larger.shape) else viewA
            out = pk.View([larger.shape[0], larger.shape[1]], pk.double)
            pk.parallel_for(
                profiler_name,
                larger.shape[0] * larger.shape[1],
                multiply_impl_2d_with_1d,
                viewA=larger,
                viewB=smaller,
                out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        if len(viewA.shape) == 1 and len(viewB.shape) == 1:
            out = pk.View([viewA.shape[0]], pk.float)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0],
                multiply_impl_1d_float,
                viewA=viewA,
                viewB=viewB,
                out=out)
        elif len(viewA.shape) == 2 and len(viewB.shape) == 2:
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.float)
            pk.parallel_for(
                profiler_name,
                viewA.shape[0] * viewA.shape[1],
                multiply_impl_2d_with_2d,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            larger = viewA if len(viewA.shape) > len(viewB.shape) else viewB
            smaller = viewB if len(viewA.shape) == len(larger.shape) else viewA
            out = pk.View([larger.shape[0], larger.shape[1]], pk.float)
            pk.parallel_for(
                profiler_name,
                larger.shape[0] * larger.shape[1],
                multiply_impl_2d_with_1d,
                viewA=larger,
                viewB=smaller,
                out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


def check_broadcastable_impl(viewA, viewB):
    """
    Check whether two views are broadcastable as defined here:
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.
    
    Returns
    -------
    _ : boolean
           True if both views are compatible.
    """

    if viewA.shape == viewB.shape:
        return False # cannot broadcast same dims

    v1_p = len(viewA.shape) -1
    v2_p = len(viewB.shape) -1

    while v1_p > -1 and v2_p > -1:
        if viewA.shape[v1_p] != viewB.shape[v2_p]:
            if viewA.shape[v1_p] != 1 and viewB.shape[v2_p] != 1:
                return False
        
        v1_p -= 1
        v2_p -= 1
    
    return True

@pk.workunit
def stretch_fill_impl_scalar_into_1d(tid, scalar, viewOut):
        viewOut[tid] = scalar

@pk.workunit
def stretch_fill_impl_scalar_into_2d(tid, cols, scalar, viewOut):
    for i in range(cols):
        viewOut[tid][i] = scalar
    
@pk.workunit
def stretch_fill_impl_1d_into_2d(tid, cols, viewIn, viewOut):
    for i in range(cols):
        viewOut[tid][i] = viewIn[i]

@pk.workunit
def stretch_fill_impl_2d(tid, inner_its, col_wise, viewIn, viewOut):
    for i in range(inner_its):
        if col_wise:
            viewOut[i][tid] = viewIn[i][0]
        else:
            viewOut[tid][i] = viewIn[0][i]

        

def broadcast_view(val, viewB):
    """
    Broadcasts val onto viewB, returns the "stretched" version of viewA

    Parameters
    ----------
    val : pykokkos view or Scalar
            View or scalar to be broadcasted (is shorter and compatible in dimensions).
    viewB : pykokkos view
            View to be broadcasted onto (is longer and compatible in dimensions).

    Returns
    -------
    out : pykokkos view
           Broadcasted version of viewA.

    """
    if len(viewB.shape) > 2:
        raise NotImplementedError("Broadcasting is only supported upto 2D views")

    is_view = False
    if isinstance(val, ViewType):
        for dim in val.shape:
            if dim != 1:
                is_view = True
        
        if not is_view:
            val = val[0] if len(val.shape) == 1 else val[0][0]

    if is_view:
        is_first_small = len(val.shape) < len(viewB.shape) or ((len(val.shape) == len(viewB.shape)) and val.shape < viewB.shape)
        if not check_broadcastable_impl(val, viewB) or not is_first_small:
            raise ValueError("Incompatible broadcast")
        if not val.dtype == viewB.dtype: 
            raise ValueError("Broadcastable views must have same dtypes")

    out = pk.View(viewB.shape, viewB.dtype)

    if is_view:
        # if both 2D
        if len(val.shape) == 2: #viewB must be 2 because of the val.shape < viewB.shape check
            # figure which orientation is val (row or col)
            col_wise = 1 if val.shape[1] == 1 else 0
            inner_its = viewB.shape[0] if col_wise else viewB.shape[1]
            outer_its = viewB.shape[1] if col_wise else viewB.shape[0]
            pk.parallel_for(outer_its, stretch_fill_impl_2d, inner_its=inner_its, col_wise=col_wise, viewIn=val, viewOut=out)
        else: # 1d to 2D
            pk.parallel_for(out.shape[0], stretch_fill_impl_1d_into_2d, cols=viewB.shape[1], viewIn=val, viewOut=out)
            
        return out

    # scalar

    if len(viewB.shape) == 1:
        out_1d = pk.View(viewB.shape)
        pk.parallel_for(viewB.shape[0], stretch_fill_impl_scalar_into_1d, scalar=val, viewOut=out_1d)
        return out_1d

    # else 2d
    pk.parallel_for(out.shape[0], stretch_fill_impl_scalar_into_2d, cols=out.shape[1], scalar=val, viewOut=out)
    return out


@pk.workunit
def subtract_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] - viewB[tid]

@pk.workunit
def subtract_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] - viewB[tid]

@pk.workunit
def subtract_impl_2d(tid, cols, viewA, viewB, viewOut):
    for i in range(cols):
        viewOut[tid][i] = viewA[tid][i] - viewB[tid][i]

@pk.workunit
def subtract_impl_scalar_1d(tid, viewA, scalar, viewOut):
    viewOut[tid] = viewA[tid] - scalar

@pk.workunit
def subtract_impl_scalar_2d(tid, cols, viewA, scalar, viewOut):
    for i in range(cols):
        viewOut[tid][i] = viewA[tid][i] - scalar
    

def subtract(viewA, valB, profiler_name: Optional[str] = None):
    """
    Subtracts positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    valB : pykokkos view or scalar
            Input view or scalar value.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    is_scalar = True
    if isinstance(valB, ViewType):
        # if this is a single valued view1D or view2D just count that as a scalar
        for dim in valB.shape:
            if dim != 1:
                is_scalar = False
        
        if is_scalar:
            valB = valB[0] if len(valB.shape) == 1 else valB[0][0]

    if len(viewA.shape) > 2 or (not is_scalar and len(valB.shape) > 2):
        raise NotImplementedError("only 1D and 2D views currently supported for subtract() ufunc.")

    if not is_scalar:

        if viewA.shape != valB.shape and not check_broadcastable_impl(viewA, valB): # if shape is not same check compatibility
            raise ValueError("Views must be broadcastable")

        # check if size is same otherwise broadcast and fix
        if len(viewA.shape) < len(valB.shape) or (len(viewA.shape) == len(valB.shape) and viewA.shape < valB.shape):
            viewA = broadcast_view(viewA, valB)
        elif len(valB.shape) < len(viewA.shape) or (len(viewA.shape) == len(valB.shape) and valB.shape < viewA.shape):
            valB = broadcast_view(valB, viewA)

        if viewA.dtype.__name__ == "float64" and valB.dtype.__name__ == "float64":

            if len(viewA.shape) == 1:
                out = pk.View(viewA.shape, pk.double)
                pk.parallel_for(
                    profiler_name,
                    viewA.shape[0],
                    subtract_impl_1d_double,
                    viewA=viewA,
                    viewB=valB,
                    out=out)

            if len(viewA.shape) == 2:
                out = pk.View([viewA.shape[0], viewA.shape[1]], pk.double)
                pk.parallel_for(
                    profiler_name,
                    viewA.shape[0],
                    subtract_impl_2d,
                    cols=viewA.shape[1],
                    viewA=viewA,
                    viewB=valB,
                    viewOut=out)

        elif viewA.dtype.__name__ == "float32" and valB.dtype.__name__ == "float32":

            if len(viewA.shape) == 1:
                out = pk.View(viewA.shape, pk.float)
                pk.parallel_for(
                    profiler_name,
                    viewA.shape[0],
                    subtract_impl_1d_float,
                    viewA=viewA,
                    viewB=valB,
                    out=out)

            if len(viewA.shape) == 2:
                out = pk.View([viewA.shape[0], viewA.shape[1]], pk.float)
                pk.parallel_for(
                    profiler_name,
                    viewA.shape[0],
                    subtract_impl_2d,
                    cols=viewA.shape[1],
                    viewA=viewA,
                    viewB=valB,
                    viewOut=out)
        else:
            raise RuntimeError("Incompatible Types")
        
        return out
    

    # is scalar subtract -----------------------
    if len(viewA.shape) == 1: # 1D
        out = None
        if viewA.dtype.__name__ == "float64":
            out = pk.View(viewA.shape, pk.double)
        if viewA.dtype.__name__ == "float32":
            out = pk.View(viewA.shape, pk.float)
        
        if out is None: raise RuntimeError("Incompatible Types")

        pk.parallel_for(profiler_name,
                        viewA.shape[0], 
                        subtract_impl_scalar_1d, 
                        viewA=viewA, 
                        scalar=valB, 
                        viewOut=out)
    
    if len(viewA.shape) == 2: # 2D
        out = None
        if viewA.dtype.__name__ == "float64":
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.double)
        if viewA.dtype.__name__ == "float32":
            out = pk.View([viewA.shape[0], viewA.shape[1]], pk.float)
        
        if out is None: raise RuntimeError("Incompatible Types")
        pk.parallel_for(profiler_name,
                        viewA.shape[0], 
                        subtract_impl_scalar_2d, 
                        cols=viewA.shape[1], 
                        viewA=viewA, 
                        scalar=valB, 
                        viewOut=out)

    return out

@pk.workunit
def copyto_impl_2d(tid, viewA, viewB):
    r_idx : int = tid / viewA.extent(1)
    c_idx : int = tid - r_idx * viewA.extent(1)

    viewA[r_idx][c_idx] = viewB[r_idx][c_idx]

@pk.workunit
def copyto_impl_1d(tid, viewA, viewB):
    viewA[tid] = viewB[tid]

def copyto(viewA, viewB, profiler_name: Optional[str] = None):
    '''
    copies values of viewB into valueA for corresponding indicies

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    valB : pykokkos view or scalar
            Input view 

    Returns
    -------
        Void
    '''

    if not isinstance(viewA, ViewType):
        raise ValueError("copyto: Cannot copy to a non-view type")
    if not isinstance(viewB, ViewType):
        raise ValueError("copyto: Cannot copy from a non-view type")
    if viewA.shape != viewB.shape:
        if not check_broadcastable_impl(viewA, viewB): # if shape is not same check compatibility
            raise ValueError("copyto: Views must be broadcastable or of the same size. {} against {}".format(viewA.shape, viewB.shape))
        # check if size is same otherwise broadcast and fix
        viewA = broadcast_view(viewB, viewA)

    # implementation constraint, for now
    if viewA.rank() > 2:
        raise NotImplementedError("copyto: This version of Pykokkos only supports copyto upto 2D views")

    if viewA.rank() == 1:
        pk.parallel_for(profiler_name, viewA.shape[0], copyto_impl_1d, viewA=viewA, viewB=viewB)

    else:   
        outRows = viewA.shape[0]
        outCols = viewA.shape[1] 
        totalThreads = outRows * outCols
        pk.parallel_for(profiler_name, totalThreads, copyto_impl_2d, viewA=viewA, viewB=viewB)

@pk.workunit
def np_matmul_impl_2d_2d(tid, cols, vec_length, viewA, viewB, viewOut):
    r_idx : int = tid / cols
    c_idx : int = tid - r_idx * cols

    for i in range(vec_length):
        viewOut[r_idx][c_idx] += viewA[r_idx][i] * viewB[i][c_idx]

@pk.workunit 
def np_matmul_impl_1d_2d(tid, vec_length, view1D, viewB, viewOut):
    for i in range(vec_length):
        viewOut[tid] += view1D[i] * viewB[i][tid]

@pk.workunit 
def np_matmul_impl_2d_1d(tid, vec_length, viewA, view1D, viewOut):
    for i in range(vec_length):
        viewOut[tid] += viewA[tid][i] * view1D[i]
    
def np_matmul(viewA, viewB, profiler_name: Optional[str] = None):
    """
    Upto 2D Matrix Multiplication of compatible views according to numpy specification

    The behavior depends on the arguments in the following way:
    [*] If both arguments are 2-D they are multiplied like conventional matrices.

    [X] Not implemented yet - If either argument is N-D, N > 2, it is treated as a 
    stack of matrices residing in the last two indexes and broadcast accordingly.

    [*] If the first argument is 1-D, it is promoted to a matrix by prepending a 1 
    to its dimensions. After matrix multiplication the prepended 1 is removed.

    [*] If the second argument is 1-D, it is promoted to a matrix by appending a 1
    to its dimensions. After matrix multiplication the appended 1 is removed.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    Pykokkos view
        Matmul result in a view or 0.0 in case views are empty


    """

    if len(viewA.shape) > 2 or len(viewB.shape) > 2:
        raise NotImplementedError("Matmul only supports upto 2D views")

    viewAType = viewA.dtype.__name__
    viewBType = viewB.dtype.__name__

    if viewAType != viewBType:
        raise RuntimeError("Cannot multiply {} with {} dtype. Types must be same.".format(viewAType, viewBType))

    if not viewA.shape and not viewB.shape:
        return 0.0

    viewALast = viewA.shape[1] if len(viewA.shape) == 2 else viewA.shape[0]
    viewBFirst = viewB.shape[0] if len(viewB.shape) == 2 else viewB.shape[0]

    if viewALast != viewBFirst:
        print(viewALast, viewBFirst)
        raise RuntimeError("Matrix dimensions are not compatible for multiplication: {} and {}".format(viewA.shape, viewB.shape))

    outRows = viewA.shape[0] if len(viewA.shape) == 2 else 1
    outCols = viewB.shape[1] if len(viewB.shape) == 2 else 1
    totalThreads = outRows * outCols

    out = None
    if len(viewA.shape) == 1 or len(viewB.shape) == 1:
        dim = max(outCols, outRows)
        out = pk.View([dim], pk.float if viewBType == "float32" else pk.double)
    else:
        out = pk.View([outRows, outCols], pk.float if viewBType == "float32" else pk.double)

    # CASE 1 BOTH 2D
    if len(viewA.shape) == len(viewB.shape) and len(viewA.shape) == 2:
        pk.parallel_for(profiler_name,
                        totalThreads, 
                        np_matmul_impl_2d_2d, 
                        cols=outCols, 
                        vec_length=viewALast, 
                        viewA=viewA, 
                        viewB=viewB, 
                        viewOut=out)

    elif len(viewA.shape) == 1 and len(viewB.shape) == 1:
        return dot(viewA, viewB)

    # CASE 2 Either is 1D
    elif len(viewA.shape) == 1:
        pk.parallel_for(profiler_name,
                        totalThreads,
                        np_matmul_impl_1d_2d,
                        vec_length= viewA.shape[0],
                        view1D=viewA,
                        viewB=viewB,
                        viewOut=out)

    elif len(viewB.shape) == 1:
        pk.parallel_for(profiler_name,
                        totalThreads,
                        np_matmul_impl_2d_1d,
                        vec_length= viewB.shape[0],
                        viewA=viewA,
                        view1D=viewB,
                        viewOut=out)

    else:
        raise RuntimeError("Unhandled case of matrix multiplication shapes: {} with {}".format(viewA.shape, viewB.shape))

    return out


def matmul(viewA, viewB, profiler_name: Optional[str] = None):
    """
    1D Matrix Multiplication of compatible views

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    Float/Double
        1D Matmul result

    """
    if len(viewA.shape) != 1 or viewA.shape[0] != viewB.shape[0]:
        raise RuntimeError(
            "Input operand 1 has a mismatch in its core dimension (Size {} is different from {})".format(viewA.shape[0], viewB.shape[0]))

    a_dtype_str = viewA.dtype.__name__
    b_dtype_str = viewB.dtype.__name__
    if not(a_dtype_str == "float64" and b_dtype_str == "float64"):
        if not(a_dtype_str == "float32" and b_dtype_str == "float32"):
            raise RuntimeError("Incompatible Types")

    return _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                                    tid=viewA.shape[0],
                                    dtype=viewA.dtype.value,
                                    ndims=1,
                                    op="matmul",
                                    sub_dispatcher=pk.parallel_reduce,
                                    viewA=viewA,
                                    viewB=viewB)

@pk.workunit
def dot_impl_1d_double(tid: int, acc: pk.Acc[pk.double], viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double]):
    acc += viewA[tid] * viewB[tid]

@pk.workunit
def dot_impl_1d_float(tid: int, acc: pk.Acc[pk.float], viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float]):
    acc += viewA[tid] * viewB[tid]

def dot(viewA, viewB):
    """
    1D Matrix Multiplication of compatible views

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    Float/Double

    """

    if len(viewA.shape) == 0 and len(viewB.shape) == 0:
        return 0

    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views supported for dot() ufunc.")

    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.parallel_reduce(
                viewA.shape[0],
                dot_impl_1d_double,
                viewA=viewA,
                viewB=viewB)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.parallel_reduce(
                viewA.shape[0],
                dot_impl_1d_float,
                viewA=viewA,
                viewB=viewB)
    else:
        raise RuntimeError("Incompatible Types")
    return out

@pk.workunit
def divide_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] / viewB[tid % viewB.extent(0)]


@pk.workunit
def divide_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] / viewB[tid]


@pk.workunit
def divide_impl_2d_1d_double(tid: int, viewA: pk.View2D[pk.double], viewB: pk.View1D[pk.double], out: pk.View2D[pk.double]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] / viewB[i % viewB.extent(0)]


def divide(viewA, viewB, profiler_name: Optional[str] = None):
    """
    Divides positionally corresponding elements
    of viewA with elements of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if not isinstance(viewB, pk.ViewType) and not isinstance(viewB, pk.ViewType):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if viewA.rank() == 2:
        out = pk.View(viewA.shape, pk.double)
        pk.parallel_for(
            profiler_name,
            viewA.shape[0],
            divide_impl_2d_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            profiler_name,
            viewA.shape[0],
            divide_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            profiler_name,
            viewA.shape[0],
            divide_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def negative_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid] * -1


@pk.workunit
def negative_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid] * -1

def negative(view, profiler_name: Optional[str] = None):
    """
    Element-wise negative of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(view.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for negative() ufunc.")
    if view.dtype.__name__ == "float64":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(profiler_name, view.shape[0], negative_impl_1d_double, view=view, out=out)
    elif view.dtype.__name__ == "float32":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(profiler_name, view.shape[0], negative_impl_1d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


def positive(view):
    """
    Element-wise positive of the view;
    Essentially returns a copy of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if view.shape == ():
        out = pk.View((), dtype=view.dtype)
    else:
        out = pk.View([*view.shape], dtype=view.dtype)
    out[...] = view
    return out


@pk.workunit
def power_impl_scalar_double(tid:int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = pow(viewA[0], viewB[tid])


@pk.workunit
def power_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = pow(viewA[tid], viewB[tid])


@pk.workunit
def power_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = pow(viewA[tid], viewB[tid])

@pk.workunit
def power_impl_2d_double(tid: int, viewA: pk.View2D[pk.double], viewB: pk.View1D[pk.double], out: pk.View2D[pk.double]):
    for i in range(viewA.extent(1)):
        out[tid][i] = pow(viewA[tid][i], viewB[i % viewB.extent(0)])


def power(viewA, viewB):
    """
    Returns a view with each val in viewA raised
    to the positionally corresponding power in viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if not isinstance(viewB, pk.ViewType):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if isinstance(viewA, int):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewA
        viewA = view_temp

        out = pk.View([viewB.shape[0]], pk.double)
        pk.parallel_for(
            viewB.shape[0],
            power_impl_scalar_double,
            viewA=viewA,
            viewB=viewB,
            out=out)
    elif viewA.rank() == 2:
        out = pk.View(viewA.shape, pk.double)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_2d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)
    elif viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def fmod_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = fmod(viewA[tid], viewB[tid])


@pk.workunit
def fmod_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = fmod(viewA[tid], viewB[tid])


def fmod(viewA, viewB):
    """
    Element-wise remainder of division when element of viewA is
    divided by positionally corresponding element of viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("fmod() ufunc only supports 1D views")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmod_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            fmod_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def square_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_uint8(tid: int, view: pk.View1D[pk.uint8], out: pk.View1D[pk.uint8]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_uint8(tid: int, view: pk.View2D[pk.uint8], out: pk.View2D[pk.uint8]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_uint16(tid: int, view: pk.View1D[pk.uint16], out: pk.View1D[pk.uint16]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_uint16(tid: int, view: pk.View2D[pk.uint16], out: pk.View2D[pk.uint16]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_uint32(tid: int, view: pk.View1D[pk.uint32], out: pk.View1D[pk.uint32]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_uint32(tid: int, view: pk.View2D[pk.uint32], out: pk.View2D[pk.uint32]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_uint64(tid: int, view: pk.View1D[pk.uint64], out: pk.View1D[pk.uint64]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_uint64(tid: int, view: pk.View2D[pk.uint64], out: pk.View2D[pk.uint64]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_int8(tid: int, view: pk.View1D[pk.int8], out: pk.View1D[pk.int8]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_int8(tid: int, view: pk.View2D[pk.int8], out: pk.View2D[pk.int8]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_int16(tid: int, view: pk.View1D[pk.int16], out: pk.View1D[pk.int16]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_int16(tid: int, view: pk.View2D[pk.int16], out: pk.View2D[pk.int16]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_int32(tid: int, view: pk.View1D[pk.int32], out: pk.View1D[pk.int32]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_int32(tid: int, view: pk.View2D[pk.int32], out: pk.View2D[pk.int32]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]


@pk.workunit
def square_impl_1d_int64(tid: int, view: pk.View1D[pk.int64], out: pk.View1D[pk.int64]):
    out[tid] = view[tid] * view[tid]


@pk.workunit
def square_impl_2d_int64(tid: int, view: pk.View2D[pk.int64], out: pk.View2D[pk.int64]):
    for i in range(view.extent(1)):
        out[tid][i] = view[tid][i] * view[tid][i]

def square(view):
    """
    Squares argument element-wise

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(view.shape) > 2:
        raise NotImplementedError("only up to 2D views currently supported for square() ufunc.")
    out = pk.View(view.shape, view.dtype)
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_float, view=view, out=out)
    elif "uint8" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint8, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint8, view=view, out=out)
    elif "uint16" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint16, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint16, view=view, out=out)
    elif "uint32" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint32, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint32, view=view, out=out)
    elif "uint64" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint64, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint64, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint64, view=view, out=out)
    elif "int8" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int8, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int8, view=view, out=out)
    elif "int16" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int16, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int16, view=view, out=out)
    elif "int32" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int32, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int32, view=view, out=out)
    elif "int64" in view.dtype.__name__:
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int64, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int64, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int64, view=view, out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def greater_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] > viewB[tid]


@pk.workunit
def greater_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] > viewB[tid]


def greater(viewA, viewB):
    """
    Return the truth value of viewA > viewB element-wise.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view (uint8)
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("greater() ufunc only supports 1D views")
    out = pk.View([viewA.shape[0]], pk.uint8)
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        pk.parallel_for(
            viewA.shape[0],
            greater_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        pk.parallel_for(
            viewA.shape[0],
            greater_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def logaddexp_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double],):
    out[tid] = log(exp(viewA[tid]) + exp(viewB[tid]))


@pk.workunit
def logaddexp_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float],):
    out[tid] = log(exp(viewA[tid]) + exp(viewB[tid]))


def logaddexp(viewA, viewB):
    """
    Return a view with log(exp(a) + exp(b)) calculate for
    positionally corresponding elements in viewA and viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logaddexp() ufunc.")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out

def true_divide(viewA, viewB):
    """
    true_divide is an alias of divide

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """

    return divide(viewA, viewB)


@pk.workunit
def logaddexp2_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double],):
    out[tid] = log2(pow(2, viewA[tid]) + pow(2, viewB[tid]))


@pk.workunit
def logaddexp2_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float],):
    out[tid] = log2(pow(2, viewA[tid]) + pow(2, viewB[tid]))


def logaddexp2(viewA, viewB):
    """
    Return a view with log(pow(2, a) + pow(2, b)) calculated for
    positionally corresponding elements in viewA and viewB

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logaddexp2() ufunc.")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp2_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp2_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def floor_divide_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] // viewB[tid]


@pk.workunit
def floor_divide_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] // viewB[tid]


def floor_divide(viewA, viewB):
    """
    Divides positionally corresponding elements
    of viewA with elements of viewB and floors the result

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for floor_divide() ufunc.")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            floor_divide_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            floor_divide_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


def sin(view, profiler_name: Optional[str] = None):
    """
    Element-wise trigonometric sine of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("sin() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=dtype)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="sin",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


@pk.workunit
def cos_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = cos(view[tid])


@pk.workunit
def cos_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[tid][i] = cos(view[tid][i])


@pk.workunit
def cos_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = cos(view[tid])


@pk.workunit
def cos_impl_2d_float(tid: int, view: pk.View2D[pk.float], out: pk.View2D[pk.float]):
    for i in range(view.extent(1)):
        out[tid][i] = cos(view[tid][i])


def cos(view):
    """
    Element-wise trigonometric cosine of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(view.shape) > 2:
        raise NotImplementedError("only up to 2D views currently supported for cos() ufunc.")
    if "double" in view.dtype.__name__ or "float64" in view.dtype.__name__:
        out = pk.View([*view.shape], dtype=pk.float64)
        if len(view.shape) == 1:
            pk.parallel_for(view.shape[0], cos_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], cos_impl_2d_double, view=view, out=out)
    elif "float" in view.dtype.__name__:
        out = pk.View([*view.shape], dtype=pk.float32)
        if len(view.shape) == 1:
            pk.parallel_for(view.shape[0], cos_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], cos_impl_2d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


def tan(view, profiler_name: Optional[str] = None):
    """
    Element-wise tangent of the view

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("tan() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=dtype)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="tan",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


@pk.workunit
def logical_and_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] and viewB[tid]


@pk.workunit
def logical_and_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] and viewB[tid]


def logical_and(viewA, viewB):
    """
    Return the element-wise truth value of viewA AND viewB.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view (uint8)
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logical_and() ufunc.")
    out = pk.View([viewA.shape[0]], pk.uint8)
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        pk.parallel_for(
            viewA.shape[0],
            logical_and_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        pk.parallel_for(
            viewA.shape[0],
            logical_and_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def logical_or_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] or viewB[tid]


@pk.workunit
def logical_or_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = viewA[tid] or viewB[tid]


def logical_or(viewA, viewB):
    """
    Return the element-wise truth value of viewA OR viewB.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view (uint8)
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logical_or() ufunc.")
    out = pk.View([viewA.shape[0]], pk.uint8)
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        pk.parallel_for(
            viewA.shape[0],
            logical_or_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        pk.parallel_for(
            viewA.shape[0],
            logical_or_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def logical_xor_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = bool(viewA[tid]) ^ bool(viewB[tid])


@pk.workunit
def logical_xor_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = bool(viewA[tid]) ^ bool(viewB[tid])


def logical_xor(viewA, viewB):
    """
    Return the element-wise truth value of viewA XOR viewB.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view (uint8)
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logical_xor() ufunc.")
    out = pk.View([viewA.shape[0]], pk.uint8)
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        pk.parallel_for(
            viewA.shape[0],
            logical_xor_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        pk.parallel_for(
            viewA.shape[0],
            logical_xor_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def logical_not_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.uint8]):
    out[tid] = not view[tid]


@pk.workunit
def logical_not_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.uint8]):
    out[tid] = not view[tid]


def logical_not(view):
    """
    Element-wise logical_not of the view.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view (uint8)
           Output view.

    """
    if len(view.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for logical_not() ufunc.")
    out = pk.View([view.shape[0]], pk.uint8)
    if view.dtype.__name__ == "float64":
        pk.parallel_for(view.shape[0], logical_not_impl_1d_double, view=view, out=out)
    elif view.dtype.__name__ == "float32":
        pk.parallel_for(view.shape[0], logical_not_impl_1d_float, view=view, out=out)
    return out


@pk.workunit
def fmax_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = fmax(viewA[tid], viewB[tid])


@pk.workunit
def fmax_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = fmax(viewA[tid], viewB[tid])


def fmax(viewA, viewB):
    """
    Return the element-wise fmax.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("fmax() ufunc only supports 1D views")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmax_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            fmax_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def fmin_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = fmin(viewA[tid], viewB[tid])


@pk.workunit
def fmin_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = fmin(viewA[tid], viewB[tid])


def fmin(viewA, viewB):
    """
    Return the element-wise fmin.

    Parameters
    ----------
    viewA : pykokkos view
            Input view.
    viewB : pykokkos view
            Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("fmax() ufunc only supports 1D views")
    if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmin_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif viewA.dtype.__name__ == "float32" and viewB.dtype.__name__ == "float32":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            fmin_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


def exp(view, profiler_name: Optional[str] = None):
    """
    Element-wise exp of the view.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("exp() ufunc only supports up to 2D views")
    if view.size == 0:
        return view
    out = pk.View([*view.shape], dtype=dtype)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="exp",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


@pk.workunit
def exp2_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = pow(2, view[tid])


@pk.workunit
def exp2_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = pow(2, view[tid])


def exp2(view):
    """
    Element-wise 2**x of the view.

    Parameters
    ----------
    view : pykokkos view
           Input view.

    Returns
    -------
    out : pykokkos view
           Output view.

    """
    if len(view.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for exp2() ufunc.")
    if view.dtype.__name__ == "float64":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], exp2_impl_1d_double, view=view, out=out)
    elif view.dtype.__name__ == "float32":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], exp2_impl_1d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


# TODO: Implement parallel max reduction with index
def argmax(view, axis=None):
    if isinstance(axis, pk.ViewType):
        raise NotImplementedError

    res = np.argmax(view, axis=axis)
    view = pk.View(res.shape, pk.int32)
    view[:] = res

    return view

# TODO: Implement parallel sorting + filtering 
def unique(view):
    res = np.unique(view)
    view = pk.View(res.shape, pk.double)
    view[:] = res

    return view

@pk.workunit
def var_impl_2d_axis0_double(tid: int, view: pk.View2D[pk.double], view_mean:pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(view.extent(0)):
        out[tid] += (pow(view[i][tid] - view_mean[tid], 2)) / view.extent(0)


@pk.workunit
def var_imple_2d_axis1_double(tid: int, view: pk.View2D[pk.double], view_mean:pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(view.extent(1)):
        out[tid] += (pow(view[tid][i] - view_mean[tid], 2)) / view.extent(1)

@pk.workunit
def var_impl_1d(tid, acc, view, mean):
    acc += pow(view[tid] - mean, 2) / view.extent(0)

def var(view, axis=None, profiler_name: Optional[str] = None): # population
    if isinstance(axis, pk.ViewType):
        raise NotImplementedError

    if view.rank() > 2:
        raise NotImplementedError("Current version of Pykokkos only supports variance for upto 2D views")
    
    if view.rank() == 2: # legacy code
        if view.dtype.__name__ == "float64":
            if axis == 0:
                view_mean = mean(view, 0, profiler_name)
                out = pk.View([view.shape[1]], pk.double)
                pk.parallel_for(profiler_name, view.shape[1], var_impl_2d_axis0_double, view=view, view_mean=view_mean, out=out)
                return out
            else:
                view_mean = mean(view, 1, profiler_name)
                out = pk.View([view.shape[0]], pk.double)
                pk.parallel_for(profiler_name, view.shape[0], var_imple_2d_axis1_double, view=view, view_mean=view_mean, out=out)
                return out
        else:
            raise RuntimeError("Incompatible Types")
    elif view.rank() == 1: # newer impl
        mean_val = mean(view, profiler_name)
        return pk.parallel_reduce(profiler_name, view.shape[0], var_impl_1d, view=view, mean=mean_val)
    else:
        raise RuntimeError("Unexpected view of shape {}".format(view.shape))


@pk.workunit
def mean_impl_1d_axis0_double(tid: int, view: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(view.extent(0)):
        out[tid] += (view[i][tid] / view.extent(0))


@pk.workunit
def mean_impl_1d_axis1_double(tid: int, view: pk.View2D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = 0
    for i in range(view.extent(1)):
        out[tid] += (view[tid][i] / view.extent(1))

@pk.workunit
def mean_impl_1d(tid, acc, view):
    acc += view[tid] / view.extent(0)

def mean(view, axis=None, profiler_name: Optional[str] = None):
    if isinstance(axis, pk.ViewType):
        raise NotImplementedError

    if view.rank() > 2:
        raise NotImplementedError("Current version of Pykokkos only supports variance for upto 2D views")

    if view.rank() == 2:
        if view.dtype.__name__ == "float64": # legacy
            if axis == 0:
                out = pk.View([view.shape[1]], pk.double)
                pk.parallel_for(profiler_name, view.shape[1], mean_impl_1d_axis0_double, view=view, out=out)
                return out
            else:
                out = pk.View([view.shape[0]], pk.double)
                pk.parallel_for(profiler_name, view.shape[0], mean_impl_1d_axis1_double, view=view, out=out)

                return out
        else:
            raise RuntimeError("Incompatible Types")

    elif view.rank() == 1:
        return pk.parallel_reduce(profiler_name, view.shape[0], mean_impl_1d, view=view)
    else:
        raise RuntimeError("Unexpected view of shape {}".format(view.shape))



@pk.workunit
def in1d_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.int8]):
    out[tid] = 0
    for i in range(viewB.extent(0)):
        if viewB[i] == viewA[tid]:
            out[tid] = 1
            break

def in1d(viewA, viewB):
    if viewA.dtype.__name__ == "float64":
        out = pk.View(viewA.shape, pk.int8)
        pk.parallel_for(
            viewA.shape[0],
            in1d_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")

    return out


@pk.workunit
def transpose_impl_2d_double(tid: int, view: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(view.extent(1)):
        out[i][tid] = view[tid][i]


def transpose(view):
    if view.rank() == 1:
        return view

    if view.rank() == 2:
        if view.dtype.__name__ == "float64":
            out = pk.View(view.shape[::-1], pk.double)
            pk.parallel_for(view.shape[0], transpose_impl_2d_double, view=view, out=out)
            return out
    
    raise RuntimeError("Transpose supports 2D views only")


@pk.workunit
def hstack_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    if tid >= viewA.extent(0):
        out[tid] = viewB[tid - viewA.extent(0)]
    else:
        out[tid] = viewA[tid]

@pk.workunit
def hstack_impl_2d_double(tid: int, viewA: pk.View2D[pk.double], viewB: pk.View2D[pk.double], out: pk.View2D[pk.double]):
    for i in range(out.extent(1)):
        if i >= viewA.extent(1):
            out[tid][i] = viewB[tid][i - viewA.extent(1)]
        else:
            out[tid][i] = viewA[tid][i]


def hstack(viewA, viewB):
    if viewA.shape != viewB.shape:
        raise RuntimeError("All the input view dimensions for the concatenation axis must match exactly")

    if viewA.rank() == 2 and viewB.rank() == 2:
        if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
            out = pk.View([viewA.shape[0], viewA.shape[1] * 2], pk.double)
            pk.parallel_for(
                out.shape[0],
                hstack_impl_2d_double,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            raise RuntimeError("hstack supports 2D views of type double only")
    elif viewA.rank() == 1 and viewB.rank() == 1:
        if viewA.dtype.__name__ == "float64" and viewB.dtype.__name__ == "float64":
            out = pk.View([viewA.shape[0] + viewB.shape[0]], pk.double)
            pk.parallel_for(
                out.shape[0],
                hstack_impl_1d_double,
                viewA=viewA,
                viewB=viewB,
                out=out)
        else:
            raise RuntimeError("hstack supports 1D views of type double only")
    else:
        raise RuntimeError("hstack supports views of same shape (1D and 2D) only")
    
    return out


@pk.workunit
def index_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.int32], out: pk.View1D[pk.double]):
    out[tid] = viewA[viewB[tid]]


def index(viewA, viewB):
    if viewB.dtype == pk.int32:
        out = pk.View(viewB.shape, pk.double)
        pk.parallel_for(
            viewB.shape[0],
            index_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


def isnan(view, profiler_name: Optional[str] = None):
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("isnan() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=pk.bool)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    if view.ndim == 0:
        new_view = pk.View([1], dtype=view.dtype)
        new_view[0] = view
        view = new_view
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isnan",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def isinf(view, profiler_name: Optional[str] = None):
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("isinf() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=pk.bool)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isinf",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def equal(view1, view2, profiler_name: Optional[str] = None):
    """
    Computes the truth value of ``view1_i`` == ``view2_i`` for each element
    ``x1_i`` of the input view ``view1`` with the respective element ``x2_i``
    of the input view ``view2``.


    Parameters
    ----------
    view1 : pykokkos view
            Input view. May have any data type.
    view2 : pykokkos view
            Input view. May have any data type, but must be shape-compatible
            with ``view1`` via broadcasting.

    Returns
    -------
    out : pykokkos view (bool)
           Output view.
    """
    if view1.size == 0 and view2.size == 0:
        ret =  pk.View((), dtype=pk.bool)
        ret[...] = 1
        return ret
    view1, view2 = _broadcast_views(view1, view2)
    dtype1 = view1.dtype
    dtype2 = view2.dtype
    view1, view2, effective_dtype = _typematch_views(view1, view2)
    ndims = len(view1.shape)
    if ndims > 5:
        raise NotImplementedError("equal() ufunc only supports up to 5D views")
    out = pk.View([*view1.shape], dtype=pk.bool)
    if view1.shape == ():
        tid = 1
    else:
        tid = view1.shape[0]
    if isinstance(view1, pk.Subview):
        new_view = pk.View((), dtype=view1.dtype)
        new_view[:] = view1.data
        view1 = new_view
    if isinstance(view2, pk.Subview):
        new_view = pk.View((), dtype=view2.dtype)
        new_view[:] = view2.data
        view2 = new_view
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=effective_dtype,
                             ndims=ndims,
                             op="equal",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view1=view1,
                             view2=view2)
    return out


def isfinite(view, profiler_name: Optional[str] = None):
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("isfinite() ufunc only supports up to 2D views")
    if view.size == 0:
        out = pk.View(view.shape, dtype=pk.bool)
        return out
    out = pk.View([*view.shape], dtype=pk.bool)
    if view.shape == ():
        new_view = pk.View([1], dtype=dtype)
        new_view[:] = view
        view = new_view
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isfinite",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def round(view, profiler_name: Optional[str] = None):
    """
    Rounds each element of the input view to the nearest integer-valued number.

    Parameters
    ----------
    view : pykokkos view
           Should have a numeric data type.

    Returns
    -------
    out: pykokkos view
         A view containing the rounded result for each element in
         the input view. The returned view must have the same data
         type as the input view.

    Notes
    -----
    If view element ``i`` is already integer-valued, the result is ``i``.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    dtype_str = str(dtype)
    if "int" in dtype_str:
        # special case defined in API std
        return view
    out = pk.View(view.shape, dtype=dtype)
    if ndims > 3:
        raise NotImplementedError("only up to 3D views currently supported for round() ufunc.")
        
    _supported_types_check(dtype_str, {"double", "float64", "float"})

    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="round",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def trunc(view, profiler_name: Optional[str] = None):
    """
    Rounds each element ``i`` of the input view to the integer-valued number
    that is closest to but no greater than ``i``.

    Parameters
    ----------
    view : pykokkos view
           Should have a numeric data type.

    Returns
    -------
    out: pykokkos view
         A view containing the rounded result for each element in
         the input view. The returned view must have the same data
         type as the input view.

    Notes
    -----
    If view element ``i`` is already integer-valued, the result is ``i``.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    dtype_str = str(dtype)
    if "int" in dtype_str:
        # special case defined in API std
        return view
    out = pk.View(view.shape, dtype=dtype)
    if ndims > 3:
        raise NotImplementedError("only up to 3D views currently supported for trunc() ufunc.")

    _supported_types_check(dtype_str, {"double", "float64", "float"})

    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="trunc",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def ceil(view, profiler_name: Optional[str] = None):
    """
    Rounds each element of the input view to the smallest (i.e., closest to -infinity)
    integer-valued number that is not less than a given element.

    Parameters
    ----------
    view : pykokkos view
           Should have a numeric data type.

    Returns
    -------
    out: pykokkos view
         A view containing the rounded result for each element in
         the input view. The returned view must have the same data
         type as the input view.

    Notes
    -----
    If view element ``i`` is already integer-valued, the result is ``i``.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    dtype_str = str(dtype)
    if "int" in dtype_str:
        # special case defined in API std
        return view
    out = pk.View(view.shape, dtype=dtype)
    if ndims > 3:
        raise NotImplementedError("only up to 3D views currently supported for ceil() ufunc.")

    _supported_types_check(dtype_str, {"double", "float64", "float"})

    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="ceil",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def floor(view, profiler_name: Optional[str] = None):
    """
    Rounds each element of the input view to the greatest (i.e., closest to +infinity)
    integer-valued number that is not greater than a given element.

    Parameters
    ----------
    view : pykokkos view
           Should have a numeric data type.

    Returns
    -------
    out: pykokkos view
         A view containing the rounded result for each element in
         the input view. The returned view must have the same data
         type as the input view.

    Notes
    -----
    If view element ``i`` is already integer-valued, the result is ``i``.

    """
    dtype = view.dtype
    ndims = len(view.shape)
    dtype_str = str(dtype)
    if "int" in dtype_str:
        # special case defined in API std
        return view
    out = pk.View(view.shape, dtype=dtype)
    if ndims > 3:
        raise NotImplementedError("only up to 3D views currently supported for floor() ufunc.")

    _supported_types_check(dtype_str, {"double", "float64", "float"})

    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="floor",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def tanh(view, profiler_name: Optional[str] = None):
    """
    Calculates an approximation to the hyperbolic tangent for each element x_i of the input view.

    Parameters
    ----------
    view : pykokkos view
            Input view whose elements each represent a hyperbolic angle. Should have a floating-point data type.

    Returns
    -------
    y : pykokkos view
        A view containing the hyperbolic tangent of each element in the input view. The returned view must
        have a floating-point data type determined by type promotion rules.
    """
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("tanh() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=dtype)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(profiler_name=profiler_name,
                             tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="tanh",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out
