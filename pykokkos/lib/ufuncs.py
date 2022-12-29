import re
import math
from inspect import getmembers, isfunction

import numpy as np
import pykokkos as pk
from pykokkos.lib import ufunc_workunits

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


def _ufunc_kernel_dispatcher(tid,
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
    ret = sub_dispatcher(tid, desired_workunit, **kwargs)
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
                view2_new[:] = view2
                view2 = view2_new
            else:
                effective_dtype = dtype2
                view1_new = pk.View([*view1.shape], dtype=effective_dtype)
                view1_new[:] = view1
                view1 = view1_new
    return view1, view2, effective_dtype


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
    _ufunc_kernel_dispatcher(tid=view.shape[0],
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
    if not isinstance(view, pk.View):
        return math.log(view)

    if len(view.shape) > 2:
        raise NotImplementedError("log() ufunc only supports up to 2D views")

    out = pk.View(view.shape, view.dtype)
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log_impl_2d_float, view=view, out=out)
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
    # TODO: support complex types when they
    # are available in pykokkos?
    if len(view.shape) > 2:
        raise NotImplementedError("only up to 2D views currently supported for sqrt() ufunc.")
    out = pk.View(view.shape, view.dtype)
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, sqrt_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], sqrt_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], sqrt_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log2_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log2_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log2_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            # NOTE: is this really worth sending to a kernel?
            pk.parallel_for(1, log10_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log10_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log10_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, log1p_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], log1p_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], log1p_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
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
    elif "float" in str(view.dtype):
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
    elif "uint32" in str(view.dtype):
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
    elif "uint16" in str(view.dtype):
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
    elif "int16" in str(view.dtype):
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
    elif "int32" in str(view.dtype):
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
    elif "uint64" in str(view.dtype):
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
    elif "int64" in str(view.dtype):
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
    elif "uint8" in str(view.dtype):
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
    elif "int8" in str(view.dtype):
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
def add_impl_2d_1d_double(tid: int, viewA: pk.View2D[pk.double], viewB: pk.View1D[pk.double], out: pk.View2D[pk.double]):
    for i in range(viewA.extent(1)):
        out[tid][i] = viewA[tid][i] + viewB[i % viewB.extent(0)]


def add(viewA, viewB):
    """
    Sums positionally corresponding elements
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
    if not isinstance(viewB, pk.View):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if viewA.rank() == 2:
        out = pk.View(viewA.shape, pk.double)
        pk.parallel_for(
            viewA.shape[0],
            add_impl_2d_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            add_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            add_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
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


def multiply(viewA, viewB):
    """
    Multiplies positionally corresponding elements
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

    if not isinstance(viewB, pk.View):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for mulitply() ufunc.")

    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            multiply_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            multiply_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


@pk.workunit
def subtract_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = viewA[tid] - viewB[tid]


@pk.workunit
def subtract_impl_1d_float(tid: int, viewA: pk.View1D[pk.float], viewB: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = viewA[tid] - viewB[tid]


def subtract(viewA, viewB):
    """
    Subtracts positionally corresponding elements
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
    if len(viewA.shape) > 1 or len(viewB.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for subtract() ufunc.")
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            subtract_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
            viewA.shape[0],
            subtract_impl_1d_float,
            viewA=viewA,
            viewB=viewB,
            out=out)
    else:
        raise RuntimeError("Incompatible Types")
    return out


def matmul(viewA, viewB):
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

    a_dtype_str = str(viewA.dtype)
    b_dtype_str = str(viewB.dtype)
    if not(a_dtype_str == "DataType.double" and b_dtype_str == "DataType.double"):
        if not(a_dtype_str == "DataType.float" and b_dtype_str == "DataType.float"):
            raise RuntimeError("Incompatible Types")

    return _ufunc_kernel_dispatcher(tid=viewA.shape[0],
                                    dtype=viewA.dtype.value,
                                    ndims=1,
                                    op="matmul",
                                    sub_dispatcher=pk.parallel_reduce,
                                    viewA=viewA,
                                    viewB=viewB)


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


def divide(viewA, viewB):
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
    if not isinstance(viewB, pk.View) and not isinstance(viewB, pk.Subview):
        view_temp = pk.View([1], pk.double)
        view_temp[0] = viewB
        viewB = view_temp

    if viewA.rank() == 2:
        out = pk.View(viewA.shape, pk.double)
        pk.parallel_for(
            viewA.shape[0],
            divide_impl_2d_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            divide_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
        out = pk.View([viewA.shape[0]], pk.float)
        pk.parallel_for(
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

def negative(view):
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
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], negative_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], negative_impl_1d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


@pk.workunit
def positive_impl_1d_double(tid: int, view: pk.View1D[pk.double], out: pk.View1D[pk.double]):
    out[tid] = view[tid]


@pk.workunit
def positive_impl_1d_float(tid: int, view: pk.View1D[pk.float], out: pk.View1D[pk.float]):
    out[tid] = view[tid]

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
    if len(view.shape) > 1:
        raise NotImplementedError("only 1D views currently supported for positive() ufunc.")
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], positive_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], positive_impl_1d_float, view=view, out=out)
    else:
        raise NotImplementedError
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
    if not isinstance(viewB, pk.View):
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
    elif str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            power_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmod_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_float, view=view, out=out)
    elif "uint8" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint8, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint8, view=view, out=out)
    elif "uint16" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint16, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint16, view=view, out=out)
    elif "uint32" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint32, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint32, view=view, out=out)
    elif "uint64" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_uint64, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_uint64, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_uint64, view=view, out=out)
    elif "int8" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int8, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int8, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int8, view=view, out=out)
    elif "int16" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int16, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int16, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int16, view=view, out=out)
    elif "int32" in str(view.dtype):
        if view.shape == ():
            pk.parallel_for(1, square_impl_1d_int32, view=view, out=out)
        elif len(view.shape) == 1:
            pk.parallel_for(view.shape[0], square_impl_1d_int32, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], square_impl_2d_int32, view=view, out=out)
    elif "int64" in str(view.dtype):
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        pk.parallel_for(
            viewA.shape[0],
            greater_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            logaddexp2_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            floor_divide_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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


def sin(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
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
    if "double" in str(view.dtype) or "float64" in str(view.dtype):
        out = pk.View([*view.shape], dtype=pk.float64)
        if len(view.shape) == 1:
            pk.parallel_for(view.shape[0], cos_impl_1d_double, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], cos_impl_2d_double, view=view, out=out)
    elif "float" in str(view.dtype):
        out = pk.View([*view.shape], dtype=pk.float32)
        if len(view.shape) == 1:
            pk.parallel_for(view.shape[0], cos_impl_1d_float, view=view, out=out)
        elif len(view.shape) == 2:
            pk.parallel_for(view.shape[0], cos_impl_2d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


def tan(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        pk.parallel_for(
            viewA.shape[0],
            logical_and_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        pk.parallel_for(
            viewA.shape[0],
            logical_or_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        pk.parallel_for(
            viewA.shape[0],
            logical_xor_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(view.dtype) == "DataType.double":
        pk.parallel_for(view.shape[0], logical_not_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmax_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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
    if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
        out = pk.View([viewA.shape[0]], pk.double)
        pk.parallel_for(
            viewA.shape[0],
            fmin_impl_1d_double,
            viewA=viewA,
            viewB=viewB,
            out=out)

    elif str(viewA.dtype) == "DataType.float" and str(viewB.dtype) == "DataType.float":
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


def exp(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
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
    if str(view.dtype) == "DataType.double":
        out = pk.View([view.shape[0]], pk.double)
        pk.parallel_for(view.shape[0], exp2_impl_1d_double, view=view, out=out)
    elif str(view.dtype) == "DataType.float":
        out = pk.View([view.shape[0]], pk.float)
        pk.parallel_for(view.shape[0], exp2_impl_1d_float, view=view, out=out)
    else:
        raise NotImplementedError
    return out


# TODO: Implement parallel max reduction with index
def argmax(view, axis=None):
    if isinstance(axis, pk.View):
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


def var(view, axis=None):
    if isinstance(axis, pk.View):
        raise NotImplementedError

    if str(view.dtype) == "DataType.double":
        if axis == 0:
            view_mean = mean(view, 0)
            out = pk.View([view.shape[1]], pk.double)
            pk.parallel_for(view.shape[1], var_impl_2d_axis0_double, view=view, view_mean=view_mean, out=out)
            return out
        else:
            view_mean = mean(view, 1)
            out = pk.View([view.shape[0]], pk.double)
            pk.parallel_for(view.shape[0], var_imple_2d_axis1_double, view=view, view_mean=view_mean, out=out)
            return out


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


def mean(view, axis=None):
    if isinstance(axis, pk.View):
        raise NotImplementedError

    if str(view.dtype) == "DataType.double":
        if axis == 0:
            out = pk.View([view.shape[1]], pk.double)
            pk.parallel_for(view.shape[1], mean_impl_1d_axis0_double, view=view, out=out)
            return out
        else:
            out = pk.View([view.shape[0]], pk.double)
            pk.parallel_for(view.shape[0], mean_impl_1d_axis1_double, view=view, out=out)

            return out
    else:
        raise RuntimeError("Incompatible Types")


@pk.workunit
def in1d_impl_1d_double(tid: int, viewA: pk.View1D[pk.double], viewB: pk.View1D[pk.double], out: pk.View1D[pk.int8]):
    out[tid] = 0
    for i in range(viewB.extent(0)):
        if viewB[i] == viewA[tid]:
            out[tid] = 1
            break

def in1d(viewA, viewB):
    if str(viewA.dtype) == "DataType.double":
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
        if str(view.dtype) == "DataType.double":
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
        if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
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
        if str(viewA.dtype) == "DataType.double" and str(viewB.dtype) == "DataType.double":
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


def isnan(view):
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("isnan() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=pk.bool)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isnan",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def isinf(view):
    dtype = view.dtype
    ndims = len(view.shape)
    if ndims > 2:
        raise NotImplementedError("isinf() ufunc only supports up to 2D views")
    out = pk.View([*view.shape], dtype=pk.bool)
    if view.shape == ():
        tid = 1
    else:
        tid = view.shape[0]
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isinf",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def equal(view1, view2):
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
        return pk.View((), dtype=pk.bool)
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=effective_dtype,
                             ndims=ndims,
                             op="equal",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view1=view1,
                             view2=view2)
    return out


def isfinite(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="isfinite",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def round(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="round",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def trunc(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="trunc",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def ceil(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="ceil",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def floor(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="floor",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out


def tanh(view):
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
    _ufunc_kernel_dispatcher(tid=tid,
                             dtype=dtype,
                             ndims=ndims,
                             op="tanh",
                             sub_dispatcher=pk.parallel_for,
                             out=out,
                             view=view)
    return out
