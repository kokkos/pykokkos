import re
import math
from inspect import getmembers, isfunction
from typing import Optional

import numpy as np
import pykokkos as pk
from pykokkos.lib import ufunc_workunits
from pykokkos.interface import ViewType

kernel_dict = dict(getmembers(ufunc_workunits, isfunction))


def _ufunc_kernel_dispatcher(
    profiler_name: Optional[str], tid, dtype, ndims, op, sub_dispatcher, **kwargs
):
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
        if ("int" in res1_dtype_str and "int" in res2_dtype_str) or (
            "float" in res1_dtype_str and "float" in res2_dtype_str
        ):
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


def _equal(view1, view2, profiler_name: Optional[str] = None):
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
        ret = pk.View((), dtype=pk.bool)
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
    _ufunc_kernel_dispatcher(
        profiler_name=profiler_name,
        tid=tid,
        dtype=effective_dtype,
        ndims=ndims,
        op="equal",
        sub_dispatcher=pk.parallel_for,
        out=out,
        view1=view1,
        view2=view2,
    )
    return out

def _isnan(view, profiler_name: Optional[str] = None):
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
    _ufunc_kernel_dispatcher(
        profiler_name=profiler_name,
        tid=tid,
        dtype=dtype,
        ndims=ndims,
        op="isnan",
        sub_dispatcher=pk.parallel_for,
        out=out,
        view=view,
    )
    return out

def _isfinite(view, profiler_name: Optional[str] = None):
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
    _ufunc_kernel_dispatcher(
        profiler_name=profiler_name,
        tid=tid,
        dtype=dtype,
        ndims=ndims,
        op="isfinite",
        sub_dispatcher=pk.parallel_for,
        out=out,
        view=view,
    )
    return out
