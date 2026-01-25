#!@PYTHON_EXECUTABLE@
# ************************************************************************
#
#                        Kokkos v. 3.0
#       Copyright (2020) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Christian R. Trott (crtrott@sandia.gov)
#
# ************************************************************************
#

from __future__ import absolute_import
from . import libpykokkos as lib

__author__ = "Jonathan R. Madsen"
__copyright__ = (
    "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "3.1.1"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


def convert_dtype(_dtype, _module=None):
    """Converts kokkos data types into numpy dtype"""
    if isinstance(_dtype, (str, int)):
        _true_dtype = lib.get_dtype(_dtype)
    else:
        _true_dtype = lib.get_dtype(int(_dtype))
    if _module is None:
        import numpy as np

        _module = np

    # Handle complex dtypes mapping
    if _true_dtype == "complex_float32_dtype":
        _true_dtype = "complex64"
    elif (
        _true_dtype == "complex_float64_dtype" or _true_dtype == "complex_double_dtype"
    ):
        _true_dtype = "complex128"

    return getattr(_module, _true_dtype)


def read_dtype(_dtype):
    """Converts generic dtype (from numpy for example) into kokkos dtype"""
    # try calling library routine
    if isinstance(_dtype, (str, int)):
        try:
            return getattr(lib, lib.get_dtype(_dtype))
        except ValueError:
            pass

    # try numpy
    try:
        import numpy as np

        if _dtype == np.int8:
            return lib.int8
        if _dtype == np.int16:
            return lib.int16
        elif _dtype == np.int32:
            return lib.int32
        elif _dtype == np.int64:
            return lib.int64
        elif _dtype == np.uint8:
            return lib.uint8
        elif _dtype == np.uint16:
            return lib.uint16
        elif _dtype == np.uint32:
            return lib.uint32
        elif _dtype == np.uint64:
            return lib.uint64
        elif _dtype == np.float32:
            return lib.float32
        elif _dtype == np.float64:
            return lib.float64
        elif _dtype == np.complex64:
            return lib.complex_float32_dtype
        elif _dtype == np.complex128:
            return lib.complex_float64_dtype
    except ImportError:
        pass

    # just return the dtype
    return _dtype


def _determine_array_input(_inp, shape, label, array, dtype, layout):
    """Determine whether first argument is shape, label, or array
    and extract subsequent info"""
    # support non-labeled variant
    if isinstance(_inp, str) and label is None:
        label = f"{_inp}"
    elif isinstance(_inp, (list, tuple)) and shape is None:
        shape = list(_inp)[:]
    elif array is None:
        array = _inp
        try:
            shape = array.shape
            _order = array.order
            if _order == "C":
                layout = lib.LayoutRight
            elif _order == "F":
                layout = lib.LayoutLeft
        except AttributeError:
            pass

    # if array has been passed in, try getting the dtype
    if array is not None:
        try:
            dtype = read_dtype(array.dtype)
        except AttributeError:
            pass

    return [shape, label, array, dtype, layout]


def array(
    shape_label_or_array,
    shape=None,
    label=None,
    array=None,
    dtype=lib.double,
    space=lib.HostSpace,
    layout=lib.LayoutRight,
    trait=lib.Managed,
    dynamic=False,
    order=None,
):
    [shape, label, array, dtype, layout] = _determine_array_input(
        shape_label_or_array, shape, label, array, dtype, layout
    )

    # layout was specified via numpy "order" field
    if order is not None and layout == lib.LayoutRight and isinstance(order, str):
        if order.upper() == "C":
            layout = lib.LayoutRight
        elif order.upper() == "F":
            layout = lib.LayoutLeft

    _prefix = "KokkosView" if not dynamic else "KokkosDynRankView"
    _space = lib.get_memory_space(space)
    _dtype = lib.get_dtype(dtype)
    _layout = lib.get_layout(layout)
    _name = None
    _label = None
    _ndim = len(shape)

    if dynamic:
        _dtype_str = "{}".format(_dtype)
    else:
        if _ndim > lib.max_concrete_rank:
            raise ValueError(
                "pykokkos-base build only supports {} ranks. Requested {} ranks".format(
                    lib.max_concrete_rank, _ndim
                )
            )
        _dtype_str = "{}{}".format(_dtype, "*" * _ndim)

    _name = f"{_prefix}_{_dtype}_{_space}_{_layout}"
    _label = f"{_prefix}<{_dtype_str}, {_layout}, {_space}"

    # handle the trait argument
    if trait is not None:
        _trait = lib.get_memory_trait(trait)
        if _trait not in ("Managed", "Unmanaged"):
            _name = f"{_name}_{_trait}"
            _label = f"{_label}, {_trait}"

    # if fixed view
    if not dynamic:
        _name = f"{_name}_{_ndim}"

    # if a label was not provided
    if label is None:
        label = f"{_label}>"

    return getattr(lib, _name)(label if array is None else array, shape)


def unmanaged_array(*_args, **_kwargs):
    """This is kept for backwards compatibility"""
    return array(*_args, **_kwargs)


def create_mirror(src, copy=False):
    """Performs Kokkos::create_mirror"""
    return src.create_mirror(copy)


def create_mirror_view(src, copy=False):
    """Performs Kokkos::create_mirror_view"""
    return src.create_mirror_view(copy)


def deep_copy(dst, src):
    """Performs Kokkos::deep_copy"""
    return dst.deep_copy(src)


def random_pool(state, space, seed=None):
    """Create a Random_XorShift Pool"""

    if state not in {64, 1024}:
        raise ValueError(f"State size {state} not supported, only 64 and 1024.")

    if seed is not None and not isinstance(seed, int):
        raise ValueError("Seed must be either None or of type int")

    _space = lib.get_execution_space(space)
    _name = f"KokkosXorShift{state}Pool_{_space}"

    _cons = getattr(lib, _name)

    if seed is None:
        return _cons()

    return _cons(seed)
