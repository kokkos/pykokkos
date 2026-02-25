import atexit
from typing import Optional

from pykokkos.runtime import runtime_singleton
from pykokkos.core import Runtime
from pykokkos.interface import *
from pykokkos.kokkos_manager import (
    initialize,
    finalize,
    get_default_space,
    set_default_space,
    get_default_precision,
    set_default_precision,
    is_uvm_enabled,
    enable_uvm,
    disable_uvm,
    set_device_id,
)

from pykokkos.lib.info import iinfo, finfo
from pykokkos.lib.create import zeros, zeros_like, ones, ones_like, full, full_like
from pykokkos.lib.manipulate import reshape, ravel, expand_dims
from pykokkos.lib.util import (
    all,
    any,
    sum,
    find_max,
    searchsorted,
    col,
    linspace,
    logspace,
)
from pykokkos.lib.constants import e, pi, inf, nan
from pykokkos.interface.views import astype

import numpy as np

class PKArray(np.ndarray):
    def __new__(cls, array):
        return np.asarray(array).view(cls)

    @property
    def dtype(self):
        from pykokkos.interface import data_types as dt
        mapping = {
            np.dtype('bool'):    dt.bool,
            np.dtype('int8'):    dt.int8,
            np.dtype('int16'):   dt.int16,
            np.dtype('int32'):   dt.int32,
            np.dtype('int64'):   dt.int64,
            np.dtype('uint8'):   dt.uint8,
            np.dtype('uint16'):  dt.uint16,
            np.dtype('uint32'):  dt.uint32,
            np.dtype('uint64'):  dt.uint64,
            np.dtype('float32'): dt.float32,
            np.dtype('float64'): dt.float64,
        }
        return mapping.get(super().dtype, super().dtype)

def _pk_func(np_func):
    def wrapper(*args, **kwargs):
        return PKArray(np_func(*args, **kwargs))
    return wrapper

isnan    = _pk_func(np.isnan)
isinf    = _pk_func(np.isinf)
isfinite = _pk_func(np.isfinite)
equal    = _pk_func(np.equal)
sign     = _pk_func(np.sign)
round    = _pk_func(np.round)
trunc    = _pk_func(np.trunc)
ceil     = _pk_func(np.ceil)
floor    = _pk_func(np.floor)


runtime_singleton.runtime = Runtime()

import weakref

_view_registry: weakref.WeakSet = weakref.WeakSet()


def cleanup():
    """
    Delete the runtime instance to avoid Kokkos errors caused by
    deallocation after calling Kokkos::finalize()
    Also cleanup all View objects before finalization
    """

    for view in list(_view_registry):
        try:
            if hasattr(view, "array"):
                view.array = None
            if hasattr(view, "data"):
                view.data = None
        except (ReferenceError, AttributeError):
            pass

    _view_registry.clear()

    global runtime_singleton
    del runtime_singleton.runtime
    del runtime_singleton

    from pykokkos.interface.parallel_dispatch import workunit_cache

    workunit_cache.clear()


# Will be called in reverse order of registration (cleanup then finalize)
atexit.register(finalize)
atexit.register(cleanup)
