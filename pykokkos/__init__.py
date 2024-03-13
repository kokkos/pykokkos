import atexit
from typing import Optional

from pykokkos.runtime import runtime_singleton
from pykokkos.core import Runtime
from pykokkos.interface import *
from pykokkos.kokkos_manager import (
    initialize, finalize,
    get_default_space, set_default_space,
    get_default_precision, set_default_precision,
    is_uvm_enabled, enable_uvm, disable_uvm,
    set_device_id
)

from pykokkos.lib.ufuncs import (reciprocal,
                                 log,
                                 log2,
                                 log10,
                                 log1p,
                                 sqrt,
                                 sign,
                                 add,
                                 copyto,
                                 subtract,
                                 dot,
                                 multiply,
                                 matmul,
                                 np_matmul,
                                 divide,
                                 negative,
                                 positive,
                                 power,
                                 fmod,
                                 square,
                                 greater,
                                 logaddexp,
                                 true_divide,
                                 logaddexp2,
                                 floor_divide,
                                 sin,
                                 cos,
                                 tan,
                                 tanh,
                                 logical_and,
                                 logical_or,
                                 logical_xor,
                                 logical_not,
                                 fmax,
                                 fmin,
                                 exp,
                                 exp2,
                                 argmax,
                                 unique,
                                 var,
                                 in1d,
                                 mean,
                                 hstack,
                                 transpose,
                                 index,
                                 isinf,
                                 isnan,
                                 equal,
                                 isfinite,
                                 round,
                                 trunc,
                                 ceil,
                                 floor,
                                 broadcast_view)
from pykokkos.lib.info import iinfo, finfo
from pykokkos.lib.create import (zeros,
                                 zeros_like,
                                 ones,
                                 ones_like,
                                 full,
                                 full_like)
from pykokkos.lib.manipulate import reshape, ravel, expand_dims
from pykokkos.lib.util import all, any, sum, find_max, searchsorted, col, linspace, logspace
from pykokkos.lib.constants import e, pi, inf, nan
from pykokkos.interface.views import astype

__array_api_version__ = "2021.12"

__all__ = ["__array_api_version__"]

runtime_singleton.runtime = Runtime()

def cleanup():
    """
    Delete the runtime instance to avoid Kokkos errors caused by
    deallocation after calling Kokkos::finalize()
    """

    global runtime_singleton
    del runtime_singleton.runtime
    del runtime_singleton

    from pykokkos.interface.parallel_dispatch import workunit_cache
    workunit_cache.clear()

# Will be called in reverse order of registration (cleanup then finalize)
atexit.register(finalize)
atexit.register(cleanup)
