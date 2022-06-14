import atexit
from typing import Optional

from pykokkos.runtime import runtime_singleton
from pykokkos.core import CompilationDefaults, Runtime
from pykokkos.interface import *
from pykokkos.kokkos_manager import (
    initialize, finalize,
    get_default_space, set_default_space,
    get_default_precision, set_default_precision,
    is_uvm_enabled, enable_uvm, disable_uvm
)

initialize()
from pykokkos.lib.ufuncs import (reciprocal, # type: ignore
                                 log,
                                 log2,
                                 log10,
                                 log1p)

runtime_singleton.runtime = Runtime()
defaults: Optional[CompilationDefaults] = runtime_singleton.runtime.compiler.read_defaults()

if defaults is not None:
    set_default_space(ExecutionSpace[defaults.space])
    if defaults.force_uvm:
        enable_uvm()
    else:
        disable_uvm()

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
