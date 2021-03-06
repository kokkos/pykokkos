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
atexit.register(finalize)

runtime_singleton.runtime = Runtime()
defaults: Optional[CompilationDefaults] = runtime_singleton.runtime.compiler.read_defaults()

if defaults is not None:
    set_default_space(ExecutionSpace[defaults.space])
    if defaults.force_uvm:
        enable_uvm()
    else:
        disable_uvm()
