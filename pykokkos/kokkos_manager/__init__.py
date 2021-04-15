import os
from typing import Any, Dict

from pykokkos.bindings import kokkos
from pykokkos.interface.execution_space import ExecutionSpace
from pykokkos.interface.data_types import DataTypeClass, double

CONSTANTS: Dict[str, Any] = {
    "EXECUTION_SPACE": ExecutionSpace.OpenMP,
    "REAL_DTYPE": double,
    "IS_INITIALIZED": False,
    "ENABLE_UVM": False
}

def get_default_space() -> ExecutionSpace:
    """
    Get the default PyKokkos execution space

    :returns: the ExecutionSpace object
    """

    if os.environ.get("DEBUG"):
        return ExecutionSpace.Debug

    return CONSTANTS["EXECUTION_SPACE"]

def set_default_space(space: ExecutionSpace) -> None:
    """
    Set the default PyKokkos execution space

    :param space: the new default
    """

    if not isinstance(space, ExecutionSpace):
        print("ERROR: space is not an ExecutionSpace")
        return

    CONSTANTS["EXECUTION_SPACE"] = space

def get_default_precision() -> ExecutionSpace:
    """
    Get the default PyKokkos precision

    :returns: the precision type object
    """

    return CONSTANTS["REAL_DTYPE"]

def set_default_precision(precision: DataTypeClass) -> None:
    """
    Set the default PyKokkos precision

    :param precision: the new default
    """

    if not issubclass(precision, DataTypeClass):
        print("ERROR: precision is not a DataType")
        return

    CONSTANTS["REAL_DTYPE"] = precision

def is_uvm_enabled() -> bool:
    """
    Check if UVM is enabled

    :returns: True or False
    """

    return CONSTANTS["ENABLE_UVM"]

def enable_uvm() -> None:
    """
    Enable CudaUVMSpace
    """

    CONSTANTS["ENABLE_UVM"] = True

def disable_uvm() -> None:
    """
    Disable CudaUVMSpace
    """

    CONSTANTS["ENABLE_UVM"] = False

def initialize() -> None:
    """
    Call Kokkos::initialize() if not already called
    """

    if CONSTANTS["IS_INITIALIZED"] == False:
        kokkos.initialize()
        CONSTANTS["IS_INITIALIZED"] = True

def finalize() -> None:
    """
    Call Kokkos::finalize() if initialize() has been called
    """

    if CONSTANTS["IS_INITIALIZED"] == True:
        kokkos.finalize()
        CONSTANTS["IS_INITIALIZED"] = False
