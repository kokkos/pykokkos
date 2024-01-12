import os
from types import ModuleType
from typing import Any, Dict, List

from pykokkos.bindings import kokkos
from pykokkos.interface.execution_space import ExecutionSpace, ExecutionSpaceInstance
from pykokkos.interface.data_types import DataTypeClass, double


CONSTANTS: Dict[str, Any] = {
    "KOKKOS_VERSION": 3.7, # default to 3.7
    "EXECUTION_SPACE": ExecutionSpace.OpenMP,
    "AVAILABLE_EXECUTION_SPACES": {},
    "REAL_DTYPE": double,
    "IS_INITIALIZED": False,
    "ENABLE_UVM": False,
    "MULTI_GPU": False,
    "NUM_GPUS": 0,
    "KOKKOS_GPU_MODULE": kokkos,
    "KOKKOS_GPU_MODULE_LIST": [],
    "KOKKOS_GPU_INSTANCE_LIST": [],
    "DEVICE_ID": 0,
    "GPU_BACKEND": None
}

pk_kokkos_version: str = os.getenv("PK_KOKKOS_INTERFACE")
if pk_kokkos_version is not None:
    try:
        CONSTANTS["KOKKOS_VERSION"] = float(pk_kokkos_version)
    except ValueError:
        print(f"WARNING: PK_KOKKOS_INTERFACE value '{pk_kokkos_version}' is invalid; reverting to {CONSTANTS['KOKKOS_VERSION']}")

def get_kokkos_version() -> float:
    """
    Get the version of the installed Kokkos library
    """

    return CONSTANTS["KOKKOS_VERSION"]

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

def get_execution_space_instance(space: ExecutionSpace) -> ExecutionSpaceInstance:
    """
    Return the default execution space instance for a given space

    :param space: the execution space required
    :returns: the kokkos execution space object
    """

    if space not in CONSTANTS["AVAILABLE_EXECUTION_SPACES"]:
        raise ValueError(f"Execution space {space} is not available")

    return CONSTANTS["AVAILABLE_EXECUTION_SPACES"][space]

def get_available_execution_spaces() -> List[str]:
    """
    Get the available execution spaces

    :returns: a list of the available spaces
    """

    return list(CONSTANTS["AVAILABLE_EXECUTION_SPACES"].keys())

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

def get_kokkos_module(is_cpu: bool) -> ModuleType:
    """
    Get the current kokkos module

    :param is_cpu: is the lib needed for cpu
    :returns: the kokkos module
    """

    if is_cpu:
        return kokkos

    return CONSTANTS["KOKKOS_GPU_MODULE"]

def set_device_id(device_id: int) -> None:
    """
    Set the current device ID

    :param device_id: the ID of the device to enable
    """

    if not isinstance(device_id, int):
        raise TypeError("'device_id' must be of type 'int'")

    num_gpus: int = CONSTANTS["NUM_GPUS"]
    if device_id >= num_gpus or device_id < 0:
        raise RuntimeError(f"Device {device_id} does not exist (range [0..{num_gpus}))")

    if num_gpus == 1:
        return

    import cupy
    cupy.cuda.runtime.setDevice(device_id)
    CONSTANTS["DEVICE_ID"] = device_id

    gpu_lib = CONSTANTS["KOKKOS_GPU_MODULE_LIST"][device_id]
    CONSTANTS["KOKKOS_GPU_MODULE"] = gpu_lib

    exec_space_instance = CONSTANTS["KOKKOS_GPU_INSTANCE_LIST"][device_id]
    CONSTANTS["AVAILABLE_EXECUTION_SPACES"][get_gpu_framework()] = exec_space_instance

def get_device_id() -> int:
    """
    Get the ID of the currently enabled device

    :returns: the ID of the enabled device
    """

    return CONSTANTS["DEVICE_ID"]

def is_multi_gpu_enabled() -> bool:
    """
    Check if pykokkos has been configured for multi-gpu use

    :returns: True or False
    """

    return CONSTANTS["MULTI_GPU"]

def get_kokkos_gpu_modules() -> List:
    """
    Get the pykokkos-base gpu modules

    :returns: the list of modules
    """

    return CONSTANTS["KOKKOS_GPU_MODULE_LIST"]

def get_num_gpus() -> bool:
    """
    Get the number of gpus pykokkos has been configured for

    :returns: the number of gpus
    """

    return CONSTANTS["NUM_GPUS"]

def get_gpu_framework() -> ExecutionSpace:
    """
    Get the framework used by the GPU

    :returns: the framework as a string
    """

    return CONSTANTS["GPU_BACKEND"]

try:
    # Save the active device ID before calling initialize(), which
    # will overwrite it
    import cupy as cp
    active_device: int = cp.cuda.runtime.getDevice()
except ImportError:
    pass

initialize()

# For every available execution space, create a default execution
# space instance
for space in ExecutionSpace:
    if space in {ExecutionSpace.Debug, ExecutionSpace.Default}:
        continue

    if kokkos.get_device_available(space.value):
        CONSTANTS["AVAILABLE_EXECUTION_SPACES"][space] = ExecutionSpaceInstance(space)

        if space in {ExecutionSpace.Cuda, ExecutionSpace.HIP}:
            CONSTANTS["GPU_BACKEND"] = space

# NOTE: multiple GPU support is almost certainly
# broken, we can't just assume that there are modules
# named gpu0, gpu1, and so on...

# Import multiple kokkos libs to support multiple devices per
# process. This assumes that there are modules named f"gpu{id}"
# that can be imported.
import atexit
import importlib
import sys

try:
    import cupy as cp
    NUM_CUDA_GPUS: int = cp.cuda.runtime.getDeviceCount()
    KOKKOS_LIBS: List[str] = [f"gpu{id}" for id in range(NUM_CUDA_GPUS)]
except ImportError:
    NUM_CUDA_GPUS = 0
    KOKKOS_LIBS = []

CONSTANTS["NUM_GPUS"] = NUM_CUDA_GPUS

KOKKOS_GPU_MODULE_LIST: List = []
for id, lib in enumerate(KOKKOS_LIBS):
    try:
        module = importlib.import_module(lib)
        KOKKOS_GPU_MODULE_LIST.append(module)

        # Can't pass device id directly to initialize(), so need to
        # append argument to select device to sys.argv.
        # (see https://github.com/kokkos/pykokkos-base/blob/d3946ed56483f3cbe2e660cc50fe73c50dad19ea/src/libpykokkos.cpp#L65)
        sys.argv.append(f"--device-id={id}")
        module.initialize()
        atexit.register(module.finalize)
        sys.argv.pop()
    except ModuleNotFoundError:
        pass

if len(KOKKOS_GPU_MODULE_LIST) > 1:
    CONSTANTS["MULTI_GPU"] = True
    CONSTANTS["KOKKOS_GPU_MODULE_LIST"] = KOKKOS_GPU_MODULE_LIST

    # Create an execution space instance per device from each GPU lib
    KOKKOS_GPU_INSTANCE_LIST: List = []
    for lib in KOKKOS_GPU_MODULE_LIST:
        CONSTANTS["KOKKOS_GPU_MODULE"] = lib
        KOKKOS_GPU_INSTANCE_LIST.append(ExecutionSpaceInstance(get_gpu_framework()))

    CONSTANTS["KOKKOS_GPU_MODULE"] = KOKKOS_GPU_MODULE_LIST[0]
    CONSTANTS["KOKKOS_GPU_INSTANCE_LIST"] = KOKKOS_GPU_INSTANCE_LIST

try:
    import cupy as cp
    cp.cuda.runtime.setDevice(active_device)
except ImportError:
    pass