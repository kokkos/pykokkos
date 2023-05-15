from enum import Enum

import pykokkos.kokkos_manager as km

class ExecutionSpace(Enum):
    Cuda = "Cuda"
    HIP = "HIP"
    OpenMP = "OpenMP"
    Pthreads = "Pthreads"
    Serial = "Serial"
    Debug = "Debug"
    Default = "Default"

def is_host_execution_space(space: ExecutionSpace) -> bool:
    """
    Check if the supplied execution space runs on the host

    :param space: the space being checked
    :returns: True if the space runs on the host
    """

    if space is ExecutionSpace.Default:
        space = km.get_default_space()

    return space in {ExecutionSpace.OpenMP, ExecutionSpace.Pthreads, ExecutionSpace.Serial}