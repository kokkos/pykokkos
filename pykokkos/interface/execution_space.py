from enum import Enum

import pykokkos.kokkos_manager as km

class ExecutionSpace(Enum):
    Cuda = "Cuda"
    HIP = "HIP"
    OpenMP = "OpenMP"
    Threads = "Threads"
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

    return space in {ExecutionSpace.OpenMP, ExecutionSpace.Threads, ExecutionSpace.Serial}

class ExecutionSpaceInstance:
    """
    An instance of the execution space, corresponding to Cuda/HIP
    streams
    """

    def __init__(self, space: ExecutionSpace, stream = None):
        """
        ExecutionSpaceInstance constructor

        :param space: the exectuion space requested
        :param stream: optional stream argument, only valid for Cuda and HIP
        """

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        self.space: ExecutionSpace = space

        instance_constructor = getattr(km.get_kokkos_module(is_host_execution_space(space)), f"KokkosExecutionSpace_{space.value}")

        if stream is not None:
            if space.value not in {"Cuda", "HIP"}:
                raise ValueError(f"Stream argument unsupported for space {space}")

            import cupy as cp

            if not isinstance(stream, cp.cuda.Stream):
                raise TypeError(f"Type {type(stream)} unsupported; Only CuPy streams allowed")

            self.instance = instance_constructor(stream.ptr, False)
        else:
            self.instance = instance_constructor()