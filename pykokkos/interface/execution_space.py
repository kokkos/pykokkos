from enum import Enum


class ExecutionSpace(Enum):
    Cuda = "Cuda"
    OpenMP = "OpenMP"
    Pthreads = "Pthreads"
    Serial = "Serial"
    Debug = "Debug"
    Default = "Default"
