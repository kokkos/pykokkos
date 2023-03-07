from pykokkos.runtime import runtime_singleton
from .execution_space import ExecutionSpace
from pathlib import Path
from typing import List

def compile_into_module(path: Path, source: List[str], module_name: str, executionSpace: ExecutionSpace = ExecutionSpace.Default):
    """
    Takes a c++ pybind11 source as a string and compiles it into a python module. The resulting python module is returned upon success

    param path: Path to write the shared object produced in the compilation to
    param source: c++ source of the module to be compiled with pybind11
    param module_name: name of the module in python
    param executionSpace: Executionspace which the module is compiled for

    return: The compiled python module
    """
    return runtime_singleton.runtime.compile_into_module(path,source,module_name,executionSpace)
