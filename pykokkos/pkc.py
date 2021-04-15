#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
from typing import List

from pykokkos.core.compiler import CompilationDefaults, Compiler
from pykokkos.interface import ExecutionSpace
from pykokkos.kokkos_manager import is_uvm_enabled, enable_uvm

def get_spaces(spaces: List[str]) -> List[ExecutionSpace]:
    """
    Get the of execution spaces from a list of strings

    :param spaces: the spaces provided by the user
    :returns: a list of ExecutionSpace values
    """

    execution_spaces: List[ExecutionSpace] = []
    for s in spaces:
        if s not in ExecutionSpace.__members__:
            sys.exit(f"ERROR: Invalid execution space \"{s}\"")

        execution_spaces.append(ExecutionSpace[s])

    return execution_spaces

def get_defaults(spaces: List[ExecutionSpace], force_uvm: bool) -> CompilationDefaults:
    """
    Get the defaults given the passed arguments

    :param spaces: the spaces provided by the user
    :param force_uvm: whether uvm is enabled
    :returns: the defaults object
    """

    space: str = ExecutionSpace.OpenMP.value

    if ExecutionSpace.Cuda in spaces:
        space = ExecutionSpace.Cuda.value
    elif ExecutionSpace.OpenMP in spaces:
        space = ExecutionSpace.OpenMP.value
    elif ExecutionSpace.Pthreads in spaces:
        space = ExecutionSpace.Pthreads.value
    elif ExecutionSpace.Serial in spaces:
        space = ExecutionSpace.Serial.value

    return CompilationDefaults(space, force_uvm)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="PyKokkos Compiler.")
    argparser.add_argument("main", type=str,
                           help="The path to the main file.")
    argparser.add_argument("-sources", type=str, nargs="+",
                           help="The list of compilation units.")
    argparser.add_argument("-spaces", type=str, nargs="+",
                           help="The list of execution spaces."
                                "If empty, all spaces are compiled")
    argparser.add_argument("-uvm", "--force_uvm", action="store_true",
                           help="Compile with force_uvm.")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           help="Verbose option.")

    args = argparser.parse_args()

    spaces: List[ExecutionSpace]
    if not args.spaces:
        spaces = [ExecutionSpace.OpenMP]
    else:
        spaces = get_spaces(args.spaces)

    sources: List[str]
    if not args.sources:
        sources = [args.main]
    else:
        sources = args.sources

    if args.force_uvm:
        enable_uvm()

    defaults: CompilationDefaults = get_defaults(spaces, args.force_uvm)

    compiler = Compiler()
    compiler.compile_sources(Path(args.main), sources, spaces, is_uvm_enabled(), defaults, args.verbose)
