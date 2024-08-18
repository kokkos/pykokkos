
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from pykokkos.runtime import runtime_singleton
import pykokkos.kokkos_manager as km

from .execution_policy import ExecutionPolicy, RangePolicy
from .execution_space import ExecutionSpace
from .views import ViewType, array

workunit_cache: Dict[int, Callable] = {}


@dataclass
class HandledArgs:
    """
    Class for holding the arguments passed to parallel_* functions
    """

    name: Optional[str]
    policy: ExecutionPolicy
    workunit: Callable
    view: Optional[ViewType]
    initial_value: Union[int, float]


def handle_args(is_for: bool, *args) -> HandledArgs:
    """
    Handle the *args passed to parallel_* functions

    :param is_for: whether the arguments belong to a parallel_for call
    :param *args: the list of arguments being checked
    :returns: a HandledArgs object containing the passed arguments
    """

    unpacked: Tuple = tuple(*args)

    name: Optional[str] = None
    policy: Union[ExecutionPolicy, int]
    workunit: Callable
    view: Optional[ViewType] = None
    initial_value: Union[int, float] = 0


    if len(unpacked) == 2:
        policy = unpacked[0]
        workunit = unpacked[1]

    elif len(unpacked) == 3:
        if isinstance(unpacked[0], str) or unpacked[0] is None:
            name = unpacked[0]
            policy = unpacked[1]
            workunit = unpacked[2]
        elif is_for and isinstance(unpacked[2], ViewType):
            policy = unpacked[0]
            workunit = unpacked[1]
            view = unpacked[2]
        elif isinstance(unpacked[2], (int, float)):
            policy = unpacked[0]
            workunit = unpacked[1]
            initial_value = unpacked[2]
        else:
            raise TypeError(f"ERROR: wrong arguments {unpacked}")

    elif len(unpacked) == 4:
        if isinstance(unpacked[0], str) or unpacked[0] is None:
            name = unpacked[0]
            policy = unpacked[1]
            workunit = unpacked[2]

            if is_for and isinstance(unpacked[3], ViewType):
                view = unpacked[3]
            elif isinstance(unpacked[3], (int, float)):
                initial_value = unpacked[3]
            else:
                raise TypeError(f"ERROR: wrong arguments {unpacked}")
        else:
            raise TypeError(f"ERROR: wrong arguments {unpacked}")

    else:
        raise ValueError(f"ERROR: incorrect number of arguments {len(unpacked)}")

    if isinstance(policy, (int, np.integer)):
        policy = RangePolicy(ExecutionSpace.Default, 0, int(policy))

    return HandledArgs(name, policy, workunit, view, initial_value)


def check_policy(policy: Any) -> None:
    """
    Check if an argument is a valid execution policy and raise an
    exception otherwise

    :param policy: the potential policy to be checked
    """

    if not isinstance(policy, (int, ExecutionPolicy)):
        raise TypeError(f"ERROR: {policy} is not a valid execution policy")


def check_workunit(workunit: Any) -> None:
    """
    Check if an argument is a valid workunit and raise an exception
    otherwise

    :param workunit: the potential workunit to be checked
    """

    if not callable(workunit):
        raise TypeError(f"ERROR: {workunit} is not a valid workunit")


def convert_arrays(kwargs: Dict[str, Any]) -> None:
    """
    Convert all numpy and cupy ndarray objects into pk Views

    :param kwargs: the list of keyword arguments passed to the workunit
    """

    cp_available: bool

    try:
        import cupy as cp
        cp_available = True
    except ImportError:
        cp_available = False

    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            kwargs[k] = array(v)
        elif cp_available and isinstance(v, cp.ndarray):
            kwargs[k] = array(v)


def parallel_for(*args, **kwargs) -> None:
    """
    Run a parallel for loop

    :param *args: 
        :param name: (optional) name of the kernel
        :param policy: the execution policy, either a RangePolicy,
            TeamPolicy, TeamThreadRange, ThreadVectorRange, or an
            integer representing the number of threads
        :param workunit: the workunit to be run in parallel
        :param view: (optional) the view being initialized

    :param **kwargs: the keyword arguments passed to a standalone
        workunit
    """

    kwargs = dict(kwargs)
    convert_arrays(kwargs)
    handled_args: HandledArgs = handle_args(True, args)

    runtime_singleton.runtime.run_workunit(
        handled_args.name,
        handled_args.policy,
        handled_args.workunit,
        "for",
        **kwargs)


def reduce_body(operation: str, *args, **kwargs) -> Union[float, int]:
    """
    Internal method to avoid duplication parallel_reduce and
    parallel_scan bodies

    :param operation: the name of the operation, "reduce" or "scan"
    """

    kwargs = dict(kwargs)
    convert_arrays(kwargs)
    args_to_hash: List = []
    args_not_to_hash: Dict = {}
    for k, v in kwargs.items():
        if not isinstance(v, int):
            args_to_hash.append(v)
        else:
            args_not_to_hash[k] = v

    for a in args:
        if callable(a):
            args_to_hash.append(a.__name__)
            break

    args_to_hash.append(operation)

    to_hash = frozenset(args_to_hash)
    cache_key: int = hash(to_hash)

    if cache_key in workunit_cache:
        func, args = workunit_cache[cache_key]
        args.update(args_not_to_hash)
        return func(**args)

    handled_args: HandledArgs = handle_args(True, args)

    return runtime_singleton.runtime.run_workunit(
        handled_args.name,
        handled_args.policy,
        handled_args.workunit,
        operation,
        **kwargs)


def parallel_reduce(*args, **kwargs) -> Union[float, int]:
    """
    Run a parallel reduction

    :param *args: 
        :param name: (optional) name of the kernel
        :param policy: the execution policy, either a RangePolicy,
            TeamPolicy, TeamThreadRange, ThreadVectorRange, or an
            integer representing the number of threads
        :param workunit: the workunit to be run in parallel
        :param initial_value: (optional) the initial value of the
            reduction

    :param **kwargs: the keyword arguments passed to a standalone
        workunit
    """

    return reduce_body("reduce", *args, **kwargs)


def parallel_scan(*args, **kwargs) -> Union[float, int]:
    """
    Run a parallel reduction

    :param *args: 
        :param name: (optional) name of the kernel
        :param policy: the execution policy, either a RangePolicy,
            TeamPolicy, TeamThreadRange, ThreadVectorRange, or an
            integer representing the number of threads
        :param workunit: the workunit to be run in parallel
        :param initial_value: (optional) the initial value of the
            reduction

    :param **kwargs: the keyword arguments passed to a standalone
        workunit
    """

    return reduce_body("scan", *args, **kwargs)


def execute(space: ExecutionSpace, workload: object) -> None:
    if space is ExecutionSpace.Default:
        runtime_singleton.runtime.run_workload(km.get_default_space(), workload)
    else:
        runtime_singleton.runtime.run_workload(space, workload)


def flush():
    runtime_singleton.runtime.flush_trace()