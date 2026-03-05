from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from pykokkos.runtime import runtime_singleton
import pykokkos.kokkos_manager as km
from pykokkos.core.cppast import BuiltinType

from .execution_policy import ExecutionPolicy, RangePolicy
from .execution_space import ExecutionSpace, DeviceExecutionSpace
from .views import ViewType, array

from .interface_util import generic_error, get_filename, get_lineno

from .memory_space import get_default_memory_space

import inspect

workunit_cache: Dict[int, Callable] = {}

# Map PyKokkos BuiltinType to numpy dtypes
# This ensures consistency with PyKokkos's type system
BUILTIN_TO_NUMPY: Dict[str, np.dtype] = {
    BuiltinType.INT.value: np.int32,
    BuiltinType.INT8.value: np.int8,
    BuiltinType.INT16.value: np.int16,
    BuiltinType.INT32.value: np.int32,
    BuiltinType.INT64.value: np.int64,
    BuiltinType.UINT8.value: np.uint8,
    BuiltinType.UINT16.value: np.uint16,
    BuiltinType.UINT32.value: np.uint32,
    BuiltinType.UINT64.value: np.uint64,
    BuiltinType.FLOAT.value: np.float32,
    BuiltinType.DOUBLE.value: np.float64,
    BuiltinType.BOOL.value: np.bool_,
}


def parse_list_annotation(annotation) -> Tuple[int, np.dtype]:
    """
    Recursively parse List[T] or List[List[T]] annotations to determine
    nesting depth and element type.

    :param annotation: Type annotation (e.g., List[int], List[List[float]])
    :returns: Tuple of (depth, numpy_dtype)
    """
    import typing

    depth = 0
    current = annotation
    element_type = None

    # Traverse nested List annotations
    while hasattr(current, "__origin__") and current.__origin__ is list:
        depth += 1
        if hasattr(current, "__args__") and len(current.__args__) > 0:
            current = current.__args__[0]
        else:
            break

    # Now current should be the element type (int, float, bool, etc.)
    element_type = current

    # Map element type to numpy dtype
    if element_type is int:
        dtype = BUILTIN_TO_NUMPY[BuiltinType.INT.value]
    elif element_type is float:
        dtype = BUILTIN_TO_NUMPY[BuiltinType.DOUBLE.value]
    elif element_type is bool:
        dtype = BUILTIN_TO_NUMPY[BuiltinType.BOOL.value]
    else:
        # Default to int32
        dtype = BUILTIN_TO_NUMPY[BuiltinType.INT.value]

    return depth, dtype


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


def convert_arrays(kwargs: Dict[str, Any], workunit: Callable, execution_space) -> None:
    """
    Convert all numpy, cupy and pytorch ndarray objects into pk Views

    :param kwargs: the list of keyword arguments passed to the workunit
    :param workunit: the workunit function (used to infer types for Python lists)
    :param execution_space: the execution space of the workunit
        (used to convert arrays to the correct memory space)
    """

    cp_available: bool
    torch_available: bool

    memory_space = get_default_memory_space(execution_space)

    try:
        import cupy as cp

        cp_available = True
    except ImportError:
        cp_available = False

    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False

    # Get type hints from workunit if available
    type_hints = {}
    if workunit is not None and callable(workunit):
        import inspect as insp

        try:
            sig = insp.signature(workunit)
            type_hints = {
                name: param.annotation
                for name, param in sig.parameters.items()
                if param.annotation != insp.Parameter.empty
            }
        except (ValueError, TypeError):
            pass

    for k, v in kwargs.items():
        if isinstance(v, ViewType) or isinstance(v, np.generic):
            continue
        elif isinstance(v, list):
            # Default to whatever PyKokkos uses for 'int'
            dtype = BUILTIN_TO_NUMPY[BuiltinType.INT.value]

            if k in type_hints:
                annotation = type_hints[k]
                if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
                    # Parse nested List annotations (List[int], List[List[int]], etc.)
                    depth, dtype = parse_list_annotation(annotation)

            # Convert Python list to numpy array, then to View
            kwargs[k] = array(np.array(v, dtype=dtype), space=memory_space)
        elif isinstance(v, np.ndarray):
            if execution_space in DeviceExecutionSpace:
                raise TypeError(
                    f"Argument '{k}' is a numpy array, which cannot be accessed "
                    f"from the {execution_space.value} execution space. "
                    f"Use a pk.View (e.g. pk.View([...], dtype)) or a CuPy array instead."
                )
            kwargs[k] = array(v, space=memory_space)
        elif cp_available and isinstance(v, cp.ndarray):
            if execution_space not in DeviceExecutionSpace:
                raise TypeError(
                    f"Argument '{k}' is a CuPy array, which cannot be accessed "
                    f"from the {execution_space.value} (host) execution space. "
                    f"Convert it to a numpy array or pk.View in host memory first."
                )
            kwargs[k] = array(v, space=memory_space)
        elif torch_available and torch.is_tensor(v):
            kwargs[k] = array(v, space=memory_space)
        elif (
            hasattr(v, "__array__")
            or hasattr(v, "__cuda_array_interface__")
            or hasattr(v, "__array_interface__")
        ):
            # This is some array-like object we don't support
            caller_frame = inspect.currentframe().f_back.f_back
            filename = get_filename(caller_frame)
            lineno = get_lineno(caller_frame)
            msg = f"Type {type(v)} is not supported. Only numpy arrays, cupy arrays, and torch tensors are supported."
            generic_error(filename, lineno, msg, "Conversion failed")


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
    handled_args: HandledArgs = handle_args(True, args)
    convert_arrays(kwargs, handled_args.workunit, handled_args.policy.space.space)

    runtime_singleton.runtime.run_workunit(
        handled_args.name, handled_args.policy, handled_args.workunit, "for", **kwargs
    )


def reduce_body(operation: str, *args, **kwargs) -> Union[float, int]:
    """
    Internal method to avoid duplication parallel_reduce and
    parallel_scan bodies

    :param operation: the name of the operation, "reduce" or "scan"
    """

    kwargs = dict(kwargs)
    handled_args: HandledArgs = handle_args(True, args)
    convert_arrays(kwargs, handled_args.workunit, handled_args.policy.space.space)

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

    return runtime_singleton.runtime.run_workunit(
        handled_args.name,
        handled_args.policy,
        handled_args.workunit,
        operation,
        **kwargs,
    )


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
