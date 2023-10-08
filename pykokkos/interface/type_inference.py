import inspect
from dataclasses import dataclass
from typing import  Callable, Dict, Optional, Tuple, Union, List, Any
import pykokkos.kokkos_manager as km
from .execution_policy import MDRangePolicy, TeamPolicy, TeamThreadRange, RangePolicy, ExecutionPolicy, ExecutionSpace
from .views import View, ViewType
from .layout import Layout, get_default_layout
from .data_types import DataType, DataTypeClass

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


@dataclass
class UpdatedTypes:
    """
    Class for storing inferred type annotation information 
    (Making Pykokkos more pythonic by automatically inferring types)
    """

    workunit: Callable
    inferred_types: Dict[str, str] # type information stored as string: identifier -> type
    param_list: List[str]
    layout_change: Dict[str, str] # if layout for a view is not Layout.Default it will be stored here


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
        if isinstance(unpacked[0], str):
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
        if isinstance(unpacked[0], str):
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

    if isinstance(policy, int):
        policy = RangePolicy(km.get_default_space(), 0, policy)

    return HandledArgs(name, policy, workunit, view, initial_value)


def get_annotations(parallel_type: str, handled_args: HandledArgs, *args, passed_kwargs) -> Optional[UpdatedTypes]:
    '''
    Infer the datatypes for arguments passed against workunit parameters

    parallel_type: A string identifying the type of parallel dispatch ("parallel_for", "parallel_reduce" ...)
    handled_args: Processed arguments passed to the dispatch
    args: raw arguments passed to the dispatch
    passed_kwargs: raw keyword arguments passed to the dispatch
    returns: UpdateTypes object or None if there are no annotations to be inferred
    '''

    param_list = list(inspect.signature(handled_args.workunit).parameters.values())
    args_list = list(*args)
    updated_types = UpdatedTypes(workunit=handled_args.workunit, inferred_types={}, param_list=param_list, layout_change={})
    policy_params: int = len(handled_args.policy.begin) if isinstance(handled_args.policy, MDRangePolicy) else 1

    # accumulator 
    if parallel_type == "parallel_reduce":
        policy_params += 1
    # accumulator + lass_pass
    if parallel_type == "parallel_scan":
        policy_params += 2

    # Handling policy parameters
    updated_types = infer_policy_args(param_list, policy_params, handled_args.policy, parallel_type, updated_types)

    # Policy parameters are the only parameters
    if len(param_list) == policy_params:
        if not len(updated_types.inferred_types): return None
        return updated_types

    # Handle Keyword args, make sure they are treated by queuing them in args
    if len(passed_kwargs):
        # add value to arguments list so the value can be assessed
        for param in param_list[policy_params:]:
            if param.name in passed_kwargs:
                args_list.append(passed_kwargs[param.name])

    # Handling arguments other than policy args, they begin at value_idx in args list
    # e.g idx=3 -> parallel_for("label", policy, workunit, <other args>...) or if name ("label") is missing: 2 
    value_idx: int = 3 if handled_args.name != None else 2 

    assert (len(param_list) - policy_params) == len(args_list) - value_idx, f"Unannotated arguments mismatch {len(param_list) - policy_params} != {len(args_list) - value_idx}"

    # At this point there must more arguments to the workunit that may not have their types annotated
    # These parameters may also not have raw values associated in the stand alone format -> infer types from the argument list

    updated_types = infer_other_args(param_list, policy_params, args_list, value_idx, handled_args.policy.space, updated_types)

    if not len(updated_types.inferred_types): return None
    return updated_types

def infer_policy_args(
    param_list: List[inspect.Parameter],
    policy_params: int,
    policy: ExecutionPolicy,
    parallel_type: str,
    updated_types: UpdatedTypes
    ) -> UpdatedTypes:
    '''
    Infer the types of policy arguments

    param_list: list of parameter objects that are present in the workunit signature
    policy_params: the number of initial parameters that are dedicated to policy (in param_list/signature)
    policy: the pykokkos execution policy for workunit
    parallel_type: "parallel_for" or "parallel_reduce" or "parallel_scan"
    updated_types: UpdatedTypes object to store inferred types information
    returns: Updated UpdatedTypes object with inferred types
    '''

    for i in range(policy_params):
        param = param_list[i]

        if param.annotation is not inspect._empty:
            continue

        # Check policy and apply annotation(s)
        if isinstance(policy, RangePolicy) or isinstance(policy, TeamThreadRange):
            # only expects one param
            if i == 0:
                updated_types.inferred_types[param.name] = "int"

        elif isinstance(policy, TeamPolicy):
            if i == 0:
                updated_types.inferred_types[param.name] = 'TeamMember'

        elif isinstance(policy, MDRangePolicy):
            total_dims = len(policy.begin) 
            if i < total_dims:
                updated_types.inferred_types[param.name] = "int"
        else:
            raise ValueError("Automatic annotations not supported for this policy")

        # last policy param for parallel reduce and second last for parallel_scan is always the accumulator; the default type is double
        if i == policy_params - 1 and parallel_type == "parallel_reduce" or i == policy_params - 2 and parallel_type == "parallel_scan":
            updated_types.inferred_types[param.name] = "Acc:double"

        if i == policy_params - 1 and parallel_type == "parallel_scan":
            updated_types.inferred_types[param.name] = "bool"

    return updated_types


def infer_other_args(
    param_list: List[inspect.Parameter], 
    policy_params: int,
    args_list: List[Any],
    start_idx: int,
    space: ExecutionSpace,
    updated_types: UpdatedTypes
    ) -> UpdatedTypes:
    '''
    Infer the types of arguments (after the policy arguments)

    param_list: list of parameter objects that are present in the workunit signature
    policy_params: the number of initial parameters that are dedicated to policy (in param_list/signature)
    args_list: List of arguments passed to the parallel dispactch (e.g args for parallal_for())
    start_idx: The index for the first non policy argument in args_list
    updated_types: UpdatedTypes object to store inferred types information
    returns: Updated UpdatedTypes object with inferred types
    '''

    # DataType class has all supported pk datatypes, we ignore class members starting with __, add enum duplicates
    supported_np_dtypes = [attr for attr in dir(DataType) if not attr.startswith("__")] + ["float64", "float32"]

    for i in range(policy_params , len(param_list)):
        param = param_list[i]
        value = args_list[start_idx + i - policy_params]

        if isinstance(value, View):
            inferred_layout = value.layout if value.layout is not Layout.LayoutDefault else get_default_layout(space)
            updated_types.layout_change[param.name] = "LayoutRight" if inferred_layout == Layout.LayoutRight else "LayoutLeft"

        if param.annotation is not inspect._empty:
            continue

        param_type = type(value).__name__

        # switch integer values over 31 bits (signed positive value) to pk.int64
        if param_type == "int" and value.bit_length() > 31:
            param_type = "numpy:int64"

        # check if package name is numpy (handling numpy primitives)
        pckg_name = type(value).__module__

        if pckg_name == "numpy":
            if param_type not in supported_np_dtypes:
                err_str = f"Numpy type {param_type} is unsupported"
                raise TypeError(err_str)
            if param_type == "float64":
                param_type = "double"
            if param_type == "float32":
                param_type = "float"
            # numpy:<type>, Will switch to pk.<type> in parser.fix_types
            param_type = pckg_name +":"+ param_type

        if isinstance(value, View):
            view_dtype = get_pk_datatype(value.dtype)
            if not view_dtype:
                raise TypeError("Cannot infer datatype for view:", param.name)

            param_type = "View"+str(len(value.shape))+"D:"+view_dtype

        updated_types.inferred_types[param.name] = param_type 
    print(updated_types.inferred_types)
    return updated_types


def get_pk_datatype(value):
    '''
    value: value whose datatype is to be determined
    returns the type of custom pkDataType as string
    '''

    dtype = None
    if isinstance(value, DataType):
        dtype = str(value.name)

    elif inspect.isclass(value) and issubclass(value, DataTypeClass):
        dtype = str(value.__name__)

    if dtype == "float64":
        dtype = "double"
    if dtype == "float32":
        dtype = "float"

    return dtype