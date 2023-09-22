import gc
import inspect
import functools
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pykokkos.runtime import runtime_singleton
import pykokkos.kokkos_manager as km

from .views import ViewType, View
from .execution_policy import *
from .execution_space import ExecutionSpace

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

@dataclass
class UpdatedTypes:
    """
    Class for storing inferred type annotation information 
    (Making Pykokkos more pythonic by automatically inferring types)
    """

    workunit: Callable
    inferred_types: Dict[str, type]
    is_arg: set[str]



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

    print("handle_args arguments: ", *args)

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


def get_annotations(parallel_type: str, handled_args: HandledArgs, *args, passed_kwargs) -> UpdatedTypes:
    
    param_list = list(inspect.signature(handled_args.workunit).parameters.values())
    args_list = list(*args)
    print("\t[get_annotations] PARAM VALUES:", param_list) 
    #! Should you be always setting this?
    updated_types = UpdatedTypes(workunit=handled_args.workunit, inferred_types={}, is_arg=set())
    
    policy_params: int = len(handled_args.policy.begin) if isinstance(handled_args.policy, MDRangePolicy) else 1
    print("\t[get_annotations] POLICY PARAMS:", policy_params)
    
    # accumulator 
    if parallel_type == "parallel_reduce":
        policy_params += 1

    for i in range(policy_params):
        # Check policy type
        param = param_list[i]
        if param.annotation is inspect._empty:
            print("\t\t[!!!] ANNOTATION IS NOT PROVIDED for policy param: ", param)
            if updated_types is None:
                updated_types: UpdatedTypes = UpdatedTypes(workunit=handled_args.workunit, inferred_types={}, is_arg=set())
            # Check policy and apply annotation(s)
            
            if isinstance(handled_args.policy, RangePolicy) or isinstance(handled_args.policy, TeamThreadRange):
                # only expects one param
                if i == 0:
                    updated_types.inferred_types[param.name] = "int"
                    updated_types.is_arg.add(param.name)
            
            elif isinstance(handled_args.policy, TeamPolicy):
                if i == 0:
                    updated_types.inferred_types[param.name] = 'pk.TeamMember'
                    updated_types.is_arg.add(param.name)
            
            elif isinstance(handled_args.policy, MDRangePolicy):
                total_dims = len(handled_args.policy.begin) 
                if i < total_dims:
                    updated_types.inferred_types[param.name] = "int"
                    updated_types.is_arg.add(param.name)
            else:
                raise ValueError("Automatic annotations not supported for this policy")
            
            if i == policy_params - 1 and parallel_type == "parallel_reduce":
                updated_types.inferred_types[param.name] = 'pk.Acc[float]'
                updated_types.is_arg.add(param.name)


    if updated_types is None:
        return None


    if len(param_list) == policy_params:
        return updated_types

    # Handle Kwargs
    print("\t[get_annotations] KWARGS RECEIVED: ", passed_kwargs)
    if len(passed_kwargs.keys()):
        # add value to arguments so the value can be assessed
        for param in param_list[policy_params:]:
            if param.name in passed_kwargs:
                args_list.append(passed_kwargs[param.name])
    

    # Handling other arguments
    # print("handled args name:", handled_args.name) # prints the profiling label
    print(args_list)
    value_idx: int = 3 if handled_args.name != None else 2 
    print("\t[get_annotations] ___ OTHER ARGS ___", args_list[value_idx:])



    print(len(args_list) - value_idx)
    assert (len(param_list) - policy_params) == len(args_list) - value_idx, f"Unannotated arguments mismatch {len(param_list) - policy_params} != {len(args_list) - value_idx}"
    
    # At this point there must more arguments to the workunit that may not have their types annotated
    # These parameters may also not have raw values associated in the stand alone format -> infer types from the parameter list


    for i in range(policy_params , len(param_list)):
        # Check policy type
        param = param_list[i]
        if param.annotation is inspect._empty:
            print("\t\t[!!!] ANNOTATION IS NOT PROVIDED PARAM", param)

            value = args_list[value_idx+i-policy_params]
            # print("DIRS:", dir(value))
            # print("Type:", value.layout)
            # Note: view class has the following constructor: 
            # def __init__(
            #     self,
            #     shape: Union[List[int], Tuple[int]],
            #     dtype: Union[DataTypeClass, type] = real,
            #     space: MemorySpace = MemorySpace.MemorySpaceDefault,
            #     layout: Layout = Layout.LayoutDefault,
            #     trait: Trait = Trait.TraitDefault,
            #     array: Optional[np.ndarray] = None
            # )

            param_type = type(value).__name__

            if isinstance(value, View):
                print("FOUND VIEW")
                print(value.shape, value.dtype, value.space)
                print(len(value.shape), value.dtype.name)
                # if this is a 1D view
                param_type = "View"+str(len(value.shape))+"D:"+str(value.dtype.name)

            updated_types.inferred_types[param.name] = param_type
            updated_types.is_arg.add(param.name)


    print("RETURNING UPDATED TYPES", updated_types)
    return updated_types
            


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

    # args_to_hash: List = []
    # args_not_to_hash: Dict = {}
    # for k, v in kwargs.items():
    #     if not isinstance(v, int):
    #         args_to_hash.append(v)
    #     else:
    #         args_not_to_hash[k] = v

    # # Hash the workunit
    # for a in args:
    #     if callable(a):
    #         args_to_hash.append(a.__name__)
    #         break

    # to_hash = frozenset(args_to_hash)
    # cache_key: int = hash(to_hash)

    # if cache_key in workunit_cache:
    #     dead_obj = 0
    #     func, newargs = workunit_cache[cache_key]
    #     for key, arg in newargs.items():
    #         # see gh-34
    #         # reject cache retrieval when an object in the
    #         # cache has a reference count of 0 (presumably
    #         # only possible because of the C++/pybind11 infra;
    #         # normally a refcount of 1 is the lowest for pure
    #         # Python objects)
    #         # NOTE: is the cache genuinely useful now though?
    #         ref_count = len(gc.get_referrers(arg))
    #         # we also can't safely retrieve from the cache
    #         # for user-defined workunit components
    #         # because they may depend on class instance state
    #         # per gh-173
    #         if ref_count == 0 or not key.startswith("pk_"):
    #             dead_obj += 1
    #             break
    #     if not dead_obj:
    #         args = newargs
    #         args.update(args_not_to_hash)
    #         func(**args)
    #         return



    handled_args: HandledArgs = handle_args(True, args)
    
    print("\n----------- PARALLEL FOR -----------------------")
    print("-----KWARGS", kwargs)
    print("-----ARGS", args)
    print("-----Workunit", handled_args.workunit)
    print("attributes for workload", (list(inspect.signature(handled_args.workunit).parameters.values())[0]).annotation)
    print("attributes for policy", handled_args.policy.begin, handled_args.policy.end)
    print("name of workunit", handled_args.workunit.__name__)
    print("----------- END -----------------------\n")

    updated_types: UpdatedTypes = get_annotations("parallel_for", handled_args, args, passed_kwargs=kwargs)
    
    func, args = runtime_singleton.runtime.run_workunit(
        handled_args.name,
        [updated_types],
        handled_args.policy,
        handled_args.workunit,
        "for",
        **kwargs)

    # workunit_cache[cache_key] = (func, args)
    func(**args)

def reduce_body(operation: str, *args, **kwargs) -> Union[float, int]:
    """
    Internal method to avoid duplication parallel_reduce and
    parallel_scan bodies

    :param operation: the name of the operation, "reduce" or "scan"
    """

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

    #* Inferring missing data types

    print("\n----------- PARALLEL REDUCE -----------------------")
    print("-----KWARGS", kwargs)
    print("-----ARGS", args)
    print("-----Workunit", handled_args.workunit)
    print("attributes for workload", (list(inspect.signature(handled_args.workunit).parameters.values())[0]).annotation)
    print("attributes for policy", handled_args.policy.begin, handled_args.policy.end)
    print("name of workunit", handled_args.workunit.__name__)
    print("----------- END -----------------------\n")

    updated_types: UpdatedTypes = get_annotations("parallel_reduce", handled_args, args, passed_kwargs=kwargs)


    func, args = runtime_singleton.runtime.run_workunit(
        handled_args.name,
        [updated_types],
        handled_args.policy,
        handled_args.workunit,
        operation,
        **kwargs)

    workunit_cache[cache_key] = (func, args)
    return func(**args)

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
