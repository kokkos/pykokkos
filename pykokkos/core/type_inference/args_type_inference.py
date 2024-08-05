import ast
from dataclasses import dataclass
import hashlib
import inspect
import pickle
from typing import  Callable, Dict, Optional, Tuple, Union, List

from pykokkos.core.fusion import fuse_workunit_kwargs_and_params
from pykokkos.core.module_setup import get_metadata
from pykokkos.interface import (
    MDRangePolicy, TeamPolicy, TeamThreadRange, RangePolicy, ExecutionPolicy, ExecutionSpace,
    View, ViewType,
    DataType, DataTypeClass
)


@dataclass
class UpdatedTypes:
    """
    Class for storing inferred type annotation information 
    (Making Pykokkos more pythonic by automatically inferring types)
    """

    workunit: Callable
    inferred_types: Dict[str, str] # type information stored as string: identifier -> type
    param_list: List[str]

@dataclass
class UpdatedDecorator:
    """
    Class for storing inferred decorator specifiers
    (Making Pykokkos more pythonic by automatically inferring types)
    """

    inferred_decorator: Dict[str, Dict[str, str]] # against each view (first dict) values for layout, space, and trait
    param_list: List[str] # Original params needed to reset AST incase user provides all annotations

# DataType class has all supported pk datatypes, we ignore class members starting with __, add enum duplicate aliases
SUPPORTED_NP_DTYPES = [attr for attr in dir(DataType) if not attr.startswith("__")] + ["float64", "float32"]

# Cache for original argument nodes: Maps stringified workunit reference, e.g str(workunit_name), to the original ast.arguments node
ORIGINAL_PARAMS: Dict[str, ast.arguments] = {}


def check_missing_annotations(param_list: List[ast.arg]) -> bool:
    '''
    Check if any annotation node for parent argument node is none

    :param param_list: list of ast.arg objects to run annotation check against
    :returns: True if any parameter is missing annotation, false otherwise
    '''

    missing = False
    for param in param_list:
        if param.annotation is None:
            missing = True
            break
    return missing

def get_annotations(parallel_type: str, workunit_trees: Union[Tuple[Callable, ast.AST], List[Tuple[Callable, ast.AST]]], policy: Union[ExecutionPolicy, int], passed_kwargs) -> Optional[UpdatedTypes]:
    '''
    Infer the datatypes for arguments passed against workunit parameters

    :param parallel_type: A string identifying the type of parallel dispatch ("parallel_for", "parallel_reduce" ...)
    :param workunit_trees: workunit object and its tree in tuples. Can be a list or standalone
    :param policy: The execution policy for this parallel dispatch - used to handle policy args
    :param passed_kwargs: raw keyword arguments passed to the dispatch
    :returns: UpdateTypes object or None if there are no annotations to be inferred
    '''
    # x.arg is param identifier (name) and x.annotation is the annotation object .id is the annotated type
    param_list: List[ast.arg]

    if isinstance(workunit_trees, list):
        if parallel_type == "parallel_scan":
            raise RuntimeError("Cannot do kernel fusion with parallel scan")
        workunit = [w for w, _ in workunit_trees]
        trees = [t for _, t in workunit_trees]
        passed_kwargs, param_list = fuse_workunit_kwargs_and_params(trees, passed_kwargs, parallel_type)
    else:
        workunit, entity_AST = workunit_trees
        param_list = [x for x in entity_AST.args.args]

    
    updated_types = UpdatedTypes(
        workunit=workunit, 
        inferred_types={}, 
        param_list=param_list, 
    )
    policy_params: int = len(policy.begin) if isinstance(policy, MDRangePolicy) else 1

    # check if all annotations are already provided
    missing = check_missing_annotations(param_list)
    if not missing: return None

    # accumulator 
    if parallel_type == "parallel_reduce":
        policy_params += 1
    # accumulator + lass_pass
    if parallel_type == "parallel_scan":
        policy_params += 2

    # Handling policy parameters
    updated_types = infer_policy_args(param_list, policy_params, policy, parallel_type, updated_types)

    # Policy parameters are the only parameters
    if not len(passed_kwargs):
        if not len(updated_types.inferred_types): return None
        return updated_types

    # Additional keyword arguments that may have been passed
    updated_types = infer_other_args(param_list, passed_kwargs, updated_types)

    if not len(updated_types.inferred_types): return None

    return updated_types


def get_views_decorator(parallel_type: str, workunit_trees: List[Tuple[Callable, ast.AST]], passed_kwargs) -> UpdatedDecorator:
    '''
    Extract the layout, space, trait information against view: will be used to construct decorator
    specifiers

    :param parallel_type: A string identifying the type of parallel dispatch ("parallel_for", "parallel_reduce" ...)
    :param handled_args: Processed arguments passed to the dispatch
    :param passed_kwargs: Keyword arguments passed to parallel dispatch (has views)
    :returns: UpdatedDecorator object 
    '''

    param_list: List[ast.arg]
    if isinstance(workunit_trees, list):
        trees = [t for _, t in workunit_trees]
        passed_kwargs, param_list = fuse_workunit_kwargs_and_params(trees, passed_kwargs, parallel_type)
        param_list = [p.arg for p in param_list]
    else:
        _, entity_AST = workunit_trees
        param_list = [p.arg for p in entity_AST.args.args]

    updated_decorator = UpdatedDecorator(
        inferred_decorator = {},
        param_list=param_list
    )

    for kwarg in passed_kwargs:
        if kwarg not in param_list:
            raise Exception(f"Unknown kwarg: {kwarg} passed")

        value = passed_kwargs[kwarg]
        if not isinstance(value, ViewType):
            continue

        if kwarg not in updated_decorator.inferred_decorator: 
            updated_decorator.inferred_decorator[kwarg] = {}

        updated_decorator.inferred_decorator[kwarg]['trait'] = str(value.trait).split(".")[1]
        updated_decorator.inferred_decorator[kwarg]['layout'] = str(value.layout).split(".")[1]
        updated_decorator.inferred_decorator[kwarg]['space'] = str(value.space).split(".")[1]

    if not len(updated_decorator.inferred_decorator):
        return None

    return updated_decorator


def infer_policy_args(
    param_list: List[ast.arg],
    policy_params: int,
    policy: ExecutionPolicy,
    parallel_type: str,
    updated_types: UpdatedTypes
    ) -> UpdatedTypes:
    '''
    Infer the types of policy arguments

    :param param_list: list of parameter objects that are present in the workunit signature
    :param policy_params: the number of initial parameters that are dedicated to policy (in param_list/signature)
    :param policy: the pykokkos execution policy for workunit
    :param parallel_type: "parallel_for" or "parallel_reduce" or "parallel_scan"
    :param updated_types: UpdatedTypes object to store inferred types information
    :returns: Updated UpdatedTypes object with inferred types
    '''

    for i in range(policy_params):
        param = param_list[i]

        if param.annotation is not None:
            continue

        # Check policy and apply annotation(s)
        if isinstance(policy, RangePolicy) or isinstance(policy, TeamThreadRange):
            # only expects one param
            if i == 0:
                updated_types.inferred_types[param.arg] = "int"

        elif isinstance(policy, TeamPolicy):
            if i == 0:
                updated_types.inferred_types[param.arg] = 'TeamMember'

        elif isinstance(policy, MDRangePolicy):
            total_dims = len(policy.begin) 
            if i < total_dims:
                updated_types.inferred_types[param.arg] = "int"
        else:
            raise ValueError("Automatic annotations not supported for this policy")

        # last policy param for parallel reduce and second last for parallel_scan is always the accumulator; the default type is double
        if i == policy_params - 1 and parallel_type == "parallel_reduce" or i == policy_params - 2 and parallel_type == "parallel_scan":
            updated_types.inferred_types[param.arg] = "Acc:double"

        if i == policy_params - 1 and parallel_type == "parallel_scan":
            updated_types.inferred_types[param.arg] = "bool"

    return updated_types


def infer_other_args(
    param_list: List[inspect.Parameter],
    passed_kwargs,
    updated_types: UpdatedTypes
    ) -> UpdatedTypes:
    '''
    Infer the types of arguments (after the policy arguments)

    :param param_list: list of parameter objects that are present in the workunit signature
    :param policy_params: the number of initial parameters that are dedicated to policy (in param_list/signature)
    :param args_list: List of arguments passed to the parallel dispactch (e.g args for parallal_for())
    :param start_idx: The index for the first non policy argument in args_list
    :param updated_types: UpdatedTypes object to store inferred types information
    :returns: Updated UpdatedTypes object with inferred types
    '''

    for name, value in passed_kwargs.items():
        param = None
        for iparam in param_list:
            if name == iparam.arg: param = iparam

        if param is None:
            continue

        if param.annotation is not None:
            continue

        param_type = type(value).__name__

        # switch integer values over 31 bits (signed positive value) to numpy:int64
        if param_type == "int" and value.bit_length() > 31:
            param_type = "numpy:int64"

        # check if package name is numpy (handling numpy primitives)
        pckg_name = type(value).__module__

        if pckg_name == "numpy":
            if param_type not in SUPPORTED_NP_DTYPES:
                err_str = f"Numpy type {param_type} is unsupported"
                raise TypeError(err_str)

            if param_type == "float64": param_type = "double"
            if param_type == "float32": param_type = "float"
            # numpy:<type>, Will switch to pk.<type> in parser.fix_types
            param_type = pckg_name +":"+ param_type

        if isinstance(value, ViewType):
            view_dtype = get_pk_datatype(value.dtype)
            if not view_dtype:
                raise TypeError("Cannot infer datatype for view:", param.arg)

            param_type = "View"+str(len(value.shape))+"D:"+view_dtype

        updated_types.inferred_types[param.arg] = param_type 

    return updated_types


def get_pk_datatype(view_dtype):
    '''
    Infer the dataype of view e.g pk.View1D[<infer this>]

    :param view_dtype: view.dtype whose datatype is to be determined as string
    :returns: the type of custom pkDataType as string
    '''

    dtype = None
    if isinstance(view_dtype, DataType):
        dtype = str(view_dtype.name)

    elif inspect.isclass(view_dtype) and issubclass(view_dtype, DataTypeClass):
        dtype = str(view_dtype.__name__)

    if dtype == "float64": dtype = "double"
    if dtype == "float32": dtype = "float"

    return dtype


def get_types_signature(updated_types: UpdatedTypes, updated_decorator: UpdatedDecorator, execution_space: ExecutionSpace) -> str:
    '''
    Generates a signature/hash to represent the signature of the workunit: used for module setup

    :param inferred_types: Dict that stores arg name against its inferred type
    :param inferred_decorator: Dict that stores the layout of view name against its inferred layout
    :param execution_space: The execution space of the workunit
    :returns: a string representing inferred types
    '''

    signature:str = ""
    if updated_types is not None:
        for name, i_type in updated_types.inferred_types.items():
            signature += i_type

    if updated_decorator is not None:
        for name in updated_decorator.inferred_decorator:
            space_str = str(execution_space)
            layout_str = updated_decorator.inferred_decorator[name]['layout']
            trait_str = updated_decorator.inferred_decorator[name]['trait']

            signature += layout_str +  space_str + trait_str

    if signature == "":
        return None

    # Compacting
    signature = hashlib.md5(signature.encode()).hexdigest()

    return signature


def get_type_str(inspect_type: inspect.Parameter.annotation) -> str:
    '''
    Given a user provided inspect.annotation string return the equivalent type inferrence string (used internally).
    This function is typically invoked when resetting the AST

    :param inspect_type: annotation object provided by inspect package
    :return: string for the same type as supported in type_inference.py
    '''

    basic_type = None
    if isinstance(inspect_type, type):
        basic_type = str(inspect_type.__name__)
    else:
        # Support for python 3.8, string manip needed :(
        t_str = str(inspect_type)
        t_str = t_str.replace("pykokkos.interface.data_types.", "")
        t_str = t_str.replace("pykokkos.interface.views.", "")
        if ".Acc[" in t_str:
            basic_type = "Acc"
        elif "TeamMember" in t_str:
            basic_type = "TeamMember"
        elif "View" in t_str:
            basic_type = (t_str.split('[')[0]).strip()

    assert basic_type is not None, f"Inference failed for {inspect_type}"

    # just a basic primitive
    if "pykokkos" not in str(inspect_type):
        return basic_type

    if basic_type == "Acc":
        return "Acc:double"

    if basic_type == "TeamMember":
        return "TeamMember"

    type_str = str(inspect_type).replace('pykokkos.interface.data_types.', 'pk.')

    if "views" in type_str:
        # is a view, only need the slice
        type_str = type_str.split('[')[1]
        type_str = type_str[:-1]
        type_str = type_str.replace("pk.", "")

        return basic_type+":"+type_str
    
    # just a numpy primitive
    if "pk." in type_str and basic_type in SUPPORTED_NP_DTYPES:
        type_str = "numpy:" + basic_type
        return type_str

    err_str = f"User provided unsupported annotation: {inspect_type}"
    raise TypeError(err_str)


def get_type_info(
        operation: str,
        parser,
        policy: ExecutionPolicy,
        workunit: Union[List[Callable], Callable],
        passed_kwargs
        ) -> Tuple[Optional[UpdatedTypes], Optional[UpdatedDecorator], Optional[str]]:
    '''
    Process arg nodes for each workunit received: 
    1) check if type inference is supported
    2) restore the original argument nodes and finally return this information
    
    :param operation: the type of parallel dispactch (parallel_for, reduce or scan)
    :param parser: the parser object for the workunit
    :param policy: the execution policy of the dispatch
    :param workunit: list of workunits to be fused or singular workunit not to be fused,
    :param passed_kwargs: original kwargs passed to the parallel dispatch
    :returns: Tuple of: 
        1) UpdatedTypes object if type inference is supported 
        2) UpdatedDecorator object if decorator inference is supported
        3) types_signature string: a hash string representing those types/layouts
        or
        (None, None, None) if not supported
    '''

    is_standalone_workunit: bool = True
    list_passed: bool = True
    execution_space: ExecutionSpace = policy.space.space

    entity_AST: Union[List[ast.AST], ast.AST] = []

    if not isinstance(workunit, list):
            workunit = [workunit] # for easier transformations
            parser = [parser]
            list_passed = False

    for this_workunit, this_parser in zip(workunit, parser):
        this_metadata = get_metadata(this_workunit)
        this_tree = this_parser.get_entity(this_metadata.name).AST
        workunit_str = str(this_workunit)

        if not isinstance(this_tree, ast.FunctionDef):
            # Workunit must be a function on its own (not in a functor)
            is_standalone_workunit = False
            entity_AST.append(this_tree)
            continue

        is_missing_annotations: bool = (
            workunit_str in ORIGINAL_PARAMS
            or
            check_missing_annotations(this_tree.args.args)
            )

        if is_missing_annotations:
            this_tree = restore_original_args(workunit_str, this_tree)
        
        entity_AST.append(this_tree)

    workunit_trees: Union[List[Tuple[Callable, ast.AST]], Tuple[Callable, ast.AST]]
    workunit_trees, workunit, entity_AST = prepare_fusion_args(list_passed, workunit, entity_AST)

    updated_types: Optional[UpdatedTypes] = None
    updated_decorator: Optional[UpdatedDecorator] = None
    types_signature: Optional[str] = None

    if is_standalone_workunit:
        updated_types = get_annotations(f"parallel_{operation}", workunit_trees, policy, passed_kwargs)
        updated_decorator = get_views_decorator(f"parallel_{operation}", workunit_trees, passed_kwargs)
        types_signature = get_types_signature(updated_types, updated_decorator, execution_space)

    return updated_types, updated_decorator, types_signature


def prepare_fusion_args(
        list_passed: bool, 
        workunit: List[Callable],
        entity_AST: List[ast.AST]
        ) -> Tuple[
            Union[List[Tuple[Callable, ast.AST]],Tuple[Callable, ast.AST]],
            Union[List[Callable], Callable],
            Optional[List[ast.AST]]
        ]:
    '''
    Invoked in run_workunit in runtime.py only: Adjust the types of the arguments according to fusion.
    That is determine if the final type will be a list (fusion intended) or not (no fusion)

    :param list_passed: True if a list of workunits was passed to the runtime (fusion intended)
    :param workunit: list of workunits to be or not to be fused
    :entity_AST: list of workunit asts to be or not to be fused
    :returns: if there is no fusion, return the singular elements, otherwise, the lists of those elements
    '''
    if not list_passed: # revert to singular tuple if not list originally
        workunit_trees = (workunit[0], entity_AST[0])
        workunit = workunit[0]
        entity_AST = None # No fusion
    else:
        workunit_trees = list(zip(workunit, entity_AST))
    
    return (workunit_trees, workunit, entity_AST)


def restore_original_args(workunit_str: str, tree: ast.AST) -> ast.AST:
    '''
    Restore the original argument nodes in the ast (as if it was freshly read from source)
        - this is achieved with caching and returning that state on the first invokation of
    the workunit in question

    :param workunit_str: Stringified version of the workunit reference, used as key to cache
    :param tree: the ast for the workunit
    :returns: workunit tree with argument nodes restored
    '''
    if workunit_str in ORIGINAL_PARAMS:
        tree.args.args = pickle.loads(ORIGINAL_PARAMS[workunit_str])
    else:
        # first call, store the original params
        ORIGINAL_PARAMS[workunit_str] = pickle.dumps(tree.args.args)

    return tree