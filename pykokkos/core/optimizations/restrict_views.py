import hashlib
import re
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pykokkos.core import cppast
from pykokkos.interface import Subview, View, ViewType, Trait


def get_restrict_views(views: Dict[str, ViewType]) -> Tuple[Set[str], str]:
    """
    Identify views that do not alias each other to apply the restrict
    keyword to

    :param views: the views passed as arguments
    :returns: the views that do not alias and a unique identifier
    """

    # Map from base view id() to a list of views that alias that view
    base_view_ids: Dict[int, Set[str]] = {}
    # Map from view name to the xp_array it has
    xp_arrays: Dict[str, Any] = {}

    xp_lib: Optional[ModuleType] = None
    for view_name, view in views.items():
        base_view: View = view.base_view if isinstance(view, Subview) else view

        if base_view.trait is Trait.Unmanaged:
            assert hasattr(base_view, "xp_array")
            xp_arrays[view_name] = base_view.xp_array

            base_type = str(type(base_view.xp_array))
            if "numpy" in base_type:
                import numpy as np
                xp_lib = np
            elif "cupy" in base_type:
                import cupy as cp
                xp_lib = cp
            else:
                raise RuntimeError(f"unsupported array type {base_type}")

        view_id: int = id(base_view)
        if view_id in base_view_ids:
            base_view_ids[view_id].add(view_name)
        else:
            base_view_ids[view_id] = {view_name}

    restricted_views: Set[str] = set()
    for view_id, view_set in base_view_ids.items():
        if len(view_set) == 1:
            restricted_views.update(view_set)

    aliasing_arrays: Set[str] = set()

    # TODO: Currently O(n^2) complexity
    for name in restricted_views:
        view: ViewType = views[name]
        has_xp_array: bool = hasattr(view, "xp_array")
        if not has_xp_array:
            continue

        xp_array = view.xp_array
        for other_name, other_array in xp_arrays.items():
            if other_name == name:
                continue

            if xp_lib.may_share_memory(xp_array, other_array):
                aliasing_arrays.add(name)
                aliasing_arrays.add(other_name)

    restricted_views -= aliasing_arrays
    restricted_signature: str = hashlib.md5("".join(sorted(restricted_views)).encode()).hexdigest()

    return restricted_views, restricted_signature


def get_stride_name(view: cppast.DeclRefExpr, dimension: int) -> cppast.DeclRefExpr:
    """
    Get the name of the stride parameter for a view

    :param view: the view being indexed
    :param dimension: the dimension of the requested stride
    :returns: the cppast representation of that stride
    """

    return cppast.DeclRefExpr(f"pk_stride_{view.declname}_stride{dimension}")


def get_restrict_ptr_name(view: cppast.DeclRefExpr) -> cppast.DeclRefExpr:
    """
    Get the name of the view pointer parameter for a view

    :param view: the view being indexed
    :returns: the cppast representation of the view pointer
    """

    return cppast.DeclRefExpr(f"pk_restrictptr_{view.declname}")


def get_restrict_ptr_type(view_type: str) -> Tuple[cppast.ClassType, int]:
    """
    Extract "double*" from "Kokkos::View<double**, ...>" as well as
    the rank to use as a parameter type

    :param view_type: the C++ type of the view
    :returns: the dtype part of the view as well as its rank
    """

    # Extract "double**" from Kokkos::View<double**, ...
    view_dtype: str = view_type.split("<")[1].split(",")[0]
    rank: int = view_dtype.count("*")

    # Replace "**" with "*"
    restrict_type: str = view_dtype.replace(rank * "*", "*") + " KOKKOS_RESTRICT"
    decltype = cppast.ClassType(restrict_type)

    return decltype, rank


def define_restrict_function(functor: cppast.RecordDecl, operation: str, workunit: cppast.MethodDecl, restrict_views: Set[str]) -> cppast.MethodDecl:
    """
    Define the kokkos function that will have restrict

    :param functor: the AST of the workunit to optimize
    :param operation: the operation type ("for", "reduce", or "scan")
    :param workunit: the translated workunit
    :param restrict_views: the views with the restrict keyword
    """

    # Add the tid, accumulator, and boolean
    params: List[cppast.ParmVarDecl] = [*workunit.params[1:]]
    for decl in functor.decls:
        if not isinstance(decl, cppast.DeclStmt):
            continue

        if isinstance(decl.decl, cppast.MethodDecl):
            continue

        field_name: cppast.DeclRefExpr = decl.decl.declname
        field_type: Union[str, cppast.Type] = decl.decl.decltype

        if field_name.declname not in restrict_views:
            params.append(cppast.ParmVarDecl(field_type, field_name))
            continue

        decltype: cppast.ClassType
        rank: int
        decltype, rank = get_restrict_ptr_type(field_type)

        params.append(cppast.ParmVarDecl(decltype, get_restrict_ptr_name(field_name)))
        for i in range(rank):
            params.append(cppast.ParmVarDecl(cppast.ClassType("size_t"), get_stride_name(field_name, i)))

    typename: str = functor.typename.typename
    name: str = f"pk_kokkos_function_{typename}"
    method = cppast.MethodDecl("KOKKOS_FUNCTION", cppast.ClassType("void"), name, params, workunit.body)
    method.is_const = True
    functor.add_decl(method)

    return method
    

def index_restrict_view(name: cppast.DeclRefExpr, indices: List[cppast.Expr]) -> cppast.ArraySubscriptExpr:
    """
    Get the indexing operation of a particular view with a given list
    of indices

    :param name: the name of the view being indexed
    :param indices: the list of indices
    """

    restrict_name: cppast.DeclRefExpr = get_restrict_ptr_name(name)

    full_index: cppast.Expr = cppast.BinaryOperator(indices[0], get_stride_name(name, 0), cppast.BinaryOperatorKind.Mul)
    for i, index in enumerate(indices[1:]):
        current_stride: cppast.DeclRefExpr = get_stride_name(name, i + 1) # Add one since we did zero before the loop
        current_mul = cppast.BinaryOperator(index, current_stride, cppast.BinaryOperatorKind.Mul)
        full_index = cppast.BinaryOperator(current_mul, full_index, cppast.BinaryOperatorKind.Add)

    return cppast.ArraySubscriptExpr(restrict_name, [full_index])


def adjust_kokkos_function_definition(
    attributes: str,
    return_type: cppast.ClassType,
    name: str,
    params: List[cppast.ParmVarDecl],
    body: cppast.CompoundStmt,
    restrict_views: Set[str]
) -> cppast.MethodDecl:
    """
    Adjust the definition of a kokkos function by replacing views with
    pointers and adding strides.

    :param attributes: the function attribute (e.g. "KOKKOS_FUNCTION")
    :param return_type: the return type of the method
    :param name: the method name
    :param params: the parameters of the method
    :param body: the body of the method
    :param restrict_views: the views with the restrict keyword
    :returns: the cppast representation of the method declaration
    """

    new_params: List[cppast.ParmVarDecl] = []
    for param in params:
        param_name: cppast.DeclRefExpr = param.declname

        # Account for fused views
        r = re.search("fused_(.*)_[0-9]*", param_name.declname)
        unfused_name: str = r.group(1) if r else param_name.declname

        if unfused_name not in restrict_views:
            new_params.append(param)
            continue

        field_type: str = param.decltype

        decltype: cppast.ClassType
        rank: int
        decltype, rank = get_restrict_ptr_type(field_type)

        new_params.append(cppast.ParmVarDecl(decltype, get_restrict_ptr_name(param_name)))
        for i in range(rank):
            new_params.append(cppast.ParmVarDecl(cppast.ClassType("size_t"), get_stride_name(param_name, i)))
    
    return cppast.MethodDecl(attributes, return_type, name, new_params, body)


def adjust_kokkos_function_call(
    function: cppast.DeclRefExpr,
    args: List[cppast.Expr],
    restrict_views: Set[str],
    views: Dict[cppast.DeclRefExpr, cppast.Type]
) -> cppast.CallExpr:
    """
    Adjust the call to a kokkos function by accounting for the new
    parameters introduced when adjusting the definition

    :param function: the kokkos function being called
    :param args: the arguments to the function
    :param restrict_views: the views with the restrict keyword
    :param views: map from view name to type
    :returns: the function call with the proper number of arguments
    """

    new_args: List[cppast.Expr] = []
    for arg in args:
        if not isinstance(arg, cppast.DeclRefExpr):
            new_args.append(arg)
            continue

        if arg.declname not in restrict_views:
            new_args.append(arg)
            continue

        new_args.append(get_restrict_ptr_name(arg))

        view_type: cppast.ClassType = views[arg]

        # View type here is of the form View2D<double, ...>
        rank = int(re.search(r'\d+', view_type.typename).group())

        for i in range(rank):
            new_args.append(get_stride_name(arg, i))

    return cppast.CallExpr(function, new_args)


def add_function_call(kokkos_function: cppast.MethodDecl, workunit: cppast.MethodDecl, restrict_views: Set[str]) -> None:
    """
    Add the call to the restricted Kokkos function call to the
    workunit body

    :param kokkos_function: the restricted kokkos function
    :param workunit: the workunit being modified
    :param restrict_views: the views with the restrict keyword
    """

    args: List[cppast.Expr] = []
    for param in kokkos_function.params:
        name: cppast.DeclRefExpr = param.declname

        if name.declname.startswith("pk_restrictptr_"):
            view_name = name.declname.replace("pk_restrictptr_", "")
            get_call = cppast.MemberCallExpr(view_name, cppast.DeclRefExpr("data"), [])
            args.append(get_call)

        elif name.declname.startswith("pk_stride_"): # Is a stride parameter
            split_name: List[str] = name.declname.split("_")
            dimension = int(split_name[-1].replace("stride", ""))
            view_name = name.declname.replace("pk_stride_", "").replace(f"_stride{dimension}", "")
            args.append(cppast.MemberCallExpr(cppast.DeclRefExpr(view_name), cppast.DeclRefExpr(f"stride_{dimension}"), []))

        else:
            args.append(name)

    call = cppast.CallExpr(kokkos_function.declname, args)
    workunit._body = cppast.CallStmt(call)


def add_restrict_views(functor: cppast.RecordDecl, operation: str, workunit: cppast.MethodDecl, restrict_views: Set[str]):
    """
    Define the restrict kokkos function and call it

    :param functor: the functor containing the workunit
    :param operation: the operation type ("for", "reduce", or "scan")
    :param workunit: the translated workunit
    :param restrict_views: the views with the restrict keyword
    """

    kokkos_function: cppast.MethodDecl = define_restrict_function(functor, operation, workunit, restrict_views)
    add_function_call(kokkos_function, workunit, restrict_views)
