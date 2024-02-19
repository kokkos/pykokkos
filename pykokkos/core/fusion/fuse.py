import ast
import os
from typing import Any, Dict, List, Set, Tuple, Union

from .util import DeclarationsVisitor, VariableRenamer


def fuse_workunit_kwargs_and_params(
    workunit_trees: List[ast.AST],
    kwargs: Dict[str, Any],
    operation: str
) -> Tuple[Dict[str, Any], List[ast.arg]]:
    """
    Fuse the parameters and runtime arguments of a list of workunits and rename them as necessary

    :param workunits_trees: the list of workunit trees (ASTs) being merged
    :param kwargs: the keyword arguments passed to the call
    :param operation: they type of parallel operation ("parallel_for", "parallel_reduce", or "parallel_scan")
    :returns: a tuple of the fused kwargs and the combined inspected parameters
    """

    if operation == "parallel_scan":
        raise RuntimeError("parallel_scan not supported for fusion")

    fused_kwargs: Dict[str, Any] = {}
    fused_params: List[ast.arg] = []
    fused_params.append(ast.arg(arg="fused_tid", annotation=int))

    if operation == "parallel_reduce":
        fused_params.append(ast.arg(arg="pk_fused_acc"))

    view_ids: Set[int] = set()

    for workunit_idx, tree in enumerate(workunit_trees):
        key: str = f"args_{workunit_idx}"
        if key not in kwargs:
            raise RuntimeError(f"kwargs not specified for workunit {workunit_idx} with key {key}")
        current_kwargs: Dict[str, Any] = kwargs[key]

        current_params: List[ast.arg] = [p for p in tree.args.args]
        if operation == "parallel_reduce" and workunit_idx == len(workunit_trees) - 1:
            # Skip the thread ID and the accumulator
            current_params = current_params[2:]
        else:
            # Skip the thread ID
            current_params = current_params[1:]

        for p in current_params:
            current_arg = current_kwargs[p.arg]
            if "PK_FUSE_ARGS" in os.environ and id(current_arg) in view_ids:
                continue

            view_ids.add(id(current_arg))

            fused_name: str = f"fused_{p.arg}_{workunit_idx}"
            fused_kwargs[fused_name] = current_kwargs[p.arg]
            fused_params.append(ast.arg(arg=fused_name, annotation=p.annotation))

    return fused_kwargs, fused_params


# def fuse_workunit_params(passed_kwargs: Dict) -> Tuple[Dict[str, Any], OrderedDict[Tuple[str, int], str]]:
#     """
#     Combine params in a workunit by renaming them

#     :param passed_kwargs: the original kwargs passed by the user
#     :returns: a tuple of Dicts, the first mapping kwargs and another mapping old name to new name
#     """

#     fused_kwargs = {}
#     fused_params = collections.OrderedDict()

#     for idx, kwargs in enumerate(passed_kwargs.values()):
#         for kwarg, value in kwargs.items():
#             param_name: str = f"fused_{kwarg}_{idx}"
#             fused_kwargs[param_name] = value
#             fused_params[(kwarg, idx)] = param_name

#     return fused_kwargs, fused_params


def fuse_arguments(all_args: List[ast.arguments], **kwargs) -> Tuple[ast.arguments, Dict[Tuple[str, int], str]]:
    """
    Fuse the ast argument object into one

    :param all_args: a list of all ast.arguments objects in the workunits
    :returns: a tuple of the new ast.arguments and a map from the old names to the new ones
    """

    name_map: Dict[Tuple[str, int], str] = {} # Maps from a tuple of old arg name and workunit idx to the new one

    # The fused args are initialized contain the thread id
    new_tid: str = "fused_tid"
    fused_args = ast.arguments(args=[ast.arg(arg=new_tid, annotation=ast.Name(id='int', ctx=ast.Load()))])

    new_acc: str = "pk_fused_acc"

    # Map from view ID to fused name
    fused_view_names: Dict[int, str] = {}

    for workunit_idx, args in enumerate(all_args):
        key: str = f"args_{workunit_idx}"
        if key not in kwargs:
            raise RuntimeError(f"kwargs not specified for workunit {workunit_idx} with key {key}")
        current_kwargs: Dict[str, Any] = kwargs[key]

        for arg_idx, arg in enumerate(args.args):
            old_name: str = arg.arg
            key = (old_name, workunit_idx)
            new_name: str

            # Record what the thread id was but do not add it again
            if arg_idx == 0:
                name_map[key] = new_tid
                continue

            # Account for accumulator
            if old_name not in current_kwargs and arg_idx == 1:
                name_map[key] = new_acc
                fused_args.args.insert(1, ast.arg(arg=new_acc, annotation=arg.annotation))
                continue

            current_arg = current_kwargs[old_name]
            if "PK_FUSE_ARGS" in os.environ and id(current_arg) in fused_view_names:
                name_map[key] = fused_view_names[id(current_arg)]
                continue

            new_name = f"fused_{old_name}_{workunit_idx}"
            fused_view_names[id(current_arg)] = new_name
            name_map[key] = new_name
            fused_args.args.append(ast.arg(arg=new_name, annotation=arg.annotation))

    return fused_args, name_map


def fuse_bodies(bodies: List[List[ast.stmt]], name_map: Dict[Tuple[str, int], str]) -> List[ast.stmt]:
    """
    Fuse the bodies of the workunits and rename all declared variables

    :param bodies: the list of statements in each workunit's body
    :param name_map: a map from the old variable names to the new ones
    :returns: the fused bodies in one list
    """

    fused_body: List[ast.stmt] = []

    for idx, body in enumerate(bodies):
        declarations = DeclarationsVisitor()
        for statement in body:
            declarations.visit(statement)

        for declaration in declarations.declarations:
            new_name: str = f"fused_{declaration}_{idx}"
            name_map[(declaration, idx)] = new_name

        renamer = VariableRenamer(name_map, idx)
        fused_body += [renamer.visit(s) for s in body]

    return fused_body


def fuse_decorators(decorators: List[Union[ast.Attribute, ast.Call]], name_map: Dict[Tuple[str, int], str]) -> List[ast.Call]:
    """
    Fuse the decorators of the workunits

    :param decorators: the list of decorators (e.g. pk.workunit())
    :param name_map: a map from the old variable names to the new ones
    :returns: the combined list of keywords
    """

    renamer = VariableRenamer(name_map)
    fused_keywords: List[ast.keyword] = [renamer.visit(d) for d in decorators]

    return ast.Call(func=decorators[0].func, args=[], keywords=fused_keywords)


def fuse_ASTs(ASTs: List[ast.FunctionDef], name: str, **kwargs) -> ast.FunctionDef:
    """
    Fuse the ASTs of multiple workunits together

    :param ASTs: the asts to be fused
    :param name: the name of the fused workunit
    :returns: the AST of the fused workunit
    """

    args: ast.arguments
    name_map: Dict[str, str]
    args, name_map = fuse_arguments([AST.args for AST in ASTs], **kwargs)

    # decorator: ast.Call = fuse_decorators([AST.decorator_list[0] for AST in ASTs], name_map)
    body: List[ast.stmt] = fuse_bodies([AST.body for AST in ASTs], name_map)

    # return ast.FunctionDef(name=name, args=args, decorator_list=[decorator], body=body)
    return ast.FunctionDef(name=name, args=args, decorator_list=ASTs[0].decorator_list, body=body)


def fuse_sources(sources: List[Tuple[List[str], int]]):
    pass


def fuse_workunits(
    fused_name: str,
    ASTs: List[ast.FunctionDef],
    sources: List[Tuple[List[str], int]],
    **kwargs
) -> Tuple[ast.FunctionDef, Tuple[List[str], int]]:
    """
    Merge a list of workunits into a single object

    :param names: the name of the fused workunit
    :param ASTs: the parsed python ASTs to be fused
    :param sources: the raw source of the workunits to be fused
    """

    AST: ast.FunctionDef = fuse_ASTs(ASTs, fused_name, **kwargs)
    source: Tuple[List[str], int] = fuse_sources(sources)

    return AST, source
