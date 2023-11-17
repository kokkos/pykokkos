import ast
import inspect
from typing import Any, Callable, Dict, List, Set, Tuple, Union


def get_node_name(node: Union[ast.Attribute, ast.Name]) -> str:
    """
    Copied from visitors_util.py due to circular import
    """

    name: str
    if isinstance(node, ast.Attribute):
        name = node.attr
    else:
        name = node.id

    return name


class DeclarationsVisitor(ast.NodeVisitor):
    """
    Get all variable declarations
    """

    def __init__(self) -> None:
        self.declarations: Set[str] = set()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self.declarations.add(get_node_name(node.target))


class VariableRenamer(ast.NodeTransformer):
    """
    Renames variables in a fused ast according to a map
    """

    def __init__(self, name_map: Dict[Tuple[str, int], str], workunit_idx: int):
        self.name_map = name_map
        self.workunit_idx = workunit_idx

    def visit_Name(self, node: ast.Name) -> Any:
        key = (node.id, self.workunit_idx)
        # If the name is not mapped, keep the original name
        node.id = self.name_map.get(key, node.id)
        return node

    def visit_keyword(self, node: ast.keyword) -> Any:
        key = (node.id, self.workunit_idx)
        # If the name is not mapped, keep the original name
        node.arg = self.name_map.get(key, node.arg)
        return node
    

def fuse_workunit_kwargs_and_params(
    workunits: List[Callable],
    kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[inspect.Parameter]]:
    """
    Fuse the parameters and runtime arguments of a list of workunits and rename them as necessary

    :param workunits: the list of workunits being merged
    :param kwargs: the keyword arguments passed to the call
    :returns: a tuple of the fused kwargs and the combined inspected parameters
    """

    fused_kwargs: Dict[str, Any] = {}
    fused_params: List[inspect.Parameter] = [inspect.Parameter("fused_tid", inspect.Parameter.POSITIONAL_OR_KEYWORD)]

    for workunit_idx, workunit in enumerate(workunits):
        key: str = f"args_{workunit_idx}"
        if key not in kwargs:
            raise RuntimeError(f"kwargs not specified for workunit {workunit_idx} with key {key}")
        current_kwargs: Dict[str, Any] = kwargs[key]
    
        current_params: List[inspect.Parameter] = list(inspect.signature(workunit).parameters.values())
        for p in current_params[1:]: # Skip the thread ID
            fused_name: str = f"fused_{p.name}_{workunit_idx}"
            fused_kwargs[fused_name] = current_kwargs[p.name]
            fused_params.append(inspect.Parameter(fused_name, p.kind))

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


def fuse_arguments(all_args: List[ast.arguments]) -> Tuple[ast.arguments, Dict[Tuple[str, int], str]]:
    """
    Fuse the ast argument object into one

    :param all_args: a list of all ast.arguments objects in the workunits
    :returns: a tuple of the new ast.arguments and a map from the old names to the new ones
    """

    name_map: Dict[Tuple[str, int], str] = {} # Maps from a tuple of old arg name and workunit idx to the new one

    # The fused args are initialized contain the thread id
    new_tid: str = "fused_tid"
    fused_args = ast.arguments(args=[ast.arg(arg=new_tid, annotation=ast.Name(id='int', ctx=ast.Load()))])

    for workunit_idx, args in enumerate(all_args):
        for arg_idx, arg in enumerate(args.args): # Skip "self"
            old_name: str = arg.arg
            key = (old_name, workunit_idx)
            new_name: str

            # Record what the thread id was but do not add it again
            if arg_idx == 0:
                name_map[key] = new_tid
                continue

            new_name = f"fused_{old_name}_{workunit_idx}"
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


def fuse_ASTs(ASTs: List[ast.FunctionDef], name: str) -> ast.FunctionDef:
    """
    Fuse the ASTs of multiple workunits together

    :param ASTs: the asts to be fused
    :param name: the name of the fused workunit
    :returns: the AST of the fused workunit
    """

    args: ast.arguments
    name_map: Dict[str, str]
    args, name_map = fuse_arguments([AST.args for AST in ASTs])

    # decorator: ast.Call = fuse_decorators([AST.decorator_list[0] for AST in ASTs], name_map)
    body: List[ast.stmt] = fuse_bodies([AST.body for AST in ASTs], name_map)

    # return ast.FunctionDef(name=name, args=args, decorator_list=[decorator], body=body)
    return ast.FunctionDef(name=name, args=args, decorator_list=ASTs[0].decorator_list, body=body)


def fuse_sources(sources: List[Tuple[List[str], int]]):
    pass


def fuse_workunits(
    names: List[str],
    ASTs: List[ast.FunctionDef],
    sources: List[Tuple[List[str], int]],
) -> Tuple[str, ast.FunctionDef, Tuple[List[str], int]]:
    """
    Merge a list of workunits into a single object

    :param names: the names of the workunits to be fused
    :param ASTs: the parsed python ASTs to be fused
    :param sources: the raw source of the workunits to be fused
    """

    name: str = "_".join(names)
    AST: ast.FunctionDef = fuse_ASTs(ASTs, name)

    source: Tuple[List[str], int] = fuse_sources(sources)

    return name, AST, source
