import ast
from typing import Dict, List, Optional, Tuple, Union

from pykokkos.core import cppast

from . import visitors_util


class ParameterVisitor(ast.NodeVisitor):
    """
    Gets the members of a workunit
    """

    def __init__(self, src: Tuple[List[str], int], param_begin: int, pk_import: str, debug: bool):
        """
        ParameterVisitor constructor

        :param src: the python source code of the workload
        :param param_begin: where workunit argument begins (excluding tid/acc)
        :param pk_import: the identifier used to access the PyKokkos package
        :param debug: if true, prints the python AST when an error is encountered
        """

        self.src: Tuple[List[str], int] = src
        self.param_begin: int = param_begin
        self.pk_import: str = pk_import
        self.debug: bool = debug

        self.fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType] = {}
        self.views: Dict[cppast.DeclRefExpr, cppast.ClassType] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit the workunit function definition

        :param node: the function definition node
        """

        self.visit(node.args)

    def visit_arguments(self, node: ast.arguments) -> None:
        """
        Visit the parameters of the workunit

        :param node: the arguments node
        """

        args: List[ast.arg] = []

        if len(node.args) < self.param_begin:
            self.error(node.parent, "Missing tid and/or accumulator argument")

        args = node.args[self.param_begin:]

        for a in args:
            self.visit(a)

    def visit_arg(self, node: ast.arg) -> None:
        """
        Visit an individual parameter

        :param node: the arg node
        """

        annotation: Union[ast.Name, ast.Attribute] = node.annotation

        declref = cppast.DeclRefExpr(node.arg)
        decltype: Optional[cppast.Type] = visitors_util.get_type(annotation, self.pk_import)

        if decltype is None:
            self.error(node, "Type is not supported")

        # just checking decltype might be enough
        is_field: bool = isinstance(annotation, ast.Name) or \
                isinstance(decltype, cppast.PrimitiveType)
        if is_field:
            self.fields[declref] = decltype
        else:
            self.views[declref] = decltype

    def error(self, node: ast.AST, message: str):
        visitors_util.error(self.src, self.debug, node, message)
