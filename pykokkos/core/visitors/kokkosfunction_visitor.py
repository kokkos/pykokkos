import ast
from typing import List

from pykokkos.core import cppast

from . import visitors_util
from .pykokkos_visitor import PyKokkosVisitor


class KokkosFunctionVisitor(PyKokkosVisitor):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> cppast.MethodDecl:
        if not self.is_valid_kokkos_function(node):
            self.error(node, "Invalid Kokkos function")

        return_type: cppast.ClassType
        if self.is_void_function(node):
            return_type = cppast.ClassType("void")
        else:
            return_type = visitors_util.get_type(node.returns, self.pk_import)

        if return_type is None:
            self.error(node, "Return type is not supported for translation")

        params: List[cppast.ParmVarDecl] = self.visit(node.args)

        name: str = node.name
        body = cppast.CompoundStmt([self.visit(b) for b in node.body])
        attributes: str = "KOKKOS_FUNCTION"

        method = cppast.MethodDecl(attributes, return_type, name, params, body)
        method.is_const = True

        return method

    # Checks that a function marked as kokkos_function
    # is annotated with a return type if it returns
    def is_valid_kokkos_function(self, node) -> bool:
        # Is the return type annotation missing
        if node.returns is None:
            return False

        # Is the type annotation for any argument missing (excluding self)
        if any(arg.annotation is None and arg.arg != "self" for arg in node.args.args):
            return False

        return True
