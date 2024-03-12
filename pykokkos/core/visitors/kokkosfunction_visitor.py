import ast
import os
import re
from typing import List, Optional

from pykokkos.core import cppast
from pykokkos.core.optimizations.restrict_views import (
    adjust_kokkos_function_call, adjust_kokkos_function_definition
)

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

        method: cppast.MethodDecl
        if "PK_RESTRICT" in os.environ:
            method = adjust_kokkos_function_definition(attributes, return_type, name, params, body, self.restrict_views)
        else:
            method = cppast.MethodDecl(attributes, return_type, name, params, body)

        method.is_const = True

        return method

    def visit_Call(self, node: ast.Call) -> cppast.CallExpr:
        # Copied from workunit_visitor.py
        name: str = visitors_util.get_node_name(node.func)
        args: List[cppast.Expr] = [self.visit(a) for a in node.args]

        function = cppast.DeclRefExpr(f"Kokkos::{name}")

        atomic_fetch_op: re.Pattern = re.compile("atomic_fetch_*")
        is_atomic_fetch_op: bool = atomic_fetch_op.match(name)
        is_atomic_compare_exchange: bool = name == "atomic_compare_exchange"

        if is_atomic_fetch_op or is_atomic_compare_exchange:
            if is_atomic_fetch_op and len(args) != 3:
                self.error(node, "atomic_fetch_op functions take exactly 3 arguments")
            if is_atomic_compare_exchange and len(args) != 4:
                self.error(node, "atomic_compare_exchange takes exactly 4 arguments")

            # convert indices
            args[0] = cppast.CallExpr(args[0], args[1].exprs)
            del args[1]

            # if not isinstance(args[0], cppast.CallExpr):
            #     self.error(
            #         node, "atomic_fetch_op functions only support views")

            # atomic_fetch_* operations need to have an address as
            # their first argument
            args[0] = cppast.UnaryOperator(args[0], cppast.BinaryOperatorKind.AddrOf)
            return cppast.CallExpr(function, args)

        return super().visit_Call(node)

    def visit_arguments(self, node: ast.arguments) -> None:
        for arg in node.args:
            if arg.arg == "self":
                continue

            declref = cppast.DeclRefExpr(arg.arg)
            if declref in self.views:
                # Do not overwrite the existing type to keep its
                # template arguments. Here we are assuming that the
                # user used the same names for both kernel and kokkos
                # function arguments.
                continue

            fused_arg: re.Pattern = re.compile(f"fused_.*_[0-9]+")
            if fused_arg.match(arg.arg):
                original_view: str = arg.arg.split("_")[1]
                original = cppast.DeclRefExpr(original_view)

                if original in self.views:
                    self.views[declref] = self.views[cppast.DeclRefExpr(original_view)]
                    continue

            decltype: Optional[cppast.Type] = visitors_util.get_type(arg.annotation, self.pk_import)
            if isinstance(decltype, cppast.ClassType) and decltype.typename.startswith("View"):
                self.views[declref] = decltype

        return super().visit_arguments(node)


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
