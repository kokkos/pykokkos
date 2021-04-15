import ast
from typing import Dict, List, Optional, Union

from pykokkos.core import cppast

from . import visitors_util
from .pykokkos_visitor import PyKokkosVisitor


class ClasstypeVisitor(PyKokkosVisitor):
    def visit_ClassDef(self, node: ast.ClassDef) -> cppast.RecordDecl:
        name: str = node.name
        # Add class as allowed type
        visitors_util.allowed_types[name] = name

        member_variables: Dict[cppast.DeclRefExpr, cppast.Type] = self.get_member_variables(node)
        decls: List[cppast.DeclStmt] = []

        if len(member_variables) == 0:
            self.error(node, "Missing constructor or no member variables detected")

        for n, t in member_variables.items():
            decls.append(cppast.DeclStmt(cppast.FieldDecl(t, n)))

        for b in node.body:
            decls.append(self.visit(b))

        if not self.has_default_constructor(node):
            decls.append(cppast.ConstructorDecl("KOKKOS_FUNCTION", name, [], None))

        classdef = cppast.RecordDecl(cppast.ClassType(name), decls)
        classdef.is_definition = True

        return classdef

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[cppast.ConstructorDecl, cppast.MethodDecl]:
        name: str = node.name
        return_type: Optional[cppast.ClassType]

        if name == "__init__":
            name = node.parent.name
            return_type = None
        elif self.is_void_function(node):
            return_type = cppast.ClassType("void")
        else:
            return_type = visitors_util.get_type(node.returns, self.pk_import)

        if len(node.args.args) == 0 or node.args.args[0].arg != "self":
            self.error(node, "Static functions are not supported")

        params: List[cppast.ParmVarDecl] = self.visit(node.args)
        body = cppast.CompoundStmt([self.visit(b) for b in node.body])
        attributes: str = "KOKKOS_FUNCTION"

        if return_type is None:
            return cppast.ConstructorDecl(attributes, name, params, body)
        else:
            return cppast.MethodDecl(attributes, return_type, name, params, body)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Union[cppast.AssignOperator, cppast.Stmt]:
        # If local variable
        if isinstance(node.target, ast.Name):
            return super().visit_AnnAssign(node)

        target: cppast.DeclRefExpr = self.visit(node.target)
        value: cppast.Expr = self.visit(node.value)

        return cppast.AssignOperator([target], value, cppast.BinaryOperatorKind.Assign)

    def visit_Assign(self, node: ast.Assign) -> cppast.AssignOperator:
        targets: List[cppast.DeclRefExpr] = [
            self.visit(t) for t in node.targets]
        value: cppast.Expr = self.visit(node.value)
        op: cppast.BinaryOperatorKind = cppast.BinaryOperatorKind.Assign
        assign = cppast.AssignOperator(targets, value, op)

        return assign

    def visit_Return(self, node: ast.Return) -> cppast.ReturnStmt:
        function: ast.FunctionDef = self.get_parent_function(node)
        if self.is_void_function(function) and node.value is not None:
            self.error(node, "Cannot return a value from a void function")

        if node.value is None:
            return cppast.ReturnStmt()

        return cppast.ReturnStmt(self.visit(node.value))

    # Returns a dictionary mapping from member variable name to type
    def get_member_variables(self, node: ast.ClassDef) -> Dict[cppast.DeclRefExpr, cppast.Type]:
        member_variables: Dict[cppast.DeclRefExpr, cppast.Type] = {}
        constructor: Optional[ast.FunctionDef] = None

        for function in node.body:
            if isinstance(function, ast.FunctionDef):
                if function.name == "__init__":
                    constructor = function
                    break

        if constructor is None:
            self.error(node, "Missing constructor")

        for b in constructor.body:
            if isinstance(b, ast.AnnAssign):
                if b.target.value.id == "self":
                    declref = cppast.DeclRefExpr(visitors_util.get_node_name(b.target))
                    typename: cppast.Type = visitors_util.get_type(b.annotation, self.pk_import)

                    if typename is None:
                        self.error(b, "Type not supported")

                    serializer = cppast.Serializer()
                    member_variables[declref] = serializer.serialize(typename)

        return member_variables

    def has_default_constructor(self, node: ast.ClassDef) -> bool:
        for b in node.body:
            if (isinstance(b, ast.FunctionDef)):
                if b.name == "__init__":
                    if len(b.args.args) == 1:
                        return True

        return False
