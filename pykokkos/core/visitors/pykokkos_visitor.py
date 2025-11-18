import ast
from ast import FunctionDef, AST
import os
import re
import sys
from typing import List, Dict, Optional, Set, Union

from pykokkos.core import cppast
from pykokkos.core.optimizations import adjust_kokkos_function_call, get_restrict_ptr_name, index_restrict_view
from pykokkos.interface import View

from . import visitors_util


class PyKokkosVisitor(ast.NodeVisitor):
    def __init__(
            self, env, src,
            views: Dict[cppast.DeclRefExpr, cppast.Type],
            work_units: Dict[str, FunctionDef],
            fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
            kokkos_functions: Dict[str, FunctionDef],
            dependency_methods: Dict[str, List[str]],
            pk_import: str,
            restrict_views: Set[str],
            debug=False
    ):
        self.env = env
        self.src = src
        self.views = views
        self.work_units = work_units
        self.fields = fields
        self.kokkos_functions = kokkos_functions
        self.dependency_methods = dependency_methods
        self.pk_import = pk_import
        self.restrict_views = restrict_views
        self.debug = debug

        # Map from subview to parent view
        self.subviews: Dict[str, str] = {}
        self.pkmain_views: Dict[str, View] = {}
        self.lists: List[str] = []

        # Maps from nested work unit name to definition
        self.nested_work_units: Dict[str, cppast.LambdaExpr] = {}

    def visit_arguments(self, node: ast.arguments) -> List[cppast.ParmVarDecl]:
        args: List[cppast.ParmVarDecl] = [self.visit(a) for a in node.args if a.arg != "self"]

        return args

    def visit_arg(self, node: ast.arg) -> Union[cppast.ParmVarDecl, str]:
        if node.arg == "self":
            return ""

        if node.annotation is None:
            self.error(node, "Missing type annotation")

        decltype: cppast.Type = visitors_util.get_type(node.annotation, self.pk_import)
        if decltype is None:
            self.error(node, "Type not supported")

        if self.is_dependency(decltype):
            decltype.is_reference = True

        declname = cppast.DeclRefExpr(node.arg)
        if isinstance(decltype, cppast.ClassType) and decltype.typename.startswith("View"):
            decltype = self.views[declname]
            decltype = visitors_util.cpp_view_type(decltype)

        arg = cppast.ParmVarDecl(decltype, declname)

        return arg

    def visit_Assign(self, node: ast.Assign) -> Union[cppast.AssignOperator, cppast.DeclStmt]:
        for target in node.targets:
            if (
                # TODO: check if target.value.id is in scope
                (isinstance(target, ast.Attribute) and target.value.id == "self")
                and type(target) not in {ast.Name, ast.Subscript}
            ):
                self.error(
                    target, "Only local variables and views supported for assignment",
                )

        # handle subview
        if isinstance(node.value, ast.Subscript):
            if (sys.version_info.minor <= 8 and not isinstance(node.value.slice, ast.Index)) or (
                sys.version_info.minor > 8 and isinstance(node.value.slice, ast.Tuple)):

                view = node.value.value
                if isinstance(view, ast.Attribute) and view.value.id == "self":
                # reference view through self
                    attr = node.value.value
                    view_name = view.attr
                elif isinstance(view, ast.Name):
                # reference views through params (standalone)
                    view_name = view.id
                else:
                    self.error(view, "View not recognized")

                if cppast.DeclRefExpr(view_name) in self.views:
                    return self.generate_subview(node, view_name)
                else:
                    self.error(node, "Can only take subview of views")

        targets: List[cppast.DeclRefExpr] = [
            self.visit(t) for t in node.targets]
        value: cppast.Expr = self.visit(node.value)
        op: cppast.BinaryOperatorKind = cppast.BinaryOperatorKind.Assign
        assign = cppast.AssignOperator(targets, value, op)

        return assign

    def generate_subview(self, node: ast.Assign, view_name: str) -> cppast.DeclStmt:
        subview_args: List[cppast.Expr] = [cppast.DeclRefExpr(view_name)]

        slice_node = node.value
        for dim in slice_node.slice.elts:
            # In Python >= 3.9, ast.Index is deprecated
            # (see # https://docs.python.org/3/whatsnew/3.9.html)
            # Instead of ast.Index, value will be used directly
            check_type: type
            if sys.version_info.minor <= 8:
                check_type: type = ast.Index
            else:
                check_type: type = ast.Name

            if isinstance(dim, check_type):
                subview_args.append(self.visit(dim))
            else:
                if dim.lower is None and dim.upper is None: 
                    subview_args.append(cppast.DeclRefExpr("Kokkos::ALL"))
                elif dim.lower is not None and dim.upper is not None:
                    make_pair = cppast.CallExpr("std::make_pair",
                            [self.visit(dim.lower), self.visit(dim.upper)])
                    subview_args.append(make_pair)
                else:
                    self.error(
                            slice_node, "Partial slice not supported, use [n:m] or [:]")

        if len(node.targets) > 1:
            self.error(node, "Multiple declarations of subview not supported")

        auto = cppast.ClassType("auto")
        target = node.targets[0]
        target_ref = cppast.DeclRefExpr(target.id)
        if target_ref in self.views:
            self.error(
                node, "Redeclaration of existing subview")
        else:
            self.views[target_ref] = None
            self.subviews[target_ref.declname] = view_name

        call = cppast.CallExpr("Kokkos::subview", subview_args)
        decl = cppast.DeclStmt(cppast.VarDecl(auto, self.visit(target), call))

        return decl 

    def visit_AugAssign(self, node: ast.AugAssign) -> cppast.CompoundAssignOperator:
        variable: cppast.DeclRefExpr = self.visit(node.target)
        value: cppast.Expr = self.visit(node.value)
        op: cppast.BinaryOperatorKind = self.visit(node.op)
        augassign = cppast.CompoundAssignOperator(variable, value, op)

        return augassign

    def visit_Attribute(self, node: ast.Attribute) -> cppast.DeclRefExpr:
        if (
            not isinstance(node.value, ast.Name)
            # TODO: implement proper scope checking
            # or node.value.id not in ("self", "math")
        ):
            self.error(node, "Unrecognized attribute")

        # Math constant
        if node.value.id == "math":
            try:
                constant: str = visitors_util.get_math_constant_str(node.attr)
            except NotImplementedError:
                self.error(node, "Unrecognized math constant")
            return cppast.DeclRefExpr(constant)

        if node.value.id == "self":
            name: str = node.attr
            
            if cppast.DeclRefExpr(name) in self.views:
                return name

            # if name not in self.env:
            #     self.error(node, "Couldn't find variable")

            field = cppast.MemberExpr(cppast.DeclRefExpr("this"), name)
            field.is_pointer = True

            return field

        return cppast.DeclRefExpr(f"{node.value.id}.{node.attr}")

    def visit_Constant(self, node: ast.Constant) -> Union[cppast.BoolLiteral,
                                                          cppast.FloatingLiteral,
                                                          cppast.IntegerLiteral,
                                                          cppast.StringLiteral]:

        if isinstance(node.value, bool):
            return cppast.BoolLiteral(node.value)

        if isinstance(node.value, float):
            return cppast.FloatingLiteral(node.value)

        if isinstance(node.value, int):
            return cppast.IntegerLiteral(node.value)

        if isinstance(node.value, str):
            return cppast.StringLiteral(node.value)

        self.error(node, "Unsupported Constant")

    def visit_Subscript(self, node: ast.Subscript) -> Union[cppast.ArraySubscriptExpr, cppast.CallExpr]:
        current_node: ast.Subscript = node
        slices: List = []
        dim: int = 0

        while isinstance(current_node, ast.Subscript):
            index = current_node.slice

            if sys.version_info.minor <= 8:
                # In Python >= 3.9, ast.Index is deprecated
                # (see # https://docs.python.org/3/whatsnew/3.9.html)
                # Instead of ast.Index, value will be used directly

                if not isinstance(index, ast.Index):
                    self.error(
                        current_node, "Slices not supported, use simple indices")

            slices.insert(0, index)
            current_node = current_node.value
            dim += 1

        name: str = visitors_util.get_node_name(current_node)
        ref = cppast.DeclRefExpr(name)

        if ref not in self.views and name not in self.lists:
            self.error(current_node, "Unknown view or list")

        dim_map: List = [cppast.ClassType("View1D"),
                         cppast.ClassType("View2D"),
                         cppast.ClassType("View3D"),
                         cppast.ClassType("View4D"),
                         cppast.ClassType("View5D"),
                         cppast.ClassType("View6D"),
                         cppast.ClassType("View7D"),
                         cppast.ClassType("View8D")]

        if name in self.lists:
            indices: List[cppast.Expr] = [self.visit(s) for s in slices]
            subscript = cppast.ArraySubscriptExpr(ref, indices)

            return subscript

        if (
            ref in self.views
            and (
                self.views[ref] is None  # For views added in @pk.main
                or self.views[ref].typename == dim_map[dim - 1].typename
            )
        ):
            args: List[cppast.Expr] = [self.visit(s) for s in slices]
            # Account for fused views
            r = re.search("fused_(.*)_[0-9]*", ref.declname)
            unfused_name: str = r.group(1) if r else ref.declname

            if "PK_RESTRICT" in os.environ and unfused_name in self.restrict_views or name in self.restrict_views:
                if unfused_name in self.restrict_views:
                    v = self.restrict_views[unfused_name]
                else:
                    v = self.restrict_views[name]
                subscript = index_restrict_view(ref, args, v)
            else:
                subscript = cppast.CallExpr(ref, args)

            return subscript

        self.error(node, f"'{name}' is not a View{dim}D")

    def visit_BinOp(self, node: ast.BinOp) -> Union[cppast.BinaryOperator, cppast.CallExpr, cppast.CastExpr]:
        lhs = cppast.ParenExpr(self.visit(node.left))
        rhs = cppast.ParenExpr(self.visit(node.right))

        if isinstance(node.op, ast.Pow):
            return cppast.CallExpr(cppast.DeclRefExpr("pow"), [lhs, rhs])

        op: cppast.BinaryOperatorKind = self.visit(node.op)

        if isinstance(node.op, ast.Div):
            # Cast one of the operands to a double
            lhs = cppast.CastExpr(
                cppast.PrimitiveType(cppast.BuiltinType.DOUBLE), lhs)

        binop = cppast.BinaryOperator(lhs, rhs, op)

        if isinstance(node.op, ast.FloorDiv):
            # Cast the result to an int
            cast = cppast.CastExpr(
                cppast.PrimitiveType(cppast.BuiltinType.INT), binop)
            return cast

        return binop

    def visit_UnaryOp(self, node: ast.UnaryOp) -> cppast.UnaryOperator:
        op: cppast.BinaryOperatorKind = self.visit(node.op)
        operand: cppast.Expr = self.visit(node.operand)
        unop = cppast.UnaryOperator(operand, op)

        return unop

    def visit_Compare(self, node: ast.Compare) -> cppast.BinaryOperator:
        if len(node.comparators) > 1:
            # TODO: possibly break out into multiple comparisons
            self.error(
                node.comparators[1],
                "Chaining comparisons not supported for translation",
            )

        left: cppast.Expr = self.visit(node.left)
        op: cppast.BinaryOperatorKind = self.visit(node.ops[0])
        comparators: cppast.Expr = self.visit(node.comparators[0])
        compare = cppast.BinaryOperator(left, comparators, op)

        return compare

    def visit_BoolOp(self, node: ast.BoolOp) -> cppast.BoolOperator:
        exprs: List[cppast.Expr] = [self.visit(v) for v in node.values]
        op: cppast.BinaryOperatorKind = self.visit(node.op)
        boolop = cppast.BoolOperator(exprs, op)

        return boolop

    def visit_Name(self, node: ast.Name) -> cppast.DeclRefExpr:
        return cppast.DeclRefExpr(node.id)

    def visit_Index(self, node: ast.Index) -> cppast.Expr:
        return self.visit(node.value)

    def visit_For(self, node: ast.For) -> cppast.ForStmt:
        if not isinstance(node.target, ast.Name):
            self.error(node.target, "Must use single loop variable")

        if node.orelse:
            self.error(node.orelse, "Else clause not supported for translation")

        if (
            not isinstance(node.iter, ast.Call)
            or node.iter.func.id != "range"
        ):
            # TODO: support other iterators?
            self.error(
                node.iter, "Only range() iterator is supported for translation")

        index: cppast.DeclRefExpr = self.visit(node.target)
        start: cppast.Expr
        end: cppast.Expr
        step: cppast.Expr = cppast.IntegerLiteral(1)
        op = cppast.BinaryOperatorKind.LT

        args = node.iter.args
        if len(args) == 1:
            start = cppast.IntegerLiteral(0)
            end = self.visit(args[0])

        else:
            start = self.visit(args[0])
            end = self.visit(args[1])

            if len(args) == 3:
                step = self.visit(args[2])

                # Negative step sizes are only handled correctly if they're
                # written with a preceeding minus sign
                if (
                    isinstance(args[2], ast.UnaryOp)
                    and isinstance(args[2].op, ast.USub)
                ):
                    op = cppast.BinaryOperatorKind.GT

        body = cppast.CompoundStmt([self.visit(b) for b in node.body])

        init = cppast.DeclStmt(cppast.VarDecl(
            cppast.PrimitiveType(cppast.BuiltinType.INT), index, start))
        condition = cppast.BinaryOperator(index, end, op)
        increment = cppast.BinaryOperator(
            index, step, cppast.BinaryOperatorKind.AddAssign)
        forstmt = cppast.ForStmt(init, condition, increment, body)

        return forstmt

    def visit_If(self, node: ast.If) -> cppast.IfStmt:
        condition: cppast.Expr = self.visit(node.test)
        then_body = cppast.CompoundStmt([self.visit(b) for b in node.body])
        else_body = cppast.CompoundStmt([self.visit(b) for b in node.orelse]) if node.orelse else None
        ifstmt = cppast.IfStmt(condition, then_body, else_body)

        return ifstmt

    def visit_While(self, node: ast.While) -> cppast.WhileStmt:
        if node.orelse:
            self.error(node.orelse, "Else clause not supported for translation")

        condition: cppast.Expr = self.visit(node.test)
        body = cppast.CompoundStmt([self.visit(b) for b in node.body])
        whilestmt = cppast.WhileStmt(condition, body)

        return whilestmt

    def visit_Call(self, node: ast.Call) -> cppast.CallExpr:
        name: str = visitors_util.get_node_name(node.func)

        if name == "print":
            self.error(
                node.func, "Function not supported, did you mean pykokkos.printf()?"
            )
        elif name in ["PerTeam", "PerThread", "fence"]:
            name = "Kokkos::" + name

        function = cppast.DeclRefExpr(name)
        args: List[cppast.Expr] = [self.visit(a) for a in node.args]

        if visitors_util.is_math_function(name) or name in ["printf", "abs", "Kokkos::PerTeam", "Kokkos::PerThread", "Kokkos::fence"]:
            return cppast.CallExpr(function, args)

        if function in self.kokkos_functions:
            if "PK_RESTRICT" in os.environ:
                return adjust_kokkos_function_call(function, args, self.restrict_views, self.views)
            else:
                return cppast.CallExpr(function, args)

        # Call to a dependency's constructor
        if function.declname in visitors_util.allowed_types:
            name = visitors_util.allowed_types[name]
            function = cppast.DeclRefExpr(name)
            return cppast.CallExpr(function, args)

        # Call to a dependency's method
        for key, value in self.dependency_methods.items():
            if function in value:
                object_name: cppast.DeclRefExpr = self.visit(node.func.value)
                return cppast.MemberCallExpr(object_name, function, args)

        self.error(node.func, f"Function {name} not supported for translation")

    def visit_Return(self, node: ast.Return) -> cppast.ReturnStmt:
        parent_function: FunctionDef = self.get_parent_function(node)
        if parent_function is None:
            self.error(node, "Cannot return outside of function")

        if node.value:
            if cppast.DeclRefExpr(parent_function.name) in self.kokkos_functions:
                return cppast.ReturnStmt(self.visit(node.value))
            else:
                self.error(
                    node.value, "Cannot return value from translated function")

        return cppast.ReturnStmt()

    def visit_Break(self, node: ast.Break) -> cppast.BreakStmt:
        return cppast.BreakStmt()

    def visit_Continue(self, node: ast.Continue) -> cppast.ContinueStmt:
        return cppast.ContinueStmt()

    def visit_Pass(self, node: ast.Pass) -> cppast.EmptyStmt:
        return cppast.EmptyStmt()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> cppast.DeclStmt:
        if not isinstance(node.target, ast.Name):
            self.generic_error(node)

        decltype: cppast.Type = visitors_util.get_type(node.annotation, self.pk_import)
        if decltype is None:
            self.error(node, "Type not supported")
        target: cppast.DeclRefExpr = self.visit(node.target)
        value: cppast.Expr = self.visit(node.value)

        # Add the name of the list to self.lists
        # and add the length of the list to its name
        if isinstance(node.annotation, ast.Subscript):
            self.lists.append(target.declname)
            target.add_length(len(node.value.elts))

        annassign = cppast.DeclStmt(cppast.VarDecl(decltype, target, value))

        return annassign

    def visit_NamedExpr(self, node: ast.NamedExpr) -> str:
        self.generic_error(node)  # TODO: check if needed

    def visit_Assert(self, node: ast.Assert) -> str:
        self.generic_error(node)  # TODO: test if possible

    def visit_IfExp(self, node: ast.IfExp) -> str:
        self.generic_error(node)  # TODO: test if possible (maybe ternary)

    def visit_Tuple(self, node: ast.Tuple) -> str:
        self.generic_error(node)  # TODO: test if possible

    def visit_Expr(self, node: ast.Expr) -> cppast.CallStmt:
        # This is needed for docstrings
        if isinstance(node.value, ast.Constant):
            return cppast.EmptyStmt()

        if not isinstance(node.value, ast.Call):
            self.error(
                node, "Only function calls are allowed as standalone statements")

        call: cppast.CallExpr = self.visit(node.value)

        return cppast.CallStmt(call)

    def visit_Delete(self, node: ast.Delete) -> str:
        self.generic_error(node)

    def visit_Global(self, node: ast.Global) -> str:
        self.generic_error(node)

    # ignore nonlocal
    def visit_Nonlocal(self, node: ast.Nonlocal) -> str:
        return ""

    def visit_Yield(self, node: ast.Yield) -> str:
        self.generic_error(node)

    def visit_With(self, node: ast.With) -> str:
        self.generic_error(node)

    def visit_List(self, node: ast.List) -> cppast.InitListExpr:
        exprs: List[cppast.Expr] = [self.visit(e) for e in node.elts]
        initlist = cppast.InitListExpr(exprs)

        return initlist

    def visit_ListComp(self, node: ast.ListComp) -> str:
        self.generic_error(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> str:
        self.generic_error(node)

    def visit_Slice(self, node: ast.slice) -> str:
        self.generic_error(node)

    def visit_UAdd(self, node: ast.UAdd) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Plus

    def visit_USub(self, node: ast.USub) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Minus

    def visit_Not(self, node: ast.Not) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.LNot

    def visit_Invert(self, node: ast.Invert) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Not

    def visit_Add(self, node: ast.Add) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Add

    def visit_Sub(self, node: ast.Sub) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Sub

    def visit_Mult(self, node: ast.Mult) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Mul

    def visit_Div(self, node: ast.Div) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Div

    def visit_FloorDiv(self, node: ast.FloorDiv) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Div

    def visit_Mod(self, node: ast.Mod) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Rem

    def visit_Pow(self, node: ast.Pow) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    def visit_LShift(self, node: ast.LShift) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Shl

    def visit_RShift(self, node: ast.RShift) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Shr

    def visit_BitOr(self, node: ast.BitOr) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Or

    def visit_BitXor(self, node: ast.BitXor) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.Xor

    def visit_BitAnd(self, node: ast.BitAnd) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.And

    def visit_MatMult(self, node: ast.MatMult) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    def visit_And(self, node: ast.And) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.LAnd

    def visit_Or(self, node: ast.Or) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.LOr

    def visit_Eq(self, node: ast.Eq) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.EQ

    def visit_NotEq(self, node: ast.NotEq) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.NE

    def visit_Lt(self, node: ast.Lt) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.LT

    def visit_LtE(self, node: ast.LtE) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.LE

    def visit_Gt(self, node: ast.Gt) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.GT

    def visit_GtE(self, node: ast.GtE) -> cppast.BinaryOperatorKind:
        return cppast.BinaryOperatorKind.GE

    def visit_Is(self, node: ast.Is) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    def visit_IsNot(self, node: ast.IsNot) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    def visit_In(self, node: ast.In) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    def visit_NotIn(self, node: ast.NotIn) -> cppast.BinaryOperatorKind:
        self.generic_error(node)

    # Returns the parent function of a node
    def get_parent_function(self, node: AST) -> FunctionDef:
        while hasattr(node, "parent"):
            if isinstance(node.parent, FunctionDef):
                return node.parent
            node = node.parent

        return None

    def is_in_loop(self, node: AST) -> bool:
        while hasattr(node, "parent"):
            if isinstance(node.parent, ast.For) or isinstance(node.parent, ast.While):
                return True
            node = node.parent

        return False

    def is_void_function(self, node: ast.FunctionDef) -> bool:
        if (
            node.returns is None
            or (
                isinstance(node.returns, ast.Constant)
                and node.returns.value is None
            )
        ):
            return True

        return False

    def is_dependency(self, input_type: cppast.Type) -> bool:
        if isinstance(input_type, cppast.PrimitiveType):
            return False

        classref = cppast.DeclRefExpr(input_type.typename)
        if classref in self.dependency_methods:
            return True

        return False

    def get_scratch_view_type(self, view_type: ast.Subscript) -> Optional[str]:
        """
        Get the cppast representation of a scratch view

        :param view_type: the subscripted type of the view
        :returns: the string representation of the view if it is valid
        """

        is_valid: bool = False

        if isinstance(view_type, ast.Subscript) and isinstance(view_type.value, ast.Attribute):
            attr: ast.Attribute = view_type.value

            if attr.value.id == self.pk_import and attr.attr.startswith("ScratchView"):
                view_dtype: cppast.PrimitiveType = visitors_util.get_type(view_type.slice, self.pk_import)
                view_type: str = attr.attr
                is_valid = True

        if not is_valid:
            return None

        scratch_view = cppast.ClassType(view_type)
        scratch_view.add_template_param(view_dtype)

        cpp_view_type: str = visitors_util.cpp_view_type(scratch_view)

        return cpp_view_type

    def error(self, node, message):
        visitors_util.error(self.src, self.debug, node, message)

    def generic_error(self, node):
        self.error(node, "Not supported for translation")
