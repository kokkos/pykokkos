import ast
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.interface import TeamMember

from . import visitors_util
from .pykokkos_visitor import PyKokkosVisitor


class WorkunitVisitor(PyKokkosVisitor):
    def __init__(
        self, env, src, views: Dict[cppast.DeclRefExpr, cppast.Type],
        work_units: Dict[str, ast.FunctionDef], fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
        kokkos_functions: Dict[str, ast.FunctionDef], dependency_methods: Dict[str, List[str]],
        pk_import: str, restrict_views: Set[str], debug=False
    ):
        self.has_rand_call: bool = False
        super().__init__(env, src, views, work_units, fields, kokkos_functions, dependency_methods, pk_import, restrict_views, debug)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Union[str, Tuple[str, cppast.MethodDecl]]:
        if self.is_nested_call(node):
            params: List[cppast.ParmVarDecl] = [a for a in self.visit(node.args)]
            body = cppast.CompoundStmt([self.visit(b) for b in node.body])

            workunit = cppast.LambdaExpr("[&]", params, body)
            self.nested_work_units[node.name] = workunit

            return ""

        else:
            operation: Optional[str] = self.get_operation_type(node)
            if operation is None:
                self.error(node.args, "Incorrect types in workunit definition")

            tag_type = cppast.ClassType(f"const {node.name}_tag")
            tag_type.is_reference = True
            tag = cppast.ParmVarDecl(tag_type, cppast.DeclRefExpr(""))

            params: List[cppast.ParmVarDecl] = [tag]
            params.extend(self.visit(node.args))

            body = cppast.CompoundStmt([self.visit(b) for b in node.body])
            attributes: str = "KOKKOS_FUNCTION"
            decltype = cppast.ClassType("void")
            declname: str = "operator()"

            method = cppast.MethodDecl(attributes, decltype, declname, params, body)
            method.is_const = True

            return (operation, method)

    def get_operation_type(self, node: ast.FunctionDef) -> Optional[str]:
        """
        Get the type of the operation ("for", "reduce", or "scan") of a workunit

        :param node: the workunit definition
        :returns: the name of the operation
        """

        args: List[ast.arg] = node.args.args
        last_arg: ast.arg = args[0]

        # Find the last argument in the workunit function definition that is not
        # a view or a field. This is important as this argument could be the thread ID,
        # the accumulator, or a boolean, which would help determine what the operation
        # is (for, reduce, or scan)
        for arg in args:
            arg_name = cppast.DeclRefExpr(arg.arg)
            if arg_name in self.views or arg_name in self.fields:
                break
            last_arg = arg

        annotation = last_arg.annotation

        if isinstance(annotation, ast.Name):
            name: str = visitors_util.get_node_name(annotation)
            if name == "bool":
                return "scan"
            if name == "int":
                return "for"

        if isinstance(annotation, ast.Attribute):
            name: str = visitors_util.get_node_name(annotation)
            if name == "TeamMember":
                return "for"

        if isinstance(annotation, ast.Subscript):
            name: str = visitors_util.get_node_name(annotation.value)
            if name == "Acc":
                return "reduce"

        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> cppast.Stmt:
        if isinstance(node.value, ast.Call):
            decltype: cppast.Type = visitors_util.get_type(node.annotation, self.pk_import)
            if decltype is None:
                self.error(node, "Type not supported")
            declname: cppast.DeclRefExpr = self.visit(node.target)
            function_name: str = visitors_util.get_node_name(node.value.func)

            # Call to a TeamMember method
            if function_name in dir(TeamMember):
                vardecl = cppast.VarDecl(
                    decltype, declname, self.visit(node.value))
                return cppast.DeclStmt(vardecl)

            # Nested parallelism
            if function_name in ("parallel_reduce", "parallel_scan"):
                args: List[cppast.Expr] = [
                    self.visit(a) for a in node.value.args]

                initial_value: cppast.Expr
                if len(args) == 3:
                    initial_value = args[2]
                else:
                    initial_value = cppast.IntegerLiteral(0)

                vardecl = cppast.VarDecl(decltype, declname, initial_value)
                declstmt = cppast.DeclStmt(vardecl)

                work_unit: str = args[1].declname
                function = cppast.DeclRefExpr(f"Kokkos::{function_name}")

                call: cppast.CallExpr
                if work_unit in self.nested_work_units:
                    call = cppast.CallExpr(function, [args[0], self.nested_work_units[work_unit], declname])
                else:
                    call = cppast.CallExpr(function, [args[0], f"pk_id_{work_unit}", declname])

                callstmt = cppast.CallStmt(call)

                return cppast.CompoundStmt([declstmt, callstmt])

            if function_name.startswith("ScratchView"):
                cpp_view_type: str = self.get_scratch_view_type(node.annotation)
                py_view_type: str = node.annotation.value.attr
                rank = int(re.search(r'\d+', py_view_type).group())

                typeref = cppast.ClassType(cpp_view_type)
                args: List[cppast.Expr] = [self.visit(a) for a in node.value.args]
                constructor = cppast.ConstructExpr(declname, args)

                view_decl = cppast.VarDecl(typeref, constructor, None)
                self.views[declname] = None

                return cppast.DeclStmt(view_decl)

        return super().visit_AnnAssign(node)

    def visit_arguments(self, node: ast.arguments) -> List[cppast.ParmVarDecl]:
        args: List[ast.arg] = node.args

        if len(args) == 0:
            return []

        # If the first argument does not have an annotation, then this is a nested workunit
        is_nested: bool = True if args[0].annotation is not None else False

        self_arg: ast.arg = args[0]
        if not is_nested and self_arg.arg != "self":
            self.error(args[0], "First argument has to be \"self\"")

        # Skip self argument
        if not is_nested:
            args = args[1:]

        cpp_args: List[cppast.ParmVarDecl] = []

        # Visit all tid args, could be more than one for MDRangePolicies.
        # Stop when the accumulator is reached or there are no more tid args.

        acc_arg_index = 0
        for i, a in enumerate(args):
            is_acc: bool = isinstance(a.annotation, ast.Subscript)
            if is_acc:
                acc_arg_index = i
                break

            arg_name = cppast.DeclRefExpr(a.arg)
            if arg_name in self.views or arg_name in self.fields:
                break

            cpp_args.append(self.visit_arg(a))

        acc_arg: ast.arg
        last_arg: ast.arg

        operation: str = self.get_operation_type(node.parent)
        if operation == "scan":
            last_arg: ast.arg = args[acc_arg_index + 1]
            acc_arg = args[acc_arg_index]
        if operation == "reduce":
            acc_arg = args[acc_arg_index]

        if operation in ("scan", "reduce"):
            acc: cppast.ParmVarDecl = self.visit_arg(acc_arg)
            acc.decltype.is_reference = True
            cpp_args.append(acc)

        if operation == "scan":
            last: cppast.ParmVarDecl = self.visit_arg(last_arg)
            cpp_args.append(last)

        return cpp_args

    def visit_arg(self, node: ast.arg) -> cppast.ParmVarDecl:
        if node.annotation is None:
            self.error(node, "Missing type annotation")

        decltype: cppast.Type = visitors_util.get_type(node.annotation, self.pk_import)
        if decltype is None:
            self.error(node, "Type not supported")

        # If argument is pk.TeamMember (hierarchical parallelism)
        is_hierachical: bool = isinstance(node.annotation, ast.Attribute)

        if is_hierachical:
            try:
                decltype.typename = f"const {decltype.typename}"
            except AttributeError:
                decltype._type = f"const {decltype.typename.value}"

            decltype.is_reference = True

        declname = cppast.DeclRefExpr(node.arg)
        arg = cppast.ParmVarDecl(decltype, declname)

        return arg

    def visit_Call(self, node: ast.Call) -> cppast.CallExpr:
        name: str = visitors_util.get_node_name(node.func)
        args: List[cppast.Expr] = [self.visit(a) for a in node.args]

        # Call to a TeamMember method
        if name in dir(TeamMember):
            team_member: str = visitors_util.get_node_name(node.func.value)
            call = cppast.MemberCallExpr(cppast.DeclRefExpr(
                team_member), cppast.DeclRefExpr(name), args)

            return call

        # Call to view.extent()
        if name == "extent":
            if len(args) != 1:
                self.error(node, "the extent method takes exactly 1 argument")

            view: str = visitors_util.get_node_name(node.func.value)
            call = cppast.MemberCallExpr(cppast.DeclRefExpr(
                view), cppast.DeclRefExpr(name), [args[0]])

            return call

        function = cppast.DeclRefExpr(f"Kokkos::{name}")
        if name in ("TeamThreadRange", "ThreadVectorRange", "TeamThreadMDRange"):
            return cppast.CallExpr(function, args)

        if name in ("parallel_for", "single"):
            work_unit: str = args[1].declname
            if work_unit in self.nested_work_units:
                return cppast.CallExpr(function, [args[0], self.nested_work_units[work_unit]])
            else:
                return cppast.CallExpr(function, [args[0], f"pk_id_{work_unit}"])

        atomic_fetch_op: re.Pattern = re.compile("atomic_*")
        is_atomic_fetch_op: bool = atomic_fetch_op.match(name)
        is_atomic_increment: bool = name == "atomic_increment"
        is_atomic_compare_exchange: bool = name == "atomic_compare_exchange"

        if is_atomic_fetch_op or is_atomic_compare_exchange or is_atomic_increment:
            if is_atomic_increment and len(args) != 2:
                self.error(node, "is_atomic_increment takes exactly 2 arguments")
            if not is_atomic_increment and is_atomic_fetch_op and len(args) != 3:
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

        if name == "rand":
            if len(args) != 1:
                self.error(node, "pk.rand() accepts only one argument, the datatype")

            self.has_rand_call = True

            rand_type = cppast.ClassType("Kokkos::rand")
            pool_type = cppast.ClassType("Kokkos::Random_XorShift64_Pool<>::generator_type")
            rand_type.add_template_param(pool_type)
            rand_type.add_template_param(visitors_util.get_type(node.args[0], self.pk_import))

            rand_call = cppast.MemberCallExpr(rand_type, cppast.DeclRefExpr("draw"), [cppast.DeclRefExpr(Keywords.RandPoolState.value)])
            rand_call.is_static = True

            return rand_call

        if name in {"cyl_bessel_j0", "cyl_bessel_j1"}:
            if len(args) != 1:
                self.error(node, "pk.cyl_bessel_j0/j1 accepts only one argument")

            s = cppast.Serializer()
            arg_str = s.serialize(args[0])
            math_call = cppast.CallExpr(cppast.DeclRefExpr(f"Kokkos::Experimental::{name}<Kokkos::complex<decltype({arg_str})>, double, int>"), args)
            real_number_call = cppast.MemberCallExpr(math_call, cppast.DeclRefExpr("real"), [])

            return real_number_call

        return super().visit_Call(node)

    def is_nested_call(self, node: ast.FunctionDef) -> bool:
        while (hasattr(node, "parent")):
            node = node.parent
            if isinstance(node, ast.FunctionDef):
                return True

        return False
