import ast
from typing import List, Dict, Optional, Set, Union
from ast import FunctionDef

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.interface import BinOp, BinSort, View, Iterate, TeamPolicy

from . import visitors_util
from .pykokkos_visitor import PyKokkosVisitor


class KokkosMainVisitor(PyKokkosVisitor):
    def __init__(
            self, env, src,
            views: Dict[str, View],
            work_units: Dict[str, FunctionDef],
            fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
            kokkos_functions: Dict[str, FunctionDef],
            dependency_methods: Dict[str, List[str]],
            functor: str,
            pk_import: str,
            restrict_views: Set[str] = set(),
            debug=False
    ):
        super().__init__(env, src, views, work_units, fields,
                         kokkos_functions, dependency_methods,
                         pk_import, restrict_views, debug)

        self.functor: str = functor
        self.reduction_result_queue: List[str] = []
        self.timer_result_queue: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> str:
        run_body: str = ""
        serializer = cppast.Serializer()
        for statement in node.body:
            run_body += serializer.serialize(self.visit(statement))

        return run_body

    def visit_Assign(self, node: ast.Assign) -> cppast.Stmt:
        target = node.targets[0]

        if isinstance(node.value, ast.Call):
            name: str = visitors_util.get_node_name(node.value.func)

            # Create Timer object
            if name == "Timer":
                decltype = cppast.ClassType("Kokkos::Timer")
                declname = cppast.DeclRefExpr("timer")
                return cppast.DeclStmt(cppast.VarDecl(decltype, declname, None))

            # Call Timer.seconds()
            if name == "seconds":
                target_name: str = visitors_util.get_node_name(target)
                if target_name not in self.timer_result_queue:
                    self.timer_result_queue.append(target_name)

                call = cppast.CallStmt(self.visit(node.value))
                target_ref = cppast.DeclRefExpr(target_name)
                target_view_ref = cppast.DeclRefExpr(
                    f"timer_result_{target_name}")
                subscript = cppast.ArraySubscriptExpr(
                    target_view_ref, [cppast.IntegerLiteral(0)])
                assign_op = cppast.BinaryOperatorKind.Assign

                # Holds the result of the reduction temporarily
                temp_ref = cppast.DeclRefExpr("pk_acc")
                target_assign = cppast.AssignOperator(
                    [target_ref], temp_ref, assign_op)
                view_assign = cppast.AssignOperator(
                    [subscript], target_ref, assign_op)

                return cppast.CompoundStmt([call, target_assign, view_assign])

            if name in ("BinSort", "BinOp1D", "BinOp3D"):
                args: List = node.value.args
                # if not isinstance(args[0], ast.Attribute):
                #     self.error(node.value, "First argument has to be a view")

                view = cppast.DeclRefExpr(visitors_util.get_node_name(args[0]))
                if view not in self.views:
                    self.error(args[0], "Undefined view")

                view_type: cppast.ClassType = self.views[view]
                is_subview: bool = view_type is None
                if is_subview:
                    parent_view_name: str = self.subviews[view.declname]

                    # Need to remove "pk_d_" from the start of the
                    # view name to get the type of the parent
                    if parent_view_name.startswith("pk_d_"):
                        parent_view_name = parent_view_name.replace("pk_d_", "", 1)
                    parent_view = cppast.DeclRefExpr(parent_view_name)
                    view_type = self.views[parent_view]

                view_type_str: str = visitors_util.cpp_view_type(view_type)

                if name != "BinSort":
                    dimension: int = 1 if name == "BinOp1D" else 3
                    cpp_type = cppast.DeclRefExpr(
                        BinOp.get_type(dimension, view_type_str))

                    # Do not translate the first argument (view)
                    constructor = cppast.CallExpr(
                        cpp_type, [self.visit(a) for a in args[1:]])

                else:
                    bin_op_type: str = f"decltype({visitors_util.get_node_name(args[1])})"

                    binsort_args: List[cppast.DeclRefExpr] = [self.visit(a) for a in args]
                    cpp_type = cppast.DeclRefExpr(BinSort.get_type(f"decltype({binsort_args[0].declname})", bin_op_type, Keywords.DefaultExecSpace.value))
                    constructor = cppast.CallExpr(cpp_type, binsort_args)

                cpp_target: cppast.DeclRefExpr = self.visit(target)
                auto_type = cppast.ClassType("auto")

                return cppast.DeclStmt(cppast.VarDecl(auto_type, cpp_target, constructor))

            if name in ("get_bin_count", "get_bin_offsets", "get_permute_vector"):
                if not isinstance(target, ast.Attribute) or target.value.id != "self":
                    self.error(
                        node, "Views defined in pk.main must be an instance variable")

                cpp_target: str = visitors_util.get_node_name(target)
                cpp_device_target = f"pk_d_{cpp_target}"
                cpp_target_ref = cppast.DeclRefExpr(cpp_device_target)
                sorter: cppast.DeclRefExpr = self.visit(node.value.func.value)

                initial_target_ref = cppast.DeclRefExpr(
                    f"_pk_{cpp_target_ref.declname}")

                function = cppast.MemberCallExpr(
                    sorter, cppast.DeclRefExpr(name), [])

                # Add to the dict of declarations made in pk.main
                if name == "get_permute_vector":
                    # This occurs when a workload is executed multiple times
                    # Initially the view has not been defined in the workload,
                    # so it needs to be classified as a pkmain_view.
                    if cpp_target in self.views:
                        self.views[cpp_target_ref].add_template_param(
                            cppast.PrimitiveType(cppast.BuiltinType.INT))

                        return cppast.AssignOperator([cpp_target_ref], function, cppast.BinaryOperatorKind.Assign)
                        # return f"{cpp_target} = {sorter}.{name}();"

                    self.pkmain_views[cpp_target_ref] = cppast.ClassType(
                        "View1D")
                else:
                    self.pkmain_views[cpp_target_ref] = None

                auto_type = cppast.ClassType("auto")
                decl = cppast.DeclStmt(cppast.VarDecl(
                    auto_type, initial_target_ref, function))

                # resize the workload's vector to match the generated vector
                resize_call = cppast.CallStmt(
                    cppast.CallExpr(cppast.DeclRefExpr("Kokkos::resize"),
                                    [cpp_target_ref,
                                     cppast.MemberCallExpr(
                                         initial_target_ref,
                                         cppast.DeclRefExpr("extent"),
                                         [cppast.IntegerLiteral(0)])]))

                copy_call = cppast.CallStmt(cppast.CallExpr(cppast.DeclRefExpr(
                    "Kokkos::deep_copy"), [cpp_target_ref, initial_target_ref]))

                # Assign to the functor after resizing
                functor = cppast.DeclRefExpr("pk_f")
                functor_access = cppast.MemberExpr(functor, cpp_target)
                functor_assign = cppast.AssignOperator([functor_access], cpp_target_ref, cppast.BinaryOperatorKind.Assign)

                return cppast.CompoundStmt([decl, resize_call, copy_call, functor_assign])

        # Assign result of parallel_reduce
        if type(target) not in {ast.Name, ast.Subscript} and target.value.id == "self":
            target_name: str = visitors_util.get_node_name(target)
            if target_name not in self.reduction_result_queue:
                self.reduction_result_queue.append(target_name)

            call = cppast.CallStmt(self.visit(node.value))
            target_ref = cppast.DeclRefExpr(target_name)
            target_view_ref = cppast.DeclRefExpr(
                f"reduction_result_{target_name}")
            subscript = cppast.ArraySubscriptExpr(
                target_view_ref, [cppast.IntegerLiteral(0)])
            assign_op = cppast.BinaryOperatorKind.Assign

            # Holds the result of the reduction temporarily
            temp_ref = cppast.DeclRefExpr("pk_acc")
            target_assign = cppast.AssignOperator(
                [target_ref], temp_ref, assign_op)
            view_assign = cppast.AssignOperator(
                [subscript], target_ref, assign_op)

            return cppast.CompoundStmt([call, target_assign, view_assign])

        return super().visit_Assign(node)

    def visit_Attribute(self, node: ast.Attribute) -> cppast.DeclRefExpr:
        name: str = visitors_util.get_node_name(node)
        if name in self.work_units:
            return cppast.DeclRefExpr(name)

        if node.value.id == "self":
            if name in self.views:
                return name

            return cppast.DeclRefExpr(name)

        return super().visit_Attribute(node)

    def visit_Lambda(self, node: ast.Lambda) -> cppast.Expr:
        #NOTE: should handle args, kwonlyargs, varargs, kwargs, defaults
        return self.visit(node.body)

    def visit_Subscript(self, node: ast.Subscript) -> Union[cppast.ArraySubscriptExpr, cppast.CallExpr]:
        call: Union[cppast.ArraySubscriptExpr, cppast.CallExpr] = super().visit_Subscript(node)
        if isinstance(call, cppast.CallExpr):
            view_name: str = call.function.declname
            call._function._declname = f"pk_d_{view_name}"

        return call

    def visit_Call(self, node: ast.Call) -> Union[cppast.Expr, cppast.Stmt]:
        name: str = visitors_util.get_node_name(node.func)
        args: List[cppast.Expr] = [self.visit(a) for a in node.args]

        # Add pk_d_ before each view name to match mirror view names
        s = cppast.Serializer()
        for i in range(len(args)):
            if args[i] in self.views:
                if self.views[args[i]] is not None:
                    view: str = s.serialize(args[i])
                    args[i] = cppast.DeclRefExpr(f"pk_d_{view}")

        # Nested parallelism
        if name == "TeamPolicy":
            function = cppast.DeclRefExpr(f"Kokkos::{name}")
            if len(args) == 2:
                args.append(cppast.IntegerLiteral(1))

            policy = cppast.ConstructExpr(function, args)

            return policy

        if name in dir(TeamPolicy):
            team_policy = self.visit(node.func.value)
            return cppast.MemberCallExpr(team_policy, cppast.DeclRefExpr(name), args)

        elif name in ["RangePolicy", "MDRangePolicy"]:
            rank = len(node.args[0].elts)
            if rank == 0:
                self.error(node.value, "RangePolicy dimension must be greater than 0")
            if rank != len(node.args[1].elts):
                self.error(node.value, "RangePolicy dimension mismatch")

            iter_outer = Iterate.Default
            iter_inner = Iterate.Default
            for keyword in node.keywords:
                if keyword.arg == "rank":
                    explicit_rank = keyword.value.args[0].value
                    if explicit_rank != rank:
                        self.error(node.value, "RangePolicy dimension mismatch")
                        
                    iter_outer = getattr(Iterate, keyword.value.args[1].attr)
                    iter_inner = getattr(Iterate, keyword.value.args[2].attr)

            function = cppast.DeclRefExpr(f"Kokkos::{name}<Kokkos::Rank<{rank},{iter_outer.value},{iter_inner.value}>>")
            policy = cppast.ConstructExpr(cppast.DeclRefExpr(f"Kokkos::{name}"), args)
            if name == "MDRangePolicy":
                policy.add_template_param(cppast.DeclRefExpr(f"Kokkos::Rank<{rank},{iter_outer.value},{iter_inner.value}>"))

            return policy

        if name == "seconds":
            fence = cppast.CallStmt(cppast.CallExpr(
                cppast.DeclRefExpr("Kokkos::fence"), []))
            temp_decl = cppast.DeclRefExpr("pk_acc")
            seconds = cppast.MemberCallExpr(cppast.DeclRefExpr(
                "timer"), cppast.DeclRefExpr("seconds"), [])
            result = cppast.AssignOperator(
                [temp_decl], seconds, cppast.BinaryOperatorKind.Assign)

            return cppast.CompoundStmt([fence, result])

        function = cppast.DeclRefExpr(f"Kokkos::{name}")
        if name == "parallel_for":
            arg_start: int = 0 # Accounts for the optional kernel name
            kernel_name: Optional[cppast.StringLiteral] = None
            if isinstance(args[0], cppast.StringLiteral):
                kernel_name = args[0]
                arg_start = 1

            policy: cppast.ConstructExpr = args[arg_start]
            policy = self.add_space_to_policy(policy)

            if isinstance(node.args[arg_start + 1], ast.Lambda):
                decl: str = "KOKKOS_LAMBDA ("
                tid = cppast.DeclRefExpr(node.args[arg_start + 1].args.args[0].arg)

                # if target exists 
                if len(args) == arg_start + 3:
                    target = cppast.ArraySubscriptExpr(args[arg_start + 2], [tid])
                    args[arg_start + 1] = cppast.AssignOperator([target], args[arg_start + 1], cppast.BinaryOperatorKind.Assign)

                serializer = cppast.Serializer()
                decl += f"int {tid.declname}) {{"
                decl += serializer.serialize(args[arg_start + 1]) + ";}\n"

                call_args: List[cppast.Expr] = [policy, decl]
                if kernel_name is not None:
                    call_args.insert(0, kernel_name)

                return cppast.CallExpr(function, call_args)

            else:
                work_unit: str = args[arg_start + 1].declname
                policy = self.add_workunit_to_policy(policy, work_unit)

                call_args: List[cppast.Expr] = [policy, cppast.DeclRefExpr("pk_f")]
                if kernel_name is not None:
                    call_args.insert(0, kernel_name)

                return cppast.CallExpr(function, call_args)

        if name in ("parallel_reduce", "parallel_scan"):
            arg_start: int = 0 # Accounts for the optional kernel name
            kernel_name: Optional[cppast.StringLiteral] = None
            if isinstance(args[0], cppast.StringLiteral):
                kernel_name = args[0]
                arg_start = 1

            initial_value: cppast.Expr
            if len(args) == arg_start + 3:
                initial_value = args[arg_start + 2]
            else:
                initial_value = cppast.IntegerLiteral(0)

            acc_decl = cppast.DeclRefExpr("pk_acc")
            init_var = cppast.BinaryOperator(acc_decl, initial_value, cppast.BinaryOperatorKind.Assign)

            policy: cppast.ConstructExpr = args[arg_start]
            policy = self.add_space_to_policy(policy)

            if isinstance(node.args[arg_start + 1], ast.Lambda):
                decl: str = "KOKKOS_LAMBDA ("
                tid = cppast.DeclRefExpr(node.args[arg_start + 1].args.args[0].arg)
                acc = cppast.DeclRefExpr(node.args[arg_start + 1].args.args[1].arg)

                # assign to accumulator
                args[arg_start + 1] = cppast.AssignOperator([acc], args[arg_start + 1], cppast.BinaryOperatorKind.Assign)

                serializer = cppast.Serializer()
                decl += f"int {tid.declname}, double& {acc.declname}) {{"
                decl += serializer.serialize(args[arg_start + 1]) + ";}\n"

                call_args: List[cppast.Expr] = [policy, decl, acc_decl]
                if kernel_name is not None:
                    call_args.insert(0, kernel_name)

                call = cppast.CallExpr(function, call_args)

            else:
                work_unit: str = args[arg_start + 1].declname
                policy = self.add_workunit_to_policy(policy, work_unit)

                call_args: List[cppast.Expr] = [policy, cppast.DeclRefExpr("pk_f"), acc_decl]
                if kernel_name is not None:
                    call_args.insert(0, kernel_name)

                return cppast.CallExpr(function, call_args)

            return cppast.BinaryOperator(init_var, call, cppast.BinaryOperatorKind.Comma)

        if name in dir(BinSort):
            sorter: str = visitors_util.get_node_name(node.func.value)
            sorter_ref = cppast.DeclRefExpr(sorter)
            function = cppast.DeclRefExpr(name)

            return cppast.MemberCallExpr(sorter_ref, function, args)

        if name == "shmem_size":
            if len(args) != 1:
                self.error(node.func, "shmem_size() accepts only a single argument")

            func: ast.Attribute = node.func
            cpp_view_type: str = self.get_scratch_view_type(func.value)

            if cpp_view_type is None:
                self.error(func, "Wrong call to shmem_size()")

            view_type = cppast.ClassType(cpp_view_type)
            call_expr = cppast.MemberCallExpr(view_type, name, args)
            call_expr.is_static = True

            return call_expr

        return super().visit_Call(node)

    def visit_Constant(self, node: ast.Constant) -> cppast.DeclRefExpr:
        if isinstance(node.value, str) and node.value == "auto":
            return cppast.DeclRefExpr("Kokkos::AUTO")

        return super().visit_Constant(node)

    def add_space_to_policy(self, policy: Union[cppast.ConstructExpr, cppast.MemberCallExpr]) -> Union[cppast.ConstructExpr, cppast.MemberCallExpr]:
        """
        Add the execution space to the execution policy

        :param policy: the execution policy (could also be an integer)
        :returns: the execution policy
        """

        # Replace the number of threads with a RangePolicy
        if type(policy) not in (cppast.ConstructExpr, cppast.MemberCallExpr):
            begin = cppast.IntegerLiteral(0)
            policy = cppast.ConstructExpr(cppast.DeclRefExpr("Kokkos::RangePolicy"), [begin, policy])

        space = cppast.DeclRefExpr(Keywords.DefaultExecSpace.value)
        policy_constructor = self.get_policy_constructor(policy)
        policy_constructor.add_template_param(space)

        return policy

    def add_workunit_to_policy(self, policy: Union[cppast.ConstructExpr, cppast.MemberCallExpr], work_unit: str) -> Union[cppast.ConstructExpr, cppast.MemberCallExpr]:
        """
        Add the workunit tag to the execution policy

        :param policy: the execution policy (could also be an integer)
        :param work_unit: the tag of the workunit
        :returns: the execution policy
        """

        policy_constructor = self.get_policy_constructor(policy)
        policy_constructor.add_template_param(cppast.DeclRefExpr(f"{self.functor}::{work_unit}_tag"))

        return policy

    def get_policy_constructor(self, policy: Union[cppast.ConstructExpr, cppast.MemberCallExpr]) -> cppast.ConstructExpr:
        """
        Get the call to the policy constructor from the policy object

        :param: the policy object
        :returns: the call to the constructor
        """

        if isinstance(policy, cppast.MemberCallExpr):
            return policy.base
        else:
            return policy

    def generate_subview(self, node: ast.Assign, view_name: str) -> cppast.DeclStmt:
        """
        Generate a subview in pk.main. This involves adding the
        "pk_d_" prefix to the parent view.
        """

        return super().generate_subview(node, f"pk_d_{view_name}")
