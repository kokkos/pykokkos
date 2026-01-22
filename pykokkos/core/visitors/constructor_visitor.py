import ast
from typing import List, Dict, Optional, Tuple, Union

from pykokkos.core import cppast
from pykokkos.interface import Layout, MemorySpace, Trait

from . import visitors_util


class ConstructorVisitor(ast.NodeVisitor):
    """
    Gets the members of a functor
    """

    def __init__(
        self, src: Tuple[List[str], int], member_type: str, pk_import: str, debug: bool
    ):
        """
        ConstructorVisitor constructor

        :param src: the python source code of the workload
        :param member_type: specifies which members to retrieve, "fields", "views", or "typeinfo"
        :param pk_import: the identifier used to access the PyKokkos package
        :param debug: if true, prints the python AST when an error is encountered
        """

        if member_type not in ("fields", "views", "typeinfo", "randpool"):
            raise ValueError(
                'member_type has to be either "fields", "views", "typeinfo", or "randpool"'
            )

        self.src: Tuple[List[str], int] = src
        self.member_type: str = member_type
        self.pk_import: str = pk_import
        self.debug: bool = debug

    # Return a list of tuples containing field name and type
    def visit_ClassDef(self, node: ast.ClassDef) -> List[Tuple]:
        if self.member_type == "typeinfo":
            return self.get_typeinfo(node)
        else:
            for function in node.body:
                if function.name == "__init__":
                    return self.visit(function)

        self.error(node, "Missing constructor")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> List[Tuple]:
        if self.member_type == "typeinfo":
            return self.get_typeinfo(node)

        members: List[Tuple] = []

        for statement in node.body:
            if self.member_type in ("fields", "views", "randpool") and isinstance(
                statement, ast.AnnAssign
            ):
                ann_assign: Tuple = self.visit(statement)
                if len(ann_assign) != 0:
                    members.append(ann_assign)
            elif self.member_type == "views" and isinstance(statement, ast.Assign):
                assign: Tuple = self.visit(statement)
                if len(assign) != 0:
                    members.append(assign)

        return members

    def visit_AnnAssign(
        self, node: ast.AnnAssign
    ) -> Tuple[cppast.DeclRefExpr, cppast.Type]:
        declref: cppast.DeclRefExpr = self.visit(node.target)
        decltype: cppast.Type

        if self.member_type == "fields":
            # get only primitive types and pykokkos datatypes
            if not isinstance(node.annotation, ast.Name):
                if not self.is_pk_dtype(node.annotation):
                    return ()

            decltype = visitors_util.get_type(node.annotation, self.pk_import)
            if decltype.typename in (
                "Random_XorShift64_Pool",
                "Random_XorShift1024_Pool",
            ):
                return ()

            if decltype is None:
                self.error(node, "Type is not supported")

        elif self.member_type == "views":
            # get only view types
            if not isinstance(node.annotation, ast.Subscript):
                return ()

            decltype: cppast.ClassType = visitors_util.get_type(
                node.annotation, self.pk_import
            )
            # do not return lists
            if isinstance(decltype, cppast.PrimitiveType):
                return ()

        elif self.member_type == "randpool":
            if not isinstance(node.annotation, ast.Attribute):
                return ()

            rand_pool_type: str = node.annotation.attr

            if rand_pool_type not in (
                "Random_XorShift64_Pool",
                "Random_XorShift1024_Pool",
            ):
                return ()

            decltype: cppast.ClassType = cppast.ClassType(rand_pool_type)

        return (declref, decltype)

    # All attributes are fields
    def visit_Attribute(self, node: ast.Attribute) -> cppast.DeclRefExpr:
        if node.value.id == "self":
            name: str = visitors_util.get_node_name(node)
            return cppast.DeclRefExpr(name)

        self.error(node, "Can only define instance variables")

    # All the views defined with ViewTypeInfo
    def visit_keyword(self, node: ast.keyword) -> List[cppast.DeclRefExpr]:
        call = node.value
        if not isinstance(call, ast.Call):
            self.error(node.parent, "Only ViewTypeInfo objects are allowed")

        func = call.func
        if not isinstance(func, ast.Attribute):
            self.error(func, "Only ViewTypeInfo objects are allowed")

        if func.value.id != self.pk_import or func.attr != "ViewTypeInfo":
            self.error(func, "Only ViewTypeInfo objects are allowed")

        template_params: List[cppast.DeclRefExpr] = []
        layout: Optional[Layout] = self.get_layout(call)
        space: Optional[cppast.DeclRefExpr] = self.get_memory_space(call)
        trait: Optional[Trait] = self.get_trait(call)

        if layout is not None:
            template_params.append(layout)

        if space is not None:
            template_params.append(space)

        if trait is not None:
            template_params.append(trait)

        return template_params

    def get_typeinfo(
        self, node: Union[ast.Call, ast.FunctionDef]
    ) -> Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]]:
        """
        Get the view type info from a decorator

        :param node: the decorator call
        :returns: a dictionary mapping from view name to template params
        """

        decorator: ast.Call = None

        for d in node.decorator_list:
            if isinstance(d, ast.Call):
                func = d.func

                if isinstance(func, ast.Attribute):
                    if func.value.id == self.pk_import and func.attr in (
                        "functor",
                        "workload",
                        "workunit",
                    ):
                        decorator = d

        type_info: Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]] = {}

        if decorator is not None:
            for k in decorator.keywords:
                view = cppast.DeclRefExpr(k.arg)
                type_info[view] = self.visit(k)

        return type_info

    def get_layout(self, node: ast.Call) -> Optional[cppast.DeclRefExpr]:
        if not hasattr(node, "keywords"):
            return None

        args: List[ast.keyword] = node.keywords

        for a in args:
            if a.arg == "layout":
                if not isinstance(a.value, ast.Attribute) and not isinstance(
                    a.value.value, ast.Attribute
                ):
                    self.error(
                        node,
                        "Layout argument should be of the form pk.layout.Layout...",
                    )

                layout: str = visitors_util.get_node_name(a.value)

                if layout in Layout.__members__:
                    return cppast.DeclRefExpr(layout)
                else:
                    self.error(node, "Unrecognized layout")

        return None

    def get_memory_space(self, node: ast.Call) -> Optional[cppast.DeclRefExpr]:
        if not hasattr(node, "keywords"):
            return None

        args: List[ast.keyword] = node.keywords

        for a in args:
            if a.arg == "space":
                if not isinstance(a.value, ast.Attribute) and not isinstance(
                    a.value.value, ast.Attribute
                ):
                    self.error(
                        node,
                        "MemorySpace argument should be of the form pk.MemorySpace.HostSpace...",
                    )

                space: str = visitors_util.get_node_name(a.value)

                if space in MemorySpace.__members__:
                    return cppast.DeclRefExpr(space)
                else:
                    self.error(node, "Unrecognized memory space")

        return None

    def get_trait(self, node: ast.Call) -> Optional[cppast.DeclRefExpr]:
        if not hasattr(node, "keywords"):
            return None

        args: List[ast.keyword] = node.keywords
        for a in args:
            if a.arg == "trait":
                if not isinstance(a.value, ast.Attribute):
                    self.error(
                        node, "Trait argument should be of the form pk.Trait.Atomic..."
                    )

                trait: str = visitors_util.get_node_name(a.value)

                if trait in Trait.__members__:
                    return cppast.DeclRefExpr(trait)
                else:
                    self.error(node, "Unrecognized trait")

        return None

    def is_pk_dtype(self, node: ast.Attribute) -> bool:
        """
        Check if a type annotation is a pykokkos datatype

        :param node: the type annotation
        :returns: true or false
        """

        if not isinstance(node, ast.Attribute):
            return False

        qualifier: str = node.value.id

        return qualifier == self.pk_import

    def visit_Assign(self, node: ast.Assign) -> Tuple[cppast.DeclRefExpr, cppast.Type]:
        """
        Extract PyKokkos View members from numpy/cupy array to allow users to
        initialize views from numpy/cupy arrays (e.g., self.v = np.array(...)),
        which are automatically converted to PyKokkos View types.

        :param node: the assignment AST node
        :returns: a tuple of (name, view_type) if it's a numpy/cupy array, else empty tuple
        """
        if self.member_type != "views":
            return ()

        # make sure that this is assignment to self attributes
        if len(node.targets) != 1:
            return ()
        target = node.targets[0]
        if (
            not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
        ):
            return ()

        # make sure that value is a numpy/cupy array function call
        declref: cppast.DeclRefExpr = self.visit(target)
        value = node.value
        if not isinstance(value, ast.Call):
            return ()
        func = value.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return ()
        array_module = func.value.id
        if array_module not in ("np", "numpy", "cp", "cupy"):
            return ()

        # infer view type from the array call
        decltype = self._infer_view_type_from_array_call(value, array_module)
        if decltype is None:
            return ()

        return (declref, decltype)

    def _infer_view_type_from_array_call(
        self, call: ast.Call, array_module: str
    ) -> Optional[cppast.ClassType]:
        """
        Infer the View type from a numpy/cupy array creation call

        :param call: the AST call node
        :param array_module: the module name (np, numpy, cp, cupy)
        :returns: the inferred view type or None
        """
        # shape from first positional argument
        if len(call.args) == 0:
            return None

        shape_arg = call.args[0]

        # determine the rank
        rank = 1
        if isinstance(shape_arg, (ast.List, ast.Tuple)):
            rank = len(shape_arg.elts)
        elif isinstance(shape_arg, ast.Constant):
            rank = 1

        # handle dtype
        dtype = None
        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = kw.value
                break
        if (
            dtype is None
            and len(call.args) > 1
            and isinstance(call.args[1], ast.Attribute)
        ):
            dtype = call.args[1]

        # translate dtype using existing get_type function
        if dtype is None:
            dtype_type = cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)
        elif (
            isinstance(dtype, ast.Attribute)
            and hasattr(dtype, "value")
            and isinstance(dtype.value, ast.Name)
        ):
            module_name = dtype.value.id
            dtype_attr = dtype.attr
            if module_name in ("np", "numpy", "cp", "cupy"):
                # edge case: numpy/cupy float64 -> double, float32 -> float
                dtype_name = (
                    "double"
                    if dtype_attr == "float64"
                    else ("float" if dtype_attr == "float32" else dtype_attr)
                )
                if dtype_name in visitors_util.view_dtypes:
                    dtype_value = visitors_util.view_dtypes[dtype_name]
                    if isinstance(dtype_value, cppast.BuiltinType):
                        dtype_type = cppast.PrimitiveType(dtype_value)
                    else:
                        dtype_type = cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)
                else:
                    dtype_type = cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)
            else:
                dtype_type = visitors_util.get_type(dtype, self.pk_import)
                if dtype_type is None or not isinstance(
                    dtype_type, cppast.PrimitiveType
                ):
                    dtype_type = cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)
        else:
            dtype_type = visitors_util.get_type(dtype, self.pk_import)
            if dtype_type is None or not isinstance(dtype_type, cppast.PrimitiveType):
                dtype_type = cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)

        view_type_name = f"View{rank}D"
        view_type = cppast.ClassType(view_type_name)
        view_type.add_template_param(dtype_type)

        return view_type

    def error(self, node: ast.AST, message: str):
        visitors_util.error(self.src, self.debug, node, message)
