import ast
import copy
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.parsers import PyKokkosEntity, PyKokkosStyles
from pykokkos.core.visitors import ConstructorVisitor, KokkosMainVisitor, ParameterVisitor, visitors_util
from pykokkos.interface import Decorator, ViewTypeInfo


class PyKokkosMembers:
    """
    Holds all PyKokkos related members which are needed for translation
    """

    def __init__(self):
        self.fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType] = {}
        self.views: Dict[cppast.DeclRefExpr, cppast.ClassType] = {}
        self.view_template_params: Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]] = {}
        self.real_dtype_views: Set[cppast.DeclRefExpr] = {}

        self.pk_workunits: Dict[cppast.DeclRefExpr, ast.FunctionDef] = {}
        self.pk_functions: Dict[cppast.DeclRefExpr, ast.FunctionDef] = {}
        self.pk_mains: Dict[cppast.DeclRefExpr, ast.FunctionDef] = {}
        self.pk_callbacks: Dict[cppast.DeclRefExpr, ast.FunctionDef] = {}

        self.classtype_methods: Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]] = {}

        self.random_pool: Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]] = None

        self.reduction_result_queue: List[str] = []
        self.timer_result_queue: List[str] = []

        self.has_real: bool = False

    def extract(self, entity: PyKokkosEntity, classtypes: List[PyKokkosEntity]) -> None:
        """
        Add all PyKokkos information relevant information (fields, views, ...)
        to the translator before translation begins

        :param entity: the ast representation and source of the entity being translated
        :param dependencies: the list of classtypes needed by the entity
        """

        AST: Union[ast.ClassDef, ast.FunctionDef] = entity.AST
        source: Tuple[List[str], int] = entity.source
        pk_import: str = entity.pk_import

        if entity.style is PyKokkosStyles.workload:
            self.pk_mains = self.get_decorated_functions(AST, Decorator.KokkosMain)
            self.fields = self.get_fields(AST, source, pk_import)
            self.views = self.get_views(AST, source, pk_import)
            self.random_pool = self.get_random_pool(AST, source, pk_import)

        elif entity.style is PyKokkosStyles.functor:
            self.fields = self.get_fields(AST, source, pk_import)
            self.views = self.get_views(AST, source, pk_import)
            self.random_pool = self.get_random_pool(AST, source, pk_import)

        elif entity.style in {PyKokkosStyles.fused, PyKokkosStyles.workunit}:
            # for operation by default
            param_begin: int = 1
            
            # check for accumulator
            args: List[ast.arg] = AST.args.args
            for i, arg in enumerate(args):
                if isinstance(arg.annotation, ast.Subscript) and arg.annotation.value.attr == "Acc":
                    param_begin = i + 1
                    # handle last_pass param for parallel_scan
                    if i + 1 <= len(args) and isinstance(args[i+1].annotation, ast.Name) and \
                            args[i+1].annotation.id == "bool":
                        param_begin += 1
                    break

            self.fields, self.views = self.get_params(AST, source, param_begin, pk_import)

        self.real_dtype_views = self.get_real_views()
        if len(self.real_dtype_views) != 0:
            self.has_real = True
        self.view_template_params = self.get_view_template_params(AST, source, pk_import)

        for n, t in self.views.items():
            if n in self.view_template_params:
                t.template_params.extend(self.view_template_params[n])

        if entity.style in (PyKokkosStyles.workload, PyKokkosStyles.functor):
            self.pk_workunits = self.get_decorated_functions(AST, Decorator.WorkUnit)
            self.pk_functions = self.get_decorated_functions(AST, Decorator.KokkosFunction)
            self.pk_callbacks = self.get_decorated_functions(AST, Decorator.KokkosCallback)
        else:
            self.pk_workunits[cppast.DeclRefExpr(AST.name)] = AST
            self.pk_functions = self.get_decorated_functions(entity.full_AST, Decorator.KokkosFunction)

        self.classtype_methods = self.get_classtype_methods(classtypes)

        if entity.style is PyKokkosStyles.workload:
            name: str = f"pk_functor_{entity.name}"
            self.reduction_result_queue, self.timer_result_queue = self.get_queues(source, name, pk_import)

        if len(self.pk_mains) > 1:
            print("ERROR: Only one pk.main function can be translated")
            sys.exit(1)

    def get_fields(self, classdef: ast.ClassDef, source: Tuple[List[str], int], pk_import: str) -> Dict[cppast.DeclRefExpr, cppast.PrimitiveType]:
        """
        Get all fields (or instance variables) in classdef by parsing the constructor

        :param classdef: the classdef being parsed
        :param source: the python source code of the workload
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: a dictionary mapping from field name to type
        """

        visitor = ConstructorVisitor(source, "fields", pk_import, True)
        fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType]
        fields = dict(visitor.visit(classdef))

        return fields

    def get_views(self, classdef: ast.ClassDef, source: Tuple[List[str], int], pk_import: str) -> Dict[cppast.DeclRefExpr, cppast.ClassType]:
        """
        Get all views defined in classdef by parsing the constructor

        :param classdef: the classdef to be parsed
        :param source: the python source code of the workload
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: a dictionary mapping from view name to type (only dimensionality and type)
        """

        visitor = ConstructorVisitor(source, "views", pk_import, True)
        views: Dict[cppast.DeclRefExpr, cppast.ClassType]
        views = dict(visitor.visit(classdef))

        return views

    def get_queues(self, source: Tuple[List[str], int], name: str, pk_import: str) -> Tuple[List[str], List[str]]:
        """
        Get all fields assigned to a reduction result or timer result

        :param source: the python source code of the workload
        :param name: the name of the workload
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: two lists, one for the reduction results and one for the timer results
        """

        views = copy.deepcopy(self.views) # Needed since KokkosMainVisitor modifies views

        # Copied from translate_mains() in bindings.py
        node_visitor = KokkosMainVisitor(
            {}, source, views, self.pk_workunits,
            self.fields, self.pk_functions,
            self.classtype_methods, name, pk_import, debug=True)

        for main in self.pk_mains.values():
            try:
                node_visitor.visit(main)
            except NotImplementedError:
                print(f"Translation of {main.name} failed")
                sys.exit(1)

        return (node_visitor.reduction_result_queue, node_visitor.timer_result_queue)

    def get_real_views(self):
        """
        Get all the views that contain a pk.real datatype

        :returns: a set of view names to type (only dimensionality and type)
        """

        views: Set[cppast.DeclRefExpr] = set()
        for n, t in self.views.items():
            dtype: cppast.PrimitiveType = t.template_params[0]
            if isinstance(dtype, cppast.PrimitiveType):
                if isinstance(dtype.typename, str) and dtype.typename == Keywords.RealPrecision.value:
                    views.add(n)

        return views

    def get_params(
        self,
        functiondef: ast.FunctionDef,
        source: Tuple[List[str], int],
        param_begin: int,
        pk_import: str
    ) -> Tuple[Dict[cppast.DeclRefExpr, cppast.PrimitiveType], Dict[cppast.DeclRefExpr, cppast.ClassType]]:
        """
        Gets all fields and views passed as parameters to the workunit

        :param functiondef: the functiondef to be parsed
        :param source: the python source code of the workload
        :param param_begin: where workunit argument begins (excluding tid/acc)
        """

        visitor = ParameterVisitor(source, param_begin, pk_import, True)
        visitor.visit(functiondef)

        return (visitor.fields, visitor.views)

    def get_view_template_params(
        self,
        node: Union[ast.ClassDef, ast.FunctionDef],
        source: Tuple[List[str], int],
        pk_import: str
    ) -> Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]]:
        """
        Get the template parameters for all views defined in the constructor

        :param node: the classdef or functiondef to be parsed
        :param source: the python source code of the workload
        :returns: a dictionary mapping from view name to a list of template parameters
        """

        visitor = ConstructorVisitor(source, "typeinfo", pk_import, True)
        type_info: Dict[cppast.DeclRefExpr, ViewTypeInfo]
        type_info = dict(visitor.visit(node))

        return type_info

    def get_decorated_functions(self, classdef: ast.ClassDef, decorator: Decorator) -> Dict[cppast.DeclRefExpr, ast.FunctionDef]:
        visitor = ast.NodeVisitor()
        functions: Dict[cppast.DeclRefExpr, ast.FunctionDef] = {}

        def visit_FunctionDef(node: ast.FunctionDef):
            if node.decorator_list:
                is_standalone_workunit_decorator: bool = isinstance(node.decorator_list[0], ast.Call)

                if is_standalone_workunit_decorator:
                    return

                node_decorator: str = visitors_util.get_node_name(node.decorator_list[0])

                if decorator.value == node_decorator:
                    functions[cppast.DeclRefExpr(node.name)] = node

        visitor.visit_FunctionDef = visit_FunctionDef
        for method in classdef.body:
            visitor.visit(method)

        return functions

    def get_classtype_methods(self, classtypes: List[PyKokkosEntity]) -> Dict[cppast.DeclRefExpr, List[cppast.DeclRefExpr]]:
        classtype_methods: Dict[
            cppast.DeclRefExpr, List[cppast.DeclRefExpr]] = {}

        for c in classtypes:
            classdef: ast.ClassDef = c.AST

            classref = cppast.DeclRefExpr(classdef.name)
            classtype_methods[classref] = []

            for node in classdef.body:
                if isinstance(node, ast.FunctionDef):
                    function: cppast.DeclRefExpr

                    # If constructor
                    if node.name == "__init__":
                        function = cppast.DeclRefExpr(classdef.name)
                    else:
                        function = cppast.DeclRefExpr(node.name)

                    classtype_methods[classref].append(function)

        return classtype_methods

    def get_random_pool(self, classdef: ast.ClassDef, source: Tuple[List[str], int], pk_import: str) -> Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]]:
        """
        Gets the type of the random pool if it exists

        :param classdef: the classdef to be parsed
        :param source: the python source code of the workload
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: the type of the random pool if it exists
        """

        visitor = ConstructorVisitor(source, "randpool", pk_import, True)
        fields = dict(visitor.visit(classdef))

        if len(fields) > 1:
            print("ERROR: Only one pk.RandPool allowed")
            sys.exit(1)

        if len(fields) == 0:
            return None

        declref: cppast.DeclRefExpr
        decltype: cppast.ClassType
        for n, t in fields.items():
            declref = n
            decltype = t

        return declref, decltype