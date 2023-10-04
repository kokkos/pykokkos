import ast
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple, Union

from pykokkos.core import cppast
from pykokkos.interface import Decorator, UpdatedTypes

class PyKokkosStyles(Enum):
    """
    An Enum of all the different styles allowed in PyKokkos
    """

    functor = auto()
    workload = auto()
    workunit = auto()
    classtype = auto()

@dataclass
class PyKokkosEntity:
    """
    The representation of a PyKokkos enity produced by Parser
    """

    style: PyKokkosStyles
    name: cppast.DeclRefExpr
    AST: Union[ast.ClassDef, ast.FunctionDef]
    source: Tuple[List[str], int]
    path: str
    pk_import: str

class Parser:
    """
    Parse a PyKokkos workload and its dependencies
    """

    def __init__(self, path: str):
        """
        Parse the file and find all entities

        :param path: the path to the file
        """
        self.lines: List[str]
        self.tree: ast.Module
        with open(path, "r") as f:
            self.lines = f.readlines()
            self.tree = ast.parse("".join(self.lines))


        self.path: str = path
        self.pk_import: str = self.get_import()
        self.workloads: Dict[str, PyKokkosEntity] = {}
        self.classtypes: Dict[str, PyKokkosEntity] = {}
        self.functors: Dict[str, PyKokkosEntity] = {}
        self.workunits: Dict[str, PyKokkosEntity] = {}

        self.workloads = self.get_entities(PyKokkosStyles.workload)
        self.classtypes = self.get_entities(PyKokkosStyles.classtype)
        self.functors = self.get_entities(PyKokkosStyles.functor)
        self.workunits = self.get_entities(PyKokkosStyles.workunit)


    def get_import(self) -> str:
        """
        Get the pykokkos import identifier

        :returns: the name of the pykokkos import
        """

        package: str = "pykokkos"
        for entity_tree in self.tree.body:
            if isinstance(entity_tree, ast.Import):
                if entity_tree.names[0].name == package:
                    alias: ast.alias = entity_tree.names[0]
                    package = alias.name if alias.asname is None else alias.asname

        return package


    def get_classtypes(self) -> List[PyKokkosEntity]:
        """
        Get a list of parsed classtypes

        :returns: the PyKokkosEntity representation of the classtypes
        """

        return list(self.classtypes.values())

    def get_entity(self, name: str) -> PyKokkosEntity:
        """
        Get the parsed entity

        :param name: the name of the functor
        :returns: the PyKokkosEntity representation of the entity
        """

        if name in self.workloads:
            return self.workloads[name]
        if name in self.functors:
            return self.functors[name]

        return self.workunits[name]

    def get_entities(self, style: PyKokkosStyles) -> Dict[str, PyKokkosEntity]:
        """
        Get the entities from path that are of a particular style

        :param style: the style of the entity to get
        :returns: a dict mapping the name of each entity to a PyKokkosEntity instance
        """

        entities: Dict[str, PyKokkosEntity] = {}
        check_entity: Callable[[ast.stmt], bool]

        if style is PyKokkosStyles.workload:
            check_entity = self.is_workload
        elif style is PyKokkosStyles.functor:
            check_entity = self.is_functor
        elif style is PyKokkosStyles.workunit:
            check_entity = self.is_workunit
        elif style is PyKokkosStyles.classtype:
            check_entity = self.is_classtype

        for i, entity_tree in enumerate(self.tree.body):
            if check_entity(entity_tree, self.pk_import):

                start: int = entity_tree.lineno - 1
                try:
                    stop: int = self.tree.body[i + 1].lineno - 1
                except IndexError:
                    stop = len(self.lines)
                
                name: str = entity_tree.name

                entity = PyKokkosEntity(style, cppast.DeclRefExpr(name), entity_tree, (self.lines[start:stop], start), self.path, self.pk_import)
                entities[name] = entity

        return entities


    def fix_types(self, entity: PyKokkosEntity, updated_types: UpdatedTypes) -> ast.AST:
        '''
        updatedTypes: object that contains info about inferred types

        This method will walk the ast for a workunit and add any mission annotations
        This method will also invoke fix_viewlayouts that will add missing decorators for user specified layouts
        '''

        style: PyKokkosStyles = entity.style
        assert style is PyKokkosStyles.workunit and updated_types is not None

        # For now, just so we can raise an error instead of unexpectedly crashing
        primitives_supported = ["int", "bool", "float"]

        entity_tree: ast.AST = entity.AST

        # if modifications to layout decorator is needed
        if len(updated_types.layout_change):
            entity_tree.decorator_list = self.fix_view_layout(entity_tree, updated_types.layout_change)

        for arg_obj in entity_tree.args.args:
            # Type already provided by the user
            if arg_obj.arg not in updated_types.inferred_types:
                continue

            update_type = updated_types.inferred_types[arg_obj.arg]

            if update_type in primitives_supported:
                arg_obj.annotation = ast.Name(id=update_type, ctx=ast.Load())

            elif "numpy:" in update_type:
                # update_type = numpy:int64
                dtype = update_type.split(':')[1]
                # Change numpy.<type> to equivalent pk.<type>
                arg_obj.annotation = ast.Attribute(
                    value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                    attr = dtype,
                    ctx = ast.Load()
                )

            elif "View" in update_type:
                # update_type = View1D:double
                view_type, dtype = update_type.split(':')

                arg_obj.annotation = ast.Subscript(
                    value = ast.Attribute(
                        value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                        attr = view_type,
                        ctx = ast.Load()
                    ),
                    slice = ast.Attribute(
                        value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                        attr = dtype,
                        ctx = ast.Load()
                    ),
                    ctx = ast.Load()
                )

            elif "Acc:" in update_type:
                dtype = update_type.split(":")[1]

                arg_obj.annotation = ast.Subscript(
                        value = ast.Attribute(
                            value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                            attr = "Acc",
                            ctx = ast.Load()
                    ),
                    slice = ast.Attribute(
                        value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                        attr = dtype,
                        ctx=ast.Load(),
                    ),
                    ctx = ast.Load()
                )

            elif "pk.TeamMember" in update_type: #"pk.TeamMember" is hard-set in get_annotations
                arg_obj.annotation = ast.Attribute(
                    value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                    attr = "TeamMember",
                    ctx = ast.Load()
                )
            else:
                raise ValueError(f"Type inference for {update_type} is not supported")

        assert entity_tree is not None
        return entity_tree


    def fix_view_layout(self, node : ast.AST, layout_change: Dict[str, str]) -> List[ast.Call]:
        '''
        node: ast object for the workunit
        layout_change: Dict that stores view variable identifier as keys against the layout type

        This function returns the modified workunit ast with corrent layout decorators
        '''

        assert len(node.decorator_list), "Decorator cannot be missing for pykokkos workunit"
        # Decorator list will have ast.Call object as first element if user has provided layout decorators
        is_layout_given: bool = isinstance(node.decorator_list[0], ast.Call)

        if is_layout_given:
            # Filter out layouts already given by user
            layout_change = self.filter_layout_change(node, layout_change)

        if len(layout_change):
            call_obj = None

            if is_layout_given: # preserve the existing call object
                call_obj = node.decorator_list[0]
            else: 
                call_obj= ast.Call()
                call_obj.func = ast.Attribute(value=ast.Name(id=self.pk_import, ctx=ast.Load()), attr='workunit', ctx=ast.Load())
                call_obj.args = []
                call_obj.keywords = []

            for view, layout in layout_change.items():
                call_obj.keywords.append(
                    ast.keyword(
                        arg=view, 
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=self.pk_import, ctx=ast.Load()), 
                                attr='ViewTypeInfo', ctx=ast.Load()
                            ), 
                            args=[], 
                            keywords=[
                                ast.keyword(
                                    arg='layout', 
                                    value=ast.Attribute(
                                        value=ast.Attribute(
                                            value=ast.Name(id=self.pk_import, ctx=ast.Load()), 
                                            attr='Layout', ctx=ast.Load()), 
                                        attr= layout, ctx=ast.Load()
                                        )
                                )
                            ]
                        )
                    )
                )

            return [call_obj]

        # no change needed
        return node.decorator_list

    @staticmethod
    def filter_layout_change(node: ast.AST, working_dict: Dict[str, str]) -> Dict[str, str]:
        #MARK ADD DOCSTRING
        call_obj = node.decorator_list[0]
        # iterating over view layout decorators in signature (of workunit)
        for keyword_obj in call_obj.keywords:
            if keyword_obj.arg in working_dict:
                # user provided layout decorator for this view, remove from working dict
                del working_dict[keyword_obj.arg]

        return working_dict

    @staticmethod
    def is_classtype(node: ast.stmt, pk_import: str) -> bool:
        """
        Checks if an ast node is a a PyKokkos class

        :param node: the node being checked
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: true or false
        """

        if not isinstance(node, ast.ClassDef):
            return False

        for attribute in node.decorator_list:
            if isinstance(attribute, ast.Attribute):
                if (attribute.value.id == pk_import
                        and Decorator.is_kokkos_classtype(attribute.attr)):
                    return True

        return False

    @staticmethod
    def is_workload(node: ast.stmt, pk_import: str) -> bool:
        """
        Checks if an ast node is a a PyKokkos workload

        :param node: the node being checked
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: true or false
        """

        if not isinstance(node, ast.ClassDef):
            return False

        for decorator in node.decorator_list:
            attribute = None

            if isinstance(decorator, ast.Call):
                attribute = decorator.func
            elif isinstance(decorator, ast.Attribute):
                attribute = decorator

            if isinstance(attribute, ast.Attribute):
                if (attribute.value.id == pk_import and Decorator.is_workload(attribute.attr)):
                    return True

        return False

    @staticmethod
    def is_functor(node: ast.stmt, pk_import: str) -> bool:
        """
        Checks if an AST node is a functor

        :param node: the node being checked
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: true or false
        """

        if not isinstance(node, ast.ClassDef):
            return False

        for decorator in node.decorator_list:
            attribute = None

            if isinstance(decorator, ast.Call):
                attribute = decorator.func
            elif isinstance(decorator, ast.Attribute):
                attribute = decorator

            if isinstance(attribute, ast.Attribute):
                if (attribute.value.id == pk_import and Decorator.is_functor(attribute.attr)):
                    return True

        return False

    @staticmethod
    def is_workunit(node: ast.stmt, pk_import: str) -> bool:
        """
        Checks if an AST node is a workunit

        :param node: the node being checked
        :param pk_import: the identifier used to access the PyKokkos package
        :returns: true if a node is decorated with @pk.workunit
        """

        if not isinstance(node, ast.FunctionDef):
            return False

        for decorator in node.decorator_list:
            attribute = None

            if isinstance(decorator, ast.Call):
                attribute = decorator.func
            elif isinstance(decorator, ast.Attribute):
                attribute = decorator

            if isinstance(attribute, ast.Attribute):
                # Needed to get the attribute when the decorator is of
                # the form A.B.C
                while isinstance(attribute.value, ast.Attribute):
                    attribute = attribute.value

                if (attribute.value.id == pk_import and Decorator.is_work_unit(attribute.attr)):
                    return True

        return False