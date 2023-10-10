import ast
import inspect
import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple, Union

from pykokkos.core import cppast
from pykokkos.interface import Decorator, UpdatedTypes, get_type_str

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
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                if node.names[0].name == package:
                    alias: ast.alias = node.names[0]
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

        for i, node in enumerate(self.tree.body):
            if check_entity(node, self.pk_import):

                start: int = node.lineno - 1
                try:
                    stop: int = self.tree.body[i + 1].lineno - 1
                except IndexError:
                    stop = len(self.lines)

                name: str = node.name

                entity = PyKokkosEntity(style, cppast.DeclRefExpr(name), node, (self.lines[start:stop], start), self.path, self.pk_import)
                entities[name] = entity

        return entities


    def fix_types(self, entity: PyKokkosEntity, updated_types: UpdatedTypes) -> ast.AST:
        '''
        Inject (into the entity AST) the missing annotations for datatypes that have been inferred 

        entity: Pykokkos entity whose AST will be patched - the entity being compiled/translated
        updated_types: UpdatedTypes object that contains info about inferred types for this entity
        returns: the updated entity AST after injecting correct annotations (from updated_types) for datatypes
        '''

        style: PyKokkosStyles = entity.style
        assert style is PyKokkosStyles.workunit and updated_types is not None

        entity_tree: ast.AST = entity.AST

        needs_reset: bool = self.check_self(entity_tree)
        if needs_reset:
            entity_tree = self.reset_entity_tree(entity_tree, updated_types)

        # if modifications to layout decorator is needed, do not change original inferences
        if len(updated_types.layout_change):
            inferred_layouts = copy.deepcopy(updated_types.layout_change)
            entity_tree.decorator_list = self.fix_view_layout(entity_tree, inferred_layouts)

        for arg_obj in entity_tree.args.args:
            # Type already provided by the user
            if arg_obj.arg not in updated_types.inferred_types:
                continue

            update_type = updated_types.inferred_types[arg_obj.arg]
            arg_obj.annotation = self.get_annotation_node(update_type)
            
        assert entity_tree is not None
        return entity_tree

    def check_self(self, entity_tree: ast.AST) -> bool:
        '''
        Check if self args exists in the AST, which implies this AST was already
        translated 

        entity_tree: entity AST that needs to be examined
        returns: True if a 'self' argument exists, False otherwise
        '''

        for arg in entity_tree.args.args:
            if arg.arg == "self":
                return True
        return False

    def reset_entity_tree(self, entity_tree: ast.AST, updated_types: UpdatedTypes) -> ast.AST:
        '''
        Remove type annotations and self argument from the entity tree. This allows
        the types to be inserted again if they change dynamically

        entity_tree: Ast of pykokkos entity being resiet
        updated_types: inferred types information
        returns: updated entity ast as it would be in the first run
        '''

        args_list: List[ast.arg] = []
        param_list = updated_types.param_list
        for param in param_list:
            arg_obj = ast.arg(arg=param.name)
            if param.annotation is not inspect._empty:
                type_str = get_type_str(param.annotation)  # simplify inspect.annotation to string
                arg_obj.annotation = self.get_annotation_node(type_str)
            args_list.append(arg_obj)

        entity_tree.args.args = args_list
        entity_tree.decorator_list = [
            ast.Attribute(
                value=ast.Name(id=self.pk_import, ctx=ast.Load()), 
                attr="workunit",
                ctx=ast.Load())
        ]

        return entity_tree

    def get_annotation_node(self, type: str) -> ast.AST:
        '''
        Given a type return ast.annotation node

        type: str representing datatype (refer to type_inference.py for string formating)
        return: annotation node that can be inserted in the AST
        '''

        # For now, just so we can raise an error instead of unexpectedly crashing
        primitives_supported = ["int", "bool", "float"]
        annotation_node : ast.AST 
        if type in primitives_supported:
            annotation_node = ast.Name(id=type, ctx=ast.Load())

        elif "numpy:" in type:
            # update_type = numpy:int64
            dtype = type.split(':')[1]
            # Change numpy.<type> to equivalent pk.<type>
            annotation_node = ast.Attribute(
                value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                attr = dtype,
                ctx = ast.Load()
            )

        elif "View" in type:
            # update_type = View1D:double
            view_type, dtype = type.split(':')

            annotation_node = ast.Subscript(
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

        elif "Acc:" in type:
            dtype = type.split(":")[1]

            annotation_node = ast.Subscript(
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

        elif "TeamMember" in type: #"TeamMember" is hard-set in get_annotations
            annotation_node = ast.Attribute(
                value = ast.Name(id=self.pk_import, ctx=ast.Load()),
                attr = "TeamMember",
                ctx = ast.Load()
            )
        else:
            raise ValueError(f"Type inference for {type} is not supported")

        return annotation_node

    def fix_view_layout(self, node : ast.AST, layout_change: Dict[str, str]) -> List[ast.Call]:
        '''
        Construct the decorator list (as in AST) with the missing layout decorators for pykokkos views

        node: ast object for the entity
        layout_change: Dict that maps [view -> layout]
        returns: decorator list 
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
        '''
        Filter out (from the working dict) the views whose layouts decorators user has already provided

        node: AST of pykokkos entity to work with
        working_dict: map of view -> layout
        returns updated working dict
        '''

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