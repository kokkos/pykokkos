import ast
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Tuple, Union

from pykokkos.core import cppast
from pykokkos.interface import Decorator, UpdatedTypes
from pykokkos.core.visitors import RemoveTransformer
from copy import deepcopy


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
        print("--------------------------------->> PARSER INITIALIZED!")
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
        print("functors:", self.functors.keys())
        print("workloads:", self.workloads.keys())
        print("workunits:", self.workunits.keys())

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

        # print("STYLE: ", style, end="\n")
        # print("PATH: ", self.path)
        if style is PyKokkosStyles.workload:
            check_entity = self.is_workload
        elif style is PyKokkosStyles.functor:
            check_entity = self.is_functor
        elif style is PyKokkosStyles.workunit:
            check_entity = self.is_workunit
        elif style is PyKokkosStyles.classtype:
            check_entity = self.is_classtype

        for i, node in enumerate(self.tree.body):
            # if style is PyKokkosStyles.functor:
            #     print("entity check:", check_entity)
            #     print(ast.dump(node)[0:50])
            #     print()
            if check_entity(node, self.pk_import):
                # print("---> TRUE")
                # print(ast.dump(node))
                # print()

                start: int = node.lineno - 1

                try:
                    stop: int = self.tree.body[i + 1].lineno - 1
                except IndexError:
                    stop = len(self.lines)
                
                name: str = node.name
                print("getting entity:", name, ":",self.lines[start:stop])
                entity = PyKokkosEntity(style, cppast.DeclRefExpr(name), node, (self.lines[start:stop], start), self.path, self.pk_import)
                entities[name] = entity

        return entities


    # @Hannan updating this to remove other workunit nodes from the AST 
    def fix_types(self, entity: PyKokkosEntity, updated_types: List[UpdatedTypes]):
        
        check_entity: Callable[[ast.stmt], bool]
        style: PyKokkosStyles = entity.style

        if style is PyKokkosStyles.workload:
            check_entity = self.is_workload
        elif style is PyKokkosStyles.functor:
            check_entity = self.is_functor
        elif style is PyKokkosStyles.workunit:
            check_entity = self.is_workunit
        elif style is PyKokkosStyles.classtype:
            check_entity = self.is_classtype

        # REMOVING NODES NOT NEEDED FROM AST
        entity_tree = Union[ast.ClassDef, ast.FunctionDef]
        working_tree = deepcopy(self.tree)
        for node in self.tree.body:
            if check_entity(node, self.pk_import):
                units = node.body
                for unit in units:
                    print(">>>>>> Scanning to remove:", unit.name)
                    for update_obj in updated_types:
                        if update_obj is not None and unit.name != "__init__" and unit.name != update_obj.workunit.__name__:
                            print("REMOVING FROM AST: ", unit.name)
                            transformer = RemoveTransformer(unit)
                            working_tree = transformer.visit(working_tree)
        self.tree = working_tree
        print()

        # Changing annotations for the needed workunit definitions
        for i, node in enumerate(self.tree.body):
            if check_entity(node, self.pk_import):
                units = node.body
                for unit in units:
                    print(">>>>> Scanning to change types:", unit.name)
                    for update_obj in updated_types:
                     
                        if update_obj is not None and update_obj.workunit.__name__ == unit.name:
                            print("Needs modification:", ast.dump(unit))

                            for arg_obj in unit.args.args:
                                for update_arg, update_type in update_obj.inferred_types.items():
                                    if update_arg == arg_obj.arg:
                                        print("Changing to", update_type.__name__)
                                        arg_obj.annotation = ast.Name(id="int", ctx=ast.Load())
                                        print(arg_obj.arg, arg_obj.annotation.id)
                                    # change the types to those of dictionaries, just args for now
                                    # update_obj.inferred_types
        print()
        # Checking to ensure changes reflect in the AST
        for i, node in enumerate(self.tree.body):       
            if check_entity(node, self.pk_import):
                entity_tree = node
                units = node.body
                for unit in units:
                    for update_obj in updated_types:
                        if update_obj is not None and update_obj.workunit.__name__ != unit.name:
                            print(unit.name, "EXISTS IN AST")
                        if update_obj is not None and update_obj.workunit.__name__ == unit.name:
                            for arg_obj in unit.args.args:
                                for update_arg, update_type in update_obj.inferred_types.items():
                                    if update_arg == arg_obj.arg:
                                        print(arg_obj.arg, arg_obj.annotation.id)
                                        print("Modified:", ast.dump(unit), "\n\n")
                                    # change the types to those of dictionaries, just args for now
                                    # update_obj.inferred_types

        # print("returning: \n", ast.dump(entity_tree))
        return entity_tree

# FunctionDef(name='y_init', args=arguments(posonlyargs=[], args=[arg(arg='self'), arg(arg='i')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Assign(targets=[Subscript(value=Attribute(value=Name(id='self', ctx=Load()), attr='y', ctx=Load()), slice=Name(id='i', ctx=Load()), ctx=Store())], value=Constant(value=1))], decorator_list=[Attribute(value=Name(id='pk', ctx=Load()), attr='workunit', ctx=Load())])
# FunctionDef(name='y_init', args=arguments(posonlyargs=[], args=[arg(arg='self'), arg(arg='i', annotation=Name(id='int', ctx=Load()))], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Assign(targets=[Subscript(value=Attribute(value=Name(id='self', ctx=Load()), attr='y', ctx=Load()), slice=Name(id='i', ctx=Load()), ctx=Store())], value=Constant(value=1))], decorator_list=[Attribute(value=Name(id='pk', ctx=Load()), attr='workunit', ctx=Load())]) 
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
