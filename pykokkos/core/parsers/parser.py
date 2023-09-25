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
                # print("getting entity:", name, ":",self.lines[start:stop])
                entity = PyKokkosEntity(style, cppast.DeclRefExpr(name), node, (self.lines[start:stop], start), self.path, self.pk_import)
                entities[name] = entity

        return entities


    # @HannanNaeem updating this to remove other workunit nodes from the AST 
    def fix_types(self, entity: PyKokkosEntity, updated_types: UpdatedTypes) -> ast.AST:
        
        check_entity: Callable[[ast.stmt], bool]
        style: PyKokkosStyles = entity.style

        # only supports standalone workunits, return the entity AST as it is
        if style is not PyKokkosStyles.workunit or updated_types == None:
            return entity.AST
        
        check_entity = self.is_workunit
        
        #*1 REMOVING NODES NOT NEEDED FROM AST
        # entity_tree = Union[ast.ClassDef, ast.FunctionDef]
        # # We keep a working tree as nodes will be removed
        # working_tree = deepcopy(self.tree)

        # for node in self.tree.body:
        #     if check_entity(node, self.pk_import):
        #         unit = node
        #         if unit.name != "__init__" and unit.name != updated_types.workunit.__name__:
        #             transformer = RemoveTransformer(unit)
        #             working_tree = transformer.visit(working_tree)
        # del self.tree
        # self.tree = working_tree


        entity_tree: ast.AST = None

        #*2 Changing annotations for the needed workunit definitions
        for node in self.tree.body:

            # At this point there will be only one such node that needs annotation changes
            if check_entity(node, self.pk_import):
                
                if updated_types.workunit.__name__ == node.name:
                    
                    entity_tree = node

                    for arg_obj in node.args.args:
                        if arg_obj.arg in updated_types.inferred_types:
                            update_type = updated_types.inferred_types[arg_obj.arg]

                            # case statements
                            if "int" in update_type:
                                arg_obj.annotation = ast.Name(id=update_type, ctx=ast.Load())
                                
                            
                                #todo expand on this
                            elif "View" in update_type:
                                # update_type = View1D:double
                                view_type, dtype = update_type.split(':')
                                # View1D annotation=
                                # Subscript(
                                #     value=Attribute(
                                #           value=Name(id='pk', ctx=Load()), 
                                #           attr='View1D', 
                                #           ctx=Load()), 
                                #     slice=Attribute(
                                #           value=Name(id='pk', ctx=Load()), 
                                #           attr='double', 
                                #           ctx=Load()), 
                                #     ctx=Load()
                                #     )
                                arg_obj.annotation = ast.Subscript(
                                    value = ast.Attribute(
                                        value = ast.Name(id="pk", ctx=ast.Load()),
                                        attr = view_type,
                                        ctx = ast.Load()
                                    ),
                                    slice = ast.Attribute(
                                        value = ast.Name(id="pk", ctx=ast.Load()),
                                        attr = dtype,
                                        ctx = ast.Load()
                                    ),
                                    ctx = ast.Load()
                                )

                            elif "Acc" in update_type:
                                # update_type = Acc:float
                                dtype = update_type.split(":")[1]
                                # Subscript(
                                #    value=Attribute(
                                #           value=Name(id='pk', ctx=Load()), 
                                #           attr='Acc', 
                                #           ctx=Load()), 
                                #     slice=Name(id='float', ctx=Load())
                                arg_obj.annotation = ast.Subscript(
                                        value = ast.Attribute(
                                            value = ast.Name(id="pk", ctx=ast.Load()),
                                            attr = "Acc",
                                            ctx = ast.Load()
                                    ),
                                    slice = ast.Name(id = dtype, ctx = ast.Load()),
                                    ctx = ast.Load()
                                )
                            
                            elif "pk.TeamMember" in update_type:
                                # Attribute(
                                #   value=Name(
                                #       id='pk', 
                                #       ctx=Load()), 
                                #   attr='TeamMember', 
                                #   ctx=Load())
                                arg_obj.annotation = ast.Attribute(
                                    value = ast.Name(id = "pk", ctx = ast.Load()),
                                    attr = "TeamMember",
                                    ctx = ast.Load()
                                )
                            else:
                                raise ValueError("ERROR: Unsupported type inference")

        # debug print
        # print("[FIX TYPES RETURN] returning: \n", ast.dump(entity_tree), "\n")
        return entity_tree


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
