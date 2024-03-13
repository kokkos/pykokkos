import ast
import copy
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.optimizations import add_restrict_views
from pykokkos.core.parsers import PyKokkosEntity, PyKokkosStyles
from pykokkos.core.visitors import (
    ClasstypeVisitor, KokkosFunctionVisitor, WorkunitVisitor
)

from .bindings import bind_main, bind_workunits
from .functor import generate_functor
from .functor_cast import generate_cast
from .members import PyKokkosMembers
from .symbols_pass import SymbolsPass

def generate_include_guard_start(symbol_name: str):
    include_guard: str = f"#ifndef {symbol_name}\n"
    include_guard += f"#define {symbol_name}\n"
    return include_guard

def generate_include_guard_end() -> str:
    return "\n#endif"

class StaticTranslator:
    """
    Translates a PyKokkos workload to C++ using static analysis only
    """

    def __init__(self, module: str, functor: str,functor_cast: str, pk_members: PyKokkosMembers):
        """
        StaticTranslator Constructor

        :param module: the name of the compiled Python module
        :param functor: the name of the generated functor file
        :param pk_members: the PyKokkos related members of the entity
        """

        self.pk_import: str

        self.module_file: str = module
        self.functor_file: str = functor
        self.functor_cast: str = functor_cast
        self.pk_members: PyKokkosMembers = pk_members

    def translate(
        self,
        entity: PyKokkosEntity,
        classtypes: List[PyKokkosEntity],
        restrict_views: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Translate an entity into C++ code

        :param entity: the type of the entity being translated
        :param classtypes: the list of classtypes needed by the entity
        :returns: a tuple of lists of strings of representing the functor code and bindings respectively
        """

        self.pk_import = entity.pk_import

        entity.AST = self.add_parent_refs(entity.AST)
        for c in classtypes:
            c.AST = self.add_parent_refs(c.AST)
        for f, AST in self.pk_members.pk_functions.items():
            self.pk_members.pk_functions[f] = self.add_parent_refs(AST)

        # Fusing will rename some symbols so this will not work
        if entity.style is not PyKokkosStyles.fused:
            self.check_symbols(classtypes, entity.path)

        source: Tuple[List[str], int] = entity.source
        functor_name: str = f"pk_functor_{entity.name}"
        classtypes: List[cppast.RecordDecl] = self.translate_classtypes(classtypes, restrict_views)
        functions: List[cppast.MethodDecl] = self.translate_functions(source, restrict_views)

        workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]]
        has_rand_call: bool
        workunits, has_rand_call = self.translate_workunits(source, restrict_views)

        struct: cppast.RecordDecl = generate_functor(functor_name, self.pk_members, workunits, functions, has_rand_call)
        if "PK_RESTRICT" in os.environ:
            for operation, workunit in workunits.values():
                add_restrict_views(struct, operation, workunit, restrict_views)

        cast: List[str] = [self.generate_header(), generate_include_guard_start(functor_name.upper()+"_CAST_"+"_HPP")]
        cast.append(self.generate_cast_includes())
        cast.extend(generate_cast(functor_name,self.pk_members))
        cast.append(generate_include_guard_end())

        bindings: List[str] = self.generate_bindings(entity, functor_name, source, workunits)

        s = cppast.Serializer()
        functor: List[str] = [self.generate_header(), generate_include_guard_start(functor_name.upper()+"_HPP")]
        functor.extend([s.serialize(c) for c in classtypes])
        functor.append(s.serialize(struct))
        functor.append(generate_include_guard_end())

        bindings.insert(0, self.generate_includes())
        bindings.insert(0, self.generate_header())

        return functor, bindings, cast

    @staticmethod
    def add_parent_refs(classdef: ast.ClassDef) -> ast.ClassDef:
        """
        Add references to each node's parent node in classdef

        :param classdef: the classdef being modified
        :returns: the modified classdef
        """

        for node in ast.walk(classdef):
            for child in ast.iter_child_nodes(node):
                child.parent = node

            for field_name, child in ast.iter_fields(node):
                if isinstance(child, ast.AST):
                    child.parent_accessor = field_name
                elif isinstance(child, list):
                    for idx, grand_child in enumerate(child):
                        if isinstance(grand_child, str):
                            continue
                        grand_child.parent_accessor = field_name
                        grand_child.idx_in_parent = idx

        return classdef

    def check_symbols(self, classtypes: List[PyKokkosEntity], path: str) -> None:
        """
        Pass over PyKokkos code and make sure that all symbols are
        valid, printing error messages and exiting if any errors are
        found

        :param classtypes: the list of PyKokkos classtypes
        :param path: the path to the file being translated
        """

        symbols_pass = SymbolsPass(self.pk_members, self.pk_import, path)

        error_messages: List[str] = []
        for AST in self.pk_members.pk_mains.values():
            error_messages.extend(symbols_pass.check_symbols(AST))
        for AST in self.pk_members.pk_workunits.values():
            error_messages.extend(symbols_pass.check_symbols(AST))
        for AST in self.pk_members.pk_functions.values():
            error_messages.extend(symbols_pass.check_symbols(AST))
        for entity in classtypes:
            error_messages.extend(symbols_pass.check_symbols(entity.AST))

        if error_messages:
            for error in error_messages:
                print(error)

            sys.exit()


    def translate_classtypes(self, classtypes: List[PyKokkosEntity], restrict_views: Set[str]) -> List[cppast.RecordDecl]:
        """
        Translate all classtypes, i.e. classes that the workload uses internally

        :param classtypes: the list of classtypes needed by the workload
        :param restrict_views: the views with the restrict keyword
        :returns: a list of strings of translated source code
        """

        declarations: List[cppast.RecordDecl] = []
        definitions: List[cppast.RecordDecl] = []

        for c in classtypes:
            classdef: ast.ClassDef = c.AST
            source: Tuple[List[str], int] = c.source

            node_visitor = ClasstypeVisitor(
                {},
                source, self.pk_members.views, self.pk_members.pk_workunits, self.pk_members.fields,
                self.pk_members.pk_functions, self.pk_members.classtype_methods, self.pk_import, restrict_views, debug=True
            )

            definition: cppast.RecordDecl = node_visitor.visit(classdef)
            declaration = copy.deepcopy(definition)
            declaration.is_definition = False

            definitions.append(definition)
            declarations.append(declaration)

        return declarations + definitions

    def translate_functions(self, source: Tuple[List[str], int], restrict_views: Set[str]) -> List[cppast.MethodDecl]:
        """
        Translate all PyKokkos functions

        :param source: the python source code of the workload
        :param restrict_views: the views with the restrict keyword
        :returns: a list of method declarations
        """

        # The visitor might add views declared as parameters
        views = copy.deepcopy(self.pk_members.views)

        node_visitor = KokkosFunctionVisitor(
            {},
            source, views, self.pk_members.pk_workunits, self.pk_members.fields, self.pk_members.pk_functions,
            self.pk_members.classtype_methods, self.pk_import, restrict_views, debug=True)

        translation: List[cppast.MethodDecl] = []

        for functiondef in self.pk_members.pk_functions.values():
            translation.append(node_visitor.visit(functiondef))

        return translation

    def translate_workunits(self, source: Tuple[List[str], int], restrict_views: Set[str]) -> Tuple[Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]], bool]:
        """
        Translate the workunits

        :param source: the python source code of the workload
        :param restrict_views: the views with the restrict keyword
        :returns: a tuple of a dictionary mapping from workload name
            to a tuple of operation name and source, and a boolean
            indicating whether the workunit has a call to pk.rand()
        """

        node_visitor = WorkunitVisitor(
            {}, source, self.pk_members.views, self.pk_members.pk_workunits,
            self.pk_members.fields, self.pk_members.pk_functions,
            self.pk_members.classtype_methods, self.pk_import, restrict_views, debug=True)

        workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]] = {}

        has_rand_call: bool = False
        for n, w in self.pk_members.pk_workunits.items():
            try:
                workunits[n] = node_visitor.visit(w)
                has_rand_call = has_rand_call or node_visitor.has_rand_call
                if node_visitor.has_rand_call:
                    workunit: cppast.MethodDecl = workunits[n][1]
                    self.add_rand_pool_state(workunit)
                    node_visitor.has_rand_call = False
            except:
                print(f"Translation of {w} {w.name} failed")
                sys.exit(1)

        return workunits, has_rand_call

    def generate_header(self) -> str:
        """
        Generate the commented header at the top of the C++ source file

        :returns: the header as a string
        """

        return "// ******* AUTOMATICALLY GENERATED BY PyKokkos *******"

    def generate_includes(self) -> str:
        """
        Generate the list of include statements

        :returns: the includes as a string
        """

        headers: List[str] = [
            "pybind11/pybind11.h",
            "Kokkos_Core.hpp",
            "Kokkos_Random.hpp",
            "Kokkos_Sort.hpp",
            "fstream",
            "iostream",
            "cmath",
            self.functor_file,
            self.functor_cast
        ]
        headers = [f"#include <{h}>\n" for h in headers]

        return "".join(headers)

    def generate_cast_includes(self) -> str:
        """
        Generate the list of include statements for the cast header file

        :returns: the includes as a string
        """

        headers: List[str] = [
            "pybind11/pybind11.h",
            "Kokkos_Core.hpp",
            "Kokkos_Random.hpp",
            "Kokkos_Sort.hpp",
            "fstream",
            "iostream",
            "cmath",
            self.functor_file
        ]
        headers = [f"#include <{h}>\n" for h in headers]

        return "".join(headers)

    def generate_bindings(
        self,
        entity: PyKokkosEntity,
        functor_name: str,
        source: Tuple[List[str], int],
        workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]]
    ) -> List[str]:
        """
        Generate the pybind bindings for a single real precision

        :param entity: the type of the entity being translated
        :param functor_name: the name of the functor
        :param workunits: the translated workunits
        :returns: the source as a list of strings
        """

        bindings: List[str]
        if entity.style is PyKokkosStyles.workload:
            bindings = bind_main(functor_name, self.pk_members, source, self.pk_import, self.module_file)
        else:
            bindings = bind_workunits(functor_name, self.pk_members, workunits, self.module_file)

        return bindings

    def add_rand_pool_state(self, workunit: cppast.MethodDecl) -> None:
        """
        Generate code to initialize and free the random pool state in
        a workunit

        :param workunit: the workunit containing a call to pk.rand()
        """

        random_pool: Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]] = self.pk_members.random_pool

        pool_type: str = random_pool[1].typename
        generator_type = cppast.ClassType(f"Kokkos::{pool_type}<>::generator_type")

        pool_state_name = cppast.DeclRefExpr(Keywords.RandPoolState.value)

        rand_pool_name: cppast.DeclRefExpr = random_pool[0]
        pool_value = cppast.MemberCallExpr(rand_pool_name, cppast.DeclRefExpr("get_state"), [])

        init_pool = cppast.DeclStmt(cppast.VarDecl(generator_type, pool_state_name, pool_value))

        free_pool = cppast.CallStmt(cppast.MemberCallExpr(rand_pool_name, cppast.DeclRefExpr("free_state"), [pool_state_name]))

        body: cppast.CompoundStmt = workunit.body
        body.statements.insert(0, init_pool)
        body.statements.append(free_pool)
