import ast
import copy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from pykokkos.core.fusion import fuse_workunits
from pykokkos.core.optimizations import loop_fuse, memory_ops_fuse
from pykokkos.core.parsers import Parser, PyKokkosEntity, PyKokkosStyles
from pykokkos.core.translators import PyKokkosMembers, StaticTranslator
from pykokkos.core.type_inference import UpdatedTypes, UpdatedDecorator
from pykokkos.interface import ExecutionSpace
import pykokkos.kokkos_manager as km

from .cpp_setup import CppSetup
from .module_setup import EntityMetadata, ModuleSetup

@dataclass
class CompilationDefaults:
    """
    Holds the default values from compilation
    """

    space: str
    force_uvm: bool

class Compiler:
    """
    Calls the translator and C++ compiler
    """

    def __init__(self):
        # maps from entity metadata to members
        self.members: Dict[str, PyKokkosMembers] = {}

        # caches the result of CppSetup.is_compiled(path)
        self.is_compiled_cache: Dict[str, bool] = {}
        self.parser_cache: Dict[str, Parser] = {}

        self.functor_file: str = "functor.hpp"
        self.functor_cast_file: str = "functor_cast.hpp"
        self.bindings_file: str = "bindings.cpp"
        self.defaults_file: str = "defaults.json"

        loglevel = os.environ.get("PK_LOG_LEVEL", "WARNING")
        numeric_level = getattr(logging, loglevel.upper(), None)
        logging.basicConfig(stream=sys.stdout, level=numeric_level)
        self.logger = logging.getLogger()

    def fuse_objects(self, metadata: List[EntityMetadata], fuse_ASTs: bool, **kwargs) -> Tuple[PyKokkosEntity, List[PyKokkosEntity]]:
        """
        Fuse two or more workunits into one

        :param metadata: the metadata of the workunits to be fused
        :param fuse_ASTs: whether to do the actual fusion of the ASTs, which is expensive
        :returns: the fused entity and all the classtypes it uses
        """

        pyk_classtypes: List[PyKokkosEntity] = []

        # used to track whether two different classtypes in different
        # files use the same name
        pyk_classtype_ids: Dict[str, str] = {}

        names: List[str] = []
        ASTs: List[ast.FunctionDef] = []
        sources: List[Tuple[List[str], int]] = []

        path: str = ""
        full_ASTs: List[ast.Module] = []
        pk_imports: List[str] = []
        for m in metadata:
            parser = self.get_parser(m.path)
            entity: PyKokkosEntity = parser.get_entity(m.name)

            for c in parser.get_classtypes():
                if c.name in pyk_classtype_ids:
                    if c.path != pyk_classtype_ids[c.name]:
                        raise RuntimeError(f"Ambiguous usage of classtype {c.name} in {c.path} and {pyk_classtype_ids[c.name]}")
                else:
                    pyk_classtype_ids[c.name] = c.path
                    pyk_classtypes.append(c)

            path += f"_{m.path}"
            full_ASTs.append(entity.full_AST)
            pk_imports.append(entity.pk_import)

            names.append(entity.name)
            if fuse_ASTs:
                ASTs.append(copy.deepcopy(entity.AST))
            sources.append(entity.source)

        if not all(pk_import == pk_imports[0] for pk_import in pk_imports):
            raise ValueError("Must use same pykokkos import alias for all fused workunits")

        fused_name: str = "_".join(names)
        if fuse_ASTs:
            AST, source = fuse_workunits(fused_name, ASTs, sources, **kwargs)
        else:
            AST = None
            source = None

        entity = PyKokkosEntity(PyKokkosStyles.fused, fused_name, AST, full_ASTs[0], source, None, pk_imports[0])

        return entity, pyk_classtypes


    def compile_object(
        self,
        module_setup: ModuleSetup,
        space: ExecutionSpace,
        force_uvm: bool,
        updated_decorator: Optional[UpdatedDecorator],
        updated_types: Optional[UpdatedTypes],
        types_signature: Optional[str],
        restrict_views: Set[str],
        **kwargs
    ) -> PyKokkosMembers:
        """
        Compile an entity object for a single execution space

        :param module_setup: the module_setup object containing module info
        :param space: the execution space to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :param updated_decorator: Object for decorator specifiers
        :param updated_types: Object with with inferred types
        :param restrict_views: a set of view names that do not alias any other views
        :returns: the PyKokkos members obtained during translation
        """

        metadata: List[EntityMetadata] = module_setup.metadata

        entity: PyKokkosEntity
        classtypes: List[PyKokkosEntity] = []
        parser = self.get_parser(metadata[0].path)

        if len(metadata) == 1:
            entity = parser.get_entity(metadata[0].name)
            classtypes = parser.get_classtypes()
        else:
            # Avoid fusing the ASTs before checking if it was already compiled
            entity, classtypes = self.fuse_objects(metadata, fuse_ASTs=False, **kwargs)

        hash: str = self.members_hash(entity.path, entity.name, types_signature)

        types_inferred: bool = updated_types is not None
        decorator_inferred: bool = updated_decorator is not None

        if types_inferred and entity.style not in {PyKokkosStyles.workunit, PyKokkosStyles.fused}:
            raise Exception(f"Types are required for style: {entity.style}")

        if self.is_compiled(module_setup.output_dir):
            if hash not in self.members: # True if pre-compiled
                if len(metadata) > 1:
                    entity, classtypes = self.fuse_objects(metadata, fuse_ASTs=True, **kwargs)

                if types_inferred:
                    entity.AST = parser.fix_types(entity, updated_types)
                if decorator_inferred:
                    entity.AST = parser.fix_decorator(entity, updated_decorator)
                self.members[hash] = self.extract_members(entity, classtypes)

            return self.members[hash]

        if len(metadata) > 1:
            entity, classtypes = self.fuse_objects(metadata, fuse_ASTs=True, **kwargs)

        self.is_compiled_cache[module_setup.output_dir] = True

        members: PyKokkosMembers

        if types_inferred:
            entity.AST = parser.fix_types(entity, updated_types)
        if decorator_inferred:
            entity.AST = parser.fix_decorator(entity, updated_decorator)

        if hash in self.members: # True if compiled with another execution space
            members = self.members[hash]
        else:
            members = self.extract_members(entity, classtypes)
            self.members[hash] = members

        self.compile_entity(module_setup.main, module_setup, entity, classtypes, space, force_uvm, members, restrict_views)
        return members

    def compile_entity(
        self,
        main: Path,
        module_setup: ModuleSetup,
        entity: PyKokkosEntity,
        classtypes: List[PyKokkosEntity],
        space: ExecutionSpace,
        force_uvm: bool,
        members: PyKokkosMembers,
        restrict_views: Set[str]
    ) -> None:
        """
        Compile the entity

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity being compiled
        :param entity: the parsed entity being compiled
        :param classtypes: the list of parsed classtypes being compiled
        :param space: the execution space to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :param members: the PyKokkos related members of the entity
        :param restrict_views: a set of view names that do not alias any other views
        """

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        if space is ExecutionSpace.Debug:
            return

        if module_setup.is_compiled():
            return

        cpp_setup = CppSetup(module_setup.module_file, module_setup.gpu_module_files)
        translator = StaticTranslator(module_setup.name, self.functor_file,self.functor_cast_file, members)
        t_start: float = time.perf_counter()
        functor: List[str]
        bindings: List[str]
        cast: List[str]

        if entity.style in {PyKokkosStyles.workunit, PyKokkosStyles.fused}:
            if "PK_LOOP_FUSE" in os.environ:
                loop_fuse(entity.AST)
            if "PK_MEM_FUSE" in os.environ:
                memory_ops_fuse(entity.AST, entity.pk_import)
        functor, bindings, cast = translator.translate(entity, classtypes, restrict_views)

        t_end: float = time.perf_counter() - t_start
        self.logger.info(f"translation {t_end}")

        output_dir: Path = module_setup.get_output_dir(main, module_setup.metadata, space, module_setup.types_signature, module_setup.restrict_signature)
        c_start: float = time.perf_counter()
        cpp_setup.compile(output_dir, functor, self.functor_file, cast, self.functor_cast_file, bindings, self.bindings_file, space, force_uvm, self.get_compiler())
        c_end: float = time.perf_counter() - c_start
        self.logger.info(f"compilation {c_end}")

    def compile_raw_source(
        self,
        output_dir: Path,
        source: List[str],
        filename: str,
        module_file: str,
        space: ExecutionSpace,
        force_uvm: bool
        ) -> None:
        """
        Compile the entity

        :param main: the path to the main file in the current PyKokkos application
        :param source: cpp source of module
        :param filename: name of the file to store the source in
        :param space: the execution space to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :param members: the PyKokkos related members of the entity
        """

        cpp_setup = CppSetup(module_file, [])
        c_start: float = time.perf_counter()
        cpp_setup.compile_raw_source(output_dir, source, filename, space, force_uvm, self.get_compiler())
        c_end: float = time.perf_counter() - c_start
        self.logger.info(f"compilation {c_end}")

    def get_compiler(self) -> str:
        """
        Get the compiler to use based on the machine name

        :returns: g++ or nvcc
        """

        from pykokkos.bindings import kokkos

        if kokkos.get_device_available("Cuda"):
            return "nvcc"

        if kokkos.get_device_available("HIP"):
            return "hipcc"

        return "g++"

    def get_defaults_file(self, main: Path) -> Path:
        """
        Get the path to the file that holds the defaults

        :param main: the path to the main file in the current PyKokkos application
        :returns: a Path object of the defaults file
        """

        return ModuleSetup.get_main_dir(main) / self.defaults_file

    def write_defaults(self, main: Path, defaults: CompilationDefaults) -> None:
        """
        Write the defaults dictionary to a file

        :param main: the path to the main file in the current PyKokkos application
        :param defaults: the default values for execution space and force_uvm
        """

        try:
            os.makedirs(ModuleSetup.get_main_dir(main), exist_ok=True)
        except FileExistsError:
            pass

        file: Path = self.get_defaults_file(main)
        with open(file, "w") as f:
            json.dump(defaults.__dict__, f)

    def read_defaults(self) -> Optional[CompilationDefaults]:
        """
        Read the defaults from the defaults file

        :returns: the default values for execution space and force_uvm
        """

        defaults: Optional[CompilationDefaults]
        module_setup = ModuleSetup(None, km.get_default_space())
        main: Path = module_setup.get_main_path()
        file: Path = self.get_defaults_file(main)

        try:
            with open(file, "r") as f:
                defaults = CompilationDefaults(**json.load(f))
        except FileNotFoundError:
            defaults = None

        return defaults

    def members_hash(self, path: List[str], name: str, types_signature: Optional[str]) -> str:
        """
        Map from entity path and name to a string to index members

        :param path: the path to the file containing the entity
        :param name: the name of the entity
        :param types_signature: string signature of inferred parameter types
        :returns: the hash of the entity
        """

        return f"{path}_{name}" if types_signature is None else f"{path}_{name}_{types_signature}"

    def extract_members(self, entity: PyKokkosEntity, classtypes: List[PyKokkosEntity]) -> PyKokkosMembers:
        """
        Extract the PyKokkos members from an entity

        :param entity: the entity being compiled
        :param classtypes: the list of classtypes used in the entity
        :returns: the PyKokkosMembers object
        """

        entity.AST = StaticTranslator.add_parent_refs(entity.AST)

        for c in classtypes:
            c.AST = StaticTranslator.add_parent_refs(c.AST)

        members = PyKokkosMembers()
        members.extract(entity, classtypes)

        return members

    def is_compiled(self, output_dir: str) -> bool:
        """
        Check if the entity is compiled. This caches the result of
        CppSetup.is_compiled() as that requires accessing the
        filesystem, which is costly.

        :param output_dir: the location of the compiled entity
        :returns: True if output_dir exists
        """

        if output_dir in self.is_compiled_cache:
            return self.is_compiled_cache[output_dir]

        is_compiled: bool = CppSetup.is_compiled(output_dir)
        self.is_compiled_cache[output_dir] = is_compiled

        return is_compiled

    def get_parser(self, path: str) -> Parser:
        """
        Get the parser for a particular file

        :param path: the path to the file
        :returns: the Parser object
        """

        if path in self.parser_cache:
            return self.parser_cache[path]

        parser = Parser(path)
        self.parser_cache[path] = parser

        return parser
