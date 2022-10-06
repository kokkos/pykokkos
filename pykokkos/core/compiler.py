from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

from pykokkos.core.parsers import Parser, PyKokkosEntity
from pykokkos.core.translators import PyKokkosMembers, StaticTranslator
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
        self.bindings_file: str = "bindings.cpp"
        self.defaults_file: str = "defaults.json"

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.logger = logging.getLogger()

    def compile_sources(
        self,
        main: Path,
        sources: List[str],
        spaces: List[ExecutionSpace],
        force_uvm: bool,
        defaults: CompilationDefaults,
        verbose: bool
    ) -> None:
        """
        Compile all entities in every source file

        :param main: the path to the main file in the current PyKokkos application
        :param sources: the list of paths to source files
        :param spaces: the list of execution spaces to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :param defaults: the default values for execution space and force_uvm
        :param verbose: print files discovered
        """

        # Remove the .py suffix
        main = main.with_suffix("")
        self.write_defaults(main, defaults)

        for path in sources:
            parser = Parser(path)
            
            self.logger.info("Path %s", path)
            self.logger.info("%d workloads", len(parser.workloads))
            self.logger.info("%d functors", len(parser.functors))
            self.logger.info("%d workunits", len(parser.workunits))
            self.logger.info("%d classtypes", len(parser.classtypes))

            classtypes: List[PyKokkosEntity] = parser.get_classtypes()
            self.compile_entities(main, parser.workloads, classtypes, spaces, force_uvm, verbose)
            self.compile_entities(main, parser.functors, classtypes, spaces, force_uvm, verbose)
            self.compile_entities(main, parser.workunits, classtypes, spaces, force_uvm, verbose)
    
    def compile_entities(
        self,
        main: Path,
        entities: Dict[str, PyKokkosEntity],
        classtypes: List[PyKokkosEntity],
        spaces: List[ExecutionSpace],
        force_uvm: bool,
        verbose: bool
    ) -> None:
        """
        Compile all entities in a single source file

        :param main: the path to the main file in the current PyKokkos application
        :param entities: a dictionary of entities
        :param classtypes: the list of parsed classtypes being compiled
        :param spaces: the list of execution spaces to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :param verbose: print entities discovered
        """

        for e in entities.values():
            for space in spaces:
                module_setup = ModuleSetup(None, space, e.name.declname, e.path)
                self.compile_entity(main, module_setup, e, classtypes, space, force_uvm)

    def compile_object(
        self,
        module_setup: ModuleSetup,
        space: ExecutionSpace,
        force_uvm: bool
    ) -> Optional[PyKokkosMembers]:
        """
        Compile an entity object for a single execution space

        :param entity_object: the module_setup object containing module info
        :param space: the execution space to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :returns: the PyKokkos members obtained during translation
        """

        metadata = module_setup.metadata

        hash: str = self.members_hash(metadata.path, metadata.name)
        if self.is_compiled(module_setup.output_dir):
            if hash not in self.members: # True if pre-compiled
                self.members[hash] = self.extract_members(metadata)

            return self.members[hash]

        self.is_compiled_cache[module_setup.output_dir] = True

        parser = self.get_parser(metadata.path)
        entity: PyKokkosEntity = parser.get_entity(metadata.name)

        members: PyKokkosMembers
        if hash in self.members: # True if compiled with another execution space
            members = self.members[hash]
        else:
            members = self.extract_members(metadata)
            self.members[hash] = members

        self.compile_entity(module_setup.main, module_setup, entity, parser.get_classtypes(), space, force_uvm, members)

        return members

    def compile_entity(
        self,
        main: Path,
        module_setup: ModuleSetup,
        entity: PyKokkosEntity,
        classtypes: List[PyKokkosEntity],
        space: ExecutionSpace,
        force_uvm: bool,
        members: PyKokkosMembers
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
        """

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        if space is ExecutionSpace.Debug:
            return

        if module_setup.is_compiled():
            return

        cpp_setup = CppSetup(module_setup.module_file, module_setup.gpu_module_files, self.functor_file, self.bindings_file)
        translator = StaticTranslator(module_setup.name, self.functor_file, members)

        t_start: float = time.perf_counter()
        functor: List[str]
        bindings: List[str]
        functor, bindings = translator.translate(entity, classtypes)
        t_end: float = time.perf_counter() - t_start
        self.logger.info(f"translation {t_end}")

        output_dir: Path = module_setup.get_output_dir(main, module_setup.metadata, space)
        c_start: float = time.perf_counter()
        cpp_setup.compile(output_dir, functor, bindings, space, force_uvm, self.get_compiler())
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

    def members_hash(self, path: str, name: str) -> str:
        """
        Map from entity path and name to a string to index members

        :param path: the path to the file containing the entity
        :param name: the name of the entity
        :returns: the hash of the entity
        """

        return f"{path}_{name}"

    def extract_members(self, metadata: EntityMetadata) -> PyKokkosMembers:
        """
        Extract the PyKokkos members from an entity

        :param module_name: the name of the module being compiled
        :param metadata: the metadata of the entity being compiled
        :returns: the PyKokkosMembers object
        """

        parser = self.get_parser(metadata.path)
        entity: PyKokkosEntity = parser.get_entity(metadata.name)

        entity.AST = StaticTranslator.add_parent_refs(entity.AST)
        classtypes = parser.get_classtypes()
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