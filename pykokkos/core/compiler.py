from dataclasses import dataclass
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import sys
import sysconfig
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from pykokkos.core.parsers import Parser, PyKokkosEntity, PyKokkosStyles
from pykokkos.core.translators import PyKokkosMembers, StaticTranslator
from pykokkos.interface import ExecutionSpace
import pykokkos.kokkos_manager as km

from .cpp_setup import CppSetup

@dataclass
class EntityMetadata:
    """
    Contains metadata about the functor or workunit
    """

    entity: Union[Callable[..., None], type, None]
    name: str
    path: str

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
        self.base_dir: str = "pk_cpp"

        # maps from entity metadata to members
        self.members: Dict[str, PyKokkosMembers] = {}
        self.module_name: str = "kernel"
        suffix: Optional[str] = sysconfig.get_config_var("EXT_SUFFIX")
        self.module_file: str = f"{self.module_name}{suffix}"

        # The path to the main file if using the console
        self.console_main: str = "pk_console"

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
            metadata = EntityMetadata(None, e.name.declname, e.path)
            self.compile_entity(main, metadata, e, classtypes, spaces, force_uvm)

    def compile_object(
        self,
        entity_object: Union[object, Callable[..., None]],
        space: ExecutionSpace,
        force_uvm: bool
    ) -> Tuple[str, Optional[PyKokkosMembers]]:
        """
        Compile an entity object for a single execution space

        :param entity_object: the entity object
        :param space: the execution space to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :returns: the path to the compiled module and PyKokkos members obtained during translation
        """

        main: Path = self.get_main_path()
        metadata: EntityMetadata = self.get_metadata(entity_object)
        output_dir: Path = self.get_output_dir(main, metadata, space)
        module_path: str = os.path.join(output_dir, self.module_file)

        hash: str = self.members_hash(metadata.path, metadata.name)
        if CppSetup.is_compiled(output_dir):
            if hash not in self.members: # True if pre-compiled
                self.members[hash] = self.extract_members(metadata)

            return (module_path, self.members[hash])

        parser = Parser(metadata.path)
        entity: PyKokkosEntity = parser.get_entity(metadata.name)
        self.compile_entity(main, metadata, entity, parser.get_classtypes(), [space], force_uvm)

        members: PyKokkosMembers = self.extract_members(metadata)
        self.members[hash] = members

        return (module_path, members)

    def get_main_path(self) -> Path:
        """
        Get the path to the main file

        :returns: a Path object to the main file
        """

        if hasattr(sys.modules["__main__"], "__file__"):
            path: str = sys.modules["__main__"].__file__
            path = path[:-3] # remove the .py extensions
            return Path(path)

        return Path(self.console_main)

    def compile_entity(
        self,
        main: Path,
        metadata: EntityMetadata,
        entity: PyKokkosEntity,
        classtypes: List[PyKokkosEntity],
        spaces: List[ExecutionSpace],
        force_uvm: bool
    ) -> None:
        """
        Compile the entity

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity being compiled
        :param entity: the parsed entity being compiled
        :param classtypes: the list of parsed classtypes being compiled
        :param spaces: the list of execution spaces to compile for
        :param force_uvm: whether CudaUVMSpace is enabled
        :returns: the PyKokkos members obtained during translation
        """

        cpp_setup = CppSetup(self.module_file, self.functor_file, self.bindings_file)
        translator = StaticTranslator(self.module_name, self.functor_file)

        spaces = [km.get_default_space() if s is ExecutionSpace.Default else s for s in spaces]
        is_compiled: List[bool] = [CppSetup.is_compiled(self.get_output_dir(main, metadata, s)) for s in spaces]

        if not all(is_compiled):
            t_start: float = time.perf_counter()
            functor: List[str]
            bindings: List[str]
            functor, bindings = translator.translate(entity, classtypes)
            t_end: float = time.perf_counter() - t_start
            self.logger.info(f"translation {t_end}")

            for i, s in enumerate(spaces):
                if s is ExecutionSpace.Debug:
                    continue

                if not is_compiled[i]:
                    output_dir: Path = self.get_output_dir(main, metadata, s)
                    c_start: float = time.perf_counter()
                    cpp_setup.compile(output_dir, functor, bindings, s, force_uvm, self.get_compiler(s))
                    c_end: float = time.perf_counter() - c_start
                    self.logger.info(f"compilation {c_end}")

    def get_compiler(self, space: ExecutionSpace) -> str:
        """
        Get the compiler to use based on the machine name

        :param machine: the name of the machine
        :param space: the execution space
        :returns: g++ or nvcc
        """

        if space is ExecutionSpace.Cuda:
            return "nvcc"

        return "g++"

    def get_output_dir(self, main: Path, metadata: EntityMetadata, space: ExecutionSpace) -> Path:
        """
        Get the output directory for an execution space

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity being compiled
        :param space: the execution space to compile for
        :returns: the path to the output directory for a specific execution space
        """

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        return self.get_entity_dir(main, metadata) / space.value

    def get_entity_dir(self, main: Path, metadata: EntityMetadata) -> Path:
        """
        Get the base output directory for an entity

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity being compiled
        :returns: the path to the base output directory
        """

        filename: str = metadata.path.split("/")[-1].split(".")[0]
        dirname: str = f"{filename}_{metadata.name}"

        return self.get_main_dir(main) / Path(dirname)

    def get_main_dir(self, main: Path) -> Path:
        """
        Get the main directory for an application from the path to the main file

        :param main: the path to the main file in the current PyKokkos application
        :returns: the path to the main directory
        """

        # If the parent directory is root, remove it so we can
        # concatenate it to pk_cpp
        main_path: Path = main
        if str(main).startswith("/"):
            main_path = Path(str(main)[1:])

        return Path(self.base_dir) / main_path

    def get_metadata(self, entity: Union[Callable[..., None], object]) -> EntityMetadata:
        """
        Gets the name and filepath of an entity

        :param entity: the workload or workunit function object
        :returns: an EntityMetadata object
        """

        entity_type: Union[Callable[..., None], type]

        if isinstance(entity, Callable):
            is_functor: bool = hasattr(entity, "__self__")
            if is_functor:
                entity_type = self.get_functor(entity)
            else:
                entity_type = entity
        else:
            entity_type = type(entity)

        name: str = entity_type.__name__
        filepath: str = inspect.getfile(entity_type)

        return EntityMetadata(entity, name, filepath)

    def get_functor(self, workunit: Callable[..., None]) -> type:
        """
        Get the functor that the workunit belongs to

        :param workunit: the workunit function object
        :returns: the functor's type
        """

        for p in inspect.getmro(workunit.__self__.__class__):
            if workunit.__name__ in p.__dict__:
                return p

        raise ValueError(workunit, "Method does not exist")

    def get_defaults_file(self, main: Path) -> Path:
        """
        Get the path to the file that holds the defaults

        :param main: the path to the main file in the current PyKokkos application
        :returns: a Path object of the defaults file
        """

        return self.get_main_dir(main) / self.defaults_file

    def write_defaults(self, main: Path, defaults: CompilationDefaults) -> None:
        """
        Write the defaults dictionary to a file

        :param main: the path to the main file in the current PyKokkos application
        :param defaults: the default values for execution space and force_uvm
        """

        try:
            os.makedirs(self.get_main_dir(main), exist_ok=True)
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
        main: Path = self.get_main_path()
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

        :param entity: the metadata of the entity being compiled
        :returns: the PyKokkosMembers object
        """

        parser = Parser(metadata.path)
        entity: PyKokkosEntity = parser.get_entity(metadata.name)

        translator = StaticTranslator(self.module_name, self.functor_file)

        translator.translate(entity, parser.get_classtypes())
        members: PyKokkosMembers = translator.pk_members

        return members