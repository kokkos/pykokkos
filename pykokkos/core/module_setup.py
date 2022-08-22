from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
import sysconfig
import time
from typing import Callable, Optional, Union

from pykokkos.interface import ExecutionSpace
import pykokkos.kokkos_manager as km

from .cpp_setup import CppSetup

BASE_DIR: str = "pk_cpp"


@dataclass
class EntityMetadata:
    """
    Contains metadata about the functor or workunit
    """

    entity: Union[Callable[..., None], type, None]
    name: str # the name of the functor/workunit
    path: str # the path to the file containing the entity


def get_functor(workunit: Callable[..., None]) -> type:
    """
    Get the functor that the workunit belongs to

    :param workunit: the workunit function object
    :returns: the functor's type
    """

    for p in inspect.getmro(workunit.__self__.__class__):
        if workunit.__name__ in p.__dict__:
            return p

    raise ValueError(workunit, "Method does not exist")


def get_metadata(entity: Union[Callable[..., None], object]) -> EntityMetadata:
    """
    Gets the name and filepath of an entity

    :param entity: the workload or workunit function object
    :returns: an EntityMetadata object
    """

    entity_type: Union[Callable[..., None], type]

    if isinstance(entity, Callable):
        is_functor: bool = hasattr(entity, "__self__")
        if is_functor:
            entity_type = get_functor(entity)
        else:
            entity_type = entity
    else:
        entity_type = type(entity)

    name: str = entity_type.__name__
    filepath: str = inspect.getfile(entity_type)

    return EntityMetadata(entity, name, filepath)


class ModuleSetup:
    """
    For a given workunit, selects the name and path to the compiled
    module
    """

    def __init__(
        self,
        entity: Union[Callable[..., None], type, None],
        space: ExecutionSpace,
        entity_name: Optional[str]=None,
        entity_path: Optional[str]=None
    ):
        """
        ModuleSetup constructor

        :param entity: the functor/workunit/workload
        :param entity_name: optionally provide the entity name (only set for pkc)
        :param entity_path: optionally provide the entity path (only set for pkc)
        """

        self.metadata: EntityMetadata
        
        for_pkc: bool = entity is None
        if for_pkc:
            self.metadata = EntityMetadata(None, entity_name, entity_path)
        else:
            self.metadata = get_metadata(entity)

        self.space: ExecutionSpace = space

        suffix: Optional[str] = sysconfig.get_config_var("EXT_SUFFIX")
        self.module_file: str = f"kernel{suffix}"

        # The path to the main file if using the console
        self.console_main: str = "pk_console"

        self.main: Path = self.get_main_path()
        self.output_dir: Optional[Path] = self.get_output_dir(self.main, self.metadata, space)

        if self.output_dir is not None:
            self.path: str = os.path.join(self.output_dir, self.module_file)
            self.name: str = self.path.replace("/", "_")
            self.name: str = self.name.replace("-", "_")
            self.name: str = self.name.replace(".", "_")

    def get_output_dir(self, main: Path, metadata: EntityMetadata, space: ExecutionSpace) -> Optional[Path]:
        """
        Get the output directory for an execution space

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity being compiled
        :param space: the execution space to compile for
        :returns: the path to the output directory for a specific execution space
        """

        if metadata.path is None:
            return None

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
        # the compilation paths do not have concurrent safety without a unique
        # identifier because i.e., compilation units can share
        # the same module/class; try using the memory loc of the Python
        # metadata object
        mem_id = id(metadata)
        dirname: str = f"{filename}_{metadata.name}_{mem_id}"

        return self.get_main_dir(main) / Path(dirname)

    @staticmethod
    def get_main_dir(main: Path) -> Path:
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

        return Path(BASE_DIR) / main_path

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

    def is_compiled(self) -> bool:
        """
        Check if this module is compiled for its execution space
        """

        return CppSetup.is_compiled(self.get_output_dir(self.main, self.metadata, self.space))
