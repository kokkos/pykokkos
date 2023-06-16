from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
import sysconfig
import time
from typing import Callable, List, Optional, Union

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
    ):
        """
        ModuleSetup constructor

        :param entity: the functor/workunit/workload
        """

        self.metadata: EntityMetadata
        
        isEntityNone: bool = entity is None
        if isEntityNone:
            self.metadata = EntityMetadata(None, None, None)
        else:
            self.metadata = get_metadata(entity)

        self.space: ExecutionSpace = space

        suffix: Optional[str] = sysconfig.get_config_var("EXT_SUFFIX")
        self.module_file: str = f"kernel{suffix}"

        # The path to the main file if using the console
        self.console_main: str = "pk_console"

        self.main: Path = self.get_main_path()
        self.output_dir: Optional[Path] = self.get_output_dir(self.main, self.metadata, space)
        self.gpu_module_files: List[str] = []
        if km.is_multi_gpu_enabled():
            self.gpu_module_files = [f"kernel{device_id}{suffix}" for device_id in range(km.get_num_gpus())]

        if self.output_dir is not None:
            self.path: str = os.path.join(self.output_dir, self.module_file)
            if km.is_multi_gpu_enabled():
                self.gpu_module_paths: str = [os.path.join(self.output_dir, module_file) for module_file in self.gpu_module_files]

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
        dirname: str = f"{filename}_{metadata.name}"

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