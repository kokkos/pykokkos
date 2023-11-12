from dataclasses import dataclass
import hashlib
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

    name: str
    filepath: str

    if isinstance(entity, Callable):
        # Workunit/functor
        is_functor: bool = hasattr(entity, "__self__")
        if is_functor:
            entity_type: type = get_functor(entity)
            name = entity_type.__name__
            filepath = inspect.getfile(entity_type)
        else:
            name = entity.__name__
            filepath = inspect.getfile(entity)

    else:
        # Workload
        entity_type: type = type(entity)
        name = entity_type.__name__
        filepath = inspect.getfile(entity_type)

    return EntityMetadata(entity, name, filepath)


class ModuleSetup:
    """
    For a given workunit, selects the name and path to the compiled
    module
    """

    def __init__(
        self,
        entity: Union[Callable[..., None], type, List[Callable[..., None]]],
        space: ExecutionSpace,
        types_signature: Optional[str] = None
    ):
        """
        ModuleSetup constructor

        :param entity: the functor/workunit/workload or list of workunits for fusion
        :param types_signature: hash/string to identify workunit signature against types
        """

        self.metadata: List[EntityMetadata]

        if entity is None:
            self.metadata = [EntityMetadata(None, None, None)]
        elif isinstance(entity, list):
            self.metadata = [get_metadata(e) for e in entity]
        else:
            self.metadata = [get_metadata(entity)]

        print(entity)

        self.space: ExecutionSpace = space
        self.types_signature = types_signature

        suffix: Optional[str] = sysconfig.get_config_var("EXT_SUFFIX")
        self.module_file: str = f"kernel{suffix}"

        # The path to the main file if using the console
        self.console_main: str = "pk_console"

        self.main: Path = self.get_main_path()
        self.output_dir: Optional[Path] = self.get_output_dir(self.main, self.metadata, space, types_signature)
        self.gpu_module_files: List[str] = []
        if km.is_multi_gpu_enabled():
            self.gpu_module_files = [f"kernel{device_id}{suffix}" for device_id in range(km.get_num_gpus())]

        if self.output_dir is not None:
            self.path: str = os.path.join(self.output_dir, self.module_file)
            if km.is_multi_gpu_enabled():
                self.gpu_module_paths: str = [os.path.join(self.output_dir, module_file) for module_file in self.gpu_module_files]

            self.name: str = hashlib.sha256(self.path.encode()).hexdigest()

    def get_output_dir(
        self,
        main: Path,
        metadata: List[EntityMetadata],
        space: ExecutionSpace,
        types_signature: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get the output directory for an execution space

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity or fused entities being compiled
        :param space: the execution space to compile for
        :param types_signature: optional identifier/hash string for types of parameters
        :returns: the path to the output directory for a specific execution space
        """

        for m in metadata:
            if m.path is None:
                return None

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        out_dir: Path = self.get_entity_dir(main, metadata) / space.value
        if types_signature is not None:
            out_dir: Path = self.get_entity_dir(main, metadata) / types_signature / space.value
        return out_dir

    def get_entity_dir(self, main: Path, metadata: List[EntityMetadata]) -> Path:
        """
        Get the base output directory for an entity

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity or fused entities being compiled
        :returns: the path to the base output directory
        """

        entity_dir: str = ""

        for m in metadata:
            filename: str = m.path.split("/")[-1].split(".")[0]
            entity_dir += f"{filename}_{m.name}"

        return self.get_main_dir(main) / Path(entity_dir)

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

        return CppSetup.is_compiled(self.get_output_dir(self.main, self.metadata, self.space, self.types_signature))