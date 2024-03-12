from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import sys
import sysconfig
from typing import Callable, List, Optional, Set, Union

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
        types_signature: Optional[str] = None,
        restricted_views: Optional[Set[str]] = None
    ):
        """
        ModuleSetup constructor

        :param entity: the functor/workunit/workload or list of workunits for fusion
        :param types_signature: hash/string to identify workunit signature against types
        :param restricted_views: a set of view names that do not alias any other views
        """

        self.metadata: List[EntityMetadata]

        if entity is None:
            self.metadata = [EntityMetadata(None, None, None)]
        elif isinstance(entity, list):
            self.metadata = [get_metadata(e) for e in entity]
        else:
            self.metadata = [get_metadata(entity)]

        self.space: ExecutionSpace = space
        self.types_signature = types_signature
        self.restrict_signature: Optional[str] = None
        if restricted_views is not None:
            self.restrict_signature = hashlib.md5("".join(sorted(restricted_views)).encode()).hexdigest()

        suffix: Optional[str] = sysconfig.get_config_var("EXT_SUFFIX")
        self.module_file: str = f"kernel{suffix}"

        # The path to the main file if using the console
        self.console_main: str = "pk_console"

        self.main: Path = self.get_main_path()
        self.output_dir: Optional[Path] = self.get_output_dir(self.main, self.metadata, space, types_signature, self.restrict_signature)
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
        types_signature: Optional[str] = None,
        restrict_signature: Optional[str] = None
    ) -> Optional[Path]:
        """
        Get the output directory for an execution space

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity or fused entities being compiled
        :param space: the execution space to compile for
        :param types_signature: optional identifier/hash string for types of parameters
        :param restrict_signature: optional identifier/hash string from the views that do not alias any other views
        :returns: the path to the output directory for a specific execution space
        """

        for m in metadata:
            if m.path is None:
                return None

        if space is ExecutionSpace.Default:
            space = km.get_default_space()

        out_dir: Path = self.get_entity_dir(main, metadata)
        if types_signature is not None:
            out_dir = out_dir / f"types_{types_signature}"
        if restrict_signature is not None:
            out_dir = out_dir / f"restrict_{restrict_signature}"

        out_dir = out_dir / space.value

        return out_dir

    def get_entity_dir(self, main: Path, metadata: List[EntityMetadata]) -> Path:
        """
        Get the base output directory for an entity

        :param main: the path to the main file in the current PyKokkos application
        :param metadata: the metadata of the entity or fused entities being compiled
        :returns: the path to the base output directory
        """

        entity_dir: str = ""

        for m in metadata[:5]:
            filename: str = m.path.split("/")[-1].split(".")[0]
            entity_dir += f"{filename}_{m.name}"

        remaining: str = ""
        for m in metadata[5:]:
            filename: str = m.path.split("/")[-1].split(".")[0]
            remaining += f"{filename}_{m.name}"

        if remaining != "":
            entity_dir += hashlib.md5(("".join(remaining)).encode()).hexdigest()

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

        return CppSetup.is_compiled(self.get_output_dir(self.main, self.metadata, self.space, self.types_signature, self.restrict_signature))