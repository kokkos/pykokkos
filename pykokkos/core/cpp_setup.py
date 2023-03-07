import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import List, Tuple

from pykokkos.interface import (
    ExecutionSpace, get_default_layout, get_default_memory_space,
    is_host_execution_space
)
import pykokkos.kokkos_manager as km


class CppSetup:
    """
    Creates the directory to hold the translation and invokes the compiler
    """

    def __init__(self, module_file: str, gpu_module_files: List[str]):
        """
        CppSetup constructor

        :param module: the name of the file containing the compiled Python module
        :param gpu_module_files: the list of names of files containing for each gpu module
        """

        self.module_file: str = module_file
        self.gpu_module_files: List[str] = gpu_module_files

        self.script: str = "compile.sh"
        self.script_path: Path = Path(__file__).resolve().parent / self.script

        self.lib_path_env: str = "PK_KOKKOS_LIB_PATH"

        self.format: bool = False

    def compile_raw_source(
        self,
        output_dir: Path,
        source: List[str],
        filename: str,
        space: ExecutionSpace,
        enable_uvm: bool,
        compiler: str
    ) -> None:
        """
        Compiles the generated C++ code

        :param output_dir: the base directory
        :param source: the translated C++ source
        :param filename: the name the source is written to
        :param space: the execution space to compile for
        :param enable_uvm: whether to enable CudaUVMSpace
        :param compiler: the compiler name
        """

        self.initialize_directory(output_dir)
        self.write_raw_source(output_dir, source, filename)
        self.copy_script(output_dir)
        self.invoke_script(output_dir, space, enable_uvm, compiler)

    def compile(
        self,
        output_dir: Path,
        functor: List[str],
        functor_filename: str,
        functor_cast: List[str],
        functor_cast_filename: str,
        bindings: List[str],
        bindings_filename: str,
        space: ExecutionSpace,
        enable_uvm: bool,
        compiler: str
    ) -> None:
        """
        Compiles the generated C++ code

        :param output_dir: the base directory
        :param functor: the translated C++ functor
        :param functor_filename: the generated C++ functor filename
        :param functor_cast: the generated C++ functor_cast
        :param functor_cast_filename: the generated C++ functor_cast filename
        :param bindings: the generated bindings
        :param bindings_filename: the generated bindings_filename
        :param space: the execution space to compile for
        :param enable_uvm: whether to enable CudaUVMSpace
        """

        self.initialize_directory(output_dir)
        self.write_source(output_dir, functor,functor_filename, functor_cast, functor_cast_filename, bindings, bindings_filename)
        self.copy_script(output_dir)
        self.invoke_script(output_dir, space, enable_uvm, compiler)
        if space in {ExecutionSpace.Cuda, ExecutionSpace.HIP} and km.is_multi_gpu_enabled():
            self.copy_multi_gpu_kernel(output_dir)


    def initialize_directory(self, name: Path) -> None:
        """
        Creates an output directory, overwriting an existing directory with the same name

        :param name: the name of the directory
        """

        try:
            shutil.rmtree(name)
        except OSError:
            pass

        try:
            os.makedirs(name, exist_ok=True)
        except FileExistsError:
            pass

    def write_source(self, output_dir: Path, functor: List[str], functor_filename: str ,functor_cast: List[str], functor_cast_filename: str, bindings: List[str],bindings_filename: str) -> None:
        """
        Writes the generated C++ source code to a file

        :param output_dir: the base directory
        :param functor: the generated C++ functor
        :param functor_filename: the generated C++ functor filename
        :param functor_cast: the generated C++ functor_cast
        :param functor_cast_filename: the generated C++ functor_cast filename
        :param bindings: the generated bindings
        :param bindings_filename: the generated bindings_filename
        """

        self.write_raw_source(output_dir.parent,functor,functor_filename)
        self.write_raw_source(output_dir.parent,functor_cast,functor_cast_filename)
        self.write_raw_source(output_dir,bindings,bindings_filename)


    def write_raw_source(self, output_dir: Path, source: List[str], filename: str) -> None:
        """
        Writes the generated C++ source code to a file

        :param output_dir: the base directory
        :param source: the generated C++ source file content
        :param filename: the filename for the code
        """

        file_path: Path = output_dir / filename

        with open(file_path, "w") as out:
            out.write("\n".join(source))

        if self.format:
            try:
                subprocess.run(["clang-format", "-i", file_path])
            except Exception as ex:
                print(f"Exception while formatting cpp: {ex}")

    def copy_script(self, output_dir: Path) -> None:
        """
        Copy the compilation script to the output directory

        :param output_dir: the base directory
        """

        file_path: Path = output_dir / "compile.sh"
        try:
            shutil.copy(self.script_path, file_path)
        except Exception as ex:
            print(f"Exception while copying views and makefile: {ex}")
            sys.exit(1)

    def get_kokkos_paths(self, space: ExecutionSpace, compiler: str) -> Tuple[Path, Path, Path]:
        """
        Get the paths of the Kokkos instal lib and include
        directories. If the environment variable is set, use that
        Kokkos install. If not, fall back to the installed
        pykokkos-base package.

        :param space: the execution space to compile for
        :param compiler: what compiler to use
        :returns: a tuple of paths to the Kokkos lib/, include/,
            and compiler to be used
        """

        lib_path: Path
        include_path: Path
        if self.lib_path_env in os.environ:
            lib_path = Path(os.environ.get(self.lib_path_env))
            if not lib_path.is_dir():
                raise RuntimeError(f"lib/ directory path {str(lib_path)} does not exist")

            include_path = lib_path.parent / "include"
            if not include_path.is_dir():
                raise RuntimeError(f"install/ directory path {str(include_path)} does not exist")

            compiler_path: Path
            if compiler != "nvcc":
                compiler_path = Path("g++")
            else:
                compiler_path = lib_path.parent / "bin/nvcc_wrapper"

            return lib_path, include_path, compiler_path

        is_cpu: bool = is_host_execution_space(space)
        kokkos_lib: ModuleType = km.get_kokkos_module(is_cpu)
        install_path = Path(kokkos_lib.__path__[0])
        lib_parent_path: Path
        if km.is_multi_gpu_enabled():
            lib_parent_path = install_path
        else:
            lib_parent_path = install_path.parent

        if (lib_parent_path / "lib").is_dir():
            lib_path = lib_parent_path / "lib"
        elif (lib_parent_path / "lib64").is_dir():
            lib_path = lib_parent_path / "lib64"
        else:
            raise RuntimeError("lib/ or lib64/ directories not found in installed pykokkos-base package."
                               f" Try setting {self.lib_path_env} instead.")

        include_path = install_path.parent / "include/kokkos"

        compiler_path: Path
        if compiler != "nvcc":
            compiler_path = Path(compiler)
        else:
            compiler_path = install_path.parent / "bin/nvcc_wrapper"

        return lib_path, include_path, compiler_path

    def get_kokkos_lib_suffix(self, space: ExecutionSpace) -> str:
        """
        Get the suffix of the libkokkoscore and libkokkoscontainers
        libraries corresponding to the enabled device

        :param space: the execution space to compile for
        :returns: the suffix as a string
        """

        if is_host_execution_space(space) or not km.is_multi_gpu_enabled():
            return ""

        return f"_{km.get_device_id()}"

    def invoke_script(self, output_dir: Path, space: ExecutionSpace, enable_uvm: bool, compiler: str) -> None:
        """
        Invoke the compilation script

        :param output_dir: the base directory
        :param space: the execution space of the workload
        :param enable_uvm: whether to enable CudaUVMSpace
        :param compiler: what compiler to use
        """

        view_space: str = "Kokkos::HostSpace"
        if space is ExecutionSpace.Cuda:
            if enable_uvm:
                view_space = "Kokkos::CudaUVMSpace"
        if space is ExecutionSpace.HIP:
            if enable_uvm:
                view_space = "Kokkos::Experimental::HIPManagedSpace"

        space_value: str
        if space.value == "HIP":
            space_value = "Experimental::HIP"
        else:
            space_value = space.value

        view_layout: str = str(get_default_layout(get_default_memory_space(space)))
        view_layout = view_layout.split(".")[-1]
        view_layout = f"Kokkos::{view_layout}"

        precision: str = km.get_default_precision().__name__.split(".")[-1]
        lib_path: Path
        include_path: Path
        compiler_path: Path
        lib_path, include_path, compiler_path = self.get_kokkos_paths(space, compiler)
        compute_capability: str = self.get_cuda_compute_capability(compiler)
        lib_suffix: str = self.get_kokkos_lib_suffix(space)

        command: List[str] = [f"./{self.script}",
                              compiler,             # What compiler to use
                              self.module_file,     # Compilation target
                              space_value,          # Execution space
                              view_space,           # Argument views memory space
                              view_layout,          # Argument views memory layout
                              precision,            # Default real precision
                              str(lib_path),        # Path to Kokkos install lib/ directory
                              str(include_path),    # Path to Kokkos install include/ directory
                              compute_capability,   # Device compute capability
                              lib_suffix,           # The libkokkos* suffix identifying the gpu
                              str(compiler_path)]   # The path to the compiler to use
        compile_result = subprocess.run(command, cwd=output_dir, capture_output=True, check=False)

        if compile_result.returncode != 0:
            print(compile_result.stderr.decode("utf-8"))
            print(f"C++ compilation in {output_dir} failed")
            sys.exit(1)

        patchelf: List[str] = ["patchelf",
                               "--set-rpath",
                               str(lib_path),
                               self.module_file]

        patchelf_result = subprocess.run(patchelf, cwd=output_dir, capture_output=True, check=False)
        if patchelf_result.returncode != 0:
            print(patchelf_result.stderr.decode("utf-8"))
            print(f"patchelf failed")
            sys.exit(1)

    def copy_multi_gpu_kernel(self, output_dir: Path) -> None:
        """
        Copy the kernel .so file once for each device and run patchelf
        to point to the right library

        :param output_dir: the base directory
        """

        original_module: Path = output_dir / self.module_file
        for id, (kernel_filename, kokkos_gpu_module) in enumerate(zip(self.gpu_module_files, km.get_kokkos_gpu_modules())):
            kernel_path: Path = output_dir / kernel_filename

            try:
                shutil.copy(original_module, kernel_path)
            except Exception as ex:
                print(f"Exception while copying kernel: {ex}")
                sys.exit(1)

            lib_path: Path = Path(kokkos_gpu_module.__path__[0]) / "lib"
            patchelf: List[str] = ["patchelf",
                                "--set-rpath",
                                str(lib_path),
                                kernel_filename]

            patchelf_result = subprocess.run(patchelf, cwd=output_dir, capture_output=True, check=False)
            if patchelf_result.returncode != 0:
                print(patchelf_result.stderr.decode("utf-8"))
                print(f"patchelf failed")
                sys.exit(1)

            # Now replace the needed libkokkos* libraries with the correct version
            needed_libraries: str = subprocess.run(["patchelf", "--print-needed", kernel_filename], cwd=output_dir, capture_output=True, check=False).stdout.decode("utf-8")

            for line in needed_libraries.splitlines():
                if "libkokkoscore" in line or "libkokkoscontainers" in line:
                    # Line will be of the form f"libkokkoscore_{id}.so.3.4"
                    # This will extract id
                    current_id: int = int(line.split("_")[1].split(".")[0])
                    to_remove: str = line
                    to_add: str = line.replace(f"_{current_id}", f"_{id}")

                    subprocess.run(["patchelf", "--replace-needed", to_remove, to_add, kernel_filename], cwd=output_dir, capture_output=True, check=False)

    def get_cuda_compute_capability(self, compiler: str) -> str:
        """
        Get the compute capability of an Nvidia GPU

        :param compiler: the compiler being used (nvcc or g++)
        :returns: the compute capability as a string or the empty
            string if g++ is the compiler
        """

        if compiler != "nvcc":
            return ""
        else:
            import cupy

        return f"sm_{cupy.cuda.Device().compute_capability}"

    @staticmethod
    def is_compiled(output_dir: Path) -> bool:
        """
        Check if an entity is compiled

        :param output_dir: the directory containing the compiled entity
        :returns: true if compiled
        """

        return output_dir.is_dir()
