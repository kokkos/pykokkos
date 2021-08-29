import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List

from pykokkos.interface import ExecutionSpace, get_default_layout, get_default_memory_space
import pykokkos.kokkos_manager as km


class CppSetup:
    """
    Creates the directory to hold the translation and invokes the compiler
    """

    def __init__(self, module_file: str, functor: str, bindings: str):
        """
        CppSetup constructor

        :param module: the name of the file containing the compiled Python module
        :param functor: the name of the generated functor file
        :param bindings: the name of the generated bindings file
        """

        self.module_file: str = module_file
        self.functor_file: str = functor
        self.bindings_file: str = bindings

        self.script: str = "compile.sh"
        self.script_path: Path = Path(__file__).resolve().parent / self.script

        self.makefile: Path = Path(__file__).resolve().parent / "template_Makefile"
        self.kokkos_devices_placeholder: str = "KOKKOS_DEVICES_PLACEHOLDER"
        self.target_placeholder: str = "TARGET_PLACEHOLDER"
        self.defines_placeholder: str = "DEFINES_PLACEHOLDER"
        self.force_uvm_placeholder: str = "FORCE_UVM_PLACEHOLDER"

        self.format: bool = False

    def compile(
        self,
        output_dir: Path,
        functor: List[str],
        bindings: List[str],
        space: ExecutionSpace,
        enable_uvm: bool,
        compiler: str
    ) -> None:
        """
        Compiles the generated C++ code

        :param output_dir: the base directory
        :param functor: the translated C++ functor
        :param bindings: the generated bindings
        :param space: the execution space to compile for
        :param enable_uvm: whether to enable CudaUVMSpace
        """

        self.initialize_directory(output_dir)
        self.write_source(output_dir, functor, bindings)
        self.copy_script(output_dir)
        self.invoke_script(output_dir, space, enable_uvm, compiler)


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

    def write_source(self, output_dir: Path, functor: List[str], bindings: List[str]) -> None:
        """
        Writes the generated C++ source code to a file

        :param output_dir: the base directory
        :param functor: the generated C++ functor
        :param bindings: the generated bindings
        """

        functor_path: Path = output_dir.parent / self.functor_file
        bindings_path: Path = output_dir / self.bindings_file

        with open(functor_path, "w") as out:
            out.write("\n".join(functor))
        with open(bindings_path, "w") as out:
            out.write("\n".join(bindings))

        if self.format:
            try:
                subprocess.run(["clang-format", "-i", functor_path])
                subprocess.run(["clang-format", "-i", bindings_path])
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

    def get_kokkos_path(self) -> Path:
        """
        Get the location of the installed Kokkos package

        :returns: path to the location
        """

        from pykokkos.bindings import kokkos

        return Path(kokkos.__path__[0]).parent

    def invoke_script(self, output_dir: Path, space: ExecutionSpace, enable_uvm: bool, compiler: str) -> None:
        """
        Invoke the compilation script

        :param output_dir: the base directory
        :param space: the execution space of the workload
        :param enable_uvm: whether to enable CudaUVMSpace
        :param compiler: what compiler to use
        """

        view_space: str = "Kokkos::HostSpace"
        if space is ExecutionSpace.Cuda and enable_uvm:
            view_space = "Kokkos::CudaUVMSpace"

        view_layout: str = str(get_default_layout(get_default_memory_space(space)))
        view_layout = view_layout.split(".")[-1]
        view_layout = f"Kokkos::{view_layout}"

        precision: str = km.get_default_precision().__name__.split(".")[-1]
        kokkos_path: Path = self.get_kokkos_path()

        compute_capability: str = ""
        if compiler == "nvcc":
            try:
                import cupy
                compute_capability = f"sm_{cupy.cuda.Device().compute_capability}"
            except:
                print(f"ERROR: could not get CUDA compute capability")

        command: List[str] = [f"./{self.script}",
                              compiler,           # What compiler to use
                              self.module_file,   # Compilation target
                              space.value,        # Execution space
                              view_space,         # Argument views memory space
                              view_layout,        # Argument views memory layout
                              precision,          # Default real precision
                              str(kokkos_path),   # Path to Kokkos install
                              compute_capability] # Device compute capability
        compile_result = subprocess.run(command, cwd=output_dir, capture_output=True, check=False)

        if compile_result.returncode != 0:
            print(compile_result.stderr.decode("utf-8"))
            print(f"C++ compilation in {output_dir} failed")
            sys.exit(1)

        rpath: str
        if space is ExecutionSpace.Cuda:
            rpath = os.environ["PK_KOKKOS_LIB_PATH_CUDA"]
        else:
            rpath = os.environ["PK_KOKKOS_LIB_PATH_OMP"]
        patchelf: List[str] = ["patchelf",
                               "--set-rpath",
                               rpath,
                               self.module_file]

        patchelf_result = subprocess.run(patchelf, cwd=output_dir, capture_output=True, check=False)
        if patchelf_result.returncode != 0:
            print(patchelf_result.stderr.decode("utf-8"))
            print(f"patchelf failed")
            sys.exit(1)

    @staticmethod
    def is_compiled(output_dir: Path) -> bool:
        """
        Check if an entity is compiled

        :param output_dir: the directory containing the compiled entity
        :returns: true if compiled
        """

        return output_dir.is_dir()
