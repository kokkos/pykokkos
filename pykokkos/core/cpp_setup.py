import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import List, Tuple

from pykokkos.interface import (
    ExecutionSpace,
    get_default_layout,
    get_default_memory_space,
    is_host_execution_space,
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

        self.cmake_template: str = "CMakeLists.txt"
        self.cmake_template_path: Path = (
            Path(__file__).resolve().parent / self.cmake_template
        )

        self.lib_path_env: str = "PK_KOKKOS_LIB_PATH"

        self.format: bool = os.getenv("PK_FORMAT") is not None

    def compile_raw_source(
        self,
        output_dir: Path,
        source: List[str],
        filename: str,
        space: ExecutionSpace,
        enable_uvm: bool,
        compiler: str,
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
        cmake_args, module_name = self.generate_cmake(
            output_dir, space, enable_uvm, compiler
        )
        self.invoke_cmake(output_dir, cmake_args, module_name)

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
        compiler: str,
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
        self.write_source(
            output_dir,
            functor,
            functor_filename,
            functor_cast,
            functor_cast_filename,
            bindings,
            bindings_filename,
        )
        cmake_args, module_name = self.generate_cmake(
            output_dir, space, enable_uvm, compiler
        )
        self.invoke_cmake(output_dir, cmake_args, module_name)
        if (
            space in {ExecutionSpace.Cuda, ExecutionSpace.HIP}
            and km.is_multi_gpu_enabled()
        ):
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

    def write_source(
        self,
        output_dir: Path,
        functor: List[str],
        functor_filename: str,
        functor_cast: List[str],
        functor_cast_filename: str,
        bindings: List[str],
        bindings_filename: str,
    ) -> None:
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

        self.write_raw_source(output_dir.parent, functor, functor_filename)
        self.write_raw_source(output_dir.parent, functor_cast, functor_cast_filename)
        self.write_raw_source(output_dir, bindings, bindings_filename)

    def write_raw_source(
        self, output_dir: Path, source: List[str], filename: str
    ) -> None:
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

    def generate_cmake(
        self, output_dir: Path, space: ExecutionSpace, enable_uvm: bool, compiler: str
    ) -> Tuple[List[str], str]:
        """
        Copy CMakeLists.txt template and prepare CMake configuration variables

        :param output_dir: the base directory
        :param space: the execution space of the workload
        :param enable_uvm: whether to enable CudaUVMSpace
        :param compiler: what compiler to use
        :returns: tuple of (cmake_args, module_name)
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
        compiler_path: Path
        lib_path, compiler_path = self.get_kokkos_paths(space, compiler)
        compute_capability: str = self.get_cuda_compute_capability(compiler)
        lib_suffix: str = self.get_kokkos_lib_suffix(space)

        cmake_file: Path = output_dir / "CMakeLists.txt"
        try:
            shutil.copy(self.cmake_template_path, cmake_file)
        except Exception as ex:
            print(f"Exception while copying CMakeLists.txt template: {ex}")
            sys.exit(1)

        # Remove the .so extension from module file name for CMake target
        module_name = self.module_file.replace(".so", "").replace(".pyd", "")

        try:
            import pybind11

            pybind11_dir = pybind11.get_cmake_dir()
        except ImportError:
            print(f"Can not get pybind11 except dir: {ex}")
            sys.exit(1)

        cmake_args = [
            f"-DMODULE_NAME={module_name}",
            f"-DKokkos_ROOT={lib_path.parent.resolve()}",
            f"-DPK_EXEC_SPACE={space_value}",
            f"-DPK_ARG_MEMSPACE={view_space}",
            f"-DPK_ARG_LAYOUT={view_layout}",
            f"-DPK_REAL={precision}",
            f"-DLIB_SUFFIX={lib_suffix}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if pybind11_dir is not None:
            cmake_args.append(f"-Dpybind11_DIR={pybind11_dir}")

        return cmake_args, module_name

    def get_kokkos_paths(
        self, space: ExecutionSpace, compiler: str
    ) -> Tuple[Path, Path]:
        """
        Get the paths of the Kokkos install lib directory.
        If the environment variable is set, use that
        Kokkos install. If not, fall back to the installed
        pykokkos-base package.

        :param space: the execution space to compile for
        :param compiler: what compiler to use
        :returns: a tuple of paths to the Kokkos lib/
            and compiler to be used
        """

        lib_path: Path
        if self.lib_path_env in os.environ:
            lib_path = Path(os.environ.get(self.lib_path_env))
            if not lib_path.is_dir():
                raise RuntimeError(
                    f"lib/ directory path {str(lib_path)} does not exist"
                )

            compiler_path: Path
            if compiler != "nvcc":
                compiler_path = Path("g++")
            else:
                compiler_path = lib_path.parent / "bin/nvcc_wrapper"

            return lib_path, compiler_path

        import sys

        is_cpu: bool = is_host_execution_space(space)
        kokkos_lib: ModuleType = km.get_kokkos_module(is_cpu)
        install_path = Path(kokkos_lib.__path__[0])
        lib_parent_path: Path
        if km.is_multi_gpu_enabled():
            lib_parent_path = install_path
        else:
            lib_parent_path = install_path.parent

        lib_path = None
        if (lib_parent_path / "lib").is_dir():
            lib_path = lib_parent_path / "lib"
        elif (lib_parent_path / "lib64").is_dir():
            lib_path = lib_parent_path / "lib64"
        else:
            # Try checking sys.prefix/lib and sys.prefix/lib64
            sys_prefix = Path(sys.prefix)
            if (sys_prefix / "lib").is_dir():
                # Verify that kokkos libraries actually exist here
                kokkos_lib_files = list((sys_prefix / "lib").glob("libkokkoscore.*"))
                if kokkos_lib_files:
                    lib_path = sys_prefix / "lib"
            if lib_path is None and (sys_prefix / "lib64").is_dir():
                kokkos_lib_files = list((sys_prefix / "lib64").glob("libkokkoscore.*"))
                if kokkos_lib_files:
                    lib_path = sys_prefix / "lib64"

        if lib_path is None:
            raise RuntimeError(
                "lib/ or lib64/ directories not found in installed pykokkos-base package."
                f" Try setting {self.lib_path_env} instead."
            )

        compiler_path: Path
        if compiler != "nvcc":
            compiler_path = Path(compiler)
        else:
            # Try traditional location first, then sys.prefix
            compiler_path = install_path.parent / "bin/nvcc_wrapper"
            if not compiler_path.exists():
                sys_prefix = Path(sys.prefix)
                alt_compiler_path = sys_prefix / "bin/nvcc_wrapper"
                if alt_compiler_path.exists():
                    compiler_path = alt_compiler_path

        return lib_path, compiler_path

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

    def invoke_cmake(
        self, output_dir: Path, cmake_args: List[str], module_name: str
    ) -> None:
        """
        Invoke CMake to configure and build the project

        :param output_dir: the base directory containing CMakeLists.txt
        :param cmake_args: list of CMake configuration arguments
        :param module_name: the name of the module being built
        """

        build_dir = output_dir / "build"

        # Run CMake configuration with arguments
        cmake_config_cmd = [
            "cmake",
            "-B",
            str(build_dir),
            "-S",
            str(output_dir),
        ] + cmake_args
        config_result = subprocess.run(
            cmake_config_cmd, capture_output=True, check=False
        )

        if config_result.returncode != 0:
            print(config_result.stderr.decode("utf-8"))
            print(f"CMake configuration in {output_dir} failed")
            sys.exit(1)

        # Run CMake build with parallel jobs
        num_jobs = multiprocessing.cpu_count()
        cmake_build_cmd = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "-j",
            str(num_jobs),
        ]
        build_result = subprocess.run(cmake_build_cmd, capture_output=True, check=False)

        if build_result.returncode != 0:
            print(build_result.stderr.decode("utf-8"))
            print(f"CMake build in {output_dir} failed")
            sys.exit(1)

        cmake_install_cmd = [
            "cmake",
            "--install",
            str(build_dir),
            "--prefix",
            str(output_dir.resolve()),
        ]
        install_result = subprocess.run(
            cmake_install_cmd, capture_output=True, check=False
        )

        if install_result.returncode != 0:
            print(install_result.stderr.decode("utf-8"))
            print(f"CMake install in {output_dir} failed")
            sys.exit(1)

    def copy_multi_gpu_kernel(self, output_dir: Path) -> None:
        """
        Copy the kernel .so file once for each device and run patchelf
        to point to the right library

        :param output_dir: the base directory
        """

        original_module: Path = output_dir / self.module_file
        for id, (kernel_filename, kokkos_gpu_module) in enumerate(
            zip(self.gpu_module_files, km.get_kokkos_gpu_modules())
        ):
            kernel_path: Path = output_dir / kernel_filename

            try:
                shutil.copy(original_module, kernel_path)
            except Exception as ex:
                print(f"Exception while copying kernel: {ex}")
                sys.exit(1)

            lib_path: Path = Path(kokkos_gpu_module.__path__[0]) / "lib"
            patchelf: List[str] = [
                "patchelf",
                "--set-rpath",
                str(lib_path),
                kernel_filename,
            ]

            patchelf_result = subprocess.run(
                patchelf, cwd=output_dir, capture_output=True, check=False
            )
            if patchelf_result.returncode != 0:
                print(patchelf_result.stderr.decode("utf-8"))
                print(f"patchelf failed")
                sys.exit(1)

            # Now replace the needed libkokkos* libraries with the correct version
            needed_libraries: str = subprocess.run(
                ["patchelf", "--print-needed", kernel_filename],
                cwd=output_dir,
                capture_output=True,
                check=False,
            ).stdout.decode("utf-8")

            for line in needed_libraries.splitlines():
                if "libkokkoscore" in line or "libkokkoscontainers" in line:
                    # Line will be of the form f"libkokkoscore_{id}.so.3.4"
                    # This will extract id
                    current_id: int = int(line.split("_")[1].split(".")[0])
                    to_remove: str = line
                    to_add: str = line.replace(f"_{current_id}", f"_{id}")

                    subprocess.run(
                        [
                            "patchelf",
                            "--replace-needed",
                            to_remove,
                            to_add,
                            kernel_filename,
                        ],
                        cwd=output_dir,
                        capture_output=True,
                        check=False,
                    )

    def get_cuda_compute_capability(self, compiler: str) -> str:
        """
        Get the compute capability of an Nvidia GPU

        :param compiler: the compiler being used (nvcc or g++)
        :returns: the compute capability as a string (e.g., "89") or the empty
            string if g++ is the compiler
        """

        if compiler != "nvcc":
            return ""
        else:
            import cupy

        return str(cupy.cuda.Device().compute_capability)

    @staticmethod
    def is_compiled(output_dir: Path) -> bool:
        """
        Check if an entity is compiled

        :param output_dir: the directory containing the compiled entity
        :returns: true if compiled
        """

        return output_dir.is_dir()
