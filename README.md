
# PyKokkos
[![Python Testing](https://github.com/kokkos/pykokkos/actions/workflows/main_ci.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/main_ci.yml)
[![Documentation](https://github.com/kokkos/pykokkos/actions/workflows/documentation.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/documentation.yml)
[![Linux](https://github.com/kokkos/pykokkos/actions/workflows/build_linux.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/build_linux.yml)
[![MacOS](https://github.com/kokkos/pykokkos/actions/workflows/build_macos.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/build_macos.yml)

PyKokkos is a framework for writing high-performance Python code
similar to Numba. In contrast to Numba, PyKokkos kernels are primarily
parallel and are also performance portable, meaning that they can run
efficiently on different hardware (CPUs, NVIDIA GPUs, and AMD GPUs)
with no changes required.

For more information about PyKokkos, see the PyKokkos GitHub pages:
https://kokkos.org/pykokkos/

## Installation

### Quick Start

PyKokkos consists of two components that need to be installed separately:

1. **pykokkos-base** (C++ bindings to Kokkos)
2. **pykokkos** (Python translation layer)

#### Installing pykokkos-base

<details>

<summary>Known issue: macOS + homebrew OpenMP</summary>
Homebrew does not add OpenMP to standard paths after
an install, leading to CMake build failures of the form:

```
...
Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
...
```

Such failures can be resolved by adding OpenMP to the `CPATH` and `LIBRARYPATH`:
```
export OMP_PREFIX=$(brew --prefix libomp)
export CPATH="${OMP_PREFIX}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${OMP_PREFIX}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Additionally, OpenMP must be added to the `DYLD_LIBRARY_PATH` for just-in-time compilation
```
export DYLD_LIBRARY_PATH="${OMP_PREFIX}/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```
Note that the `DYDL_LIBRARY_PATH` must be set for each shell session; we recommend
adding it to the shell profile.

</details>

```bash
# Clone the repository
git clone https://github.com/kokkos/pykokkos.git
cd pykokkos/

# Create and activate conda environment
conda create -n pyk python=3.13 -y
conda env update -n pyk -f base/environment.yml
conda activate pyk

# Install pykokkos-base from the root directory
python install_base.py install --verbose -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON
```

#### Installing pykokkos

After installing pykokkos-base:

```bash
# Install pykokkos (ensure you're in the pyk environment)
conda install -c conda-forge pybind11 cupy patchelf
pip install -e .
```

For more detailed installation instructions, please visit:
https://kokkos.org/pykokkos/installation.html

## Documentation
The documentation is available online at https://kokkos.org/pykokkos.
It can be built locally with the sphinx package by updating the `pyk` conda environment with
`conda install -c conda-forge sphinx sphinx_rtd_theme`
and running `cd docs; make html`.
The resulting html files reside in `_build/html` and
can be viewed in a browser (e.g., in a bash terminal run `open _build/html/index.html`).

## Citation

If you have used PyKokkos in a research project, please cite this
research paper:

```bibtex
@inproceedings{AlAwarETAL21PyKokkos,
  author = {Al Awar, Nader and Zhu, Steven and Biros, George and Gligoric, Milos},
  title = {A Performance Portability Framework for Python},
  booktitle = {International Conference on Supercomputing},
  pages = {467-478},
  year = {2021},
}
```

## Acknowledgments

This project is partially funded by the U.S. Department of Energy,
National Nuclear Security Administration under Award Number
DE-NA0003969 (PSAAP III).
