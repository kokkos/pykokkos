
# PyKokkos
[![Python Testing](https://github.com/kokkos/pykokkos/actions/workflows/main_ci.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/main_ci.yml)
[![Documentation](https://github.com/kokkos/pykokkos/actions/workflows/documentation.yml/badge.svg)](https://github.com/kokkos/pykokkos/actions/workflows/documentation.yml)

PyKokkos is a framework for writing high-performance Python code
similar to Numba. In contrast to Numba, PyKokkos kernels are primarily
parallel and are also performance portable, meaning that they can run
efficiently on different hardware (CPUs, NVIDIA GPUs, and AMD GPUs)
with no changes required.

For more information about PyKokkos, see the PyKokkos GitHub pages:
https://kokkos.github.io/pykokkos/index.html

## Installation

### Quick Start

PyKokkos consists of two components that need to be installed separately:

1. **pykokkos-base** (C++ bindings to Kokkos)
2. **pykokkos** (Python translation layer)

#### Installing pykokkos-base

```bash
# Clone the repository
git clone https://github.com/kokkos/pykokkos.git
cd pykokkos/

# Create and activate conda environment
conda create --name pyk --file base/requirements.txt python=3.11
conda activate pyk

# Install pykokkos-base from the root directory
python install_base.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON
```

#### Installing pykokkos

After installing pykokkos-base:

```bash
# Install pykokkos (ensure you're in the pyk environment)
conda install -c conda-forge pybind11 cupy patchelf
pip install -e .
```

For more detailed installation instructions, please visit:
https://kokkos.github.io/pykokkos/installation.html

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
