# PyKokkos

### Requirements

PyKokkos has only been tested with GCC 7.5.0 and NVCC 10.2. Different
compiler versions will probably require recompilation of the
kokkos-python bindings.

### Setting up your environment

Create a new conda environment with Python 3.8 and install the
packages from requirements.txt:

```bash
conda create --name pyk --file requirements.txt -c conda-forge
```

Install PyKokkos as an editable python package by running:

```bash
pip install --user -e .
```

Clone Kokkos:

```bash
git clone https://github.com/kokkos/kokkos.git $HOME/Kokkos/kokkos
pushd $HOME/Kokkos/kokkos
git checkout 953d7968e8fc5908af954f883e2e38d02c279cf2
popd
```

Install Kokkos as a shared library, once with only OpenMP enabled, and
once with CUDA and OpenMP enabled.

The following environment variables need to be set to point to the
path to the Kokkos installation directories:

PK_KOKKOS_LIB_PATH_CUDA: this is the path to the lib/ directory in your Kokkos CUDA install
PK_KOKKOS_INCLUDE_PATH_CUDA: this is the path to the include/ directory in your Kokkos CUDA install
PK_KOKKOS_NVCC: this is the path to bin/nvcc_wrapper in your Kokkos CUDA install
PK_KOKKOS_LIB_PATH_OMP: same as above for openmp
PK_KOKKOS_INCLUDE_PATH_OMP: same as above for openmp

### Developers

Nader Al Awar (nader.alawar@utexas.edu)

Steven Zhu

### Acknowledgments

This project is partially funded by the U.S. Department of Energy,
National Nuclear Security Administration under Award Number
DE-NA0003969 (PSAAP III).
