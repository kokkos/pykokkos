#!/bin/bash



# Show warning about processor usage
echo "WARNING: Using too many processors for building may slow down your system significantly."
echo "Available processors: $(nproc)"

# Determine PARALLEL_LEVEL:
# - prefer environment variable PARALLEL_LEVEL if set and >0 (can be baked-in via Docker ARG/ENV)
# - otherwise fall back to container/host nproc
if [ -n "${PARALLEL_LEVEL:-}" ] && [ "${PARALLEL_LEVEL}" -gt 0 ] 2>/dev/null; then
    PARALLEL_LEVEL="${PARALLEL_LEVEL}"
else
    read -p "Enter number of processors to use [default: $(nproc)]: " USER_PARALLEL
    PARALLEL_LEVEL=${USER_PARALLEL:-$(nproc)}
fi

pushd "/home/${USERNAME}"
# Prefer the Miniconda installed in the user's home; fall back to any `conda` in PATH.
if [ -n $(which conda) ]; then
    echo "ERROR: conda not found."
    dirs -c
    exit 1
fi

# Begin pykokkos-base installation process
git clone https://github.com/kokkos/pykokkos-base.git
pushd pykokkos-base

# Create the conda environment using the resolved conda command
"${CONDA_CMD}" create -y --name pyk --file requirements.txt python=3.11
"${CONDA_CMD}" activate pyk

python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF \
    -DCMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
    -DENABLE_VIEW_RANKS=4 \
    -DENABLE_CUDA=ON -DKokkos_ENABLE_CUDA=ON \
    -DENABLE_OPENMP=ON -DKokkos_ENABLE_OPENMP=ON \
    -DENABLE_THREADS=OFF \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_POLICY_VERSION_MINIMUM=3.5

popd
dirs -c

# Make hidden file to check that pykokkos is installed now
# touch /home/"${USERNAME}"/.pyk-installed
