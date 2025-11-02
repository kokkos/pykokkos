#!/bin/bash

# Constants and variables
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_FLAGS="-DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=4 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_THREADS=OFF"
KOKKOS_ARCHITECTURES="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_THREADS=OFF"
PYKOKKOS_ARCHITECTURES=""
PARALLEL_LEVEL=1
CXX_COMPILER=""
C_COMPILER=""

# Define architecture configurations
declare -A ARCH_CONFIG=(
    ["CUDA"]="command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null"
    ["HIP"]="command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null"
    ["OPENMP"]="echo '#include <omp.h>' | cpp -fopenmp &> /dev/null"
)

# List of available architectures
declare -A ARCH_AVAILABLE
ARCH_AVAILABLE["SERIAL"]=1  # SERIAL is always available


# Detect and validate compilers
function detect_compiler {
    echo "Detecting C/C++ compilers..."

    local cxx_compilers=("c++" "g++" "clang++")
    for compiler in "${cxx_compilers[@]}"; do
        if command -v "$compiler" &> /dev/null; then
            CXX_COMPILER=$(command -v "$compiler")
            echo "Found C++ compiler: $CXX_COMPILER"
            break
        fi
    done

    local c_compilers=("cc" "gcc" "clang")
    for compiler in "${c_compilers[@]}"; do
        if command -v "$compiler" &> /dev/null; then
            C_COMPILER=$(command -v "$compiler")
            echo "Found C compiler: $C_COMPILER"
            break
        fi
    done

    # Exit if either compiler is not found
    if [ -z "$CXX_COMPILER" ]; then
        echo "ERROR: No C++ compiler found"
        exit 1
    fi
    if [ -z "$C_COMPILER" ]; then
        echo "ERROR: No C compiler found"
        exit 1
    fi
}

# Check if conda is available
function check_conda {
    if [[ -x $(conda --version) ]]; then
        echo "ERROR: conda not found."
        dirs -c
        exit 1
    fi
}

# Detect available HPC architecture
function detect_arch {
    echo "Detecting available architectures..."
    echo "SERIAL architecture is enabled by default"

    # Detect each architecture
    for arch in "${!ARCH_CONFIG[@]}"; do
        detection_cmd="${ARCH_CONFIG[$arch]}"
        ARCH_AVAILABLE[$arch]=0

        if [ -n "$detection_cmd" ]; then
            if eval "$detection_cmd"; then
                ARCH_AVAILABLE[$arch]=1
                echo "$arch detected"
            else
                echo "$arch not detected"
            fi
        fi
    done

    # THREADS is always disabled to avoid conflicts with OpenMP
}

# Ask which architecture(s) to enable
function prompt_arch {
    echo -e "\nSelect architectures to enable:"

    for arch in "${!ARCH_CONFIG[@]}"; do
        if [ ${ARCH_AVAILABLE[$arch]:-0} -eq 0 ] && [ "$arch" != "Threads" ]; then
            continue
        fi

        # Set default based on availability
        if [ ${ARCH_AVAILABLE[$arch]:-0} -eq 1 ]; then
            prompt="[Y/n]"
            default="y"
        else
            prompt="[y/N]"
            default="n"
        fi

        read -p "Enable $arch support? $prompt: " enable
        if [[ "$default" = "y" && "$enable" != "n" && "$enable" != "N" ]] || \
           [[ "$default" = "n" && ("$enable" = "y" || "$enable" = "Y") ]]; then
            # Both PyKokkos and Kokkos flags for enabled architectures
            KOKKOS_ARCHITECTURES="$KOKKOS_ARCHITECTURES -DKokkos_ENABLE_${arch}=ON"
            PYKOKKOS_ARCHITECTURES="$PYKOKKOS_ARCHITECTURES -DENABLE_${arch}=ON"
        fi
    done

    echo "Selected architectures: $KOKKOS_ARCHITECTURES"
}

# Ask user about how many processors should be used during pykokkos build
function prompt_build_proc {
    echo "WARNING: Using too many processors for building may slow down your system significantly."
    echo "Available processors: $(nproc)"
    read -p "Enter number of processors to use [default: $(nproc)]: " USER_PARALLEL

    PARALLEL_LEVEL=${USER_PARALLEL:-$(nproc)}
}

# install pykokkos-base
function install_base {
    # Clone pykokkos-base and build it
    mkdir -p "$SCRIPT_DIR"/external
    pushd external
    git clone https://github.com/kokkos/pykokkos-base.git
    pushd pykokkos-base

    # Create the conda environment using the resolved conda command
    conda create -y --name pyk --file requirements.txt python=3.11
    
    # Initialize conda in the current shell
    eval "$(conda shell.bash hook)"
    conda activate pyk

    echo "=== Build Configuration ==="
    echo "Default flags: $DEFAULT_FLAGS"
    echo "PyKokkos architectures: $PYKOKKOS_ARCHITECTURES"
    echo "Kokkos architectures: $KOKKOS_ARCHITECTURES"
    echo "Compilers: -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_C_COMPILER=${C_COMPILER}"
    echo "Parallel: -DCMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL}"
    echo "==========================="

    # Split flags into array to avoid quote issues
    python setup.py install -- \
        $DEFAULT_FLAGS \
        $PYKOKKOS_ARCHITECTURES \
        $KOKKOS_ARCHITECTURES \
        -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
        -DCMAKE_C_COMPILER=${C_COMPILER} \
        -DCMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL}

}
function install {
    install_base

    conda install -c conda-forge pybind11 cupy patchelf -y
    pip install --user -e .

    conda deactivate
    dirs -c
}

function farewell {
    echo "PyKokkos successfully installed"
    echo "Enter 'conda activate pyk' to enable python environment"
}

# Main execution process
check_conda
detect_arch
detect_compiler
prompt_arch
prompt_build_proc
install || exit 1
farewell