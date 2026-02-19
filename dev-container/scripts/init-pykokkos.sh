#!/bin/bash

# Wait a bit for CUDA to be fully initialized
sleep 2

# Initialize conda and create PyKokkos environment if it doesn't exist
if [ -f /tmp/conda_path ]; then
    export CONDA_PATH
    CONDA_PATH=$(cat /tmp/conda_path)
    source "${CONDA_PATH}/../etc/profile.d/conda.sh"

    export USERNAME
    USERNAME=$(cat /tmp/username 2>/dev/null || echo "root")
    if [ "${USERNAME}" != "root" ]; then
        export HOME_DIR="/home/${USERNAME}"
    else
        export HOME_DIR="/root"
    fi

    cd ${HOME_DIR} || exit

    # Accept conda TOS for this user
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    # Check if pyk environment exists
    if ! conda env list | grep -q "^pyk "; then
        echo "[PyKokkos Init] Creating conda environment..."
        cd pykokkos || exit
        if conda create --name pyk --file base/requirements.txt python=3.11 -y > /tmp/pyk-install.log 2>&1; then
            echo "[PyKokkos Init] Installing PyKokkos with CUDA support..." > /tmp/pyk-install.log
            conda activate pyk
            echo "[PyKokkos Init] PYK env activated..." > /tmp/pyk-install.log
            python install_base.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON >> /tmp/pyk-install.log 2>&1
            conda install -c conda-forge pybind11 cupy patchelf -y
            pip install -e ./

            if [ $? -eq 0 ]; then
                echo "[PyKokkos Init] Installation complete! Log: /tmp/pyk-install.log"
            else
                echo "[PyKokkos Init] Installation failed! Check /tmp/pyk-install.log for details"
            fi
        else
            echo "[PyKokkos Init] Failed to create conda environment! Check /tmp/pyk-install.log for details"
        fi
    else
        echo "[PyKokkos Init] Environment 'pyk' already exists, skipping installation"
    fi
fi
