#!/bin/bash

COMPILER="${1}"
MODULE="${2}"
EXEC_SPACE="${3}"
PK_ARG_MEMSPACE="${4}"
PK_ARG_LAYOUT="${5}"
PK_REAL="${6}"
KOKKOS_PATH="${7}"
COMPUTE_CAPABILITY="${8}"
SRC=$(find -name "*.cpp")


if [ "${COMPILER}" == "g++" ]; then
        g++ \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -isystem "${KOKKOS_PATH}/include/kokkos" \
        -fPIC \
        -fopenmp -std=c++14 \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        g++ \
        -I.. \
        -O3 \
        -shared \
        -fopenmp \
        "${SRC}".o -o "${MODULE}" \
        "${KOKKOS_PATH}/lib/libkokkoscontainers.so" \
        "${KOKKOS_PATH}/lib/libkokkoscore.so"

elif [ "${COMPILER}" == "nvcc" ]; then
        "${KOKKOS_PATH}/bin/nvcc_wrapper" \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -isystem "${KOKKOS_PATH}/include/kokkos" \
        -arch="${COMPUTE_CAPABILITY}" \
        --expt-extended-lambda -fPIC \
        -Xcompiler -fopenmp -std=c++14 \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        "${KOKKOS_PATH}/bin/nvcc_wrapper" \
        -I.. \
        -O3 \
        -shared \
        -arch="${COMPUTE_CAPABILITY}" \
        --expt-extended-lambda \
        -fopenmp \
        "${SRC}".o -o "${MODULE}" \
        "${KOKKOS_PATH}/lib/libkokkoscontainers.so" \
        "${KOKKOS_PATH}/lib/libkokkoscore.so"
fi