#!/bin/bash

COMPILER="${1}"
MODULE="${2}"
EXEC_SPACE="${3}"
PK_ARG_MEMSPACE="${4}"
PK_ARG_LAYOUT="${5}"
PK_REAL="${6}"
SRC=$(find -name "*.cpp")


if [ "${COMPILER}" == "g++" ]; then
        g++ \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -isystem "${PK_KOKKOS_INCLUDE_PATH_OMP}" \
        -fPIC \
        -fopenmp -std=c++11 \
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
        "${PK_KOKKOS_LIB_PATH_OMP}/libkokkoscontainers.so" \
        "${PK_KOKKOS_LIB_PATH_OMP}/libkokkoscore.so"

elif [ "${COMPILER}" == "nvcc" ]; then
        "${PK_KOKKOS_NVCC}" \
        `python3 -m pybind11 --includes` \
        -I.. \
        -O3 \
        -isystem "${PK_KOKKOS_INCLUDE_PATH_CUDA}" \
        -arch=sm_75 \
        --expt-extended-lambda -fPIC \
        -Xcompiler -fopenmp -std=c++11 \
        -DSPACE="${EXEC_SPACE}" \
        -o "${SRC}".o \
        -c "${SRC}" \
        -Dpk_arg_memspace="${PK_ARG_MEMSPACE}" \
        -Dpk_arg_layout="${PK_ARG_LAYOUT}" \
        -Dpk_exec_space="Kokkos::${EXEC_SPACE}" \
        -Dpk_real="${PK_REAL}"

        "${PK_KOKKOS_NVCC}" \
        -I.. \
        -O3 \
        -shared \
        -arch=sm_75 \
        --expt-extended-lambda \
        -fopenmp \
        "${SRC}".o -o "${MODULE}" \
        "${PK_KOKKOS_LIB_PATH_CUDA}/libkokkoscontainers.so" \
        "${PK_KOKKOS_LIB_PATH_CUDA}/libkokkoscore.so"
fi